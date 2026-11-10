"""Extract decision points: the moment right before each tool call executes.

At step t the monitor may see steps 0..t-1 in full, plus step t's thought, tool and args.
It may NOT see step t's observation (that only exists after the call runs).
Target y = 1 if step t is labeled 'hijacked', else 0. An injection_point step is y = 0: the call
itself was requested by the user, the poison only arrives in its observation.

By default every tool call is a decision point. The `state_changing` column lets you restrict to
state-changing calls (the narrower definition in the plan): see `select()`.
Why the default is "all calls": with state-changing calls only, 696 of 5,536 attacked traces
(including 555 of 1,500 partial hijacks) have no hijacked decision point at all, so the monitor
could never catch them. The script prints this check.

Run:  python -m l2e.decision_points     (writes data/processed/decision_points.csv)
"""
from collections import Counter

import pandas as pd

from . import config
from .data import load_anon
from .tools import is_state_changing


def prefix_steps(traj: dict, t: int):
    """Steps visible at decision point t: earlier steps in full, current step without obs/label."""
    visible = [dict(s) for s in traj["steps"][:t]]
    cur = traj["steps"][t]
    visible.append({"thought": cur["thought"], "tool": cur["tool"], "args": cur["args"]})
    for s in visible:
        s.pop("label", None)  # labels are never model input
    return visible


def extract(traj: dict, part: str):
    rows = []
    for t, step in enumerate(traj["steps"]):
        rows.append({
            "dp_id": f"{traj['id']}:{t}", "traj_id": traj["id"], "step": t, "part": part,
            "domain": traj["agent"], "category": traj["category"],
            "source_category": traj["source_category"], "compliance": traj.get("compliance") or "none",
            "tool": step["tool"], "state_changing": is_state_changing(step["tool"]),
            "step_label": step["label"], "y": int(step["label"] == "hijacked"),
        })
    return rows


def build(split="taskdisjoint"):
    rows = []
    for part in config.PARTS:
        for traj in load_anon(split, part):
            rows += extract(traj, part)
    return pd.DataFrame(rows)


def select(df, mode="all"):
    """mode='all' (every call) or 'state_changing' (plan's narrower definition)."""
    return df if mode == "all" else df[df["state_changing"]]


def reachable_attacks(df, mode):
    """How many attacked traces have at least one hijacked decision point under this mode."""
    att = df[df["category"] == "attacked"]
    hit = select(att, mode).groupby("traj_id")["y"].max()
    hit = hit.reindex(att["traj_id"].unique(), fill_value=0)
    by_c = att.drop_duplicates("traj_id").set_index("traj_id")["compliance"]
    return {c: (int(hit[by_c == c].sum()), int((by_c == c).sum())) for c in sorted(by_c.unique())} | \
        {"all": (int(hit.sum()), len(hit))}


def main():
    config.ensure_dirs()
    df = build("taskdisjoint")
    path = config.PROCESSED / "decision_points.csv"
    df.to_csv(path, index=False)

    md = ["# Week 1 · Decision points (task-disjoint split)", "",
          f"{len(df)} decision points from {df['traj_id'].nunique()} traces, saved to `{path.relative_to(config.ROOT)}`.", "",
          "| part | all calls | hijacked | state-changing calls | hijacked (state-changing) |", "|---|---|---|---|---|"]
    for part in config.PARTS:
        d = df[df["part"] == part]
        s = d[d["state_changing"]]
        md.append(f"| {part} | {len(d)} | {d['y'].sum()} | {len(s)} | {s['y'].sum()} |")
    md += ["", "Attacked traces with at least one hijacked decision point (reachable / total):", "",
           "| compliance | every call | state-changing only |", "|---|---|---|"]
    a, s = reachable_attacks(df, "all"), reachable_attacks(df, "state_changing")
    for c in a:
        md.append(f"| {c} | {a[c][0]} / {a[c][1]} | {s[c][0]} / {s[c][1]} |")
    md += ["", "Step labels at decision points: " + ", ".join(f"{k} {v}" for k, v in Counter(df['step_label']).items()), ""]
    (config.REPORTS / "04_decision_points.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
