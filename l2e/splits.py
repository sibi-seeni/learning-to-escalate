"""Shift splits: tool coverage (primary) and leave-one-domain-out (secondary).

Tool coverage, one file per random seed:
  1. Hold out TOOL_HOLDOUT_FRAC of each domain's tools (set H).
  2. Train/val: task-disjoint train/val decision points that come BEFORE the trace's first use of
     any H tool. Because the monitor only reads prefixes, it never sees an H tool anywhere.
     (Dropping whole traces instead would throw away 60-85% of training data.)
  3. Test pool (task-disjoint test part):
       seen   = call tool not in H, and no H tool earlier in the prefix
       unseen = call tool in H
  4. Coverage sets at 100/75/50/25%: same size N and the same hijacked rate at every level, so only
     tool familiarity changes. Coverage = share of test calls whose tool was seen in training.

Leave-one-domain-out, one file per domain:
  train/val = task-disjoint train/val decision points from the other four domains
  test      = every decision point of the held-out domain (all parts; none were trained on)

Files: data/processed/splits/*.json, each holding lists of dp_id.
Run:   python -m l2e.splits
"""
import json
import random

import numpy as np
import pandas as pd

from . import config


def held_out_tools(dps, seed, frac=config.TOOL_HOLDOUT_FRAC):
    rng = random.Random(seed)
    held = set()
    for dom in config.DOMAINS:
        tools = sorted(dps.loc[dps["domain"] == dom, "tool"].unique())
        held |= set(rng.sample(tools, max(1, round(frac * len(tools)))))
    return held


def before_first_held(dps, held):
    """True for decision points whose prefix (steps 0..t) contains no held-out tool."""
    in_h = dps["tool"].isin(held).astype(int)
    seen_h = in_h.groupby(dps["traj_id"]).cumsum()  # rows are in step order within each trace
    return seen_h == 0


def coverage_sets(seen, unseen, levels, seed):
    """Sample equal-size sets with a fixed hijacked rate at each coverage level."""
    pi = pd.concat([seen, unseen])["y"].mean()
    pools = {(g, y): df[df["y"] == y] for g, df in (("seen", seen), ("unseen", unseen)) for y in (0, 1)}
    # Largest N that every level can fill without reusing rows.
    caps = []
    for c in levels:
        for g, share in (("seen", c), ("unseen", 1 - c)):
            for y, rate in ((1, pi), (0, 1 - pi)):
                if share * rate > 0:
                    caps.append(len(pools[(g, y)]) / (share * rate))
    n = int(min(caps))
    rng = np.random.RandomState(seed)
    out = {}
    for c in levels:
        ids = []
        for g, share in (("seen", c), ("unseen", 1 - c)):
            for y, rate in ((1, pi), (0, 1 - pi)):
                k = int(round(n * share * rate))
                if k:
                    ids += list(pools[(g, y)].sample(n=k, random_state=rng)["dp_id"])
        out[f"{int(c * 100)}"] = sorted(ids)
    return out, n, pi


def tool_coverage(dps, seed):
    held = held_out_tools(dps, seed)
    clean = before_first_held(dps, held)
    train = dps[(dps["part"] == "train") & clean]
    val = dps[(dps["part"] == "val") & clean]
    test = dps[dps["part"] == "test"]
    unseen = test[test["tool"].isin(held)]
    seen = test[clean.loc[test.index] & ~test["tool"].isin(held)]
    sets, n, pi = coverage_sets(seen, unseen, config.COVERAGE_LEVELS, seed)
    return {"seed": seed, "held_out_tools": sorted(held), "train": list(train["dp_id"]), "val": list(val["dp_id"]),
            "test": sets, "test_size": n, "test_hijacked_rate": pi}, train


def lodo(dps, domain):
    keep = dps["domain"] != domain
    return {"held_out_domain": domain,
            "train": list(dps[keep & (dps["part"] == "train")]["dp_id"]),
            "val": list(dps[keep & (dps["part"] == "val")]["dp_id"]),
            "test": list(dps[~keep]["dp_id"])}


def main():
    config.ensure_dirs()
    dps = pd.read_csv(config.PROCESSED / "decision_points.csv")
    y = dps.set_index("dp_id")["y"]
    md = ["# Week 1 · Shift splits", "",
          f"## Tool coverage (hold out {config.TOOL_HOLDOUT_FRAC:.0%} of each domain's tools)", "",
          "| seed | held-out tools | train DPs kept | train hijacked | smallest domain (train DPs) | test N per level | test hijacked rate |",
          "|---|---|---|---|---|---|---|"]
    full_train = (dps["part"] == "train").sum()
    for seed in config.TOOL_HOLDOUT_SEEDS:
        split, train = tool_coverage(dps, seed)
        (config.SPLITS_DIR / f"tool_coverage_seed{seed}.json").write_text(json.dumps(split))
        per_dom = train["domain"].value_counts()
        md.append(f"| {seed} | {len(split['held_out_tools'])} | {len(train)} / {full_train} | {int(train['y'].sum())} | "
                  f"{per_dom.idxmin()} ({per_dom.min()}) | {split['test_size']} | {split['test_hijacked_rate']:.3f} |")
    md += ["", "Held-out tools, seed 0: " + ", ".join(
        json.loads((config.SPLITS_DIR / "tool_coverage_seed0.json").read_text())["held_out_tools"]), "",
        "## Leave one domain out", "",
        "| held-out domain | train DPs | val DPs | test DPs | test hijacked |", "|---|---|---|---|---|"]
    for dom in config.DOMAINS:
        split = lodo(dps, dom)
        (config.SPLITS_DIR / f"lodo_{dom}.json").write_text(json.dumps(split))
        md.append(f"| {dom} | {len(split['train'])} | {len(split['val'])} | {len(split['test'])} | {int(y[split['test']].sum())} |")
    md.append("")
    (config.REPORTS / "06_splits.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
