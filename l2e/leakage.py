"""Re-check the world-identity shortcut before and after anonymization.

The AgentDrift paper looks up each test trajectory's world object among training trajectories and
predicts that world's majority category (unseen world -> benign). Raw worlds give ~86% binary
accuracy. After anonymizing, the same lookup should fall to about the majority-class rate, and no
world strings should survive in the text a model reads.

Run:  python -m l2e.leakage      (needs `python -m l2e.anonymize` first)
"""
import json
from collections import Counter, defaultdict

from . import config
from .anonymize import residual_hits
from .data import load_anon, load_part


def world_key(rec):
    return json.dumps(rec["world"], sort_keys=True)


def lookup_accuracy(train, test, key=world_key):
    """Binary accuracy (attacked vs not) of the majority-category-per-world lookup, per domain."""
    table = defaultdict(Counter)
    for r in train:
        table[key(r)][r["category"]] += 1
    rows = defaultdict(lambda: [0, 0, 0])  # domain -> [n, correct, world_seen]
    for r in test:
        k = key(r)
        pred = table[k].most_common(1)[0][0] if k in table else "benign"
        for d in (r["agent"], "all"):
            rows[d][0] += 1
            rows[d][1] += (pred == "attacked") == (r["category"] == "attacked")
            rows[d][2] += k in table
    return {d: {"n": n, "acc": c / n, "seen": s} for d, (n, c, s) in rows.items()}


def majority_rate(test):
    return sum(r["category"] != "attacked" for r in test) / len(test)


def main():
    config.ensure_dirs()
    out = {}
    md = ["# Week 1 · World-identity leakage check", "",
          "Lookup = predict each test trace's category from its world object's majority category in train "
          "(unseen world -> benign). Binary accuracy, attacked vs. not.", ""]
    for split in config.SPLIT_DIRS:
        raw_tr, raw_te = load_part(split, "train"), load_part(split, "test")
        an_tr, an_te = load_anon(split, "train"), load_anon(split, "test")
        raw = lookup_accuracy(raw_tr, raw_te)
        anon = lookup_accuracy(an_tr, an_te)
        maj = majority_rate(raw_te)
        out[split] = {"raw": raw, "anonymized": anon, "majority_rate": maj}
        md += [f"## {split} split (majority-class rate {maj:.1%})", "",
               "| domain | test traces | raw lookup acc | raw worlds seen | anonymized lookup acc |",
               "|---|---|---|---|---|"]
        for d in list(config.DOMAINS) + ["all"]:
            md.append(f"| {d} | {raw[d]['n']} | {raw[d]['acc']:.1%} | {raw[d]['seen']} | {anon[d]['acc']:.1%} |")
        md.append("")

    # Residual audit on the task-disjoint split, every part.
    n_traces, n_hit, strings = 0, 0, Counter()
    for part in config.PARTS:
        raw_by_id = {r["id"]: r for r in load_part("taskdisjoint", part)}
        for a in load_anon("taskdisjoint", part):
            hits = residual_hits(raw_by_id[a["id"]], a)
            n_traces += 1
            n_hit += bool(hits)
            strings.update(hits)
    out["residual"] = {"traces": n_traces, "traces_with_residual": n_hit, "top": strings.most_common(20)}
    md += ["## Residual world strings after anonymization (task-disjoint, all parts)", "",
           f"{n_hit} of {n_traces} traces still contain a world name, company or address string.",
           "Top leftovers: " + (", ".join(f"`{s}` x{c}" for s, c in strings.most_common(10)) or "none"),
           "", "A leftover can be a false alarm, e.g. a surname that is also an ordinary word in an "
           "args key such as 'White Blood Cell Count'. Look at the list before worrying.", ""]

    (config.REPORTS / "03_leakage.json").write_text(json.dumps(out, indent=2))
    (config.REPORTS / "03_leakage.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
