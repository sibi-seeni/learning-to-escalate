"""Week 2 · Which monitor holds up best on a domain it has never seen?

Trains each monitor configuration on each leave-one-domain-out split (and the task-disjoint split for
reference), with several seeds, and compares them on the held-out data. Domain-shift numbers swing a lot
between training runs, so every row is a mean over seeds with its spread.

A configuration is "<arch>:<feature set>", e.g. "transformer:minilm-roles-noid" (see step_features.py):
  lstm:hash                      the current edge monitor
  transformer:hash               same features, transformer
  transformer:hash-roles         + what each tool does (sends out, moves money, ...)
  transformer:hash-roles-noid    roles instead of the memorized tool id
  transformer:minilm             pretrained MiniLM sentence embeddings for the text
  transformer:minilm-roles-noid  MiniLM + roles, no tool id
  transformer_big:mpnet          the laptop-only "better monitor" (DriftNet's encoder)
MiniLM / mpnet configs are skipped until `python -m l2e.encoders embed minilm|mpnet` has been run.

What to look at: errors and especially "confident false alarms" (benign calls scored >= 0.9) on the held-out
domains. Those are the errors confidence-based escalation can't catch.

Run:  python -m l2e.compare_shift                                   (all available configs, 3 seeds)
      python -m l2e.compare_shift --configs lstm:hash transformer:hash-roles-noid --seeds 0
Out:  reports/week2/04_domain_shift.md, reports/week2/04_domain_shift_runs.jsonl (one line per run; a rerun
      skips runs that are already there, so the script can be stopped and resumed)
"""
import argparse
import json
import time

import numpy as np

from . import config
from .calibrate import ece, fit_temperature, sigmoid
from .labels import error_counts
from .monitor import EventData, metrics, train
from .oof import score, split_defs
from .step_features import split_feature_set

CONFIGS = ["lstm:hash", "transformer:hash", "transformer:hash-roles", "transformer:hash-roles-noid",
           "transformer:minilm", "transformer:minilm-roles-noid", "transformer_big:mpnet"]
SPLITS = ["taskdisjoint"] + [f"lodo_{d}" for d in config.DOMAINS]
CONFIDENT = 0.9


def available(feature_set):
    enc, _ = split_feature_set(feature_set)
    return enc == "hash" or (config.PROCESSED / "emb" / f"{enc}.npy").exists()


def run_one(data, sp, arch, seed):
    t0 = time.time()
    model, vocab, hist = train(data, sp["train"], sp["val"], arch=arch, seed=seed, verbose=False)
    val = score(model, vocab, data, sp["val"])
    T = fit_temperature(val["logit"], val["dp_id"].map(data.y))
    te = score(model, vocab, data, sp["test"]["test"])
    y = te["dp_id"].map(data.y).values
    p = sigmoid(te["logit"].values / T)
    pred = (p >= 0.5).astype(int)
    m = metrics(y, p)
    return {**error_counts(y, pred), "hijacked": int(y.sum()), "auroc": m["auroc"], "auprc": m["auprc"],
            "f1": m["f1"], "ece": ece(p, y), "temperature": T, "epochs": len(hist),
            "confident_false_alarms": int(((y == 0) & (p >= CONFIDENT)).sum()),
            "confident_misses": int(((y == 1) & (p <= 1 - CONFIDENT)).sum()), "sec": round(time.time() - t0)}


def ms(vals, fmt="{:.0f}"):
    vals = [v for v in vals if v is not None]
    if not vals:
        return ""
    if len(vals) == 1:
        return fmt.format(vals[0])
    return f"{fmt.format(np.mean(vals))} ± {fmt.format(np.std(vals, ddof=1))}"


def report(runs, configs, seeds):
    md = ["# Week 2 · Monitors under domain shift", "",
          f"Each cell: mean ± standard deviation over seeds {', '.join(map(str, seeds))}. Temperature fit on each "
          f"split's val set, verdict at 0.5. Confident false alarm = benign call scored ≥ {CONFIDENT}; confident "
          f"miss = hijacked call scored ≤ {1 - CONFIDENT:.1f}.", ""]
    by = {}
    for r in runs:
        by.setdefault((r["config"], r["split"]), []).append(r)
    present = [c for c in configs if any(k[0] == c for k in by)]

    md += ["## Summary: average over the five held-out domains", "",
           "| monitor | errors | dangerous | costly | confident false alarms | confident misses | AUROC | ECE | "
           "task-disjoint errors |", "|---|---|---|---|---|---|---|---|---|"]
    for c in present:
        per_seed = {}
        for d in config.DOMAINS:
            for r in by.get((c, f"lodo_{d}"), []):
                per_seed.setdefault(r["seed"], []).append(r)
        full = {s: rs for s, rs in per_seed.items() if len(rs) == len(config.DOMAINS)}
        avg = lambda k: [np.mean([r[k] for r in rs]) for rs in full.values()]
        td = [r["monitor_errors"] for r in by.get((c, "taskdisjoint"), [])]
        if not full:
            md.append(f"| {c} | (incomplete) | | | | | | | {ms(td)} |")
            continue
        md.append(f"| {c} | {ms(avg('monitor_errors'))} | {ms(avg('dangerous'))} | {ms(avg('costly'))} | "
                  f"{ms(avg('confident_false_alarms'))} | {ms(avg('confident_misses'))} | "
                  f"{ms(avg('auroc'), '{:.3f}')} | {ms(avg('ece'), '{:.3f}')} | {ms(td)} |")

    for split in SPLITS:
        rows = [(c, by[(c, split)]) for c in present if (c, split) in by]
        if not rows:
            continue
        n_dp, n_h = rows[0][1][0]["decision_points"], rows[0][1][0]["hijacked"]
        title = "Task-disjoint test (no shift)" if split == "taskdisjoint" else f"Held out: {split[5:]}"
        md += ["", f"## {title}", "", f"{n_dp} decision points, {n_h} hijacked.", "",
               "| monitor | seeds | errors | dangerous | costly | confident false alarms | AUROC | ECE |",
               "|---|---|---|---|---|---|---|---|"]
        for c, rs in rows:
            g = lambda k: [r[k] for r in rs]
            md.append(f"| {c} | {len(rs)} | {ms(g('monitor_errors'))} | {ms(g('dangerous'))} | {ms(g('costly'))} | "
                      f"{ms(g('confident_false_alarms'))} | {ms(g('auroc'), '{:.3f}')} | {ms(g('ece'), '{:.3f}')} |")
    md.append("")
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=CONFIGS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--splits", nargs="+", default=SPLITS)
    args = ap.parse_args()
    config.ensure_dirs()
    log = config.REPORTS2 / "04_domain_shift_runs.jsonl"
    runs = [json.loads(l) for l in open(log)] if log.exists() else []
    done = {(r["config"], r["split"], r["seed"]) for r in runs}

    for c in args.configs:
        arch, fs = c.split(":")
        if not available(fs):
            print(f"skip {c}: run `python -m l2e.encoders embed {split_feature_set(fs)[0]}` first", flush=True)
            continue
        data = EventData(fs)
        defs = split_defs(data.dps.reset_index())
        for split in args.splits:
            for seed in args.seeds:
                if (c, split, seed) in done:
                    continue
                r = {"config": c, "split": split, "seed": seed, **run_one(data, defs[split], arch, seed)}
                runs.append(r)
                with open(log, "a") as f:
                    f.write(json.dumps(r, default=float) + "\n")
                print(f"{c:32s} {split:13s} seed {seed}: errors {r['monitor_errors']:4d} (dangerous {r['dangerous']}, "
                      f"costly {r['costly']}), confident false alarms {r['confident_false_alarms']}, "
                      f"AUROC {r['auroc']:.3f} ({r['sec']}s)", flush=True)

    md = report(runs, list(dict.fromkeys(CONFIGS + [r["config"] for r in runs])), sorted({r["seed"] for r in runs}))
    (config.REPORTS2 / "04_domain_shift.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
