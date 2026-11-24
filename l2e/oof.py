"""Out-of-fold monitor predictions and error labels, for every split (week 2; week 3 adds domain folds).

The deferrer learns "is the monitor wrong here?". Its training labels must come from monitor
predictions on data that monitor never trained on, otherwise the monitor looks better than it is and
the deferrer learns from too few, too easy errors. So, for each split:

  1. Folds over the split's training traces (l2e/folds.py):
       --fold-by task    (week 2) K folds grouped by task text (a task is never on both sides)
       --fold-by domain  (week 3) one fold per training domain: each fold model never saw the domain it scores
     Each fold's model scores its held-out fold -> out-of-fold ("oof") scores for every training DP.
  2. A full model on all training DPs scores val and test, plus "context": the other calls of any trace
     that is only partly in a test set (the tool-coverage sets), so the deferrer's "earlier in this trace"
     features are exact. With --fold-by domain the full model is the same as with task folds, so it is
     reused from the task-fold run of the same seed when that exists.
  3. Every model gets its own temperature, fit on its val DPs (domain folds: only its own domains' val DPs).
  4. Verdict at 0.5 -> error labels (labels.error_labels): dangerous = missed hijack, costly = false alarm.

Splits: taskdisjoint · tool_seed0..2 (test sets at 100/75/50/25% tool coverage) · lodo_<domain> x5.

Run:  python -m l2e.oof                                    (best hash monitor, all splits, ~6 trainings per split)
      python -m l2e.oof --splits taskdisjoint tool_seed0   (a subset)
      python -m l2e.oof --encoder hash-roles --arch transformer --fold-by domain --splits lodo_web
      python -m l2e.oof --encoder hash-roles --arch transformer --seed 1     (another monitor seed)
      python -m l2e.oof --encoder hash-roles --arch transformer --context-only --splits tool_seed0
                                                           (add context rows to an existing run, no training)
      python -m l2e.oof --encoder hash-roles --arch transformer --seed 1 --full-only --splits taskdisjoint
                                                           (only the full model: a seed-ensemble member)
Out:  data/processed/oof/<tag>/<split>.csv  one row per (set, decision point), tag = config.oof_tag(...):
        dp_id, split, set (oof | val | test | cov100 ... | context), fold, logit, temperature, score, pred, y,
        tool_seen (tool in that model's training vocabulary), monitor_error, dangerous, costly, weight
      models/oof/<tag>/<split>_full.pt
      reports/week2/03_errors_<tag>.md and .json (the week 2 run), reports/week3/... for domain folds and seeds
"""
import argparse
import json
import shutil
import time

import numpy as np
import pandas as pd

from . import config
from .calibrate import fit_temperature, sigmoid
from .folds import context_ids, fold_plan, oof_path, split_defs
from .labels import error_counts, error_labels
from .monitor import EventData, load, metrics, predict, save, train, train_device
from .step_features import default_cfg

COLS = ["dp_id", "split", "set", "fold", "logit", "temperature", "score", "pred", "y", "tool_seen",
        "monitor_error", "dangerous", "costly", "weight"]


def score(model, vocab, data, ids):
    ids_out, lg = predict(model, data.items(ids, vocab, truncate=False))
    df = pd.DataFrame({"dp_id": ids_out, "logit": lg})
    df["tool_seen"] = data.dps.loc[df["dp_id"], "tool"].isin(set(vocab)).values.astype(int)
    return df


def finish(df, name, y):
    """Calibrated score, verdict and error labels for scored rows."""
    df = df.copy()
    df["split"] = name
    df["y"] = df["dp_id"].map(y)
    df["score"] = sigmoid(df["logit"] / df["temperature"])
    df["pred"] = (df["score"] >= 0.5).astype(int)
    df = pd.concat([df.reset_index(drop=True), error_labels(df["y"], df["pred"])], axis=1)
    return df[COLS]


def full_model_rows(model, vocab, data, sp):
    """val, test sets and context, scored by the full model."""
    y = data.y
    val = score(model, vocab, data, sp["val"])
    T = fit_temperature(val["logit"], val["dp_id"].map(y))
    rows = [val.assign(set="val", fold=-1, temperature=T)]
    for set_name, ids in sp["test"].items():
        rows.append(score(model, vocab, data, ids).assign(set=set_name, fold=-1, temperature=T))
    ctx = context_ids(data.dps, sp["test"])
    if ctx:
        rows.append(score(model, vocab, data, ctx).assign(set="context", fold=-1, temperature=T))
    return rows, T


def reusable_full(base_tag, name):
    """Full-model rows (val, test, context) from the task-fold run of the same seed, if it has them all."""
    path = oof_path(base_tag, name)
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df[df["set"] != "oof"]


def run_split(data, name, sp, arch, folds, hp, out_dir, model_dir, feature_set, fold_by="task", seed=config.SEED,
              base_tag=None):
    y = data.y
    task_of = {t: tr["task"] for t, tr in data.trajs.items()}
    rows = []
    plan = [] if fold_by == "full" else fold_plan(data.dps, sp["train"], sp["val"], fold_by, folds, task_of)
    for k, tr_ids, ho_ids, va_ids in plan:
        t0 = time.time()
        model, vocab, _ = train(data, tr_ids, va_ids, arch=arch, seed=seed, hp=hp, verbose=False)
        val = score(model, vocab, data, va_ids)
        T = fit_temperature(val["logit"], val["dp_id"].map(y))
        rows.append(score(model, vocab, data, ho_ids).assign(set="oof", fold=k, temperature=T))
        print(f"  {name} fold {k + 1} ({fold_by}): {len(tr_ids)} train DPs, {len(ho_ids)} held out, T {T:.2f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    reused = reusable_full(base_tag, name) if fold_by == "domain" and base_tag else None
    if reused is not None:
        src = config.MODELS / "oof" / base_tag / f"{name}_full.pt"
        if src.exists():
            shutil.copy(src, model_dir / f"{name}_full.pt")
        df = pd.concat([finish(pd.concat(rows, ignore_index=True), name, y), reused[COLS]], ignore_index=True)
        print(f"  {name} full model: reused from {base_tag}", flush=True)
        if "context" not in set(df["set"]) and context_ids(data.dps, sp["test"]):
            df.to_csv(out_dir / f"{name}.csv", index=False)
            return add_context(data, name, sp, out_dir, model_dir)
    else:
        t0 = time.time()
        model, vocab, _ = train(data, sp["train"], sp["val"], arch=arch, seed=seed, hp=hp, verbose=False)
        full_rows, T = full_model_rows(model, vocab, data, sp)
        save(model_dir / f"{name}_full.pt", model, vocab, T, arch, feature_set, default_cfg(feature_set),
             extra={"seed": seed})
        print(f"  {name} full model: T {T:.2f} ({time.time() - t0:.0f}s)", flush=True)
        df = finish(pd.concat(rows + full_rows, ignore_index=True), name, y)
    df.to_csv(out_dir / f"{name}.csv", index=False)
    return df


def add_context(data, name, sp, out_dir, model_dir):
    """Add (or refresh) the context rows of an existing run, using its saved full model. No training."""
    path = out_dir / f"{name}.csv"
    df = pd.read_csv(path)
    model, ck = load(model_dir / f"{name}_full.pt")
    ctx = context_ids(data.dps, sp["test"])
    df = df[df["set"] != "context"]
    if ctx:
        rows = score(model, ck["vocab"], data, ctx).assign(set="context", fold=-1, temperature=ck["temperature"])
        df = pd.concat([df, finish(rows, name, data.y)], ignore_index=True)
    df.to_csv(path, index=False)
    print(f"  {name}: {len(ctx)} context rows", flush=True)
    return df


def summary(df):
    out = {}
    for s, g in df.groupby("set", sort=False):
        c = error_counts(g["y"], g["pred"])
        c["hijacked"] = int(g["y"].sum())
        c["auroc"] = round(metrics(g["y"], g["score"])["auroc"], 4) if g["y"].nunique() > 1 else None
        c["tool_seen_rate"] = round(float(g["tool_seen"].mean()), 3)
        out[s] = c
    return out


def write_report(all_sum, tag, folds):
    how = {"domain": "one fold per training domain (each fold model scores a domain it never saw)",
           "full": "none here: this run trains only the full model, as a seed-ensemble member"}.get(
        folds, f"{folds} folds grouped by task")
    md = [f"# Out-of-fold errors and held-out error counts ({tag})", "",
          f"Out-of-fold predictions on each split's training DPs ({how}), and a full "
          "model on all training DPs for val and test. Each model is temperature-scaled on the split's val set; "
          "verdict at 0.5. Dangerous = missed hijack, costly = false alarm.", "",
          "## Held-out error counts (the plan's table)", "",
          "| held-out split | decision points | hijacked | monitor errors | dangerous | costly | AUROC |",
          "|---|---|---|---|---|---|---|"]

    def line(label, c):
        au = f"{c['auroc']:.3f}" if c["auroc"] is not None else "n/a"
        return (f"| {label} | {c['decision_points']} | {c['hijacked']} | {c['monitor_errors']} | {c['dangerous']} | "
                f"{c['costly']} | {au} |")

    if "taskdisjoint" in all_sum:
        md.append(line("Task-disjoint test", all_sum["taskdisjoint"]["test"]))
    seeds = [k for k in all_sum if k.startswith("tool_seed")]
    for lvl in ("cov100", "cov75", "cov50", "cov25"):
        for s in seeds:
            md.append(line(f"Tool coverage {lvl[3:]}%, {s.replace('tool_', '')}", all_sum[s][lvl]))
    for dom in config.DOMAINS:
        if f"lodo_{dom}" in all_sum:
            md.append(line(f"Leave out {dom}", all_sum[f"lodo_{dom}"]["test"]))
    if seeds:
        md += ["", "Tool coverage, mean over seeds:", "", "| coverage | decision points | errors | dangerous | costly | AUROC |",
               "|---|---|---|---|---|---|"]
        for lvl in ("cov100", "cov75", "cov50", "cov25"):
            cs = [all_sum[s][lvl] for s in seeds]
            m = lambda k: np.mean([c[k] for c in cs])
            md.append(f"| {lvl[3:]}% | {m('decision_points'):.0f} | {m('monitor_errors'):.1f} | {m('dangerous'):.1f} | "
                      f"{m('costly'):.1f} | {m('auroc'):.3f} |")
    md += ["", "## Deferrer training material: out-of-fold errors on each split's training DPs", "",
           "| split | training DPs | hijacked | OOF errors | dangerous | costly | OOF AUROC | val AUROC |",
           "|---|---|---|---|---|---|---|---|"]
    for name, s in all_sum.items():
        if "oof" not in s:  # a full-only run has no out-of-fold rows
            continue
        o, v = s["oof"], s["val"]
        md.append(f"| {name} | {o['decision_points']} | {o['hijacked']} | {o['monitor_errors']} | {o['dangerous']} | "
                  f"{o['costly']} | {o['auroc']:.3f} | {v['auroc']:.3f} |")
    md.append("")
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="hash", help="feature set (hash, hash-nothought, minilm, ...)")
    ap.add_argument("--arch", default=None, help="default: models/monitor_best_<encoder>.json")
    ap.add_argument("--splits", nargs="+", default=None)
    ap.add_argument("--folds", type=int, default=config.OOF_FOLDS, help="number of task folds")
    ap.add_argument("--fold-by", choices=config.FOLD_BY, default="task")
    ap.add_argument("--seed", type=int, default=config.SEED, help="monitor training seed")
    ap.add_argument("--context-only", action="store_true", help="only add context rows to an existing run")
    ap.add_argument("--full-only", action="store_true",
                    help="train only the full model (an extra seed for the seed-ensemble policies, no folds)")
    ap.add_argument("--quick", action="store_true", help="3 epochs per model, smoke test")
    args = ap.parse_args()
    arch = args.arch or json.loads((config.MODELS / f"monitor_best_{args.encoder}.json").read_text())["arch"]
    if args.full_only:
        args.fold_by = "full"
    tag = config.oof_tag(arch, args.encoder, args.seed, args.fold_by)
    base_tag = config.oof_tag(arch, args.encoder, args.seed, "task")
    hp = {"max_epochs": 3} if args.quick else {}
    out_dir = config.PROCESSED / "oof" / tag
    model_dir = config.MODELS / "oof" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    config.ensure_dirs()

    data = EventData(args.encoder)
    print(f"monitor training device: {train_device()} (set L2E_DEVICE=auto to use a GPU)", flush=True)
    defs = split_defs(data.dps.reset_index())
    names = args.splits or list(defs)
    if args.context_only:
        for name in names:
            add_context(data, name, defs[name], out_dir, model_dir)
        return
    week2 = args.fold_by == "task" and args.seed == config.SEED
    rep_dir = config.REPORTS2 if week2 else config.REPORTS3
    sum_path = rep_dir / f"03_errors_{tag}.json"
    all_sum = json.loads(sum_path.read_text()) if sum_path.exists() and args.splits else {}
    for name in names:
        print(f"split {name} ({arch}, {args.encoder}, seed {args.seed}, {args.fold_by} folds) ...", flush=True)
        df = run_split(data, name, defs[name], arch, args.folds, hp, out_dir, model_dir, args.encoder,
                       args.fold_by, args.seed, base_tag)
        all_sum[name] = summary(df)
        sum_path.write_text(json.dumps(all_sum, indent=2))  # save as we go
    ordered = {k: all_sum[k] for k in defs if k in all_sum}
    md = write_report(ordered, tag, args.folds if args.fold_by == "task" else args.fold_by)
    if args.quick:
        md = md.replace("\n\n", "\n\n> QUICK RUN (3 epochs). Do not report these numbers.\n\n", 1)
    (rep_dir / f"03_errors_{tag}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
