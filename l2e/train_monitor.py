"""Week 2 · Train the compact neural monitors on the task-disjoint split and compare with the LR.

For each architecture: train on train, early-stop and fit the temperature on val, report on test.
The logistic regression baseline gets the same temperature scaling, so the comparison is fair.

Run:  python -m l2e.train_monitor                       (hashed text; lstm, gru, transformer)
      python -m l2e.train_monitor --encoder minilm      (after `python -m l2e.encoders embed minilm`)
      python -m l2e.train_monitor --encoder mpnet --archs transformer_big    ("better monitor", laptop only)
      python -m l2e.train_monitor --encoder hash-nothought   (ablation: agent thoughts hidden)
      python -m l2e.train_monitor --quick               (3 epochs, smoke test)
Out:  models/monitor_<arch>_<encoder>.pt, models/monitor_best_<encoder>.json,
      data/processed/preds/monitor_<arch>_<encoder>.csv, reports/week2/01_monitor_<encoder>.md and .json
"""
import argparse
import io
import json
import time

import joblib
import pandas as pd
import torch

from . import config
from .calibrate import ece, fit_temperature, sigmoid
from .features import prefix_matrix
from .labels import error_counts
from .monitor import EventData, metrics, n_params, predict, save, train
from .step_features import default_cfg

LR_NAME = "logistic regression (6 features)"
# The main (edge) monitor must quantize to int8 on the board; ONNX Runtime can't quantize GRU (week 1).
INT8_ARCHS = ("lstm", "transformer")
TIE = 0.001  # validation AUPRC gap treated as a tie; ties go to the smaller model (it's an edge monitor)


def state_kb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 1024


def lr_baseline(data, parts):
    """Six-feature LR from week 1, temperature-scaled on val."""
    model = joblib.load(config.MODELS / "lr_prefix_monitor.joblib")
    out = {p: pd.DataFrame({"dp_id": parts[p]["dp_id"].values,
                            "logit": model.decision_function(prefix_matrix(data.trajs, parts[p]))})
           for p in ("val", "test")}
    val = parts["val"].merge(out["val"], on="dp_id")
    return out, fit_temperature(val["logit"], val["y"])


def summarize(name, preds, T, parts, params=None, kb=None, secs=None):
    te = parts["test"].merge(preds["test"], on="dp_id")
    y = te["y"].values
    p_raw, p_cal = sigmoid(te["logit"].values), sigmoid(te["logit"].values / T)
    sc = te["state_changing"].values.astype(bool)
    m, msc = metrics(y, p_cal), metrics(y[sc], p_cal[sc])
    row = {"model": name, "params": params, "size_kb": kb, "train_sec": secs, "temperature": T,
           **{k: round(v, 4) for k, v in m.items()}, "ece_raw": round(ece(p_raw, y), 4),
           "auroc_state_changing": round(msc["auroc"], 4), "f1_state_changing": round(msc["f1"], 4),
           **error_counts(y, (p_cal >= 0.5).astype(int))}
    te = te.assign(score=p_cal, pred=(p_cal >= 0.5).astype(int))
    per_dom = {dom: error_counts(g["y"], g["pred"]) for dom, g in te.groupby("domain")}
    return row, per_dom, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="hash", help="feature set: hash, minilm, mpnet, optionally + '-nothought'")
    ap.add_argument("--archs", nargs="+", default=["lstm", "gru", "transformer"])
    ap.add_argument("--quick", action="store_true", help="3 epochs, for a smoke test")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    config.ensure_dirs()
    (config.PROCESSED / "preds").mkdir(exist_ok=True)
    hp = {"max_epochs": 3} if args.quick else {}

    data = EventData(args.encoder)
    dps = data.dps.reset_index()
    parts = {p: dps[dps["part"] == p].reset_index(drop=True) for p in config.PARTS}

    rows, doms, hists = [], {}, {}
    lr_preds, lr_T = lr_baseline(data, parts)
    r, d, _ = summarize(LR_NAME, lr_preds, lr_T, parts, params=7)
    rows.append(r)
    doms[r["model"]] = d

    for arch in args.archs:
        print(f"training {arch} on '{args.encoder}' features ...", flush=True)
        t0 = time.time()
        model, vocab, hist = train(data, parts["train"]["dp_id"], parts["val"]["dp_id"], arch=arch, hp=hp)
        secs = round(time.time() - t0)
        preds = {}
        for part in ("val", "test"):
            ids, lg = predict(model, data.items(parts[part]["dp_id"], vocab, truncate=False))
            preds[part] = pd.DataFrame({"dp_id": ids, "logit": lg})
        val = parts["val"].merge(preds["val"], on="dp_id")
        T = fit_temperature(val["logit"], val["y"])
        name = f"{arch} + {args.encoder}"
        r, d, te = summarize(name, preds, T, parts, n_params(model), round(state_kb(model), 1), secs)
        r["epochs"] = len(hist)
        r["best_val_auprc"] = round(max(h["val_auprc"] for h in hist), 4)
        rows.append(r)
        doms[name] = d
        hists[name] = hist
        save(config.MODELS / f"monitor_{arch}_{args.encoder}.pt", model, vocab, T, arch, args.encoder, default_cfg(args.encoder))
        te[["dp_id", "domain", "tool", "state_changing", "y", "logit", "score", "pred"]].to_csv(
            config.PROCESSED / "preds" / f"monitor_{arch}_{args.encoder}.csv", index=False)
        print(f"  test AUROC {r['auroc']:.4f}  AUPRC {r['auprc']:.4f}  F1 {r['f1']:.4f}  T {T:.2f}  ({secs}s)", flush=True)

    neural = [r for r in rows if r["model"] != LR_NAME]
    cands = [r for r in neural if r["model"].split(" + ")[0] in INT8_ARCHS] or neural
    top = max(r["best_val_auprc"] for r in cands)
    best = min([r for r in cands if r["best_val_auprc"] >= top - TIE], key=lambda r: r["params"])
    best_arch = best["model"].split(" + ")[0]
    (config.MODELS / f"monitor_best_{args.encoder}.json").write_text(json.dumps(
        {"arch": best_arch, "encoder": args.encoder, "path": f"monitor_{best_arch}_{args.encoder}.pt",
         "chosen_by": f"best validation AUPRC among int8-quantizable architectures; within {TIE} the smaller model"},
        indent=2))

    md = [f"# Week 2 · Compact neural monitors ({args.encoder} text features)", "",
          "Task-disjoint split, every decision point. Temperature fit on val; test metrics use calibrated "
          "scores and threshold 0.5.",
          f"{len(parts['test'])} test decision points ({int(parts['test']['y'].sum())} hijacked).", "",
          "| model | params | size (KB) | epochs | AUROC | AUPRC | F1 | precision | recall | ECE raw → calibrated | T |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['params'] or ''} | {r['size_kb'] or ''} | {r.get('epochs', '')} | "
                  f"{r['auroc']:.3f} | {r['auprc']:.3f} | {r['f1']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | "
                  f"{r['ece_raw']:.3f} → {r['ece']:.3f} | {r['temperature']:.2f} |")
    md += ["", "State-changing calls only:", "", "| model | AUROC | F1 |", "|---|---|---|"]
    md += [f"| {r['model']} | {r['auroc_state_changing']:.3f} | {r['f1_state_changing']:.3f} |" for r in rows]
    md += ["", "Monitor errors on test (threshold 0.5), with dangerous / costly per domain:", "",
           "| model | errors | dangerous | costly | " + " | ".join(config.DOMAINS) + " |",
           "|---|---|---|---|" + "---|" * len(config.DOMAINS)]
    for r in rows:
        d = doms[r["model"]]
        md.append(f"| {r['model']} | {r['monitor_errors']} | {r['dangerous']} | {r['costly']} | " +
                  " | ".join(f"{d[x]['dangerous']} / {d[x]['costly']}" for x in config.DOMAINS) + " |")
    md += ["", "Best validation AUPRC: " + ", ".join(f"{r['model'].split(' + ')[0]} {r['best_val_auprc']:.4f}" for r in neural) + ".",
           f"Chosen edge monitor (best validation AUPRC among int8-quantizable architectures, {', '.join(INT8_ARCHS)}; "
           f"gaps under {TIE} count as ties and go to the smaller model): "
           f"**{best_arch}**, recorded in `models/monitor_best_{args.encoder}.json`.", ""]
    if args.quick:
        md.insert(1, "> QUICK RUN (3 epochs). Do not report these numbers.\n")
    (config.REPORTS2 / f"01_monitor_{args.encoder}.md").write_text("\n".join(md))
    (config.REPORTS2 / f"01_monitor_{args.encoder}.json").write_text(json.dumps(
        {"rows": rows, "per_domain": doms, "history": hists, "best": best_arch, "quick": args.quick},
        indent=2, default=float))
    print("\n".join(md))


if __name__ == "__main__":
    main()
