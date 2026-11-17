"""Week 2 · Export a trained monitor to ONNX (float + int8) and build the board bundle.

One ONNX call = one decision: the prefix's event stream goes in, calibrated P(hijacked) comes out.
int8 uses ONNX Runtime dynamic quantization (int8 weights, activations quantized on the fly).
ORT cannot quantize GRU, so a GRU's "int8" file stays float; use the LSTM or transformer for int8.

Checks on the laptop:
  - torch vs ONNX float vs ONNX int8 on every test decision point (max difference, AUROC, F1, verdict flips)
  - featurize + model latency per decision (median / p95, single thread)

Run:  python -m l2e.export_onnx                          (the best hash monitor from train_monitor)
      python -m l2e.export_onnx --arch transformer        (any trained monitor)
Out:  board/models/monitor_<arch>_<features>[_int8].onnx, board/models/monitor_bundle_<arch>_<features>.json,
      board/models/monitor_sample_traces.jsonl, board/step_features.py (copy for the board),
      reports/week2/02_export_<arch>_<features>.md and .json
"""
import argparse
import json
import shutil
import statistics
import time

import numpy as np
import pandas as pd
import torch

from . import config
from .monitor import EventData, LastStep, load, make_featurizer, metrics
from .step_features import tool_ids

BOARD = config.ROOT / "board"
BOARD_MODELS = BOARD / "models"
N_SAMPLE_TRACES = 120


def export(wrapper, tool, x, path):
    kw = dict(input_names=["tool", "x"], output_names=["p"], opset_version=17,
              dynamic_axes={"tool": {1: "events"}, "x": {1: "events"}})
    try:
        torch.onnx.export(wrapper, (tool, x), str(path), dynamo=False, **kw)  # TorchScript exporter
    except TypeError:  # older torch without the `dynamo` argument
        torch.onnx.export(wrapper, (tool, x), str(path), **kw)


def quantize(src, dst):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)


def session(path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def prefix(steps, t):
    """What the board sees at decision t: earlier steps in full, the current call without its result."""
    cur = steps[t]
    return [dict(s) for s in steps[:t]] + [{"thought": cur["thought"], "tool": cur["tool"], "args": cur["args"]}]


def timed(fn, n):
    for _ in range(20):
        fn()
    ts = []
    for _ in range(n):
        s = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - s) * 1000)
    ts.sort()
    return round(statistics.median(ts), 4), round(ts[int(0.95 * len(ts)) - 1], 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="hash", help="feature set the monitor was trained on")
    ap.add_argument("--arch", default=None, help="default: models/monitor_best_<encoder>.json")
    args = ap.parse_args()
    arch = args.arch or json.loads((config.MODELS / f"monitor_best_{args.encoder}.json").read_text())["arch"]
    name = f"{arch}_{args.encoder}"
    BOARD_MODELS.mkdir(parents=True, exist_ok=True)
    config.ensure_dirs()
    torch.set_num_threads(1)

    model, ck = load(config.MODELS / f"monitor_{name}.pt")
    wrapper = LastStep(model, ck["temperature"]).eval()
    data = EventData(args.encoder)
    dps = data.dps.reset_index()
    test = dps[dps["part"] == "test"]

    # 1. export float + int8
    tools, X, _ = data.ev[test["traj_id"].iloc[0]]
    f_path, q_path = BOARD_MODELS / f"monitor_{name}.onnx", BOARD_MODELS / f"monitor_{name}_int8.onnx"
    export(wrapper, torch.from_numpy(tool_ids(tools, ck["vocab"]))[None], torch.from_numpy(X)[None], f_path)
    quantize(f_path, q_path)
    sess = {"float": session(f_path), "int8": session(q_path)}

    # 2. torch vs ONNX on every test decision point
    rows = []
    for tid, g in test.groupby("traj_id", sort=False):
        tools, X, calls = data.ev[tid]
        ti = tool_ids(tools, ck["vocab"])
        with torch.no_grad():
            lg = model(torch.from_numpy(ti)[None], torch.from_numpy(X)[None])[0]
        for t, y in zip(g["step"], g["y"]):
            L = calls[t] + 1
            feed = {"tool": ti[None, :L], "x": X[None, :L]}
            rows.append({"dp_id": f"{tid}:{t}", "y": y,
                         "torch": float(torch.sigmoid(lg[calls[t]] / ck["temperature"])),
                         "float": float(sess["float"].run(None, feed)[0][0, 0]),
                         "int8": float(sess["int8"].run(None, feed)[0][0, 0])})
    r = pd.DataFrame(rows)
    acc = {k: {m: round(v, 4) for m, v in metrics(r["y"], r[k]).items()} for k in ("torch", "float", "int8")}
    diff = {k: float((r[k] - r["torch"]).abs().max()) for k in ("float", "int8")}
    flips = int(((r["int8"] >= 0.5) != (r["torch"] >= 0.5)).sum())

    # 3. board bundle: featurizer config + vocab, a sample of test traces, expected outputs
    feat = make_featurizer(args.encoder, ck["featurizer"])
    rng = np.random.RandomState(config.SEED)
    sample = [data.trajs[i] for i in rng.choice(sorted(test["traj_id"].unique()), N_SAMPLE_TRACES, replace=False)]
    expected = {}
    for tr in sample:
        for t in range(len(tr["steps"])):
            tools, X, _ = feat.events(prefix(tr["steps"], t))
            expected[f"{tr['id']}:{t}"] = float(sess["float"].run(
                None, {"tool": tool_ids(tools, ck["vocab"])[None], "x": X[None]})[0][0, 0])
    with open(BOARD_MODELS / "monitor_sample_traces.jsonl", "w") as f:
        for tr in sample:
            f.write(json.dumps({"id": tr["id"], "steps": tr["steps"]}) + "\n")
    bundle = {"name": name, "arch": arch, "features": args.encoder, "temperature": ck["temperature"],
              "vocab": ck["vocab"], "featurizer": ck["featurizer"], "expected": expected,
              "float": f_path.name, "int8": q_path.name}
    (BOARD_MODELS / f"monitor_bundle_{name}.json").write_text(json.dumps(bundle))
    shutil.copy(config.ROOT / "l2e" / "step_features.py", BOARD / "step_features.py")

    # 4. laptop latency on the longest sample trace: featurize the prefix, then run the model
    tr = max(sample, key=lambda s: len(s["steps"]))
    pre = prefix(tr["steps"], len(tr["steps"]) - 1)
    tools, X, _ = feat.events(pre)
    feed = {"tool": tool_ids(tools, ck["vocab"])[None], "x": X[None]}
    lat = {"featurize": timed(lambda: feat.events(pre), 300)}
    for k in ("float", "int8"):
        lat[f"model ({k})"] = timed(lambda: sess[k].run(None, feed), 300)
    size = {k: round(p.stat().st_size / 1024, 1) for k, p in (("float", f_path), ("int8", q_path))}

    md = [f"# Week 2 · ONNX export: {arch} + {args.encoder}", "",
          f"Files: `board/models/{f_path.name}` ({size['float']} KB) and `board/models/{q_path.name}` ({size['int8']} KB).", "",
          f"All {len(r)} task-disjoint test decision points, calibrated scores, threshold 0.5:", "",
          "| version | AUROC | AUPRC | F1 | max abs diff vs torch |", "|---|---|---|---|---|",
          f"| torch | {acc['torch']['auroc']:.4f} | {acc['torch']['auprc']:.4f} | {acc['torch']['f1']:.4f} | 0 |",
          f"| ONNX float | {acc['float']['auroc']:.4f} | {acc['float']['auprc']:.4f} | {acc['float']['f1']:.4f} | {diff['float']:.1e} |",
          f"| ONNX int8 | {acc['int8']['auroc']:.4f} | {acc['int8']['auprc']:.4f} | {acc['int8']['f1']:.4f} | {diff['int8']:.1e} |", "",
          f"int8 changes the verdict on {flips} of {len(r)} decision points.", "",
          f"Laptop latency for one decision on the longest sample trace ({len(tr['steps'])} steps, {X.shape[0]} events), "
          "single thread:", "", "| stage | median ms | p95 ms |", "|---|---|---|"]
    md += [f"| {k} | {v[0]:.3f} | {v[1]:.3f} |" for k, v in lat.items()]
    md += ["", f"Board bundle: `board/models/monitor_bundle_{name}.json` (featurizer config, tool vocabulary, expected "
           f"outputs for {len(expected)} decisions from {N_SAMPLE_TRACES} test traces) plus `board/step_features.py`.",
           f"On the board: `python board/time_monitor.py --bundle models/monitor_bundle_{name}.json`", ""]
    (config.REPORTS2 / f"02_export_{name}.md").write_text("\n".join(md))
    (config.REPORTS2 / f"02_export_{name}.json").write_text(json.dumps(
        {"size_kb": size, "accuracy": acc, "max_abs_diff": diff, "int8_verdict_flips": flips,
         "latency_ms_median_p95": lat}, indent=2))
    print("\n".join(md))


if __name__ == "__main__":
    main()
