"""Week 2 board timing: replay sample test traces through the real monitor, one decision at a time.

Runs ON the UNO Q (also on a laptop, for comparison). Needs only numpy and onnxruntime, plus the files
that `python -m l2e.export_onnx` writes into board/: step_features.py and models/monitor_*.

For every decision point of every sample trace it times
  featurize  build the event stream for the prefix (earlier steps + the proposed call)
  model      one ONNX Runtime call, float and int8
and checks that the float model's output matches the laptop's.

MiniLM monitors also run the sentence encoder on the board (models/minilm_int8.onnx + the tokenizer,
from `python -m l2e.encoders export-minilm`; needs `pip install tokenizers` on the board). Each new
event is encoded once and cached, as a live monitor would: decision t encodes only obs_{t-1} and call_t.
The laptop used float sentence-transformers embeddings, so outputs differ slightly; the check then
reports verdict agreement instead of an exact match.

Run:  python board/time_monitor.py --bundle models/monitor_bundle_lstm_hash.json
      python board/time_monitor.py --bundle models/monitor_bundle_lstm_minilm.json
Out:  board/results/monitor_timing_<name>_<hostname>.json
"""
import argparse
import json
import platform
import resource
import socket
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

from step_features import Featurizer, tool_ids

HERE = Path(__file__).resolve().parent


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 / 1024 if platform.system() == "Darwin" else r / 1024


def stats(ms):
    ms = sorted(ms)
    return {"median_ms": round(statistics.median(ms), 3), "p95_ms": round(ms[int(0.95 * len(ms)) - 1], 3),
            "max_ms": round(ms[-1], 3)}


def session(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def prefix(steps, t):
    cur = steps[t]
    return [dict(s) for s in steps[:t]] + [{"thought": cur["thought"], "tool": cur["tool"], "args": cur["args"]}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="path relative to board/, e.g. models/monitor_bundle_lstm_hash.json")
    ap.add_argument("--repeats", type=int, default=3, help="passes over the sample traces")
    args = ap.parse_args()
    bundle = json.loads((HERE / args.bundle).read_text())
    encoder = bundle["features"].split("-")[0]
    cache, t_enc = {}, []
    if encoder == "hash":
        feat = Featurizer(bundle["featurizer"])
    elif encoder == "minilm":
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(HERE / "models" / "minilm_tokenizer.json"))
        tok.enable_truncation(256)
        enc_sess = session(HERE / "models" / "minilm_int8.onnx")

        def lookup(text):
            if text not in cache:
                s = time.perf_counter()
                e = tok.encode(text)
                cache[text] = enc_sess.run(None, {"input_ids": np.array([e.ids], np.int64),
                                                  "attention_mask": np.array([e.attention_mask], np.int64)})[0][0]
                t_enc.append((time.perf_counter() - s) * 1000)
            return cache[text]
        feat = Featurizer(bundle["featurizer"], text_lookup=lookup, text_dim=384)
    else:
        raise SystemExit(f"{encoder} is laptop-only")
    sess = {k: session(HERE / "models" / bundle[k]) for k in ("float", "int8")}
    traces = [json.loads(l) for l in open(HERE / "models" / "monitor_sample_traces.jsonl")]

    t_feat, t_model = [], {"float": [], "int8": []}
    max_diff, n, flips = 0.0, 0, 0
    for _ in range(args.repeats):
        for tr in traces:
            cache.clear()  # a fresh agent session: nothing encoded yet
            for t in range(len(tr["steps"])):
                pre = prefix(tr["steps"], t)
                s = time.perf_counter()
                tools, X, _ = feat.events(pre)
                feed = {"tool": tool_ids(tools, bundle["vocab"])[None], "x": X[None]}
                t_feat.append((time.perf_counter() - s) * 1000)
                for k in ("float", "int8"):
                    s = time.perf_counter()
                    p = float(sess[k].run(None, feed)[0][0, 0])
                    t_model[k].append((time.perf_counter() - s) * 1000)
                    if k == "float":
                        ref = bundle["expected"][f"{tr['id']}:{t}"]
                        max_diff = max(max_diff, abs(p - ref))
                        flips += (p >= 0.5) != (ref >= 0.5)
                n += 1

    total = {k: [a + b for a, b in zip(t_feat, t_model[k])] for k in t_model}
    report = {"host": socket.gethostname(), "machine": platform.machine(), "python": platform.python_version(),
              "onnxruntime": ort.__version__, "model": bundle["name"], "decisions_timed": n,
              "size_kb": {k: round((HERE / "models" / bundle[k]).stat().st_size / 1024, 1) for k in ("float", "int8")},
              "featurize": stats(t_feat), "model_float": stats(t_model["float"]), "model_int8": stats(t_model["int8"]),
              "total_float": stats(total["float"]), "total_int8": stats(total["int8"]),
              "max_abs_diff_vs_laptop": max_diff, "verdict_flips_vs_laptop": int(flips),
              "matches_laptop": max_diff < 1e-3 if encoder == "hash" else flips == 0,
              "peak_rss_mb": round(peak_rss_mb(), 1)}
    if t_enc:
        report["encoder_per_new_text"] = stats(t_enc)
    out = HERE / "results" / f"monitor_timing_{bundle['name']}_{socket.gethostname()}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("RESULT:", "PASS" if report["matches_laptop"] else "MISMATCH", "->", out)


if __name__ == "__main__":
    main()
