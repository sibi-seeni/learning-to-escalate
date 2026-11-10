"""Week 1 board check. Run ON the UNO Q (also works on a laptop, for comparison).

Checks that numpy, scikit-learn, onnxruntime and lightgbm import, that the stand-in models give the
same outputs as on the laptop, and times one decision (batch of 1) for each model.

Run:  python board/check_board.py          (needs board/models/ from make_test_models.py)
Out:  board/results/board_check_<hostname>.json
"""
import json
import platform
import resource
import socket
import statistics
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"


def ram_mb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemTotal"):
                return int(line.split()[1]) / 1024
    except OSError:
        return None


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024 / 1024 if platform.system() == "Darwin" else r / 1024  # macOS reports bytes, Linux KB


def time_it(fn, n=300, warmup=20):
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(n):
        s = time.perf_counter()
        fn()
        t.append((time.perf_counter() - s) * 1000)
    t.sort()
    return {"median_ms": round(statistics.median(t), 3), "p95_ms": round(t[int(0.95 * len(t)) - 1], 3)}


def main():
    report = {"host": socket.gethostname(), "machine": platform.machine(), "system": platform.platform(),
              "python": platform.python_version(), "cpus": __import__("os").cpu_count(),
              "ram_total_mb": ram_mb(), "packages": {}, "checks": {}}

    import numpy as np
    report["packages"]["numpy"] = np.__version__
    for mod in ("sklearn", "onnxruntime", "lightgbm"):
        try:
            report["packages"][mod] = __import__(mod).__version__
        except Exception as e:  # keep going so we see every missing package at once
            report["packages"][mod] = f"MISSING: {e}"

    exp = np.load(MODELS / "expected.npz")

    if not str(report["packages"]["onnxruntime"]).startswith("MISSING"):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1  # one decision at a time; also keeps cores free for the demo
        for name in ("tiny_gru", "tiny_gru_int8", "tiny_lstm", "tiny_lstm_int8"):
            sess = ort.InferenceSession(str(MODELS / f"{name}.onnx"), opts, providers=["CPUExecutionProvider"])
            out = sess.run(None, {"x": exp["x_seq"]})[0]
            one = exp["x_seq"][:1]
            report["checks"][name] = {
                "matches_laptop": bool(np.allclose(out, exp[name], atol=1e-3)),
                "max_abs_diff": float(np.abs(out - exp[name]).max()),
                "size_kb": round((MODELS / f"{name}.onnx").stat().st_size / 1024, 1),
                **time_it(lambda: sess.run(None, {"x": one})),
            }

    if not str(report["packages"]["lightgbm"]).startswith("MISSING"):
        import lightgbm as lgb
        booster = lgb.Booster(model_file=str(MODELS / "tiny_lgbm.txt"))
        out = booster.predict(exp["x_tab"])
        one = exp["x_tab"][:1]
        report["checks"]["tiny_lgbm"] = {
            "matches_laptop": bool(np.allclose(out, exp["lgbm"], atol=1e-6)),
            **time_it(lambda: booster.predict(one, num_threads=1)),
        }

    if not str(report["packages"]["sklearn"]).startswith("MISSING"):
        from sklearn.linear_model import LogisticRegression
        rng = np.random.RandomState(0)
        lr = LogisticRegression().fit(rng.randn(500, 6), rng.randint(0, 2, 500))
        one = rng.randn(1, 6)
        report["checks"]["sklearn_lr_6feat"] = time_it(lambda: lr.predict_proba(one))

    report["peak_rss_mb"] = round(peak_rss_mb(), 1)
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"board_check_{report['host']}.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    ok = all(v.get("matches_laptop", True) for v in report["checks"].values()) and \
        not any(str(v).startswith("MISSING") for v in report["packages"].values())
    print("\nRESULT:", "PASS" if ok else "FAIL (see above)", f"-> {path.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()
