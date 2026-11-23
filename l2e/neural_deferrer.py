"""Neural deferrer: a small causal transformer that reads the trace itself, not just the monitor's confidence.

The LightGBM deferrer (l2e/deferrer.py) sees numbers about each call: the monitor's score, the tool's roles,
novelty. This one reads the same event stream the monitor reads (hashed text, tool id, structured features)
and, on every call event, six extra numbers:

  d_logit      the monitor's calibrated logit at that call (clipped to +-10, divided by 5)
  d_margin     2 x |score - 0.5|
  d_scored     1 on call events that have a monitor score
  d_tool_seen  is the tool in the monitor's training vocabulary
  d_nov_call   text novelty of the call (l2e/novelty.py)
  d_nov_obs    text novelty of the result before it (0 at step 0)

So at call t it knows what the agent did and saw up to t, and how the monitor scored every call up to t.
It outputs P(the monitor is wrong at this call). Same architecture as the main monitor (2-layer causal
transformer, width 64, l2e/monitor.py), so it would fit on the board the same way.

Training data: the monitor's out-of-fold errors, missed attacks weighted 5x, as for LightGBM. One out-of-fold
fold is kept aside for early stopping (with domain folds that is a whole training domain, so it stops when
it is best at a domain it never trained on). Risk: about 400 errors among 50k calls is little for a sequence
model, and it could drift into a second attack detector. That is what the comparison in
l2e/evaluate_deferral.py checks.

Run:  python -m l2e.neural_deferrer                              (all splits, task + domain folds if present)
      python -m l2e.neural_deferrer --splits lodo_web --fold-by domain --seeds 1
Out:  data/processed/defer_preds/<tag>/<split>_neural.csv   dp_id, set, seed, p
      models/deferrer/neural_<tag>_taskdisjoint.pt            (seed 42, for the board later)
      reports/week3/02_neural_deferrer_training.json

Runs in its own process: it needs PyTorch, and LightGBM must not be loaded next to it (macOS crash).
"""
import argparse
import copy
import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd

from . import config
from .deferrer import ALL_SPLITS, MONITOR_TAG, available_runs, load_table
from .step_features import UNK, build_vocab, tool_ids

EXTRA = ["d_logit", "d_margin", "d_scored", "d_tool_seen", "d_nov_call", "d_nov_obs"]
TRAIN_HP = dict(batch=64, lr=2e-3, weight_decay=1e-4, max_epochs=25, patience=4, tool_dropout=0.1, dropout=0.1)
N_SEEDS = 3


def extra_values(row):
    """The six per-call numbers from one row of the deferrer table (l2e.deferrer.load_table)."""
    nov_obs = row["nov_obs"]
    return [float(np.clip(row["logit_cal"], -10, 10) / 5.0), float(2 * row["margin"]), 1.0, float(row["tool_seen"]),
            float(np.nan_to_num(row["nov_call"], nan=1.0)), float(0.0 if np.isnan(nov_obs) else nov_obs)]


def extras_by_model(d):
    """(fold, traj_id) -> {step: extra vector}, one entry per (scoring model, call)."""
    u = d.drop_duplicates(["fold", "dp_id"])
    out = defaultdict(dict)
    for row in u[["fold", "traj_id", "step", "logit_cal", "margin", "tool_seen", "nov_call", "nov_obs"]].itertuples(
            index=False):
        out[(row.fold, row.traj_id)][int(row.step)] = extra_values(row._asdict())
    return out


def augment(X, calls, extras):
    """Event matrix + the EXTRA columns, filled on the call events that have a monitor score."""
    A = np.zeros((len(X), len(EXTRA)), np.float32)
    for step, vec in extras.items():
        if step < len(calls) and calls[step] < len(X):
            A[calls[step]] = vec
    return np.concatenate([X, A], axis=1)


def make_items(data, rows, extras, vocab, truncate=True):
    """rows: deferrer-table rows scored by ONE model (one fold, or the full model).
    -> [(tool ids, X+extras, call positions, error labels, weights, dp_ids)] one item per trace."""
    out = []
    for (fold, tid), g in rows.groupby(["fold", "traj_id"], sort=False):
        tools, X, calls = data.ev[tid]
        g = g.drop_duplicates("dp_id").sort_values("step")
        pos = [calls[int(t)] for t in g["step"]]
        L = max(pos) + 1 if truncate else len(tools)
        Xa = augment(X[:L], calls, extras.get((fold, tid), {}))
        out.append((tool_ids(tools[:L], vocab), Xa, pos, g["monitor_error"].tolist(), g["weight"].tolist(),
                    g["dp_id"].tolist()))
    return out


def collate(batch, tool_dropout=0.0, rng=None):
    import torch
    L = max(len(b[0]) for b in batch)
    F = batch[0][1].shape[1]
    tool = np.zeros((len(batch), L), np.int64)
    X = np.zeros((len(batch), L, F), np.float32)
    mask = np.zeros((len(batch), L), bool)
    y = np.zeros((len(batch), L), np.float32)
    w = np.zeros((len(batch), L), np.float32)
    for i, (tid, x, pos, lab, wt, _) in enumerate(batch):
        n = len(tid)
        tool[i, :n], X[i, :n] = tid, x
        mask[i, pos], y[i, pos], w[i, pos] = True, lab, wt
    if tool_dropout and rng is not None:
        drop = (rng.random(tool.shape) < tool_dropout) & (tool > 0)
        tool[drop] = UNK
    return tuple(torch.from_numpy(a) for a in (tool, X, mask, y, w))


def predict(model, items, batch=256):
    import torch
    model.eval()
    ids, out = [], []
    with torch.no_grad():
        for i in range(0, len(items), batch):
            b = items[i:i + batch]
            tool, X, *_ = collate(b)
            p = torch.sigmoid(model(tool, X))
            for j, it in enumerate(b):
                ids += it[5]
                out += p[j, it[2]].tolist()
    return ids, np.array(out)


def train(data, d, vocab, seed, hp=None, verbose=False):
    """Train on the out-of-fold rows of d, early stopping on the last fold. -> (model, history)."""
    import torch
    from sklearn.metrics import average_precision_score
    from torch import nn

    from .monitor import ARCHS, Monitor
    hp = {**TRAIN_HP, **(hp or {})}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    oof = d[d["set"] == "oof"]
    stop_fold = oof["fold"].max()
    extras = extras_by_model(oof)
    tr = make_items(data, oof[oof["fold"] != stop_fold], extras, vocab)
    es = make_items(data, oof[oof["fold"] == stop_fold], extras, vocab)
    es_y = np.concatenate([it[3] for it in es])
    es_w = np.concatenate([it[4] for it in es])
    model = Monitor(len(vocab) + 2, data.dim + len(EXTRA), arch=config.MAIN_ARCH, dropout=hp["dropout"],
                    **ARCHS[config.MAIN_ARCH])
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    lossf = nn.BCEWithLogitsLoss(reduction="none")
    best, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(hp["max_epochs"]):
        model.train()
        t0, tot, n = time.time(), 0.0, 0.0
        order = rng.permutation(len(tr))
        for i in range(0, len(order), hp["batch"]):
            tool, X, mask, y, w = collate([tr[j] for j in order[i:i + hp["batch"]]], hp["tool_dropout"], rng)
            loss = (lossf(model(tool, X)[mask], y[mask]) * w[mask]).sum() / w[mask].sum()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot, n = tot + loss.item(), n + 1
        _, p = predict(model, es)
        ap = average_precision_score(es_y, p, sample_weight=es_w) if es_y.any() else 0.0
        hist.append({"epoch": ep + 1, "train_loss": tot / max(n, 1), "stop_fold_auprc": ap, "sec": time.time() - t0})
        if verbose:
            print(f"    epoch {ep + 1:2d}  loss {tot / max(n, 1):.4f}  error AUPRC on fold {stop_fold} {ap:.4f}", flush=True)
        if ap > best + 1e-4:
            best, best_state, bad = ap, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, hist


def run(data, tag, split, dps, n_seeds=N_SEEDS, hp=None, save_model=False):
    import torch
    d = load_table(tag, split, dps)
    vocab = build_vocab(d.loc[d["set"] == "oof", "tool"])
    full = d[d["fold"] == -1]
    ev_items = make_items(data, full[~full["set"].isin(["context"])], extras_by_model(full), vocab, truncate=False)
    want = d[~d["set"].isin(["oof", "context"])][["dp_id", "set"]]
    preds, hists = [], []
    for k in range(n_seeds):
        seed = config.SEED + k
        t0 = time.time()
        model, hist = train(data, d, vocab, seed, hp)
        ids, p = predict(model, ev_items)
        pm = pd.DataFrame({"dp_id": ids, "p": p}).drop_duplicates("dp_id")
        preds.append(want.merge(pm, on="dp_id", how="left").assign(seed=seed))
        hists.append({"seed": seed, "epochs": len(hist), "best_stop_fold_auprc": max(h["stop_fold_auprc"] for h in hist),
                      "sec": round(time.time() - t0, 1)})
        print(f"  {tag}/{split} seed {seed}: {len(hist)} epochs, best error AUPRC on the held-aside fold "
              f"{hists[-1]['best_stop_fold_auprc']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if save_model and k == 0:
            torch.save({"state": model.state_dict(), "vocab": vocab, "extra": EXTRA, "arch": config.MAIN_ARCH,
                        "in_dim": data.dim + len(EXTRA), "n_tools": len(vocab) + 2, "monitor": tag, "split": split},
                       config.MODELS / "deferrer" / f"neural_{tag}_{split}.pt")
    out = config.PROCESSED / "defer_preds" / tag
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(preds, ignore_index=True)[["dp_id", "set", "seed", "p"]].to_csv(out / f"{split}_neural.csv", index=False)
    return hists


def main():
    from .monitor import EventData
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=None)
    ap.add_argument("--fold-by", nargs="+", choices=config.FOLD_BY, default=list(config.FOLD_BY))
    ap.add_argument("--monitor-seeds", nargs="+", type=int, default=list(config.MONITOR_SEEDS))
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--quick", action="store_true", help="2 epochs, smoke test")
    args = ap.parse_args()
    config.ensure_dirs()
    (config.MODELS / "deferrer").mkdir(parents=True, exist_ok=True)
    hp = {"max_epochs": 2} if args.quick else None
    data = EventData(config.MAIN_FEATURES)
    dps = data.dps.reset_index()
    path = config.REPORTS3 / "02_neural_deferrer_training.json"
    summary = json.loads(path.read_text()) if path.exists() else {}
    for split in args.splits or ALL_SPLITS:
        for tag, mseed, fb in available_runs(args.fold_by, args.monitor_seeds, split):
            print(f"neural deferrer on {split} / {tag} ...", flush=True)
            summary[f"{tag}/{split}"] = {"monitor": tag, "monitor_seed": mseed, "fold_by": fb, "split": split,
                                         "quick": bool(args.quick),
                                         "runs": run(data, tag, split, dps, args.seeds, hp,
                                                     save_model=split == "taskdisjoint" and tag == MONITOR_TAG)}
            path.write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
