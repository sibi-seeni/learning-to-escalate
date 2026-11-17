"""Compact neural monitors over the event stream (see step_features.py).

  lstm             1-layer LSTM, hidden 64        (int8-quantizable in ONNX Runtime)
  gru              1-layer GRU, hidden 64         (float only: ORT can't quantize GRU)
  transformer      2-layer causal transformer, width 64, 4 heads
  transformer_big  4-layer causal transformer, width 256 (laptop-only "better monitor", use with mpnet)

Every model: [tool embedding (16) ; event features] -> linear + ReLU -> sequence layer -> P(hijacked)
at each call event. Training puts the loss only on the decision points in the split's training set, and
the model only ever looks backwards (recurrent, or a causal mask), so no decision sees the future.

Tool dropout: during training 10% of tool ids are replaced by "unknown", so the model has practiced
on unfamiliar tools before the tool-shift tests.
"""
import copy
import math
import time
from collections import defaultdict

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from torch import nn

from . import config
from .calibrate import ece, sigmoid
from .data import load_anon
from .step_features import UNK, Featurizer, build_vocab, default_cfg, split_feature_set, tool_ids

ARCHS = {
    "lstm": dict(d=64, layers=1),
    "gru": dict(d=64, layers=1),
    "transformer": dict(d=64, layers=2, heads=4),
    "transformer_big": dict(d=256, layers=4, heads=8),
}
TRAIN = dict(batch=64, lr=2e-3, weight_decay=1e-4, max_epochs=25, patience=4, tool_dropout=0.1, dropout=0.1)
TOOL_DIM = 16


# --------------------------------------------------------------------------- models

class CausalBlock(nn.Module):
    """Pre-norm transformer block written out by hand, so it exports to plain ONNX ops."""

    def __init__(self, d, heads, dropout):
        super().__init__()
        self.h = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        B, L, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).chunk(3, dim=-1)
        q, k, v = (t.reshape(B, L, self.h, D // self.h).transpose(1, 2) for t in (q, k, v))
        att = (q @ k.transpose(-2, -1)) / math.sqrt(D // self.h) + mask
        out = (self.drop(att.softmax(-1)) @ v).transpose(1, 2).reshape(B, L, D)
        x = x + self.drop(self.proj(out))
        return x + self.drop(self.ff(self.ln2(x)))


class Monitor(nn.Module):
    def __init__(self, n_tools, in_dim, arch="lstm", d=64, layers=1, heads=4, dropout=0.1, max_len=48):
        super().__init__()
        self.arch = arch
        self.tool = nn.Embedding(n_tools, TOOL_DIM, padding_idx=0)
        self.inp = nn.Sequential(nn.Linear(in_dim + TOOL_DIM, d), nn.ReLU(), nn.Dropout(dropout))
        if arch == "lstm":
            self.seq = nn.LSTM(d, d, num_layers=layers, batch_first=True)
        elif arch == "gru":
            self.seq = nn.GRU(d, d, num_layers=layers, batch_first=True)
        else:
            self.pos = nn.Embedding(max_len, d)
            self.blocks = nn.ModuleList([CausalBlock(d, heads, dropout) for _ in range(layers)])
            self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)

    def forward(self, tool, x):
        """tool (B, L) int64, x (B, L, F) float -> logits (B, L)."""
        h = self.inp(torch.cat([x, self.tool(tool)], dim=-1))
        if self.arch in ("lstm", "gru"):
            h, _ = self.seq(h)
        else:
            L = h.shape[1]
            h = h + self.pos(torch.arange(L, device=h.device))
            mask = torch.triu(torch.ones(L, L, device=h.device), diagonal=1) * -1e9
            for blk in self.blocks:
                h = blk(h, mask)
            h = self.ln(h)
        return self.head(h).squeeze(-1)


class LastStep(nn.Module):
    """Export wrapper: one decision = the prefix's event stream in, calibrated P(hijacked) at its end out."""

    def __init__(self, model, temperature):
        super().__init__()
        self.model, self.T = model, float(temperature)

    def forward(self, tool, x):
        return torch.sigmoid(self.model(tool, x)[:, -1:] / self.T)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- data

class EventData:
    """Every task-disjoint trace featurized once, for one feature set (see make_featurizer). Cached on disk."""

    def __init__(self, encoder="hash", rebuild=False):
        self.encoder = encoder
        self.dps = pd.read_csv(config.PROCESSED / "decision_points.csv").set_index("dp_id")
        self.y = self.dps["y"].to_dict()
        self.trajs = {r["id"]: r for part in config.PARTS for r in load_anon("taskdisjoint", part)}
        cache = config.PROCESSED / f"events_{encoder}.joblib"
        if cache.exists() and not rebuild:
            self.ev = joblib.load(cache)
        else:
            feat = make_featurizer(encoder)
            t0 = time.time()
            self.ev = {tid: feat.events(tr["steps"]) for tid, tr in self.trajs.items()}
            joblib.dump(self.ev, cache)
            print(f"featurized {len(self.ev)} traces with '{encoder}' in {time.time() - t0:.0f}s -> {cache.name}")
        self.dim = next(iter(self.ev.values()))[1].shape[1]

    def items(self, dp_ids, vocab, truncate=True):
        """Group decision points by trace -> list of (tool ids, X, call positions, labels, dp_ids).

        truncate: cut each trace after its last requested decision point (training). Without it the
        whole trace goes in; that changes nothing for the scores because the model is causal.
        """
        by_traj = defaultdict(list)
        for d in dp_ids:
            tid, t = d.rsplit(":", 1)
            by_traj[tid].append(int(t))
        out = []
        for tid, steps in by_traj.items():
            tools, X, calls = self.ev[tid]
            steps = sorted(steps)
            pos = [calls[t] for t in steps]
            L = max(pos) + 1 if truncate else len(tools)
            ids = [f"{tid}:{t}" for t in steps]
            out.append((tool_ids(tools[:L], vocab), X[:L], pos, [self.y[i] for i in ids], ids))
        return out

    def vocab_for(self, train_ids):
        """Tools visible in any training prefix (= the tools of the training decision points).

        With the "noid" option the vocabulary is empty: every tool is "unknown" and the model has to go by
        the tool's roles and text."""
        if split_feature_set(self.encoder)[1]["noid"]:
            return {}
        return build_vocab(self.dps.loc[list(train_ids), "tool"])


def make_featurizer(feature_set, cfg=None):
    """feature_set: 'hash', 'minilm', 'mpnet', optionally with '-nothought'."""
    cfg = cfg or default_cfg(feature_set)
    enc, _ = split_feature_set(feature_set)
    if enc == "hash":
        return Featurizer(cfg)
    from .encoders import EmbeddingLookup
    look = EmbeddingLookup(enc)
    return Featurizer(cfg, text_lookup=look, text_dim=look.dim)


def collate(batch, tool_dropout=0.0, rng=None):
    L = max(len(b[0]) for b in batch)
    F = batch[0][1].shape[1]
    tool = np.zeros((len(batch), L), np.int64)
    X = np.zeros((len(batch), L, F), np.float32)
    mask = np.zeros((len(batch), L), bool)
    y = np.zeros((len(batch), L), np.float32)
    for i, (tid, x, pos, lab, _) in enumerate(batch):
        n = len(tid)
        tool[i, :n], X[i, :n] = tid, x
        mask[i, pos], y[i, pos] = True, lab
    if tool_dropout and rng is not None:
        drop = (rng.random(tool.shape) < tool_dropout) & (tool > 0)
        tool[drop] = UNK
    return torch.from_numpy(tool), torch.from_numpy(X), torch.from_numpy(mask), torch.from_numpy(y)


# --------------------------------------------------------------------------- train / predict

def predict(model, items, batch=256):
    """-> (dp_ids, raw logits)"""
    model.eval()
    ids, out = [], []
    with torch.no_grad():
        for i in range(0, len(items), batch):
            b = items[i:i + batch]
            tool, X, _, _ = collate(b)
            lg = model(tool, X)
            for j, it in enumerate(b):
                ids += it[4]
                out += lg[j, it[2]].tolist()
    return ids, np.array(out)


def metrics(y, p, thr=0.5):
    y, p = np.asarray(y), np.asarray(p)
    yhat = (p >= thr).astype(int)
    return {"auroc": roc_auc_score(y, p), "auprc": average_precision_score(y, p), "f1": f1_score(y, yhat),
            "precision": precision_score(y, yhat, zero_division=0), "recall": recall_score(y, yhat), "ece": ece(p, y)}


def train(data, train_ids, val_ids, arch="lstm", seed=config.SEED, hp=None, verbose=True):
    """Train one monitor, early stopping on validation AUPRC. Returns (model, vocab, history)."""
    hp = {**TRAIN, **(hp or {})}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    vocab = data.vocab_for(train_ids)
    tr_items = data.items(train_ids, vocab)
    va_items = data.items(val_ids, vocab)
    va_y = np.concatenate([it[3] for it in va_items])
    model = Monitor(len(vocab) + 2, data.dim, arch=arch, dropout=hp["dropout"], **ARCHS[arch])
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    lossf = nn.BCEWithLogitsLoss(reduction="sum")
    best, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(hp["max_epochs"]):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        order = rng.permutation(len(tr_items))
        for i in range(0, len(order), hp["batch"]):
            tool, X, mask, y = collate([tr_items[j] for j in order[i:i + hp["batch"]]], hp["tool_dropout"], rng)
            k = mask.sum()
            loss = lossf(model(tool, X)[mask], y[mask]) / k
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot, n = tot + loss.item() * k.item(), n + k.item()
        _, va_lg = predict(model, va_items)
        ap = average_precision_score(va_y, va_lg)
        hist.append({"epoch": ep + 1, "train_loss": tot / n, "val_auprc": ap, "sec": time.time() - t0})
        if verbose:
            print(f"  [{arch}] epoch {ep + 1:2d}  loss {tot / n:.4f}  val AUPRC {ap:.4f}  ({time.time() - t0:.1f}s)", flush=True)
        if ap > best + 1e-4:
            best, best_state, bad = ap, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= hp["patience"]:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, vocab, hist


def save(path, model, vocab, temperature, arch, encoder, featurizer_cfg, extra=None):
    torch.save({"state": model.state_dict(), "arch": arch, "encoder": encoder, "vocab": vocab,
                "temperature": float(temperature), "n_tools": len(vocab) + 2,
                "in_dim": model.inp[0].in_features - TOOL_DIM, "featurizer": featurizer_cfg, **(extra or {})}, path)


def load(path):
    ck = torch.load(path, weights_only=False)
    m = Monitor(ck["n_tools"], ck["in_dim"], arch=ck["arch"], **ARCHS[ck["arch"]])
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck


def score_trace(model, ck, steps, featurizer):
    """Calibrated P(hijacked) at every call of a trace, in one causal pass (score k only uses the past)."""
    tools, X, calls = featurizer.events(steps)
    with torch.no_grad():
        lg = model(torch.from_numpy(tool_ids(tools, ck["vocab"]))[None], torch.from_numpy(X)[None])[0]
    return sigmoid(lg[calls].numpy() / ck["temperature"])
