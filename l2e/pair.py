"""The best escalation policy from week 3, packaged for the app (and later the board).

Two different monitors score every call: the main transformer (hashed text + tool roles) and the week 2 LSTM
(hashed text). Their calibrated logits are averaged and divided by one more temperature, fit on validation.
The pair's verdict is P(hijacked) >= 0.5. A call goes to the human when the pair is among the least sure
calls: |P - 0.5| at or below a threshold that sends 5% of validation calls to review.

On unseen domains this leaves 2.7 missed attacks per 1,000 calls at a 5% review budget, against 3.5 for the
main monitor alone, and is better in every resample (reports/week3/03_deferral_eval.md).

Everything comes from the task-disjoint split's full models and their saved out-of-fold tables, the same
ones the evaluation used, so the app's scores match the report:
  models/oof/transformer_hash-roles/taskdisjoint_full.pt, models/oof/lstm_hash/taskdisjoint_full.pt
  data/processed/oof/<same tags>/taskdisjoint.csv   (val rows: the pair's temperature and review threshold)

The novelty bank (models/deferrer/novelty_bank_*.npz) adds plain-language reasons when it exists.
"""
import json

import numpy as np
import pandas as pd

from . import config
from .calibrate import fit_temperature, sigmoid

MAIN_TAG = f"{config.MAIN_ARCH}_{config.MAIN_FEATURES}"
PARTNER_TAG = "lstm_hash"
BUDGET = 0.05
NAMES = {MAIN_TAG: "transformer", PARTNER_TAG: "LSTM"}


def fit_pair(main_df, partner_df, budget=BUDGET):
    """From two out-of-fold tables (val rows used): -> pair temperature and the margin threshold for review."""
    def zval(df):
        v = df[df["set"] == "val"].drop_duplicates("dp_id")
        return pd.Series((v["logit"] / v["temperature"]).values, index=v["dp_id"].values), v.set_index("dp_id")["y"]
    z1, y = zval(main_df)
    z2, _ = zval(partner_df)
    z2 = z2.reindex(z1.index)
    if z2.isna().any():
        raise ValueError("the two monitors were not scored on the same validation calls")
    zavg = 0.5 * (z1.values + z2.values)
    T = fit_temperature(zavg, y.reindex(z1.index).values)
    margin = np.abs(sigmoid(zavg / T) - 0.5)
    return {"temperature": float(T), "margin_threshold": float(np.quantile(margin, budget)), "budget": budget,
            "val_calls": int(len(margin))}


def pair_scores(z_main, z_partner, T):
    """Calibrated logits of the two monitors -> the pair's P(hijacked)."""
    return sigmoid(0.5 * (np.asarray(z_main) + np.asarray(z_partner)) / T)


def reasons(p_pair, p_main, p_partner, margin_threshold, tool_seen, nov_call=None, nov_obs=None):
    """Plain-language reasons for one call's decision (largest first, at most three)."""
    out = []
    gap = abs(p_main - p_partner)
    if abs(p_pair - 0.5) <= margin_threshold:
        out.append(f"the two monitors together are unsure (P = {p_pair:.2f}, close to the 0.5 line)")
    if gap >= 0.3:
        out.append(f"the monitors disagree: transformer {p_main:.2f}, LSTM {p_partner:.2f}")
    if not tool_seen:
        out.append("this tool was never seen in training")
    if nov_obs is not None and not np.isnan(nov_obs) and nov_obs >= 0.6:
        out.append(f"the tool result before this call is unlike training data (novelty {nov_obs:.2f})")
    if nov_call is not None and not np.isnan(nov_call) and nov_call >= 0.6:
        out.append(f"this call's text is unlike the training calls (novelty {nov_call:.2f})")
    return out[:3]


class PairMonitor:
    """Score a whole trace with both monitors in one causal pass each (a score at call t uses only the past)."""

    def __init__(self, split="taskdisjoint"):
        from .monitor import load, make_featurizer
        self.members = []
        for tag in (MAIN_TAG, PARTNER_TAG):
            model, ck = load(config.MODELS / "oof" / tag / f"{split}_full.pt")
            self.members.append((tag, model, ck, make_featurizer(ck["encoder"], ck["featurizer"])))
        tables = [pd.read_csv(config.PROCESSED / "oof" / tag / f"{split}.csv") for tag in (MAIN_TAG, PARTNER_TAG)]
        self.meta = fit_pair(*tables)
        self.vocab = self.members[0][2]["vocab"]
        self.bank, self.text_featurizer = None, None
        bank = config.MODELS / "deferrer" / f"novelty_bank_{MAIN_TAG}_{split}.npz"
        if bank.exists():
            z = np.load(bank)
            self.bank = (z["call"].astype(np.float32), z["obs"].astype(np.float32))
            self.text_featurizer = self.members[0][3]
        self._check(tables[0], split)

    def _check(self, main_df, split):
        """The app's main-monitor logits must match the saved table the report used (one validation trace)."""
        v = main_df[main_df["set"] == "val"]
        tid = v["dp_id"].iloc[0].rsplit(":", 1)[0]
        from .data import load_anon
        try:
            tr = next((r for r in load_anon("taskdisjoint", "val") if r["id"] == tid), None)
        except (FileNotFoundError, OSError):
            return
        if tr is None:
            return
        z = self.calibrated_logits(tr["steps"])[0]
        saved = v[v["dp_id"].str.startswith(tid + ":")].set_index("dp_id")
        want = [saved.loc[f"{tid}:{t}", "logit"] / saved.loc[f"{tid}:{t}", "temperature"]
                for t in range(len(z)) if f"{tid}:{t}" in saved.index]
        diff = float(np.max(np.abs(np.array(want) - z[:len(want)]))) if want else 0.0
        self.meta["check_max_diff"] = diff
        if diff > 1e-3:
            print(f"WARNING: app scores differ from the saved {split} table by {diff:.2e}", flush=True)

    def calibrated_logits(self, steps):
        import torch
        from .step_features import tool_ids
        out = []
        for _, model, ck, feat in self.members:
            tools, X, calls = feat.events(steps)
            with torch.no_grad():
                lg = model(torch.from_numpy(tool_ids(tools, ck["vocab"]))[None], torch.from_numpy(X)[None])[0]
            out.append(lg[calls].numpy() / ck["temperature"])
        return out

    def novelty(self, steps):
        if self.bank is None:
            return None, None
        from .novelty import nearest_distance
        _, X, calls = self.text_featurizer.events(steps)
        n = self.text_featurizer.n_struct
        call = X[calls, n:]
        obs = np.full_like(call, np.nan)
        for t, p in enumerate(calls):
            if p > 0:
                obs[t] = X[p - 1, n:]
        return nearest_distance(call, self.bank[0]), nearest_distance(obs, self.bank[1])

    def score_trace(self, steps):
        """-> list of dicts, one per call: p_main, p_partner, p_pair, verdict, defer, reasons."""
        z1, z2 = self.calibrated_logits(steps)
        pp = pair_scores(z1, z2, self.meta["temperature"])
        nc, no = self.novelty(steps)
        rows = []
        for t, s in enumerate(steps):
            p1, p2, p = float(sigmoid(z1[t])), float(sigmoid(z2[t])), float(pp[t])
            seen = s["tool"] in self.vocab
            rows.append({"p_main": p1, "p_partner": p2, "p_pair": p, "verdict": "HIJACKED" if p >= 0.5 else "SAFE",
                         "defer": abs(p - 0.5) <= self.meta["margin_threshold"], "tool_seen": seen,
                         "reasons": reasons(p, p1, p2, self.meta["margin_threshold"], seen,
                                            None if nc is None else nc[t], None if no is None else no[t])})
        return rows

    @staticmethod
    def available(split="taskdisjoint"):
        return all((config.MODELS / "oof" / t / f"{split}_full.pt").exists() and
                   (config.PROCESSED / "oof" / t / f"{split}.csv").exists() for t in (MAIN_TAG, PARTNER_TAG))

    def describe(self):
        return json.dumps(self.meta)
