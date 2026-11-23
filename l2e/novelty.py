"""Text novelty: how far is this call (and the tool result just before it) from anything seen in training?

The deferrer's v1 novelty signal was only "is this tool in the training vocabulary". This adds two numbers per
decision point, both from the monitor's own hashed text vectors (no extra model):

  nov_call   1 - cosine similarity between the call's text and its nearest neighbour in a bank of training calls
  nov_obs    the same for the tool result right before the call (the place an injection sits); NaN at step 0

The bank is a fixed random sample of BANK_SIZE training decision points (call and result vectors), so the cost
is one small matrix product per decision and it fits on the board (4096 x 512 float16 = 4 MB per bank).

Out of fold, like the error labels: a training row is compared only with the training data of the fold model
that scored it (the other folds), never with itself or its own fold. Rows scored by the full model (val, test,
context) are compared with a bank drawn from all training rows. Otherwise every training row would look
familiar and the deferrer could not learn what novelty means.

Run:  python -m l2e.novelty                                  (main monitor, task folds, every split)
      python -m l2e.novelty --fold-by domain --splits lodo_web
Out:  data/processed/novelty/<tag>/<split>.csv   dp_id, set, fold, nov_call, nov_obs
      models/deferrer/novelty_bank_<tag>_taskdisjoint.npz  (the full-model bank, for the app and the board)

The top of this file needs only numpy and pandas, so the app and the board can import the distance function.
"""
import argparse
import time

import numpy as np

BANK_SIZE = 4096
NOVELTY_FEATURES = ["nov_call", "nov_obs"]


def nearest_distance(q, bank, chunk=8192):
    """1 - max cosine similarity of each row of q to the bank. Hashed text vectors are unit length (or all zero:
    empty text, which gets distance 1). Rows of q that are NaN (no previous result) give NaN."""
    q = np.asarray(q, np.float32)
    bank = np.asarray(bank, np.float32)
    out = np.full(len(q), np.nan)
    ok = ~np.isnan(q).any(axis=1)
    idx = np.flatnonzero(ok)
    for i in range(0, len(idx), chunk):
        j = idx[i:i + chunk]
        out[j] = 1.0 - np.clip((q[j] @ bank.T).max(axis=1), -1.0, 1.0)
    return out


def sample_bank(n, size, seed):
    """Row positions of a bank: all rows if there are fewer than `size`, else a fixed random sample."""
    rng = np.random.default_rng(seed)
    return np.arange(n) if n <= size else np.sort(rng.choice(n, size, replace=False))


def bank_vectors(call_vecs, obs_vecs, rows):
    """Bank from the chosen training rows. Results that don't exist (step 0) are left out of the result bank."""
    obs = obs_vecs[rows]
    return call_vecs[rows], obs[~np.isnan(obs).any(axis=1)]


def novelty_for(df, call_vecs, obs_vecs, size=BANK_SIZE, seed=0):
    """df: one split's out-of-fold table (dp_id, set, fold), aligned with call_vecs / obs_vecs row by row.

    -> nov_call, nov_obs arrays. Fold k rows use a bank from the oof rows of the other folds; full-model rows
    (fold -1) use a bank from all oof rows. Returns also the full-model bank for saving.
    """
    nov_call = np.full(len(df), np.nan)
    nov_obs = np.full(len(df), np.nan)
    is_oof = (df["set"] == "oof").values
    fold = df["fold"].values
    full_bank = None
    for k in sorted(set(fold)):
        rows = np.flatnonzero(fold == k)
        train_rows = np.flatnonzero(is_oof & (fold != k)) if k >= 0 else np.flatnonzero(is_oof)
        pick = train_rows[sample_bank(len(train_rows), size, seed + 1000 * (k + 1))]
        bc, bo = bank_vectors(call_vecs, obs_vecs, pick)
        nov_call[rows] = nearest_distance(call_vecs[rows], bc)
        nov_obs[rows] = nearest_distance(obs_vecs[rows], bo)
        if k < 0:
            full_bank = (bc, bo)
    return nov_call, nov_obs, full_bank


# --------------------------------------------------------------------------- laptop: build the features

def text_vectors(data, dp_ids, n_struct):
    """Hashed text of each decision point's call and of the result just before it (NaN at step 0)."""
    dim = data.dim - n_struct
    call = np.zeros((len(dp_ids), dim), np.float32)
    obs = np.full((len(dp_ids), dim), np.nan, np.float32)
    for i, d in enumerate(dp_ids):
        tid, t = d.rsplit(":", 1)
        _, X, calls = data.ev[tid]
        p = calls[int(t)]
        call[i] = X[p, n_struct:]
        if p > 0:
            obs[i] = X[p - 1, n_struct:]
    return call, obs


def main():
    from . import config
    from .folds import read_oof
    from .monitor import EventData
    from .step_features import Featurizer, default_cfg

    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default=config.MAIN_ARCH)
    ap.add_argument("--encoder", default=config.MAIN_FEATURES)
    ap.add_argument("--fold-by", choices=config.FOLD_BY, default="task")
    ap.add_argument("--seed", type=int, default=config.SEED, help="monitor seed (picks the out-of-fold run)")
    ap.add_argument("--splits", nargs="+", default=None)
    args = ap.parse_args()
    if not args.encoder.startswith("hash"):
        raise SystemExit("novelty uses the hashed text vectors; run it with a hash feature set")
    tag = config.oof_tag(args.arch, args.encoder, args.seed, args.fold_by)
    n_struct = Featurizer(default_cfg(args.encoder)).n_struct
    data = EventData(args.encoder)
    splits = args.splits or ["taskdisjoint", *[f"tool_seed{s}" for s in config.TOOL_HOLDOUT_SEEDS],
                             *[f"lodo_{d}" for d in config.DOMAINS]]
    out_dir = config.PROCESSED / "novelty" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        t0 = time.time()
        df = read_oof(tag, split)[["dp_id", "set", "fold"]].reset_index(drop=True)
        call, obs = text_vectors(data, df["dp_id"].tolist(), n_struct)
        nc, no, bank = novelty_for(df, call, obs, seed=config.SEED)
        df["nov_call"], df["nov_obs"] = nc, no
        df.to_csv(out_dir / f"{split}.csv", index=False)
        if split == "taskdisjoint" and bank is not None:
            (config.MODELS / "deferrer").mkdir(parents=True, exist_ok=True)
            np.savez_compressed(config.MODELS / "deferrer" / f"novelty_bank_{tag}_{split}.npz",
                                call=bank[0].astype(np.float16), obs=bank[1].astype(np.float16))
        g = df.groupby("set")["nov_call"].mean().round(3).to_dict()
        print(f"novelty {tag}/{split}: mean nov_call by set {g} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
