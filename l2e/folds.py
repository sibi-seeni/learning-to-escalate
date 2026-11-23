"""Split definitions and out-of-fold plans. Plain pandas/scikit-learn, no PyTorch or LightGBM.

Kept apart from l2e/oof.py so the deferrer (LightGBM), the novelty features and the evaluation can use the
same fold and split logic without importing PyTorch (on macOS the two can crash in one process).

Two ways to make out-of-fold labels:
  task     week 2: K folds grouped by task text. The held-out fold has new tasks from the same domains.
  domain   week 3: one fold per training domain. Each fold model trains on the other domains and scores a
           domain it never saw, so its errors look like the ones the deployed monitor makes under domain
           shift. Early stopping and the temperature use only the fold model's own domains, as they would
           at deployment (the val traces of the held-out domain are not used for that fold).
"""
import json

import pandas as pd
from sklearn.model_selection import GroupKFold

from . import config


def split_defs(dps):
    """name -> {train, val, test: {set name: dp ids}}"""
    by_part = {p: list(dps.loc[dps["part"] == p, "dp_id"]) for p in config.PARTS}
    out = {"taskdisjoint": {"train": by_part["train"], "val": by_part["val"], "test": {"test": by_part["test"]}}}
    for s in config.TOOL_HOLDOUT_SEEDS:
        d = json.loads((config.SPLITS_DIR / f"tool_coverage_seed{s}.json").read_text())
        out[f"tool_seed{s}"] = {"train": d["train"], "val": d["val"],
                                "test": {f"cov{k}": v for k, v in d["test"].items()}}
    for dom in config.DOMAINS:
        d = json.loads((config.SPLITS_DIR / f"lodo_{dom}.json").read_text())
        out[f"lodo_{dom}"] = {"train": d["train"], "val": d["val"], "test": {"test": d["test"]}}
    return out


def fold_plan(dps, train_ids, val_ids, fold_by="task", folds=config.OOF_FOLDS, task_of=None):
    """-> list of (fold name, fold training ids, held-out ids, val ids for that fold model).

    dps: decision points indexed by dp_id, with traj_id and domain. task_of: traj_id -> task text (task mode).
    """
    train_ids, val_ids = list(train_ids), list(val_ids)
    if fold_by == "task":
        groups = dps.loc[train_ids, "traj_id"].map(task_of).values
        out = []
        for k, (tr, ho) in enumerate(GroupKFold(n_splits=folds).split(train_ids, groups=groups)):
            out.append((k, [train_ids[i] for i in tr], [train_ids[i] for i in ho], val_ids))
        return out
    if fold_by != "domain":
        raise ValueError(f"fold_by must be one of {config.FOLD_BY}, got {fold_by!r}")
    dom = dps.loc[train_ids, "domain"].values
    vdom = dps.loc[val_ids, "domain"].values
    domains = sorted(set(dom))
    if len(domains) < 2:
        raise ValueError("domain folds need at least two training domains")
    out = []
    for k, d in enumerate(domains):
        out.append((k, [i for i, x in zip(train_ids, dom) if x != d], [i for i, x in zip(train_ids, dom) if x == d],
                    [i for i, x in zip(val_ids, vdom) if x != d]))
    return out


def context_ids(dps, test_sets):
    """Every decision point of a trace that appears in a test set, minus those already in a test set.

    Some test sets hold only part of a trace (the tool-coverage sets sample single calls). The deferrer's
    "earlier in this trace" features need the monitor's scores on the earlier calls too, which the monitor
    would have produced live. These are scored by the same full model and stored as set "context".
    """
    tested = set().union(*map(set, test_sets.values())) if test_sets else set()
    trajs = set(dps.loc[list(tested), "traj_id"])
    all_ids = dps.index[dps["traj_id"].isin(trajs)]
    return [i for i in all_ids if i not in tested]


def oof_path(tag, split):
    return config.PROCESSED / "oof" / tag / f"{split}.csv"


def read_oof(tag, split):
    path = oof_path(tag, split)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run l2e.oof for this monitor first (scripts/run_week3.sh).")
    return pd.read_csv(path)
