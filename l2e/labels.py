"""Monitor error labels: the deferrer's training target.

  dangerous error  monitor says safe (0), call is hijacked (1)   -> a missed attack
  costly error     monitor says hijacked (1), call is benign (0) -> a false alarm someone must review

The deferrer learns "is the monitor wrong here?", with dangerous errors weighted more. In week 2 these
labels must come from out-of-fold predictions (the monitor never trained on that row).
"""
import numpy as np
import pandas as pd

from . import config


def error_labels(y_true, y_pred, dangerous_weight=config.DANGEROUS_WEIGHT):
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    dangerous = (y_pred == 0) & (y_true == 1)
    costly = (y_pred == 1) & (y_true == 0)
    return pd.DataFrame({
        "monitor_error": (dangerous | costly).astype(int),
        "dangerous": dangerous.astype(int),
        "costly": costly.astype(int),
        # sample weight for training the deferrer
        "weight": np.where(dangerous, dangerous_weight, 1.0),
    })


def error_counts(y_true, y_pred):
    e = error_labels(y_true, y_pred)
    return {"decision_points": len(e), "monitor_errors": int(e["monitor_error"].sum()),
            "dangerous": int(e["dangerous"].sum()), "costly": int(e["costly"].sum())}
