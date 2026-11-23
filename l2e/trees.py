"""Run a trained LightGBM model with numpy only, from its JSON dump (booster.dump_model()).

Why: on macOS, LightGBM and PyTorch each load their own OpenMP library, and using both in one process can
crash (segfault). The app needs the PyTorch monitor and the deferrer together, so the deferrer runs from its
JSON dump instead. This also lets the board run it with nothing but numpy.

l2e.deferrer checks on validation data that this gives the same predictions as LightGBM itself.
Only what the deferrer uses is supported: numerical splits ("<=") and the binary objective.
"""
import json

import numpy as np


class TreeModel:
    def __init__(self, dump):
        if isinstance(dump, (str, bytes)) or hasattr(dump, "read_text"):
            dump = json.loads(dump.read_text() if hasattr(dump, "read_text") else dump)
        self.features = dump["feature_names"]
        self.trees = [t["tree_structure"] for t in dump["tree_info"]]
        obj = dump.get("objective", "binary sigmoid:1")
        if not obj.startswith("binary"):
            raise ValueError(f"only the binary objective is supported, got {obj!r}")
        self.sigmoid = float(obj.split("sigmoid:")[1].split()[0]) if "sigmoid:" in obj else 1.0

    @classmethod
    def load(cls, path):
        with open(path) as f:
            return cls(json.load(f))

    def raw(self, X):
        X = np.asarray(X, dtype=float)
        out = np.zeros(len(X))
        idx = np.arange(len(X))
        for t in self.trees:
            self._add(t, X, idx, out)
        return out

    def predict(self, X):
        """X: 2-D array with columns in self.features order (or a DataFrame). -> P(class 1)."""
        if hasattr(X, "columns"):
            X = X[self.features].to_numpy(dtype=float)
        return 1.0 / (1.0 + np.exp(-self.sigmoid * self.raw(X)))

    def contributions(self, X):
        """Per-feature contributions to the raw score (Saabas path attribution): for each tree, every split on
        the path moves the prediction from the node's value to the child's, and that change is credited to the
        split's feature. -> (bias, contrib) with bias + contrib.sum(axis=1) == raw(X). Used by the app to say
        why a call was deferred; the trees must have been dumped with internal values (LightGBM does this)."""
        X = np.asarray(X, dtype=float)
        contrib = np.zeros(X.shape)
        bias = 0.0
        idx = np.arange(len(X))
        for t in self.trees:
            bias += self._value(t)
            self._contrib(t, X, idx, contrib)
        return bias, contrib

    @staticmethod
    def _value(node):
        return node["leaf_value"] if "leaf_value" in node else node["internal_value"]

    def _go_left(self, node, X, idx):
        v = X[idx, node["split_feature"]]
        nan = np.isnan(v)
        dl = bool(node["default_left"])
        if node.get("missing_type", "None") == "NaN":
            return np.where(nan, dl, v <= node["threshold"])
        v = np.where(nan, 0.0, v)
        left = v <= node["threshold"]
        if node.get("missing_type") == "Zero":
            left = np.where(np.abs(v) <= 1e-35, dl, left)
        return left

    def _contrib(self, node, X, idx, contrib):
        if "leaf_value" in node or len(idx) == 0:
            return
        left = self._go_left(node, X, idx)
        f, here = node["split_feature"], self._value(node)
        for child, rows in ((node["left_child"], idx[left]), (node["right_child"], idx[~left])):
            contrib[rows, f] += self._value(child) - here
            self._contrib(child, X, rows, contrib)

    def _add(self, node, X, idx, out):
        if "leaf_value" in node:
            out[idx] += node["leaf_value"]
            return
        if len(idx) == 0:
            return
        if node.get("decision_type", "<=") != "<=":
            raise ValueError("categorical splits are not supported")
        v = X[idx, node["split_feature"]]
        nan = np.isnan(v)
        missing = node.get("missing_type", "None")
        dl = bool(node["default_left"])
        if missing == "NaN":
            left = np.where(nan, dl, v <= node["threshold"])
        else:  # "None": NaN counts as 0;  "Zero": NaN and 0 both take the default branch
            v = np.where(nan, 0.0, v)
            left = v <= node["threshold"]
            if missing == "Zero":
                left = np.where(np.abs(v) <= 1e-35, dl, left)
        self._add(node["left_child"], X, idx[left], out)
        self._add(node["right_child"], X, idx[~left], out)
