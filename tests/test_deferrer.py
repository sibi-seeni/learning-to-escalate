import numpy as np
import pandas as pd
import pytest

# No LightGBM here on purpose: on macOS, LightGBM and PyTorch in one process (the torch tests) can segfault.
from l2e.deferrer import FEATURES, add_features, caught_curve
from l2e.trees import TreeModel


def test_caught_curve_counts_top_k():
    err = [1, 0, 1, 0, 0, 0, 0, 0, 0, 1]
    dang = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    prio = [9, 8, 7, 1, 1, 1, 1, 1, 1, 0]  # ranks errors 0 and 2 first, error 9 last
    c = {r["budget"]: r for r in caught_curve(prio, err, dang, budgets=[0.0, 0.2, 0.3, 1.0])}
    assert c[0.0]["errors_caught"] == 0
    assert (c[0.2]["errors_caught"], c[0.2]["dangerous_caught"], c[0.2]["wasted_reviews"]) == (1, 1, 1)
    assert c[0.3]["errors_caught"] == 2
    assert c[1.0]["errors_caught"] == 3


def test_features_only_use_earlier_calls():
    d = pd.DataFrame({"set": "test", "fold": -1, "traj_id": ["a"] * 3 + ["b"] * 2, "step": [0, 1, 2, 0, 1],
                      "tool": ["read_email", "forward_email", "mystery_upload", "get_balance", "transfer_money"],
                      "state_changing": [False, True, False, False, True],
                      "score": [0.1, 0.9, 0.2, 0.3, 0.6], "tool_seen": [1, 1, 0, 1, 1]})
    f = add_features(d)
    assert set(FEATURES) <= set(f.columns)
    a = f[f["traj_id"] == "a"]
    assert np.isnan(a["prev_score"].iloc[0]) and a["prev_score"].iloc[2] == 0.9
    assert list(a["prev_max_score"].iloc[1:]) == [0.1, 0.9]
    assert list(a["n_prev_alarms"]) == [0, 0, 1]
    assert f[f["traj_id"] == "b"]["prev_score"].iloc[1] == 0.3  # no leak across traces
    assert a["role_sends_out"].iloc[2] == 1  # unknown tool: roles from its name
    assert a["margin"].iloc[1] == pytest.approx(0.4)


def test_numpy_trees_route_values_and_missing():
    # one tree: x0 <= 0.5 ? (x1 <= 2 ? 1.0 : 2.0) : 3.0 ; NaN in x0 goes right, NaN in x1 counts as 0
    tree = {"split_feature": 0, "threshold": 0.5, "decision_type": "<=", "default_left": False, "missing_type": "NaN",
            "left_child": {"split_feature": 1, "threshold": 2.0, "decision_type": "<=", "default_left": True,
                           "missing_type": "None", "left_child": {"leaf_value": 1.0},
                           "right_child": {"leaf_value": 2.0}},
            "right_child": {"leaf_value": 3.0}}
    m = TreeModel({"feature_names": ["a", "b"], "objective": "binary sigmoid:1",
                   "tree_info": [{"tree_structure": tree}, {"tree_structure": {"leaf_value": -1.0}}]})
    X = np.array([[0.1, 1.0], [0.1, 5.0], [0.9, 0.0], [np.nan, 0.0], [0.1, np.nan]])
    assert list(m.raw(X)) == [0.0, 1.0, 2.0, 2.0, 0.0]
    assert m.predict(X)[0] == pytest.approx(0.5)


PARITY = """
import numpy as np, lightgbm as lgb
from l2e.trees import TreeModel
rng = np.random.default_rng(0)
X = rng.normal(size=(3000, 6)); X[rng.random(X.shape) < 0.15] = np.nan; X[:, 5] = (X[:, 5] > 0).astype(float)
y = ((np.nan_to_num(X[:, 0]) + np.nan_to_num(X[:, 1]) * X[:, 5]) > 0.3).astype(int)
m = lgb.train(dict(objective="binary", num_leaves=15, verbose=-1, seed=1), lgb.Dataset(X, y), 60)
d = float(np.abs(TreeModel(m.dump_model()).predict(X) - m.predict(X)).max())
assert d < 1e-9, d
print("ok", d)
"""


def test_numpy_trees_match_lightgbm_in_a_separate_process():
    import subprocess
    import sys
    import importlib.util
    if importlib.util.find_spec("lightgbm") is None:  # find_spec doesn't load LightGBM into this process
        pytest.skip("lightgbm not installed")
    r = subprocess.run([sys.executable, "-c", PARITY], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
