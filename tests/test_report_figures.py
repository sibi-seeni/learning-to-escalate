"""Deliverable 3 report figures (l2e/report_figures.py) on tiny fake data."""
import json

import numpy as np
import pandas as pd
import pytest

from l2e import config
from l2e import report_figures as rf


def _oof(rng, n_val, n_test, noise, prefix):
    rows = []
    for set_name, n in (("val", n_val), ("test", n_test)):
        y = (rng.random(n) < 0.2).astype(int)
        z = 5 * y - 2.5 + rng.normal(0, noise, n)
        T = 1.2
        rows.append(pd.DataFrame({"dp_id": [f"{prefix}{set_name}_{i // 4}:{i % 4}" for i in range(n)], "set": set_name,
                                  "logit": z * T, "temperature": T, "y": y}))
    return pd.concat(rows, ignore_index=True)


def test_confusion_and_roc():
    y = np.array([0, 0, 1, 1, 1])
    pred = np.array([0, 1, 0, 1, 1])
    m = rf.confusion(y, pred, review=np.array([False, False, True, False, False]))
    assert m.tolist() == [[1, 1, 0], [0, 2, 1]]
    fpr, tpr, auc = rf.roc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert auc == pytest.approx(1.0) and fpr[0] == 0 and tpr[-1] == 1


def test_pair_table_reviews_the_budget():
    rng = np.random.default_rng(0)
    main = _oof(rng, 400, 400, 2.0, "")
    partner = main.copy()
    partner["logit"] = partner["logit"] + rng.normal(0, 1.0, len(partner))
    t = rf.pair_table(main, partner)
    assert len(t) == 400 and t["review"].sum() == 20 and t["review_main"].sum() == 20
    assert ((t["p_pair"] >= 0) & (t["p_pair"] <= 1)).all()
    unsure = np.abs(t["p_pair"] - 0.5)
    assert unsure[t["review"]].max() <= unsure[~t["review"]].min()


def test_main_end_to_end(tmp_path, monkeypatch):
    rng = np.random.default_rng(1)
    proc, figs, rep = tmp_path / "processed", tmp_path / "figures", tmp_path / "reports"
    for p in (proc, figs, rep):
        p.mkdir()
    monkeypatch.setattr(config, "PROCESSED", proc)
    monkeypatch.setattr(config, "FIGURES", figs)
    monkeypatch.setattr(config, "REPORTS3", rep)
    monkeypatch.setattr(config, "ensure_dirs", lambda: None)
    dp_rows = []
    for split in ["taskdisjoint"] + [f"lodo_{d}" for d in config.DOMAINS]:
        main = _oof(rng, 200, 200, 2.0, f"{split}_")
        partner = main.copy()
        partner["logit"] = partner["logit"] + rng.normal(0, 1.5, len(partner))
        for tag, df in ((rf.MAIN, main), (rf.PARTNER, partner)):
            (proc / "oof" / tag).mkdir(parents=True, exist_ok=True)
            df.to_csv(proc / "oof" / tag / f"{split}.csv", index=False)
        dp_rows.append(pd.DataFrame({"dp_id": main["dp_id"], "traj_id": main["dp_id"].str.split(":").str[0],
                                     "step": main["dp_id"].str.split(":").str[1].astype(int), "tool": "send_email"}))
    pd.concat(dp_rows).drop_duplicates("dp_id").to_csv(proc / "decision_points.csv", index=False)

    def cell(m):
        return {"budget": 0.05, "dangerous_left_per_1k": {"mean": m, "lo": m - 0.5, "hi": m + 0.5}}
    budgets = [0.0, 0.05]
    res = {s: {"natural": {"counts": {"calls": 1000.0, "dangerous": 3.0, "errors": 10.0},
                           "table": {"confidence": [cell(3.0), cell(1.5)], "stronger": [cell(2.5), cell(1.0)]}}}
           for s in ["Task-disjoint test", rf.UNSEEN]}
    (rep / "03_deferral_eval.json").write_text(json.dumps({"budgets": budgets, "results": res}))
    rf.main()
    for f in ("11_d2_vs_d3.png", "12_confusion.png", "13_roc.png"):
        assert (figs / f).stat().st_size > 1000
    nums = json.loads((rep / "04_report_numbers.json").read_text())
    assert nums["d2_vs_d3"][rf.UNSEEN]["d3"]["mean"] == 1.0
    assert "Missed attack" in (rep / "04_examples.md").read_text()
