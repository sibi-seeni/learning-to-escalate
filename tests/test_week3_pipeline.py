"""End-to-end smoke test of the week 3 pipeline on synthetic files: neural deferrer -> evaluation -> report, figures.

No real data: a fake event stream and fake out-of-fold tables in a temporary folder. Checks that every piece
reads what the previous one writes, not the numbers.
"""
import json

import numpy as np
import pandas as pd
import pytest

from l2e import config

DOMS = ("banking", "email", "web")


def make_world(tmp, rng):
    """Decision points, an event stream, and out-of-fold tables for two splits and two fold kinds."""
    dps, ev = [], {}
    for dom in DOMS:
        for t in range(60):
            tid = f"{dom}_{t}"
            attacked = t % 3 == 0
            n = 4
            part = "train" if t < 36 else ("val" if t < 48 else "test")
            for s in range(n):
                y = int(attacked and s >= 2)
                dps.append(dict(dp_id=f"{tid}:{s}", traj_id=tid, step=s, part=part, domain=dom, tool=f"{dom}_tool{s}",
                                state_changing=s % 2 == 1, y=y))
            L = 2 * n - 1
            ev[tid] = ([f"{dom}_tool{i // 2}" for i in range(L)], rng.normal(size=(L, 6)).astype(np.float32),
                       list(range(0, L, 2)))
    dps = pd.DataFrame(dps)
    (tmp / "oof").mkdir()
    dps.to_csv(tmp / "decision_points.csv", index=False)
    return dps, ev


def oof_table(dps, split, fold_by, rng, sets):
    rows = []
    for name, ids in sets.items():
        sub = dps.set_index("dp_id").loc[ids]
        for dp, r in sub.iterrows():
            if name == "oof":
                fold = DOMS.index(r["domain"]) if fold_by == "domain" else int(r["traj_id"].split("_")[1]) % 3
            else:
                fold = -1
            sc = float(np.clip(r["y"] * 0.7 + rng.normal(0.15, 0.3), 0.001, 0.999))
            pred = int(sc >= 0.5)
            err = int(pred != r["y"])
            rows.append(dict(dp_id=dp, split=split, set=name, fold=fold, logit=float(np.log(sc / (1 - sc))),
                             temperature=1.0, score=sc, pred=pred, y=int(r["y"]), tool_seen=int(rng.random() > 0.1),
                             monitor_error=err, dangerous=int(err and r["y"]), costly=int(err and not r["y"]),
                             weight=5.0 if err and r["y"] else 1.0))
    return pd.DataFrame(rows)


@pytest.fixture
def world(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    for k, v in {"PROCESSED": tmp_path, "MODELS": tmp_path / "models", "REPORTS3": tmp_path / "rep",
                 "FIGURES": tmp_path / "fig", "REPORTS": tmp_path / "rep1", "REPORTS2": tmp_path / "rep2",
                 "ANON": tmp_path / "anon", "SPLITS_DIR": tmp_path / "splits"}.items():
        monkeypatch.setattr(config, k, v)
    monkeypatch.setattr(config, "MONITOR_SEEDS", (config.SEED, 1, 2))
    dps, ev = make_world(tmp_path, rng)
    ids = lambda m: dps.loc[m, "dp_id"].tolist()
    splits = {
        "taskdisjoint": {"oof": ids(dps.part == "train"), "val": ids(dps.part == "val"), "test": ids(dps.part == "test")},
        "lodo_web": {"oof": ids((dps.part == "train") & (dps.domain != "web")),
                     "val": ids((dps.part == "val") & (dps.domain != "web")), "test": ids(dps.domain == "web")},
    }
    for split, sets in splits.items():
        for fb in (("task", "domain") if split.startswith("lodo") else ("task",)):
            tag = config.oof_tag(fold_by=fb)
            df = oof_table(dps, split, fb, rng, sets)
            if fb == "domain":  # the full-model rows are shared with the task-fold run
                base = pd.read_csv(tmp_path / "oof" / config.oof_tag() / f"{split}.csv")
                df = pd.concat([df[df.set == "oof"], base[base.set != "oof"]])
            for sub in ("oof", "novelty", "defer_preds"):
                (tmp_path / sub / tag).mkdir(parents=True, exist_ok=True)
            df.to_csv(tmp_path / "oof" / tag / f"{split}.csv", index=False)
            nov = df[["dp_id", "set", "fold"]].assign(nov_call=rng.random(len(df)), nov_obs=rng.random(len(df)))
            nov.to_csv(tmp_path / "novelty" / tag / f"{split}.csv", index=False)
            rest = df[df.set != "oof"][["dp_id", "set"]]
            lg = pd.concat([rest.assign(seed=config.SEED + k, p=rng.random(len(rest))) for k in range(2)])
            lg.to_csv(tmp_path / "defer_preds" / tag / f"{split}_lgbm.csv", index=False)
        for seed in (1, 2):  # full-only seeds: ensemble members, no deferrer runs
            ftag = config.oof_tag(seed=seed, fold_by="full")
            (tmp_path / "oof" / ftag).mkdir(parents=True, exist_ok=True)
            base = pd.read_csv(tmp_path / "oof" / config.oof_tag() / f"{split}.csv")
            full = base[base.set != "oof"].copy()
            full["logit"] = full["logit"] + rng.normal(0, 1.5, len(full))
            full.to_csv(tmp_path / "oof" / ftag / f"{split}.csv", index=False)
        mtag = config.oof_tag("transformer_big", "mpnet-roles", fold_by="full")  # the big mpnet monitor
        (tmp_path / "oof" / mtag).mkdir(parents=True, exist_ok=True)
        big = pd.read_csv(tmp_path / "oof" / config.oof_tag() / f"{split}.csv")
        big = big[big.set != "oof"].copy()
        big["score"] = np.clip(big["score"] + rng.normal(0, 0.2, len(big)), 0.001, 0.999)
        big.to_csv(tmp_path / "oof" / mtag / f"{split}.csv", index=False)
        (tmp_path / "oof" / "lstm_hash").mkdir(exist_ok=True)
        oof_table(dps, split, "task", rng, sets).to_csv(tmp_path / "oof" / "lstm_hash" / f"{split}.csv", index=False)

    class Data:
        pass
    data = Data()
    data.ev, data.dim = ev, 6
    return tmp_path, dps, data


def test_neural_then_evaluation(world, monkeypatch):
    pytest.importorskip("torch")
    tmp, dps, data = world
    from l2e import evaluate_deferral, neural_deferrer
    (config.MODELS / "deferrer").mkdir(parents=True)
    for split, fbs in (("taskdisjoint", ("task",)), ("lodo_web", ("task", "domain"))):
        for fb in fbs:
            tag = config.oof_tag(fold_by=fb)
            hist = neural_deferrer.run(data, tag, split, dps, n_seeds=2, hp={"max_epochs": 2},
                                       save_model=split == "taskdisjoint")
            assert len(hist) == 2
            p = pd.read_csv(tmp / "defer_preds" / tag / f"{split}_neural.csv")
            assert p["p"].notna().all() and set(p["set"]) == {"val", "test"}
    assert (config.MODELS / "deferrer" / f"neural_{config.oof_tag()}_taskdisjoint.pt").exists()

    config.ensure_dirs()
    monkeypatch.setattr("sys.argv", ["x", "--resamples", "12"])
    evaluate_deferral.main()
    res = json.loads((config.REPORTS3 / "03_deferral_eval.json").read_text())
    web = res["settings"]["Unseen domain: web"]["policies"]
    assert {"confidence", "stronger", "novelty", "lgbm_task", "lgbm_domain", "neural_task", "neural_domain"} <= set(web)
    assert {"seed_ensemble", "disagreement", "ens_disagree", "arch_disagree", "pair_disagree", "mpnet"} <= set(web)
    assert res["settings"]["Task-disjoint test"]["ensemble_seeds"] == 3
    assert res["settings"]["Task-disjoint test"]["variants"] == 1  # full-only seeds add no deferrer variants
    assert "Unseen domains (mean)" in res["results"]
    t = res["results"]["Task-disjoint test"]["2%"]["table"]
    assert t["oracle"][-1]["dangerous_left_per_1k"]["mean"] <= t["random"][-1]["dangerous_left_per_1k"]["mean"]
    md = (config.REPORTS3 / "03_deferral_eval.md").read_text()
    assert "Attack rate 2%" in md and "LightGBM deferrer (domain folds)" in md and "3-seed ensemble" in md
    assert (config.FIGURES / "09_risk_vs_escalation.png").exists()
    assert (config.FIGURES / "10_domain_fold_labels.png").exists()
