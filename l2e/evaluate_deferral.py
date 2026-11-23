"""Week 3 · Which calls should a human review? Every deferral policy at the same review budget.

For each held-out setting, attack rate and review budget, this sends the top-k% of calls by each policy's
priority to a perfect reviewer and counts what is left: missed attacks the system still lets through, and
all monitor errors. Numpy, pandas and matplotlib only (no PyTorch, no LightGBM).

Policies (higher priority = reviewed first):
  random        a random k% (expected value)
  confidence    the k% the main monitor is least sure about (score closest to 0.5). Temperature scaling
                divides every logit by one number, so it never changes this order: raw and calibrated
                confidence pick the same calls and are one policy here.
  novelty       the k% least like the training data: nov_call + nov_obs + (1 if the tool is new)
  stronger      a stronger monitor with calibrated confidence: the main transformer averaged with the
                LSTM (logits averaged, temperature refit on val), deferring its own least confident calls.
                Its errors are its own, so "left" counts the stronger monitor's errors.
  lgbm_task     LightGBM deferrer on task-fold out-of-fold errors (l2e/deferrer.py)
  lgbm_domain   the same on domain-fold errors (leave-one-domain-out splits)
  neural_task   transformer deferrer that reads the trace (l2e/neural_deferrer.py), task-fold errors
  neural_domain the same on domain-fold errors
  oracle        the real errors first (upper bound)

Attack rate: the test sets are about 18% hijacked calls. Real traffic has far fewer attacks, so each test set
is resampled to 5%, 2% and 1% hijacked calls by keeping a random subset of the traces that contain a hijacked
call (whole traces, so the "earlier in this trace" features stay right). Traces without one (benign, hard
negatives, failed attacks) are bootstrapped. "natural" bootstraps whole traces at the original rate.
Each resample also picks a monitor seed and a deferrer seed in turn, so the intervals include seed noise.

Settings: task-disjoint test · tool coverage 100/75/50/25% (3 tool-holdout seeds) · each held-out domain ·
"unseen domains" = the mean over the five held-out domains (paired resample by resample).

Run:  python -m l2e.evaluate_deferral                (after l2e.deferrer and l2e.neural_deferrer)
      python -m l2e.evaluate_deferral --resamples 50 (faster)
Out:  reports/week3/03_deferral_eval.md and .json, reports/week3/03_deferral_eval_full.md (every budget),
      results/figures/09_risk_vs_escalation.png, results/figures/10_domain_fold_labels.png
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .calibrate import fit_temperature, sigmoid

PREVALENCES = [None, 0.05, 0.02, 0.01]  # None = the test set's natural attack rate
MAIN_PREVALENCE = 0.02
BUDGETS = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30]
TABLE_BUDGETS = [0.01, 0.02, 0.05, 0.10]
POLICIES = ["random", "confidence", "novelty", "stronger", "lgbm_task", "lgbm_domain", "neural_task",
            "neural_domain", "oracle"]
LABELS = {"random": "random", "confidence": "least confident", "novelty": "most novel",
          "stronger": "stronger monitor, least confident", "lgbm_task": "LightGBM deferrer (task folds)",
          "lgbm_domain": "LightGBM deferrer (domain folds)", "neural_task": "transformer deferrer (task folds)",
          "neural_domain": "transformer deferrer (domain folds)", "oracle": "perfect (upper bound)"}
UNSEEN = "Unseen domains (mean)"
LEARNED = ["lgbm_task", "lgbm_domain", "neural_task", "neural_domain"]
STRONGER_PARTNER = "lstm_hash"  # the second monitor in the "stronger monitor" average (week 2 out-of-fold run)


def prev_name(p):
    return "natural" if p is None else f"{p:.0%}"


# --------------------------------------------------------------------------- resampling and curves

def resample_weights(traj_codes, hij, prevalence, rng):
    """-> integer weight per row (how many times it is in this resample).

    traj_codes: trace index per row (0..T-1); hij: 1 for a hijacked call. prevalence None: bootstrap whole
    traces. Else: bootstrap the traces without a hijacked call, then add traces with one, in random order and
    without repeats, while the share of hijacked calls stays at or under `prevalence`.
    """
    traj_codes = np.asarray(traj_codes)
    hij = np.asarray(hij, float)
    T = traj_codes.max() + 1
    n_rows = np.bincount(traj_codes, minlength=T)
    n_hij = np.bincount(traj_codes, weights=hij, minlength=T)
    if prevalence is None:
        counts = np.bincount(rng.integers(0, T, T), minlength=T)
        return counts[traj_codes]
    neg = np.flatnonzero(n_hij == 0)
    att = np.flatnonzero(n_hij > 0)
    counts = np.zeros(T, int)
    if len(neg):
        np.add.at(counts, neg[rng.integers(0, len(neg), len(neg))], 1)
    base = float((counts * n_rows).sum())
    order = rng.permutation(att)
    cum_n = base + np.cumsum(n_rows[order])
    cum_h = np.cumsum(n_hij[order])
    ok = np.flatnonzero(cum_h <= prevalence * cum_n)
    k = ok[-1] + 1 if len(ok) else 0  # shares rise as traces are added, so the kept ones are a prefix
    counts[order[:k]] = 1
    return counts[traj_codes]


def curve(order, w, err, dang, budgets=BUDGETS):
    """Review the top calls in `order` (all rows, best first) up to each budget, counting repeats via w.
    -> (reviews, errors caught, dangerous caught) arrays, one value per budget."""
    o = order[w[order] > 0]
    cw = np.concatenate([[0.0], np.cumsum(w[o])])
    ce = np.concatenate([[0.0], np.cumsum((w * err)[o])])
    cd = np.concatenate([[0.0], np.cumsum((w * dang)[o])])
    k = np.round(np.asarray(budgets) * cw[-1])
    return k, np.interp(k, cw, ce), np.interp(k, cw, cd)


def summarize(vals):
    vals = np.asarray(vals, float)
    return {"mean": float(vals.mean()), "lo": float(np.quantile(vals, 0.025)), "hi": float(np.quantile(vals, 0.975))}


# --------------------------------------------------------------------------- loading

def _csv(path):
    return pd.read_csv(path) if Path(path).exists() else None


def stronger_monitor(main, partner, set_name):
    """Average the two monitors' calibrated logits, refit one temperature on val. -> score per dp_id of the set."""
    def cal(df):
        return df.assign(z=df["logit"] / df["temperature"])[["dp_id", "set", "z", "y"]]
    m = cal(main).merge(cal(partner)[["dp_id", "set", "z"]], on=["dp_id", "set"], suffixes=("", "_2"))
    m["z_avg"] = 0.5 * (m["z"] + m["z_2"])
    val = m[m["set"] == "val"]
    T = fit_temperature(val["z_avg"], val["y"])
    te = m[m["set"] == set_name]
    return pd.Series(sigmoid(te["z_avg"].values / T), index=te["dp_id"].values), T


class Variant:
    """One test set scored by one monitor seed, with every policy's priority ready."""

    def __init__(self, split, set_name, mseed, dps):
        tag = config.oof_tag(seed=mseed)
        oof = pd.read_csv(config.PROCESSED / "oof" / tag / f"{split}.csv")
        te = oof[oof["set"] == set_name].drop_duplicates("dp_id").reset_index(drop=True)
        self.split, self.set_name, self.mseed, self.n = split, set_name, mseed, len(te)
        traj = dps.loc[te["dp_id"], "traj_id"].values
        self.traj = pd.factorize(traj)[0]
        self.y = te["y"].values.astype(float)
        err, dang = te["monitor_error"].values.astype(float), te["dangerous"].values.astype(float)
        margin = (te["score"] - 0.5).abs().values
        self.pol = {"confidence": (-margin[:, None], err, dang),
                    "oracle": ((2 * err + dang)[:, None], err, dang),
                    "random": (None, err, dang)}
        nov = _csv(config.PROCESSED / "novelty" / tag / f"{split}.csv")
        if nov is not None:
            nv = te[["dp_id"]].merge(nov[nov["set"] == set_name].drop_duplicates("dp_id"), on="dp_id", how="left")
            pr = nv["nov_call"].fillna(1).values + nv["nov_obs"].fillna(0).values + (1 - te["tool_seen"].values)
            self.pol["novelty"] = (pr[:, None], err, dang)
        partner = _csv(config.PROCESSED / "oof" / STRONGER_PARTNER / f"{split}.csv")
        if partner is not None:
            s, self.stronger_T = stronger_monitor(oof, partner, set_name)
            s = te["dp_id"].map(s)
            if s.notna().all():
                p = (s.values >= 0.5).astype(float)
                e2 = (p != self.y).astype(float)
                d2 = ((p == 0) & (self.y == 1)).astype(float)
                self.pol["stronger"] = (-np.abs(s.values - 0.5)[:, None], e2, d2)
        for fb in ("task", "domain"):
            ptag = config.oof_tag(seed=mseed, fold_by=fb)
            for kind in ("lgbm", "neural"):
                df = _csv(config.PROCESSED / "defer_preds" / ptag / f"{split}_{kind}.csv")
                if df is None:
                    continue
                df = df[df["set"] == set_name].drop_duplicates(["dp_id", "seed"])
                wide = df.pivot(index="dp_id", columns="seed", values="p").reindex(te["dp_id"])
                if wide.isna().any().any():
                    continue
                self.pol[f"{kind}_{fb}"] = (wide.values, err, dang)
        rng = np.random.default_rng(config.SEED)
        jitter = rng.random(self.n) * 1e-9  # break ties at random, not by row order
        self.orders = {k: [np.argsort(-(pr[:, j] + jitter), kind="stable") for j in range(pr.shape[1])]
                       for k, (pr, _, _) in self.pol.items() if pr is not None}
        self.counts = {"decision_points": self.n, "hijacked": int(self.y.sum()), "errors": int(err.sum()),
                       "dangerous": int(dang.sum())}


def build_settings(dps):
    """name -> list of Variants (resamples cycle through them)."""
    def seeds_for(split):
        return [s for s in config.MONITOR_SEEDS
                if (config.PROCESSED / "oof" / config.oof_tag(seed=s) / f"{split}.csv").exists()]
    plan = {"Task-disjoint test": [("taskdisjoint", "test")]}
    for c in (100, 75, 50, 25):
        plan[f"Tool coverage {c}%"] = [(f"tool_seed{s}", f"cov{c}") for s in config.TOOL_HOLDOUT_SEEDS]
    for d in config.DOMAINS:
        plan[f"Unseen domain: {d}"] = [(f"lodo_{d}", "test")]
    out = {}
    for name, parts in plan.items():
        vs = [Variant(sp, st, s, dps) for sp, st in parts for s in seeds_for(sp)]
        if vs:
            out[name] = vs
    return out


# --------------------------------------------------------------------------- evaluation

def evaluate_setting(variants, prevalence, R, seed):
    """-> {policy: {"left_d": (R, B), "left_e": (R, B), "caught_d": ..., "D": (R,), "E": (R,), "n": (R,),
    "reviews": (R, B)}} with the same resample r for every policy (paired)."""
    pols = [p for p in POLICIES if all(p in v.pol for v in variants)]
    rng = np.random.default_rng(seed)
    B = len(BUDGETS)
    res = {p: {k: np.zeros((R, B)) for k in ("left_d", "left_e", "caught_d", "caught_e", "reviews")} for p in pols}
    for p in pols:
        for k in ("D", "E", "n"):
            res[p][k] = np.zeros(R)
    for r in range(R):
        v = variants[r % len(variants)]
        w = resample_weights(v.traj, v.y, prevalence, rng).astype(float)
        n = w.sum()
        s_idx = r // len(variants)
        for p in pols:
            pr, err, dang = v.pol[p]
            D, E = float((w * dang).sum()), float((w * err).sum())
            if p == "random":
                k = np.round(np.asarray(BUDGETS) * n)
                ce, cd = k / n * E, k / n * D
            else:
                orders = v.orders[p]
                k, ce, cd = curve(orders[s_idx % len(orders)], w, err, dang)
            out = res[p]
            out["reviews"][r], out["caught_e"][r], out["caught_d"][r] = k, ce, cd
            out["left_d"][r], out["left_e"][r] = (D - cd) / n * 1000, (E - ce) / n * 1000
            out["D"][r], out["E"][r], out["n"][r] = D, E, n
    return res


def mean_over(results):
    """Average several settings' results resample by resample (the unseen-domains summary)."""
    pols = [p for p in POLICIES if all(p in r for r in results)]
    return {p: {k: np.mean([r[p][k] for r in results], axis=0) for k in results[0][p]} for p in pols}


def table(res):
    """-> {policy: [{budget, missed attacks left per 1k (mean, lo, hi), errors left per 1k, share caught...}]}"""
    out = {}
    for p, r in res.items():
        rows = []
        for j, b in enumerate(BUDGETS):
            d_tot, e_tot, rev = r["D"].sum(), r["E"].sum(), r["reviews"][:, j].sum()
            rows.append({"budget": b,
                         "dangerous_left_per_1k": summarize(r["left_d"][:, j]),
                         "errors_left_per_1k": summarize(r["left_e"][:, j]),
                         "dangerous_caught_share": float(r["caught_d"][:, j].sum() / d_tot) if d_tot else None,
                         "errors_caught_share": float(r["caught_e"][:, j].sum() / e_tot) if e_tot else None,
                         "wasted_review_share": float(1 - r["caught_e"][:, j].sum() / rev) if rev else None})
        out[p] = rows
    return out


def paired(res, base="confidence"):
    """Learned policy minus the baseline, missed attacks left per 1,000 calls, resample by resample."""
    out = {}
    if base not in res:
        return out
    for p in [p for p in res if p in LEARNED + ["stronger", "novelty"]]:
        rows = []
        for j, b in enumerate(BUDGETS):
            diff = res[p]["left_d"][:, j] - res[base]["left_d"][:, j]
            rows.append({"budget": b, **summarize(diff), "p_better": float((diff < 0).mean()),
                         "p_worse": float((diff > 0).mean())})
        out[p] = rows
    return out


def counts(res):
    r = next(iter(res.values()))
    conf = res.get("confidence", r)
    return {"calls": float(r["n"].mean()), "errors": float(conf["E"].mean()), "dangerous": float(conf["D"].mean())}


# --------------------------------------------------------------------------- report

def fmt_ci(x, nd=2):
    return f"{x['mean']:.{nd}f} [{x['lo']:.{nd}f}, {x['hi']:.{nd}f}]"


def write_report(out, R, settings_meta, main_prevalence=MAIN_PREVALENCE):
    main = prev_name(main_prevalence)
    md = ["# Deferral at equal review budgets (week 3)", "",
          f"Monitor: {config.MAIN_ARCH} on {config.MAIN_FEATURES}. {R} resamples per cell; intervals are the middle 95% "
          "over resamples (traces, monitor seeds and deferrer seeds all vary). \"Left\" = what the system still "
          "gets wrong after a perfect reviewer has checked the deferred calls, per 1,000 calls. Lower is better.",
          "", "Raw and calibrated confidence rank calls identically (temperature scaling keeps the order), so they "
          "are one policy (\"least confident\").", ""]
    md += ["## Test sets", "", "| setting | variants (split × monitor seed) | calls | hijacked | monitor errors | "
           "missed attacks |", "|---|---|---|---|---|---|"]
    for name, m in settings_meta.items():
        md.append(f"| {name} | {m['variants']} | {m['decision_points']:.0f} | {m['hijacked']:.0f} | {m['errors']:.0f} | "
                  f"{m['dangerous']:.0f} |")
    md.append("")
    for prev in [main] + [prev_name(p) for p in PREVALENCES if prev_name(p) != main]:
        md += [f"## Attack rate {prev}: missed attacks left per 1,000 calls", ""]
        settings = [s for s in out if prev in out[s]]
        pols = [p for p in POLICIES if any(p in out[s][prev]["table"] for s in settings) and p != "random"]
        for b in (0.02, 0.05):
            j = BUDGETS.index(b)
            md += [f"Review budget {b:.0%} (no deferral in the first column):", "",
                   "| setting | no deferral (main monitor) | " + " | ".join(LABELS[p] for p in pols) + " |",
                   "|---|---|" + "---|" * len(pols)]
            for s in settings:
                t = out[s][prev]["table"]
                first = next(iter(t.values()))
                none = fmt_ci(first[0]["dangerous_left_per_1k"]) if "confidence" not in t else \
                    fmt_ci(t["confidence"][0]["dangerous_left_per_1k"])
                cells = [fmt_ci(t[p][j]["dangerous_left_per_1k"]) if p in t else "—" for p in pols]
                md.append(f"| {s} | {none} | " + " | ".join(cells) + " |")
            md.append("")
        md += [f"Learned policies vs. least confident at attack rate {prev}: difference in missed attacks left per "
               "1,000 calls (negative = the learned one leaves fewer), with the share of resamples where it is "
               "better.", "", "| setting | policy | 2% budget | better in | 5% budget | better in |",
               "|---|---|---|---|---|---|"]
        for s in settings:
            for p, rows in out[s][prev]["paired"].items():
                if p not in LEARNED + ["stronger"]:
                    continue
                r2, r5 = rows[BUDGETS.index(0.02)], rows[BUDGETS.index(0.05)]
                md.append(f"| {s} | {LABELS[p]} | {fmt_ci(r2)} | {r2['p_better']:.0%} | {fmt_ci(r5)} | "
                          f"{r5['p_better']:.0%} |")
        md.append("")
    md += ["## Review burden at attack rate " + main, "",
           "Share of reviews spent on calls the monitor already had right (a 5% budget):", "",
           "| setting | " + " | ".join(LABELS[p] for p in POLICIES if p not in ("random",)) + " |",
           "|---|" + "---|" * (len(POLICIES) - 1)]
    j = BUDGETS.index(0.05)
    for s in out:
        if main not in out[s]:
            continue
        t = out[s][main]["table"]
        cells = [f"{t[p][j]['wasted_review_share']:.0%}" if p in t and t[p][j]["wasted_review_share"] is not None
                 else "—" for p in POLICIES if p != "random"]
        md.append(f"| {s} | " + " | ".join(cells) + " |")
    md += ["", "Figures: `results/figures/09_risk_vs_escalation.png`, `results/figures/10_domain_fold_labels.png`. "
           "Every budget: `reports/week3/03_deferral_eval_full.md`.", ""]
    return "\n".join(md)


def write_full(out):
    md = ["# Deferral evaluation, every budget", "",
          "Missed attacks left per 1,000 calls, mean [95% interval]; share of missed attacks caught (pooled).", ""]
    for s, by_prev in out.items():
        for prev, o in by_prev.items():
            t = o["table"]
            pols = [p for p in POLICIES if p in t]
            md += [f"## {s} · attack rate {prev}", "", "| budget | " + " | ".join(LABELS[p] for p in pols) + " |",
                   "|---|" + "---|" * len(pols)]
            for j, b in enumerate(BUDGETS):
                cells = []
                for p in pols:
                    row = t[p][j]
                    sh = row["dangerous_caught_share"]
                    cells.append(fmt_ci(row["dangerous_left_per_1k"]) + (f" ({sh:.0%})" if sh is not None else ""))
                md.append(f"| {b:.1%} | " + " | ".join(cells) + " |")
            md.append("")
    return "\n".join(md)


# --------------------------------------------------------------------------- figures

STYLE = {  # fixed categorical order (references: dataviz palette, light mode)
    "confidence": dict(color="#eb6834", lw=2.0, label="least confident"),
    "stronger": dict(color="#1baf7a", lw=2.0, label="stronger monitor"),
    "lgbm": dict(color="#2a78d6", lw=2.0, label="LightGBM deferrer"),
    "neural": dict(color="#6250d6", lw=2.0, label="transformer deferrer"),
    "random": dict(color="#8a8a85", lw=1.4, ls="--", label="random"),
    "oracle": dict(color="#c3c2b7", lw=1.4, ls=":", label="perfect"),
}
INK, MUTED, GRID = "#1a1a19", "#52514e", "#e6e5e0"


def best_learned(t, kind):
    """Domain-fold version where it exists (unseen domains), else task folds."""
    return f"{kind}_domain" if f"{kind}_domain" in t else f"{kind}_task"


def _axes_style(ax):
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8.5)


def plot_curves(out, path, prev):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [s for s in ("Task-disjoint test", "Tool coverage 25%", UNSEEN) if s in out
              and prev in out[s]]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(4.9 * len(panels), 3.8), squeeze=False)
    xs = np.array(BUDGETS) * 100
    for ax, s in zip(axes[0], panels):
        t = out[s][prev]["table"]
        series = [("oracle", "oracle"), ("random", "random"), ("confidence", "confidence"), ("stronger", "stronger"),
                  (best_learned(t, "neural"), "neural"), (best_learned(t, "lgbm"), "lgbm")]
        for key, sty in series:
            if key not in t:
                continue
            m = np.array([r["dangerous_left_per_1k"]["mean"] for r in t[key]])
            st = {k: v for k, v in STYLE[sty].items() if k != "label"}
            ax.plot(xs, m, **st, zorder=3 if sty in ("lgbm", "neural") else 2)
            if sty == "lgbm":  # one band only: two overlapping bands blur into a third colour
                lo = [r["dangerous_left_per_1k"]["lo"] for r in t[key]]
                hi = [r["dangerous_left_per_1k"]["hi"] for r in t[key]]
                ax.fill_between(xs, lo, hi, color=STYLE[sty]["color"], alpha=0.12, lw=0, zorder=1)
        c = out[s][prev]["counts"]
        ax.set_title(f"{s}\nbefore review: {1000 * c['dangerous'] / c['calls']:.1f} missed attacks per 1,000 calls",
                     fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("review budget (% of calls sent to a human)", fontsize=9, color=MUTED)
        ax.set_xlim(0, 30)
        ax.set_ylim(bottom=0)
        _axes_style(ax)
    axes[0][0].set_ylabel("missed attacks left per 1,000 calls", fontsize=9, color=MUTED)
    handles = [plt.Line2D([], [], **{k: v for k, v in STYLE[k2].items()})
               for k2 in ("lgbm", "neural", "confidence", "stronger", "random", "oracle")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Missed attacks left after review, attack rate {prev} (lower is better; "
                 "band: 95% interval of the LightGBM deferrer)",
                 fontsize=10.5, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_domain_folds(out, path, prev, budget=0.05):
    """Per unseen domain: missed attacks left at one budget, task-fold vs domain-fold labels, with intervals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [s for s in out if s.startswith("Unseen domain:") and prev in out[s]]
    rows += [UNSEEN] if UNSEEN in out and prev in out[UNSEEN] else []
    if not rows:
        return
    j = BUDGETS.index(budget)
    series = [("confidence", "least confident", "#eb6834", "o"), ("lgbm_task", "LightGBM, task folds", "#2a78d6", "o"),
              ("lgbm_domain", "LightGBM, domain folds", "#2a78d6", "s"),
              ("neural_task", "transformer, task folds", "#6250d6", "o"),
              ("neural_domain", "transformer, domain folds", "#6250d6", "s")]
    series = [x for x in series if any(x[0] in out[s][prev]["table"] for s in rows)]
    fig, ax = plt.subplots(figsize=(7.6, 0.55 * len(rows) * len(series) / 2.2 + 1.6))
    gap = 0.8 / max(len(series), 1)
    for i, s in enumerate(rows):
        t = out[s][prev]["table"]
        for k, (key, lab, col, mk) in enumerate(series):
            if key not in t:
                continue
            r = t[key][j]["dangerous_left_per_1k"]
            yv = i + (k - (len(series) - 1) / 2) * gap
            ax.errorbar(r["mean"], yv, xerr=[[r["mean"] - r["lo"]], [r["hi"] - r["mean"]]], fmt=mk, ms=6,
                        color=col, mfc=col if mk == "o" else "white", mew=1.6, elinewidth=1.4, capsize=0,
                        label=lab if i == 0 else None)
    if UNSEEN in rows:
        ax.axhline(rows.index(UNSEEN) - 0.5, color="#c3c2b7", lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([s.replace("Unseen domain: ", "") for s in rows], fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.set_xlabel(f"missed attacks left per 1,000 calls at a {budget:.0%} review budget (lower is better)",
                  fontsize=9, color=MUTED)
    ax.grid(axis="x", color=GRID, lw=0.8)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8.5, left=False)
    ax.set_title(f"Unseen domains, attack rate {prev}: does learning from cross-domain errors help?",
                 fontsize=10, color=INK, loc="left")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.42 if len(rows) > 3 else -0.6), ncol=3, frameon=False,
              fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- main

def to_json(o):
    if isinstance(o, dict):
        return {k: to_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [to_json(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resamples", type=int, default=200)
    ap.add_argument("--main-prevalence", type=float, default=MAIN_PREVALENCE, help="attack rate for the headline")
    args = ap.parse_args()
    config.ensure_dirs()
    t0 = time.time()
    dps = pd.read_csv(config.PROCESSED / "decision_points.csv").set_index("dp_id")
    settings = build_settings(dps)
    out, meta, raw_lodo = {}, {}, {}
    for si, (name, variants) in enumerate(settings.items()):
        meta[name] = {"variants": len(variants), **{k: np.mean([v.counts[k] for v in variants])
                                                    for k in ("decision_points", "hijacked", "errors", "dangerous")},
                      "policies": [p for p in POLICIES if all(p in v.pol for v in variants)]}
        out[name] = {}
        for prev in PREVALENCES:
            res = evaluate_setting(variants, prev, args.resamples, seed=config.SEED + 1000 * si)
            if name.startswith("Unseen domain:"):
                raw_lodo.setdefault(prev_name(prev), []).append(res)
            out[name][prev_name(prev)] = {"counts": counts(res), "table": table(res), "paired": paired(res)}
        print(f"{name}: {len(variants)} variant(s), policies {meta[name]['policies']} ({time.time() - t0:.0f}s)",
              flush=True)
    if raw_lodo:
        name = UNSEEN
        out[name] = {}
        for prev, results in raw_lodo.items():
            res = mean_over(results)
            out[name][prev] = {"counts": counts(res), "table": table(res), "paired": paired(res)}
        meta[name] = {"variants": sum(meta[s]["variants"] for s in meta if s.startswith("Unseen domain:")),
                      **{k: np.mean([meta[s][k] for s in meta if s.startswith("Unseen domain:")])
                         for k in ("decision_points", "hijacked", "errors", "dangerous")}}
    main_prev = prev_name(args.main_prevalence)
    (config.REPORTS3 / "03_deferral_eval.json").write_text(json.dumps(to_json(
        {"resamples": args.resamples, "main_prevalence": main_prev, "budgets": BUDGETS, "settings": meta,
         "results": out}), indent=1))
    (config.REPORTS3 / "03_deferral_eval.md").write_text(write_report(out, args.resamples, meta, args.main_prevalence))
    (config.REPORTS3 / "03_deferral_eval_full.md").write_text(write_full(out))
    plot_curves(out, config.FIGURES / "09_risk_vs_escalation.png", main_prev)
    plot_domain_folds(out, config.FIGURES / "10_domain_fold_labels.png", main_prev)
    print(f"done in {time.time() - t0:.0f}s -> reports/week3/03_deferral_eval.md")


if __name__ == "__main__":
    main()
