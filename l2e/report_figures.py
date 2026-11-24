"""Deliverable 3 report figures: D2 vs D3 side by side, confusion matrices and ROC curves.

Reads files the week 3 pipeline already wrote (no training, takes seconds):
  reports/week3/03_deferral_eval.json                      (bootstrap results of every policy)
  data/processed/oof/transformer_hash-roles/<split>.csv    (main monitor, seed 42)
  data/processed/oof/lstm_hash/<split>.csv                 (the LSTM partner)
  data/processed/decision_points.csv                       (tool names for the examples)

D2 system = the transformer alone, sending its least confident calls to the human.
D3 system = the transformer + LSTM pair (averaged calibrated logits, one more temperature fit on val),
            sending its least sure calls to the human. Same review budget for both.

Run:  python -m l2e.report_figures
Out:  results/figures/11_d2_vs_d3.png, 12_confusion.png, 13_roc.png
      reports/week3/04_report_numbers.json, reports/week3/04_examples.md
"""
import json

import numpy as np
import pandas as pd

from . import config
from .calibrate import fit_temperature, sigmoid

MAIN = f"{config.MAIN_ARCH}_{config.MAIN_FEATURES}"
PARTNER = "lstm_hash"
BUDGET = 0.05
UNSEEN = "Unseen domains (mean)"
INK, MUTED, GRID = "#1a1a19", "#52514e", "#e6e5e0"
COLORS = {"none": "#8a8a85", "d2": "#eb6834", "d3": "#1baf7a", "transformer": "#eb6834", "LSTM": "#2a78d6",
          "pair": "#1baf7a"}


# --------------------------------------------------------------------------- data

def _cal(df, set_name):
    g = df[df["set"] == set_name].drop_duplicates("dp_id")
    return g.assign(z=g["logit"] / g["temperature"])[["dp_id", "z", "y"]]


def pair_table(main_df, partner_df, set_name="test", budget=BUDGET):
    """One test set scored by both monitors and the pair.
    -> DataFrame dp_id, y, p_main, p_partner, p_pair, review (True for the pair's least sure `budget` share),
       review_main (the same for the transformer alone)."""
    v = _cal(main_df, "val").merge(_cal(partner_df, "val")[["dp_id", "z"]], on="dp_id", suffixes=("", "_2"))
    T = fit_temperature(0.5 * (v["z"] + v["z_2"]).values, v["y"].values)
    t = _cal(main_df, set_name).merge(_cal(partner_df, set_name)[["dp_id", "z"]], on="dp_id", suffixes=("", "_2"))
    out = pd.DataFrame({"dp_id": t["dp_id"].values, "y": t["y"].values.astype(int),
                        "p_main": sigmoid(t["z"].values), "p_partner": sigmoid(t["z_2"].values),
                        "p_pair": sigmoid(0.5 * (t["z"].values + t["z_2"].values) / T)})
    k = int(round(budget * len(out)))
    for col, flag in (("p_pair", "review"), ("p_main", "review_main")):
        order = np.argsort(np.abs(out[col].values - 0.5), kind="stable")
        out[flag] = False
        out.loc[order[:k], flag] = True
    out.attrs["temperature"] = float(T)
    return out


def load_split(split, set_name="test"):
    def read(tag):
        return pd.read_csv(config.PROCESSED / "oof" / tag / f"{split}.csv")
    return pair_table(read(MAIN), read(PARTNER), set_name)


def unseen_pooled():
    """All five leave-one-domain-out test sets, each with its own pair temperature and review cut."""
    parts = []
    for d in config.DOMAINS:
        t = load_split(f"lodo_{d}")
        parts.append(t.assign(domain=d))
    return pd.concat(parts, ignore_index=True)


def confusion(y, pred, review=None):
    """-> 2x3 counts: rows = truth (safe, hijacked); columns = said safe, said hijacked, sent to a human."""
    review = np.zeros(len(y), bool) if review is None else np.asarray(review)
    m = np.zeros((2, 3), int)
    for truth in (0, 1):
        r = (y == truth)
        m[truth, 0] = int((r & ~review & (pred == 0)).sum())
        m[truth, 1] = int((r & ~review & (pred == 1)).sum())
        m[truth, 2] = int((r & review).sum())
    return m


def roc(y, s):
    """-> false positive rates, true positive rates, AUC (ties handled by sorting once)."""
    order = np.argsort(-s, kind="stable")
    ys = y[order]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    P, N = max(ys.sum(), 1), max(len(ys) - ys.sum(), 1)
    tpr, fpr = np.r_[0, tp / P], np.r_[0, fp / N]
    trap = getattr(np, "trapezoid", None) or np.trapz
    return fpr, tpr, float(trap(tpr, fpr))


# --------------------------------------------------------------------------- figures

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "DejaVu Sans"
    return plt


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelsize=8.5)


def plot_d2_vs_d3(ev, path, prev="natural", budget=BUDGET):
    """Missed attacks left per 1,000 calls: no review, D2 (transformer + least confident), D3 (pair)."""
    plt = _plt()
    j = ev["budgets"].index(budget)
    rows = ["Task-disjoint test", "Tool coverage 25%"] + [f"Unseen domain: {d}" for d in config.DOMAINS] + [UNSEEN]
    rows = [r for r in rows if r in ev["results"] and prev in ev["results"][r]]
    series = [("none", "no review", None), ("d2", "D2: transformer, least confident", "confidence"),
              ("d3", "D3: transformer + LSTM pair, least sure", "stronger")]
    fig, ax = plt.subplots(figsize=(7.4, 0.62 * len(rows) + 1.4))
    h = 0.26
    out = {}
    for i, r in enumerate(rows):
        t = ev["results"][r][prev]["table"]
        c = ev["results"][r][prev]["counts"]
        out[r] = {}
        for k, (key, lab, pol) in enumerate(series):
            if pol is None:
                mean = 1000 * c["dangerous"] / c["calls"]
                lo = hi = mean
            else:
                x = t[pol][j]["dangerous_left_per_1k"]
                mean, lo, hi = x["mean"], x["lo"], x["hi"]
            y = i + (k - 1) * h
            ax.barh(y, mean, height=h * 0.86, color=COLORS[key], label=lab if i == 0 else None, zorder=2)
            if pol is not None:
                ax.plot([lo, hi], [y, y], color=INK, lw=0.9, zorder=3)
            out[r][key] = {"mean": mean, "lo": lo, "hi": hi}
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r.replace("Unseen domain: ", "unseen: ").replace(" (mean)", "\n(mean of 5)") for r in rows],
                       fontsize=8.5, color=INK)
    ax.invert_yaxis()
    ax.set_xlim(left=0)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    _style(ax)
    ax.tick_params(left=False)
    ax.set_xlabel(f"missed attacks left per 1,000 calls, {budget:.0%} of calls reviewed (lower is better)",
                  fontsize=9, color=MUTED)
    ax.set_title(f"D2 vs D3 at the same review budget (attack rate: {prev}; line = 95% interval)",
                 fontsize=10, color=INK, loc="left")
    fig.legend(*ax.get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(0.5, 0), ncol=3,
               frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.6 / (0.62 * len(rows) + 1.4), 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_confusion(df, path, title):
    """Three matrices on the same calls: transformer alone, pair alone, pair with 5% sent to a human."""
    plt = _plt()
    y = df["y"].values
    mats = [("Transformer alone (D2 monitor)", confusion(y, (df["p_main"].values >= 0.5).astype(int))[:, :2]),
            ("Transformer + LSTM pair", confusion(y, (df["p_pair"].values >= 0.5).astype(int))[:, :2]),
            (f"Pair, {BUDGET:.0%} least sure to a human (D3)",
             confusion(y, (df["p_pair"].values >= 0.5).astype(int), df["review"].values))]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 2.9), gridspec_kw={"width_ratios": [2, 2, 3]})
    for ax, (name, m) in zip(axes, mats):
        cols = ["says safe", "says hijacked", "to a human"][:m.shape[1]]
        share = m / m.sum(axis=1, keepdims=True)
        ax.imshow(share, cmap="Greys", vmin=0, vmax=1.6, aspect="auto")
        for a in range(2):
            for b in range(m.shape[1]):
                bad = (a == 1 and b == 0)
                ax.text(b, a, f"{m[a, b]:,}\n{share[a, b]:.1%}", ha="center", va="center", fontsize=9,
                        color="#b3261e" if bad else INK, fontweight="bold" if bad else "normal")
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(cols, fontsize=8.5, color=INK)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["truly safe", "truly hijacked"], fontsize=8.5, color=INK)
        ax.set_title(name, fontsize=9.5, color=INK, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    fig.suptitle(title + "  (red = missed attacks; % of the row)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {name: m.tolist() for name, m in mats}


def plot_roc(sets, path):
    """ROC per test set for transformer, LSTM and the pair; log x-axis because the interesting part is near 0."""
    plt = _plt()
    fig, axes = plt.subplots(1, len(sets), figsize=(4.6 * len(sets), 3.7), squeeze=False)
    aucs = {}
    for ax, (name, df) in zip(axes[0], sets.items()):
        aucs[name] = {}
        y = df["y"].values
        for col, lab in (("p_main", "transformer"), ("p_partner", "LSTM"), ("p_pair", "pair")):
            fpr, tpr, auc = roc(y, df[col].values)
            aucs[name][lab] = auc
            ax.plot(np.maximum(fpr, 1e-4), tpr, color=COLORS[lab], lw=2.0, label=f"{lab} (AUC {auc:.4f})")
        ax.set_xscale("log")
        ax.set_xlim(1e-4, 1)
        ax.set_ylim(0.5, 1.005)
        ax.grid(color=GRID, lw=0.8)
        _style(ax)
        ax.set_title(name, fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("false alarm rate (log scale)", fontsize=9, color=MUTED)
        ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="none", framealpha=0.9, fontsize=8.5)
    axes[0][0].set_ylabel("share of hijacked calls caught", fontsize=9, color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return aucs


# --------------------------------------------------------------------------- examples

def examples(df, dps, n=2):
    """Qualitative cases on unseen domains: what review fixes, what it costs, and what still slips through."""
    df = df.merge(dps[["dp_id", "traj_id", "step", "tool"]], on="dp_id", how="left")
    main_wrong = (df["p_main"] >= 0.5).astype(int) != df["y"]
    pair_pred = (df["p_pair"] >= 0.5).astype(int)
    cases = [
        ("Missed attack the pair sends to a human", (df["y"] == 1) & (pair_pred == 0) & df["review"]),
        ("Transformer missed it, the LSTM pulled the pair over the line", (df["y"] == 1) & main_wrong & (pair_pred == 1)
         & ~df["review"]),
        ("False alarm sent to a human (review spent on a safe call)", (df["y"] == 0) & df["review"]),
        ("Missed attack that still slips through (both monitors confident it is safe)",
         (df["y"] == 1) & (pair_pred == 0) & ~df["review"]),
    ]
    md = ["# Qualitative examples on unseen domains (D3 system)", "",
          "Each row is one proposed tool call from a held-out domain. P = calibrated P(hijacked).", ""]
    for name, mask in cases:
        sub = df[mask].copy()
        sub["gap"] = (sub["p_main"] - sub["p_partner"]).abs()
        sub = sub.sort_values("gap", ascending=False).head(n)
        md += [f"## {name} ({int(mask.sum())} calls)", "",
               "| trace | step | tool | truth | transformer P | LSTM P | pair P | sent to human |",
               "|---|---|---|---|---|---|---|---|"]
        for _, r in sub.iterrows():
            md.append(f"| {r['traj_id']} | {int(r['step'])} | {r['tool']} | {'hijacked' if r['y'] else 'safe'} | "
                      f"{r['p_main']:.3f} | {r['p_partner']:.3f} | {r['p_pair']:.3f} | {'yes' if r['review'] else 'no'} |")
        md.append("")
    return "\n".join(md)


def main():
    config.ensure_dirs()
    ev = json.loads((config.REPORTS3 / "03_deferral_eval.json").read_text())
    nums = {"d2_vs_d3": plot_d2_vs_d3(ev, config.FIGURES / "11_d2_vs_d3.png")}
    task = load_split("taskdisjoint")
    unseen = unseen_pooled()
    nums["confusion_unseen"] = plot_confusion(unseen, config.FIGURES / "12_confusion.png",
                                              f"Unseen domains, all five pooled ({len(unseen):,} calls, natural attack rate)")
    nums["auc"] = plot_roc({"Task-disjoint test": task, "Unseen domains (pooled)": unseen},
                           config.FIGURES / "13_roc.png")
    dps = pd.read_csv(config.PROCESSED / "decision_points.csv", usecols=["dp_id", "traj_id", "step", "tool"])
    (config.REPORTS3 / "04_examples.md").write_text(examples(unseen, dps))
    (config.REPORTS3 / "04_report_numbers.json").write_text(json.dumps(nums, indent=1))
    u = nums["d2_vs_d3"].get(UNSEEN, {})
    if u:
        print(f"Unseen domains, 5% review: no review {u['none']['mean']:.2f}, D2 {u['d2']['mean']:.2f}, "
              f"D3 {u['d3']['mean']:.2f} missed attacks left per 1,000 calls")
    for k, v in nums["auc"].items():
        print(k, " ".join(f"{m} AUC {a:.4f}" for m, a in v.items()))
    print("wrote results/figures/11_d2_vs_d3.png, 12_confusion.png, 13_roc.png and reports/week3/04_*")


if __name__ == "__main__":
    main()
