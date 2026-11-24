"""Trace replay console, version 3 (Deliverable 3).

Pick a held-out agent trace and step through it. At each tool call the app shows what the monitors can see
(earlier steps + the proposed call, never its result), both monitors' scores, the decision, and why.
Deferred calls wait for a reviewer to approve or reject; the reviewer log says whether each review was right.

Modes (L2E_APP_MODE):
  pair    (default) the week 3 result: the main transformer and the LSTM, calibrated logits averaged and
          recalibrated. A call goes to the human when the pair is among the 5% least sure calls (threshold
          set on validation). Reasons: the pair is unsure, the two monitors disagree, a new tool, novel text.
          See l2e/pair.py.
  single  the Deliverable 2 setup: the main monitor and the learned LightGBM deferrer (v2 with text novelty
          when it exists), deferring the top 5% P(monitor is wrong), with reasons from its trees.
  lr      the week 1 logistic regression with a placeholder rule (defer scores between 0.3 and 0.7).
The app falls back pair -> single -> lr when a mode's files are missing, and says so at the top.

New in version 3: the pair monitor and its two scores side by side, Previous / Next buttons, a labelled
trace list (examples where deferral matters come first), a "show the answer" switch for live demos, and a
reviewer log that marks each review right or wrong.

Run (after scripts/run_week1.sh, scripts/finalize_monitor.sh, scripts/run_week3.sh):
  python ui/app.py   ->  http://127.0.0.1:7860
"""
import html
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr
import joblib
import pandas as pd

from l2e import config
from l2e.data import load_anon
from l2e.decision_points import prefix_steps
from l2e.features import prefix_features

DEFER_BAND = (0.3, 0.7)  # placeholder rule for the lr mode
TRACES = {r["id"]: r for r in load_anon("taskdisjoint", "test")}
NOTICE = ""


# --------------------------------------------------------------------------- the three modes

class PairMode:
    def __init__(self):
        from l2e.pair import PairMonitor
        self.pm = PairMonitor()
        m = self.pm.meta
        self.name = ("**Monitors:** transformer (hashed text + tool roles) and LSTM (hashed text), averaged and "
                     f"recalibrated (T = {m['temperature']:.2f})  \n**Escalation:** a call goes to the human when "
                     f"the pair is less than {0.5 + m['margin_threshold']:.1%} sure either way, i.e. P(hijacked) "
                     f"between {0.5 - m['margin_threshold']:.3f} and {0.5 + m['margin_threshold']:.3f}. That sends the "
                     f"{m['budget']:.0%} least sure calls to review on validation.")

    def analyze(self, tr):
        rows = self.pm.score_trace(tr["steps"])
        return [{"score": r["p_pair"], "verdict": r["verdict"], "defer": r["defer"], "reasons": r["reasons"],
                 "p_main": r["p_main"], "p_partner": r["p_partner"]} for r in rows]


class SingleMode:
    def __init__(self):
        from l2e.deferrer import LiveDeferrer
        from l2e.monitor import load, make_featurizer, score_trace
        model, ck = load(config.MODELS / config.MAIN_MONITOR)
        feat = make_featurizer(ck["encoder"], ck["featurizer"])
        self.score = lambda tr: [float(p) for p in score_trace(model, ck, tr["steps"], feat)]
        self.vocab = ck["vocab"]
        self.d = LiveDeferrer() if LiveDeferrer.available() else None
        self.name = (f"**Monitor:** {ck['arch']} on {ck['encoder']} features, calibrated (T = {ck['temperature']:.2f})"
                     + (f"  \n**Escalation:** learned deferrer v{self.d.version}, defers the top "
                        f"{self.d.meta['app_budget']:.0%} most likely monitor errors (P(wrong) ≥ "
                        f"{self.d.meta['threshold']:.3f})" if self.d else
                        f"  \n**Escalation:** placeholder rule, defer scores between {DEFER_BAND[0]} and {DEFER_BAND[1]}"))

    def analyze(self, tr):
        sc = self.score(tr)
        if self.d is None:
            return [{"score": p, "verdict": "HIJACKED" if p >= 0.5 else "SAFE",
                     "defer": DEFER_BAND[0] <= p <= DEFER_BAND[1], "reasons": []} for p in sc]
        q = self.d.score_trace(tr, sc, self.vocab)
        why = self.d.reasons(tr, sc, self.vocab)
        return [{"score": p, "verdict": "HIJACKED" if p >= 0.5 else "SAFE", "p_wrong": float(qi),
                 "defer": qi >= self.d.meta["threshold"], "reasons": w} for p, qi, w in zip(sc, q, why)]


class LRMode:
    name = (f"**Monitor:** six-feature logistic regression (week 1 baseline)  \n**Escalation:** placeholder rule, "
            f"defer scores between {DEFER_BAND[0]} and {DEFER_BAND[1]}")

    def __init__(self):
        self.lr = joblib.load(config.MODELS / "lr_prefix_monitor.joblib")

    def analyze(self, tr):
        out = []
        for t in range(len(tr["steps"])):
            p = float(self.lr.predict_proba([prefix_features(tr, t)])[0, 1])
            out.append({"score": p, "verdict": "HIJACKED" if p >= 0.5 else "SAFE",
                        "defer": DEFER_BAND[0] <= p <= DEFER_BAND[1], "reasons": []})
        return out


def pick_mode():
    global NOTICE
    want = os.environ.get("L2E_APP_MODE", "pair")
    order = {"pair": [PairMode, SingleMode, LRMode], "single": [SingleMode, LRMode], "lr": [LRMode]}.get(want)
    if order is None:
        raise SystemExit(f"L2E_APP_MODE must be pair, single or lr, not {want!r}")
    failed = ""
    for cls in order:
        try:
            mode = cls()
            if cls is not order[0]:
                NOTICE = (f"⚠ The {want} mode's files are missing ({failed}); showing the "
                          f"{cls.__name__.replace('Mode', '').lower()} mode instead.")
            return mode
        except (FileNotFoundError, OSError, ValueError) as e:
            failed = f"{type(e).__name__}: {e}"[:160]
    raise SystemExit("No monitor found. Run scripts/run_week1.sh first.")


MODE = pick_mode()
_CACHE = {}


def analyze(trace_id):
    # Causal monitors: the result at step t uses only steps before t and the call itself, so a whole trace is
    # scored once and cached.
    if trace_id not in _CACHE:
        _CACHE[trace_id] = MODE.analyze(TRACES[trace_id])
    return _CACHE[trace_id]


# --------------------------------------------------------------------------- the trace list

def showcase(limit=2):
    """Traces where deferral matters (first monitor error in the trace), found at startup (a few seconds)."""
    found = {"defers a missed attack": [], "defers a false alarm": [], "misses an error": []}
    for tid, tr in TRACES.items():
        for st, r in zip(tr["steps"], analyze(tid)):
            y = st["label"] == "hijacked"
            if (r["verdict"] == "HIJACKED") == y:
                continue
            key = ("defers a missed attack" if y else "defers a false alarm") if r["defer"] else "misses an error"
            if len(found[key]) < limit:
                found[key].append(tid)
            break
        if all(len(v) >= limit for v in found.values()):
            break
    return [(f"★ {k}: {t}", t) for k, v in found.items() for t in v]


FEATURED = showcase()
seen = {v for _, v in FEATURED}
for cat in ("attacked", "failed_attack", "hard_negative", "benign"):
    tid = next((i for i, r in TRACES.items() if r["category"] == cat and i not in seen), None)
    if tid is None:
        continue
    FEATURED.append((f"example ({cat.replace('_', ' ')}): {tid}", tid))
    seen.add(tid)
CHOICES = FEATURED + [(t, t) for t in sorted(set(TRACES) - seen)]
FIRST = CHOICES[0][1]


# --------------------------------------------------------------------------- rendering

def bar(label, p, color):
    w = max(1, round(100 * p))
    return (f"<div style='display:flex;align-items:center;gap:8px;margin:2px 0'>"
            f"<span style='width:88px'>{label}</span>"
            f"<span style='flex:1;height:12px;background:#e6e5e0;border-radius:6px;position:relative'>"
            f"<span style='position:absolute;left:0;top:0;bottom:0;width:{w}%;background:{color};border-radius:6px'></span>"
            f"<span style='position:absolute;left:50%;top:-3px;bottom:-3px;width:2px;background:#52514e'></span></span>"
            f"<b style='width:44px;text-align:right'>{p:.2f}</b></div>")


def badge(text, bg):
    return (f"<span style='display:inline-block;padding:2px 10px;border-radius:12px;background:{bg};color:#fff;"
            f"font-weight:600'>{text}</span>")


def render(trace_id, t, show_truth):
    tr = TRACES[trace_id]
    t = int(min(max(t, 0), len(tr["steps"]) - 1))
    rows = analyze(trace_id)
    r = rows[t]

    esc = html.escape  # placeholders like <USER> must not be read as HTML tags
    lines = [f"**Task:** {esc(tr['task'])}", ""]
    for i, s in enumerate(prefix_steps(tr, t)):
        tag = "**▶ proposed call (the monitors decide before it runs)**" if i == t else f"step {i + 1}"
        lines.append(f"{tag} · `{s['tool']}({json.dumps(s['args'])[:120]})`  \n*thought:* {esc(s['thought'][:200])}")
        if "obs" in s:
            lines.append(f"> result: {esc(s['obs'][:200])}")
    trace_md = "\n\n".join(lines)

    verdict = badge(r["verdict"], "#c0392b" if r["verdict"] == "HIJACKED" else "#2e7d32")
    decision = badge("ASK A HUMAN", "#e07b00") if r["defer"] else badge("TRUST", "#52514e")
    parts = ["<div style='font-size:15px;line-height:1.6'>",
             f"<div>Verdict: {verdict} &nbsp; Decision: {decision}</div>",
             "<div style='margin-top:10px'><b>P(hijacked)</b> <span style='opacity:.7'>(line = 0.5)</span></div>"]
    if "p_main" in r:
        parts += [bar("transformer", r["p_main"], "#2a78d6"), bar("LSTM", r["p_partner"], "#6250d6"),
                  bar("pair", r["score"], "#14161a")]
    else:
        parts.append(bar("monitor", r["score"], "#2a78d6"))
        if "p_wrong" in r:
            parts.append(f"<div>Deferrer: P(monitor is wrong) = <b>{r['p_wrong']:.3f}</b></div>")
    if r["reasons"]:
        head = "Why it asks a human" if r["defer"] else "Worth noting"
        parts.append(f"<div style='margin-top:8px'>{head}:<ul style='margin:2px 0 4px 18px'>"
                     + "".join(f"<li>{esc(x)}</li>" for x in r["reasons"]) + "</ul></div>")
    if show_truth:
        truth = tr["steps"][t]["label"]
        right = (r["verdict"] == "HIJACKED") == (truth == "hijacked")
        parts.append(f"<div style='margin-top:8px;opacity:.8'>Answer: this call is <b>{truth}</b> · the verdict is "
                     f"<b>{'right' if right else 'wrong'}</b> · trace type: {tr['category']} "
                     f"{tr.get('compliance') or ''}</div>")
    parts.append("</div>")

    cols = []
    for i in range(t + 1):
        ri = rows[i]
        row = {"step": i + 1, "tool": tr["steps"][i]["tool"]}
        if "p_main" in ri:
            row.update({"transformer": round(ri["p_main"], 3), "LSTM": round(ri["p_partner"], 3)})
        row.update({"P(hijacked)": round(ri["score"], 3), "verdict": ri["verdict"],
                    "decision": "ask a human" if ri["defer"] else "trust"})
        if show_truth:
            row["answer"] = tr["steps"][i]["label"]
        cols.append(row)
    slider = gr.Slider(minimum=1, maximum=len(tr["steps"]), value=t + 1, step=1, label="Step")
    return trace_md, "".join(parts), pd.DataFrame(cols), slider


def review(action, trace_id, step, log):
    tr = TRACES[trace_id]
    t = int(step) - 1
    truth = tr["steps"][t]["label"]
    right = (action == "rejected") == (truth == "hijacked")
    log = (log or []) + [{"trace": trace_id, "step": t + 1, "tool": tr["steps"][t]["tool"], "reviewer": action,
                          "answer": truth, "review was": "right" if right else "wrong"}]
    n_right = sum(e["review was"] == "right" for e in log)
    return pd.DataFrame(log), log, f"{len(log)} review(s), {n_right} right"


# --------------------------------------------------------------------------- layout

with gr.Blocks(title="Learning When to Escalate · trace replay") as demo:
    gr.Markdown("## Learning When to Escalate · trace replay console\n"
                "Step through a held-out agent trace. At each tool call the monitors see only the past and the "
                "proposed call, then either trust their verdict or ask a human.\n\n" + MODE.name
                + (f"\n\n{NOTICE}" if NOTICE else ""))
    log_state = gr.State([])
    with gr.Row():
        trace_dd = gr.Dropdown(CHOICES, value=FIRST, label="Trace (held-out test set; ★ = deferral matters)", scale=4)
        show_truth = gr.Checkbox(value=True, label="Show the answer", scale=1)
    with gr.Row():
        prev_b = gr.Button("◀ Previous", scale=1)
        step_sl = gr.Slider(minimum=1, maximum=len(TRACES[FIRST]["steps"]), value=1, step=1, label="Step", scale=4)
        next_b = gr.Button("Next ▶", scale=1)
    with gr.Row():
        with gr.Column(scale=3):
            trace_view = gr.Markdown()
        with gr.Column(scale=2):
            status_view = gr.HTML()
            with gr.Row():
                approve = gr.Button("Approve call", variant="secondary")
                reject = gr.Button("Reject call", variant="stop")
            timeline_view = gr.Dataframe(label="Decisions so far", interactive=False)
    review_count = gr.Markdown("0 reviews")
    review_view = gr.Dataframe(pd.DataFrame(columns=["trace", "step", "tool", "reviewer", "answer", "review was"]),
                               label="Reviewer log", interactive=False)

    outs = [trace_view, status_view, timeline_view, step_sl]
    trace_dd.change(lambda tid, st: render(tid, 0, st), [trace_dd, show_truth], outs)
    step_sl.change(lambda tid, s, st: render(tid, s - 1, st), [trace_dd, step_sl, show_truth], outs)
    show_truth.change(lambda tid, s, st: render(tid, s - 1, st), [trace_dd, step_sl, show_truth], outs)
    prev_b.click(lambda tid, s, st: render(tid, s - 2, st), [trace_dd, step_sl, show_truth], outs)
    next_b.click(lambda tid, s, st: render(tid, s, st), [trace_dd, step_sl, show_truth], outs)
    approve.click(lambda *a: review("approved", *a), [trace_dd, step_sl, log_state],
                  [review_view, log_state, review_count])
    reject.click(lambda *a: review("rejected", *a), [trace_dd, step_sl, log_state],
                 [review_view, log_state, review_count])
    demo.load(lambda tid, st: render(tid, 0, st), [trace_dd, show_truth], outs)

if __name__ == "__main__":
    demo.launch()
