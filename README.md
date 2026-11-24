# Learning When to Escalate on the Edge

Error-aware deferral for compact AI agent security monitors under distribution shift.

An AI agent with tools (send email, move money, run code) can be hijacked by instructions hidden in the
content it reads. A small **monitor** reads the agent's trace up to each tool call and says *safe* or
*hijacked*. Within a fixed review budget (for example 5% of calls), the system decides which calls to send to a
human instead of trusting the monitor. The project asks which escalation signal works best when the agent
meets new tools and new domains: the monitor's own confidence, a learned **deferrer** trained on the monitor's
past mistakes, or a second monitor. The models are small enough for an Arduino UNO Q, a small edge board.

Current system (Deliverable 3): two small monitors (a transformer and an LSTM) score every call; their averaged
score is the verdict, and the 5% of calls the pair is least sure about go to a human.

AML2 final project (University of Florida, fall 2026), and the pilot for a spring 2027 capstone.

**Author:** Sibi Seenivasan · MS Applied Data Science, University of Florida · GitHub [@sibi-seeni](https://github.com/sibi-seeni)

## Installation

Python 3.10 or newer.

```bash
git clone https://github.com/sibi-seeni/learning-to-escalate.git
cd learning-to-escalate
python3 -m venv .venv && source .venv/bin/activate
brew install libomp            # macOS only, needed by LightGBM
pip install -r requirements.txt
bash scripts/get_data.sh       # downloads AgentDrift into data/raw/ (pinned commit)
bash scripts/run_week1.sh      # anonymize -> leakage check -> decision points -> baseline -> splits -> tests
bash scripts/run_week2.sh      # neural monitors -> ONNX + int8 -> out-of-fold labels -> domain-shift comparison
bash scripts/finalize_monitor.sh  # main monitor (transformer + tool roles): 10 seeds, export, out-of-fold labels
bash scripts/run_week3.sh        # deferrer v2 + transformer deferrer + domain-fold labels -> full evaluation
bash scripts/run_d3_check.sh     # tests + report figures + notebook end to end, log in results/logs/
```

`run_week3.sh` takes about 1–1.5 hours on a laptop CPU; `EXTRA_SEEDS=1` adds two more monitor seeds on the
unseen-domain splits. After that run, `bash scripts/run_ensemble.sh` (~10 min) adds the 3-seed ensemble and
seed-disagreement policies everywhere, and `bash scripts/run_mpnet.sh` (~1 hour) adds the plan's "better
monitor": a bigger transformer on mpnet text, laptop only. The Deliverable 2 deferrer is still there as `python -m l2e.deferrer --v1`.

`WITH_ENCODERS=1 bash scripts/run_week2.sh` also builds the MiniLM and mpnet versions (downloads the models
from Hugging Face once).

## Run the notebooks

```bash
jupyter notebook notebooks/setup.ipynb       # week 1: environment, data, exploration, leakage (about 30 s)
jupyter notebook notebooks/train_eval.ipynb  # train + evaluate the monitor and the deferrer, D3 results (about 2 min)
```

`setup.ipynb` checks the environment, loads the dataset, explores it, and reproduces the week 1 checks.
`train_eval.ipynb` trains the main monitor, evaluates it against the baseline (metrics, confusion matrix, a
sample run), checks the int8 board model, plots the domain-shift results and trains the D2 deferrer. Section 7
(Deliverable 3) reads the week 3 results instead of retraining them: every escalation policy at equal review
budgets, D2 vs D3 side by side, confusion matrices, ROC curves, the transformer + LSTM pair on a sample trace, and
qualitative examples. It needs `run_week1.sh`, `finalize_monitor.sh` and `run_week3.sh` first. The report figures
alone: `python -m l2e.report_figures` (seconds). Figures are saved to `results/figures/`.

## Run the interface

```bash
python ui/app.py               # open http://127.0.0.1:7860
```

Pick a held-out trace (★ marks showcase examples) and step through it with ◀ Previous / Next ▶. For each
proposed tool call the page shows what the monitors can see, three bars (transformer, LSTM and the pair), the
verdict (SAFE / HIJACKED), the decision (TRUST / ASK A HUMAN) and, when it asks a human, why in plain words
("the monitors disagree: transformer 0.56, LSTM 0.01"). "Show the answer" reveals the true label. A reviewer can
approve or reject a call, and the reviewer log counts how many reviews were right.

The app runs the D3 system by default: the transformer + LSTM pair, sending the 5% least sure calls to a human.
It needs `finalize_monitor.sh` and `run_week3.sh`. Other modes: `L2E_APP_MODE=single` (the D2 system: the
transformer with the learned deferrer) and `L2E_APP_MODE=lr` (the week 1 baseline). If the files for a mode are
missing, the app falls back to the next one and says so at the top of the page.

Screenshots: `docs/figures/ui_d3_*.jpg` (D3) and `docs/figures/ui_d2_*.png` (D2).

## Current results (Deliverable 3)

Main monitor on the task-disjoint test set (9,796 tool calls, 1,755 hijacked): AUROC 0.9997, 55 errors, 23
missed attacks, against 1,252 errors for the six-feature baseline. 161 KB in int8.

Which calls should a human review? Missed attacks still let through **per 1,000 calls** when 5% of calls go to a
perfect reviewer (natural attack rate, mean over 200 resamples by whole traces; lower is better):

| test set | no review | D2: transformer, least confident | LightGBM deferrer v2 | transformer deferrer | 3-seed ensemble | big mpnet monitor | **D3: transformer + LSTM pair** |
|---|---|---|---|---|---|---|---|
| task-disjoint | 2.31 | 0.22 | 0.22 | 0.67 | 0.28 | 1.48 | **0.18** |
| 25% of tools seen | 5.44 | 1.78 | 2.05 | 3.26 | 1.57 | 3.56 | **1.46** |
| unseen domains (mean of 5) | 8.49 | 3.46 | 3.50 | 4.75 | 2.94 | 6.95 | **2.71** |

- No learned deferrer beats simply reviewing the calls the monitor is least sure about.
- A second, different monitor does: on unseen domains the transformer + LSTM pair leaves 2.71 [2.05, 3.39]
  missed attacks per 1,000 calls against 3.46 [2.35, 4.49] for D2, and is better in every resample. The app
  now runs this pair.
- Disagreement between seeds or models on its own, domain-fold labels and the 4-layer mpnet monitor don't help.

Details: `docs/WEEK3.md`, `reports/week3/03_deferral_eval.md` (every policy, budget and attack rate),
`reports/week3/04_examples.md`, `results/figures/09`–`13`. Deliverable 2 results: `docs/WEEK2.md`,
`reports/week3/01_deferrer_v1.md`. Run logs: `results/logs/`.

## Known issues

- **One synthetic generator:** every trace comes from the same LLM, and the pair was chosen from about 15
  policies, so the gain may be smaller on real traces. The policy list is frozen; an AgentDojo check is planned.
- **LightGBM + PyTorch on macOS:** using both in one Python process can crash (segfault). The deferrers are
  trained in their own processes (`python -m l2e.deferrer`, `python -m l2e.neural_deferrer`); the evaluation
  and the app need neither (the app runs the trees with numpy).
- **Board timing for the trained models is not done yet** (Deliverable 4). Week 1 measured stand-in models on
  the board; `board/time_monitor.py` is ready. The pair needs both monitors on the board (about 240 KB in int8).
- **A perfect reviewer is assumed** in every deferral result, so the gains are an upper bound.
- **Unseen-domain results vary a lot between training seeds**, so single runs there should not be trusted.
- `run_week3.sh` takes 1–1.5 hours on a laptop CPU, `run_mpnet.sh` about 1 hour more; set `L2E_DEVICE=auto` to
  use a GPU or Apple MPS. MiniLM and mpnet need a one-time Hugging Face download.

## Dataset

[AgentDrift](https://github.com/Asif-0209/AgentDrift) (CC BY 4.0): 12,536 synthetic tool-using agent traces
(71,024 steps) over five domains (banking, coding, email, medical, web). It is text stored as JSON lines, about
32 MB per split. Every step has the agent's thought, tool call, tool result and a label: benign, injection point,
hijacked or failed injection. I use the task-disjoint split (9,081 / 1,733 / 1,722 traces). All names and
companies are synthetic, and I anonymize them anyway, because they leak the label (see `docs/WEEK1.md`).

## Layout

```
data/            raw and processed data (generated, not committed; see data/README.md)
notebooks/       setup.ipynb (week 1), train_eval.ipynb (train + evaluate, D2 and D3 results)
l2e/             source code: anonymizer, decision points, features, baseline, splits,
                 step featurizer, neural monitors, calibration, ONNX export, out-of-fold labels,
                 domain-shift comparison, fold plans, text novelty, learned deferrer (LightGBM),
                 transformer deferrer, deferral evaluation, numpy tree runner,
                 transformer + LSTM pair for the app, report figures
ui/              Gradio trace replay console: two monitors + pair + reasons + reviewer log
board/           Arduino UNO Q: environment check, monitor timing, results
results/         figures/ (from the pipeline and notebook), logs/ (run logs)
reports/         short reports written by each pipeline step (week1/, week2/, week3/)
docs/            WEEK1–3.md checkpoints, D2 and D3 reports, figures and interface screenshots
scripts/         get_data.sh, run_week1.sh, run_week2.sh, finalize_monitor.sh, run_week3.sh,
                 run_ensemble.sh, run_mpnet.sh
tests/           pytest
```

## Status

- Week 1 (done): data pipeline, anonymization, leakage check, decision points, baseline monitor, shift splits, board setup
- Week 2 (done on the laptop; board timing to run): compact neural monitor, int8 export, out-of-fold error
  labels on every shift split, hidden-thoughts ablation, domain-shift comparison over 10 seeds. See `docs/WEEK2.md`.
- Deliverable 2 (Nov 18): learned deferrer v1 and the app wired to it, training/evaluation notebook.
- Week 3 / Deliverable 3 (Nov 25): deferrer v2 with text novelty, a transformer deferrer that reads the trace,
  domain-grouped error labels, 3 monitor seeds, stronger-monitor and mpnet baselines, seed and model
  disagreement, evaluation at 1–5% attack rates with intervals, and the app switched to the transformer + LSTM
  pair with reasons. See `docs/WEEK3.md`.
- Deliverable 4 (Dec 9): both monitors timed on the board, AgentDojo check, final report

## Contact

Sibi Seenivasan · MS Applied Data Science, University of Florida · GitHub [@sibi-seeni](https://github.com/sibi-seeni).
Questions and bug reports: open an issue on this repository.
