# Learning When to Escalate on the Edge

Error-aware deferral for compact AI agent security monitors under distribution shift.

An AI agent with tools (send email, move money, run code) can be hijacked by instructions hidden in the
content it reads. A small **monitor** reads the agent's trace up to each tool call and says *safe* or
*hijacked*. A second small model, the **deferrer**, predicts when the monitor is likely wrong and sends those
calls to a human, within a fixed review budget. Both run on an Arduino UNO Q, a small edge board.

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
python -m l2e.deferrer           # learned deferrer v1 vs. confidence at equal review budgets
```

`WITH_ENCODERS=1 bash scripts/run_week2.sh` also builds the MiniLM and mpnet versions (downloads the models
from Hugging Face once).

## Run the notebooks

```bash
jupyter notebook notebooks/setup.ipynb       # week 1: environment, data, exploration, leakage (about 30 s)
jupyter notebook notebooks/train_eval.ipynb  # train + evaluate the monitor and the deferrer (about 1 min)
```

`setup.ipynb` checks the environment, loads the dataset, explores it, and reproduces the week 1 checks.
`train_eval.ipynb` trains the main monitor, evaluates it against the baseline (metrics, confusion matrix, a
sample run), checks the int8 board model, plots the domain-shift results, and trains the deferrer. It needs
`run_week1.sh` and `finalize_monitor.sh` first. Figures are saved to `results/figures/`.

## Run the interface (prototype)

```bash
python ui/app.py               # open http://127.0.0.1:7860  (needs run_week1.sh, and run_week2.sh for the neural monitor)
```

Pick a held-out trace and step through it. The page shows what the monitor can see, its score and verdict,
and whether the call is trusted or deferred. Reviewers can approve or reject deferred calls. The monitor is
the main monitor from week 2 (a small transformer on hashed text + tool roles, calibrated); `L2E_MONITOR=lr python ui/app.py`
switches back to the week 1 baseline. The deferrer is the learned deferrer v1 (`python -m l2e.deferrer`); without it the app falls back to a placeholder rule.

## Current results (Deliverable 2)

Task-disjoint test set: 9,796 decision points (tool calls), 1,755 of them hijacked.

| monitor | parameters | AUROC | F1 | errors | missed attacks |
|---|---|---|---|---|---|
| six-feature logistic regression (week 1 baseline) | 7 | 0.905 | 0.689 | 1,252 | 366 |
| main monitor: causal transformer, hashed text + tool roles | 107k | 0.9997 | 0.984 | 55 | 23 |

- The main monitor is 161 KB in int8 (3 of 9,796 verdicts change) and takes about 0.3 ms per decision on a laptop.
- On a domain it never trained on, it makes 160 to 975 errors depending on the domain, and the number
  swings a lot between training seeds (on web: 373 to 1,308). It still ranks calls well (AUROC ≈ 0.99); what
  breaks is where its 0.5 line falls.
- Learned deferrer v1, at a 5% review budget: on the task-disjoint test it catches 49 of 55 errors (sending
  the least confident calls catches 50). On the unseen web domain it catches 38 of 81 missed attacks against 21 for
  confidence, but fewer errors overall (172 vs 307 of 975), because it focuses on missed attacks.

Details: `docs/WEEK2.md`, `reports/week2/`, `reports/week3/01_deferrer_v1.md`, `results/figures/`.

## Known issues

- **LightGBM + PyTorch on macOS:** using both in one Python process can crash (segfault). The deferrer is
  trained in its own process (`python -m l2e.deferrer`); the app and board run its trees with numpy.
- **Board timing for the trained models is not done yet** (the board is away until late November). Week 1 measured
  stand-in models on it; `board/time_monitor.py` is ready.
- **Deferrer v1 is a first version:** one seed, the dataset's natural attack rate (about 18% of calls), no
  confidence intervals, and a perfect reviewer is assumed. Week 3 adds 1–5% prevalence, several seeds, intervals
  and a text-novelty signal.
- Results on unseen domains vary a lot between training seeds, so single runs there should not be trusted.
- MiniLM and mpnet need a one-time Hugging Face download; MiniLM was tested and not adopted.

## Dataset

[AgentDrift](https://github.com/Asif-0209/AgentDrift) (CC BY 4.0): 12,536 synthetic tool-using agent traces
(71,024 steps) over five domains (banking, coding, email, medical, web). It is text stored as JSON lines, about
32 MB per split. Every step has the agent's thought, tool call, tool result and a label: benign, injection point,
hijacked or failed injection. We use the task-disjoint split (9,081 / 1,733 / 1,722 traces). All names and
companies are synthetic, and we anonymize them anyway, because they leak the label (see `docs/WEEK1.md`).

## Layout

```
data/            raw and processed data (generated, not committed; see data/README.md)
notebooks/       setup.ipynb (week 1), train_eval.ipynb (train + evaluate monitor and deferrer)
l2e/             source code: anonymizer, decision points, features, baseline, splits,
                 step featurizer, neural monitors, calibration, ONNX export, out-of-fold labels,
                 domain-shift comparison, learned deferrer, numpy tree runner
ui/              Gradio trace replay console: monitor + learned deferrer + reviewer buttons
board/           Arduino UNO Q: environment check, monitor timing, results
results/         figures from the notebook
reports/         short reports written by each pipeline step (week1/, week2/, week3/)
docs/            WEEK1.md and WEEK2.md checkpoints, D1.2 and D2 reports, figures and interface screenshots
scripts/         get_data.sh, run_week1.sh, run_week2.sh, finalize_monitor.sh
tests/           pytest
```

## Status

- Week 1 (done): data pipeline, anonymization, leakage check, decision points, baseline monitor, shift splits, board setup
- Week 2 (done on the laptop; board timing to run): compact neural monitor, int8 export, out-of-fold error
  labels on every shift split, hidden-thoughts ablation, domain-shift comparison over 10 seeds. See `docs/WEEK2.md`.
- Deliverable 2 (Nov 18): learned deferrer v1 and the app wired to it, training/evaluation notebook.
- Week 3: stronger deferrer (novelty signals, 1–5% prevalence, seeds and intervals), shift experiments
- Week 4: on-board demo, final report
