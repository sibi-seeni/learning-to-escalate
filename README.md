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
```

## Run the notebook

```bash
jupyter notebook notebooks/setup.ipynb       # then Run All, about 30 seconds
```

It checks the environment, loads the dataset, explores it, and reproduces the week 1 checks. Figures are
saved to `results/figures/`.

## Run the interface (prototype)

```bash
python ui/app.py               # open http://127.0.0.1:7860  (needs run_week1.sh first)
```

Pick a held-out trace and step through it. The page shows what the monitor can see, its score and verdict,
and whether the call is trusted or deferred. Reviewers can approve or reject deferred calls. For now the
monitor is the logistic regression baseline, and the deferrer is a placeholder rule.

## Dataset

[AgentDrift](https://github.com/Asif-0209/AgentDrift) (CC BY 4.0): 12,536 synthetic tool-using agent traces
(71,024 steps) over five domains (banking, coding, email, medical, web). It is text stored as JSON lines, about
32 MB per split. Every step has the agent's thought, tool call, tool result and a label: benign, injection point,
hijacked or failed injection. We use the task-disjoint split (9,081 / 1,733 / 1,722 traces). All names and
companies are synthetic, and we anonymize them anyway, because they leak the label (see `docs/WEEK1.md`).

## Layout

```
data/            raw and processed data (generated, not committed; see data/README.md)
notebooks/       setup.ipynb: environment check, data loading, exploration
l2e/             source code: anonymizer, decision points, features, baseline, splits
ui/              Gradio trace replay console (prototype)
board/           Arduino UNO Q environment check and results
results/         figures from the notebook
reports/week1/   short reports written by each pipeline step
docs/            WEEK1.md research checkpoint, blueprint report, figures
scripts/         get_data.sh, run_week1.sh
tests/           pytest
```

## Status

- Week 1 (done): data pipeline, anonymization, leakage check, decision points, baseline monitor, shift splits, board setup
- Week 2: compact neural monitor (LSTM / tiny transformer), int8 export, timing on the board
- Week 3: learned deferrer and shift experiments
- Week 4: on-board demo, final report
