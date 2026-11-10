# Learning When to Escalate on the Edge

Error-aware deferral for compact agent security monitors under distribution shift.
AML2 final project (UF, fall 2026) and the pilot for the spring 2027 capstone.

A small monitor reads an AI agent's trace up to the next tool call and says *safe* or *hijacked*.
A second small model decides whether to trust that verdict or send the call to a human. Both run on
an Arduino UNO Q. Data: [AgentDrift](https://github.com/Asif-0209/AgentDrift) (CC BY 4.0).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
brew install libomp            # macOS only, needed by LightGBM
pip install -r requirements.txt
bash scripts/get_data.sh       # clones AgentDrift into data/raw/ (pinned commit)
bash scripts/run_week1.sh      # data -> anonymize -> leakage -> decision points -> LR baseline -> splits -> tests
```

## Layout

```
l2e/                  the pipeline (run each step with python -m l2e.<step>)
  config.py           paths and constants
  data.py             load AgentDrift splits
  anonymize.py        world anonymizer + target/amount masker
  leakage.py          world-lookup shortcut, before vs after anonymizing
  tools.py            state-changing tools and the paper's sink verbs
  decision_points.py  one row per tool call, prefix-only view
  labels.py           dangerous / costly monitor-error labels
  features.py         the paper's six surface features (whole trace or prefix)
  baseline_lr.py      six-feature logistic regression (paper sanity check + prefix monitor)
  splits.py           tool-coverage and leave-one-domain-out splits
board/                UNO Q environment check (stand-in models, timing)
scripts/              get_data.sh, run_week1.sh
tests/                pytest
reports/week1/        small markdown/json reports written by each step
```

`data/`, `models/` and `board/models/` are generated and git-ignored.
