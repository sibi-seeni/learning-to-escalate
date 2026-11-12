# Week 1 Checkpoint: Data, Leakage Controls and Baselines

**Period:** Nov 9–15, 2026 · **Dataset:** AgentDrift, commit `014a514` · **Split:** task-disjoint unless stated

## Summary

- The AgentDrift world-identity shortcut is reproduced (86.1% lookup accuracy, matching the paper). After
  anonymization it falls to 55.4%, the same as the 55.0% majority-class rate.
- We define one decision point per tool call, using a prefix-only view. Restricting to state-changing calls
  would leave 13% of attacked traces (37% of partial hijacks) with nothing for the monitor to catch.
- The six-feature logistic regression reproduces the paper's whole-trace result (F1 0.649 vs. 0.647). As a
  prefix monitor it reaches AUROC 0.905, with 366 dangerous and 886 costly errors on the test set.
- Tool-coverage (100/75/50/25%) and leave-one-domain-out splits are built. No held-out tool appears in any
  training prefix.
- The Arduino UNO Q runs ONNX Runtime and LightGBM, with outputs identical to the laptop's. Small stand-in
  models take 0.3–0.45 ms per decision. ONNX Runtime cannot quantize GRU layers to int8, so int8 recurrent
  monitors will use an LSTM.

## 1. Dataset

AgentDrift (CC BY 4.0) contains 12,536 synthetic tool-using agent traces (71,024 steps) over five domains
(banking, coding, email, medical, web) and 103 tools. Each step records the agent's thought, the tool call
and its arguments, the tool result, and a label: benign, injection point, hijacked or failed injection.

| part | traces | steps | attacked | benign | failed attack | hard negative |
|---|---|---|---|---|---|---|
| train | 9,081 | 51,507 | 4,008 | 2,891 | 1,089 | 1,093 |
| val | 1,733 | 9,721 | 753 | 584 | 193 | 203 |
| test | 1,722 | 9,796 | 775 | 525 | 218 | 204 |

The task-disjoint split keeps every task in a single part, so the model is never tested on a task it saw
in training.

## 2. Leakage controls

The dataset's authors report that a lookup on the world object alone (user, company, contacts) predicts the
attacked class with ~86% accuracy, and that 28.9% of injections reuse one fictitious domain. We replace
identities with placeholders that keep only the relation, and we mask repeated attack targets:

| original | placeholder |
|---|---|
| user's name, its parts, usernames (`ethan.kim`) | `<USER>` |
| contact's name | `<CONTACT_MANAGER>`, `<CONTACT_TEAM_LEAD>`, … (by relation) |
| company, company email-domain stem | `<COMPANY>` |
| user / contact email | `<USER_EMAIL>` / `<CONTACT_EMAIL>` |
| other email on a world domain / elsewhere | `<INTERNAL_EMAIL>` / `<EXTERNAL_EMAIL>` |
| URLs, bare domains | `<INTERNAL_URL>` / `<EXTERNAL_URL>`, `<INTERNAL_DOMAIN>` / `<EXTERNAL_DOMAIN>` |
| money amounts (text and numeric args) | `<AMOUNT>` |

Example: `Transferring $1,000 to Emily Chen` → `Transferring <AMOUNT> to <CONTACT_COLLEAGUE>`.

**Result.** World-lookup accuracy on the test set:

| split | raw worlds | anonymized | majority class |
|---|---|---|---|
| stratified | 86.1% | 54.2% | 55.8% |
| task-disjoint | 86.4% | 55.4% | 55.0% |

The world date and contact relations are kept, because on their own they score at or below the majority rate.
A residual audit finds world strings in 2 of 12,536 traces. Both are false positives: "White" from the lab
test "White Blood Cell Count". A test also confirms that the anonymizer's external-email count matches the
dataset's own rule on 400 real traces.

## 3. Decision points and error labels

A decision point is the moment right before a tool call executes. The monitor sees all earlier steps in
full, plus the current call's thought, tool and arguments. It never sees the current call's result. The
target is 1 when the call is labeled hijacked. An injection-point call is 0, because the user requested it
and the poison only arrives in its result.

**Deviation from the plan.** The plan placed decision points only before state-changing calls. Many
hijacked steps, however, are reads (for example `get_patient_record` or `copy_text`) that gather data for a
later leak, and some attacks end without a state-changing hijacked call:

| compliance | reachable, every call | reachable, state-changing only |
|---|---|---|
| full hijack | 3,136 / 3,136 | 3,090 / 3,136 |
| partial hijack | 1,500 / 1,500 | 945 / 1,500 |
| delayed execution | 900 / 900 | 805 / 900 |
| all attacked | 5,536 / 5,536 | 4,840 / 5,536 |

We therefore treat every tool call as a decision point and report the state-changing subset alongside it.
This also gives the deferrer more monitor errors to learn from. Totals: 51,507 / 9,721 / 9,796 decision
points (train / val / test), of which 8,947 / 1,647 / 1,755 are hijacked.

**Error labels.** A *dangerous* error is when the monitor says safe and the call is hijacked. A *costly*
error is when the monitor says hijacked and the call is benign. The deferrer's target is "the monitor is
wrong here", with dangerous errors weighted 5×. From week 2, these labels will come from out-of-fold
predictions.

## 4. Baseline monitor

The baseline is the AgentDrift paper's six-feature logistic regression. The features are step count,
external recipients, an external-email flag, an external-URL flag, sink-tool count and suspicious-keyword
count.

**Reproduction** (whole traces, stratified split, target = attacked trace):

| | precision | recall | F1 | AUROC |
|---|---|---|---|---|
| paper | 0.778 | 0.554 | 0.647 | 0.734 |
| ours | 0.741 | 0.576 | 0.649 | 0.747 |

Per-group flag rates match closely: full hijack 0.93 vs. 0.88, partial 0.08 vs. 0.08, delayed 0.22 vs.
0.23, benign 0.10 vs. 0.11, hard negative 0.10 vs. 0.12. Failed attacks are the exception, at 0.38 vs.
0.19. The paper publishes only 3 of its 26 keywords, and our reconstructed list also fires on poison the
agent resisted.

**Prefix monitor** (task-disjoint, every decision point, threshold 0.5): AUROC 0.905, AUPRC 0.807, F1
0.689. On state-changing calls only, AUROC is 0.895 and F1 0.756. Test errors:

| subset | decision points | errors | dangerous | costly |
|---|---|---|---|---|
| all domains | 9,796 | 1,252 | 366 | 886 |

Email accounts for the largest share of costly errors (426). The prefix model leans mostly on keywords
(coefficient 1.14) and the external-email flag (0.61).

## 5. Shift splits

**Tool coverage (primary).** For each of three seeds, 20% of each domain's tools are held out. Training and
validation keep only the decision points that come before a trace's first held-out tool. Because the
monitor reads prefixes, it never sees a held-out tool. Dropping whole traces instead would discard 60–85%
of the training data. The test sets at each coverage level have equal size and an equal hijacked rate, so
only tool familiarity varies.

| seed | train DPs kept | train hijacked | smallest domain in train | test size per level |
|---|---|---|---|---|
| 0 | 17,062 | 1,800 | email (771) | 2,818 |
| 1 | 26,044 | 3,244 | banking (1,733) | 2,560 |
| 2 | 23,625 | 2,921 | web (3,575) | 2,923 |

Seed 0 holds out `search_inbox`, which most email traces call early, so email is underrepresented there.
That is why we report all three seeds.

**Leave one domain out (secondary).** Each fold trains on the other four domains and tests on every
decision point of the held-out domain, 11,634 (banking) to 14,986 (email).

## 6. Edge feasibility

Stand-in models with random weights, one decision at a time, single thread:

| model | size | UNO Q median (ms) | MacBook Air median (ms) |
|---|---|---|---|
| GRU, hidden 64, 11 steps | 98 KB | 0.30 | 0.021 |
| LSTM, same shape | 131 KB | 0.45 | 0.020 |
| LSTM, int8 | 35 KB | 0.31 | 0.018 |
| LightGBM, 100 trees | 173 KB | 0.41 | 0.011 |
| logistic regression (scikit-learn) | n/a | 1.00 | 0.031 |

The board (Debian 13, 4× Cortex-A53, 3.7 GB RAM, Python 3.13) peaked at 159 MB. All outputs match the
laptop's to within 0.001.

- **int8 helps on the board:** it makes the LSTM 1.4× faster and 3.7× smaller.
- **scikit-learn's per-call overhead dominates the logistic regression**, so the on-board version should be
  a plain dot product.
- **ONNX Runtime's dynamic quantizer does not support GRU.** The "int8" GRU stays float (98 KB), so int8
  recurrent monitors will use an LSTM.

## 7. Deviations from the original plan

1. Every tool call is a decision point (Section 3).
2. Tool shift uses prefix truncation, not trace removal, at 20% held out over three seeds (Section 5).
3. Leave-one-domain-out tests on all parts of the held-out domain.
4. 23 of the 26 baseline keywords are reconstructed (Section 4).
5. The edge table's recurrent int8 rows use an LSTM instead of a GRU (Section 6).

Assumption: the monitor sees the agent's thought for the current call. If a deployment hides thoughts,
they can be removed in `prefix_steps()`.

## 8. Next (week 2)

- Train the compact neural monitor (LSTM and tiny causal transformer) on light hashed features, then on
  MiniLM features.
- Export to ONNX, quantize to int8, and time one real decision on the board, including the step encoder.
- Calibrate all monitors on validation and produce K-fold out-of-fold error labels.
- Fill the error-count table per held-out split.

## Reproducing these results

```bash
python3 -m venv .venv && source .venv/bin/activate
brew install libomp                 # macOS only (LightGBM)
pip install -r requirements.txt
bash scripts/get_data.sh            # AgentDrift at the pinned commit
bash scripts/run_week1.sh           # sections 1-5, writes reports/week1/, runs the tests
```

| step | command | report |
|---|---|---|
| dataset | `python -m l2e.data` | `reports/week1/01_data_summary.md` |
| anonymize + mask | `python -m l2e.anonymize` | `data/processed/anon/` |
| leakage check | `python -m l2e.leakage` | `reports/week1/03_leakage.md` |
| decision points | `python -m l2e.decision_points` | `reports/week1/04_decision_points.md` |
| baseline | `python -m l2e.baseline_lr` | `reports/week1/05_baseline_lr.md` |
| shift splits | `python -m l2e.splits` | `reports/week1/06_splits.md` |
| tests | `python -m pytest -q` | 11 tests |

**Board check (section 6).** Set up the UNO Q once in Arduino App Lab, which enables SSH as user `arduino`.

```bash
# laptop
python board/make_test_models.py
scp -r board arduino@<board-name>.local:~/l2e-board
# board
sudo apt install -y python3-venv python3-pip
python3 -m venv ~/l2e-venv && source ~/l2e-venv/bin/activate
pip install -r ~/l2e-board/requirements-board.txt
python ~/l2e-board/check_board.py   # prints RESULT: PASS, writes results/board_check_<host>.json
```

## References

- AgentDrift dataset and paper: arXiv 2609.06972, github.com/Asif-0209/AgentDrift
- DriftNet: arXiv 2609.10892
- What Standard Evaluations Hide About Detectors of Hijacked LLM Agents: arXiv 2610.41803
