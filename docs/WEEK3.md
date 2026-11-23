# Week 3 Checkpoint: Learned Deferral Under Shift

**Period:** Nov 23–29, 2026 · **Dataset:** AgentDrift, commit `014a514` · **Monitor:** the main monitor from week 2
(causal transformer on hashed text + tool roles, 107k parameters)

> **Status:** the code is done and tested on synthetic data. The numbers come from `bash scripts/run_week3.sh`.
> Sections marked *(after the run)* get filled from `reports/week3/03_deferral_eval.md`.

## Summary *(after the run)*

- RQ2, no shift: …
- RQ2, stronger monitor: …
- RQ3, unseen tools: …
- RQ3, unseen domains: …
- Does learning from cross-domain errors help (domain folds vs. task folds)? …
- Does reading the trace help (transformer deferrer vs. LightGBM)? …

## 1. What changed since Deliverable 2

| area | Deliverable 2 (v1) | Week 3 |
|---|---|---|
| deferrer inputs | monitor score, tool seen, tool roles, earlier scores | + text novelty of the call and of the tool result before it |
| deferrer models | LightGBM | LightGBM (5 seeds) and a causal transformer that reads the trace (3 seeds) |
| error labels | out of fold, folds grouped by task | + folds grouped by **domain** on the unseen-domain splits |
| baselines | random, confidence | + novelty, + a stronger monitor with calibrated confidence |
| attack rate | natural (about 18% of calls) | 1%, 2%, 5%, and natural |
| test sets | task-disjoint, held-out web | task-disjoint, 4 tool-coverage levels × 3 seeds, all 5 held-out domains |
| uncertainty | none | 95% intervals over 200 resamples of traces, monitor seeds and deferrer seeds |
| app | P(wrong) and a decision | + up to three plain-language reasons per deferral |

## 2. Text novelty

`l2e/novelty.py`. For each decision point, two numbers from the monitor's own hashed text vectors (no extra
model):

- `nov_call`: 1 − cosine similarity between the call's text and its nearest neighbour among 4,096 training calls
- `nov_obs`: the same for the tool result just before the call (where an injection sits)

Both are computed **out of fold**: a training row is only compared with the training data of the fold model
that scored it. Otherwise every training row would look familiar and the deferrer could not learn what novelty
means. The bank is 4,096 × 512 values per kind (4 MB in float16), so it fits on the board.

## 3. Domain-grouped error labels

Week 2's out-of-fold labels came from folds grouped by task. Every fold model had seen every domain, so the
deferrer only learned from in-domain mistakes and never from the kind it has to catch on a new domain.

On each leave-one-domain-out split I now also train one fold model per training domain
(`python -m l2e.oof --fold-by domain`). Each one trains on 3 of the 4 training domains and scores the 4th.
Its early stopping and temperature use only its own domains' validation data, as they would at deployment.
The full model, and so the test scores, stay the same as in week 2.

Training material for the deferrer *(after the run, from `reports/week3/03_errors_*_domfold.md`)*:

| split | task-fold errors (missed attacks) | domain-fold errors (missed attacks) |
|---|---|---|
| leave out banking | … | … |
| leave out coding | … | … |
| leave out email | … | … |
| leave out medical | … | … |
| leave out web | … | … |

## 4. Transformer deferrer

`l2e/neural_deferrer.py`. The same causal transformer as the monitor (2 layers, width 64). It reads the event
stream (text, tool, structured features) plus six numbers on every call: the monitor's calibrated logit and
margin, whether the tool is known, and both novelty values. It predicts P(monitor wrong) at each call, using
only the past. Loss: weighted cross-entropy on monitor errors (missed attacks 5×). Early stopping: one
out-of-fold fold held aside. With domain folds that fold is a whole domain.

The risk: about 400 errors in 50,000 calls is little for a sequence model, and it could turn into a second
attack detector. LightGBM stays the default unless the transformer clearly does better.

## 5. Evaluation protocol

`l2e/evaluate_deferral.py`.

- **Policies at the same review budget:** random, least confident, most novel, stronger monitor (least
  confident), LightGBM deferrer (task / domain folds), transformer deferrer (task / domain folds), perfect.
- **Raw vs. calibrated confidence:** temperature scaling divides every logit by one number. That never changes
  which calls are least confident, so the two are one policy.
- **Stronger monitor:** the main transformer and the week 2 LSTM, logits averaged, temperature refit on val.
  It defers its own least confident calls, and its remaining errors are its own. The mpnet monitor from the
  plan is planned for Deliverable 4.
- **Attack rate:** each test set is resampled to 1%, 2% and 5% hijacked calls. Whole traces are kept: traces
  without a hijacked call (benign, hard negatives, failed attacks) are bootstrapped, and traces with one are
  added in random order until the rate is reached.
- **Metric:** missed attacks left per 1,000 calls after a perfect reviewer checks the deferred calls. Also all
  errors left, and the share of reviews spent on calls the monitor had right.
- **Intervals:** middle 95% over 200 resamples. Each resample also cycles through monitor seeds,
  tool-holdout seeds and deferrer seeds. Learned vs. confidence is compared resample by resample (paired).

## 6. Results *(after the run)*

### 6.1 No shift (task-disjoint test)

### 6.2 Unseen tools (coverage 100 → 25%)

### 6.3 Unseen domains

Figure: `results/figures/09_risk_vs_escalation.png`, `results/figures/10_domain_fold_labels.png`.

### 6.4 What the deferrer leans on

From `reports/week3/02_deferrer_v2_training.json` (share of gain, task folds vs. domain folds).

### 6.5 Review burden

## 7. Interface

The replay console now loads deferrer v2 and lists, under each decision, up to three reasons that raised
P(monitor wrong) for that call, for example "the tool `fill_form` was never seen in training" or "the tool
result before this call is unlike training data". The reasons come from the trees' path contributions
(`TreeModel.contributions`, numpy only), so the board can show them too.

## 8. Deviations from the plan

1. **Stronger monitor = transformer + LSTM average**, not the mpnet monitor (Deliverable 4).
2. **Domain-grouped error labels** are new: not in the plan, added because week 2 showed that unseen domains
   are where confidence fails.
3. **A transformer deferrer** next to LightGBM, to test whether reading the trace helps more than numbers
   about the monitor's confidence.
4. **Board timing and the on-board demo** move to week 4, as planned.

## Reproducing these results

```bash
source .venv/bin/activate
bash scripts/run_week3.sh                 # ~1-1.5 h; EXTRA_SEEDS=1 adds monitor seeds 1 and 2 (~2 h more)
```

| step | command | output |
|---|---|---|
| context rows (tool coverage) | `python -m l2e.oof --encoder hash-roles --arch transformer --context-only --splits tool_seed0 ...` | rows with set "context" in the out-of-fold tables |
| domain-fold labels | `python -m l2e.oof ... --fold-by domain --splits lodo_*` | `data/processed/oof/transformer_hash-roles_domfold/`, `reports/week3/03_errors_*` |
| text novelty | `python -m l2e.novelty` | `data/processed/novelty/<run>/` |
| LightGBM deferrer v2 | `python -m l2e.deferrer` | `data/processed/defer_preds/`, `reports/week3/02_deferrer_v2_training.json` |
| transformer deferrer | `python -m l2e.neural_deferrer` | `data/processed/defer_preds/`, `reports/week3/02_neural_deferrer_training.json` |
| evaluation | `python -m l2e.evaluate_deferral` | `reports/week3/03_deferral_eval.md`, figures 09 and 10 |
| tests | `python -m pytest -q` | |
