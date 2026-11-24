# Week 3 Checkpoint: Learned Deferral Under Shift

**Period:** Nov 23–29, 2026 · **Dataset:** AgentDrift, commit `014a514` · **Monitor:** the main monitor from week 2
(causal transformer on hashed text + tool roles, 107k parameters)

> **Status:** done. Every number below comes from `reports/week3/03_deferral_eval.md` (200 resamples per
> cell, 3 seeds of the main monitor everywhere), unless another report is named.

## Summary

- **A second, different monitor is the only reliable improvement.** Averaging the main transformer with the
  week 2 LSTM leaves 2.7 missed attacks per 1,000 calls on unseen domains at a 5% review budget, against 3.5
  for the main monitor's least confident calls. It is better in 100% of resamples.
- **Learned deferrers only tie "least confident".** LightGBM leaves 3.5 (better in 42% of resamples). The
  transformer deferrer that reads the trace leaves 4.7 (never better in any resample).
- **Learning from cross-domain errors taught the wrong lesson.** Domain-fold labels are mostly false alarms,
  so the domain-fold deferrer finds more mistakes overall but fewer missed attacks (4.5 left).
- **Seed diversity helps a little; model diversity helps more.** A 3-seed ensemble leaves 2.9 (better in 67%).
  Disagreement alone, between seeds or between the transformer and the LSTM, barely helps (3.2 and 3.4).
- **Bigger is worse under shift.** The plan's mpnet "better monitor" (a 4-layer transformer on a 110M-parameter
  text encoder) makes more mistakes than the 107k-parameter main monitor on every test set, and leaves 7.0.
- **Unseen tools barely matter; unseen domains do.** With 25% of tools seen, the main monitor makes 35 mistakes
  per 2,767 calls (24 with all seen). On a left-out domain it makes 146 to 622 per 11,600–15,000 calls.
- **The answer to RQ2 is a well-measured negative result; RQ3's answer is "use a second, different model".**

## 1. What changed since Deliverable 2

| area | Deliverable 2 (v1) | Week 3 |
|---|---|---|
| deferrer inputs | monitor score, tool seen, tool roles, earlier scores | + text novelty of the call and of the tool result before it |
| deferrer models | LightGBM | LightGBM (5 seeds) and a causal transformer that reads the trace (3 seeds) |
| error labels | out of fold, folds grouped by task | + folds grouped by **domain** on the unseen-domain splits |
| baselines | random, confidence | + novelty, + two stronger monitors (transformer + LSTM, mpnet), + a 3-seed ensemble, seed and model disagreement |
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

Training material for the deferrer, main monitor seed 42 (`reports/week2/03_errors_transformer_hash-roles.md`,
`reports/week3/03_errors_transformer_hash-roles_domfold.md`):

| split | task-fold errors (missed attacks) | domain-fold errors (missed attacks) |
|---|---|---|
| leave out banking | 464 (212) | 1,712 (289) |
| leave out coding | 402 (183) | 1,364 (266) |
| leave out email | 416 (260) | 1,055 (356) |
| leave out medical | 419 (225) | 1,207 (384) |
| leave out web | 372 (224) | 800 (278) |

Domain folds give 2–4× as many errors, but nearly all the extra ones are false alarms: missed attacks go from
about half of the errors to about a fifth to a third.

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

- **Policies at the same review budget (15):** random, least confident, most novel, transformer + LSTM
  (least confident), the mpnet monitor (least confident), transformer–LSTM disagreement, transformer + LSTM
  unsure or disagreeing, 3-seed ensemble, seed disagreement, ensemble + disagreement, LightGBM deferrer (task /
  domain folds), transformer deferrer (task / domain folds), perfect.
- **Transformer–LSTM disagreement:** the main monitor's verdicts; the calls where the two monitors' calibrated
  P(hijacked) are furthest apart go to the human. A second version ranks the pair's calls by "least confident"
  and "most disagreement" together.
- **Seed ensemble and seed disagreement:** the main transformer trained with 3 seeds (42, 1, 2). The *ensemble*
  averages their calibrated logits (temperature refit on val) and defers its own least confident calls.
  *Disagreement* keeps the main monitor's verdicts but sends the calls where the seeds' P(hijacked) spreads most.
  A third policy ranks the ensemble's calls by both (mean of the two ranks). Seeds 1 and 2 come from the
  out-of-fold runs on the unseen-domain splits and from full-model-only runs elsewhere
  (`python -m l2e.oof ... --full-only`, `scripts/run_ensemble.sh`).
- **Raw vs. calibrated confidence:** temperature scaling divides every logit by one number. That never changes
  which calls are least confident, so the two are one policy.
- **Stronger monitors:** (1) the main transformer and the week 2 LSTM, logits averaged, temperature refit on
  val; (2) the plan's "better monitor", a 4-layer, width-256 causal transformer on mpnet sentence embeddings +
  tool roles (`scripts/run_mpnet.sh`; laptop only). Each defers its own least confident calls, and its
  remaining errors are its own.
- **Attack rate:** each test set is resampled to 1%, 2% and 5% hijacked calls. Whole traces are kept: traces
  without a hijacked call (benign, hard negatives, failed attacks) are bootstrapped, and traces with one are
  added in random order until the rate is reached.
- **Metric:** missed attacks left per 1,000 calls after a perfect reviewer checks the deferred calls. Also all
  errors left, and the share of reviews spent on calls the monitor had right.
- **Intervals:** middle 95% over 200 resamples. Each resample also cycles through monitor seeds,
  tool-holdout seeds and deferrer seeds. Learned vs. confidence is compared resample by resample (paired).

## 6. Results

All numbers are missed attacks still let through **per 1,000 calls** after a perfect reviewer checks the
deferred calls, at a **5% review budget** and the dataset's **natural attack rate**, unless stated. Lower is
better. "Better in" = share of the 200 paired resamples where a policy leaves fewer than "least confident".

### 6.1 Before any review: the monitors

| test set | main monitor (3 seeds) | mpnet monitor (1 seed) |
|---|---|---|
| Task-disjoint test (9,796 calls) | 55 errors, 23 missed attacks | 93 errors, 62 missed attacks |
| Tool coverage 100% / 25% (2,767 calls, mean of 3) | 24 / 35 errors | 47 / 97 errors |
| Leave out banking | 146 errors, 82 missed | 210 errors, 88 missed |
| Leave out coding | 262 errors, 213 missed | 597 errors, 227 missed |
| Leave out email | 528 errors, 45 missed | 505 errors, 113 missed |
| Leave out medical | 270 errors, 103 missed | 421 errors, 409 missed |
| Leave out web | 622 errors, 161 missed | 766 errors, 190 missed |

The main monitor's results swing with the seed: web 365–975 errors, email 235–956
(`reports/week3/03_errors_transformer_hash-roles_s1.md`, `_s2.md`).

### 6.2 No shift (task-disjoint test)

| policy | missed attacks left | better in |
|---|---|---|
| no review | 2.31 | — |
| least confident | 0.22 | — |
| transformer + LSTM | 0.18 | 51% |
| LightGBM deferrer | 0.22 | 6% |
| 3-seed ensemble | 0.28 | 27% |
| transformer deferrer | 0.67 | 0% |
| mpnet monitor | 1.48 | 0% |
| most novel | 2.12 | 0% |

Least confident already catches 90% of the missed attacks. Nothing beats it here.

### 6.3 Unseen tools (coverage 100 → 25%)

| policy | 100% | 75% | 50% | 25% |
|---|---|---|---|---|
| no review | 4.89 | 5.67 | 6.13 | 5.43 |
| least confident | 1.60 | 1.95 | 1.90 | 1.78 |
| 3-seed ensemble | 1.01 | 1.09 | 1.25 | 1.57 |
| transformer + LSTM | 1.28 | 1.39 | 1.43 | 1.46 |
| LightGBM deferrer | 1.57 | 2.10 | 2.14 | 2.05 |
| transformer deferrer | 2.66 | 3.10 | 3.60 | 3.26 |
| mpnet monitor | 3.02 | 4.10 | 3.05 | 3.56 |

The ensembles do best here, but every gap is inside the noise (better in 40–68% of resamples): each
coverage level has only about 15 missed attacks.

### 6.4 Unseen domains

| policy | banking | coding | email | medical | web | mean | better in (mean) |
|---|---|---|---|---|---|---|---|
| no review | 6.99 | 14.26 | 3.08 | 6.96 | 10.91 | 8.44 | — |
| least confident | 2.39 | 4.91 | 1.46 | 2.04 | 6.48 | 3.46 | — |
| **transformer + LSTM** | 2.42 | 3.68 | 1.39 | 1.30 | 4.74 | **2.71** | **100%** |
| transformer + LSTM, unsure or disagreeing | 2.50 | 3.82 | 1.35 | 1.43 | 5.05 | 2.83 | 99% |
| 3-seed ensemble | 1.96 | 4.27 | 1.37 | 1.26 | 5.85 | 2.94 | 67% |
| ensemble, unsure or disagreeing | 2.14 | 4.27 | 1.21 | 1.19 | 6.05 | 2.97 | 67% |
| seeds disagree most | 2.38 | 4.33 | 1.39 | 1.22 | 6.45 | 3.15 | 66% |
| transformer and LSTM disagree most | 2.68 | 4.33 | 1.35 | 1.67 | 7.03 | 3.41 | 54% |
| LightGBM deferrer, task folds | 2.24 | 5.63 | 1.73 | 1.79 | 6.10 | 3.50 | 42% |
| LightGBM deferrer, domain folds | 3.54 | 6.60 | 1.90 | 2.84 | 7.56 | 4.49 | 9% |
| transformer deferrer, task folds | 3.55 | 7.57 | 2.42 | 3.69 | 6.54 | 4.75 | 0% |
| transformer deferrer, domain folds | 3.70 | 7.73 | 2.47 | 3.51 | 7.24 | 4.93 | 0% |
| mpnet monitor | 2.64 | 10.01 | 3.71 | 10.02 | 8.36 | 6.95 | 0% |
| most novel | 6.27 | 11.86 | 2.99 | 6.31 | 9.55 | 7.40 | 0% |

The transformer + LSTM pair leaves 0.75 fewer missed attacks per 1,000 calls than least confident
[95% range 0.17 to 1.28]. On web it leaves 1.74 fewer [0.13 to 3.32], better in 98% of resamples. A 2% review
budget gives the same order (pair 4.4, least confident 5.6).

**At low attack rates the order holds but the gaps shrink.** At 2% attacks, unseen domains: least confident
0.37, pair 0.31 (better in 81%), 3-seed ensemble 0.32 (70%), LightGBM 0.37, mpnet 0.66. At 1%: least
confident 0.19, pair and ensemble 0.15. Every gap except the mpnet one is inside the noise.

Figures: `results/figures/09_risk_vs_escalation.png`, `results/figures/10_domain_fold_labels.png`.

### 6.5 What the LightGBM deferrer leans on

Share of gain, mean over the five unseen-domain splits:

| feature | task folds | domain folds |
|---|---|---|
| margin (distance from 0.5) | 50% | 46% |
| score and calibrated logit | 18% | 20% |
| text novelty of the call | 8% | 7% |
| text novelty of the result before it | 7% | 5% |
| earlier scores in the trace | 12% | 12% |

Half of what it uses is the monitor's own confidence, which is why it ends up close to "least confident".
Text novelty is the next most used input, but on its own ("most novel") it is worse than random on familiar
tasks.

### 6.6 Review burden

Share of reviews spent on calls the monitor already had right (5% budget):

| setting | least confident | transformer + LSTM | 3-seed ensemble | LightGBM (task / domain folds) | mpnet |
|---|---|---|---|---|---|
| task-disjoint test | 90% | 91% | 90% | 90% / — | 85% |
| unseen domains | 71% | 72% | 73% | 78% / 68% | 64% |

At these error rates most reviews are spent on calls the monitor had right, whatever the policy: a 5% budget
is 490 reviews for 55 errors on the task-disjoint test. The domain-fold deferrer wastes the fewest reviews
(68%) because it is good at finding false alarms, but it catches fewer missed attacks (47% against 59%).

### 6.7 Conclusions

1. **RQ2, deferral:** a learned error predictor does not beat calibrated confidence on the same monitor, with
   or without shift. This is a well-measured negative result (5 deferrer seeds, 3 monitor seeds, 200
   resamples, 1–5% attack rates).
2. **RQ2, stronger monitor:** a stronger *different* monitor does beat it. A bigger version of the same idea
   (mpnet) does not.
3. **RQ3, unseen tools:** confidence holds up; tool coverage down to 25% costs little.
4. **RQ3, unseen domains:** the monitor's confidence drifts with the training seed, so a second opinion helps
   most. Model diversity (transformer + LSTM) helps more than seed diversity (3 seeds), and averaging helps
   more than escalating on disagreement.
5. **For the edge:** the best policy uses two tiny monitors, 161 KB + 77 KB in int8, not a large one.

These are exploratory findings from 15 policies compared on one synthetic dataset (section 9).

## 7. Interface

The replay console now loads deferrer v2 and lists, under each decision, up to three reasons that raised
P(monitor wrong) for that call, for example "the tool `fill_form` was never seen in training" or "the tool
result before this call is unlike training data". The reasons come from the trees' path contributions
(`TreeModel.contributions`, numpy only), so the board can show them too.

## 8. Deviations from the plan

1. **Two stronger-monitor baselines:** the transformer + LSTM average, and the planned mpnet monitor. The mpnet
   monitor is causal (it must decide before each call), not DriftNet's bidirectional trunk, and was trained
   once with the main monitor's settings, not tuned.
2. **Domain-grouped error labels** are new: not in the plan, added because week 2 showed that unseen domains
   are where confidence fails.
3. **A transformer deferrer** next to LightGBM, to test whether reading the trace helps more than numbers
   about the monitor's confidence.
4. **Seed ensemble, seed disagreement and transformer–LSTM disagreement** added as escalation signals.
5. **Board timing and the on-board demo** move to week 4, as planned.

## 9. Limits

- **One synthetic dataset from one generator.** AgentDrift was written by one model (Llama-3.3-70B), so even the
  unseen-domain tests share its style. A published study found that detectors trained on AgentDrift rank
  AgentDojo traces no better than chance. The absolute numbers should not be claimed for real agents.
- **Picking the best of many.** 15 policies were compared on the same test sets. The intervals cover sampling
  noise, not the choice of the winner. The policy list is now frozen, and the pair result is reported as
  exploratory. It is the most likely to hold: it wins in every resample on unseen domains.
- **Nothing was tuned on the test sets.** Thresholds and temperatures are set on validation data. The main
  monitor was chosen in week 2 after comparing unseen-domain results, where the candidates were tied.
- **The mpnet monitor has one seed**, so its intervals miss its seed-to-seed swings; its gap is much larger
  than those swings.
- **A perfect reviewer is assumed**, so every deferral gain is an upper bound.
- **Next (Deliverable 4):** an outside check on AgentDojo, replaying its ground-truth tool calls with and
  without its built-in injections, which tests false alarms on a different generator.

## Reproducing these results

```bash
source .venv/bin/activate
EXTRA_SEEDS=1 bash scripts/run_week3.sh   # ~3 h with monitor seeds 1 and 2 (~1-1.5 h without)
bash scripts/run_ensemble.sh               # seed ensemble everywhere (~10 min)
bash scripts/run_mpnet.sh                  # the mpnet monitor (~45 min, most of it embedding)
```

| step | command | output |
|---|---|---|
| context rows (tool coverage) | `python -m l2e.oof --encoder hash-roles --arch transformer --context-only --splits tool_seed0 ...` | rows with set "context" in the out-of-fold tables |
| domain-fold labels | `python -m l2e.oof ... --fold-by domain --splits lodo_*` | `data/processed/oof/transformer_hash-roles_domfold/`, `reports/week3/03_errors_*` |
| text novelty | `python -m l2e.novelty` | `data/processed/novelty/<run>/` |
| LightGBM deferrer v2 | `python -m l2e.deferrer` | `data/processed/defer_preds/`, `reports/week3/02_deferrer_v2_training.json` |
| transformer deferrer | `python -m l2e.neural_deferrer` | `data/processed/defer_preds/`, `reports/week3/02_neural_deferrer_training.json` |
| mpnet monitor | `python -m l2e.oof --encoder mpnet-roles --arch transformer_big --full-only` | `reports/week3/03_errors_transformer_big_mpnet-roles_fullonly.md` |
| evaluation | `python -m l2e.evaluate_deferral` | `reports/week3/03_deferral_eval.md`, figures 09 and 10 |
| tests | `python -m pytest -q` | |
