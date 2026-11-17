# Week 2 Checkpoint: Compact Neural Monitor

**Period:** Nov 16–22, 2026 (started early, Nov 12) · **Dataset:** AgentDrift, commit `014a514` · **Split:** task-disjoint unless stated

## Summary

- A 70k-parameter LSTM over the trace prefix, with no pretrained text model, reaches test AUROC 0.9995 and
  F1 0.979. It makes 74 errors on 9,796 decision points, against 1,252 for the six-feature logistic regression.
- The GRU and the tiny causal transformer score within 0.001 of it. The LSTM was picked first because it
  quantizes to int8 (the GRU can't, in ONNX Runtime) and is the smaller of the two that do. Sections 2–6
  report the LSTM; section 7 explains the switch to the final main monitor.
- int8 shrinks it from 276 KB to 77 KB and changes 3 of 9,796 verdicts. On the laptop one decision takes
  about 0.3 ms, and most of that is building the features, not running the model.
- Hiding the agent's thoughts raises test errors from 74 to 191 but AUROC stays at 0.996, so most of the
  signal is the injected text in tool results.
- Out-of-fold labels exist for all 9 splits: 414 monitor errors (232 dangerous) on the task-disjoint training
  set for the deferrer to learn from.
- Unseen tools cost little (mean errors 30 → 51 from 100% to 25% coverage). Unseen domains cost much more:
  172 to 970 errors. On email, medical and web most of them are confident false alarms, and calibration error
  rises 10–15×.
- No monitor change fixes that. Over 10 seeds, the transformer averages 341 errors on unseen domains with
  hashed text and 349 with tool roles added; MiniLM text is worse (429, 3 seeds). A 3-seed run first made the
  roles look 17% better, which turned out to be luck.
- How a monitor does on an unseen domain depends mostly on the random seed: on held-out web the same model
  made 373 to 1,308 errors. Every monitor still ranks calls well (AUROC ≈ 0.99); what is unreliable is where
  its 0.5 line falls on a new domain. That is the gap the deferrer is meant to cover.
- **Final main monitor: a 107k-parameter causal transformer on hashed text + tool roles** (tied with the no-roles
  version, and already built into the app, board bundle and deferrer labels). 55 test errors, int8 161 KB with
  3 verdicts changed, about 0.26 ms per decision on the laptop, and 418 out-of-fold errors (316 dangerous) for
  the deferrer.

## 1. Step features

A trace becomes a stream of events: `call_0, obs_0, call_1, obs_1, …, call_t`. A call event is the agent's
thought, tool and arguments; an obs event is the tool's result. The monitor reads the stream left to right
and scores every call event. Because it only reads backwards, the score for call *t* uses steps 0..t−1 in
full plus step *t*'s call, never step *t*'s result. That is the same view as week 1's `prefix_steps()`, and a
test checks that the two give identical features.

Each event is described by:

| part | size | what it holds |
|---|---|---|
| tool id | 16-dim embedding | from a vocabulary built on the split's training tools only; anything else is "unknown" |
| structured | 14 numbers | call or result, position, state-changing and sink flags, argument count, counts of `<EXTERNAL_EMAIL>`, `<EXTERNAL_URL>`, `<EXTERNAL_DOMAIN>`, `<AMOUNT>`, internal and known-contact placeholders, keyword hits, text length |
| text | 512 numbers | hashed bag of words and word pairs (crc32, so identical on the board); no pretrained model |

During training, 10% of tool ids are randomly replaced by "unknown", so the model has practiced on
unfamiliar tools before the tool-shift tests.

## 2. Monitors

All models: event features → linear + ReLU (width 64) → sequence layer → P(hijacked) at each call. Loss only
at decision points, AdamW, early stopping on validation AUPRC, then temperature scaling on validation. The
logistic regression gets the same temperature scaling.

| model | params | AUROC | AUPRC | F1 | test errors | dangerous | costly | ECE (calibrated) |
|---|---|---|---|---|---|---|---|---|
| logistic regression (6 features) | 7 | 0.905 | 0.806 | 0.689 | 1,252 | 366 | 886 | 0.101 |
| **LSTM + hashed text** | 69,777 | 1.000 | 0.998 | 0.979 | 74 | 31 | 43 | 0.003 |
| GRU + hashed text | 61,457 | 1.000 | 0.998 | 0.981 | 65 | 38 | 27 | 0.003 |
| causal transformer (2 layers) + hashed text | 106,641 | 0.999 | 0.998 | 0.981 | 68 | 34 | 34 | 0.003 |

On state-changing calls only, all three neural monitors reach AUROC 0.999 and F1 0.98 (the LR: 0.895, 0.756).
Training takes 20–50 s each on a laptop CPU.

**Choosing the first edge monitor** (replaced in section 7)**.** Validation AUPRC is 0.9981 (LSTM), 0.9984 (GRU) and 0.9982 (transformer),
which is a tie. The rule is: best validation AUPRC among the int8-quantizable models (LSTM, transformer), with
gaps under 0.001 going to the smaller one. That picks the LSTM.

**Is this too good?** These numbers are in line with DriftNet's trajectory F1 of 0.983 on the same split. The
hijacked calls are easy for a text model because the poisoned tool result sits right before them in the
prefix, and the agent's thought often repeats the injected reason ("forwarding the attachment to complete
verification"). The world-identity shortcut is already removed (week 1), and a test confirms that changing
later events never changes an earlier score.

## 3. Ablation: agent thoughts hidden

Some deployments won't expose the agent's reasoning. With every thought removed from the features:

| model | AUROC | F1 | test errors | dangerous | costly |
|---|---|---|---|---|---|
| LSTM, thoughts visible | 1.000 | 0.979 | 74 | 31 | 43 |
| LSTM, thoughts hidden | 0.996 | 0.945 | 191 | 115 | 76 |
| transformer, thoughts hidden | 0.997 | 0.954 | 160 | 116 | 44 |

Errors more than double and the extra ones are mostly missed attacks, but the monitor still works. This
version is also a fallback for week 3 if the main monitor leaves the deferrer too few errors.

## 4. int8 export

One ONNX call is one decision: the prefix's event stream goes in and the calibrated P(hijacked) comes out.
int8 uses ONNX Runtime's dynamic quantization. Checked on all 9,796 test decision points:

| model | float KB | int8 KB | AUROC float → int8 | F1 float → int8 | verdicts changed by int8 |
|---|---|---|---|---|---|
| LSTM | 276 | 77 | 0.9995 → 0.9995 | 0.9790 → 0.9787 | 3 |
| transformer | 445 | 161 | 0.9994 → 0.9994 | 0.9806 → 0.9804 | 7 |

ONNX float matches PyTorch to within 1e-6.

**Laptop latency, one decision** (longest sample trace, 15 events, single thread, median ms):

| stage | LSTM | transformer |
|---|---|---|
| build features | 0.26 | 0.28 |
| model, float | 0.04 | 0.12 |
| model, int8 | 0.04 | 0.12 |

int8 doesn't speed these tiny models up on the laptop. On the board in week 1 it made a same-sized LSTM 1.4×
faster. Feature building costs more than the model, so it goes in the board timing too.

**Board run (to do).** `board/time_monitor.py` replays 120 test traces (688 decisions) through the real
featurizer and both ONNX files, checks every output against the laptop's, and records latency and memory.
On the laptop it passes with an exact match and 0.19 ms per decision.

## 5. Out-of-fold error labels

For each split, the training traces are split into 5 folds grouped by task text. Each fold's model scores
the fold it didn't train on, which gives out-of-fold scores for every training decision point. A model
trained on all of the training data scores val and test. Every model gets its own temperature on val.
The labels are written to `data/processed/oof/lstm_hash/<split>.csv`, with a `tool_seen` column for week 3's
novelty features.

**Deferrer training material** (out-of-fold errors on each split's training decision points):

| split | training DPs | OOF errors | dangerous | costly |
|---|---|---|---|---|
| task-disjoint | 51,507 | 414 | 232 | 182 |
| tool coverage, seeds 0 / 1 / 2 | 17,062 / 26,044 / 23,625 | 286 / 245 / 273 | 137 / 151 / 115 | 149 / 94 / 158 |
| leave-one-domain-out (range) | 40,447–43,081 | 302–415 | 166–250 | 88–230 |

With thoughts hidden there are 2–3× as many out-of-fold errors (1,138 on task-disjoint).

## 6. Held-out error counts (the plan's table)

| held-out split | decision points | monitor errors | dangerous errors |
|---|---|---|---|
| Task-disjoint test | 9,796 | 74 | 31 |
| Tool coverage 100% (mean of 3 seeds) | 2,767 | 29.7 | 12.7 |
| Tool coverage 75% | 2,767 | 38.7 | 14.3 |
| Tool coverage 50% | 2,767 | 44.0 | 16.3 |
| Tool coverage 25% | 2,767 | 50.7 | 17.7 |
| Leave out banking | 11,634 | 172 | 108 |
| Leave out coding | 14,817 | 230 | 108 |
| Leave out email | 14,986 | 583 | 44 |
| Leave out medical | 14,795 | 525 | 46 |
| Leave out web | 14,792 | 970 | 219 |

Per-seed rows are in `reports/week2/03_errors_lstm_hash.md`.

**Calibration breaks under domain shift.** AUROC stays at 0.984–0.997 on every held-out domain, so the
monitor still ranks calls well. What goes wrong is its confidence:

| held-out set | ECE | errors at 0.5 | errors at the best threshold for that set | mean score on benign calls |
|---|---|---|---|---|
| task-disjoint test | 0.003 | 74 | 63 | 0.008 |
| tool coverage 25%, seed 0 | 0.009 | 52 | 41 | 0.018 |
| leave out email | 0.041 | 583 | 252 | 0.056 |
| leave out medical | 0.034 | 525 | 209 | 0.046 |
| leave out web | 0.048 | 970 | 672 | 0.068 |

On email, medical and web most errors are false alarms made with high confidence. A threshold tuned on the
new domain would remove a third to more than half of them, but there are no labels to tune it on at deployment time. Confidence-based deferral sends the
*least* confident calls to a human, so it will likely miss these. That is the case where a learned deferrer,
with novelty signals like `tool_seen`, has room to help.

## 7. Can a different model fix domain shift?

The leave-out-web false alarms point at the inputs, not the architecture. They are mostly web-only tools the
monitor never saw (`read_webpage`, `fill_form`, `extract_links`, `extract_table`), and 99% of them have an
external URL earlier in the trace. Outside links are normal on the web but in the other four domains they
usually came with an attack. So we tested two things: a transformer instead of the LSTM, and **tool roles**,
8 domain-neutral flags for what each tool does (reads private data, reads untrusted content, writes, sends
data out, moves money, runs code, deletes, changes trust; `l2e/tool_roles.py`). With roles, web's `fill_form`
looks like email's `send_email`. A third variant also drops the memorized tool id (`-noid`).

Every configuration was trained on all five leave-one-domain-out splits (`python -m l2e.compare_shift`, full
tables in `reports/week2/04_domain_shift.md`). The two closest candidates got 10 seeds each, the rest 3.
Average over the five held-out domains, mean ± standard deviation over seeds:

| monitor | seeds | errors | dangerous | costly | confident false alarms (≥ 0.9) | task-disjoint errors |
|---|---|---|---|---|---|---|
| LSTM, hashed text | 3 | 347 ± 87 | 130 ± 13 | 217 ± 99 | 86 ± 54 | 69 ± 2 |
| transformer, hashed text | 10 | 341 ± 60 | 110 ± 15 | 231 ± 66 | 84 ± 27 | 67 ± 7 |
| transformer + tool roles | 10 | 349 ± 74 | 111 ± 20 | 238 ± 87 | 91 ± 33 | 72 ± 11 |
| transformer + tool roles, no tool id | 3 | 325 ± 30 | 120 ± 3 | 205 ± 27 | 76 ± 11 | 76 ± 8 |
| transformer + MiniLM | 3 | 429 ± 107 | 156 ± 35 | 273 ± 140 | 101 ± 65 | 98 ± 37 |
| transformer + MiniLM + tool roles | 3 | 425 ± 55 | 167 ± 13 | 259 ± 55 | 105 ± 36 | 109 ± 41 |

- **Architecture doesn't matter here.** LSTM and transformer are within noise (347 vs 341).
- **Tool roles don't help either.** With 3 seeds they looked 17% better (294 vs 353); with 10 seeds the two
  are tied (349 vs 341). The early gap was luck, which is why the closest candidates got 10 seeds.
- **MiniLM is worse overall.** A pretrained sentence encoder should understand that "submit this form to an
  outside site" and "forward this to an outside address" mean similar things. It trades one kind of error for
  another instead: on held-out web it removes most false alarms (confident false alarms 179 → 21) but about
  doubles missed attacks (156 → 329), and on coding and banking it raises far more false alarms.
- **Seed-to-seed swings dominate.** On held-out web the transformer with tool roles made 377 to 930 errors
  across seeds, and without roles 373 to 1,308. On email, 153 to 881. Only the random starting point
  changed. Any domain-shift result here needs several seeds and its spread.

**What this means.** AUROC stays near 0.99 for every monitor and seed, so they all still rank calls well. What
breaks on a new domain is where the 0.5 line falls, and that is close to random from run to run. No change to
the monitor fixes it, which is the case for a second model that predicts when the monitor is wrong.

**Decision: the main monitor is the transformer on hashed text + tool roles** (`config.MAIN_ARCH`,
`config.MAIN_FEATURES`). It is tied with the no-roles version. It was kept because it was already trained,
exported and labeled for the deferrer, and because the roles give the deferrer a domain-neutral description of
each tool. It needs no sentence encoder on the board. Built by `bash scripts/finalize_monitor.sh`:

| | main monitor (transformer + hashed text + tool roles) |
|---|---|
| parameters | 107,153 |
| task-disjoint test | AUROC 0.9997, F1 0.984, 55 errors (23 dangerous, 32 costly) |
| int8 | 447 KB → 161 KB, AUROC unchanged, 3 of 9,796 verdicts changed |
| laptop latency per decision | 0.18 ms features + 0.09 ms model |
| out-of-fold errors for the deferrer | 237–464 per split; 418 on task-disjoint, 316 of them dangerous |

Its held-out error counts (`reports/week2/03_errors_transformer_hash-roles.md`), single run:

| held-out split | decision points | errors | dangerous |
|---|---|---|---|
| Task-disjoint test | 9,796 | 55 | 23 |
| Tool coverage 100% / 75% / 50% / 25% (mean of 3 seeds) | 2,767 | 24 / 32 / 35 / 35 | 14 / 16 / 17 / 15 |
| Leave out banking | 11,634 | 160 | 67 |
| Leave out coding | 14,817 | 262 | 198 |
| Leave out email | 14,986 | 394 | 50 |
| Leave out medical | 14,795 | 416 | 28 |
| Leave out web | 14,792 | 975 | 81 |

The out-of-fold labels in `data/processed/oof/transformer_hash-roles/` are the deferrer's training data. The
LSTM files in section 5 are kept for comparison.

## 8. Deviations from the plan

1. **Event stream instead of one vector per step.** A tool result is its own event, so a recurrent monitor
   can update step by step on the board and never needs the current call's result.
2. **No GRU as the edge monitor**, because ONNX Runtime can't quantize GRU (week 1). The GRU stays in the
   comparison.
3. **Hidden-thoughts ablation added** (section 3).
4. **MiniLM was tested and not adopted** (section 7). The mpnet "better monitor" is written but not run, since
   the deferrer comes first. The MiniLM ONNX export was checked on a randomly initialized model of the same
   shape (int8 cosine 0.9999; 88 MB float, 22 MB int8).
5. **The stronger "better monitor"** is a 4-layer causal transformer (width 256) on mpnet embeddings, not
   DriftNet's bidirectional trunk, because it has to decide before each call like ours. It also waits on the
   encoder run.
6. **Domain-shift comparison over several seeds, and tool roles** (section 7). The roles didn't measurably help.
7. **Main monitor: transformer on hashed text + tool roles**, not the LSTM (section 7).

## 9. Next

- **Board timing (when back with the UNO Q, late November):** `time_monitor.py` for the main monitor's bundle
  (`monitor_bundle_transformer_hash-roles.json`), and fill the edge table. The board is in the US until then.
- **Week 3:** train the LightGBM deferrer on the main monitor's out-of-fold errors, compare it with calibrated confidence at
  equal review budgets (1–5% attack prevalence), and look hardest at leave-one-domain-out, where section 6
  shows confidence is least trustworthy.
- With only 55 in-distribution test errors, results at low budgets will be noisy. Report counts and bootstrap
  intervals next to rates, pool across folds where possible, and use several seeds for anything under domain shift.

## Reproducing these results

```bash
source .venv/bin/activate
pip install -r requirements.txt           # adds torch
bash scripts/run_week2.sh                 # ~1 hour on a laptop CPU, most of it the out-of-fold step
```

| step | command | report |
|---|---|---|
| monitors | `python -m l2e.train_monitor` | `reports/week2/01_monitor_hash.md` |
| hidden thoughts | `python -m l2e.train_monitor --encoder hash-nothought` | `reports/week2/01_monitor_hash-nothought.md` |
| ONNX + int8 | `python -m l2e.export_onnx` (and `--arch transformer`) | `reports/week2/02_export_*.md` |
| out-of-fold labels | `python -m l2e.oof` | `reports/week2/03_errors_lstm_hash.md` |
| domain shift | `python -m l2e.compare_shift` | `reports/week2/04_domain_shift.md` |
| fix the main monitor | `bash scripts/finalize_monitor.sh` | `01_monitor_hash-roles.md`, `03_errors_transformer_hash-roles.md` |
| tests | `python -m pytest -q` | 22 tests |

**Board timing.** After `export_onnx`, copy `board/` to the UNO Q as in week 1, then:

```bash
python ~/l2e-board/time_monitor.py --bundle models/monitor_bundle_lstm_hash.json
python ~/l2e-board/time_monitor.py --bundle models/monitor_bundle_transformer_hash.json
```
---