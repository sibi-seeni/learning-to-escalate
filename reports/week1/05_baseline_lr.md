# Week 1 · Six-feature logistic regression

## A. Sanity check: whole traces, stratified split (paper setup)

Ours:  precision 0.741 · recall 0.576 · f1 0.649 · auroc 0.747 · auprc 0.773
Paper: precision 0.778 · recall 0.554 · f1 0.647 · auroc 0.734 · auprc 0.754

| true group | flagged / total | flag rate |
|---|---|---|
| benign | 60 / 600 | 0.100 |
| delayed_execution | 30 / 134 | 0.224 |
| failed_attack | 85 / 225 | 0.378 |
| full_hijack | 431 / 466 | 0.925 |
| hard_negative | 22 / 225 | 0.098 |
| partial_hijack | 18 / 231 | 0.078 |

Standardized coefficients: {'n': 0.14, 'c_sum': 0.41, 'e_max': 0.54, 'u_max': 0.21, 's_sum': 0.06, 'k': 0.46} (paper: e_max 0.60, c_sum 0.48, u_max 0.23, k 0.19, n 0.14, s_sum 0.07)

## B. Prefix monitor: task-disjoint split, every decision point

Test decision points: 9796 (1755 hijacked). Threshold 0.5.

All calls:            precision 0.611 · recall 0.791 · f1 0.689 · auroc 0.905 · auprc 0.807
State-changing calls: precision 0.693 · recall 0.831 · f1 0.756 · auroc 0.895 · auprc 0.865

Monitor errors on the task-disjoint test part (this fills row 1 of the plan's error-count table for the LR baseline; the main monitor comes in week 2):

| subset | decision points | monitor errors | dangerous | costly |
|---|---|---|---|---|
| all | 9796 | 1252 | 366 | 886 |
| banking | 1725 | 182 | 83 | 99 |
| coding | 2226 | 152 | 91 | 61 |
| email | 2012 | 470 | 44 | 426 |
| medical | 2117 | 244 | 73 | 171 |
| web | 1716 | 204 | 75 | 129 |

Coefficients: {'n': 0.14, 'c_sum': 0.09, 'e_max': 0.61, 'u_max': 0.3, 's_sum': 0.16, 'k': 1.14}
Saved `models/lr_prefix_monitor.joblib` and `data/processed/lr_prefix_test_predictions.csv`.
