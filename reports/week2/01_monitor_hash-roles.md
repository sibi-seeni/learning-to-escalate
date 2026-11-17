# Week 2 · Compact neural monitors (hash-roles text features)

Task-disjoint split, every decision point. Temperature fit on val; test metrics use calibrated scores and threshold 0.5.
9796 test decision points (1755 hijacked).

| model | params | size (KB) | epochs | AUROC | AUPRC | F1 | precision | recall | ECE raw → calibrated | T |
|---|---|---|---|---|---|---|---|---|---|---|
| logistic regression (6 features) | 7 |  |  | 0.905 | 0.806 | 0.689 | 0.611 | 0.791 | 0.153 → 0.101 | 0.65 |
| transformer + hash-roles | 107153 | 428.6 | 14 | 1.000 | 0.999 | 0.984 | 0.982 | 0.987 | 0.003 → 0.003 | 1.17 |

State-changing calls only:

| model | AUROC | F1 |
|---|---|---|
| logistic regression (6 features) | 0.895 | 0.756 |
| transformer + hash-roles | 1.000 | 0.985 |

Monitor errors on test (threshold 0.5), with dangerous / costly per domain:

| model | errors | dangerous | costly | banking | coding | email | medical | web |
|---|---|---|---|---|---|---|---|---|
| logistic regression (6 features) | 1252 | 366 | 886 | 83 / 99 | 91 / 61 | 44 / 426 | 73 / 171 | 75 / 129 |
| transformer + hash-roles | 55 | 23 | 32 | 5 / 6 | 4 / 16 | 6 / 5 | 4 / 0 | 4 / 5 |

Best validation AUPRC: transformer 0.9983.
Chosen edge monitor (best validation AUPRC among int8-quantizable architectures, lstm, transformer; gaps under 0.001 count as ties and go to the smaller model): **transformer**, recorded in `models/monitor_best_hash-roles.json`.
