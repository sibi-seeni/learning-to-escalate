# Week 2 · Compact neural monitors (hash-nothought text features)

Task-disjoint split, every decision point. Temperature fit on val; test metrics use calibrated scores and threshold 0.5.
9796 test decision points (1755 hijacked).

| model | params | size (KB) | epochs | AUROC | AUPRC | F1 | precision | recall | ECE raw → calibrated | T |
|---|---|---|---|---|---|---|---|---|---|---|
| logistic regression (6 features) | 7 |  |  | 0.905 | 0.806 | 0.689 | 0.611 | 0.791 | 0.153 → 0.101 | 0.65 |
| lstm + hash-nothought | 69777 | 275.8 | 11 | 0.996 | 0.986 | 0.943 | 0.967 | 0.920 | 0.007 → 0.005 | 1.15 |
| gru + hash-nothought | 61457 | 243.3 | 11 | 0.996 | 0.988 | 0.949 | 0.953 | 0.945 | 0.007 → 0.007 | 1.07 |
| transformer + hash-nothought | 106641 | 426.6 | 15 | 0.997 | 0.989 | 0.946 | 0.942 | 0.950 | 0.012 → 0.009 | 1.34 |

State-changing calls only:

| model | AUROC | F1 |
|---|---|---|
| logistic regression (6 features) | 0.895 | 0.756 |
| lstm + hash-nothought | 0.995 | 0.952 |
| gru + hash-nothought | 0.996 | 0.959 |
| transformer + hash-nothought | 0.996 | 0.956 |

Monitor errors on test (threshold 0.5), with dangerous / costly per domain:

| model | errors | dangerous | costly | banking | coding | email | medical | web |
|---|---|---|---|---|---|---|---|---|
| logistic regression (6 features) | 1252 | 366 | 886 | 83 / 99 | 91 / 61 | 44 / 426 | 73 / 171 | 75 / 129 |
| lstm + hash-nothought | 197 | 141 | 56 | 22 / 10 | 34 / 21 | 23 / 13 | 28 / 5 | 34 / 7 |
| gru + hash-nothought | 178 | 96 | 82 | 13 / 11 | 23 / 22 | 10 / 18 | 20 / 16 | 30 / 15 |
| transformer + hash-nothought | 190 | 88 | 102 | 13 / 17 | 16 / 23 | 11 / 23 | 23 / 14 | 25 / 25 |

Best validation AUPRC: lstm 0.9870, gru 0.9881, transformer 0.9885.
Chosen edge monitor (best validation AUPRC among int8-quantizable architectures, lstm, transformer; gaps under 0.001 count as ties and go to the smaller model): **transformer**, recorded in `models/monitor_best_hash-nothought.json`.
