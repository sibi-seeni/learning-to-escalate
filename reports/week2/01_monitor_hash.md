# Week 2 · Compact neural monitors (hash text features)

Task-disjoint split, every decision point. Temperature fit on val; test metrics use calibrated scores and threshold 0.5.
9796 test decision points (1755 hijacked).

| model | params | size (KB) | epochs | AUROC | AUPRC | F1 | precision | recall | ECE raw → calibrated | T |
|---|---|---|---|---|---|---|---|---|---|---|
| logistic regression (6 features) | 7 |  |  | 0.905 | 0.806 | 0.689 | 0.611 | 0.791 | 0.153 → 0.101 | 0.65 |
| lstm + hash | 69777 | 275.8 | 16 | 1.000 | 0.998 | 0.980 | 0.977 | 0.984 | 0.004 → 0.004 | 1.31 |
| gru + hash | 61457 | 243.3 | 18 | 0.999 | 0.998 | 0.980 | 0.981 | 0.979 | 0.005 → 0.003 | 1.49 |
| transformer + hash | 106641 | 426.6 | 16 | 0.999 | 0.998 | 0.980 | 0.982 | 0.978 | 0.005 → 0.003 | 1.33 |

State-changing calls only:

| model | AUROC | F1 |
|---|---|---|
| logistic regression (6 features) | 0.895 | 0.756 |
| lstm + hash | 0.999 | 0.983 |
| gru + hash | 0.999 | 0.982 |
| transformer + hash | 0.999 | 0.984 |

Monitor errors on test (threshold 0.5), with dangerous / costly per domain:

| model | errors | dangerous | costly | banking | coding | email | medical | web |
|---|---|---|---|---|---|---|---|---|
| logistic regression (6 features) | 1252 | 366 | 886 | 83 / 99 | 91 / 61 | 44 / 426 | 73 / 171 | 75 / 129 |
| lstm + hash | 69 | 29 | 40 | 5 / 5 | 4 / 16 | 7 / 10 | 4 / 3 | 9 / 6 |
| gru + hash | 71 | 37 | 34 | 8 / 2 | 6 / 7 | 9 / 15 | 3 / 3 | 11 / 7 |
| transformer + hash | 71 | 39 | 32 | 8 / 4 | 6 / 8 | 7 / 13 | 5 / 0 | 13 / 7 |

Best validation AUPRC: lstm 0.9980, gru 0.9985, transformer 0.9977.
Chosen edge monitor (best validation AUPRC among int8-quantizable architectures, lstm, transformer; gaps under 0.001 count as ties and go to the smaller model): **lstm**, recorded in `models/monitor_best_hash.json`.
