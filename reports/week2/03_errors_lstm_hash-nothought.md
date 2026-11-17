# Week 2 · Out-of-fold errors and held-out error counts (lstm_hash-nothought)

5-fold out-of-fold predictions on each split's training DPs (folds grouped by task), and a full model on all training DPs for val and test. Each model is temperature-scaled on the split's val set; verdict at 0.5. Dangerous = missed hijack, costly = false alarm.

## Held-out error counts (the plan's table)

| held-out split | decision points | hijacked | monitor errors | dangerous | costly | AUROC |
|---|---|---|---|---|---|---|
| Task-disjoint test | 9796 | 1755 | 191 | 115 | 76 | 0.996 |
| Tool coverage 100%, seed0 | 2818 | 437 | 81 | 59 | 22 | 0.991 |
| Tool coverage 100%, seed1 | 2560 | 367 | 47 | 20 | 27 | 0.995 |
| Tool coverage 100%, seed2 | 2923 | 360 | 46 | 38 | 8 | 0.995 |
| Tool coverage 75%, seed0 | 2818 | 437 | 85 | 60 | 25 | 0.990 |
| Tool coverage 75%, seed1 | 2560 | 367 | 83 | 30 | 53 | 0.993 |
| Tool coverage 75%, seed2 | 2923 | 360 | 50 | 33 | 17 | 0.993 |
| Tool coverage 50%, seed0 | 2818 | 436 | 82 | 60 | 22 | 0.991 |
| Tool coverage 50%, seed1 | 2560 | 368 | 79 | 29 | 50 | 0.992 |
| Tool coverage 50%, seed2 | 2924 | 360 | 74 | 44 | 30 | 0.992 |
| Tool coverage 25%, seed0 | 2818 | 437 | 73 | 47 | 26 | 0.992 |
| Tool coverage 25%, seed1 | 2560 | 367 | 88 | 36 | 52 | 0.991 |
| Tool coverage 25%, seed2 | 2923 | 360 | 90 | 56 | 34 | 0.988 |
| Leave out banking | 11634 | 1170 | 457 | 268 | 189 | 0.977 |
| Leave out coding | 14817 | 2808 | 566 | 316 | 250 | 0.988 |
| Leave out email | 14986 | 2862 | 720 | 178 | 542 | 0.988 |
| Leave out medical | 14795 | 2758 | 640 | 331 | 309 | 0.984 |
| Leave out web | 14792 | 2751 | 860 | 504 | 356 | 0.971 |

Tool coverage, mean over seeds:

| coverage | decision points | errors | dangerous | costly | AUROC |
|---|---|---|---|---|---|
| 100% | 2767 | 58.0 | 39.0 | 19.0 | 0.994 |
| 75% | 2767 | 72.7 | 41.0 | 31.7 | 0.992 |
| 50% | 2767 | 78.3 | 44.3 | 34.0 | 0.992 |
| 25% | 2767 | 83.7 | 46.3 | 37.3 | 0.990 |

## Deferrer training material: out-of-fold errors on each split's training DPs

| split | training DPs | hijacked | OOF errors | dangerous | costly | OOF AUROC | val AUROC |
|---|---|---|---|---|---|---|---|
| taskdisjoint | 51507 | 8947 | 1138 | 696 | 442 | 0.995 | 0.997 |
| tool_seed0 | 17062 | 1800 | 573 | 300 | 273 | 0.986 | 0.989 |
| tool_seed1 | 26044 | 3244 | 597 | 301 | 296 | 0.993 | 0.996 |
| tool_seed2 | 23625 | 2921 | 567 | 294 | 273 | 0.993 | 0.997 |
| lodo_banking | 43081 | 8098 | 1033 | 468 | 565 | 0.996 | 0.997 |
| lodo_coding | 40955 | 6985 | 970 | 437 | 533 | 0.995 | 0.997 |
| lodo_email | 40663 | 6788 | 1140 | 539 | 601 | 0.993 | 0.995 |
| lodo_medical | 40882 | 6980 | 918 | 452 | 466 | 0.995 | 0.996 |
| lodo_web | 40447 | 6937 | 885 | 431 | 454 | 0.996 | 0.997 |
