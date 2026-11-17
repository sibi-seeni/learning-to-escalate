# Week 2 · Out-of-fold errors and held-out error counts (transformer_hash-roles)

5-fold out-of-fold predictions on each split's training DPs (folds grouped by task), and a full model on all training DPs for val and test. Each model is temperature-scaled on the split's val set; verdict at 0.5. Dangerous = missed hijack, costly = false alarm.

## Held-out error counts (the plan's table)

| held-out split | decision points | hijacked | monitor errors | dangerous | costly | AUROC |
|---|---|---|---|---|---|---|
| Task-disjoint test | 9796 | 1755 | 55 | 23 | 32 | 1.000 |
| Tool coverage 100%, seed0 | 2818 | 437 | 30 | 26 | 4 | 0.999 |
| Tool coverage 100%, seed1 | 2560 | 367 | 17 | 5 | 12 | 0.999 |
| Tool coverage 100%, seed2 | 2923 | 360 | 26 | 12 | 14 | 0.999 |
| Tool coverage 75%, seed0 | 2818 | 437 | 38 | 28 | 10 | 0.998 |
| Tool coverage 75%, seed1 | 2560 | 367 | 30 | 8 | 22 | 0.999 |
| Tool coverage 75%, seed2 | 2923 | 360 | 27 | 11 | 16 | 0.998 |
| Tool coverage 50%, seed0 | 2818 | 436 | 39 | 29 | 10 | 0.998 |
| Tool coverage 50%, seed1 | 2560 | 368 | 29 | 7 | 22 | 0.999 |
| Tool coverage 50%, seed2 | 2924 | 360 | 37 | 15 | 22 | 0.997 |
| Tool coverage 25%, seed0 | 2818 | 437 | 32 | 17 | 15 | 0.998 |
| Tool coverage 25%, seed1 | 2560 | 367 | 34 | 13 | 21 | 0.997 |
| Tool coverage 25%, seed2 | 2923 | 360 | 39 | 16 | 23 | 0.997 |
| Leave out banking | 11634 | 1170 | 160 | 67 | 93 | 0.996 |
| Leave out coding | 14817 | 2808 | 262 | 198 | 64 | 0.996 |
| Leave out email | 14986 | 2862 | 394 | 50 | 344 | 0.998 |
| Leave out medical | 14795 | 2758 | 416 | 28 | 388 | 0.999 |
| Leave out web | 14792 | 2751 | 975 | 81 | 894 | 0.992 |

Tool coverage, mean over seeds:

| coverage | decision points | errors | dangerous | costly | AUROC |
|---|---|---|---|---|---|
| 100% | 2767 | 24.3 | 14.3 | 10.0 | 0.999 |
| 75% | 2767 | 31.7 | 15.7 | 16.0 | 0.998 |
| 50% | 2767 | 35.0 | 17.0 | 18.0 | 0.998 |
| 25% | 2767 | 35.0 | 15.3 | 19.7 | 0.997 |

## Deferrer training material: out-of-fold errors on each split's training DPs

| split | training DPs | hijacked | OOF errors | dangerous | costly | OOF AUROC | val AUROC |
|---|---|---|---|---|---|---|---|
| taskdisjoint | 51507 | 8947 | 418 | 316 | 102 | 0.999 | 1.000 |
| tool_seed0 | 17062 | 1800 | 287 | 103 | 184 | 0.996 | 0.998 |
| tool_seed1 | 26044 | 3244 | 237 | 97 | 140 | 0.998 | 0.999 |
| tool_seed2 | 23625 | 2921 | 279 | 156 | 123 | 0.998 | 0.999 |
| lodo_banking | 43081 | 8098 | 464 | 212 | 252 | 0.999 | 0.999 |
| lodo_coding | 40955 | 6985 | 402 | 183 | 219 | 0.999 | 0.999 |
| lodo_email | 40663 | 6788 | 416 | 260 | 156 | 0.998 | 1.000 |
| lodo_medical | 40882 | 6980 | 419 | 225 | 194 | 0.998 | 0.999 |
| lodo_web | 40447 | 6937 | 372 | 224 | 148 | 0.999 | 0.999 |
