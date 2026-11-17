# Week 2 · Out-of-fold errors and held-out error counts (lstm_hash)

5-fold out-of-fold predictions on each split's training DPs (folds grouped by task), and a full model on all training DPs for val and test. Each model is temperature-scaled on the split's val set; verdict at 0.5. Dangerous = missed hijack, costly = false alarm.

## Held-out error counts (the plan's table)

| held-out split | decision points | hijacked | monitor errors | dangerous | costly | AUROC |
|---|---|---|---|---|---|---|
| Task-disjoint test | 9796 | 1755 | 69 | 29 | 40 | 1.000 |
| Tool coverage 100%, seed0 | 2818 | 437 | 34 | 19 | 15 | 0.999 |
| Tool coverage 100%, seed1 | 2560 | 367 | 23 | 8 | 15 | 0.999 |
| Tool coverage 100%, seed2 | 2923 | 360 | 21 | 16 | 5 | 0.998 |
| Tool coverage 75%, seed0 | 2818 | 437 | 42 | 22 | 20 | 0.997 |
| Tool coverage 75%, seed1 | 2560 | 367 | 32 | 11 | 21 | 0.999 |
| Tool coverage 75%, seed2 | 2923 | 360 | 26 | 17 | 9 | 0.998 |
| Tool coverage 50%, seed0 | 2818 | 436 | 52 | 21 | 31 | 0.996 |
| Tool coverage 50%, seed1 | 2560 | 368 | 30 | 13 | 17 | 0.998 |
| Tool coverage 50%, seed2 | 2924 | 360 | 32 | 22 | 10 | 0.997 |
| Tool coverage 25%, seed0 | 2818 | 437 | 52 | 21 | 31 | 0.995 |
| Tool coverage 25%, seed1 | 2560 | 367 | 34 | 19 | 15 | 0.997 |
| Tool coverage 25%, seed2 | 2923 | 360 | 46 | 29 | 17 | 0.995 |
| Leave out banking | 11634 | 1170 | 163 | 93 | 70 | 0.995 |
| Leave out coding | 14817 | 2808 | 251 | 139 | 112 | 0.996 |
| Leave out email | 14986 | 2862 | 536 | 48 | 488 | 0.997 |
| Leave out medical | 14795 | 2758 | 406 | 60 | 346 | 0.998 |
| Leave out web | 14792 | 2751 | 1290 | 103 | 1187 | 0.986 |

Tool coverage, mean over seeds:

| coverage | decision points | errors | dangerous | costly | AUROC |
|---|---|---|---|---|---|
| 100% | 2767 | 26.0 | 14.3 | 11.7 | 0.999 |
| 75% | 2767 | 33.3 | 16.7 | 16.7 | 0.998 |
| 50% | 2767 | 38.0 | 18.7 | 19.3 | 0.997 |
| 25% | 2767 | 44.0 | 23.0 | 21.0 | 0.996 |

## Deferrer training material: out-of-fold errors on each split's training DPs

| split | training DPs | hijacked | OOF errors | dangerous | costly | OOF AUROC | val AUROC |
|---|---|---|---|---|---|---|---|
| taskdisjoint | 51507 | 8947 | 457 | 236 | 221 | 0.999 | 1.000 |
| tool_seed0 | 17062 | 1800 | 286 | 137 | 149 | 0.996 | 0.998 |
| tool_seed1 | 26044 | 3244 | 258 | 144 | 114 | 0.998 | 1.000 |
| tool_seed2 | 23625 | 2921 | 260 | 114 | 146 | 0.998 | 0.999 |
| lodo_banking | 43081 | 8098 | 419 | 180 | 239 | 0.999 | 1.000 |
| lodo_coding | 40955 | 6985 | 335 | 204 | 131 | 0.999 | 1.000 |
| lodo_email | 40663 | 6788 | 487 | 187 | 300 | 0.998 | 0.999 |
| lodo_medical | 40882 | 6980 | 339 | 185 | 154 | 0.999 | 0.999 |
| lodo_web | 40447 | 6937 | 294 | 188 | 106 | 0.999 | 0.999 |
