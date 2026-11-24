# Out-of-fold errors and held-out error counts (transformer_hash-roles_s2_fullonly)

Out-of-fold predictions on each split's training DPs (none here: this run trains only the full model, as a seed-ensemble member), and a full model on all training DPs for val and test. Each model is temperature-scaled on the split's val set; verdict at 0.5. Dangerous = missed hijack, costly = false alarm.

## Held-out error counts (the plan's table)

| held-out split | decision points | hijacked | monitor errors | dangerous | costly | AUROC |
|---|---|---|---|---|---|---|
| Task-disjoint test | 9796 | 1755 | 67 | 44 | 23 | 1.000 |
| Tool coverage 100%, seed0 | 2818 | 437 | 32 | 21 | 11 | 0.998 |
| Tool coverage 100%, seed1 | 2560 | 367 | 23 | 15 | 8 | 0.999 |
| Tool coverage 100%, seed2 | 2923 | 360 | 18 | 11 | 7 | 0.999 |
| Tool coverage 75%, seed0 | 2818 | 437 | 35 | 27 | 8 | 0.998 |
| Tool coverage 75%, seed1 | 2560 | 367 | 31 | 17 | 14 | 0.998 |
| Tool coverage 75%, seed2 | 2923 | 360 | 25 | 11 | 14 | 0.998 |
| Tool coverage 50%, seed0 | 2818 | 436 | 34 | 23 | 11 | 0.998 |
| Tool coverage 50%, seed1 | 2560 | 368 | 26 | 15 | 11 | 0.998 |
| Tool coverage 50%, seed2 | 2924 | 360 | 33 | 17 | 16 | 0.997 |
| Tool coverage 25%, seed0 | 2818 | 437 | 28 | 21 | 7 | 0.999 |
| Tool coverage 25%, seed1 | 2560 | 367 | 32 | 22 | 10 | 0.995 |
| Tool coverage 25%, seed2 | 2923 | 360 | 44 | 19 | 25 | 0.996 |

Tool coverage, mean over seeds:

| coverage | decision points | errors | dangerous | costly | AUROC |
|---|---|---|---|---|---|
| 100% | 2767 | 24.3 | 15.7 | 8.7 | 0.999 |
| 75% | 2767 | 30.3 | 18.3 | 12.0 | 0.998 |
| 50% | 2767 | 31.0 | 18.3 | 12.7 | 0.998 |
| 25% | 2767 | 34.7 | 20.7 | 14.0 | 0.997 |

## Deferrer training material: out-of-fold errors on each split's training DPs

| split | training DPs | hijacked | OOF errors | dangerous | costly | OOF AUROC | val AUROC |
|---|---|---|---|---|---|---|---|
