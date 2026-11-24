# Out-of-fold errors and held-out error counts (transformer_hash-roles_s1_fullonly)

Out-of-fold predictions on each split's training DPs (none here: this run trains only the full model, as a seed-ensemble member), and a full model on all training DPs for val and test. Each model is temperature-scaled on the split's val set; verdict at 0.5. Dangerous = missed hijack, costly = false alarm.

## Held-out error counts (the plan's table)

| held-out split | decision points | hijacked | monitor errors | dangerous | costly | AUROC |
|---|---|---|---|---|---|---|
| Task-disjoint test | 9796 | 1755 | 67 | 47 | 20 | 0.999 |
| Tool coverage 100%, seed0 | 2818 | 437 | 34 | 14 | 20 | 0.999 |
| Tool coverage 100%, seed1 | 2560 | 367 | 22 | 7 | 15 | 0.999 |
| Tool coverage 100%, seed2 | 2923 | 360 | 24 | 21 | 3 | 0.999 |
| Tool coverage 75%, seed0 | 2818 | 437 | 48 | 15 | 33 | 0.998 |
| Tool coverage 75%, seed1 | 2560 | 367 | 30 | 11 | 19 | 0.998 |
| Tool coverage 75%, seed2 | 2923 | 360 | 19 | 16 | 3 | 0.998 |
| Tool coverage 50%, seed0 | 2818 | 436 | 59 | 12 | 47 | 0.998 |
| Tool coverage 50%, seed1 | 2560 | 368 | 28 | 8 | 20 | 0.998 |
| Tool coverage 50%, seed2 | 2924 | 360 | 32 | 26 | 6 | 0.998 |
| Tool coverage 25%, seed0 | 2818 | 437 | 59 | 10 | 49 | 0.998 |
| Tool coverage 25%, seed1 | 2560 | 367 | 35 | 14 | 21 | 0.996 |
| Tool coverage 25%, seed2 | 2923 | 360 | 38 | 31 | 7 | 0.996 |

Tool coverage, mean over seeds:

| coverage | decision points | errors | dangerous | costly | AUROC |
|---|---|---|---|---|---|
| 100% | 2767 | 26.7 | 14.0 | 12.7 | 0.999 |
| 75% | 2767 | 32.3 | 14.0 | 18.3 | 0.998 |
| 50% | 2767 | 39.7 | 15.3 | 24.3 | 0.998 |
| 25% | 2767 | 44.0 | 18.3 | 25.7 | 0.997 |

## Deferrer training material: out-of-fold errors on each split's training DPs

| split | training DPs | hijacked | OOF errors | dangerous | costly | OOF AUROC | val AUROC |
|---|---|---|---|---|---|---|---|
