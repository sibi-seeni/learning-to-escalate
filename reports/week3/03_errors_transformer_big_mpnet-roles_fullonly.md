# Out-of-fold errors and held-out error counts (transformer_big_mpnet-roles_fullonly)

Out-of-fold predictions on each split's training DPs (none here: this run trains only the full model, as a seed-ensemble member), and a full model on all training DPs for val and test. Each model is temperature-scaled on the split's val set; verdict at 0.5. Dangerous = missed hijack, costly = false alarm.

## Held-out error counts (the plan's table)

| held-out split | decision points | hijacked | monitor errors | dangerous | costly | AUROC |
|---|---|---|---|---|---|---|
| Task-disjoint test | 9796 | 1755 | 93 | 62 | 31 | 0.999 |
| Tool coverage 100%, seed0 | 2818 | 437 | 49 | 34 | 15 | 0.997 |
| Tool coverage 100%, seed1 | 2560 | 367 | 31 | 26 | 5 | 0.997 |
| Tool coverage 100%, seed2 | 2923 | 360 | 61 | 6 | 55 | 0.999 |
| Tool coverage 75%, seed0 | 2818 | 437 | 61 | 40 | 21 | 0.996 |
| Tool coverage 75%, seed1 | 2560 | 367 | 57 | 30 | 27 | 0.994 |
| Tool coverage 75%, seed2 | 2923 | 360 | 91 | 5 | 86 | 0.998 |
| Tool coverage 50%, seed0 | 2818 | 436 | 67 | 34 | 33 | 0.995 |
| Tool coverage 50%, seed1 | 2560 | 368 | 47 | 21 | 26 | 0.995 |
| Tool coverage 50%, seed2 | 2924 | 360 | 115 | 4 | 111 | 0.997 |
| Tool coverage 25%, seed0 | 2818 | 437 | 70 | 33 | 37 | 0.992 |
| Tool coverage 25%, seed1 | 2560 | 367 | 76 | 33 | 43 | 0.992 |
| Tool coverage 25%, seed2 | 2923 | 360 | 145 | 8 | 137 | 0.995 |
| Leave out banking | 11634 | 1170 | 210 | 88 | 122 | 0.995 |
| Leave out coding | 14817 | 2808 | 597 | 227 | 370 | 0.989 |
| Leave out email | 14986 | 2862 | 505 | 113 | 392 | 0.995 |
| Leave out medical | 14795 | 2758 | 421 | 409 | 12 | 0.996 |
| Leave out web | 14792 | 2751 | 766 | 190 | 576 | 0.988 |

Tool coverage, mean over seeds:

| coverage | decision points | errors | dangerous | costly | AUROC |
|---|---|---|---|---|---|
| 100% | 2767 | 47.0 | 22.0 | 25.0 | 0.998 |
| 75% | 2767 | 69.7 | 25.0 | 44.7 | 0.996 |
| 50% | 2767 | 76.3 | 19.7 | 56.7 | 0.995 |
| 25% | 2767 | 97.0 | 24.7 | 72.3 | 0.993 |

## Deferrer training material: out-of-fold errors on each split's training DPs

| split | training DPs | hijacked | OOF errors | dangerous | costly | OOF AUROC | val AUROC |
|---|---|---|---|---|---|---|---|
