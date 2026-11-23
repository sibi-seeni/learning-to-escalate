# Out-of-fold errors and held-out error counts (transformer_hash-roles_domfold)

Out-of-fold predictions on each split's training DPs (one fold per training domain (each fold model scores a domain it never saw)), and a full model on all training DPs for val and test. Each model is temperature-scaled on the split's val set; verdict at 0.5. Dangerous = missed hijack, costly = false alarm.

## Held-out error counts (the plan's table)

| held-out split | decision points | hijacked | monitor errors | dangerous | costly | AUROC |
|---|---|---|---|---|---|---|
| Leave out banking | 11634 | 1170 | 160 | 67 | 93 | 0.996 |
| Leave out coding | 14817 | 2808 | 262 | 198 | 64 | 0.996 |
| Leave out email | 14986 | 2862 | 394 | 50 | 344 | 0.998 |
| Leave out medical | 14795 | 2758 | 416 | 28 | 388 | 0.999 |
| Leave out web | 14792 | 2751 | 975 | 81 | 894 | 0.992 |

## Deferrer training material: out-of-fold errors on each split's training DPs

| split | training DPs | hijacked | OOF errors | dangerous | costly | OOF AUROC | val AUROC |
|---|---|---|---|---|---|---|---|
| lodo_banking | 43081 | 8098 | 1712 | 289 | 1423 | 0.994 | 0.999 |
| lodo_coding | 40955 | 6985 | 1364 | 266 | 1098 | 0.995 | 0.999 |
| lodo_email | 40663 | 6788 | 1055 | 356 | 699 | 0.993 | 1.000 |
| lodo_medical | 40882 | 6980 | 1207 | 384 | 823 | 0.994 | 0.999 |
| lodo_web | 40447 | 6937 | 800 | 278 | 522 | 0.996 | 0.999 |
