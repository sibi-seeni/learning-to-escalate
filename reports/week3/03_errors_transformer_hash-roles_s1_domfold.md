# Out-of-fold errors and held-out error counts (transformer_hash-roles_s1_domfold)

Out-of-fold predictions on each split's training DPs (one fold per training domain (each fold model scores a domain it never saw)), and a full model on all training DPs for val and test. Each model is temperature-scaled on the split's val set; verdict at 0.5. Dangerous = missed hijack, costly = false alarm.

## Held-out error counts (the plan's table)

| held-out split | decision points | hijacked | monitor errors | dangerous | costly | AUROC |
|---|---|---|---|---|---|---|
| Leave out banking | 11634 | 1170 | 147 | 84 | 63 | 0.995 |
| Leave out coding | 14817 | 2808 | 225 | 154 | 71 | 0.997 |
| Leave out email | 14986 | 2862 | 956 | 15 | 941 | 0.998 |
| Leave out medical | 14795 | 2758 | 195 | 141 | 54 | 0.998 |
| Leave out web | 14792 | 2751 | 365 | 197 | 168 | 0.993 |

## Deferrer training material: out-of-fold errors on each split's training DPs

| split | training DPs | hijacked | OOF errors | dangerous | costly | OOF AUROC | val AUROC |
|---|---|---|---|---|---|---|---|
| lodo_banking | 43081 | 8098 | 1265 | 514 | 751 | 0.993 | 1.000 |
| lodo_coding | 40955 | 6985 | 1012 | 275 | 737 | 0.995 | 0.999 |
| lodo_email | 40663 | 6788 | 975 | 419 | 556 | 0.994 | 0.999 |
| lodo_medical | 40882 | 6980 | 1297 | 342 | 955 | 0.993 | 0.999 |
| lodo_web | 40447 | 6937 | 944 | 379 | 565 | 0.994 | 0.999 |
