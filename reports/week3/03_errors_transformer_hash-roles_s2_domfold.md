# Out-of-fold errors and held-out error counts (transformer_hash-roles_s2_domfold)

Out-of-fold predictions on each split's training DPs (one fold per training domain (each fold model scores a domain it never saw)), and a full model on all training DPs for val and test. Each model is temperature-scaled on the split's val set; verdict at 0.5. Dangerous = missed hijack, costly = false alarm.

## Held-out error counts (the plan's table)

| held-out split | decision points | hijacked | monitor errors | dangerous | costly | AUROC |
|---|---|---|---|---|---|---|
| Leave out banking | 11634 | 1170 | 132 | 95 | 37 | 0.997 |
| Leave out coding | 14817 | 2808 | 299 | 287 | 12 | 0.995 |
| Leave out email | 14986 | 2862 | 235 | 71 | 164 | 0.998 |
| Leave out medical | 14795 | 2758 | 198 | 141 | 57 | 0.998 |
| Leave out web | 14792 | 2751 | 525 | 204 | 321 | 0.992 |

## Deferrer training material: out-of-fold errors on each split's training DPs

| split | training DPs | hijacked | OOF errors | dangerous | costly | OOF AUROC | val AUROC |
|---|---|---|---|---|---|---|---|
| lodo_banking | 43081 | 8098 | 2092 | 274 | 1818 | 0.992 | 1.000 |
| lodo_coding | 40955 | 6985 | 1394 | 414 | 980 | 0.991 | 0.999 |
| lodo_email | 40663 | 6788 | 1312 | 297 | 1015 | 0.994 | 0.999 |
| lodo_medical | 40882 | 6980 | 1870 | 231 | 1639 | 0.993 | 0.999 |
| lodo_web | 40447 | 6937 | 1009 | 295 | 714 | 0.993 | 0.999 |
