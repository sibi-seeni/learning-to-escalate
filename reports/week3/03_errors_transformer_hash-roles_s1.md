# Out-of-fold errors and held-out error counts (transformer_hash-roles_s1)

Out-of-fold predictions on each split's training DPs (5 folds grouped by task), and a full model on all training DPs for val and test. Each model is temperature-scaled on the split's val set; verdict at 0.5. Dangerous = missed hijack, costly = false alarm.

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
| lodo_banking | 43081 | 8098 | 478 | 177 | 301 | 0.999 | 1.000 |
| lodo_coding | 40955 | 6985 | 361 | 185 | 176 | 0.999 | 0.999 |
| lodo_email | 40663 | 6788 | 412 | 203 | 209 | 0.998 | 0.999 |
| lodo_medical | 40882 | 6980 | 396 | 203 | 193 | 0.999 | 0.999 |
| lodo_web | 40447 | 6937 | 380 | 229 | 151 | 0.999 | 0.999 |
