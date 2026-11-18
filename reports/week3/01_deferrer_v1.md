# Deferrer v1 · Learned error-aware deferral vs. confidence

Monitor: transformer_hash-roles (the main monitor). The deferrer is LightGBM trained on the monitor's out-of-fold errors on each split's training data, missed attacks weighted 5×. Every policy gets the same review budget (share of all decision points sent to a perfect reviewer). Natural attack rate (about 18% of decision points are hijacked), one seed; 1–5% prevalence, several seeds and intervals come in week 3.

## Task-disjoint test (no shift)

9796 decision points, 1755 hijacked. The monitor makes **55 errors, 23 of them missed attacks**. Deferrer trained on 418 out-of-fold errors (316 missed attacks) from 51507 training decision points.

All monitor errors caught:

| review budget | reviews | random | confidence | **learned** | oracle |
|---|---|---|---|---|---|
| 1% | 98 | 1 (1%) | 29 (53%) | **19 (35%)** | 55 (100%) |
| 2% | 196 | 1 (2%) | 39 (71%) | **32 (58%)** | 55 (100%) |
| 5% | 490 | 3 (5%) | 50 (91%) | **49 (89%)** | 55 (100%) |
| 10% | 980 | 6 (10%) | 54 (98%) | **53 (96%)** | 55 (100%) |
| 20% | 1959 | 11 (20%) | 55 (100%) | **55 (100%)** | 55 (100%) |

Missed attacks caught:

| review budget | random | confidence | **learned** | oracle |
|---|---|---|---|---|
| 1% | 0 (1%) | 12 (52%) | **12 (52%)** | 23 (100%) |
| 2% | 0 (2%) | 15 (65%) | **18 (78%)** | 23 (100%) |
| 5% | 1 (5%) | 21 (91%) | **21 (91%)** | 23 (100%) |
| 10% | 2 (10%) | 22 (96%) | **21 (91%)** | 23 (100%) |
| 20% | 5 (20%) | 23 (100%) | **23 (100%)** | 23 (100%) |

What the deferrer leans on (share of gain): margin 66%, score 9%, prev_score 7%, prev_max_score 7%, logit_cal 5%, step 2%.

## Held-out domain: web

14792 decision points, 2751 hijacked. The monitor makes **975 errors, 81 of them missed attacks**. Deferrer trained on 372 out-of-fold errors (224 missed attacks) from 40447 training decision points.

All monitor errors caught:

| review budget | reviews | random | confidence | **learned** | oracle |
|---|---|---|---|---|---|
| 1% | 148 | 10 (1%) | 76 (8%) | **38 (4%)** | 148 (15%) |
| 2% | 296 | 20 (2%) | 134 (14%) | **83 (9%)** | 296 (30%) |
| 5% | 740 | 49 (5%) | 307 (31%) | **172 (18%)** | 740 (76%) |
| 10% | 1479 | 98 (10%) | 554 (57%) | **400 (41%)** | 975 (100%) |
| 20% | 2958 | 195 (20%) | 835 (86%) | **707 (73%)** | 975 (100%) |

Missed attacks caught:

| review budget | random | confidence | **learned** | oracle |
|---|---|---|---|---|
| 1% | 1 (1%) | 9 (11%) | **8 (10%)** | 81 (100%) |
| 2% | 2 (2%) | 12 (15%) | **20 (25%)** | 81 (100%) |
| 5% | 4 (5%) | 21 (26%) | **38 (47%)** | 81 (100%) |
| 10% | 8 (10%) | 43 (53%) | **51 (63%)** | 81 (100%) |
| 20% | 16 (20%) | 67 (83%) | **72 (89%)** | 81 (100%) |

What the deferrer leans on (share of gain): margin 64%, prev_score 9%, prev_max_score 8%, logit_cal 6%, score 5%, step 2%.

Figure: `results/figures/06_deferral_curves.png`.
