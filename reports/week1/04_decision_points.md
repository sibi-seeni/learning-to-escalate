# Week 1 · Decision points (task-disjoint split)

71024 decision points from 12536 traces, saved to `data/processed/decision_points.csv`.

| part | all calls | hijacked | state-changing calls | hijacked (state-changing) |
|---|---|---|---|---|
| train | 51507 | 8947 | 23287 | 7166 |
| val | 9721 | 1647 | 4318 | 1288 |
| test | 9796 | 1755 | 4539 | 1373 |

Attacked traces with at least one hijacked decision point (reachable / total):

| compliance | every call | state-changing only |
|---|---|---|
| delayed_execution | 900 / 900 | 805 / 900 |
| full_hijack | 3136 / 3136 | 3090 / 3136 |
| partial_hijack | 1500 / 1500 | 945 / 1500 |
| all | 5536 / 5536 | 4840 / 5536 |

Step labels at decision points: benign 51639, injection_point 5536, hijacked 12349, failed_injection 1500
