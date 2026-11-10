# Week 1 · World-identity leakage check

Lookup = predict each test trace's category from its world object's majority category in train (unseen world -> benign). Binary accuracy, attacked vs. not.

## stratified split (majority-class rate 55.8%)

| domain | test traces | raw lookup acc | raw worlds seen | anonymized lookup acc |
|---|---|---|---|---|
| banking | 321 | 96.9% | 317 | 48.0% |
| coding | 390 | 96.2% | 390 | 58.7% |
| email | 390 | 47.7% | 390 | 45.4% |
| medical | 390 | 98.2% | 383 | 61.8% |
| web | 390 | 93.3% | 388 | 56.2% |
| all | 1881 | 86.1% | 1868 | 54.2% |

## taskdisjoint split (majority-class rate 55.0%)

| domain | test traces | raw lookup acc | raw worlds seen | anonymized lookup acc |
|---|---|---|---|---|
| banking | 317 | 95.3% | 316 | 52.7% |
| coding | 382 | 97.1% | 377 | 59.4% |
| email | 343 | 45.8% | 343 | 43.4% |
| medical | 376 | 99.5% | 374 | 59.8% |
| web | 304 | 93.4% | 300 | 61.2% |
| all | 1722 | 86.4% | 1710 | 55.4% |

## Residual world strings after anonymization (task-disjoint, all parts)

2 of 12536 traces still contain a world name, company or address string.
Top leftovers: `White` x2

A leftover can be a false alarm, e.g. a surname that is also an ordinary word in an args key such as 'White Blood Cell Count'. Look at the list before worrying.
