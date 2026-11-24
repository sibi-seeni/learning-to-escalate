# Deferral at equal review budgets (week 3)

Monitor: transformer on hash-roles. 200 resamples per cell; intervals are the middle 95% over resamples (traces, monitor seeds and deferrer seeds all vary). "Left" = what the system still gets wrong after a perfect reviewer has checked the deferred calls, per 1,000 calls. Lower is better.

Raw and calibrated confidence rank calls identically (temperature scaling keeps the order), so they are one policy ("least confident"). The 3-seed policies need the main monitor trained with several seeds on that split; where it was not, they are left out (—).

## Test sets

| setting | variants (split × monitor seed) | seeds in the ensemble | calls | hijacked | monitor errors | missed attacks |
|---|---|---|---|---|---|---|
| Task-disjoint test | 1 | 3 | 9796 | 1755 | 55 | 23 |
| Tool coverage 100% | 3 | 3 | 2767 | 388 | 24 | 14 |
| Tool coverage 75% | 3 | 3 | 2767 | 388 | 32 | 16 |
| Tool coverage 50% | 3 | 3 | 2767 | 388 | 35 | 17 |
| Tool coverage 25% | 3 | 3 | 2767 | 388 | 35 | 15 |
| Unseen domain: banking | 3 | 3 | 11634 | 1170 | 146 | 82 |
| Unseen domain: coding | 3 | 3 | 14817 | 2808 | 262 | 213 |
| Unseen domain: email | 3 | 3 | 14986 | 2862 | 528 | 45 |
| Unseen domain: medical | 3 | 3 | 14795 | 2758 | 270 | 103 |
| Unseen domain: web | 3 | 3 | 14792 | 2751 | 622 | 161 |
| Unseen domains (mean) | 15 | 3 | 14205 | 2470 | 366 | 121 |

## Attack rate 2%: missed attacks left per 1,000 calls

Review budget 2% (no deferral in the first column):

| setting | no deferral (main monitor) | least confident | most novel | stronger monitor, least confident | big mpnet monitor, least confident | transformer and LSTM disagree most | transformer + LSTM, unsure or disagreeing | 3-seed ensemble, least confident | seeds disagree most | 3-seed ensemble, unsure or disagreeing | LightGBM deferrer (task folds) | LightGBM deferrer (domain folds) | transformer deferrer (task folds) | transformer deferrer (domain folds) | perfect (upper bound) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Task-disjoint test | 0.26 [0.00, 0.74] | 0.07 [0.00, 0.37] | 0.25 [0.00, 0.73] | 0.05 [0.00, 0.19] | 0.22 [0.00, 0.73] | 0.08 [0.00, 0.37] | 0.07 [0.00, 0.36] | 0.07 [0.00, 0.36] | 0.09 [0.00, 0.37] | 0.08 [0.00, 0.37] | 0.08 [0.00, 0.37] | — | 0.10 [0.00, 0.37] | — | 0.00 [0.00, 0.00] |
| Tool coverage 100% | 0.66 [0.00, 2.31] | 0.24 [0.00, 1.11] | 0.56 [0.00, 2.22] | 0.22 [0.00, 1.11] | 0.47 [0.00, 2.19] | 0.27 [0.00, 1.14] | 0.24 [0.00, 1.12] | 0.20 [0.00, 1.10] | 0.22 [0.00, 1.09] | 0.20 [0.00, 1.09] | 0.29 [0.00, 1.40] | — | 0.38 [0.00, 1.67] | — | 0.00 [0.00, 0.00] |
| Tool coverage 75% | 0.70 [0.00, 2.09] | 0.34 [0.00, 1.52] | 0.63 [0.00, 2.04] | 0.30 [0.00, 1.04] | 0.70 [0.00, 2.10] | 0.30 [0.00, 1.04] | 0.26 [0.00, 1.04] | 0.28 [0.00, 1.04] | 0.26 [0.00, 1.02] | 0.24 [0.00, 1.02] | 0.46 [0.00, 1.60] | — | 0.47 [0.00, 2.03] | — | 0.00 [0.00, 0.00] |
| Tool coverage 50% | 0.82 [0.00, 2.43] | 0.40 [0.00, 1.47] | 0.79 [0.00, 2.43] | 0.43 [0.00, 1.45] | 0.69 [0.00, 2.39] | 0.48 [0.00, 1.91] | 0.44 [0.00, 1.46] | 0.27 [0.00, 0.97] | 0.36 [0.00, 1.41] | 0.33 [0.00, 1.41] | 0.56 [0.00, 1.95] | — | 0.58 [0.00, 2.00] | — | 0.00 [0.00, 0.00] |
| Tool coverage 25% | 0.79 [0.00, 1.91] | 0.38 [0.00, 1.29] | 0.73 [0.00, 1.87] | 0.48 [0.00, 1.44] | 0.85 [0.00, 2.82] | 0.44 [0.00, 1.33] | 0.46 [0.00, 1.37] | 0.51 [0.00, 1.52] | 0.49 [0.00, 1.37] | 0.48 [0.00, 1.35] | 0.53 [0.00, 1.44] | — | 0.57 [0.00, 1.72] | — | 0.00 [0.00, 0.00] |
| Unseen domain: banking | 1.37 [0.75, 2.25] | 0.63 [0.24, 1.36] | 1.32 [0.74, 2.12] | 0.66 [0.13, 1.25] | 0.86 [0.37, 1.38] | 0.77 [0.37, 1.36] | 0.68 [0.25, 1.25] | 0.62 [0.13, 1.24] | 0.71 [0.25, 1.39] | 0.66 [0.24, 1.35] | 0.73 [0.25, 1.36] | 0.89 [0.37, 1.64] | 0.88 [0.37, 1.61] | 0.83 [0.36, 1.62] | 0.00 [0.00, 0.00] |
| Unseen domain: coding | 1.53 [0.62, 2.82] | 0.74 [0.13, 1.46] | 1.40 [0.61, 2.67] | 0.67 [0.12, 1.35] | 1.30 [0.50, 2.24] | 0.71 [0.13, 1.35] | 0.66 [0.12, 1.32] | 0.64 [0.12, 1.33] | 0.71 [0.12, 1.35] | 0.66 [0.12, 1.35] | 0.87 [0.25, 1.70] | 0.92 [0.25, 1.84] | 1.05 [0.37, 2.37] | 1.09 [0.25, 2.44] | 0.00 [0.00, 0.00] |
| Unseen domain: email | 0.31 [0.00, 0.98] | 0.21 [0.00, 0.85] | 0.30 [0.00, 0.98] | 0.19 [0.00, 0.72] | 0.54 [0.12, 1.23] | 0.22 [0.00, 0.85] | 0.20 [0.00, 0.74] | 0.21 [0.00, 0.73] | 0.25 [0.00, 0.86] | 0.21 [0.00, 0.73] | 0.24 [0.00, 0.85] | 0.27 [0.00, 0.98] | 0.27 [0.00, 0.98] | 0.28 [0.00, 0.98] | 0.00 [0.00, 0.00] |
| Unseen domain: medical | 0.74 [0.00, 1.84] | 0.39 [0.00, 1.23] | 0.74 [0.00, 1.73] | 0.28 [0.00, 0.74] | 1.27 [0.61, 2.05] | 0.40 [0.00, 0.97] | 0.31 [0.00, 0.86] | 0.34 [0.00, 0.86] | 0.32 [0.00, 0.85] | 0.30 [0.00, 0.86] | 0.37 [0.00, 1.11] | 0.47 [0.00, 1.23] | 0.54 [0.00, 1.36] | 0.47 [0.00, 1.35] | 0.00 [0.00, 0.00] |
| Unseen domain: web | 1.14 [0.25, 2.29] | 0.91 [0.13, 2.01] | 1.08 [0.24, 2.16] | 0.72 [0.13, 1.39] | 1.17 [0.38, 2.24] | 1.02 [0.13, 2.12] | 0.74 [0.13, 1.51] | 0.93 [0.37, 1.66] | 1.03 [0.25, 2.03] | 0.92 [0.37, 1.65] | 0.88 [0.13, 1.77] | 1.02 [0.24, 2.23] | 0.89 [0.13, 1.79] | 0.96 [0.13, 2.01] | 0.00 [0.00, 0.00] |
| Unseen domains (mean) | 1.02 [0.57, 1.68] | 0.58 [0.30, 0.91] | 0.97 [0.52, 1.55] | 0.51 [0.30, 0.81] | 1.03 [0.72, 1.38] | 0.62 [0.35, 0.98] | 0.52 [0.32, 0.84] | 0.55 [0.33, 0.87] | 0.60 [0.35, 0.97] | 0.55 [0.32, 0.84] | 0.62 [0.37, 0.94] | 0.71 [0.40, 1.16] | 0.73 [0.39, 1.24] | 0.73 [0.37, 1.33] | 0.00 [0.00, 0.00] |

Review budget 5% (no deferral in the first column):

| setting | no deferral (main monitor) | least confident | most novel | stronger monitor, least confident | big mpnet monitor, least confident | transformer and LSTM disagree most | transformer + LSTM, unsure or disagreeing | 3-seed ensemble, least confident | seeds disagree most | 3-seed ensemble, unsure or disagreeing | LightGBM deferrer (task folds) | LightGBM deferrer (domain folds) | transformer deferrer (task folds) | transformer deferrer (domain folds) | perfect (upper bound) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Task-disjoint test | 0.26 [0.00, 0.74] | 0.02 [0.00, 0.18] | 0.25 [0.00, 0.73] | 0.01 [0.00, 0.18] | 0.15 [0.00, 0.54] | 0.03 [0.00, 0.19] | 0.02 [0.00, 0.19] | 0.01 [0.00, 0.18] | 0.03 [0.00, 0.18] | 0.01 [0.00, 0.18] | 0.03 [0.00, 0.19] | — | 0.08 [0.00, 0.37] | — | 0.00 [0.00, 0.00] |
| Tool coverage 100% | 0.66 [0.00, 2.31] | 0.14 [0.00, 0.93] | 0.49 [0.00, 2.22] | 0.08 [0.00, 0.58] | 0.18 [0.00, 1.09] | 0.17 [0.00, 0.95] | 0.14 [0.00, 0.93] | 0.13 [0.00, 0.92] | 0.11 [0.00, 0.58] | 0.13 [0.00, 0.90] | 0.15 [0.00, 0.94] | — | 0.24 [0.00, 1.64] | — | 0.00 [0.00, 0.00] |
| Tool coverage 75% | 0.70 [0.00, 2.09] | 0.18 [0.00, 1.00] | 0.63 [0.00, 2.04] | 0.10 [0.00, 0.53] | 0.32 [0.00, 1.56] | 0.12 [0.00, 0.88] | 0.11 [0.00, 0.87] | 0.11 [0.00, 0.53] | 0.11 [0.00, 0.53] | 0.11 [0.00, 0.53] | 0.25 [0.00, 1.04] | — | 0.34 [0.00, 1.52] | — | 0.00 [0.00, 0.00] |
| Tool coverage 50% | 0.82 [0.00, 2.43] | 0.19 [0.00, 0.96] | 0.74 [0.00, 2.38] | 0.14 [0.00, 0.88] | 0.35 [0.00, 1.45] | 0.19 [0.00, 0.97] | 0.16 [0.00, 0.89] | 0.12 [0.00, 0.52] | 0.15 [0.00, 0.88] | 0.12 [0.00, 0.52] | 0.24 [0.00, 0.99] | — | 0.44 [0.00, 1.54] | — | 0.00 [0.00, 0.00] |
| Tool coverage 25% | 0.79 [0.00, 1.91] | 0.27 [0.00, 0.98] | 0.67 [0.00, 1.87] | 0.21 [0.00, 0.94] | 0.42 [0.00, 1.88] | 0.22 [0.00, 0.90] | 0.19 [0.00, 0.89] | 0.22 [0.00, 0.90] | 0.27 [0.00, 0.99] | 0.22 [0.00, 0.91] | 0.32 [0.00, 1.40] | — | 0.47 [0.00, 1.44] | — | 0.00 [0.00, 0.00] |
| Unseen domain: banking | 1.37 [0.75, 2.25] | 0.35 [0.00, 0.75] | 1.25 [0.63, 2.01] | 0.36 [0.00, 0.86] | 0.43 [0.12, 0.87] | 0.43 [0.12, 0.87] | 0.39 [0.00, 0.87] | 0.30 [0.00, 0.74] | 0.38 [0.00, 0.86] | 0.35 [0.00, 0.85] | 0.30 [0.00, 0.75] | 0.47 [0.12, 1.11] | 0.61 [0.13, 1.34] | 0.57 [0.12, 1.24] | 0.00 [0.00, 0.00] |
| Unseen domain: coding | 1.53 [0.62, 2.82] | 0.45 [0.00, 0.98] | 1.28 [0.49, 2.44] | 0.33 [0.00, 0.85] | 1.02 [0.37, 1.84] | 0.41 [0.00, 0.97] | 0.36 [0.00, 0.85] | 0.36 [0.00, 0.85] | 0.39 [0.00, 0.87] | 0.39 [0.00, 0.86] | 0.53 [0.12, 1.10] | 0.54 [0.00, 1.23] | 0.81 [0.12, 1.89] | 0.80 [0.12, 2.08] | 0.00 [0.00, 0.00] |
| Unseen domain: email | 0.31 [0.00, 0.98] | 0.17 [0.00, 0.62] | 0.30 [0.00, 0.98] | 0.15 [0.00, 0.61] | 0.37 [0.00, 0.97] | 0.15 [0.00, 0.72] | 0.14 [0.00, 0.61] | 0.15 [0.00, 0.72] | 0.16 [0.00, 0.49] | 0.14 [0.00, 0.61] | 0.19 [0.00, 0.74] | 0.20 [0.00, 0.74] | 0.25 [0.00, 0.98] | 0.26 [0.00, 0.98] | 0.00 [0.00, 0.00] |
| Unseen domain: medical | 0.74 [0.00, 1.84] | 0.23 [0.00, 0.86] | 0.68 [0.00, 1.60] | 0.17 [0.00, 0.49] | 0.64 [0.12, 1.23] | 0.20 [0.00, 0.61] | 0.17 [0.00, 0.49] | 0.15 [0.00, 0.37] | 0.13 [0.00, 0.38] | 0.13 [0.00, 0.38] | 0.21 [0.00, 0.86] | 0.25 [0.00, 0.98] | 0.40 [0.00, 1.23] | 0.34 [0.00, 1.11] | 0.00 [0.00, 0.00] |
| Unseen domain: web | 1.14 [0.25, 2.29] | 0.65 [0.13, 1.37] | 1.01 [0.13, 2.12] | 0.52 [0.00, 1.11] | 0.86 [0.25, 1.78] | 0.77 [0.13, 1.61] | 0.56 [0.12, 1.12] | 0.65 [0.13, 1.25] | 0.81 [0.13, 1.62] | 0.68 [0.24, 1.26] | 0.63 [0.00, 1.50] | 0.74 [0.13, 1.76] | 0.69 [0.12, 1.64] | 0.75 [0.13, 1.74] | 0.00 [0.00, 0.00] |
| Unseen domains (mean) | 1.02 [0.57, 1.68] | 0.37 [0.17, 0.67] | 0.90 [0.48, 1.50] | 0.31 [0.15, 0.50] | 0.66 [0.42, 0.96] | 0.39 [0.22, 0.64] | 0.32 [0.17, 0.55] | 0.32 [0.15, 0.55] | 0.38 [0.20, 0.60] | 0.34 [0.17, 0.55] | 0.37 [0.17, 0.67] | 0.44 [0.20, 0.79] | 0.55 [0.25, 1.00] | 0.54 [0.23, 1.07] | 0.00 [0.00, 0.00] |

Learned policies vs. least confident at attack rate 2%: difference in missed attacks left per 1,000 calls (negative = the learned one leaves fewer), with the share of resamples where it is better.

| setting | policy | 2% budget | better in | 5% budget | better in |
|---|---|---|---|---|---|
| Task-disjoint test | stronger monitor, least confident | -0.02 [-0.18, 0.00] | 13% | -0.01 [-0.18, 0.00] | 8% |
| Task-disjoint test | big mpnet monitor, least confident | 0.15 [-0.19, 0.73] | 10% | 0.13 [-0.18, 0.54] | 6% |
| Task-disjoint test | transformer and LSTM disagree most | 0.02 [-0.18, 0.19] | 10% | 0.01 [-0.18, 0.19] | 8% |
| Task-disjoint test | transformer + LSTM, unsure or disagreeing | 0.00 [-0.18, 0.18] | 10% | -0.00 [-0.18, 0.19] | 8% |
| Task-disjoint test | 3-seed ensemble, least confident | -0.00 [-0.19, 0.18] | 8% | -0.01 [-0.18, 0.00] | 6% |
| Task-disjoint test | seeds disagree most | 0.02 [-0.19, 0.18] | 6% | 0.01 [0.00, 0.18] | 0% |
| Task-disjoint test | 3-seed ensemble, unsure or disagreeing | 0.01 [-0.19, 0.18] | 6% | -0.01 [-0.18, 0.18] | 6% |
| Task-disjoint test | LightGBM deferrer (task folds) | 0.01 [-0.18, 0.19] | 8% | 0.01 [0.00, 0.18] | 1% |
| Task-disjoint test | transformer deferrer (task folds) | 0.04 [0.00, 0.18] | 2% | 0.05 [0.00, 0.36] | 0% |
| Tool coverage 100% | stronger monitor, least confident | -0.02 [-0.47, 0.00] | 6% | -0.06 [-0.56, 0.00] | 14% |
| Tool coverage 100% | big mpnet monitor, least confident | 0.23 [-0.59, 1.67] | 14% | 0.03 [-0.56, 1.07] | 14% |
| Tool coverage 100% | transformer and LSTM disagree most | 0.03 [0.00, 0.56] | 2% | 0.02 [-0.56, 0.48] | 4% |
| Tool coverage 100% | transformer + LSTM, unsure or disagreeing | -0.00 [-0.01, 0.01] | 2% | -0.01 [-0.56, 0.48] | 8% |
| Tool coverage 100% | 3-seed ensemble, least confident | -0.04 [-0.55, 0.54] | 12% | -0.01 [-0.53, 0.01] | 6% |
| Tool coverage 100% | seeds disagree most | -0.02 [-0.56, 0.56] | 9% | -0.04 [-0.56, 0.48] | 10% |
| Tool coverage 100% | 3-seed ensemble, unsure or disagreeing | -0.04 [-0.56, 0.54] | 10% | -0.02 [-0.56, 0.48] | 7% |
| Tool coverage 100% | LightGBM deferrer (task folds) | 0.05 [0.00, 0.57] | 2% | 0.01 [-0.56, 0.57] | 10% |
| Tool coverage 100% | transformer deferrer (task folds) | 0.14 [-0.47, 1.10] | 5% | 0.09 [-0.55, 1.14] | 8% |
| Tool coverage 75% | stronger monitor, least confident | -0.04 [-0.52, 0.52] | 14% | -0.08 [-0.53, 0.50] | 18% |
| Tool coverage 75% | big mpnet monitor, least confident | 0.36 [-0.45, 1.59] | 12% | 0.14 [-0.52, 1.09] | 15% |
| Tool coverage 75% | transformer and LSTM disagree most | -0.04 [-0.54, 0.52] | 15% | -0.06 [-0.53, 0.01] | 12% |
| Tool coverage 75% | transformer + LSTM, unsure or disagreeing | -0.08 [-0.55, 0.51] | 18% | -0.07 [-0.53, 0.00] | 14% |
| Tool coverage 75% | 3-seed ensemble, least confident | -0.06 [-0.53, 0.52] | 19% | -0.06 [-0.52, 0.00] | 12% |
| Tool coverage 75% | seeds disagree most | -0.08 [-1.02, 0.52] | 24% | -0.06 [-0.52, 0.00] | 12% |
| Tool coverage 75% | 3-seed ensemble, unsure or disagreeing | -0.10 [-0.55, 0.52] | 23% | -0.06 [-0.52, 0.00] | 12% |
| Tool coverage 75% | LightGBM deferrer (task folds) | 0.12 [-0.51, 1.03] | 4% | 0.07 [-0.52, 1.01] | 6% |
| Tool coverage 75% | transformer deferrer (task folds) | 0.13 [-0.43, 1.03] | 3% | 0.16 [-0.44, 1.49] | 7% |
| Tool coverage 50% | stronger monitor, least confident | 0.04 [-0.49, 0.51] | 11% | -0.05 [-0.49, 0.00] | 11% |
| Tool coverage 50% | big mpnet monitor, least confident | 0.29 [-0.90, 1.89] | 18% | 0.16 [-0.49, 0.99] | 11% |
| Tool coverage 50% | transformer and LSTM disagree most | 0.09 [-0.48, 0.96] | 10% | 0.00 [-0.49, 0.48] | 8% |
| Tool coverage 50% | transformer + LSTM, unsure or disagreeing | 0.05 [-0.49, 0.51] | 10% | -0.03 [-0.50, 0.44] | 10% |
| Tool coverage 50% | 3-seed ensemble, least confident | -0.13 [-0.88, 0.50] | 28% | -0.07 [-0.51, 0.00] | 14% |
| Tool coverage 50% | seeds disagree most | -0.04 [-0.89, 0.51] | 21% | -0.04 [-0.49, 0.47] | 12% |
| Tool coverage 50% | 3-seed ensemble, unsure or disagreeing | -0.06 [-0.50, 0.51] | 22% | -0.07 [-0.50, 0.00] | 14% |
| Tool coverage 50% | LightGBM deferrer (task folds) | 0.16 [-0.48, 0.98] | 4% | 0.05 [-0.48, 0.51] | 10% |
| Tool coverage 50% | transformer deferrer (task folds) | 0.18 [-0.49, 0.99] | 5% | 0.25 [-0.01, 1.46] | 2% |
| Tool coverage 25% | stronger monitor, least confident | 0.10 [-0.51, 0.87] | 9% | -0.06 [-0.52, 0.48] | 18% |
| Tool coverage 25% | big mpnet monitor, least confident | 0.47 [-0.52, 2.55] | 13% | 0.15 [-0.86, 1.43] | 18% |
| Tool coverage 25% | transformer and LSTM disagree most | 0.06 [-0.51, 0.50] | 12% | -0.05 [-0.51, 0.48] | 16% |
| Tool coverage 25% | transformer + LSTM, unsure or disagreeing | 0.08 [-0.51, 0.87] | 10% | -0.08 [-0.52, 0.01] | 17% |
| Tool coverage 25% | 3-seed ensemble, least confident | 0.13 [-0.47, 0.88] | 6% | -0.05 [-0.48, 0.43] | 14% |
| Tool coverage 25% | seeds disagree most | 0.11 [-0.48, 0.89] | 8% | 0.00 [-0.48, 0.48] | 6% |
| Tool coverage 25% | 3-seed ensemble, unsure or disagreeing | 0.10 [-0.47, 0.53] | 6% | -0.05 [-0.48, 0.00] | 12% |
| Tool coverage 25% | LightGBM deferrer (task folds) | 0.15 [-0.51, 0.95] | 6% | 0.05 [-0.44, 0.93] | 8% |
| Tool coverage 25% | transformer deferrer (task folds) | 0.19 [-0.51, 0.98] | 5% | 0.20 [-0.49, 0.98] | 6% |
| Unseen domain: banking | stronger monitor, least confident | 0.03 [-0.25, 0.37] | 24% | 0.02 [-0.25, 0.25] | 23% |
| Unseen domain: banking | big mpnet monitor, least confident | 0.23 [-0.50, 0.86] | 18% | 0.08 [-0.38, 0.51] | 29% |
| Unseen domain: banking | transformer and LSTM disagree most | 0.13 [-0.25, 0.50] | 16% | 0.08 [-0.13, 0.37] | 14% |
| Unseen domain: banking | transformer + LSTM, unsure or disagreeing | 0.05 [-0.25, 0.38] | 26% | 0.05 [-0.13, 0.37] | 20% |
| Unseen domain: banking | 3-seed ensemble, least confident | -0.01 [-0.26, 0.37] | 38% | -0.05 [-0.37, 0.25] | 41% |
| Unseen domain: banking | seeds disagree most | 0.08 [-0.25, 0.38] | 24% | 0.04 [-0.25, 0.37] | 21% |
| Unseen domain: banking | 3-seed ensemble, unsure or disagreeing | 0.03 [-0.25, 0.38] | 28% | -0.00 [-0.25, 0.25] | 28% |
| Unseen domain: banking | LightGBM deferrer (task folds) | 0.10 [-0.25, 0.38] | 12% | -0.04 [-0.25, 0.25] | 40% |
| Unseen domain: banking | LightGBM deferrer (domain folds) | 0.26 [-0.13, 0.87] | 10% | 0.12 [-0.25, 0.62] | 14% |
| Unseen domain: banking | transformer deferrer (task folds) | 0.25 [-0.25, 0.87] | 12% | 0.26 [-0.25, 0.99] | 10% |
| Unseen domain: banking | transformer deferrer (domain folds) | 0.20 [-0.13, 0.87] | 10% | 0.23 [-0.25, 0.99] | 11% |
| Unseen domain: coding | stronger monitor, least confident | -0.07 [-0.37, 0.25] | 50% | -0.12 [-0.49, 0.12] | 66% |
| Unseen domain: coding | big mpnet monitor, least confident | 0.56 [-0.13, 1.36] | 6% | 0.57 [0.00, 1.37] | 2% |
| Unseen domain: coding | transformer and LSTM disagree most | -0.03 [-0.38, 0.37] | 42% | -0.04 [-0.37, 0.37] | 44% |
| Unseen domain: coding | transformer + LSTM, unsure or disagreeing | -0.08 [-0.49, 0.25] | 50% | -0.09 [-0.49, 0.13] | 55% |
| Unseen domain: coding | 3-seed ensemble, least confident | -0.10 [-0.50, 0.13] | 52% | -0.09 [-0.37, 0.12] | 54% |
| Unseen domain: coding | seeds disagree most | -0.03 [-0.49, 0.25] | 39% | -0.06 [-0.37, 0.13] | 42% |
| Unseen domain: coding | 3-seed ensemble, unsure or disagreeing | -0.08 [-0.50, 0.25] | 50% | -0.06 [-0.37, 0.12] | 44% |
| Unseen domain: coding | LightGBM deferrer (task folds) | 0.13 [-0.12, 0.62] | 14% | 0.08 [-0.12, 0.37] | 11% |
| Unseen domain: coding | LightGBM deferrer (domain folds) | 0.18 [-0.25, 0.74] | 16% | 0.09 [-0.25, 0.73] | 21% |
| Unseen domain: coding | transformer deferrer (task folds) | 0.31 [-0.25, 1.13] | 13% | 0.36 [-0.12, 1.22] | 6% |
| Unseen domain: coding | transformer deferrer (domain folds) | 0.35 [-0.12, 1.23] | 8% | 0.35 [-0.12, 1.23] | 3% |
| Unseen domain: email | stronger monitor, least confident | -0.01 [-0.36, 0.24] | 28% | -0.01 [-0.25, 0.24] | 24% |
| Unseen domain: email | big mpnet monitor, least confident | 0.33 [-0.25, 0.97] | 10% | 0.21 [-0.36, 0.73] | 14% |
| Unseen domain: email | transformer and LSTM disagree most | 0.02 [-0.25, 0.36] | 20% | -0.01 [-0.25, 0.24] | 26% |
| Unseen domain: email | transformer + LSTM, unsure or disagreeing | -0.00 [-0.37, 0.24] | 25% | -0.02 [-0.36, 0.24] | 26% |
| Unseen domain: email | 3-seed ensemble, least confident | 0.01 [-0.24, 0.25] | 26% | -0.01 [-0.24, 0.24] | 28% |
| Unseen domain: email | seeds disagree most | 0.04 [-0.13, 0.36] | 12% | -0.00 [-0.25, 0.25] | 22% |
| Unseen domain: email | 3-seed ensemble, unsure or disagreeing | 0.00 [-0.25, 0.25] | 29% | -0.03 [-0.24, 0.24] | 30% |
| Unseen domain: email | LightGBM deferrer (task folds) | 0.03 [-0.12, 0.24] | 8% | 0.02 [-0.12, 0.24] | 9% |
| Unseen domain: email | LightGBM deferrer (domain folds) | 0.06 [-0.12, 0.36] | 4% | 0.04 [-0.12, 0.25] | 10% |
| Unseen domain: email | transformer deferrer (task folds) | 0.07 [-0.12, 0.37] | 6% | 0.09 [-0.12, 0.37] | 3% |
| Unseen domain: email | transformer deferrer (domain folds) | 0.08 [-0.12, 0.37] | 4% | 0.10 [0.00, 0.48] | 2% |
| Unseen domain: medical | stronger monitor, least confident | -0.11 [-0.74, 0.25] | 52% | -0.06 [-0.49, 0.13] | 33% |
| Unseen domain: medical | big mpnet monitor, least confident | 0.89 [0.00, 1.83] | 2% | 0.41 [-0.25, 0.99] | 6% |
| Unseen domain: medical | transformer and LSTM disagree most | 0.01 [-0.49, 0.37] | 26% | -0.03 [-0.50, 0.25] | 31% |
| Unseen domain: medical | transformer + LSTM, unsure or disagreeing | -0.08 [-0.74, 0.25] | 42% | -0.06 [-0.62, 0.25] | 35% |
| Unseen domain: medical | 3-seed ensemble, least confident | -0.05 [-0.63, 0.49] | 50% | -0.07 [-0.49, 0.12] | 42% |
| Unseen domain: medical | seeds disagree most | -0.07 [-0.62, 0.25] | 38% | -0.09 [-0.50, 0.12] | 46% |
| Unseen domain: medical | 3-seed ensemble, unsure or disagreeing | -0.09 [-0.86, 0.49] | 52% | -0.09 [-0.49, 0.12] | 46% |
| Unseen domain: medical | LightGBM deferrer (task folds) | -0.02 [-0.25, 0.25] | 33% | -0.01 [-0.24, 0.12] | 24% |
| Unseen domain: medical | LightGBM deferrer (domain folds) | 0.08 [-0.12, 0.25] | 4% | 0.03 [-0.25, 0.25] | 16% |
| Unseen domain: medical | transformer deferrer (task folds) | 0.15 [-0.13, 0.62] | 8% | 0.17 [-0.12, 0.61] | 6% |
| Unseen domain: medical | transformer deferrer (domain folds) | 0.08 [-0.13, 0.49] | 10% | 0.11 [-0.12, 0.51] | 10% |
| Unseen domain: web | stronger monitor, least confident | -0.19 [-0.74, 0.25] | 67% | -0.13 [-0.63, 0.25] | 60% |
| Unseen domain: web | big mpnet monitor, least confident | 0.26 [-0.63, 1.25] | 30% | 0.21 [-0.51, 1.02] | 27% |
| Unseen domain: web | transformer and LSTM disagree most | 0.11 [-0.25, 0.50] | 17% | 0.12 [-0.37, 0.64] | 24% |
| Unseen domain: web | transformer + LSTM, unsure or disagreeing | -0.17 [-0.63, 0.25] | 63% | -0.10 [-0.51, 0.25] | 50% |
| Unseen domain: web | 3-seed ensemble, least confident | 0.01 [-0.63, 0.88] | 50% | -0.01 [-0.50, 0.50] | 39% |
| Unseen domain: web | seeds disagree most | 0.12 [-0.25, 0.50] | 9% | 0.16 [-0.25, 0.62] | 12% |
| Unseen domain: web | 3-seed ensemble, unsure or disagreeing | 0.01 [-0.64, 0.76] | 46% | 0.03 [-0.50, 0.50] | 31% |
| Unseen domain: web | LightGBM deferrer (task folds) | -0.03 [-0.37, 0.26] | 39% | -0.02 [-0.38, 0.38] | 38% |
| Unseen domain: web | LightGBM deferrer (domain folds) | 0.11 [-0.25, 0.50] | 14% | 0.09 [-0.25, 0.62] | 19% |
| Unseen domain: web | transformer deferrer (task folds) | -0.02 [-0.38, 0.38] | 36% | 0.04 [-0.37, 0.51] | 34% |
| Unseen domain: web | transformer deferrer (domain folds) | 0.05 [-0.25, 0.38] | 24% | 0.10 [-0.25, 0.75] | 24% |
| Unseen domains (mean) | stronger monitor, least confident | -0.07 [-0.25, 0.10] | 78% | -0.06 [-0.20, 0.07] | 81% |
| Unseen domains (mean) | big mpnet monitor, least confident | 0.45 [0.10, 0.79] | 0% | 0.30 [0.05, 0.62] | 2% |
| Unseen domains (mean) | transformer and LSTM disagree most | 0.05 [-0.12, 0.25] | 28% | 0.02 [-0.12, 0.17] | 34% |
| Unseen domains (mean) | transformer + LSTM, unsure or disagreeing | -0.06 [-0.27, 0.12] | 68% | -0.04 [-0.20, 0.08] | 74% |
| Unseen domains (mean) | 3-seed ensemble, least confident | -0.03 [-0.22, 0.21] | 60% | -0.05 [-0.20, 0.10] | 70% |
| Unseen domains (mean) | seeds disagree most | 0.03 [-0.15, 0.17] | 30% | 0.01 [-0.15, 0.15] | 38% |
| Unseen domains (mean) | 3-seed ensemble, unsure or disagreeing | -0.02 [-0.24, 0.23] | 57% | -0.03 [-0.20, 0.08] | 57% |
| Unseen domains (mean) | LightGBM deferrer (task folds) | 0.04 [-0.07, 0.15] | 28% | 0.00 [-0.10, 0.12] | 50% |
| Unseen domains (mean) | LightGBM deferrer (domain folds) | 0.14 [-0.05, 0.37] | 13% | 0.07 [-0.08, 0.30] | 22% |
| Unseen domains (mean) | transformer deferrer (task folds) | 0.15 [-0.05, 0.49] | 12% | 0.18 [-0.05, 0.55] | 6% |
| Unseen domains (mean) | transformer deferrer (domain folds) | 0.15 [-0.03, 0.45] | 6% | 0.18 [-0.05, 0.52] | 6% |

## Attack rate natural: missed attacks left per 1,000 calls

Review budget 2% (no deferral in the first column):

| setting | no deferral (main monitor) | least confident | most novel | stronger monitor, least confident | big mpnet monitor, least confident | transformer and LSTM disagree most | transformer + LSTM, unsure or disagreeing | 3-seed ensemble, least confident | seeds disagree most | 3-seed ensemble, unsure or disagreeing | LightGBM deferrer (task folds) | LightGBM deferrer (domain folds) | transformer deferrer (task folds) | transformer deferrer (domain folds) | perfect (upper bound) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Task-disjoint test | 2.31 [1.44, 3.15] | 0.76 [0.31, 1.31] | 2.12 [1.23, 2.94] | 0.53 [0.10, 0.93] | 2.46 [1.45, 3.58] | 0.82 [0.30, 1.42] | 0.73 [0.30, 1.24] | 0.98 [0.41, 1.64] | 1.08 [0.51, 1.75] | 0.96 [0.41, 1.63] | 0.69 [0.21, 1.22] | — | 0.90 [0.31, 1.53] | — | 0.00 [0.00, 0.00] |
| Tool coverage 100% | 4.89 [0.79, 12.12] | 2.65 [0.38, 7.87] | 4.08 [0.40, 9.47] | 2.50 [0.39, 6.95] | 5.11 [0.00, 10.96] | 2.67 [0.38, 6.94] | 2.49 [0.38, 6.81] | 1.93 [0.35, 4.66] | 2.15 [0.35, 5.08] | 1.87 [0.34, 4.21] | 3.39 [0.38, 10.04] | — | 3.47 [0.34, 10.07] | — | 0.00 [0.00, 0.00] |
| Tool coverage 75% | 5.67 [1.58, 13.46] | 3.06 [0.35, 8.54] | 5.05 [1.14, 12.76] | 3.15 [0.70, 7.85] | 6.45 [0.35, 13.53] | 3.38 [0.79, 8.17] | 3.14 [0.69, 7.86] | 2.67 [0.68, 6.06] | 2.64 [0.68, 5.72] | 2.54 [0.39, 6.04] | 4.11 [0.39, 11.54] | — | 3.94 [0.38, 11.46] | — | 0.00 [0.00, 0.00] |
| Tool coverage 50% | 6.13 [1.54, 12.43] | 3.24 [0.00, 8.03] | 5.91 [1.54, 12.43] | 3.41 [0.76, 7.87] | 5.56 [0.33, 11.86] | 3.82 [0.78, 8.62] | 3.59 [0.78, 7.92] | 2.65 [0.38, 6.08] | 2.87 [0.70, 5.96] | 2.87 [0.39, 6.22] | 4.45 [0.77, 11.08] | — | 4.71 [0.40, 11.50] | — | 0.00 [0.00, 0.00] |
| Tool coverage 25% | 5.43 [2.74, 8.45] | 2.83 [1.06, 5.03] | 5.00 [2.40, 7.69] | 3.38 [1.39, 5.78] | 6.81 [1.03, 13.01] | 3.39 [1.17, 5.66] | 3.53 [1.38, 6.21] | 3.62 [0.71, 6.76] | 3.61 [0.72, 6.92] | 3.58 [0.72, 6.86] | 3.74 [1.19, 6.64] | — | 3.97 [1.52, 7.02] | — | 0.00 [0.00, 0.00] |
| Unseen domain: banking | 6.99 [4.72, 9.45] | 4.21 [2.93, 5.73] | 6.72 [4.50, 9.03] | 4.04 [3.00, 5.50] | 4.93 [3.64, 6.31] | 4.55 [3.38, 5.91] | 4.30 [3.09, 5.58] | 4.06 [2.94, 5.41] | 4.38 [3.17, 5.73] | 4.18 [2.94, 5.62] | 4.28 [2.94, 5.67] | 5.88 [3.28, 9.33] | 4.76 [2.92, 6.80] | 5.14 [3.28, 7.69] | 0.00 [0.00, 0.00] |
| Unseen domain: coding | 14.26 [9.25, 21.02] | 9.13 [5.91, 13.54] | 13.10 [8.07, 19.41] | 6.93 [5.29, 9.34] | 12.57 [10.94, 14.54] | 7.29 [5.47, 9.76] | 6.94 [5.13, 9.40] | 8.21 [6.65, 10.06] | 8.83 [7.20, 10.51] | 8.40 [6.87, 10.19] | 9.67 [6.65, 13.55] | 11.25 [6.91, 15.76] | 9.91 [5.91, 14.56] | 10.44 [6.47, 16.33] | 0.07 [0.00, 1.02] |
| Unseen domain: email | 3.08 [0.53, 6.01] | 2.09 [0.47, 4.01] | 3.01 [0.53, 5.93] | 1.81 [0.81, 3.05] | 5.38 [3.86, 6.90] | 2.03 [0.40, 4.07] | 1.91 [0.73, 3.40] | 2.05 [1.07, 3.15] | 2.41 [0.53, 4.66] | 1.91 [1.00, 2.94] | 2.32 [0.53, 4.07] | 2.76 [0.40, 5.47] | 2.71 [0.53, 5.30] | 2.80 [0.53, 5.29] | 0.00 [0.00, 0.00] |
| Unseen domain: medical | 6.96 [1.42, 10.94] | 3.89 [0.87, 6.48] | 6.79 [1.42, 10.53] | 2.53 [1.28, 3.85] | 18.46 [16.22, 21.00] | 3.63 [1.36, 5.62] | 2.92 [1.41, 4.37] | 3.20 [2.38, 4.12] | 3.20 [1.41, 4.91] | 2.86 [1.77, 3.71] | 3.29 [0.61, 5.90] | 6.10 [1.41, 10.00] | 5.01 [0.95, 8.17] | 4.85 [0.87, 8.05] | 0.00 [0.00, 0.00] |
| Unseen domain: web | 10.91 [4.80, 15.60] | 8.78 [3.96, 12.60] | 10.28 [4.52, 14.84] | 6.66 [4.09, 8.93] | 11.25 [9.10, 13.46] | 9.61 [4.21, 13.72] | 6.85 [3.98, 9.06] | 8.66 [7.14, 10.27] | 9.36 [4.80, 13.06] | 8.37 [6.95, 9.93] | 8.25 [3.71, 11.83] | 9.98 [4.43, 15.11] | 8.42 [3.23, 12.85] | 9.09 [3.84, 13.35] | 0.00 [0.00, 0.00] |
| Unseen domains (mean) | 8.44 [5.43, 11.70] | 5.62 [3.72, 7.62] | 7.98 [5.16, 11.07] | 4.39 [3.34, 5.34] | 10.52 [9.61, 11.50] | 5.42 [3.68, 7.01] | 4.58 [3.42, 5.73] | 5.24 [4.60, 5.83] | 5.63 [4.16, 6.96] | 5.14 [4.58, 5.79] | 5.56 [3.97, 7.12] | 7.19 [4.94, 10.34] | 6.16 [3.90, 8.83] | 6.46 [4.14, 9.34] | 0.01 [0.00, 0.20] |

Review budget 5% (no deferral in the first column):

| setting | no deferral (main monitor) | least confident | most novel | stronger monitor, least confident | big mpnet monitor, least confident | transformer and LSTM disagree most | transformer + LSTM, unsure or disagreeing | 3-seed ensemble, least confident | seeds disagree most | 3-seed ensemble, unsure or disagreeing | LightGBM deferrer (task folds) | LightGBM deferrer (domain folds) | transformer deferrer (task folds) | transformer deferrer (domain folds) | perfect (upper bound) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Task-disjoint test | 2.31 [1.44, 3.15] | 0.22 [0.00, 0.52] | 2.12 [1.23, 2.94] | 0.18 [0.00, 0.52] | 1.48 [0.81, 2.34] | 0.38 [0.00, 0.81] | 0.19 [0.00, 0.42] | 0.28 [0.00, 0.70] | 0.37 [0.10, 0.72] | 0.28 [0.00, 0.72] | 0.22 [0.00, 0.52] | — | 0.67 [0.20, 1.22] | — | 0.00 [0.00, 0.00] |
| Tool coverage 100% | 4.89 [0.79, 12.12] | 1.60 [0.34, 4.30] | 3.59 [0.40, 8.07] | 1.28 [0.33, 2.92] | 3.02 [0.00, 7.43] | 1.60 [0.36, 3.60] | 1.58 [0.35, 3.42] | 1.01 [0.00, 2.66] | 0.95 [0.00, 2.45] | 1.00 [0.00, 2.66] | 1.57 [0.00, 5.01] | — | 2.66 [0.00, 9.31] | — | 0.00 [0.00, 0.00] |
| Tool coverage 75% | 5.67 [1.58, 13.46] | 1.95 [0.00, 5.39] | 4.98 [1.14, 12.76] | 1.39 [0.00, 4.25] | 4.10 [0.00, 9.37] | 1.74 [0.00, 4.25] | 1.64 [0.00, 3.95] | 1.09 [0.00, 2.66] | 1.15 [0.00, 3.21] | 1.09 [0.00, 2.66] | 2.10 [0.00, 6.74] | — | 3.10 [0.00, 9.63] | — | 0.00 [0.00, 0.00] |
| Tool coverage 50% | 6.13 [1.54, 12.43] | 1.90 [0.00, 5.37] | 5.41 [1.17, 12.43] | 1.43 [0.00, 4.68] | 3.05 [0.00, 8.17] | 1.78 [0.00, 5.32] | 1.68 [0.00, 5.01] | 1.25 [0.00, 3.55] | 1.39 [0.00, 3.84] | 1.30 [0.00, 3.58] | 2.14 [0.00, 6.39] | — | 3.60 [0.39, 9.91] | — | 0.00 [0.00, 0.00] |
| Tool coverage 25% | 5.43 [2.74, 8.45] | 1.78 [0.35, 3.81] | 4.65 [2.01, 7.69] | 1.46 [0.00, 3.47] | 3.56 [0.35, 8.40] | 1.80 [0.00, 4.44] | 1.61 [0.00, 3.75] | 1.57 [0.00, 3.41] | 1.83 [0.35, 3.72] | 1.69 [0.00, 3.68] | 2.05 [0.34, 4.30] | — | 3.26 [0.77, 6.34] | — | 0.00 [0.00, 0.00] |
| Unseen domain: banking | 6.99 [4.72, 9.45] | 2.39 [1.44, 3.56] | 6.27 [4.22, 8.42] | 2.42 [1.63, 3.36] | 2.64 [1.88, 3.42] | 2.68 [1.81, 3.61] | 2.50 [1.61, 3.65] | 1.96 [1.19, 2.75] | 2.38 [1.54, 3.30] | 2.14 [1.21, 2.93] | 2.24 [1.30, 3.46] | 3.54 [1.80, 6.24] | 3.55 [1.98, 5.48] | 3.70 [2.13, 5.72] | 0.00 [0.00, 0.00] |
| Unseen domain: coding | 14.26 [9.25, 21.02] | 4.91 [3.23, 7.07] | 11.86 [7.21, 17.92] | 3.68 [2.55, 5.32] | 10.01 [8.58, 11.77] | 4.33 [3.28, 5.80] | 3.82 [2.65, 5.45] | 4.27 [3.23, 5.42] | 4.33 [3.36, 5.47] | 4.27 [3.25, 5.42] | 5.63 [4.07, 7.93] | 6.60 [3.77, 10.35] | 7.57 [4.30, 12.29] | 7.73 [4.30, 13.67] | 0.00 [0.00, 0.00] |
| Unseen domain: email | 3.08 [0.53, 6.01] | 1.46 [0.34, 2.82] | 2.99 [0.53, 5.87] | 1.39 [0.60, 2.49] | 3.71 [2.54, 5.22] | 1.35 [0.27, 2.74] | 1.35 [0.46, 2.49] | 1.37 [0.59, 2.47] | 1.39 [0.53, 2.48] | 1.21 [0.53, 2.07] | 1.73 [0.40, 3.39] | 1.90 [0.27, 3.60] | 2.42 [0.47, 4.89] | 2.47 [0.53, 4.48] | 0.00 [0.00, 0.00] |
| Unseen domain: medical | 6.96 [1.42, 10.94] | 2.04 [0.47, 3.75] | 6.31 [1.23, 10.13] | 1.30 [0.54, 2.14] | 10.02 [8.36, 12.05] | 1.67 [0.75, 2.70] | 1.43 [0.61, 2.39] | 1.26 [0.74, 1.76] | 1.22 [0.74, 1.76] | 1.19 [0.73, 1.77] | 1.79 [0.33, 3.87] | 2.84 [0.74, 4.74] | 3.69 [0.61, 7.23] | 3.51 [0.54, 6.33] | 0.00 [0.00, 0.00] |
| Unseen domain: web | 10.91 [4.80, 15.60] | 6.48 [3.46, 9.06] | 9.55 [3.86, 14.17] | 4.74 [2.85, 6.35] | 8.36 [6.51, 10.07] | 7.03 [3.18, 10.29] | 5.05 [2.88, 7.03] | 5.85 [4.63, 7.11] | 6.45 [4.80, 8.24] | 6.05 [4.93, 7.50] | 6.10 [2.37, 9.11] | 7.56 [4.01, 11.86] | 6.54 [2.51, 9.56] | 7.24 [2.88, 11.84] | 0.00 [0.00, 0.00] |
| Unseen domains (mean) | 8.44 [5.43, 11.70] | 3.46 [2.35, 4.49] | 7.40 [4.71, 10.33] | 2.71 [2.05, 3.39] | 6.95 [6.27, 7.74] | 3.41 [2.41, 4.25] | 2.83 [2.07, 3.59] | 2.94 [2.53, 3.43] | 3.15 [2.58, 3.80] | 2.97 [2.54, 3.40] | 3.50 [2.47, 4.36] | 4.49 [3.15, 6.73] | 4.75 [2.94, 6.99] | 4.93 [3.02, 7.57] | 0.00 [0.00, 0.00] |

Learned policies vs. least confident at attack rate natural: difference in missed attacks left per 1,000 calls (negative = the learned one leaves fewer), with the share of resamples where it is better.

| setting | policy | 2% budget | better in | 5% budget | better in |
|---|---|---|---|---|---|
| Task-disjoint test | stronger monitor, least confident | -0.24 [-0.61, 0.10] | 88% | -0.05 [-0.31, 0.31] | 51% |
| Task-disjoint test | big mpnet monitor, least confident | 1.70 [0.81, 2.76] | 0% | 1.25 [0.61, 2.13] | 0% |
| Task-disjoint test | transformer and LSTM disagree most | 0.06 [-0.41, 0.61] | 32% | 0.16 [-0.20, 0.61] | 13% |
| Task-disjoint test | transformer + LSTM, unsure or disagreeing | -0.03 [-0.41, 0.31] | 48% | -0.03 [-0.31, 0.21] | 44% |
| Task-disjoint test | 3-seed ensemble, least confident | 0.22 [0.00, 0.61] | 2% | 0.06 [-0.21, 0.41] | 27% |
| Task-disjoint test | seeds disagree most | 0.32 [0.00, 0.72] | 0% | 0.15 [0.00, 0.41] | 0% |
| Task-disjoint test | 3-seed ensemble, unsure or disagreeing | 0.20 [-0.10, 0.71] | 4% | 0.05 [-0.31, 0.40] | 26% |
| Task-disjoint test | LightGBM deferrer (task folds) | -0.07 [-0.51, 0.41] | 52% | 0.00 [-0.10, 0.10] | 6% |
| Task-disjoint test | transformer deferrer (task folds) | 0.14 [-0.21, 0.61] | 19% | 0.45 [0.10, 0.92] | 0% |
| Tool coverage 100% | stronger monitor, least confident | -0.15 [-1.78, 0.80] | 27% | -0.33 [-2.10, 0.67] | 40% |
| Tool coverage 100% | big mpnet monitor, least confident | 2.46 [-1.71, 7.22] | 32% | 1.41 [-1.05, 4.98] | 26% |
| Tool coverage 100% | transformer and LSTM disagree most | 0.02 [-2.54, 1.37] | 20% | 0.00 [-1.46, 1.39] | 22% |
| Tool coverage 100% | transformer + LSTM, unsure or disagreeing | -0.16 [-2.10, 0.77] | 22% | -0.03 [-1.76, 1.39] | 22% |
| Tool coverage 100% | 3-seed ensemble, least confident | -0.71 [-3.51, 0.36] | 46% | -0.59 [-2.52, 0.68] | 46% |
| Tool coverage 100% | seeds disagree most | -0.50 [-3.89, 1.17] | 44% | -0.66 [-2.92, 0.67] | 50% |
| Tool coverage 100% | 3-seed ensemble, unsure or disagreeing | -0.78 [-3.61, 0.36] | 50% | -0.60 [-2.54, 0.68] | 47% |
| Tool coverage 100% | LightGBM deferrer (task folds) | 0.75 [0.00, 3.20] | 1% | -0.03 [-1.90, 1.44] | 34% |
| Tool coverage 100% | transformer deferrer (task folds) | 0.83 [-1.04, 3.61] | 12% | 1.06 [-1.15, 4.93] | 22% |
| Tool coverage 75% | stronger monitor, least confident | 0.09 [-1.43, 1.57] | 33% | -0.56 [-2.54, 0.38] | 68% |
| Tool coverage 75% | big mpnet monitor, least confident | 3.39 [-1.99, 8.68] | 32% | 2.15 [-2.01, 6.77] | 26% |
| Tool coverage 75% | transformer and LSTM disagree most | 0.32 [-2.08, 2.00] | 24% | -0.22 [-2.54, 1.11] | 40% |
| Tool coverage 75% | transformer + LSTM, unsure or disagreeing | 0.08 [-1.73, 1.76] | 34% | -0.32 [-2.54, 1.00] | 44% |
| Tool coverage 75% | 3-seed ensemble, least confident | -0.39 [-3.21, 2.33] | 57% | -0.86 [-3.46, 0.00] | 59% |
| Tool coverage 75% | seeds disagree most | -0.42 [-4.28, 2.34] | 52% | -0.80 [-3.60, 0.00] | 59% |
| Tool coverage 75% | 3-seed ensemble, unsure or disagreeing | -0.52 [-3.46, 2.36] | 62% | -0.87 [-3.62, 0.00] | 60% |
| Tool coverage 75% | LightGBM deferrer (task folds) | 1.05 [-0.74, 4.05] | 10% | 0.15 [-1.12, 2.49] | 34% |
| Tool coverage 75% | transformer deferrer (task folds) | 0.88 [-0.80, 3.54] | 17% | 1.15 [-1.34, 4.98] | 16% |
| Tool coverage 50% | stronger monitor, least confident | 0.18 [-1.43, 2.31] | 32% | -0.47 [-1.80, 0.36] | 66% |
| Tool coverage 50% | big mpnet monitor, least confident | 2.32 [-3.33, 8.08] | 32% | 1.14 [-2.08, 4.59] | 27% |
| Tool coverage 50% | transformer and LSTM disagree most | 0.58 [-1.10, 2.13] | 16% | -0.12 [-1.43, 1.02] | 40% |
| Tool coverage 50% | transformer + LSTM, unsure or disagreeing | 0.35 [-1.42, 2.02] | 26% | -0.22 [-1.43, 0.72] | 40% |
| Tool coverage 50% | 3-seed ensemble, least confident | -0.58 [-3.22, 1.21] | 60% | -0.66 [-2.87, 0.34] | 50% |
| Tool coverage 50% | seeds disagree most | -0.37 [-3.94, 1.97] | 48% | -0.51 [-2.77, 0.69] | 46% |
| Tool coverage 50% | 3-seed ensemble, unsure or disagreeing | -0.37 [-3.28, 1.58] | 52% | -0.61 [-2.82, 0.00] | 48% |
| Tool coverage 50% | LightGBM deferrer (task folds) | 1.21 [-0.77, 4.27] | 8% | 0.24 [-1.69, 2.15] | 33% |
| Tool coverage 50% | transformer deferrer (task folds) | 1.47 [-0.39, 4.27] | 4% | 1.69 [-0.41, 5.33] | 4% |
| Tool coverage 25% | stronger monitor, least confident | 0.55 [-1.08, 2.12] | 19% | -0.33 [-2.37, 1.08] | 41% |
| Tool coverage 25% | big mpnet monitor, least confident | 3.97 [-1.73, 9.77] | 20% | 1.77 [-2.37, 7.02] | 28% |
| Tool coverage 25% | transformer and LSTM disagree most | 0.55 [-1.17, 2.40] | 19% | 0.02 [-2.31, 2.06] | 40% |
| Tool coverage 25% | transformer + LSTM, unsure or disagreeing | 0.70 [-1.05, 2.41] | 12% | -0.18 [-2.74, 1.76] | 36% |
| Tool coverage 25% | 3-seed ensemble, least confident | 0.79 [-1.77, 2.77] | 28% | -0.22 [-1.73, 0.79] | 46% |
| Tool coverage 25% | seeds disagree most | 0.78 [-1.73, 3.03] | 26% | 0.05 [-0.76, 1.19] | 25% |
| Tool coverage 25% | 3-seed ensemble, unsure or disagreeing | 0.74 [-1.76, 2.75] | 27% | -0.09 [-1.08, 1.01] | 36% |
| Tool coverage 25% | LightGBM deferrer (task folds) | 0.91 [-0.79, 2.82] | 7% | 0.26 [-1.39, 2.17] | 35% |
| Tool coverage 25% | transformer deferrer (task folds) | 1.14 [-0.79, 3.21] | 6% | 1.48 [-0.67, 4.33] | 8% |
| Unseen domain: banking | stronger monitor, least confident | -0.17 [-1.22, 0.77] | 56% | 0.03 [-0.78, 0.96] | 43% |
| Unseen domain: banking | big mpnet monitor, least confident | 0.72 [-1.03, 1.96] | 16% | 0.25 [-0.86, 1.37] | 36% |
| Unseen domain: banking | transformer and LSTM disagree most | 0.34 [-0.60, 1.12] | 19% | 0.29 [-0.51, 1.03] | 23% |
| Unseen domain: banking | transformer + LSTM, unsure or disagreeing | 0.10 [-0.94, 1.03] | 38% | 0.11 [-0.77, 1.13] | 42% |
| Unseen domain: banking | 3-seed ensemble, least confident | -0.14 [-0.96, 0.52] | 57% | -0.43 [-1.30, 0.34] | 82% |
| Unseen domain: banking | seeds disagree most | 0.17 [-0.78, 1.03] | 31% | -0.01 [-0.87, 0.78] | 43% |
| Unseen domain: banking | 3-seed ensemble, unsure or disagreeing | -0.02 [-0.85, 0.69] | 48% | -0.25 [-1.12, 0.60] | 72% |
| Unseen domain: banking | LightGBM deferrer (task folds) | 0.07 [-0.69, 0.95] | 42% | -0.15 [-1.04, 0.68] | 59% |
| Unseen domain: banking | LightGBM deferrer (domain folds) | 1.68 [-0.00, 4.13] | 2% | 1.15 [-0.77, 3.78] | 30% |
| Unseen domain: banking | transformer deferrer (task folds) | 0.56 [-0.60, 1.79] | 15% | 1.16 [-0.17, 3.28] | 6% |
| Unseen domain: banking | transformer deferrer (domain folds) | 0.94 [-0.09, 2.22] | 4% | 1.32 [-0.52, 3.33] | 18% |
| Unseen domain: coding | stronger monitor, least confident | -2.20 [-4.95, -0.13] | 99% | -1.23 [-2.43, -0.40] | 100% |
| Unseen domain: coding | big mpnet monitor, least confident | 3.44 [-0.89, 7.09] | 9% | 5.10 [2.63, 7.29] | 0% |
| Unseen domain: coding | transformer and LSTM disagree most | -1.84 [-4.41, 0.00] | 96% | -0.58 [-2.16, 0.68] | 72% |
| Unseen domain: coding | transformer + LSTM, unsure or disagreeing | -2.19 [-4.79, -0.40] | 100% | -1.09 [-2.51, -0.13] | 99% |
| Unseen domain: coding | 3-seed ensemble, least confident | -0.92 [-4.66, 2.02] | 58% | -0.64 [-2.61, 0.47] | 60% |
| Unseen domain: coding | seeds disagree most | -0.30 [-3.80, 2.51] | 40% | -0.58 [-2.52, 0.54] | 55% |
| Unseen domain: coding | 3-seed ensemble, unsure or disagreeing | -0.73 [-4.56, 2.17] | 49% | -0.65 [-2.63, 0.41] | 58% |
| Unseen domain: coding | LightGBM deferrer (task folds) | 0.55 [-0.61, 1.62] | 22% | 0.72 [-0.47, 2.62] | 19% |
| Unseen domain: coding | LightGBM deferrer (domain folds) | 2.12 [0.47, 3.99] | 0% | 1.69 [-0.20, 3.66] | 6% |
| Unseen domain: coding | transformer deferrer (task folds) | 0.78 [-0.53, 2.41] | 17% | 2.66 [0.27, 5.80] | 0% |
| Unseen domain: coding | transformer deferrer (domain folds) | 1.31 [-0.20, 3.49] | 5% | 2.82 [0.47, 6.71] | 0% |
| Unseen domain: email | stronger monitor, least confident | -0.28 [-1.54, 1.13] | 66% | -0.07 [-1.00, 0.86] | 55% |
| Unseen domain: email | big mpnet monitor, least confident | 3.29 [1.15, 5.52] | 0% | 2.25 [0.80, 3.94] | 0% |
| Unseen domain: email | transformer and LSTM disagree most | -0.06 [-0.80, 1.00] | 58% | -0.10 [-0.86, 0.47] | 54% |
| Unseen domain: email | transformer + LSTM, unsure or disagreeing | -0.18 [-1.14, 0.88] | 60% | -0.11 [-0.94, 0.67] | 55% |
| Unseen domain: email | 3-seed ensemble, least confident | -0.04 [-1.34, 1.74] | 66% | -0.08 [-0.81, 1.07] | 66% |
| Unseen domain: email | seeds disagree most | 0.32 [-0.40, 1.47] | 17% | -0.06 [-0.87, 0.67] | 50% |
| Unseen domain: email | 3-seed ensemble, unsure or disagreeing | -0.18 [-1.48, 1.41] | 66% | -0.24 [-1.01, 0.81] | 66% |
| Unseen domain: email | LightGBM deferrer (task folds) | 0.23 [-0.20, 0.73] | 10% | 0.27 [-0.33, 0.93] | 18% |
| Unseen domain: email | LightGBM deferrer (domain folds) | 0.67 [-0.20, 1.75] | 13% | 0.44 [-0.20, 1.27] | 15% |
| Unseen domain: email | transformer deferrer (task folds) | 0.62 [0.00, 1.76] | 0% | 0.96 [0.07, 2.43] | 0% |
| Unseen domain: email | transformer deferrer (domain folds) | 0.72 [0.07, 1.68] | 0% | 1.01 [0.13, 2.08] | 0% |
| Unseen domain: medical | stronger monitor, least confident | -1.36 [-3.18, 0.81] | 67% | -0.74 [-2.19, 0.47] | 74% |
| Unseen domain: medical | big mpnet monitor, least confident | 14.57 [10.94, 18.94] | 0% | 7.98 [5.48, 10.69] | 0% |
| Unseen domain: medical | transformer and LSTM disagree most | -0.26 [-1.95, 0.95] | 57% | -0.37 [-1.91, 0.54] | 56% |
| Unseen domain: medical | transformer + LSTM, unsure or disagreeing | -0.97 [-2.79, 1.01] | 66% | -0.61 [-2.10, 0.34] | 70% |
| Unseen domain: medical | 3-seed ensemble, least confident | -0.68 [-3.09, 2.57] | 66% | -0.78 [-2.45, 0.75] | 66% |
| Unseen domain: medical | seeds disagree most | -0.69 [-2.69, 1.01] | 64% | -0.83 [-2.52, 0.74] | 66% |
| Unseen domain: medical | 3-seed ensemble, unsure or disagreeing | -1.02 [-3.38, 2.18] | 66% | -0.86 [-2.68, 0.68] | 67% |
| Unseen domain: medical | LightGBM deferrer (task folds) | -0.59 [-1.70, 0.14] | 90% | -0.25 [-0.88, 0.27] | 78% |
| Unseen domain: medical | LightGBM deferrer (domain folds) | 2.22 [0.34, 4.56] | 0% | 0.80 [-0.07, 2.32] | 4% |
| Unseen domain: medical | transformer deferrer (task folds) | 1.12 [-0.07, 2.57] | 4% | 1.65 [-0.14, 4.80] | 4% |
| Unseen domain: medical | transformer deferrer (domain folds) | 0.96 [-0.20, 2.35] | 15% | 1.46 [-0.20, 3.91] | 8% |
| Unseen domain: web | stronger monitor, least confident | -2.12 [-4.65, 0.61] | 74% | -1.74 [-3.32, -0.13] | 98% |
| Unseen domain: web | big mpnet monitor, least confident | 2.47 [-1.43, 7.70] | 24% | 1.88 [-1.08, 5.42] | 14% |
| Unseen domain: web | transformer and LSTM disagree most | 0.84 [-0.34, 2.31] | 10% | 0.55 [-0.96, 2.64] | 34% |
| Unseen domain: web | transformer + LSTM, unsure or disagreeing | -1.93 [-4.42, 0.55] | 82% | -1.43 [-3.18, -0.07] | 98% |
| Unseen domain: web | 3-seed ensemble, least confident | -0.12 [-3.05, 4.88] | 66% | -0.63 [-2.64, 2.31] | 66% |
| Unseen domain: web | seeds disagree most | 0.58 [-0.88, 1.76] | 19% | -0.03 [-1.89, 1.96] | 57% |
| Unseen domain: web | 3-seed ensemble, unsure or disagreeing | -0.41 [-3.37, 4.49] | 66% | -0.43 [-2.59, 2.62] | 66% |
| Unseen domain: web | LightGBM deferrer (task folds) | -0.53 [-1.69, 0.47] | 83% | -0.38 [-1.59, 1.16] | 69% |
| Unseen domain: web | LightGBM deferrer (domain folds) | 1.21 [0.14, 2.90] | 0% | 1.08 [-0.48, 3.57] | 17% |
| Unseen domain: web | transformer deferrer (task folds) | -0.36 [-1.47, 0.68] | 67% | 0.06 [-1.46, 1.56] | 46% |
| Unseen domain: web | transformer deferrer (domain folds) | 0.31 [-0.66, 1.35] | 35% | 0.76 [-0.96, 3.45] | 38% |
| Unseen domains (mean) | stronger monitor, least confident | -1.23 [-2.71, -0.11] | 100% | -0.75 [-1.28, -0.17] | 100% |
| Unseen domains (mean) | big mpnet monitor, least confident | 4.90 [2.53, 6.81] | 0% | 3.49 [2.31, 4.75] | 0% |
| Unseen domains (mean) | transformer and LSTM disagree most | -0.20 [-0.97, 0.42] | 66% | -0.04 [-0.50, 0.38] | 54% |
| Unseen domains (mean) | transformer + LSTM, unsure or disagreeing | -1.03 [-2.31, -0.07] | 100% | -0.63 [-1.16, -0.12] | 99% |
| Unseen domains (mean) | 3-seed ensemble, least confident | -0.38 [-2.29, 1.30] | 64% | -0.51 [-1.26, 0.49] | 67% |
| Unseen domains (mean) | seeds disagree most | 0.02 [-1.04, 0.64] | 35% | -0.30 [-1.01, 0.48] | 66% |
| Unseen domains (mean) | 3-seed ensemble, unsure or disagreeing | -0.47 [-2.37, 1.18] | 66% | -0.48 [-1.25, 0.52] | 66% |
| Unseen domains (mean) | LightGBM deferrer (task folds) | -0.05 [-0.70, 0.45] | 44% | 0.04 [-0.35, 0.47] | 42% |
| Unseen domains (mean) | LightGBM deferrer (domain folds) | 1.58 [0.72, 2.90] | 0% | 1.03 [-0.11, 2.46] | 9% |
| Unseen domains (mean) | transformer deferrer (task folds) | 0.54 [-0.13, 1.46] | 8% | 1.30 [0.39, 2.84] | 0% |
| Unseen domains (mean) | transformer deferrer (domain folds) | 0.85 [0.15, 1.89] | 0% | 1.47 [0.39, 3.36] | 0% |

## Attack rate 5%: missed attacks left per 1,000 calls

Review budget 2% (no deferral in the first column):

| setting | no deferral (main monitor) | least confident | most novel | stronger monitor, least confident | big mpnet monitor, least confident | transformer and LSTM disagree most | transformer + LSTM, unsure or disagreeing | 3-seed ensemble, least confident | seeds disagree most | 3-seed ensemble, unsure or disagreeing | LightGBM deferrer (task folds) | LightGBM deferrer (domain folds) | transformer deferrer (task folds) | transformer deferrer (domain folds) | perfect (upper bound) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Task-disjoint test | 0.67 [0.17, 1.20] | 0.20 [0.00, 0.51] | 0.61 [0.17, 1.18] | 0.13 [0.00, 0.34] | 0.60 [0.17, 1.18] | 0.22 [0.00, 0.51] | 0.18 [0.00, 0.51] | 0.18 [0.00, 0.50] | 0.25 [0.00, 0.67] | 0.21 [0.00, 0.51] | 0.21 [0.00, 0.52] | — | 0.26 [0.00, 0.67] | — | 0.00 [0.00, 0.00] |
| Tool coverage 100% | 1.80 [0.00, 4.44] | 0.73 [0.00, 2.00] | 1.50 [0.00, 3.52] | 0.70 [0.00, 1.56] | 1.35 [0.00, 3.60] | 0.79 [0.00, 2.05] | 0.72 [0.00, 2.04] | 0.60 [0.00, 1.57] | 0.63 [0.00, 1.54] | 0.59 [0.00, 1.54] | 0.92 [0.00, 2.58] | — | 1.11 [0.00, 3.61] | — | 0.00 [0.00, 0.00] |
| Tool coverage 75% | 1.95 [0.41, 4.75] | 0.91 [0.00, 2.91] | 1.78 [0.00, 4.67] | 0.91 [0.00, 2.45] | 1.97 [0.00, 5.21] | 0.97 [0.00, 2.45] | 0.86 [0.00, 2.40] | 0.79 [0.00, 1.98] | 0.76 [0.00, 1.95] | 0.71 [0.00, 1.93] | 1.27 [0.00, 3.90] | — | 1.31 [0.00, 4.22] | — | 0.00 [0.00, 0.00] |
| Tool coverage 50% | 2.01 [0.00, 4.58] | 0.99 [0.00, 2.82] | 1.94 [0.00, 4.57] | 1.09 [0.00, 2.77] | 1.68 [0.00, 3.66] | 1.23 [0.00, 3.16] | 1.15 [0.00, 3.11] | 0.73 [0.00, 2.20] | 0.83 [0.00, 1.90] | 0.81 [0.00, 1.87] | 1.38 [0.00, 3.63] | — | 1.47 [0.00, 4.09] | — | 0.00 [0.00, 0.00] |
| Tool coverage 25% | 1.94 [0.47, 3.58] | 0.97 [0.00, 2.26] | 1.75 [0.45, 3.33] | 1.18 [0.00, 2.48] | 2.18 [0.00, 4.82] | 1.17 [0.00, 2.51] | 1.22 [0.00, 2.71] | 1.26 [0.00, 2.87] | 1.22 [0.00, 2.87] | 1.18 [0.00, 2.87] | 1.29 [0.00, 2.88] | — | 1.40 [0.40, 2.92] | — | 0.00 [0.00, 0.00] |
| Unseen domain: banking | 3.48 [2.18, 4.72] | 1.83 [1.11, 2.58] | 3.35 [2.17, 4.65] | 1.77 [1.11, 2.53] | 2.30 [1.65, 2.95] | 2.09 [1.32, 2.79] | 1.91 [1.22, 2.62] | 1.79 [1.09, 2.58] | 1.99 [1.32, 2.74] | 1.86 [1.10, 2.60] | 1.96 [1.22, 2.83] | 2.54 [1.45, 3.85] | 2.31 [1.32, 3.43] | 2.30 [1.24, 3.44] | 0.00 [0.00, 0.00] |
| Unseen domain: coding | 3.70 [2.04, 5.77] | 1.91 [1.12, 2.68] | 3.40 [1.82, 5.54] | 1.65 [0.81, 2.49] | 3.24 [2.05, 4.75] | 1.76 [0.81, 2.63] | 1.63 [0.80, 2.52] | 1.70 [0.90, 2.51] | 1.85 [1.02, 2.65] | 1.72 [1.01, 2.53] | 2.20 [1.25, 3.23] | 2.36 [1.25, 3.50] | 2.52 [1.26, 4.07] | 2.63 [1.36, 4.41] | 0.00 [0.00, 0.00] |
| Unseen domain: email | 0.76 [0.11, 1.76] | 0.51 [0.00, 1.36] | 0.75 [0.11, 1.71] | 0.45 [0.00, 1.01] | 1.33 [0.67, 2.21] | 0.53 [0.00, 1.33] | 0.48 [0.00, 1.12] | 0.51 [0.00, 1.25] | 0.61 [0.11, 1.56] | 0.49 [0.11, 1.23] | 0.56 [0.11, 1.47] | 0.67 [0.00, 1.65] | 0.67 [0.11, 1.59] | 0.70 [0.11, 1.65] | 0.00 [0.00, 0.00] |
| Unseen domain: medical | 1.87 [0.23, 3.52] | 0.99 [0.11, 2.16] | 1.85 [0.23, 3.49] | 0.70 [0.11, 1.36] | 3.43 [2.26, 4.73] | 1.00 [0.23, 2.04] | 0.79 [0.22, 1.59] | 0.84 [0.23, 1.58] | 0.82 [0.23, 1.49] | 0.74 [0.23, 1.36] | 0.89 [0.11, 2.03] | 1.23 [0.22, 2.50] | 1.35 [0.11, 2.78] | 1.22 [0.11, 2.58] | 0.00 [0.00, 0.00] |
| Unseen domain: web | 2.87 [0.81, 4.90] | 2.29 [0.79, 3.97] | 2.71 [0.81, 4.58] | 1.81 [0.80, 3.09] | 3.05 [1.73, 4.41] | 2.57 [0.79, 4.30] | 1.86 [0.80, 3.12] | 2.33 [1.38, 3.43] | 2.59 [0.81, 4.24] | 2.28 [1.26, 3.35] | 2.18 [0.58, 3.78] | 2.58 [0.79, 4.62] | 2.23 [0.57, 3.88] | 2.41 [0.58, 4.16] | 0.00 [0.00, 0.00] |
| Unseen domains (mean) | 2.54 [1.56, 3.67] | 1.51 [0.97, 2.03] | 2.41 [1.50, 3.48] | 1.28 [0.90, 1.59] | 2.67 [2.18, 3.17] | 1.59 [1.06, 2.09] | 1.33 [0.92, 1.72] | 1.43 [1.12, 1.74] | 1.57 [1.08, 2.02] | 1.42 [1.10, 1.73] | 1.56 [1.08, 2.07] | 1.87 [1.28, 2.77] | 1.82 [1.07, 2.75] | 1.85 [1.16, 2.79] | 0.00 [0.00, 0.00] |

Review budget 5% (no deferral in the first column):

| setting | no deferral (main monitor) | least confident | most novel | stronger monitor, least confident | big mpnet monitor, least confident | transformer and LSTM disagree most | transformer + LSTM, unsure or disagreeing | 3-seed ensemble, least confident | seeds disagree most | 3-seed ensemble, unsure or disagreeing | LightGBM deferrer (task folds) | LightGBM deferrer (domain folds) | transformer deferrer (task folds) | transformer deferrer (domain folds) | perfect (upper bound) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Task-disjoint test | 0.67 [0.17, 1.20] | 0.05 [0.00, 0.17] | 0.61 [0.17, 1.18] | 0.02 [0.00, 0.17] | 0.40 [0.00, 0.84] | 0.08 [0.00, 0.33] | 0.05 [0.00, 0.17] | 0.05 [0.00, 0.17] | 0.08 [0.00, 0.34] | 0.05 [0.00, 0.17] | 0.07 [0.00, 0.34] | — | 0.20 [0.00, 0.51] | — | 0.00 [0.00, 0.00] |
| Tool coverage 100% | 1.80 [0.00, 4.44] | 0.42 [0.00, 1.05] | 1.32 [0.00, 3.49] | 0.23 [0.00, 1.02] | 0.52 [0.00, 1.56] | 0.42 [0.00, 1.31] | 0.39 [0.00, 1.31] | 0.36 [0.00, 1.05] | 0.27 [0.00, 1.02] | 0.33 [0.00, 1.02] | 0.43 [0.00, 1.51] | — | 0.76 [0.00, 3.07] | — | 0.00 [0.00, 0.00] |
| Tool coverage 75% | 1.95 [0.41, 4.75] | 0.52 [0.00, 1.50] | 1.77 [0.00, 4.63] | 0.30 [0.00, 0.97] | 1.05 [0.00, 2.85] | 0.42 [0.00, 1.44] | 0.33 [0.00, 1.22] | 0.30 [0.00, 0.99] | 0.30 [0.00, 1.00] | 0.30 [0.00, 0.99] | 0.65 [0.00, 2.40] | — | 0.96 [0.00, 3.75] | — | 0.00 [0.00, 0.00] |
| Tool coverage 50% | 2.01 [0.00, 4.58] | 0.54 [0.00, 1.83] | 1.78 [0.00, 4.57] | 0.37 [0.00, 1.79] | 0.92 [0.00, 2.71] | 0.53 [0.00, 1.79] | 0.41 [0.00, 1.36] | 0.29 [0.00, 0.92] | 0.40 [0.00, 1.32] | 0.32 [0.00, 0.92] | 0.62 [0.00, 2.26] | — | 1.14 [0.00, 3.57] | — | 0.00 [0.00, 0.00] |
| Tool coverage 25% | 1.94 [0.47, 3.58] | 0.63 [0.00, 1.64] | 1.62 [0.44, 3.30] | 0.50 [0.00, 1.63] | 1.04 [0.00, 3.12] | 0.61 [0.00, 2.07] | 0.50 [0.00, 1.66] | 0.52 [0.00, 1.42] | 0.65 [0.00, 1.65] | 0.55 [0.00, 1.49] | 0.76 [0.00, 2.26] | — | 1.16 [0.00, 2.69] | — | 0.00 [0.00, 0.00] |
| Unseen domain: banking | 3.48 [2.18, 4.72] | 0.96 [0.33, 1.64] | 3.14 [1.99, 4.33] | 1.00 [0.44, 1.56] | 1.27 [0.77, 1.77] | 1.17 [0.67, 1.68] | 1.05 [0.55, 1.64] | 0.89 [0.34, 1.43] | 1.02 [0.44, 1.55] | 1.01 [0.44, 1.54] | 0.90 [0.44, 1.56] | 1.35 [0.66, 2.22] | 1.66 [0.87, 2.82] | 1.59 [0.78, 2.69] | 0.00 [0.00, 0.00] |
| Unseen domain: coding | 3.70 [2.04, 5.77] | 1.12 [0.45, 1.82] | 3.10 [1.60, 5.14] | 0.83 [0.34, 1.46] | 2.58 [1.59, 3.85] | 1.05 [0.46, 1.72] | 0.90 [0.34, 1.49] | 0.94 [0.34, 1.58] | 1.01 [0.35, 1.70] | 1.00 [0.35, 1.62] | 1.33 [0.58, 1.98] | 1.44 [0.57, 2.51] | 1.93 [0.80, 3.28] | 1.93 [0.79, 3.62] | 0.00 [0.00, 0.00] |
| Unseen domain: email | 0.76 [0.11, 1.76] | 0.38 [0.00, 1.02] | 0.74 [0.11, 1.71] | 0.37 [0.00, 0.90] | 0.91 [0.34, 1.69] | 0.37 [0.00, 1.01] | 0.35 [0.00, 0.90] | 0.35 [0.00, 1.02] | 0.37 [0.00, 0.90] | 0.32 [0.00, 0.90] | 0.45 [0.00, 1.14] | 0.49 [0.00, 1.37] | 0.61 [0.00, 1.58] | 0.64 [0.11, 1.56] | 0.00 [0.00, 0.00] |
| Unseen domain: medical | 1.87 [0.23, 3.52] | 0.54 [0.00, 1.38] | 1.73 [0.22, 3.29] | 0.40 [0.11, 0.80] | 1.71 [1.01, 2.40] | 0.50 [0.11, 1.13] | 0.42 [0.11, 0.90] | 0.37 [0.11, 0.79] | 0.34 [0.00, 0.69] | 0.34 [0.00, 0.68] | 0.50 [0.00, 1.36] | 0.66 [0.11, 1.48] | 1.00 [0.11, 2.23] | 0.89 [0.11, 2.03] | 0.00 [0.00, 0.00] |
| Unseen domain: web | 2.87 [0.81, 4.90] | 1.65 [0.58, 2.97] | 2.54 [0.79, 4.32] | 1.32 [0.46, 2.29] | 2.23 [1.15, 3.44] | 1.95 [0.68, 3.23] | 1.42 [0.57, 2.53] | 1.61 [0.80, 2.50] | 1.97 [0.81, 3.09] | 1.68 [0.81, 2.59] | 1.60 [0.34, 2.92] | 1.90 [0.69, 3.69] | 1.74 [0.45, 3.21] | 1.90 [0.57, 3.55] | 0.00 [0.00, 0.00] |
| Unseen domains (mean) | 2.54 [1.56, 3.67] | 0.93 [0.60, 1.22] | 2.25 [1.36, 3.28] | 0.78 [0.54, 1.03] | 1.74 [1.35, 2.13] | 1.01 [0.65, 1.31] | 0.83 [0.54, 1.07] | 0.83 [0.59, 1.08] | 0.94 [0.68, 1.22] | 0.87 [0.61, 1.11] | 0.95 [0.63, 1.31] | 1.17 [0.75, 1.75] | 1.39 [0.81, 2.17] | 1.39 [0.77, 2.30] | 0.00 [0.00, 0.00] |

Learned policies vs. least confident at attack rate 5%: difference in missed attacks left per 1,000 calls (negative = the learned one leaves fewer), with the share of resamples where it is better.

| setting | policy | 2% budget | better in | 5% budget | better in |
|---|---|---|---|---|---|
| Task-disjoint test | stronger monitor, least confident | -0.07 [-0.34, 0.00] | 36% | -0.03 [-0.17, 0.00] | 20% |
| Task-disjoint test | big mpnet monitor, least confident | 0.41 [-0.17, 1.02] | 6% | 0.35 [0.00, 0.84] | 1% |
| Task-disjoint test | transformer and LSTM disagree most | 0.02 [-0.17, 0.34] | 22% | 0.03 [-0.17, 0.17] | 15% |
| Task-disjoint test | transformer + LSTM, unsure or disagreeing | -0.01 [-0.18, 0.18] | 29% | -0.01 [-0.17, 0.17] | 17% |
| Task-disjoint test | 3-seed ensemble, least confident | -0.02 [-0.17, 0.17] | 21% | -0.00 [-0.17, 0.17] | 13% |
| Task-disjoint test | seeds disagree most | 0.05 [-0.17, 0.34] | 8% | 0.03 [0.00, 0.17] | 0% |
| Task-disjoint test | 3-seed ensemble, unsure or disagreeing | 0.01 [-0.17, 0.33] | 12% | -0.00 [-0.17, 0.17] | 13% |
| Task-disjoint test | LightGBM deferrer (task folds) | 0.01 [-0.17, 0.17] | 20% | 0.02 [-0.16, 0.17] | 3% |
| Task-disjoint test | transformer deferrer (task folds) | 0.06 [-0.17, 0.34] | 4% | 0.15 [0.00, 0.51] | 0% |
| Tool coverage 100% | stronger monitor, least confident | -0.02 [-0.53, 0.52] | 14% | -0.19 [-1.02, 0.44] | 38% |
| Tool coverage 100% | big mpnet monitor, least confident | 0.63 [-0.88, 2.59] | 26% | 0.10 [-0.87, 1.54] | 28% |
| Tool coverage 100% | transformer and LSTM disagree most | 0.06 [-1.00, 1.03] | 9% | 0.01 [-1.02, 0.53] | 16% |
| Tool coverage 100% | transformer + LSTM, unsure or disagreeing | -0.01 [-0.99, 0.53] | 10% | -0.03 [-1.02, 0.53] | 20% |
| Tool coverage 100% | 3-seed ensemble, least confident | -0.12 [-1.00, 0.51] | 30% | -0.06 [-0.52, 0.50] | 20% |
| Tool coverage 100% | seeds disagree most | -0.09 [-1.03, 0.53] | 28% | -0.14 [-1.02, 0.44] | 30% |
| Tool coverage 100% | 3-seed ensemble, unsure or disagreeing | -0.13 [-1.03, 0.52] | 28% | -0.09 [-0.55, 0.44] | 24% |
| Tool coverage 100% | LightGBM deferrer (task folds) | 0.19 [-0.50, 1.06] | 4% | 0.02 [-0.55, 1.05] | 25% |
| Tool coverage 100% | transformer deferrer (task folds) | 0.38 [-0.44, 2.02] | 9% | 0.34 [-0.52, 2.65] | 23% |
| Tool coverage 75% | stronger monitor, least confident | -0.01 [-0.96, 0.84] | 22% | -0.22 [-0.98, 0.48] | 41% |
| Tool coverage 75% | big mpnet monitor, least confident | 1.06 [-0.82, 3.79] | 22% | 0.53 [-0.84, 2.42] | 18% |
| Tool coverage 75% | transformer and LSTM disagree most | 0.06 [-1.41, 0.97] | 20% | -0.10 [-1.01, 0.48] | 28% |
| Tool coverage 75% | transformer + LSTM, unsure or disagreeing | -0.06 [-1.40, 0.95] | 28% | -0.19 [-1.41, 0.41] | 34% |
| Tool coverage 75% | 3-seed ensemble, least confident | -0.12 [-1.43, 0.98] | 38% | -0.22 [-0.99, 0.00] | 34% |
| Tool coverage 75% | seeds disagree most | -0.15 [-1.48, 0.99] | 40% | -0.21 [-0.99, 0.00] | 34% |
| Tool coverage 75% | 3-seed ensemble, unsure or disagreeing | -0.20 [-1.47, 0.50] | 40% | -0.22 [-0.99, 0.00] | 34% |
| Tool coverage 75% | LightGBM deferrer (task folds) | 0.35 [-0.49, 1.47] | 4% | 0.13 [-0.49, 0.99] | 15% |
| Tool coverage 75% | transformer deferrer (task folds) | 0.40 [-0.41, 1.88] | 7% | 0.44 [-0.49, 2.40] | 14% |
| Tool coverage 50% | stronger monitor, least confident | 0.10 [-0.48, 0.95] | 16% | -0.17 [-0.50, 0.45] | 36% |
| Tool coverage 50% | big mpnet monitor, least confident | 0.69 [-1.21, 2.84] | 28% | 0.38 [-0.81, 1.82] | 18% |
| Tool coverage 50% | transformer and LSTM disagree most | 0.24 [-0.46, 0.97] | 11% | -0.01 [-0.50, 0.91] | 21% |
| Tool coverage 50% | transformer + LSTM, unsure or disagreeing | 0.16 [-0.47, 0.95] | 16% | -0.13 [-0.92, 0.45] | 28% |
| Tool coverage 50% | 3-seed ensemble, least confident | -0.26 [-1.34, 0.48] | 51% | -0.25 [-1.38, 0.01] | 38% |
| Tool coverage 50% | seeds disagree most | -0.17 [-1.37, 0.91] | 42% | -0.14 [-0.95, 0.47] | 29% |
| Tool coverage 50% | 3-seed ensemble, unsure or disagreeing | -0.18 [-1.22, 0.49] | 44% | -0.22 [-1.37, 0.45] | 35% |
| Tool coverage 50% | LightGBM deferrer (task folds) | 0.39 [-0.41, 1.79] | 4% | 0.08 [-0.48, 0.94] | 22% |
| Tool coverage 50% | transformer deferrer (task folds) | 0.48 [-0.48, 1.90] | 8% | 0.60 [0.00, 2.24] | 2% |
| Tool coverage 25% | stronger monitor, least confident | 0.21 [-0.49, 0.98] | 16% | -0.13 [-0.98, 0.46] | 34% |
| Tool coverage 25% | big mpnet monitor, least confident | 1.21 [-0.84, 3.77] | 14% | 0.42 [-0.95, 2.69] | 26% |
| Tool coverage 25% | transformer and LSTM disagree most | 0.20 [-0.51, 0.90] | 18% | -0.02 [-0.96, 0.84] | 30% |
| Tool coverage 25% | transformer + LSTM, unsure or disagreeing | 0.25 [-0.51, 1.25] | 16% | -0.13 [-0.97, 0.45] | 30% |
| Tool coverage 25% | 3-seed ensemble, least confident | 0.29 [-0.48, 1.25] | 10% | -0.10 [-0.50, 0.42] | 26% |
| Tool coverage 25% | seeds disagree most | 0.25 [-0.48, 1.24] | 12% | 0.02 [-0.48, 0.48] | 11% |
| Tool coverage 25% | 3-seed ensemble, unsure or disagreeing | 0.21 [-0.48, 1.25] | 11% | -0.07 [-0.48, 0.41] | 20% |
| Tool coverage 25% | LightGBM deferrer (task folds) | 0.33 [-0.95, 1.34] | 12% | 0.13 [-0.81, 1.36] | 24% |
| Tool coverage 25% | transformer deferrer (task folds) | 0.43 [-0.49, 1.75] | 9% | 0.53 [-0.48, 1.82] | 8% |
| Unseen domain: banking | stronger monitor, least confident | -0.06 [-0.56, 0.34] | 52% | 0.05 [-0.33, 0.44] | 28% |
| Unseen domain: banking | big mpnet monitor, least confident | 0.48 [-0.33, 1.22] | 12% | 0.32 [-0.45, 1.07] | 16% |
| Unseen domain: banking | transformer and LSTM disagree most | 0.26 [-0.22, 0.77] | 10% | 0.21 [-0.22, 0.76] | 14% |
| Unseen domain: banking | transformer + LSTM, unsure or disagreeing | 0.09 [-0.33, 0.66] | 34% | 0.10 [-0.33, 0.55] | 24% |
| Unseen domain: banking | 3-seed ensemble, least confident | -0.04 [-0.44, 0.34] | 46% | -0.07 [-0.45, 0.43] | 59% |
| Unseen domain: banking | seeds disagree most | 0.17 [-0.33, 0.66] | 18% | 0.07 [-0.34, 0.65] | 38% |
| Unseen domain: banking | 3-seed ensemble, unsure or disagreeing | 0.03 [-0.34, 0.56] | 38% | 0.05 [-0.33, 0.55] | 38% |
| Unseen domain: banking | LightGBM deferrer (task folds) | 0.13 [-0.33, 0.65] | 22% | -0.06 [-0.45, 0.33] | 51% |
| Unseen domain: banking | LightGBM deferrer (domain folds) | 0.71 [-0.22, 1.98] | 10% | 0.40 [-0.22, 1.55] | 16% |
| Unseen domain: banking | transformer deferrer (task folds) | 0.48 [-0.22, 1.53] | 10% | 0.71 [-0.11, 2.08] | 4% |
| Unseen domain: banking | transformer deferrer (domain folds) | 0.48 [-0.33, 1.54] | 8% | 0.63 [-0.11, 1.85] | 6% |
| Unseen domain: coding | stronger monitor, least confident | -0.26 [-0.79, 0.23] | 80% | -0.29 [-0.79, 0.11] | 88% |
| Unseen domain: coding | big mpnet monitor, least confident | 1.33 [0.23, 2.61] | 1% | 1.46 [0.57, 2.58] | 0% |
| Unseen domain: coding | transformer and LSTM disagree most | -0.14 [-0.58, 0.35] | 64% | -0.07 [-0.58, 0.46] | 51% |
| Unseen domain: coding | transformer + LSTM, unsure or disagreeing | -0.28 [-0.79, 0.23] | 80% | -0.22 [-0.69, 0.23] | 75% |
| Unseen domain: coding | 3-seed ensemble, least confident | -0.21 [-0.80, 0.23] | 70% | -0.18 [-0.58, 0.12] | 72% |
| Unseen domain: coding | seeds disagree most | -0.06 [-0.56, 0.45] | 49% | -0.11 [-0.57, 0.34] | 57% |
| Unseen domain: coding | 3-seed ensemble, unsure or disagreeing | -0.18 [-0.69, 0.33] | 68% | -0.12 [-0.56, 0.23] | 57% |
| Unseen domain: coding | LightGBM deferrer (task folds) | 0.29 [-0.23, 0.91] | 11% | 0.21 [-0.22, 0.67] | 6% |
| Unseen domain: coding | LightGBM deferrer (domain folds) | 0.45 [-0.23, 1.24] | 9% | 0.32 [-0.23, 1.26] | 12% |
| Unseen domain: coding | transformer deferrer (task folds) | 0.61 [-0.23, 1.98] | 10% | 0.81 [-0.11, 2.18] | 4% |
| Unseen domain: coding | transformer deferrer (domain folds) | 0.72 [-0.11, 2.09] | 3% | 0.81 [0.00, 2.16] | 0% |
| Unseen domain: email | stronger monitor, least confident | -0.06 [-0.67, 0.46] | 44% | -0.02 [-0.56, 0.45] | 38% |
| Unseen domain: email | big mpnet monitor, least confident | 0.82 [0.00, 1.79] | 2% | 0.53 [-0.12, 1.32] | 6% |
| Unseen domain: email | transformer and LSTM disagree most | 0.02 [-0.45, 0.46] | 32% | -0.02 [-0.46, 0.34] | 36% |
| Unseen domain: email | transformer + LSTM, unsure or disagreeing | -0.03 [-0.56, 0.45] | 38% | -0.03 [-0.56, 0.34] | 40% |
| Unseen domain: email | 3-seed ensemble, least confident | 0.00 [-0.45, 0.57] | 44% | -0.03 [-0.34, 0.45] | 48% |
| Unseen domain: email | seeds disagree most | 0.10 [-0.34, 0.56] | 14% | -0.02 [-0.45, 0.33] | 34% |
| Unseen domain: email | 3-seed ensemble, unsure or disagreeing | -0.02 [-0.56, 0.57] | 46% | -0.06 [-0.45, 0.34] | 48% |
| Unseen domain: email | LightGBM deferrer (task folds) | 0.05 [-0.11, 0.33] | 14% | 0.06 [-0.22, 0.34] | 15% |
| Unseen domain: email | LightGBM deferrer (domain folds) | 0.15 [-0.11, 0.56] | 7% | 0.11 [-0.11, 0.45] | 16% |
| Unseen domain: email | transformer deferrer (task folds) | 0.16 [-0.11, 0.68] | 3% | 0.23 [-0.11, 0.80] | 3% |
| Unseen domain: email | transformer deferrer (domain folds) | 0.19 [0.00, 0.67] | 2% | 0.25 [0.00, 0.79] | 1% |
| Unseen domain: medical | stronger monitor, least confident | -0.29 [-1.03, 0.34] | 68% | -0.13 [-0.80, 0.23] | 57% |
| Unseen domain: medical | big mpnet monitor, least confident | 2.43 [0.79, 3.94] | 0% | 1.17 [0.00, 2.06] | 2% |
| Unseen domain: medical | transformer and LSTM disagree most | 0.00 [-0.80, 0.58] | 30% | -0.03 [-0.68, 0.34] | 38% |
| Unseen domain: medical | transformer + LSTM, unsure or disagreeing | -0.21 [-1.02, 0.45] | 56% | -0.11 [-0.79, 0.23] | 50% |
| Unseen domain: medical | 3-seed ensemble, least confident | -0.16 [-1.13, 0.90] | 63% | -0.16 [-0.80, 0.23] | 58% |
| Unseen domain: medical | seeds disagree most | -0.17 [-1.03, 0.45] | 50% | -0.20 [-0.91, 0.23] | 57% |
| Unseen domain: medical | 3-seed ensemble, unsure or disagreeing | -0.25 [-1.24, 0.69] | 62% | -0.20 [-0.91, 0.23] | 60% |
| Unseen domain: medical | LightGBM deferrer (task folds) | -0.10 [-0.56, 0.23] | 52% | -0.04 [-0.34, 0.23] | 42% |
| Unseen domain: medical | LightGBM deferrer (domain folds) | 0.24 [0.00, 0.67] | 2% | 0.12 [-0.23, 0.57] | 16% |
| Unseen domain: medical | transformer deferrer (task folds) | 0.36 [-0.11, 1.04] | 7% | 0.46 [-0.22, 1.45] | 10% |
| Unseen domain: medical | transformer deferrer (domain folds) | 0.23 [-0.12, 0.81] | 12% | 0.36 [-0.12, 1.15] | 12% |
| Unseen domain: web | stronger monitor, least confident | -0.48 [-1.38, 0.35] | 74% | -0.34 [-1.13, 0.23] | 76% |
| Unseen domain: web | big mpnet monitor, least confident | 0.75 [-0.95, 3.02] | 23% | 0.58 [-0.81, 2.15] | 22% |
| Unseen domain: web | transformer and LSTM disagree most | 0.28 [-0.35, 1.03] | 18% | 0.29 [-0.45, 1.13] | 22% |
| Unseen domain: web | transformer + LSTM, unsure or disagreeing | -0.43 [-1.27, 0.35] | 74% | -0.24 [-0.91, 0.34] | 72% |
| Unseen domain: web | 3-seed ensemble, least confident | 0.04 [-1.02, 1.63] | 62% | -0.04 [-0.91, 0.81] | 52% |
| Unseen domain: web | seeds disagree most | 0.30 [-0.35, 0.91] | 8% | 0.32 [-0.34, 0.90] | 10% |
| Unseen domain: web | 3-seed ensemble, unsure or disagreeing | -0.01 [-1.17, 1.59] | 59% | 0.02 [-0.92, 0.81] | 44% |
| Unseen domain: web | LightGBM deferrer (task folds) | -0.11 [-0.68, 0.35] | 61% | -0.06 [-0.69, 0.59] | 54% |
| Unseen domain: web | LightGBM deferrer (domain folds) | 0.29 [-0.34, 1.02] | 10% | 0.25 [-0.24, 0.92] | 13% |
| Unseen domain: web | transformer deferrer (task folds) | -0.06 [-0.58, 0.57] | 55% | 0.08 [-0.47, 0.92] | 40% |
| Unseen domain: web | transformer deferrer (domain folds) | 0.12 [-0.35, 0.69] | 27% | 0.25 [-0.35, 1.16] | 30% |
| Unseen domains (mean) | stronger monitor, least confident | -0.23 [-0.59, 0.14] | 86% | -0.15 [-0.39, 0.04] | 93% |
| Unseen domains (mean) | big mpnet monitor, least confident | 1.16 [0.54, 1.88] | 0% | 0.81 [0.38, 1.31] | 0% |
| Unseen domains (mean) | transformer and LSTM disagree most | 0.08 [-0.16, 0.29] | 26% | 0.08 [-0.14, 0.28] | 20% |
| Unseen domains (mean) | transformer + LSTM, unsure or disagreeing | -0.17 [-0.52, 0.16] | 78% | -0.10 [-0.32, 0.07] | 82% |
| Unseen domains (mean) | 3-seed ensemble, least confident | -0.07 [-0.52, 0.43] | 64% | -0.10 [-0.34, 0.14] | 73% |
| Unseen domains (mean) | seeds disagree most | 0.06 [-0.19, 0.29] | 26% | 0.01 [-0.20, 0.20] | 44% |
| Unseen domains (mean) | 3-seed ensemble, unsure or disagreeing | -0.09 [-0.52, 0.44] | 66% | -0.06 [-0.32, 0.19] | 62% |
| Unseen domains (mean) | LightGBM deferrer (task folds) | 0.05 [-0.18, 0.27] | 36% | 0.02 [-0.14, 0.21] | 41% |
| Unseen domains (mean) | LightGBM deferrer (domain folds) | 0.37 [-0.05, 0.83] | 6% | 0.24 [-0.09, 0.67] | 12% |
| Unseen domains (mean) | transformer deferrer (task folds) | 0.31 [-0.00, 0.86] | 4% | 0.46 [0.05, 1.17] | 2% |
| Unseen domains (mean) | transformer deferrer (domain folds) | 0.35 [0.00, 0.86] | 2% | 0.46 [0.05, 1.21] | 0% |

## Attack rate 1%: missed attacks left per 1,000 calls

Review budget 2% (no deferral in the first column):

| setting | no deferral (main monitor) | least confident | most novel | stronger monitor, least confident | big mpnet monitor, least confident | transformer and LSTM disagree most | transformer + LSTM, unsure or disagreeing | 3-seed ensemble, least confident | seeds disagree most | 3-seed ensemble, unsure or disagreeing | LightGBM deferrer (task folds) | LightGBM deferrer (domain folds) | transformer deferrer (task folds) | transformer deferrer (domain folds) | perfect (upper bound) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Task-disjoint test | 0.13 [0.00, 0.56] | 0.03 [0.00, 0.19] | 0.12 [0.00, 0.39] | 0.02 [0.00, 0.19] | 0.12 [0.00, 0.39] | 0.05 [0.00, 0.19] | 0.04 [0.00, 0.19] | 0.03 [0.00, 0.19] | 0.04 [0.00, 0.20] | 0.03 [0.00, 0.19] | 0.04 [0.00, 0.19] | — | 0.05 [0.00, 0.37] | — | 0.00 [0.00, 0.00] |
| Tool coverage 100% | 0.30 [0.00, 1.18] | 0.12 [0.00, 0.93] | 0.25 [0.00, 1.15] | 0.10 [0.00, 0.59] | 0.24 [0.00, 1.18] | 0.13 [0.00, 0.60] | 0.11 [0.00, 0.60] | 0.09 [0.00, 0.58] | 0.11 [0.00, 0.59] | 0.10 [0.00, 0.59] | 0.13 [0.00, 0.93] | — | 0.16 [0.00, 0.99] | — | 0.00 [0.00, 0.00] |
| Tool coverage 75% | 0.36 [0.00, 1.35] | 0.17 [0.00, 1.04] | 0.31 [0.00, 1.35] | 0.14 [0.00, 1.04] | 0.35 [0.00, 1.56] | 0.14 [0.00, 0.56] | 0.13 [0.00, 0.54] | 0.14 [0.00, 1.03] | 0.11 [0.00, 0.55] | 0.11 [0.00, 0.54] | 0.23 [0.00, 1.06] | — | 0.24 [0.00, 1.07] | — | 0.00 [0.00, 0.00] |
| Tool coverage 50% | 0.40 [0.00, 1.49] | 0.19 [0.00, 0.98] | 0.38 [0.00, 1.47] | 0.20 [0.00, 0.98] | 0.34 [0.00, 1.49] | 0.22 [0.00, 0.99] | 0.21 [0.00, 0.98] | 0.13 [0.00, 0.51] | 0.18 [0.00, 0.98] | 0.17 [0.00, 0.97] | 0.27 [0.00, 1.35] | — | 0.29 [0.00, 1.44] | — | 0.00 [0.00, 0.00] |
| Tool coverage 25% | 0.41 [0.00, 1.43] | 0.20 [0.00, 0.95] | 0.38 [0.00, 1.43] | 0.25 [0.00, 0.97] | 0.40 [0.00, 1.54] | 0.23 [0.00, 0.96] | 0.25 [0.00, 1.01] | 0.25 [0.00, 1.04] | 0.23 [0.00, 0.98] | 0.24 [0.00, 0.98] | 0.26 [0.00, 1.00] | — | 0.28 [0.00, 1.35] | — | 0.00 [0.00, 0.00] |
| Unseen domain: banking | 0.71 [0.13, 1.28] | 0.31 [0.00, 0.78] | 0.68 [0.13, 1.28] | 0.33 [0.00, 0.78] | 0.43 [0.12, 0.90] | 0.38 [0.00, 0.89] | 0.33 [0.00, 0.89] | 0.30 [0.00, 0.77] | 0.34 [0.00, 0.90] | 0.32 [0.00, 0.90] | 0.37 [0.00, 0.90] | 0.44 [0.00, 1.01] | 0.45 [0.00, 1.16] | 0.42 [0.00, 1.03] | 0.00 [0.00, 0.00] |
| Unseen domain: coding | 0.77 [0.13, 1.53] | 0.36 [0.00, 0.89] | 0.70 [0.13, 1.40] | 0.31 [0.00, 0.87] | 0.66 [0.13, 1.51] | 0.34 [0.00, 0.87] | 0.31 [0.00, 0.87] | 0.30 [0.00, 0.76] | 0.34 [0.00, 0.89] | 0.31 [0.00, 0.77] | 0.42 [0.00, 1.02] | 0.44 [0.00, 1.02] | 0.52 [0.00, 1.15] | 0.53 [0.00, 1.15] | 0.00 [0.00, 0.00] |
| Unseen domain: email | 0.16 [0.00, 0.74] | 0.11 [0.00, 0.51] | 0.16 [0.00, 0.74] | 0.10 [0.00, 0.50] | 0.28 [0.00, 0.75] | 0.12 [0.00, 0.62] | 0.11 [0.00, 0.51] | 0.10 [0.00, 0.62] | 0.13 [0.00, 0.62] | 0.11 [0.00, 0.51] | 0.12 [0.00, 0.61] | 0.14 [0.00, 0.63] | 0.14 [0.00, 0.63] | 0.15 [0.00, 0.62] | 0.00 [0.00, 0.00] |
| Unseen domain: medical | 0.37 [0.00, 1.26] | 0.19 [0.00, 0.75] | 0.37 [0.00, 1.14] | 0.13 [0.00, 0.51] | 0.60 [0.13, 1.25] | 0.18 [0.00, 0.52] | 0.15 [0.00, 0.51] | 0.16 [0.00, 0.51] | 0.15 [0.00, 0.50] | 0.14 [0.00, 0.51] | 0.18 [0.00, 0.63] | 0.23 [0.00, 0.76] | 0.26 [0.00, 0.89] | 0.23 [0.00, 0.88] | 0.00 [0.00, 0.00] |
| Unseen domain: web | 0.57 [0.00, 1.31] | 0.45 [0.00, 1.02] | 0.54 [0.00, 1.29] | 0.36 [0.00, 0.78] | 0.56 [0.13, 1.29] | 0.50 [0.00, 1.15] | 0.38 [0.00, 0.78] | 0.47 [0.00, 0.92] | 0.52 [0.00, 1.18] | 0.48 [0.00, 1.03] | 0.43 [0.00, 1.03] | 0.51 [0.00, 1.28] | 0.44 [0.00, 1.04] | 0.47 [0.00, 1.16] | 0.00 [0.00, 0.00] |
| Unseen domains (mean) | 0.52 [0.23, 0.87] | 0.28 [0.13, 0.51] | 0.49 [0.23, 0.84] | 0.25 [0.08, 0.44] | 0.51 [0.28, 0.78] | 0.30 [0.13, 0.51] | 0.26 [0.10, 0.46] | 0.27 [0.10, 0.46] | 0.30 [0.13, 0.51] | 0.27 [0.13, 0.48] | 0.30 [0.15, 0.51] | 0.35 [0.13, 0.64] | 0.36 [0.13, 0.66] | 0.36 [0.13, 0.67] | 0.00 [0.00, 0.00] |

Review budget 5% (no deferral in the first column):

| setting | no deferral (main monitor) | least confident | most novel | stronger monitor, least confident | big mpnet monitor, least confident | transformer and LSTM disagree most | transformer + LSTM, unsure or disagreeing | 3-seed ensemble, least confident | seeds disagree most | 3-seed ensemble, unsure or disagreeing | LightGBM deferrer (task folds) | LightGBM deferrer (domain folds) | transformer deferrer (task folds) | transformer deferrer (domain folds) | perfect (upper bound) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Task-disjoint test | 0.13 [0.00, 0.56] | 0.01 [0.00, 0.19] | 0.12 [0.00, 0.39] | 0.00 [0.00, 0.00] | 0.08 [0.00, 0.37] | 0.02 [0.00, 0.19] | 0.01 [0.00, 0.19] | 0.00 [0.00, 0.00] | 0.01 [0.00, 0.19] | 0.00 [0.00, 0.00] | 0.01 [0.00, 0.19] | — | 0.03 [0.00, 0.19] | — | 0.00 [0.00, 0.00] |
| Tool coverage 100% | 0.30 [0.00, 1.18] | 0.06 [0.00, 0.57] | 0.22 [0.00, 1.15] | 0.03 [0.00, 0.56] | 0.09 [0.00, 0.58] | 0.08 [0.00, 0.59] | 0.06 [0.00, 0.57] | 0.06 [0.00, 0.57] | 0.06 [0.00, 0.57] | 0.06 [0.00, 0.57] | 0.06 [0.00, 0.58] | — | 0.10 [0.00, 0.95] | — | 0.00 [0.00, 0.00] |
| Tool coverage 75% | 0.36 [0.00, 1.35] | 0.09 [0.00, 0.54] | 0.31 [0.00, 1.35] | 0.05 [0.00, 0.53] | 0.14 [0.00, 1.05] | 0.04 [0.00, 0.53] | 0.04 [0.00, 0.53] | 0.06 [0.00, 0.54] | 0.06 [0.00, 0.54] | 0.06 [0.00, 0.54] | 0.11 [0.00, 0.54] | — | 0.17 [0.00, 1.06] | — | 0.00 [0.00, 0.00] |
| Tool coverage 50% | 0.40 [0.00, 1.49] | 0.08 [0.00, 0.50] | 0.35 [0.00, 1.47] | 0.08 [0.00, 0.50] | 0.19 [0.00, 1.00] | 0.09 [0.00, 0.50] | 0.08 [0.00, 0.50] | 0.05 [0.00, 0.49] | 0.08 [0.00, 0.50] | 0.06 [0.00, 0.49] | 0.12 [0.00, 0.90] | — | 0.22 [0.00, 1.02] | — | 0.00 [0.00, 0.00] |
| Tool coverage 25% | 0.41 [0.00, 1.43] | 0.15 [0.00, 0.90] | 0.35 [0.00, 1.35] | 0.13 [0.00, 0.90] | 0.20 [0.00, 0.96] | 0.13 [0.00, 0.87] | 0.12 [0.00, 0.87] | 0.11 [0.00, 0.52] | 0.14 [0.00, 0.91] | 0.12 [0.00, 0.52] | 0.16 [0.00, 0.95] | — | 0.22 [0.00, 0.97] | — | 0.00 [0.00, 0.00] |
| Unseen domain: banking | 0.71 [0.13, 1.28] | 0.18 [0.00, 0.53] | 0.65 [0.13, 1.18] | 0.18 [0.00, 0.52] | 0.21 [0.00, 0.52] | 0.21 [0.00, 0.52] | 0.19 [0.00, 0.52] | 0.13 [0.00, 0.40] | 0.20 [0.00, 0.53] | 0.16 [0.00, 0.52] | 0.14 [0.00, 0.52] | 0.22 [0.00, 0.65] | 0.31 [0.00, 0.90] | 0.28 [0.00, 0.79] | 0.00 [0.00, 0.00] |
| Unseen domain: coding | 0.77 [0.13, 1.53] | 0.21 [0.00, 0.52] | 0.64 [0.13, 1.38] | 0.14 [0.00, 0.51] | 0.53 [0.13, 1.15] | 0.19 [0.00, 0.51] | 0.16 [0.00, 0.51] | 0.17 [0.00, 0.51] | 0.18 [0.00, 0.63] | 0.18 [0.00, 0.63] | 0.25 [0.00, 0.64] | 0.26 [0.00, 0.76] | 0.39 [0.00, 1.01] | 0.38 [0.00, 1.01] | 0.00 [0.00, 0.00] |
| Unseen domain: email | 0.16 [0.00, 0.74] | 0.09 [0.00, 0.50] | 0.16 [0.00, 0.74] | 0.08 [0.00, 0.38] | 0.19 [0.00, 0.63] | 0.08 [0.00, 0.50] | 0.07 [0.00, 0.50] | 0.07 [0.00, 0.49] | 0.08 [0.00, 0.38] | 0.07 [0.00, 0.38] | 0.10 [0.00, 0.50] | 0.10 [0.00, 0.51] | 0.13 [0.00, 0.62] | 0.14 [0.00, 0.62] | 0.00 [0.00, 0.00] |
| Unseen domain: medical | 0.37 [0.00, 1.26] | 0.12 [0.00, 0.51] | 0.35 [0.00, 1.14] | 0.08 [0.00, 0.38] | 0.30 [0.00, 0.76] | 0.09 [0.00, 0.38] | 0.08 [0.00, 0.38] | 0.07 [0.00, 0.25] | 0.05 [0.00, 0.25] | 0.06 [0.00, 0.25] | 0.11 [0.00, 0.51] | 0.13 [0.00, 0.51] | 0.20 [0.00, 0.76] | 0.16 [0.00, 0.63] | 0.00 [0.00, 0.00] |
| Unseen domain: web | 0.57 [0.00, 1.31] | 0.33 [0.00, 0.77] | 0.50 [0.00, 1.28] | 0.26 [0.00, 0.64] | 0.41 [0.00, 1.03] | 0.39 [0.00, 0.91] | 0.28 [0.00, 0.64] | 0.33 [0.00, 0.77] | 0.42 [0.00, 1.03] | 0.35 [0.00, 0.77] | 0.31 [0.00, 0.78] | 0.37 [0.00, 0.89] | 0.34 [0.00, 0.91] | 0.36 [0.00, 0.90] | 0.00 [0.00, 0.00] |
| Unseen domains (mean) | 0.52 [0.23, 0.87] | 0.19 [0.05, 0.36] | 0.46 [0.21, 0.82] | 0.15 [0.03, 0.28] | 0.33 [0.15, 0.54] | 0.19 [0.05, 0.36] | 0.16 [0.03, 0.30] | 0.15 [0.03, 0.30] | 0.19 [0.07, 0.31] | 0.16 [0.05, 0.31] | 0.18 [0.05, 0.33] | 0.22 [0.08, 0.43] | 0.27 [0.10, 0.53] | 0.26 [0.08, 0.54] | 0.00 [0.00, 0.00] |

Learned policies vs. least confident at attack rate 1%: difference in missed attacks left per 1,000 calls (negative = the learned one leaves fewer), with the share of resamples where it is better.

| setting | policy | 2% budget | better in | 5% budget | better in |
|---|---|---|---|---|---|
| Task-disjoint test | stronger monitor, least confident | -0.01 [-0.19, 0.00] | 6% | -0.01 [-0.19, 0.00] | 3% |
| Task-disjoint test | big mpnet monitor, least confident | 0.08 [-0.19, 0.38] | 8% | 0.07 [-0.19, 0.37] | 3% |
| Task-disjoint test | transformer and LSTM disagree most | 0.02 [-0.19, 0.19] | 4% | 0.01 [-0.19, 0.19] | 3% |
| Task-disjoint test | transformer + LSTM, unsure or disagreeing | 0.01 [-0.19, 0.19] | 4% | 0.00 [-0.19, 0.19] | 3% |
| Task-disjoint test | 3-seed ensemble, least confident | -0.00 [-0.19, 0.19] | 6% | -0.00 [-0.19, 0.00] | 3% |
| Task-disjoint test | seeds disagree most | 0.00 [-0.19, 0.19] | 4% | 0.00 [0.00, 0.00] | 1% |
| Task-disjoint test | 3-seed ensemble, unsure or disagreeing | -0.00 [-0.19, 0.19] | 4% | -0.00 [-0.19, 0.00] | 3% |
| Task-disjoint test | LightGBM deferrer (task folds) | 0.01 [-0.19, 0.19] | 5% | 0.00 [0.00, 0.00] | 0% |
| Task-disjoint test | transformer deferrer (task folds) | 0.01 [0.00, 0.19] | 0% | 0.02 [0.00, 0.19] | 0% |
| Tool coverage 100% | stronger monitor, least confident | -0.01 [-0.48, 0.00] | 4% | -0.03 [-0.50, 0.00] | 6% |
| Tool coverage 100% | big mpnet monitor, least confident | 0.12 [-0.56, 1.14] | 6% | 0.03 [-0.57, 0.57] | 4% |
| Tool coverage 100% | transformer and LSTM disagree most | 0.01 [0.00, 0.57] | 2% | 0.02 [0.00, 0.49] | 0% |
| Tool coverage 100% | transformer + LSTM, unsure or disagreeing | -0.01 [0.00, 0.00] | 2% | 0.00 [-0.01, 0.48] | 2% |
| Tool coverage 100% | 3-seed ensemble, least confident | -0.03 [-0.57, 0.00] | 6% | 0.00 [0.00, 0.00] | 1% |
| Tool coverage 100% | seeds disagree most | -0.01 [-0.57, 0.00] | 4% | 0.00 [-0.01, 0.01] | 2% |
| Tool coverage 100% | 3-seed ensemble, unsure or disagreeing | -0.02 [-0.57, 0.00] | 4% | 0.00 [0.00, 0.00] | 2% |
| Tool coverage 100% | LightGBM deferrer (task folds) | 0.01 [0.00, 0.56] | 2% | 0.00 [-0.57, 0.58] | 4% |
| Tool coverage 100% | transformer deferrer (task folds) | 0.04 [-0.48, 0.58] | 4% | 0.04 [-0.50, 0.59] | 5% |
| Tool coverage 75% | stronger monitor, least confident | -0.03 [-0.53, 0.52] | 10% | -0.04 [-0.54, 0.00] | 10% |
| Tool coverage 75% | big mpnet monitor, least confident | 0.17 [-0.53, 1.54] | 12% | 0.06 [-0.52, 1.05] | 9% |
| Tool coverage 75% | transformer and LSTM disagree most | -0.03 [-0.53, 0.52] | 10% | -0.04 [-0.54, 0.00] | 8% |
| Tool coverage 75% | transformer + LSTM, unsure or disagreeing | -0.05 [-0.54, 0.52] | 12% | -0.05 [-0.54, 0.00] | 10% |
| Tool coverage 75% | 3-seed ensemble, least confident | -0.03 [-0.53, 0.52] | 11% | -0.02 [-0.46, 0.00] | 4% |
| Tool coverage 75% | seeds disagree most | -0.06 [-0.54, 0.53] | 16% | -0.02 [-0.46, 0.00] | 4% |
| Tool coverage 75% | 3-seed ensemble, unsure or disagreeing | -0.06 [-0.54, 0.44] | 14% | -0.02 [-0.46, 0.00] | 4% |
| Tool coverage 75% | LightGBM deferrer (task folds) | 0.05 [-0.52, 0.53] | 3% | 0.03 [-0.53, 0.53] | 5% |
| Tool coverage 75% | transformer deferrer (task folds) | 0.07 [0.00, 0.53] | 2% | 0.08 [-0.45, 1.04] | 4% |
| Tool coverage 50% | stronger monitor, least confident | 0.01 [-0.49, 0.50] | 6% | -0.01 [-0.01, 0.00] | 2% |
| Tool coverage 50% | big mpnet monitor, least confident | 0.15 [-0.50, 1.04] | 12% | 0.10 [-0.48, 0.97] | 8% |
| Tool coverage 50% | transformer and LSTM disagree most | 0.03 [-0.49, 0.51] | 6% | 0.00 [-0.49, 0.49] | 4% |
| Tool coverage 50% | transformer + LSTM, unsure or disagreeing | 0.02 [-0.49, 0.51] | 6% | -0.01 [-0.49, 0.01] | 4% |
| Tool coverage 50% | 3-seed ensemble, least confident | -0.06 [-0.52, 0.51] | 16% | -0.03 [-0.49, 0.00] | 7% |
| Tool coverage 50% | seeds disagree most | -0.01 [-0.49, 0.51] | 12% | -0.01 [-0.49, 0.49] | 6% |
| Tool coverage 50% | 3-seed ensemble, unsure or disagreeing | -0.02 [-0.49, 0.51] | 11% | -0.03 [-0.49, 0.00] | 7% |
| Tool coverage 50% | LightGBM deferrer (task folds) | 0.08 [0.00, 0.53] | 2% | 0.03 [-0.45, 0.51] | 4% |
| Tool coverage 50% | transformer deferrer (task folds) | 0.10 [-0.50, 0.97] | 4% | 0.14 [0.00, 1.00] | 2% |
| Tool coverage 25% | stronger monitor, least confident | 0.05 [-0.50, 0.52] | 4% | -0.02 [-0.51, 0.48] | 7% |
| Tool coverage 25% | big mpnet monitor, least confident | 0.19 [-0.52, 1.47] | 14% | 0.06 [-0.52, 0.96] | 16% |
| Tool coverage 25% | transformer and LSTM disagree most | 0.03 [-0.50, 0.51] | 5% | -0.02 [-0.51, 0.48] | 8% |
| Tool coverage 25% | transformer + LSTM, unsure or disagreeing | 0.05 [-0.48, 0.52] | 4% | -0.03 [-0.51, 0.00] | 8% |
| Tool coverage 25% | 3-seed ensemble, least confident | 0.05 [-0.45, 0.51] | 4% | -0.03 [-0.48, 0.00] | 8% |
| Tool coverage 25% | seeds disagree most | 0.03 [-0.48, 0.50] | 6% | -0.00 [-0.48, 0.48] | 4% |
| Tool coverage 25% | 3-seed ensemble, unsure or disagreeing | 0.04 [-0.45, 0.48] | 4% | -0.03 [-0.48, 0.00] | 7% |
| Tool coverage 25% | LightGBM deferrer (task folds) | 0.06 [-0.49, 0.51] | 4% | 0.01 [-0.45, 0.48] | 5% |
| Tool coverage 25% | transformer deferrer (task folds) | 0.08 [-0.49, 0.52] | 6% | 0.08 [-0.46, 0.51] | 6% |
| Unseen domain: banking | stronger monitor, least confident | 0.02 [-0.26, 0.26] | 16% | 0.00 [-0.26, 0.13] | 20% |
| Unseen domain: banking | big mpnet monitor, least confident | 0.12 [-0.39, 0.52] | 20% | 0.03 [-0.39, 0.39] | 28% |
| Unseen domain: banking | transformer and LSTM disagree most | 0.07 [-0.26, 0.38] | 12% | 0.03 [-0.13, 0.26] | 15% |
| Unseen domain: banking | transformer + LSTM, unsure or disagreeing | 0.03 [-0.26, 0.26] | 19% | 0.01 [-0.13, 0.13] | 16% |
| Unseen domain: banking | 3-seed ensemble, least confident | -0.01 [-0.14, 0.26] | 28% | -0.04 [-0.26, 0.13] | 33% |
| Unseen domain: banking | seeds disagree most | 0.04 [-0.26, 0.26] | 20% | 0.02 [-0.13, 0.26] | 14% |
| Unseen domain: banking | 3-seed ensemble, unsure or disagreeing | 0.01 [-0.26, 0.26] | 24% | -0.01 [-0.26, 0.13] | 25% |
| Unseen domain: banking | LightGBM deferrer (task folds) | 0.06 [-0.13, 0.26] | 15% | -0.04 [-0.26, 0.13] | 34% |
| Unseen domain: banking | LightGBM deferrer (domain folds) | 0.13 [-0.13, 0.53] | 8% | 0.05 [-0.13, 0.39] | 16% |
| Unseen domain: banking | transformer deferrer (task folds) | 0.14 [-0.13, 0.65] | 14% | 0.14 [-0.13, 0.64] | 12% |
| Unseen domain: banking | transformer deferrer (domain folds) | 0.11 [-0.13, 0.64] | 10% | 0.10 [-0.13, 0.65] | 14% |
| Unseen domain: coding | stronger monitor, least confident | -0.05 [-0.38, 0.25] | 40% | -0.07 [-0.26, 0.13] | 48% |
| Unseen domain: coding | big mpnet monitor, least confident | 0.30 [-0.26, 0.90] | 12% | 0.31 [-0.25, 1.02] | 10% |
| Unseen domain: coding | transformer and LSTM disagree most | -0.02 [-0.38, 0.25] | 35% | -0.03 [-0.26, 0.25] | 36% |
| Unseen domain: coding | transformer + LSTM, unsure or disagreeing | -0.05 [-0.38, 0.13] | 40% | -0.06 [-0.26, 0.13] | 44% |
| Unseen domain: coding | 3-seed ensemble, least confident | -0.06 [-0.38, 0.13] | 42% | -0.04 [-0.25, 0.13] | 35% |
| Unseen domain: coding | seeds disagree most | -0.02 [-0.26, 0.25] | 28% | -0.03 [-0.25, 0.13] | 30% |
| Unseen domain: coding | 3-seed ensemble, unsure or disagreeing | -0.04 [-0.38, 0.13] | 38% | -0.03 [-0.25, 0.13] | 29% |
| Unseen domain: coding | LightGBM deferrer (task folds) | 0.06 [-0.13, 0.38] | 14% | 0.04 [-0.13, 0.25] | 6% |
| Unseen domain: coding | LightGBM deferrer (domain folds) | 0.08 [-0.25, 0.50] | 16% | 0.05 [-0.25, 0.39] | 17% |
| Unseen domain: coding | transformer deferrer (task folds) | 0.16 [-0.13, 0.64] | 13% | 0.18 [-0.13, 0.76] | 10% |
| Unseen domain: coding | transformer deferrer (domain folds) | 0.17 [-0.13, 0.76] | 7% | 0.17 [-0.13, 0.76] | 4% |
| Unseen domain: email | stronger monitor, least confident | -0.01 [-0.25, 0.13] | 22% | -0.01 [-0.25, 0.13] | 17% |
| Unseen domain: email | big mpnet monitor, least confident | 0.17 [-0.25, 0.62] | 8% | 0.10 [-0.25, 0.50] | 12% |
| Unseen domain: email | transformer and LSTM disagree most | 0.01 [-0.25, 0.25] | 14% | -0.01 [-0.25, 0.25] | 16% |
| Unseen domain: email | transformer + LSTM, unsure or disagreeing | -0.00 [-0.25, 0.25] | 19% | -0.02 [-0.25, 0.13] | 18% |
| Unseen domain: email | 3-seed ensemble, least confident | -0.01 [-0.13, 0.13] | 16% | -0.02 [-0.13, 0.13] | 17% |
| Unseen domain: email | seeds disagree most | 0.02 [-0.13, 0.25] | 10% | -0.00 [-0.13, 0.13] | 14% |
| Unseen domain: email | 3-seed ensemble, unsure or disagreeing | -0.00 [-0.13, 0.13] | 20% | -0.02 [-0.13, 0.12] | 18% |
| Unseen domain: email | LightGBM deferrer (task folds) | 0.01 [-0.12, 0.13] | 4% | 0.01 [-0.12, 0.13] | 5% |
| Unseen domain: email | LightGBM deferrer (domain folds) | 0.03 [-0.12, 0.25] | 4% | 0.02 [-0.13, 0.13] | 6% |
| Unseen domain: email | transformer deferrer (task folds) | 0.03 [-0.12, 0.25] | 4% | 0.04 [-0.00, 0.25] | 2% |
| Unseen domain: email | transformer deferrer (domain folds) | 0.04 [0.00, 0.25] | 2% | 0.05 [0.00, 0.25] | 2% |
| Unseen domain: medical | stronger monitor, least confident | -0.06 [-0.38, 0.13] | 31% | -0.04 [-0.38, 0.13] | 26% |
| Unseen domain: medical | big mpnet monitor, least confident | 0.41 [-0.13, 1.13] | 4% | 0.19 [-0.25, 0.64] | 8% |
| Unseen domain: medical | transformer and LSTM disagree most | -0.00 [-0.50, 0.26] | 22% | -0.02 [-0.38, 0.13] | 22% |
| Unseen domain: medical | transformer + LSTM, unsure or disagreeing | -0.04 [-0.50, 0.13] | 28% | -0.04 [-0.38, 0.13] | 25% |
| Unseen domain: medical | 3-seed ensemble, least confident | -0.02 [-0.38, 0.37] | 32% | -0.05 [-0.39, 0.13] | 28% |
| Unseen domain: medical | seeds disagree most | -0.03 [-0.51, 0.25] | 30% | -0.06 [-0.38, 0.13] | 34% |
| Unseen domain: medical | 3-seed ensemble, unsure or disagreeing | -0.05 [-0.50, 0.37] | 38% | -0.06 [-0.39, 0.13] | 34% |
| Unseen domain: medical | LightGBM deferrer (task folds) | -0.01 [-0.25, 0.13] | 20% | -0.01 [-0.13, 0.13] | 16% |
| Unseen domain: medical | LightGBM deferrer (domain folds) | 0.05 [-0.13, 0.25] | 4% | 0.01 [-0.13, 0.13] | 14% |
| Unseen domain: medical | transformer deferrer (task folds) | 0.08 [-0.13, 0.38] | 6% | 0.08 [-0.13, 0.38] | 5% |
| Unseen domain: medical | transformer deferrer (domain folds) | 0.04 [-0.13, 0.26] | 8% | 0.04 [-0.13, 0.26] | 6% |
| Unseen domain: web | stronger monitor, least confident | -0.09 [-0.51, 0.13] | 46% | -0.08 [-0.51, 0.25] | 50% |
| Unseen domain: web | big mpnet monitor, least confident | 0.11 [-0.52, 0.90] | 32% | 0.08 [-0.39, 0.65] | 31% |
| Unseen domain: web | transformer and LSTM disagree most | 0.06 [-0.26, 0.40] | 18% | 0.06 [-0.38, 0.52] | 22% |
| Unseen domain: web | transformer + LSTM, unsure or disagreeing | -0.07 [-0.51, 0.26] | 48% | -0.05 [-0.51, 0.26] | 41% |
| Unseen domain: web | 3-seed ensemble, least confident | 0.02 [-0.38, 0.52] | 30% | -0.01 [-0.39, 0.26] | 28% |
| Unseen domain: web | seeds disagree most | 0.07 [-0.26, 0.39] | 8% | 0.08 [-0.38, 0.51] | 13% |
| Unseen domain: web | 3-seed ensemble, unsure or disagreeing | 0.03 [-0.39, 0.52] | 28% | 0.02 [-0.39, 0.39] | 22% |
| Unseen domain: web | LightGBM deferrer (task folds) | -0.01 [-0.26, 0.26] | 29% | -0.02 [-0.27, 0.26] | 32% |
| Unseen domain: web | LightGBM deferrer (domain folds) | 0.06 [-0.13, 0.39] | 10% | 0.04 [-0.26, 0.38] | 14% |
| Unseen domain: web | transformer deferrer (task folds) | -0.01 [-0.26, 0.38] | 30% | 0.01 [-0.26, 0.39] | 30% |
| Unseen domain: web | transformer deferrer (domain folds) | 0.02 [-0.26, 0.26] | 18% | 0.03 [-0.26, 0.39] | 24% |
| Unseen domains (mean) | stronger monitor, least confident | -0.04 [-0.15, 0.08] | 67% | -0.04 [-0.15, 0.05] | 76% |
| Unseen domains (mean) | big mpnet monitor, least confident | 0.22 [-0.03, 0.48] | 6% | 0.14 [-0.05, 0.35] | 10% |
| Unseen domains (mean) | transformer and LSTM disagree most | 0.02 [-0.10, 0.13] | 32% | 0.01 [-0.13, 0.10] | 43% |
| Unseen domains (mean) | transformer + LSTM, unsure or disagreeing | -0.03 [-0.15, 0.10] | 60% | -0.03 [-0.15, 0.08] | 70% |
| Unseen domains (mean) | 3-seed ensemble, least confident | -0.01 [-0.13, 0.13] | 56% | -0.03 [-0.13, 0.05] | 69% |
| Unseen domains (mean) | seeds disagree most | 0.01 [-0.13, 0.13] | 33% | 0.00 [-0.10, 0.10] | 40% |
| Unseen domains (mean) | 3-seed ensemble, unsure or disagreeing | -0.01 [-0.18, 0.15] | 53% | -0.02 [-0.15, 0.08] | 56% |
| Unseen domains (mean) | LightGBM deferrer (task folds) | 0.02 [-0.05, 0.13] | 32% | -0.00 [-0.08, 0.08] | 54% |
| Unseen domains (mean) | LightGBM deferrer (domain folds) | 0.07 [-0.05, 0.23] | 14% | 0.03 [-0.05, 0.15] | 27% |
| Unseen domains (mean) | transformer deferrer (task folds) | 0.08 [-0.05, 0.30] | 16% | 0.09 [-0.05, 0.28] | 13% |
| Unseen domains (mean) | transformer deferrer (domain folds) | 0.08 [-0.03, 0.26] | 10% | 0.08 [-0.05, 0.28] | 16% |

## Review burden at attack rate 2%

Share of reviews spent on calls the monitor already had right (a 5% budget):

| setting | least confident | most novel | stronger monitor, least confident | big mpnet monitor, least confident | transformer and LSTM disagree most | transformer + LSTM, unsure or disagreeing | 3-seed ensemble, least confident | seeds disagree most | 3-seed ensemble, unsure or disagreeing | LightGBM deferrer (task folds) | LightGBM deferrer (domain folds) | transformer deferrer (task folds) | transformer deferrer (domain folds) | perfect (upper bound) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Task-disjoint test | 93% | 98% | 95% | 91% | 93% | 95% | 95% | 93% | 95% | 93% | — | 94% | — | 93% |
| Tool coverage 100% | 92% | 98% | 94% | 80% | 92% | 94% | 94% | 92% | 94% | 92% | — | 93% | — | 92% |
| Tool coverage 75% | 87% | 96% | 89% | 73% | 87% | 89% | 90% | 87% | 90% | 89% | — | 91% | — | 86% |
| Tool coverage 50% | 86% | 98% | 87% | 71% | 85% | 87% | 88% | 85% | 88% | 88% | — | 91% | — | 83% |
| Tool coverage 25% | 86% | 96% | 86% | 64% | 85% | 87% | 87% | 84% | 87% | 89% | — | 90% | — | 82% |
| Unseen domain: banking | 88% | 99% | 90% | 85% | 88% | 90% | 91% | 88% | 91% | 89% | 88% | 90% | 89% | 87% |
| Unseen domain: coding | 89% | 99% | 88% | 66% | 90% | 88% | 93% | 89% | 93% | 89% | 89% | 92% | 90% | 88% |
| Unseen domain: email | 66% | 94% | 66% | 65% | 58% | 70% | 59% | 57% | 65% | 79% | 52% | 75% | 66% | 37% |
| Unseen domain: medical | 79% | 97% | 74% | 93% | 82% | 77% | 82% | 71% | 84% | 85% | 69% | 80% | 78% | 66% |
| Unseen domain: web | 63% | 91% | 56% | 60% | 88% | 59% | 60% | 60% | 65% | 86% | 51% | 71% | 61% | 36% |
| Unseen domains (mean) | 77% | 96% | 75% | 74% | 81% | 77% | 77% | 73% | 80% | 86% | 70% | 82% | 77% | 63% |

Figures: `results/figures/09_risk_vs_escalation.png`, `results/figures/10_domain_fold_labels.png`. Every budget: `reports/week3/03_deferral_eval_full.md`.
