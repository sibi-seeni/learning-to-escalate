# Deferral at equal review budgets (week 3)

Monitor: transformer on hash-roles. 200 resamples per cell; intervals are the middle 95% over resamples (traces, monitor seeds and deferrer seeds all vary). "Left" = what the system still gets wrong after a perfect reviewer has checked the deferred calls, per 1,000 calls. Lower is better.

Raw and calibrated confidence rank calls identically (temperature scaling keeps the order), so they are one policy ("least confident").

## Test sets

| setting | variants (split × monitor seed) | calls | hijacked | monitor errors | missed attacks |
|---|---|---|---|---|---|
| Task-disjoint test | 1 | 9796 | 1755 | 55 | 23 |
| Tool coverage 100% | 3 | 2767 | 388 | 24 | 14 |
| Tool coverage 75% | 3 | 2767 | 388 | 32 | 16 |
| Tool coverage 50% | 3 | 2767 | 388 | 35 | 17 |
| Tool coverage 25% | 3 | 2767 | 388 | 35 | 15 |
| Unseen domain: banking | 3 | 11634 | 1170 | 146 | 82 |
| Unseen domain: coding | 3 | 14817 | 2808 | 262 | 213 |
| Unseen domain: email | 3 | 14986 | 2862 | 528 | 45 |
| Unseen domain: medical | 3 | 14795 | 2758 | 270 | 103 |
| Unseen domain: web | 3 | 14792 | 2751 | 622 | 161 |
| Unseen domains (mean) | 15 | 14205 | 2470 | 366 | 121 |

## Attack rate 2%: missed attacks left per 1,000 calls

Review budget 2% (no deferral in the first column):

| setting | no deferral (main monitor) | least confident | most novel | stronger monitor, least confident | LightGBM deferrer (task folds) | LightGBM deferrer (domain folds) | transformer deferrer (task folds) | transformer deferrer (domain folds) | perfect (upper bound) |
|---|---|---|---|---|---|---|---|---|---|
| Task-disjoint test | 0.26 [0.00, 0.74] | 0.07 [0.00, 0.37] | 0.25 [0.00, 0.73] | 0.05 [0.00, 0.19] | 0.08 [0.00, 0.37] | — | 0.10 [0.00, 0.37] | — | 0.00 [0.00, 0.00] |
| Tool coverage 100% | 0.66 [0.00, 2.31] | 0.24 [0.00, 1.11] | 0.56 [0.00, 2.22] | 0.22 [0.00, 1.11] | 0.29 [0.00, 1.40] | — | 0.38 [0.00, 1.67] | — | 0.00 [0.00, 0.00] |
| Tool coverage 75% | 0.70 [0.00, 2.09] | 0.34 [0.00, 1.52] | 0.63 [0.00, 2.04] | 0.30 [0.00, 1.04] | 0.46 [0.00, 1.60] | — | 0.47 [0.00, 2.03] | — | 0.00 [0.00, 0.00] |
| Tool coverage 50% | 0.82 [0.00, 2.43] | 0.40 [0.00, 1.47] | 0.79 [0.00, 2.43] | 0.43 [0.00, 1.45] | 0.56 [0.00, 1.95] | — | 0.58 [0.00, 2.00] | — | 0.00 [0.00, 0.00] |
| Tool coverage 25% | 0.79 [0.00, 1.91] | 0.38 [0.00, 1.29] | 0.73 [0.00, 1.87] | 0.48 [0.00, 1.44] | 0.53 [0.00, 1.44] | — | 0.57 [0.00, 1.72] | — | 0.00 [0.00, 0.00] |
| Unseen domain: banking | 1.37 [0.75, 2.25] | 0.63 [0.24, 1.36] | 1.32 [0.74, 2.12] | 0.66 [0.13, 1.25] | 0.73 [0.25, 1.36] | 0.89 [0.37, 1.64] | 0.88 [0.37, 1.61] | 0.83 [0.36, 1.62] | 0.00 [0.00, 0.00] |
| Unseen domain: coding | 1.53 [0.62, 2.82] | 0.74 [0.13, 1.46] | 1.40 [0.61, 2.67] | 0.67 [0.12, 1.35] | 0.87 [0.25, 1.70] | 0.92 [0.25, 1.84] | 1.05 [0.37, 2.37] | 1.09 [0.25, 2.44] | 0.00 [0.00, 0.00] |
| Unseen domain: email | 0.31 [0.00, 0.98] | 0.21 [0.00, 0.85] | 0.30 [0.00, 0.98] | 0.19 [0.00, 0.72] | 0.24 [0.00, 0.85] | 0.27 [0.00, 0.98] | 0.27 [0.00, 0.98] | 0.28 [0.00, 0.98] | 0.00 [0.00, 0.00] |
| Unseen domain: medical | 0.74 [0.00, 1.84] | 0.39 [0.00, 1.23] | 0.74 [0.00, 1.73] | 0.28 [0.00, 0.74] | 0.37 [0.00, 1.11] | 0.47 [0.00, 1.23] | 0.54 [0.00, 1.36] | 0.47 [0.00, 1.35] | 0.00 [0.00, 0.00] |
| Unseen domain: web | 1.14 [0.25, 2.29] | 0.91 [0.13, 2.01] | 1.08 [0.24, 2.16] | 0.72 [0.13, 1.39] | 0.88 [0.13, 1.77] | 1.02 [0.24, 2.23] | 0.89 [0.13, 1.79] | 0.96 [0.13, 2.01] | 0.00 [0.00, 0.00] |
| Unseen domains (mean) | 1.02 [0.57, 1.68] | 0.58 [0.30, 0.91] | 0.97 [0.52, 1.55] | 0.51 [0.30, 0.81] | 0.62 [0.37, 0.94] | 0.71 [0.40, 1.16] | 0.73 [0.39, 1.24] | 0.73 [0.37, 1.33] | 0.00 [0.00, 0.00] |

Review budget 5% (no deferral in the first column):

| setting | no deferral (main monitor) | least confident | most novel | stronger monitor, least confident | LightGBM deferrer (task folds) | LightGBM deferrer (domain folds) | transformer deferrer (task folds) | transformer deferrer (domain folds) | perfect (upper bound) |
|---|---|---|---|---|---|---|---|---|---|
| Task-disjoint test | 0.26 [0.00, 0.74] | 0.02 [0.00, 0.18] | 0.25 [0.00, 0.73] | 0.01 [0.00, 0.18] | 0.03 [0.00, 0.19] | — | 0.08 [0.00, 0.37] | — | 0.00 [0.00, 0.00] |
| Tool coverage 100% | 0.66 [0.00, 2.31] | 0.14 [0.00, 0.93] | 0.49 [0.00, 2.22] | 0.08 [0.00, 0.58] | 0.15 [0.00, 0.94] | — | 0.24 [0.00, 1.64] | — | 0.00 [0.00, 0.00] |
| Tool coverage 75% | 0.70 [0.00, 2.09] | 0.18 [0.00, 1.00] | 0.63 [0.00, 2.04] | 0.10 [0.00, 0.53] | 0.25 [0.00, 1.04] | — | 0.34 [0.00, 1.52] | — | 0.00 [0.00, 0.00] |
| Tool coverage 50% | 0.82 [0.00, 2.43] | 0.19 [0.00, 0.96] | 0.74 [0.00, 2.38] | 0.14 [0.00, 0.88] | 0.24 [0.00, 0.99] | — | 0.44 [0.00, 1.54] | — | 0.00 [0.00, 0.00] |
| Tool coverage 25% | 0.79 [0.00, 1.91] | 0.27 [0.00, 0.98] | 0.67 [0.00, 1.87] | 0.21 [0.00, 0.94] | 0.32 [0.00, 1.40] | — | 0.47 [0.00, 1.44] | — | 0.00 [0.00, 0.00] |
| Unseen domain: banking | 1.37 [0.75, 2.25] | 0.35 [0.00, 0.75] | 1.25 [0.63, 2.01] | 0.36 [0.00, 0.86] | 0.30 [0.00, 0.75] | 0.47 [0.12, 1.11] | 0.61 [0.13, 1.34] | 0.57 [0.12, 1.24] | 0.00 [0.00, 0.00] |
| Unseen domain: coding | 1.53 [0.62, 2.82] | 0.45 [0.00, 0.98] | 1.28 [0.49, 2.44] | 0.33 [0.00, 0.85] | 0.53 [0.12, 1.10] | 0.54 [0.00, 1.23] | 0.81 [0.12, 1.89] | 0.80 [0.12, 2.08] | 0.00 [0.00, 0.00] |
| Unseen domain: email | 0.31 [0.00, 0.98] | 0.17 [0.00, 0.62] | 0.30 [0.00, 0.98] | 0.15 [0.00, 0.61] | 0.19 [0.00, 0.74] | 0.20 [0.00, 0.74] | 0.25 [0.00, 0.98] | 0.26 [0.00, 0.98] | 0.00 [0.00, 0.00] |
| Unseen domain: medical | 0.74 [0.00, 1.84] | 0.23 [0.00, 0.86] | 0.68 [0.00, 1.60] | 0.17 [0.00, 0.49] | 0.21 [0.00, 0.86] | 0.25 [0.00, 0.98] | 0.40 [0.00, 1.23] | 0.34 [0.00, 1.11] | 0.00 [0.00, 0.00] |
| Unseen domain: web | 1.14 [0.25, 2.29] | 0.65 [0.13, 1.37] | 1.01 [0.13, 2.12] | 0.52 [0.00, 1.11] | 0.63 [0.00, 1.50] | 0.74 [0.13, 1.76] | 0.69 [0.12, 1.64] | 0.75 [0.13, 1.74] | 0.00 [0.00, 0.00] |
| Unseen domains (mean) | 1.02 [0.57, 1.68] | 0.37 [0.17, 0.67] | 0.90 [0.48, 1.50] | 0.31 [0.15, 0.50] | 0.37 [0.17, 0.67] | 0.44 [0.20, 0.79] | 0.55 [0.25, 1.00] | 0.54 [0.23, 1.07] | 0.00 [0.00, 0.00] |

Learned policies vs. least confident at attack rate 2%: difference in missed attacks left per 1,000 calls (negative = the learned one leaves fewer), with the share of resamples where it is better.

| setting | policy | 2% budget | better in | 5% budget | better in |
|---|---|---|---|---|---|
| Task-disjoint test | stronger monitor, least confident | -0.02 [-0.18, 0.00] | 13% | -0.01 [-0.18, 0.00] | 8% |
| Task-disjoint test | LightGBM deferrer (task folds) | 0.01 [-0.18, 0.19] | 8% | 0.01 [0.00, 0.18] | 1% |
| Task-disjoint test | transformer deferrer (task folds) | 0.04 [0.00, 0.18] | 2% | 0.05 [0.00, 0.36] | 0% |
| Tool coverage 100% | stronger monitor, least confident | -0.02 [-0.47, 0.00] | 6% | -0.06 [-0.56, 0.00] | 14% |
| Tool coverage 100% | LightGBM deferrer (task folds) | 0.05 [0.00, 0.57] | 2% | 0.01 [-0.56, 0.57] | 10% |
| Tool coverage 100% | transformer deferrer (task folds) | 0.14 [-0.47, 1.10] | 5% | 0.09 [-0.55, 1.14] | 8% |
| Tool coverage 75% | stronger monitor, least confident | -0.04 [-0.52, 0.52] | 14% | -0.08 [-0.53, 0.50] | 18% |
| Tool coverage 75% | LightGBM deferrer (task folds) | 0.12 [-0.51, 1.03] | 4% | 0.07 [-0.52, 1.01] | 6% |
| Tool coverage 75% | transformer deferrer (task folds) | 0.13 [-0.43, 1.03] | 3% | 0.16 [-0.44, 1.49] | 7% |
| Tool coverage 50% | stronger monitor, least confident | 0.04 [-0.49, 0.51] | 11% | -0.05 [-0.49, 0.00] | 11% |
| Tool coverage 50% | LightGBM deferrer (task folds) | 0.16 [-0.48, 0.98] | 4% | 0.05 [-0.48, 0.51] | 10% |
| Tool coverage 50% | transformer deferrer (task folds) | 0.18 [-0.49, 0.99] | 5% | 0.25 [-0.01, 1.46] | 2% |
| Tool coverage 25% | stronger monitor, least confident | 0.10 [-0.51, 0.87] | 9% | -0.06 [-0.52, 0.48] | 18% |
| Tool coverage 25% | LightGBM deferrer (task folds) | 0.15 [-0.51, 0.95] | 6% | 0.05 [-0.44, 0.93] | 8% |
| Tool coverage 25% | transformer deferrer (task folds) | 0.19 [-0.51, 0.98] | 5% | 0.20 [-0.49, 0.98] | 6% |
| Unseen domain: banking | stronger monitor, least confident | 0.03 [-0.25, 0.37] | 24% | 0.02 [-0.25, 0.25] | 23% |
| Unseen domain: banking | LightGBM deferrer (task folds) | 0.10 [-0.25, 0.38] | 12% | -0.04 [-0.25, 0.25] | 40% |
| Unseen domain: banking | LightGBM deferrer (domain folds) | 0.26 [-0.13, 0.87] | 10% | 0.12 [-0.25, 0.62] | 14% |
| Unseen domain: banking | transformer deferrer (task folds) | 0.25 [-0.25, 0.87] | 12% | 0.26 [-0.25, 0.99] | 10% |
| Unseen domain: banking | transformer deferrer (domain folds) | 0.20 [-0.13, 0.87] | 10% | 0.23 [-0.25, 0.99] | 11% |
| Unseen domain: coding | stronger monitor, least confident | -0.07 [-0.37, 0.25] | 50% | -0.12 [-0.49, 0.12] | 66% |
| Unseen domain: coding | LightGBM deferrer (task folds) | 0.13 [-0.12, 0.62] | 14% | 0.08 [-0.12, 0.37] | 11% |
| Unseen domain: coding | LightGBM deferrer (domain folds) | 0.18 [-0.25, 0.74] | 16% | 0.09 [-0.25, 0.73] | 21% |
| Unseen domain: coding | transformer deferrer (task folds) | 0.31 [-0.25, 1.13] | 13% | 0.36 [-0.12, 1.22] | 6% |
| Unseen domain: coding | transformer deferrer (domain folds) | 0.35 [-0.12, 1.23] | 8% | 0.35 [-0.12, 1.23] | 3% |
| Unseen domain: email | stronger monitor, least confident | -0.01 [-0.36, 0.24] | 28% | -0.01 [-0.25, 0.24] | 24% |
| Unseen domain: email | LightGBM deferrer (task folds) | 0.03 [-0.12, 0.24] | 8% | 0.02 [-0.12, 0.24] | 9% |
| Unseen domain: email | LightGBM deferrer (domain folds) | 0.06 [-0.12, 0.36] | 4% | 0.04 [-0.12, 0.25] | 10% |
| Unseen domain: email | transformer deferrer (task folds) | 0.07 [-0.12, 0.37] | 6% | 0.09 [-0.12, 0.37] | 3% |
| Unseen domain: email | transformer deferrer (domain folds) | 0.08 [-0.12, 0.37] | 4% | 0.10 [0.00, 0.48] | 2% |
| Unseen domain: medical | stronger monitor, least confident | -0.11 [-0.74, 0.25] | 52% | -0.06 [-0.49, 0.13] | 33% |
| Unseen domain: medical | LightGBM deferrer (task folds) | -0.02 [-0.25, 0.25] | 33% | -0.01 [-0.24, 0.12] | 24% |
| Unseen domain: medical | LightGBM deferrer (domain folds) | 0.08 [-0.12, 0.25] | 4% | 0.03 [-0.25, 0.25] | 16% |
| Unseen domain: medical | transformer deferrer (task folds) | 0.15 [-0.13, 0.62] | 8% | 0.17 [-0.12, 0.61] | 6% |
| Unseen domain: medical | transformer deferrer (domain folds) | 0.08 [-0.13, 0.49] | 10% | 0.11 [-0.12, 0.51] | 10% |
| Unseen domain: web | stronger monitor, least confident | -0.19 [-0.74, 0.25] | 67% | -0.13 [-0.63, 0.25] | 60% |
| Unseen domain: web | LightGBM deferrer (task folds) | -0.03 [-0.37, 0.26] | 39% | -0.02 [-0.38, 0.38] | 38% |
| Unseen domain: web | LightGBM deferrer (domain folds) | 0.11 [-0.25, 0.50] | 14% | 0.09 [-0.25, 0.62] | 19% |
| Unseen domain: web | transformer deferrer (task folds) | -0.02 [-0.38, 0.38] | 36% | 0.04 [-0.37, 0.51] | 34% |
| Unseen domain: web | transformer deferrer (domain folds) | 0.05 [-0.25, 0.38] | 24% | 0.10 [-0.25, 0.75] | 24% |
| Unseen domains (mean) | stronger monitor, least confident | -0.07 [-0.25, 0.10] | 78% | -0.06 [-0.20, 0.07] | 81% |
| Unseen domains (mean) | LightGBM deferrer (task folds) | 0.04 [-0.07, 0.15] | 28% | 0.00 [-0.10, 0.12] | 50% |
| Unseen domains (mean) | LightGBM deferrer (domain folds) | 0.14 [-0.05, 0.37] | 13% | 0.07 [-0.08, 0.30] | 22% |
| Unseen domains (mean) | transformer deferrer (task folds) | 0.15 [-0.05, 0.49] | 12% | 0.18 [-0.05, 0.55] | 6% |
| Unseen domains (mean) | transformer deferrer (domain folds) | 0.15 [-0.03, 0.45] | 6% | 0.18 [-0.05, 0.52] | 6% |

## Attack rate natural: missed attacks left per 1,000 calls

Review budget 2% (no deferral in the first column):

| setting | no deferral (main monitor) | least confident | most novel | stronger monitor, least confident | LightGBM deferrer (task folds) | LightGBM deferrer (domain folds) | transformer deferrer (task folds) | transformer deferrer (domain folds) | perfect (upper bound) |
|---|---|---|---|---|---|---|---|---|---|
| Task-disjoint test | 2.31 [1.44, 3.15] | 0.76 [0.31, 1.31] | 2.12 [1.23, 2.94] | 0.53 [0.10, 0.93] | 0.69 [0.21, 1.22] | — | 0.90 [0.31, 1.53] | — | 0.00 [0.00, 0.00] |
| Tool coverage 100% | 4.89 [0.79, 12.12] | 2.65 [0.38, 7.87] | 4.08 [0.40, 9.47] | 2.50 [0.39, 6.95] | 3.39 [0.38, 10.04] | — | 3.47 [0.34, 10.07] | — | 0.00 [0.00, 0.00] |
| Tool coverage 75% | 5.67 [1.58, 13.46] | 3.06 [0.35, 8.54] | 5.05 [1.14, 12.76] | 3.15 [0.70, 7.85] | 4.11 [0.39, 11.54] | — | 3.94 [0.38, 11.46] | — | 0.00 [0.00, 0.00] |
| Tool coverage 50% | 6.13 [1.54, 12.43] | 3.24 [0.00, 8.03] | 5.91 [1.54, 12.43] | 3.41 [0.76, 7.87] | 4.45 [0.77, 11.08] | — | 4.71 [0.40, 11.50] | — | 0.00 [0.00, 0.00] |
| Tool coverage 25% | 5.43 [2.74, 8.45] | 2.83 [1.06, 5.03] | 5.00 [2.40, 7.69] | 3.38 [1.39, 5.78] | 3.74 [1.19, 6.64] | — | 3.97 [1.52, 7.02] | — | 0.00 [0.00, 0.00] |
| Unseen domain: banking | 6.99 [4.72, 9.45] | 4.21 [2.93, 5.73] | 6.72 [4.50, 9.03] | 4.04 [3.00, 5.50] | 4.28 [2.94, 5.67] | 5.88 [3.28, 9.33] | 4.76 [2.92, 6.80] | 5.14 [3.28, 7.69] | 0.00 [0.00, 0.00] |
| Unseen domain: coding | 14.26 [9.25, 21.02] | 9.13 [5.91, 13.54] | 13.10 [8.07, 19.41] | 6.93 [5.29, 9.34] | 9.67 [6.65, 13.55] | 11.25 [6.91, 15.76] | 9.91 [5.91, 14.56] | 10.44 [6.47, 16.33] | 0.07 [0.00, 1.02] |
| Unseen domain: email | 3.08 [0.53, 6.01] | 2.09 [0.47, 4.01] | 3.01 [0.53, 5.93] | 1.81 [0.81, 3.05] | 2.32 [0.53, 4.07] | 2.76 [0.40, 5.47] | 2.71 [0.53, 5.30] | 2.80 [0.53, 5.29] | 0.00 [0.00, 0.00] |
| Unseen domain: medical | 6.96 [1.42, 10.94] | 3.89 [0.87, 6.48] | 6.79 [1.42, 10.53] | 2.53 [1.28, 3.85] | 3.29 [0.61, 5.90] | 6.10 [1.41, 10.00] | 5.01 [0.95, 8.17] | 4.85 [0.87, 8.05] | 0.00 [0.00, 0.00] |
| Unseen domain: web | 10.91 [4.80, 15.60] | 8.78 [3.96, 12.60] | 10.28 [4.52, 14.84] | 6.66 [4.09, 8.93] | 8.25 [3.71, 11.83] | 9.98 [4.43, 15.11] | 8.42 [3.23, 12.85] | 9.09 [3.84, 13.35] | 0.00 [0.00, 0.00] |
| Unseen domains (mean) | 8.44 [5.43, 11.70] | 5.62 [3.72, 7.62] | 7.98 [5.16, 11.07] | 4.39 [3.34, 5.34] | 5.56 [3.97, 7.12] | 7.19 [4.94, 10.34] | 6.16 [3.90, 8.83] | 6.46 [4.14, 9.34] | 0.01 [0.00, 0.20] |

Review budget 5% (no deferral in the first column):

| setting | no deferral (main monitor) | least confident | most novel | stronger monitor, least confident | LightGBM deferrer (task folds) | LightGBM deferrer (domain folds) | transformer deferrer (task folds) | transformer deferrer (domain folds) | perfect (upper bound) |
|---|---|---|---|---|---|---|---|---|---|
| Task-disjoint test | 2.31 [1.44, 3.15] | 0.22 [0.00, 0.52] | 2.12 [1.23, 2.94] | 0.18 [0.00, 0.52] | 0.22 [0.00, 0.52] | — | 0.67 [0.20, 1.22] | — | 0.00 [0.00, 0.00] |
| Tool coverage 100% | 4.89 [0.79, 12.12] | 1.60 [0.34, 4.30] | 3.59 [0.40, 8.07] | 1.28 [0.33, 2.92] | 1.57 [0.00, 5.01] | — | 2.66 [0.00, 9.31] | — | 0.00 [0.00, 0.00] |
| Tool coverage 75% | 5.67 [1.58, 13.46] | 1.95 [0.00, 5.39] | 4.98 [1.14, 12.76] | 1.39 [0.00, 4.25] | 2.10 [0.00, 6.74] | — | 3.10 [0.00, 9.63] | — | 0.00 [0.00, 0.00] |
| Tool coverage 50% | 6.13 [1.54, 12.43] | 1.90 [0.00, 5.37] | 5.41 [1.17, 12.43] | 1.43 [0.00, 4.68] | 2.14 [0.00, 6.39] | — | 3.60 [0.39, 9.91] | — | 0.00 [0.00, 0.00] |
| Tool coverage 25% | 5.43 [2.74, 8.45] | 1.78 [0.35, 3.81] | 4.65 [2.01, 7.69] | 1.46 [0.00, 3.47] | 2.05 [0.34, 4.30] | — | 3.26 [0.77, 6.34] | — | 0.00 [0.00, 0.00] |
| Unseen domain: banking | 6.99 [4.72, 9.45] | 2.39 [1.44, 3.56] | 6.27 [4.22, 8.42] | 2.42 [1.63, 3.36] | 2.24 [1.30, 3.46] | 3.54 [1.80, 6.24] | 3.55 [1.98, 5.48] | 3.70 [2.13, 5.72] | 0.00 [0.00, 0.00] |
| Unseen domain: coding | 14.26 [9.25, 21.02] | 4.91 [3.23, 7.07] | 11.86 [7.21, 17.92] | 3.68 [2.55, 5.32] | 5.63 [4.07, 7.93] | 6.60 [3.77, 10.35] | 7.57 [4.30, 12.29] | 7.73 [4.30, 13.67] | 0.00 [0.00, 0.00] |
| Unseen domain: email | 3.08 [0.53, 6.01] | 1.46 [0.34, 2.82] | 2.99 [0.53, 5.87] | 1.39 [0.60, 2.49] | 1.73 [0.40, 3.39] | 1.90 [0.27, 3.60] | 2.42 [0.47, 4.89] | 2.47 [0.53, 4.48] | 0.00 [0.00, 0.00] |
| Unseen domain: medical | 6.96 [1.42, 10.94] | 2.04 [0.47, 3.75] | 6.31 [1.23, 10.13] | 1.30 [0.54, 2.14] | 1.79 [0.33, 3.87] | 2.84 [0.74, 4.74] | 3.69 [0.61, 7.23] | 3.51 [0.54, 6.33] | 0.00 [0.00, 0.00] |
| Unseen domain: web | 10.91 [4.80, 15.60] | 6.48 [3.46, 9.06] | 9.55 [3.86, 14.17] | 4.74 [2.85, 6.35] | 6.10 [2.37, 9.11] | 7.56 [4.01, 11.86] | 6.54 [2.51, 9.56] | 7.24 [2.88, 11.84] | 0.00 [0.00, 0.00] |
| Unseen domains (mean) | 8.44 [5.43, 11.70] | 3.46 [2.35, 4.49] | 7.40 [4.71, 10.33] | 2.71 [2.05, 3.39] | 3.50 [2.47, 4.36] | 4.49 [3.15, 6.73] | 4.75 [2.94, 6.99] | 4.93 [3.02, 7.57] | 0.00 [0.00, 0.00] |

Learned policies vs. least confident at attack rate natural: difference in missed attacks left per 1,000 calls (negative = the learned one leaves fewer), with the share of resamples where it is better.

| setting | policy | 2% budget | better in | 5% budget | better in |
|---|---|---|---|---|---|
| Task-disjoint test | stronger monitor, least confident | -0.24 [-0.61, 0.10] | 88% | -0.05 [-0.31, 0.31] | 51% |
| Task-disjoint test | LightGBM deferrer (task folds) | -0.07 [-0.51, 0.41] | 52% | 0.00 [-0.10, 0.10] | 6% |
| Task-disjoint test | transformer deferrer (task folds) | 0.14 [-0.21, 0.61] | 19% | 0.45 [0.10, 0.92] | 0% |
| Tool coverage 100% | stronger monitor, least confident | -0.15 [-1.78, 0.80] | 27% | -0.33 [-2.10, 0.67] | 40% |
| Tool coverage 100% | LightGBM deferrer (task folds) | 0.75 [0.00, 3.20] | 1% | -0.03 [-1.90, 1.44] | 34% |
| Tool coverage 100% | transformer deferrer (task folds) | 0.83 [-1.04, 3.61] | 12% | 1.06 [-1.15, 4.93] | 22% |
| Tool coverage 75% | stronger monitor, least confident | 0.09 [-1.43, 1.57] | 33% | -0.56 [-2.54, 0.38] | 68% |
| Tool coverage 75% | LightGBM deferrer (task folds) | 1.05 [-0.74, 4.05] | 10% | 0.15 [-1.12, 2.49] | 34% |
| Tool coverage 75% | transformer deferrer (task folds) | 0.88 [-0.80, 3.54] | 17% | 1.15 [-1.34, 4.98] | 16% |
| Tool coverage 50% | stronger monitor, least confident | 0.18 [-1.43, 2.31] | 32% | -0.47 [-1.80, 0.36] | 66% |
| Tool coverage 50% | LightGBM deferrer (task folds) | 1.21 [-0.77, 4.27] | 8% | 0.24 [-1.69, 2.15] | 33% |
| Tool coverage 50% | transformer deferrer (task folds) | 1.47 [-0.39, 4.27] | 4% | 1.69 [-0.41, 5.33] | 4% |
| Tool coverage 25% | stronger monitor, least confident | 0.55 [-1.08, 2.12] | 19% | -0.33 [-2.37, 1.08] | 41% |
| Tool coverage 25% | LightGBM deferrer (task folds) | 0.91 [-0.79, 2.82] | 7% | 0.26 [-1.39, 2.17] | 35% |
| Tool coverage 25% | transformer deferrer (task folds) | 1.14 [-0.79, 3.21] | 6% | 1.48 [-0.67, 4.33] | 8% |
| Unseen domain: banking | stronger monitor, least confident | -0.17 [-1.22, 0.77] | 56% | 0.03 [-0.78, 0.96] | 43% |
| Unseen domain: banking | LightGBM deferrer (task folds) | 0.07 [-0.69, 0.95] | 42% | -0.15 [-1.04, 0.68] | 59% |
| Unseen domain: banking | LightGBM deferrer (domain folds) | 1.68 [-0.00, 4.13] | 2% | 1.15 [-0.77, 3.78] | 30% |
| Unseen domain: banking | transformer deferrer (task folds) | 0.56 [-0.60, 1.79] | 15% | 1.16 [-0.17, 3.28] | 6% |
| Unseen domain: banking | transformer deferrer (domain folds) | 0.94 [-0.09, 2.22] | 4% | 1.32 [-0.52, 3.33] | 18% |
| Unseen domain: coding | stronger monitor, least confident | -2.20 [-4.95, -0.13] | 99% | -1.23 [-2.43, -0.40] | 100% |
| Unseen domain: coding | LightGBM deferrer (task folds) | 0.55 [-0.61, 1.62] | 22% | 0.72 [-0.47, 2.62] | 19% |
| Unseen domain: coding | LightGBM deferrer (domain folds) | 2.12 [0.47, 3.99] | 0% | 1.69 [-0.20, 3.66] | 6% |
| Unseen domain: coding | transformer deferrer (task folds) | 0.78 [-0.53, 2.41] | 17% | 2.66 [0.27, 5.80] | 0% |
| Unseen domain: coding | transformer deferrer (domain folds) | 1.31 [-0.20, 3.49] | 5% | 2.82 [0.47, 6.71] | 0% |
| Unseen domain: email | stronger monitor, least confident | -0.28 [-1.54, 1.13] | 66% | -0.07 [-1.00, 0.86] | 55% |
| Unseen domain: email | LightGBM deferrer (task folds) | 0.23 [-0.20, 0.73] | 10% | 0.27 [-0.33, 0.93] | 18% |
| Unseen domain: email | LightGBM deferrer (domain folds) | 0.67 [-0.20, 1.75] | 13% | 0.44 [-0.20, 1.27] | 15% |
| Unseen domain: email | transformer deferrer (task folds) | 0.62 [0.00, 1.76] | 0% | 0.96 [0.07, 2.43] | 0% |
| Unseen domain: email | transformer deferrer (domain folds) | 0.72 [0.07, 1.68] | 0% | 1.01 [0.13, 2.08] | 0% |
| Unseen domain: medical | stronger monitor, least confident | -1.36 [-3.18, 0.81] | 67% | -0.74 [-2.19, 0.47] | 74% |
| Unseen domain: medical | LightGBM deferrer (task folds) | -0.59 [-1.70, 0.14] | 90% | -0.25 [-0.88, 0.27] | 78% |
| Unseen domain: medical | LightGBM deferrer (domain folds) | 2.22 [0.34, 4.56] | 0% | 0.80 [-0.07, 2.32] | 4% |
| Unseen domain: medical | transformer deferrer (task folds) | 1.12 [-0.07, 2.57] | 4% | 1.65 [-0.14, 4.80] | 4% |
| Unseen domain: medical | transformer deferrer (domain folds) | 0.96 [-0.20, 2.35] | 15% | 1.46 [-0.20, 3.91] | 8% |
| Unseen domain: web | stronger monitor, least confident | -2.12 [-4.65, 0.61] | 74% | -1.74 [-3.32, -0.13] | 98% |
| Unseen domain: web | LightGBM deferrer (task folds) | -0.53 [-1.69, 0.47] | 83% | -0.38 [-1.59, 1.16] | 69% |
| Unseen domain: web | LightGBM deferrer (domain folds) | 1.21 [0.14, 2.90] | 0% | 1.08 [-0.48, 3.57] | 17% |
| Unseen domain: web | transformer deferrer (task folds) | -0.36 [-1.47, 0.68] | 67% | 0.06 [-1.46, 1.56] | 46% |
| Unseen domain: web | transformer deferrer (domain folds) | 0.31 [-0.66, 1.35] | 35% | 0.76 [-0.96, 3.45] | 38% |
| Unseen domains (mean) | stronger monitor, least confident | -1.23 [-2.71, -0.11] | 100% | -0.75 [-1.28, -0.17] | 100% |
| Unseen domains (mean) | LightGBM deferrer (task folds) | -0.05 [-0.70, 0.45] | 44% | 0.04 [-0.35, 0.47] | 42% |
| Unseen domains (mean) | LightGBM deferrer (domain folds) | 1.58 [0.72, 2.90] | 0% | 1.03 [-0.11, 2.46] | 9% |
| Unseen domains (mean) | transformer deferrer (task folds) | 0.54 [-0.13, 1.46] | 8% | 1.30 [0.39, 2.84] | 0% |
| Unseen domains (mean) | transformer deferrer (domain folds) | 0.85 [0.15, 1.89] | 0% | 1.47 [0.39, 3.36] | 0% |

## Attack rate 5%: missed attacks left per 1,000 calls

Review budget 2% (no deferral in the first column):

| setting | no deferral (main monitor) | least confident | most novel | stronger monitor, least confident | LightGBM deferrer (task folds) | LightGBM deferrer (domain folds) | transformer deferrer (task folds) | transformer deferrer (domain folds) | perfect (upper bound) |
|---|---|---|---|---|---|---|---|---|---|
| Task-disjoint test | 0.67 [0.17, 1.20] | 0.20 [0.00, 0.51] | 0.61 [0.17, 1.18] | 0.13 [0.00, 0.34] | 0.21 [0.00, 0.52] | — | 0.26 [0.00, 0.67] | — | 0.00 [0.00, 0.00] |
| Tool coverage 100% | 1.80 [0.00, 4.44] | 0.73 [0.00, 2.00] | 1.50 [0.00, 3.52] | 0.70 [0.00, 1.56] | 0.92 [0.00, 2.58] | — | 1.11 [0.00, 3.61] | — | 0.00 [0.00, 0.00] |
| Tool coverage 75% | 1.95 [0.41, 4.75] | 0.91 [0.00, 2.91] | 1.78 [0.00, 4.67] | 0.91 [0.00, 2.45] | 1.27 [0.00, 3.90] | — | 1.31 [0.00, 4.22] | — | 0.00 [0.00, 0.00] |
| Tool coverage 50% | 2.01 [0.00, 4.58] | 0.99 [0.00, 2.82] | 1.94 [0.00, 4.57] | 1.09 [0.00, 2.77] | 1.38 [0.00, 3.63] | — | 1.47 [0.00, 4.09] | — | 0.00 [0.00, 0.00] |
| Tool coverage 25% | 1.94 [0.47, 3.58] | 0.97 [0.00, 2.26] | 1.75 [0.45, 3.33] | 1.18 [0.00, 2.48] | 1.29 [0.00, 2.88] | — | 1.40 [0.40, 2.92] | — | 0.00 [0.00, 0.00] |
| Unseen domain: banking | 3.48 [2.18, 4.72] | 1.83 [1.11, 2.58] | 3.35 [2.17, 4.65] | 1.77 [1.11, 2.53] | 1.96 [1.22, 2.83] | 2.54 [1.45, 3.85] | 2.31 [1.32, 3.43] | 2.30 [1.24, 3.44] | 0.00 [0.00, 0.00] |
| Unseen domain: coding | 3.70 [2.04, 5.77] | 1.91 [1.12, 2.68] | 3.40 [1.82, 5.54] | 1.65 [0.81, 2.49] | 2.20 [1.25, 3.23] | 2.36 [1.25, 3.50] | 2.52 [1.26, 4.07] | 2.63 [1.36, 4.41] | 0.00 [0.00, 0.00] |
| Unseen domain: email | 0.76 [0.11, 1.76] | 0.51 [0.00, 1.36] | 0.75 [0.11, 1.71] | 0.45 [0.00, 1.01] | 0.56 [0.11, 1.47] | 0.67 [0.00, 1.65] | 0.67 [0.11, 1.59] | 0.70 [0.11, 1.65] | 0.00 [0.00, 0.00] |
| Unseen domain: medical | 1.87 [0.23, 3.52] | 0.99 [0.11, 2.16] | 1.85 [0.23, 3.49] | 0.70 [0.11, 1.36] | 0.89 [0.11, 2.03] | 1.23 [0.22, 2.50] | 1.35 [0.11, 2.78] | 1.22 [0.11, 2.58] | 0.00 [0.00, 0.00] |
| Unseen domain: web | 2.87 [0.81, 4.90] | 2.29 [0.79, 3.97] | 2.71 [0.81, 4.58] | 1.81 [0.80, 3.09] | 2.18 [0.58, 3.78] | 2.58 [0.79, 4.62] | 2.23 [0.57, 3.88] | 2.41 [0.58, 4.16] | 0.00 [0.00, 0.00] |
| Unseen domains (mean) | 2.54 [1.56, 3.67] | 1.51 [0.97, 2.03] | 2.41 [1.50, 3.48] | 1.28 [0.90, 1.59] | 1.56 [1.08, 2.07] | 1.87 [1.28, 2.77] | 1.82 [1.07, 2.75] | 1.85 [1.16, 2.79] | 0.00 [0.00, 0.00] |

Review budget 5% (no deferral in the first column):

| setting | no deferral (main monitor) | least confident | most novel | stronger monitor, least confident | LightGBM deferrer (task folds) | LightGBM deferrer (domain folds) | transformer deferrer (task folds) | transformer deferrer (domain folds) | perfect (upper bound) |
|---|---|---|---|---|---|---|---|---|---|
| Task-disjoint test | 0.67 [0.17, 1.20] | 0.05 [0.00, 0.17] | 0.61 [0.17, 1.18] | 0.02 [0.00, 0.17] | 0.07 [0.00, 0.34] | — | 0.20 [0.00, 0.51] | — | 0.00 [0.00, 0.00] |
| Tool coverage 100% | 1.80 [0.00, 4.44] | 0.42 [0.00, 1.05] | 1.32 [0.00, 3.49] | 0.23 [0.00, 1.02] | 0.43 [0.00, 1.51] | — | 0.76 [0.00, 3.07] | — | 0.00 [0.00, 0.00] |
| Tool coverage 75% | 1.95 [0.41, 4.75] | 0.52 [0.00, 1.50] | 1.77 [0.00, 4.63] | 0.30 [0.00, 0.97] | 0.65 [0.00, 2.40] | — | 0.96 [0.00, 3.75] | — | 0.00 [0.00, 0.00] |
| Tool coverage 50% | 2.01 [0.00, 4.58] | 0.54 [0.00, 1.83] | 1.78 [0.00, 4.57] | 0.37 [0.00, 1.79] | 0.62 [0.00, 2.26] | — | 1.14 [0.00, 3.57] | — | 0.00 [0.00, 0.00] |
| Tool coverage 25% | 1.94 [0.47, 3.58] | 0.63 [0.00, 1.64] | 1.62 [0.44, 3.30] | 0.50 [0.00, 1.63] | 0.76 [0.00, 2.26] | — | 1.16 [0.00, 2.69] | — | 0.00 [0.00, 0.00] |
| Unseen domain: banking | 3.48 [2.18, 4.72] | 0.96 [0.33, 1.64] | 3.14 [1.99, 4.33] | 1.00 [0.44, 1.56] | 0.90 [0.44, 1.56] | 1.35 [0.66, 2.22] | 1.66 [0.87, 2.82] | 1.59 [0.78, 2.69] | 0.00 [0.00, 0.00] |
| Unseen domain: coding | 3.70 [2.04, 5.77] | 1.12 [0.45, 1.82] | 3.10 [1.60, 5.14] | 0.83 [0.34, 1.46] | 1.33 [0.58, 1.98] | 1.44 [0.57, 2.51] | 1.93 [0.80, 3.28] | 1.93 [0.79, 3.62] | 0.00 [0.00, 0.00] |
| Unseen domain: email | 0.76 [0.11, 1.76] | 0.38 [0.00, 1.02] | 0.74 [0.11, 1.71] | 0.37 [0.00, 0.90] | 0.45 [0.00, 1.14] | 0.49 [0.00, 1.37] | 0.61 [0.00, 1.58] | 0.64 [0.11, 1.56] | 0.00 [0.00, 0.00] |
| Unseen domain: medical | 1.87 [0.23, 3.52] | 0.54 [0.00, 1.38] | 1.73 [0.22, 3.29] | 0.40 [0.11, 0.80] | 0.50 [0.00, 1.36] | 0.66 [0.11, 1.48] | 1.00 [0.11, 2.23] | 0.89 [0.11, 2.03] | 0.00 [0.00, 0.00] |
| Unseen domain: web | 2.87 [0.81, 4.90] | 1.65 [0.58, 2.97] | 2.54 [0.79, 4.32] | 1.32 [0.46, 2.29] | 1.60 [0.34, 2.92] | 1.90 [0.69, 3.69] | 1.74 [0.45, 3.21] | 1.90 [0.57, 3.55] | 0.00 [0.00, 0.00] |
| Unseen domains (mean) | 2.54 [1.56, 3.67] | 0.93 [0.60, 1.22] | 2.25 [1.36, 3.28] | 0.78 [0.54, 1.03] | 0.95 [0.63, 1.31] | 1.17 [0.75, 1.75] | 1.39 [0.81, 2.17] | 1.39 [0.77, 2.30] | 0.00 [0.00, 0.00] |

Learned policies vs. least confident at attack rate 5%: difference in missed attacks left per 1,000 calls (negative = the learned one leaves fewer), with the share of resamples where it is better.

| setting | policy | 2% budget | better in | 5% budget | better in |
|---|---|---|---|---|---|
| Task-disjoint test | stronger monitor, least confident | -0.07 [-0.34, 0.00] | 36% | -0.03 [-0.17, 0.00] | 20% |
| Task-disjoint test | LightGBM deferrer (task folds) | 0.01 [-0.17, 0.17] | 20% | 0.02 [-0.16, 0.17] | 3% |
| Task-disjoint test | transformer deferrer (task folds) | 0.06 [-0.17, 0.34] | 4% | 0.15 [0.00, 0.51] | 0% |
| Tool coverage 100% | stronger monitor, least confident | -0.02 [-0.53, 0.52] | 14% | -0.19 [-1.02, 0.44] | 38% |
| Tool coverage 100% | LightGBM deferrer (task folds) | 0.19 [-0.50, 1.06] | 4% | 0.02 [-0.55, 1.05] | 25% |
| Tool coverage 100% | transformer deferrer (task folds) | 0.38 [-0.44, 2.02] | 9% | 0.34 [-0.52, 2.65] | 23% |
| Tool coverage 75% | stronger monitor, least confident | -0.01 [-0.96, 0.84] | 22% | -0.22 [-0.98, 0.48] | 41% |
| Tool coverage 75% | LightGBM deferrer (task folds) | 0.35 [-0.49, 1.47] | 4% | 0.13 [-0.49, 0.99] | 15% |
| Tool coverage 75% | transformer deferrer (task folds) | 0.40 [-0.41, 1.88] | 7% | 0.44 [-0.49, 2.40] | 14% |
| Tool coverage 50% | stronger monitor, least confident | 0.10 [-0.48, 0.95] | 16% | -0.17 [-0.50, 0.45] | 36% |
| Tool coverage 50% | LightGBM deferrer (task folds) | 0.39 [-0.41, 1.79] | 4% | 0.08 [-0.48, 0.94] | 22% |
| Tool coverage 50% | transformer deferrer (task folds) | 0.48 [-0.48, 1.90] | 8% | 0.60 [0.00, 2.24] | 2% |
| Tool coverage 25% | stronger monitor, least confident | 0.21 [-0.49, 0.98] | 16% | -0.13 [-0.98, 0.46] | 34% |
| Tool coverage 25% | LightGBM deferrer (task folds) | 0.33 [-0.95, 1.34] | 12% | 0.13 [-0.81, 1.36] | 24% |
| Tool coverage 25% | transformer deferrer (task folds) | 0.43 [-0.49, 1.75] | 9% | 0.53 [-0.48, 1.82] | 8% |
| Unseen domain: banking | stronger monitor, least confident | -0.06 [-0.56, 0.34] | 52% | 0.05 [-0.33, 0.44] | 28% |
| Unseen domain: banking | LightGBM deferrer (task folds) | 0.13 [-0.33, 0.65] | 22% | -0.06 [-0.45, 0.33] | 51% |
| Unseen domain: banking | LightGBM deferrer (domain folds) | 0.71 [-0.22, 1.98] | 10% | 0.40 [-0.22, 1.55] | 16% |
| Unseen domain: banking | transformer deferrer (task folds) | 0.48 [-0.22, 1.53] | 10% | 0.71 [-0.11, 2.08] | 4% |
| Unseen domain: banking | transformer deferrer (domain folds) | 0.48 [-0.33, 1.54] | 8% | 0.63 [-0.11, 1.85] | 6% |
| Unseen domain: coding | stronger monitor, least confident | -0.26 [-0.79, 0.23] | 80% | -0.29 [-0.79, 0.11] | 88% |
| Unseen domain: coding | LightGBM deferrer (task folds) | 0.29 [-0.23, 0.91] | 11% | 0.21 [-0.22, 0.67] | 6% |
| Unseen domain: coding | LightGBM deferrer (domain folds) | 0.45 [-0.23, 1.24] | 9% | 0.32 [-0.23, 1.26] | 12% |
| Unseen domain: coding | transformer deferrer (task folds) | 0.61 [-0.23, 1.98] | 10% | 0.81 [-0.11, 2.18] | 4% |
| Unseen domain: coding | transformer deferrer (domain folds) | 0.72 [-0.11, 2.09] | 3% | 0.81 [0.00, 2.16] | 0% |
| Unseen domain: email | stronger monitor, least confident | -0.06 [-0.67, 0.46] | 44% | -0.02 [-0.56, 0.45] | 38% |
| Unseen domain: email | LightGBM deferrer (task folds) | 0.05 [-0.11, 0.33] | 14% | 0.06 [-0.22, 0.34] | 15% |
| Unseen domain: email | LightGBM deferrer (domain folds) | 0.15 [-0.11, 0.56] | 7% | 0.11 [-0.11, 0.45] | 16% |
| Unseen domain: email | transformer deferrer (task folds) | 0.16 [-0.11, 0.68] | 3% | 0.23 [-0.11, 0.80] | 3% |
| Unseen domain: email | transformer deferrer (domain folds) | 0.19 [0.00, 0.67] | 2% | 0.25 [0.00, 0.79] | 1% |
| Unseen domain: medical | stronger monitor, least confident | -0.29 [-1.03, 0.34] | 68% | -0.13 [-0.80, 0.23] | 57% |
| Unseen domain: medical | LightGBM deferrer (task folds) | -0.10 [-0.56, 0.23] | 52% | -0.04 [-0.34, 0.23] | 42% |
| Unseen domain: medical | LightGBM deferrer (domain folds) | 0.24 [0.00, 0.67] | 2% | 0.12 [-0.23, 0.57] | 16% |
| Unseen domain: medical | transformer deferrer (task folds) | 0.36 [-0.11, 1.04] | 7% | 0.46 [-0.22, 1.45] | 10% |
| Unseen domain: medical | transformer deferrer (domain folds) | 0.23 [-0.12, 0.81] | 12% | 0.36 [-0.12, 1.15] | 12% |
| Unseen domain: web | stronger monitor, least confident | -0.48 [-1.38, 0.35] | 74% | -0.34 [-1.13, 0.23] | 76% |
| Unseen domain: web | LightGBM deferrer (task folds) | -0.11 [-0.68, 0.35] | 61% | -0.06 [-0.69, 0.59] | 54% |
| Unseen domain: web | LightGBM deferrer (domain folds) | 0.29 [-0.34, 1.02] | 10% | 0.25 [-0.24, 0.92] | 13% |
| Unseen domain: web | transformer deferrer (task folds) | -0.06 [-0.58, 0.57] | 55% | 0.08 [-0.47, 0.92] | 40% |
| Unseen domain: web | transformer deferrer (domain folds) | 0.12 [-0.35, 0.69] | 27% | 0.25 [-0.35, 1.16] | 30% |
| Unseen domains (mean) | stronger monitor, least confident | -0.23 [-0.59, 0.14] | 86% | -0.15 [-0.39, 0.04] | 93% |
| Unseen domains (mean) | LightGBM deferrer (task folds) | 0.05 [-0.18, 0.27] | 36% | 0.02 [-0.14, 0.21] | 41% |
| Unseen domains (mean) | LightGBM deferrer (domain folds) | 0.37 [-0.05, 0.83] | 6% | 0.24 [-0.09, 0.67] | 12% |
| Unseen domains (mean) | transformer deferrer (task folds) | 0.31 [-0.00, 0.86] | 4% | 0.46 [0.05, 1.17] | 2% |
| Unseen domains (mean) | transformer deferrer (domain folds) | 0.35 [0.00, 0.86] | 2% | 0.46 [0.05, 1.21] | 0% |

## Attack rate 1%: missed attacks left per 1,000 calls

Review budget 2% (no deferral in the first column):

| setting | no deferral (main monitor) | least confident | most novel | stronger monitor, least confident | LightGBM deferrer (task folds) | LightGBM deferrer (domain folds) | transformer deferrer (task folds) | transformer deferrer (domain folds) | perfect (upper bound) |
|---|---|---|---|---|---|---|---|---|---|
| Task-disjoint test | 0.13 [0.00, 0.56] | 0.03 [0.00, 0.19] | 0.12 [0.00, 0.39] | 0.02 [0.00, 0.19] | 0.04 [0.00, 0.19] | — | 0.05 [0.00, 0.37] | — | 0.00 [0.00, 0.00] |
| Tool coverage 100% | 0.30 [0.00, 1.18] | 0.12 [0.00, 0.93] | 0.25 [0.00, 1.15] | 0.10 [0.00, 0.59] | 0.13 [0.00, 0.93] | — | 0.16 [0.00, 0.99] | — | 0.00 [0.00, 0.00] |
| Tool coverage 75% | 0.36 [0.00, 1.35] | 0.17 [0.00, 1.04] | 0.31 [0.00, 1.35] | 0.14 [0.00, 1.04] | 0.23 [0.00, 1.06] | — | 0.24 [0.00, 1.07] | — | 0.00 [0.00, 0.00] |
| Tool coverage 50% | 0.40 [0.00, 1.49] | 0.19 [0.00, 0.98] | 0.38 [0.00, 1.47] | 0.20 [0.00, 0.98] | 0.27 [0.00, 1.35] | — | 0.29 [0.00, 1.44] | — | 0.00 [0.00, 0.00] |
| Tool coverage 25% | 0.41 [0.00, 1.43] | 0.20 [0.00, 0.95] | 0.38 [0.00, 1.43] | 0.25 [0.00, 0.97] | 0.26 [0.00, 1.00] | — | 0.28 [0.00, 1.35] | — | 0.00 [0.00, 0.00] |
| Unseen domain: banking | 0.71 [0.13, 1.28] | 0.31 [0.00, 0.78] | 0.68 [0.13, 1.28] | 0.33 [0.00, 0.78] | 0.37 [0.00, 0.90] | 0.44 [0.00, 1.01] | 0.45 [0.00, 1.16] | 0.42 [0.00, 1.03] | 0.00 [0.00, 0.00] |
| Unseen domain: coding | 0.77 [0.13, 1.53] | 0.36 [0.00, 0.89] | 0.70 [0.13, 1.40] | 0.31 [0.00, 0.87] | 0.42 [0.00, 1.02] | 0.44 [0.00, 1.02] | 0.52 [0.00, 1.15] | 0.53 [0.00, 1.15] | 0.00 [0.00, 0.00] |
| Unseen domain: email | 0.16 [0.00, 0.74] | 0.11 [0.00, 0.51] | 0.16 [0.00, 0.74] | 0.10 [0.00, 0.50] | 0.12 [0.00, 0.61] | 0.14 [0.00, 0.63] | 0.14 [0.00, 0.63] | 0.15 [0.00, 0.62] | 0.00 [0.00, 0.00] |
| Unseen domain: medical | 0.37 [0.00, 1.26] | 0.19 [0.00, 0.75] | 0.37 [0.00, 1.14] | 0.13 [0.00, 0.51] | 0.18 [0.00, 0.63] | 0.23 [0.00, 0.76] | 0.26 [0.00, 0.89] | 0.23 [0.00, 0.88] | 0.00 [0.00, 0.00] |
| Unseen domain: web | 0.57 [0.00, 1.31] | 0.45 [0.00, 1.02] | 0.54 [0.00, 1.29] | 0.36 [0.00, 0.78] | 0.43 [0.00, 1.03] | 0.51 [0.00, 1.28] | 0.44 [0.00, 1.04] | 0.47 [0.00, 1.16] | 0.00 [0.00, 0.00] |
| Unseen domains (mean) | 0.52 [0.23, 0.87] | 0.28 [0.13, 0.51] | 0.49 [0.23, 0.84] | 0.25 [0.08, 0.44] | 0.30 [0.15, 0.51] | 0.35 [0.13, 0.64] | 0.36 [0.13, 0.66] | 0.36 [0.13, 0.67] | 0.00 [0.00, 0.00] |

Review budget 5% (no deferral in the first column):

| setting | no deferral (main monitor) | least confident | most novel | stronger monitor, least confident | LightGBM deferrer (task folds) | LightGBM deferrer (domain folds) | transformer deferrer (task folds) | transformer deferrer (domain folds) | perfect (upper bound) |
|---|---|---|---|---|---|---|---|---|---|
| Task-disjoint test | 0.13 [0.00, 0.56] | 0.01 [0.00, 0.19] | 0.12 [0.00, 0.39] | 0.00 [0.00, 0.00] | 0.01 [0.00, 0.19] | — | 0.03 [0.00, 0.19] | — | 0.00 [0.00, 0.00] |
| Tool coverage 100% | 0.30 [0.00, 1.18] | 0.06 [0.00, 0.57] | 0.22 [0.00, 1.15] | 0.03 [0.00, 0.56] | 0.06 [0.00, 0.58] | — | 0.10 [0.00, 0.95] | — | 0.00 [0.00, 0.00] |
| Tool coverage 75% | 0.36 [0.00, 1.35] | 0.09 [0.00, 0.54] | 0.31 [0.00, 1.35] | 0.05 [0.00, 0.53] | 0.11 [0.00, 0.54] | — | 0.17 [0.00, 1.06] | — | 0.00 [0.00, 0.00] |
| Tool coverage 50% | 0.40 [0.00, 1.49] | 0.08 [0.00, 0.50] | 0.35 [0.00, 1.47] | 0.08 [0.00, 0.50] | 0.12 [0.00, 0.90] | — | 0.22 [0.00, 1.02] | — | 0.00 [0.00, 0.00] |
| Tool coverage 25% | 0.41 [0.00, 1.43] | 0.15 [0.00, 0.90] | 0.35 [0.00, 1.35] | 0.13 [0.00, 0.90] | 0.16 [0.00, 0.95] | — | 0.22 [0.00, 0.97] | — | 0.00 [0.00, 0.00] |
| Unseen domain: banking | 0.71 [0.13, 1.28] | 0.18 [0.00, 0.53] | 0.65 [0.13, 1.18] | 0.18 [0.00, 0.52] | 0.14 [0.00, 0.52] | 0.22 [0.00, 0.65] | 0.31 [0.00, 0.90] | 0.28 [0.00, 0.79] | 0.00 [0.00, 0.00] |
| Unseen domain: coding | 0.77 [0.13, 1.53] | 0.21 [0.00, 0.52] | 0.64 [0.13, 1.38] | 0.14 [0.00, 0.51] | 0.25 [0.00, 0.64] | 0.26 [0.00, 0.76] | 0.39 [0.00, 1.01] | 0.38 [0.00, 1.01] | 0.00 [0.00, 0.00] |
| Unseen domain: email | 0.16 [0.00, 0.74] | 0.09 [0.00, 0.50] | 0.16 [0.00, 0.74] | 0.08 [0.00, 0.38] | 0.10 [0.00, 0.50] | 0.10 [0.00, 0.51] | 0.13 [0.00, 0.62] | 0.14 [0.00, 0.62] | 0.00 [0.00, 0.00] |
| Unseen domain: medical | 0.37 [0.00, 1.26] | 0.12 [0.00, 0.51] | 0.35 [0.00, 1.14] | 0.08 [0.00, 0.38] | 0.11 [0.00, 0.51] | 0.13 [0.00, 0.51] | 0.20 [0.00, 0.76] | 0.16 [0.00, 0.63] | 0.00 [0.00, 0.00] |
| Unseen domain: web | 0.57 [0.00, 1.31] | 0.33 [0.00, 0.77] | 0.50 [0.00, 1.28] | 0.26 [0.00, 0.64] | 0.31 [0.00, 0.78] | 0.37 [0.00, 0.89] | 0.34 [0.00, 0.91] | 0.36 [0.00, 0.90] | 0.00 [0.00, 0.00] |
| Unseen domains (mean) | 0.52 [0.23, 0.87] | 0.19 [0.05, 0.36] | 0.46 [0.21, 0.82] | 0.15 [0.03, 0.28] | 0.18 [0.05, 0.33] | 0.22 [0.08, 0.43] | 0.27 [0.10, 0.53] | 0.26 [0.08, 0.54] | 0.00 [0.00, 0.00] |

Learned policies vs. least confident at attack rate 1%: difference in missed attacks left per 1,000 calls (negative = the learned one leaves fewer), with the share of resamples where it is better.

| setting | policy | 2% budget | better in | 5% budget | better in |
|---|---|---|---|---|---|
| Task-disjoint test | stronger monitor, least confident | -0.01 [-0.19, 0.00] | 6% | -0.01 [-0.19, 0.00] | 3% |
| Task-disjoint test | LightGBM deferrer (task folds) | 0.01 [-0.19, 0.19] | 5% | 0.00 [0.00, 0.00] | 0% |
| Task-disjoint test | transformer deferrer (task folds) | 0.01 [0.00, 0.19] | 0% | 0.02 [0.00, 0.19] | 0% |
| Tool coverage 100% | stronger monitor, least confident | -0.01 [-0.48, 0.00] | 4% | -0.03 [-0.50, 0.00] | 6% |
| Tool coverage 100% | LightGBM deferrer (task folds) | 0.01 [0.00, 0.56] | 2% | 0.00 [-0.57, 0.58] | 4% |
| Tool coverage 100% | transformer deferrer (task folds) | 0.04 [-0.48, 0.58] | 4% | 0.04 [-0.50, 0.59] | 5% |
| Tool coverage 75% | stronger monitor, least confident | -0.03 [-0.53, 0.52] | 10% | -0.04 [-0.54, 0.00] | 10% |
| Tool coverage 75% | LightGBM deferrer (task folds) | 0.05 [-0.52, 0.53] | 3% | 0.03 [-0.53, 0.53] | 5% |
| Tool coverage 75% | transformer deferrer (task folds) | 0.07 [0.00, 0.53] | 2% | 0.08 [-0.45, 1.04] | 4% |
| Tool coverage 50% | stronger monitor, least confident | 0.01 [-0.49, 0.50] | 6% | -0.01 [-0.01, 0.00] | 2% |
| Tool coverage 50% | LightGBM deferrer (task folds) | 0.08 [0.00, 0.53] | 2% | 0.03 [-0.45, 0.51] | 4% |
| Tool coverage 50% | transformer deferrer (task folds) | 0.10 [-0.50, 0.97] | 4% | 0.14 [0.00, 1.00] | 2% |
| Tool coverage 25% | stronger monitor, least confident | 0.05 [-0.50, 0.52] | 4% | -0.02 [-0.51, 0.48] | 7% |
| Tool coverage 25% | LightGBM deferrer (task folds) | 0.06 [-0.49, 0.51] | 4% | 0.01 [-0.45, 0.48] | 5% |
| Tool coverage 25% | transformer deferrer (task folds) | 0.08 [-0.49, 0.52] | 6% | 0.08 [-0.46, 0.51] | 6% |
| Unseen domain: banking | stronger monitor, least confident | 0.02 [-0.26, 0.26] | 16% | 0.00 [-0.26, 0.13] | 20% |
| Unseen domain: banking | LightGBM deferrer (task folds) | 0.06 [-0.13, 0.26] | 15% | -0.04 [-0.26, 0.13] | 34% |
| Unseen domain: banking | LightGBM deferrer (domain folds) | 0.13 [-0.13, 0.53] | 8% | 0.05 [-0.13, 0.39] | 16% |
| Unseen domain: banking | transformer deferrer (task folds) | 0.14 [-0.13, 0.65] | 14% | 0.14 [-0.13, 0.64] | 12% |
| Unseen domain: banking | transformer deferrer (domain folds) | 0.11 [-0.13, 0.64] | 10% | 0.10 [-0.13, 0.65] | 14% |
| Unseen domain: coding | stronger monitor, least confident | -0.05 [-0.38, 0.25] | 40% | -0.07 [-0.26, 0.13] | 48% |
| Unseen domain: coding | LightGBM deferrer (task folds) | 0.06 [-0.13, 0.38] | 14% | 0.04 [-0.13, 0.25] | 6% |
| Unseen domain: coding | LightGBM deferrer (domain folds) | 0.08 [-0.25, 0.50] | 16% | 0.05 [-0.25, 0.39] | 17% |
| Unseen domain: coding | transformer deferrer (task folds) | 0.16 [-0.13, 0.64] | 13% | 0.18 [-0.13, 0.76] | 10% |
| Unseen domain: coding | transformer deferrer (domain folds) | 0.17 [-0.13, 0.76] | 7% | 0.17 [-0.13, 0.76] | 4% |
| Unseen domain: email | stronger monitor, least confident | -0.01 [-0.25, 0.13] | 22% | -0.01 [-0.25, 0.13] | 17% |
| Unseen domain: email | LightGBM deferrer (task folds) | 0.01 [-0.12, 0.13] | 4% | 0.01 [-0.12, 0.13] | 5% |
| Unseen domain: email | LightGBM deferrer (domain folds) | 0.03 [-0.12, 0.25] | 4% | 0.02 [-0.13, 0.13] | 6% |
| Unseen domain: email | transformer deferrer (task folds) | 0.03 [-0.12, 0.25] | 4% | 0.04 [-0.00, 0.25] | 2% |
| Unseen domain: email | transformer deferrer (domain folds) | 0.04 [0.00, 0.25] | 2% | 0.05 [0.00, 0.25] | 2% |
| Unseen domain: medical | stronger monitor, least confident | -0.06 [-0.38, 0.13] | 31% | -0.04 [-0.38, 0.13] | 26% |
| Unseen domain: medical | LightGBM deferrer (task folds) | -0.01 [-0.25, 0.13] | 20% | -0.01 [-0.13, 0.13] | 16% |
| Unseen domain: medical | LightGBM deferrer (domain folds) | 0.05 [-0.13, 0.25] | 4% | 0.01 [-0.13, 0.13] | 14% |
| Unseen domain: medical | transformer deferrer (task folds) | 0.08 [-0.13, 0.38] | 6% | 0.08 [-0.13, 0.38] | 5% |
| Unseen domain: medical | transformer deferrer (domain folds) | 0.04 [-0.13, 0.26] | 8% | 0.04 [-0.13, 0.26] | 6% |
| Unseen domain: web | stronger monitor, least confident | -0.09 [-0.51, 0.13] | 46% | -0.08 [-0.51, 0.25] | 50% |
| Unseen domain: web | LightGBM deferrer (task folds) | -0.01 [-0.26, 0.26] | 29% | -0.02 [-0.27, 0.26] | 32% |
| Unseen domain: web | LightGBM deferrer (domain folds) | 0.06 [-0.13, 0.39] | 10% | 0.04 [-0.26, 0.38] | 14% |
| Unseen domain: web | transformer deferrer (task folds) | -0.01 [-0.26, 0.38] | 30% | 0.01 [-0.26, 0.39] | 30% |
| Unseen domain: web | transformer deferrer (domain folds) | 0.02 [-0.26, 0.26] | 18% | 0.03 [-0.26, 0.39] | 24% |
| Unseen domains (mean) | stronger monitor, least confident | -0.04 [-0.15, 0.08] | 67% | -0.04 [-0.15, 0.05] | 76% |
| Unseen domains (mean) | LightGBM deferrer (task folds) | 0.02 [-0.05, 0.13] | 32% | -0.00 [-0.08, 0.08] | 54% |
| Unseen domains (mean) | LightGBM deferrer (domain folds) | 0.07 [-0.05, 0.23] | 14% | 0.03 [-0.05, 0.15] | 27% |
| Unseen domains (mean) | transformer deferrer (task folds) | 0.08 [-0.05, 0.30] | 16% | 0.09 [-0.05, 0.28] | 13% |
| Unseen domains (mean) | transformer deferrer (domain folds) | 0.08 [-0.03, 0.26] | 10% | 0.08 [-0.05, 0.28] | 16% |

## Review burden at attack rate 2%

Share of reviews spent on calls the monitor already had right (a 5% budget):

| setting | least confident | most novel | stronger monitor, least confident | LightGBM deferrer (task folds) | LightGBM deferrer (domain folds) | transformer deferrer (task folds) | transformer deferrer (domain folds) | perfect (upper bound) |
|---|---|---|---|---|---|---|---|---|
| Task-disjoint test | 93% | 98% | 95% | 93% | — | 94% | — | 93% |
| Tool coverage 100% | 92% | 98% | 94% | 92% | — | 93% | — | 92% |
| Tool coverage 75% | 87% | 96% | 89% | 89% | — | 91% | — | 86% |
| Tool coverage 50% | 86% | 98% | 87% | 88% | — | 91% | — | 83% |
| Tool coverage 25% | 86% | 96% | 86% | 89% | — | 90% | — | 82% |
| Unseen domain: banking | 88% | 99% | 90% | 89% | 88% | 90% | 89% | 87% |
| Unseen domain: coding | 89% | 99% | 88% | 89% | 89% | 92% | 90% | 88% |
| Unseen domain: email | 66% | 94% | 66% | 79% | 52% | 75% | 66% | 37% |
| Unseen domain: medical | 79% | 97% | 74% | 85% | 69% | 80% | 78% | 66% |
| Unseen domain: web | 63% | 91% | 56% | 86% | 51% | 71% | 61% | 36% |
| Unseen domains (mean) | 77% | 96% | 75% | 86% | 70% | 82% | 77% | 63% |

Figures: `results/figures/09_risk_vs_escalation.png`, `results/figures/10_domain_fold_labels.png`. Every budget: `reports/week3/03_deferral_eval_full.md`.
