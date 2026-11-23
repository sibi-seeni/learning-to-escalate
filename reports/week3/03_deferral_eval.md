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
| Unseen domain: banking | 1 | 11634 | 1170 | 160 | 67 |
| Unseen domain: coding | 1 | 14817 | 2808 | 262 | 198 |
| Unseen domain: email | 1 | 14986 | 2862 | 394 | 50 |
| Unseen domain: medical | 1 | 14795 | 2758 | 416 | 28 |
| Unseen domain: web | 1 | 14792 | 2751 | 975 | 81 |
| Unseen domains (mean) | 5 | 14205 | 2470 | 441 | 85 |

## Attack rate 2%: missed attacks left per 1,000 calls

Review budget 2% (no deferral in the first column):

| setting | no deferral (main monitor) | least confident | most novel | stronger monitor, least confident | LightGBM deferrer (task folds) | LightGBM deferrer (domain folds) | transformer deferrer (task folds) | transformer deferrer (domain folds) | perfect (upper bound) |
|---|---|---|---|---|---|---|---|---|---|
| Task-disjoint test | 0.26 [0.00, 0.74] | 0.07 [0.00, 0.37] | 0.25 [0.00, 0.73] | 0.05 [0.00, 0.19] | 0.08 [0.00, 0.37] | — | 0.10 [0.00, 0.37] | — | 0.00 [0.00, 0.00] |
| Tool coverage 100% | 0.66 [0.00, 2.31] | 0.24 [0.00, 1.11] | 0.56 [0.00, 2.22] | 0.22 [0.00, 1.11] | 0.29 [0.00, 1.40] | — | 0.38 [0.00, 1.67] | — | 0.00 [0.00, 0.00] |
| Tool coverage 75% | 0.70 [0.00, 2.09] | 0.34 [0.00, 1.52] | 0.63 [0.00, 2.04] | 0.30 [0.00, 1.04] | 0.46 [0.00, 1.60] | — | 0.47 [0.00, 2.03] | — | 0.00 [0.00, 0.00] |
| Tool coverage 50% | 0.82 [0.00, 2.43] | 0.40 [0.00, 1.47] | 0.79 [0.00, 2.43] | 0.43 [0.00, 1.45] | 0.56 [0.00, 1.95] | — | 0.58 [0.00, 2.00] | — | 0.00 [0.00, 0.00] |
| Tool coverage 25% | 0.79 [0.00, 1.91] | 0.38 [0.00, 1.29] | 0.73 [0.00, 1.87] | 0.48 [0.00, 1.44] | 0.53 [0.00, 1.44] | — | 0.57 [0.00, 1.72] | — | 0.00 [0.00, 0.00] |
| Unseen domain: banking | 1.13 [0.50, 1.87] | 0.63 [0.12, 1.36] | 1.09 [0.50, 1.84] | 0.70 [0.25, 1.27] | 0.77 [0.25, 1.38] | 0.90 [0.37, 1.61] | 0.76 [0.25, 1.49] | 0.77 [0.25, 1.50] | 0.00 [0.00, 0.00] |
| Unseen domain: coding | 1.41 [0.73, 2.43] | 0.75 [0.25, 1.47] | 1.29 [0.61, 2.29] | 0.68 [0.24, 1.36] | 0.97 [0.37, 1.84] | 0.94 [0.36, 1.82] | 0.94 [0.37, 1.82] | 0.96 [0.25, 1.72] | 0.00 [0.00, 0.00] |
| Unseen domain: email | 0.36 [0.00, 0.99] | 0.25 [0.00, 0.84] | 0.35 [0.00, 0.99] | 0.20 [0.00, 0.73] | 0.30 [0.00, 0.86] | 0.34 [0.00, 0.98] | 0.32 [0.00, 0.98] | 0.35 [0.00, 0.99] | 0.00 [0.00, 0.00] |
| Unseen domain: medical | 0.20 [0.00, 0.61] | 0.13 [0.00, 0.38] | 0.20 [0.00, 0.61] | 0.19 [0.00, 0.61] | 0.13 [0.00, 0.49] | 0.19 [0.00, 0.50] | 0.16 [0.00, 0.49] | 0.14 [0.00, 0.48] | 0.00 [0.00, 0.00] |
| Unseen domain: web | 0.58 [0.13, 1.14] | 0.52 [0.13, 1.02] | 0.56 [0.12, 1.14] | 0.55 [0.12, 1.12] | 0.47 [0.12, 1.00] | 0.57 [0.12, 1.14] | 0.43 [0.12, 0.90] | 0.50 [0.12, 1.01] | 0.00 [0.00, 0.00] |
| Unseen domains (mean) | 0.74 [0.47, 1.05] | 0.46 [0.25, 0.70] | 0.70 [0.45, 1.00] | 0.46 [0.27, 0.69] | 0.53 [0.30, 0.77] | 0.59 [0.35, 0.84] | 0.52 [0.32, 0.80] | 0.54 [0.30, 0.82] | 0.00 [0.00, 0.00] |

Review budget 5% (no deferral in the first column):

| setting | no deferral (main monitor) | least confident | most novel | stronger monitor, least confident | LightGBM deferrer (task folds) | LightGBM deferrer (domain folds) | transformer deferrer (task folds) | transformer deferrer (domain folds) | perfect (upper bound) |
|---|---|---|---|---|---|---|---|---|---|
| Task-disjoint test | 0.26 [0.00, 0.74] | 0.02 [0.00, 0.18] | 0.25 [0.00, 0.73] | 0.01 [0.00, 0.18] | 0.03 [0.00, 0.19] | — | 0.08 [0.00, 0.37] | — | 0.00 [0.00, 0.00] |
| Tool coverage 100% | 0.66 [0.00, 2.31] | 0.14 [0.00, 0.93] | 0.49 [0.00, 2.22] | 0.08 [0.00, 0.58] | 0.15 [0.00, 0.94] | — | 0.24 [0.00, 1.64] | — | 0.00 [0.00, 0.00] |
| Tool coverage 75% | 0.70 [0.00, 2.09] | 0.18 [0.00, 1.00] | 0.63 [0.00, 2.04] | 0.10 [0.00, 0.53] | 0.25 [0.00, 1.04] | — | 0.34 [0.00, 1.52] | — | 0.00 [0.00, 0.00] |
| Tool coverage 50% | 0.82 [0.00, 2.43] | 0.19 [0.00, 0.96] | 0.74 [0.00, 2.38] | 0.14 [0.00, 0.88] | 0.24 [0.00, 0.99] | — | 0.44 [0.00, 1.54] | — | 0.00 [0.00, 0.00] |
| Tool coverage 25% | 0.79 [0.00, 1.91] | 0.27 [0.00, 0.98] | 0.67 [0.00, 1.87] | 0.21 [0.00, 0.94] | 0.32 [0.00, 1.40] | — | 0.47 [0.00, 1.44] | — | 0.00 [0.00, 0.00] |
| Unseen domain: banking | 1.13 [0.50, 1.87] | 0.41 [0.12, 0.88] | 1.04 [0.49, 1.74] | 0.36 [0.12, 0.74] | 0.38 [0.00, 0.88] | 0.48 [0.00, 1.02] | 0.51 [0.12, 1.11] | 0.53 [0.12, 1.13] | 0.00 [0.00, 0.00] |
| Unseen domain: coding | 1.41 [0.73, 2.43] | 0.40 [0.00, 0.86] | 1.19 [0.50, 2.25] | 0.31 [0.00, 0.74] | 0.50 [0.12, 1.02] | 0.44 [0.12, 0.87] | 0.68 [0.24, 1.37] | 0.57 [0.12, 1.22] | 0.00 [0.00, 0.00] |
| Unseen domain: email | 0.36 [0.00, 0.99] | 0.20 [0.00, 0.73] | 0.35 [0.00, 0.99] | 0.16 [0.00, 0.61] | 0.28 [0.00, 0.86] | 0.29 [0.00, 0.86] | 0.29 [0.00, 0.97] | 0.33 [0.00, 0.99] | 0.00 [0.00, 0.00] |
| Unseen domain: medical | 0.20 [0.00, 0.61] | 0.11 [0.00, 0.37] | 0.17 [0.00, 0.49] | 0.14 [0.00, 0.37] | 0.09 [0.00, 0.37] | 0.14 [0.00, 0.49] | 0.11 [0.00, 0.37] | 0.10 [0.00, 0.37] | 0.00 [0.00, 0.00] |
| Unseen domain: web | 0.58 [0.13, 1.14] | 0.47 [0.12, 0.99] | 0.50 [0.12, 1.01] | 0.42 [0.00, 0.87] | 0.34 [0.00, 0.64] | 0.52 [0.12, 1.13] | 0.36 [0.00, 0.88] | 0.43 [0.12, 0.89] | 0.00 [0.00, 0.00] |
| Unseen domains (mean) | 0.74 [0.47, 1.05] | 0.32 [0.15, 0.52] | 0.65 [0.40, 0.94] | 0.28 [0.15, 0.45] | 0.32 [0.15, 0.50] | 0.37 [0.20, 0.57] | 0.39 [0.20, 0.65] | 0.39 [0.22, 0.65] | 0.00 [0.00, 0.00] |

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
| Unseen domain: banking | stronger monitor, least confident | 0.07 [-0.25, 0.37] | 18% | -0.05 [-0.26, 0.25] | 44% |
| Unseen domain: banking | LightGBM deferrer (task folds) | 0.14 [-0.25, 0.50] | 13% | -0.03 [-0.37, 0.25] | 44% |
| Unseen domain: banking | LightGBM deferrer (domain folds) | 0.27 [0.00, 0.62] | 0% | 0.07 [-0.13, 0.38] | 18% |
| Unseen domain: banking | transformer deferrer (task folds) | 0.13 [-0.25, 0.50] | 18% | 0.10 [-0.25, 0.50] | 20% |
| Unseen domain: banking | transformer deferrer (domain folds) | 0.14 [-0.12, 0.38] | 7% | 0.12 [-0.25, 0.50] | 15% |
| Unseen domain: coding | stronger monitor, least confident | -0.07 [-0.49, 0.25] | 48% | -0.09 [-0.36, 0.12] | 56% |
| Unseen domain: coding | LightGBM deferrer (task folds) | 0.22 [-0.13, 0.73] | 10% | 0.10 [-0.12, 0.37] | 6% |
| Unseen domain: coding | LightGBM deferrer (domain folds) | 0.19 [-0.12, 0.62] | 7% | 0.04 [-0.13, 0.25] | 22% |
| Unseen domain: coding | transformer deferrer (task folds) | 0.18 [-0.13, 0.62] | 13% | 0.28 [-0.12, 0.75] | 3% |
| Unseen domain: coding | transformer deferrer (domain folds) | 0.20 [-0.24, 0.61] | 10% | 0.17 [-0.12, 0.50] | 6% |
| Unseen domain: email | stronger monitor, least confident | -0.05 [-0.37, 0.12] | 39% | -0.04 [-0.37, 0.24] | 34% |
| Unseen domain: email | LightGBM deferrer (task folds) | 0.05 [-0.12, 0.25] | 10% | 0.07 [-0.12, 0.36] | 4% |
| Unseen domain: email | LightGBM deferrer (domain folds) | 0.09 [0.00, 0.37] | 2% | 0.08 [-0.12, 0.37] | 6% |
| Unseen domain: email | transformer deferrer (task folds) | 0.07 [-0.12, 0.25] | 7% | 0.09 [-0.12, 0.37] | 10% |
| Unseen domain: email | transformer deferrer (domain folds) | 0.09 [-0.12, 0.37] | 4% | 0.13 [0.00, 0.49] | 2% |
| Unseen domain: medical | stronger monitor, least confident | 0.06 [-0.12, 0.25] | 4% | 0.03 [-0.12, 0.25] | 10% |
| Unseen domain: medical | LightGBM deferrer (task folds) | 0.00 [-0.13, 0.13] | 16% | -0.01 [-0.24, 0.12] | 18% |
| Unseen domain: medical | LightGBM deferrer (domain folds) | 0.07 [0.00, 0.25] | 0% | 0.03 [-0.12, 0.25] | 6% |
| Unseen domain: medical | transformer deferrer (task folds) | 0.03 [-0.12, 0.25] | 8% | 0.01 [-0.13, 0.24] | 16% |
| Unseen domain: medical | transformer deferrer (domain folds) | 0.02 [-0.12, 0.13] | 11% | -0.00 [-0.12, 0.12] | 16% |
| Unseen domain: web | stronger monitor, least confident | 0.03 [-0.25, 0.25] | 25% | -0.05 [-0.38, 0.25] | 44% |
| Unseen domain: web | LightGBM deferrer (task folds) | -0.05 [-0.25, 0.25] | 42% | -0.13 [-0.38, 0.12] | 64% |
| Unseen domain: web | LightGBM deferrer (domain folds) | 0.05 [-0.13, 0.25] | 4% | 0.05 [-0.13, 0.25] | 6% |
| Unseen domain: web | transformer deferrer (task folds) | -0.09 [-0.38, 0.13] | 54% | -0.10 [-0.38, 0.13] | 58% |
| Unseen domain: web | transformer deferrer (domain folds) | -0.02 [-0.25, 0.25] | 26% | -0.04 [-0.26, 0.13] | 36% |
| Unseen domains (mean) | stronger monitor, least confident | 0.01 [-0.12, 0.15] | 45% | -0.04 [-0.17, 0.07] | 74% |
| Unseen domains (mean) | LightGBM deferrer (task folds) | 0.07 [-0.07, 0.20] | 17% | -0.00 [-0.12, 0.10] | 55% |
| Unseen domains (mean) | LightGBM deferrer (domain folds) | 0.13 [0.02, 0.25] | 1% | 0.05 [-0.07, 0.17] | 15% |
| Unseen domains (mean) | transformer deferrer (task folds) | 0.06 [-0.07, 0.20] | 17% | 0.07 [-0.05, 0.22] | 18% |
| Unseen domains (mean) | transformer deferrer (domain folds) | 0.09 [-0.03, 0.20] | 6% | 0.07 [-0.05, 0.22] | 12% |

## Attack rate natural: missed attacks left per 1,000 calls

Review budget 2% (no deferral in the first column):

| setting | no deferral (main monitor) | least confident | most novel | stronger monitor, least confident | LightGBM deferrer (task folds) | LightGBM deferrer (domain folds) | transformer deferrer (task folds) | transformer deferrer (domain folds) | perfect (upper bound) |
|---|---|---|---|---|---|---|---|---|---|
| Task-disjoint test | 2.31 [1.44, 3.15] | 0.76 [0.31, 1.31] | 2.12 [1.23, 2.94] | 0.53 [0.10, 0.93] | 0.69 [0.21, 1.22] | — | 0.90 [0.31, 1.53] | — | 0.00 [0.00, 0.00] |
| Tool coverage 100% | 4.89 [0.79, 12.12] | 2.65 [0.38, 7.87] | 4.08 [0.40, 9.47] | 2.50 [0.39, 6.95] | 3.39 [0.38, 10.04] | — | 3.47 [0.34, 10.07] | — | 0.00 [0.00, 0.00] |
| Tool coverage 75% | 5.67 [1.58, 13.46] | 3.06 [0.35, 8.54] | 5.05 [1.14, 12.76] | 3.15 [0.70, 7.85] | 4.11 [0.39, 11.54] | — | 3.94 [0.38, 11.46] | — | 0.00 [0.00, 0.00] |
| Tool coverage 50% | 6.13 [1.54, 12.43] | 3.24 [0.00, 8.03] | 5.91 [1.54, 12.43] | 3.41 [0.76, 7.87] | 4.45 [0.77, 11.08] | — | 4.71 [0.40, 11.50] | — | 0.00 [0.00, 0.00] |
| Tool coverage 25% | 5.43 [2.74, 8.45] | 2.83 [1.06, 5.03] | 5.00 [2.40, 7.69] | 3.38 [1.39, 5.78] | 3.74 [1.19, 6.64] | — | 3.97 [1.52, 7.02] | — | 0.00 [0.00, 0.00] |
| Unseen domain: banking | 5.75 [4.28, 7.31] | 4.03 [2.83, 5.25] | 5.58 [4.12, 7.08] | 3.96 [2.88, 5.19] | 4.13 [2.93, 5.47] | 5.39 [3.86, 6.92] | 4.03 [2.66, 5.40] | 4.99 [3.58, 6.38] | 0.00 [0.00, 0.00] |
| Unseen domain: coding | 13.26 [11.40, 15.38] | 8.44 [6.91, 10.23] | 12.20 [10.15, 14.24] | 6.78 [5.41, 8.48] | 9.53 [7.83, 11.44] | 11.60 [9.78, 13.50] | 9.23 [7.29, 11.16] | 9.12 [7.48, 11.15] | 0.00 [0.00, 0.00] |
| Unseen domain: email | 3.40 [2.26, 4.71] | 2.45 [1.46, 3.66] | 3.33 [2.22, 4.66] | 1.82 [1.00, 2.81] | 2.83 [1.75, 4.04] | 3.21 [2.08, 4.56] | 2.97 [1.81, 4.20] | 3.24 [2.13, 4.60] | 0.00 [0.00, 0.00] |
| Unseen domain: medical | 1.90 [1.28, 2.62] | 1.21 [0.74, 1.69] | 1.90 [1.28, 2.62] | 1.59 [1.01, 2.35] | 1.02 [0.54, 1.50] | 1.83 [1.21, 2.62] | 1.46 [0.88, 2.08] | 1.26 [0.74, 1.81] | 0.00 [0.00, 0.00] |
| Unseen domain: web | 5.51 [4.45, 6.75] | 4.66 [3.77, 5.65] | 5.23 [4.18, 6.49] | 4.97 [3.89, 6.22] | 4.38 [3.39, 5.44] | 5.17 [4.18, 6.28] | 3.95 [2.91, 4.88] | 4.55 [3.64, 5.56] | 0.00 [0.00, 0.00] |
| Unseen domains (mean) | 5.96 [5.41, 6.60] | 4.16 [3.66, 4.73] | 5.65 [5.09, 6.30] | 3.83 [3.27, 4.39] | 4.38 [3.81, 5.05] | 5.44 [4.86, 6.08] | 4.33 [3.82, 4.94] | 4.63 [4.09, 5.30] | 0.00 [0.00, 0.00] |

Review budget 5% (no deferral in the first column):

| setting | no deferral (main monitor) | least confident | most novel | stronger monitor, least confident | LightGBM deferrer (task folds) | LightGBM deferrer (domain folds) | transformer deferrer (task folds) | transformer deferrer (domain folds) | perfect (upper bound) |
|---|---|---|---|---|---|---|---|---|---|
| Task-disjoint test | 2.31 [1.44, 3.15] | 0.22 [0.00, 0.52] | 2.12 [1.23, 2.94] | 0.18 [0.00, 0.52] | 0.22 [0.00, 0.52] | — | 0.67 [0.20, 1.22] | — | 0.00 [0.00, 0.00] |
| Tool coverage 100% | 4.89 [0.79, 12.12] | 1.60 [0.34, 4.30] | 3.59 [0.40, 8.07] | 1.28 [0.33, 2.92] | 1.57 [0.00, 5.01] | — | 2.66 [0.00, 9.31] | — | 0.00 [0.00, 0.00] |
| Tool coverage 75% | 5.67 [1.58, 13.46] | 1.95 [0.00, 5.39] | 4.98 [1.14, 12.76] | 1.39 [0.00, 4.25] | 2.10 [0.00, 6.74] | — | 3.10 [0.00, 9.63] | — | 0.00 [0.00, 0.00] |
| Tool coverage 50% | 6.13 [1.54, 12.43] | 1.90 [0.00, 5.37] | 5.41 [1.17, 12.43] | 1.43 [0.00, 4.68] | 2.14 [0.00, 6.39] | — | 3.60 [0.39, 9.91] | — | 0.00 [0.00, 0.00] |
| Tool coverage 25% | 5.43 [2.74, 8.45] | 1.78 [0.35, 3.81] | 4.65 [2.01, 7.69] | 1.46 [0.00, 3.47] | 2.05 [0.34, 4.30] | — | 3.26 [0.77, 6.34] | — | 0.00 [0.00, 0.00] |
| Unseen domain: banking | 5.75 [4.28, 7.31] | 2.31 [1.37, 3.18] | 5.22 [3.77, 6.76] | 2.31 [1.46, 3.09] | 2.24 [1.13, 3.33] | 3.27 [2.07, 4.67] | 2.84 [1.72, 4.15] | 3.81 [2.39, 5.43] | 0.00 [0.00, 0.00] |
| Unseen domain: coding | 13.26 [11.40, 15.38] | 4.17 [3.16, 5.27] | 11.08 [9.31, 13.02] | 3.39 [2.37, 4.39] | 6.00 [4.59, 7.63] | 6.30 [4.99, 7.81] | 6.79 [4.95, 9.03] | 5.86 [4.40, 7.59] | 0.00 [0.00, 0.00] |
| Unseen domain: email | 3.40 [2.26, 4.71] | 1.78 [0.94, 2.79] | 3.33 [2.22, 4.66] | 1.55 [0.81, 2.60] | 2.38 [1.40, 3.51] | 2.39 [1.41, 3.55] | 2.69 [1.61, 3.92] | 2.89 [1.82, 4.26] | 0.00 [0.00, 0.00] |
| Unseen domain: medical | 1.90 [1.28, 2.62] | 0.81 [0.34, 1.22] | 1.69 [1.08, 2.36] | 0.90 [0.47, 1.42] | 0.65 [0.27, 1.09] | 1.04 [0.54, 1.68] | 1.03 [0.54, 1.61] | 0.92 [0.47, 1.48] | 0.00 [0.00, 0.00] |
| Unseen domain: web | 5.51 [4.45, 6.75] | 4.09 [3.25, 4.97] | 4.58 [3.59, 5.56] | 3.47 [2.64, 4.40] | 3.02 [2.11, 3.97] | 4.51 [3.60, 5.47] | 3.15 [2.37, 4.00] | 3.62 [2.76, 4.57] | 0.00 [0.00, 0.00] |
| Unseen domains (mean) | 5.96 [5.41, 6.60] | 2.63 [2.25, 3.12] | 5.18 [4.64, 5.85] | 2.32 [1.90, 2.73] | 2.86 [2.39, 3.40] | 3.50 [3.02, 4.11] | 3.30 [2.84, 3.96] | 3.42 [2.97, 4.03] | 0.00 [0.00, 0.00] |

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
| Unseen domain: banking | stronger monitor, least confident | -0.07 [-0.77, 0.60] | 54% | -0.00 [-0.43, 0.44] | 46% |
| Unseen domain: banking | LightGBM deferrer (task folds) | 0.10 [-0.70, 0.85] | 36% | -0.08 [-0.87, 0.69] | 55% |
| Unseen domain: banking | LightGBM deferrer (domain folds) | 1.36 [0.69, 1.99] | 0% | 0.96 [0.09, 1.90] | 1% |
| Unseen domain: banking | transformer deferrer (task folds) | -0.01 [-0.77, 0.77] | 46% | 0.53 [-0.26, 1.46] | 10% |
| Unseen domain: banking | transformer deferrer (domain folds) | 0.96 [0.43, 1.63] | 0% | 1.50 [0.59, 2.58] | 0% |
| Unseen domain: coding | stronger monitor, least confident | -1.66 [-2.55, -0.87] | 100% | -0.78 [-1.30, -0.20] | 100% |
| Unseen domain: coding | LightGBM deferrer (task folds) | 1.10 [0.27, 2.04] | 0% | 1.83 [0.94, 2.96] | 0% |
| Unseen domain: coding | LightGBM deferrer (domain folds) | 3.16 [2.17, 4.13] | 0% | 2.13 [1.34, 3.16] | 0% |
| Unseen domain: coding | transformer deferrer (task folds) | 0.79 [-0.54, 1.96] | 18% | 2.62 [0.81, 4.39] | 0% |
| Unseen domain: coding | transformer deferrer (domain folds) | 0.68 [-0.27, 1.62] | 6% | 1.69 [0.81, 2.97] | 0% |
| Unseen domain: email | stronger monitor, least confident | -0.63 [-1.27, 0.00] | 97% | -0.23 [-0.87, 0.40] | 74% |
| Unseen domain: email | LightGBM deferrer (task folds) | 0.38 [0.00, 0.87] | 2% | 0.60 [0.20, 1.01] | 0% |
| Unseen domain: email | LightGBM deferrer (domain folds) | 0.76 [0.34, 1.19] | 0% | 0.60 [0.07, 1.12] | 1% |
| Unseen domain: email | transformer deferrer (task folds) | 0.52 [0.00, 0.94] | 2% | 0.91 [0.40, 1.46] | 0% |
| Unseen domain: email | transformer deferrer (domain folds) | 0.80 [0.40, 1.27] | 0% | 1.11 [0.66, 1.62] | 0% |
| Unseen domain: medical | stronger monitor, least confident | 0.38 [-0.14, 0.87] | 4% | 0.08 [-0.27, 0.54] | 30% |
| Unseen domain: medical | LightGBM deferrer (task folds) | -0.19 [-0.60, 0.20] | 78% | -0.17 [-0.48, 0.14] | 79% |
| Unseen domain: medical | LightGBM deferrer (domain folds) | 0.62 [0.27, 1.02] | 0% | 0.22 [-0.13, 0.68] | 11% |
| Unseen domain: medical | transformer deferrer (task folds) | 0.25 [-0.21, 0.75] | 10% | 0.21 [-0.21, 0.67] | 14% |
| Unseen domain: medical | transformer deferrer (domain folds) | 0.05 [-0.27, 0.41] | 34% | 0.10 [-0.27, 0.54] | 28% |
| Unseen domain: web | stronger monitor, least confident | 0.31 [-0.41, 1.01] | 16% | -0.62 [-1.28, 0.00] | 96% |
| Unseen domain: web | LightGBM deferrer (task folds) | -0.28 [-0.88, 0.34] | 78% | -1.07 [-1.83, -0.34] | 100% |
| Unseen domain: web | LightGBM deferrer (domain folds) | 0.51 [0.07, 0.95] | 1% | 0.43 [-0.14, 0.88] | 6% |
| Unseen domain: web | transformer deferrer (task folds) | -0.71 [-1.29, 0.00] | 97% | -0.94 [-1.70, -0.27] | 100% |
| Unseen domain: web | transformer deferrer (domain folds) | -0.11 [-0.66, 0.54] | 60% | -0.47 [-1.16, 0.14] | 94% |
| Unseen domains (mean) | stronger monitor, least confident | -0.33 [-0.62, -0.05] | 98% | -0.31 [-0.61, -0.08] | 99% |
| Unseen domains (mean) | LightGBM deferrer (task folds) | 0.22 [-0.08, 0.54] | 10% | 0.22 [-0.09, 0.54] | 9% |
| Unseen domains (mean) | LightGBM deferrer (domain folds) | 1.28 [1.02, 1.55] | 0% | 0.87 [0.56, 1.19] | 0% |
| Unseen domains (mean) | transformer deferrer (task folds) | 0.17 [-0.18, 0.52] | 16% | 0.67 [0.34, 1.06] | 0% |
| Unseen domains (mean) | transformer deferrer (domain folds) | 0.48 [0.17, 0.76] | 0% | 0.79 [0.49, 1.09] | 0% |

## Attack rate 5%: missed attacks left per 1,000 calls

Review budget 2% (no deferral in the first column):

| setting | no deferral (main monitor) | least confident | most novel | stronger monitor, least confident | LightGBM deferrer (task folds) | LightGBM deferrer (domain folds) | transformer deferrer (task folds) | transformer deferrer (domain folds) | perfect (upper bound) |
|---|---|---|---|---|---|---|---|---|---|
| Task-disjoint test | 0.67 [0.17, 1.20] | 0.20 [0.00, 0.51] | 0.61 [0.17, 1.18] | 0.13 [0.00, 0.34] | 0.21 [0.00, 0.52] | — | 0.26 [0.00, 0.67] | — | 0.00 [0.00, 0.00] |
| Tool coverage 100% | 1.80 [0.00, 4.44] | 0.73 [0.00, 2.00] | 1.50 [0.00, 3.52] | 0.70 [0.00, 1.56] | 0.92 [0.00, 2.58] | — | 1.11 [0.00, 3.61] | — | 0.00 [0.00, 0.00] |
| Tool coverage 75% | 1.95 [0.41, 4.75] | 0.91 [0.00, 2.91] | 1.78 [0.00, 4.67] | 0.91 [0.00, 2.45] | 1.27 [0.00, 3.90] | — | 1.31 [0.00, 4.22] | — | 0.00 [0.00, 0.00] |
| Tool coverage 50% | 2.01 [0.00, 4.58] | 0.99 [0.00, 2.82] | 1.94 [0.00, 4.57] | 1.09 [0.00, 2.77] | 1.38 [0.00, 3.63] | — | 1.47 [0.00, 4.09] | — | 0.00 [0.00, 0.00] |
| Tool coverage 25% | 1.94 [0.47, 3.58] | 0.97 [0.00, 2.26] | 1.75 [0.45, 3.33] | 1.18 [0.00, 2.48] | 1.29 [0.00, 2.88] | — | 1.40 [0.40, 2.92] | — | 0.00 [0.00, 0.00] |
| Unseen domain: banking | 2.84 [1.96, 3.78] | 1.71 [0.90, 2.52] | 2.75 [1.84, 3.59] | 1.83 [1.22, 2.47] | 1.97 [1.24, 2.78] | 2.44 [1.56, 3.31] | 1.94 [1.31, 2.70] | 2.18 [1.34, 3.07] | 0.00 [0.00, 0.00] |
| Unseen domain: coding | 3.47 [2.27, 4.56] | 1.92 [1.13, 2.72] | 3.18 [2.15, 4.24] | 1.68 [0.92, 2.42] | 2.43 [1.48, 3.51] | 2.48 [1.47, 3.38] | 2.34 [1.26, 3.26] | 2.35 [1.28, 3.28] | 0.00 [0.00, 0.00] |
| Unseen domain: email | 0.84 [0.34, 1.76] | 0.61 [0.11, 1.34] | 0.83 [0.33, 1.69] | 0.46 [0.00, 1.12] | 0.70 [0.11, 1.54] | 0.81 [0.33, 1.65] | 0.74 [0.22, 1.65] | 0.80 [0.23, 1.65] | 0.00 [0.00, 0.00] |
| Unseen domain: medical | 0.50 [0.11, 0.91] | 0.33 [0.00, 0.69] | 0.50 [0.11, 0.91] | 0.46 [0.11, 1.01] | 0.31 [0.00, 0.68] | 0.48 [0.11, 0.91] | 0.39 [0.00, 0.79] | 0.36 [0.00, 0.79] | 0.00 [0.00, 0.00] |
| Unseen domain: web | 1.44 [0.70, 2.21] | 1.26 [0.57, 1.95] | 1.37 [0.58, 2.09] | 1.33 [0.69, 2.07] | 1.15 [0.45, 1.84] | 1.38 [0.69, 2.14] | 1.06 [0.46, 1.82] | 1.22 [0.57, 1.95] | 0.00 [0.00, 0.00] |
| Unseen domains (mean) | 1.82 [1.42, 2.19] | 1.17 [0.91, 1.45] | 1.73 [1.38, 2.10] | 1.15 [0.88, 1.43] | 1.31 [1.03, 1.64] | 1.52 [1.18, 1.86] | 1.29 [1.02, 1.59] | 1.38 [1.09, 1.70] | 0.00 [0.00, 0.00] |

Review budget 5% (no deferral in the first column):

| setting | no deferral (main monitor) | least confident | most novel | stronger monitor, least confident | LightGBM deferrer (task folds) | LightGBM deferrer (domain folds) | transformer deferrer (task folds) | transformer deferrer (domain folds) | perfect (upper bound) |
|---|---|---|---|---|---|---|---|---|---|
| Task-disjoint test | 0.67 [0.17, 1.20] | 0.05 [0.00, 0.17] | 0.61 [0.17, 1.18] | 0.02 [0.00, 0.17] | 0.07 [0.00, 0.34] | — | 0.20 [0.00, 0.51] | — | 0.00 [0.00, 0.00] |
| Tool coverage 100% | 1.80 [0.00, 4.44] | 0.42 [0.00, 1.05] | 1.32 [0.00, 3.49] | 0.23 [0.00, 1.02] | 0.43 [0.00, 1.51] | — | 0.76 [0.00, 3.07] | — | 0.00 [0.00, 0.00] |
| Tool coverage 75% | 1.95 [0.41, 4.75] | 0.52 [0.00, 1.50] | 1.77 [0.00, 4.63] | 0.30 [0.00, 0.97] | 0.65 [0.00, 2.40] | — | 0.96 [0.00, 3.75] | — | 0.00 [0.00, 0.00] |
| Tool coverage 50% | 2.01 [0.00, 4.58] | 0.54 [0.00, 1.83] | 1.78 [0.00, 4.57] | 0.37 [0.00, 1.79] | 0.62 [0.00, 2.26] | — | 1.14 [0.00, 3.57] | — | 0.00 [0.00, 0.00] |
| Tool coverage 25% | 1.94 [0.47, 3.58] | 0.63 [0.00, 1.64] | 1.62 [0.44, 3.30] | 0.50 [0.00, 1.63] | 0.76 [0.00, 2.26] | — | 1.16 [0.00, 2.69] | — | 0.00 [0.00, 0.00] |
| Unseen domain: banking | 2.84 [1.96, 3.78] | 1.09 [0.45, 1.65] | 2.59 [1.65, 3.35] | 1.03 [0.45, 1.64] | 1.02 [0.54, 1.64] | 1.29 [0.57, 1.94] | 1.33 [0.76, 1.98] | 1.51 [0.77, 2.30] | 0.00 [0.00, 0.00] |
| Unseen domain: coding | 3.47 [2.27, 4.56] | 1.01 [0.45, 1.60] | 2.91 [1.84, 4.11] | 0.79 [0.23, 1.34] | 1.30 [0.58, 2.06] | 1.19 [0.56, 1.80] | 1.68 [0.90, 2.49] | 1.44 [0.68, 2.27] | 0.00 [0.00, 0.00] |
| Unseen domain: email | 0.84 [0.34, 1.76] | 0.47 [0.00, 1.13] | 0.83 [0.33, 1.69] | 0.39 [0.00, 1.01] | 0.63 [0.11, 1.37] | 0.66 [0.22, 1.46] | 0.67 [0.11, 1.48] | 0.75 [0.22, 1.58] | 0.00 [0.00, 0.00] |
| Unseen domain: medical | 0.50 [0.11, 0.91] | 0.26 [0.00, 0.57] | 0.44 [0.11, 0.80] | 0.33 [0.00, 0.69] | 0.23 [0.00, 0.56] | 0.34 [0.00, 0.69] | 0.28 [0.00, 0.68] | 0.28 [0.00, 0.58] | 0.00 [0.00, 0.00] |
| Unseen domain: web | 1.44 [0.70, 2.21] | 1.11 [0.46, 1.72] | 1.24 [0.46, 1.97] | 1.02 [0.46, 1.72] | 0.83 [0.23, 1.47] | 1.25 [0.46, 1.96] | 0.88 [0.35, 1.50] | 1.00 [0.35, 1.61] | 0.00 [0.00, 0.00] |
| Unseen domains (mean) | 1.82 [1.42, 2.19] | 0.79 [0.59, 1.04] | 1.60 [1.27, 1.92] | 0.71 [0.50, 0.95] | 0.80 [0.59, 1.06] | 0.95 [0.72, 1.24] | 0.97 [0.75, 1.24] | 0.99 [0.73, 1.28] | 0.00 [0.00, 0.00] |

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
| Unseen domain: banking | stronger monitor, least confident | 0.12 [-0.22, 0.55] | 24% | -0.06 [-0.44, 0.22] | 44% |
| Unseen domain: banking | LightGBM deferrer (task folds) | 0.26 [-0.23, 0.87] | 12% | -0.07 [-0.55, 0.44] | 56% |
| Unseen domain: banking | LightGBM deferrer (domain folds) | 0.73 [0.32, 1.11] | 0% | 0.20 [-0.22, 0.66] | 8% |
| Unseen domain: banking | transformer deferrer (task folds) | 0.23 [-0.22, 0.67] | 12% | 0.24 [-0.22, 0.77] | 10% |
| Unseen domain: banking | transformer deferrer (domain folds) | 0.46 [0.00, 0.97] | 2% | 0.42 [0.00, 0.88] | 2% |
| Unseen domain: coding | stronger monitor, least confident | -0.24 [-0.70, 0.23] | 76% | -0.22 [-0.57, 0.11] | 84% |
| Unseen domain: coding | LightGBM deferrer (task folds) | 0.51 [0.00, 1.04] | 2% | 0.29 [0.00, 0.79] | 0% |
| Unseen domain: coding | LightGBM deferrer (domain folds) | 0.56 [0.00, 1.13] | 1% | 0.18 [-0.23, 0.68] | 13% |
| Unseen domain: coding | transformer deferrer (task folds) | 0.42 [-0.23, 1.02] | 6% | 0.67 [0.00, 1.36] | 2% |
| Unseen domain: coding | transformer deferrer (domain folds) | 0.43 [-0.12, 1.02] | 4% | 0.43 [-0.00, 1.02] | 2% |
| Unseen domain: email | stronger monitor, least confident | -0.15 [-0.67, 0.22] | 62% | -0.08 [-0.66, 0.23] | 50% |
| Unseen domain: email | LightGBM deferrer (task folds) | 0.08 [-0.11, 0.34] | 12% | 0.16 [-0.11, 0.45] | 4% |
| Unseen domain: email | LightGBM deferrer (domain folds) | 0.19 [0.00, 0.45] | 2% | 0.19 [-0.11, 0.55] | 4% |
| Unseen domain: email | transformer deferrer (task folds) | 0.12 [-0.11, 0.45] | 12% | 0.20 [-0.11, 0.56] | 4% |
| Unseen domain: email | transformer deferrer (domain folds) | 0.19 [-0.11, 0.45] | 5% | 0.28 [0.00, 0.66] | 2% |
| Unseen domain: medical | stronger monitor, least confident | 0.13 [-0.11, 0.45] | 6% | 0.07 [-0.22, 0.34] | 17% |
| Unseen domain: medical | LightGBM deferrer (task folds) | -0.02 [-0.23, 0.23] | 34% | -0.04 [-0.34, 0.22] | 39% |
| Unseen domain: medical | LightGBM deferrer (domain folds) | 0.16 [0.00, 0.45] | 0% | 0.08 [-0.11, 0.34] | 9% |
| Unseen domain: medical | transformer deferrer (task folds) | 0.06 [-0.12, 0.34] | 13% | 0.02 [-0.23, 0.34] | 28% |
| Unseen domain: medical | transformer deferrer (domain folds) | 0.03 [-0.22, 0.23] | 20% | 0.01 [-0.23, 0.23] | 28% |
| Unseen domain: web | stronger monitor, least confident | 0.08 [-0.34, 0.46] | 25% | -0.09 [-0.57, 0.35] | 55% |
| Unseen domain: web | LightGBM deferrer (task folds) | -0.10 [-0.46, 0.34] | 60% | -0.28 [-0.70, 0.12] | 84% |
| Unseen domain: web | LightGBM deferrer (domain folds) | 0.12 [-0.11, 0.46] | 6% | 0.14 [-0.12, 0.46] | 8% |
| Unseen domain: web | transformer deferrer (task folds) | -0.20 [-0.58, 0.12] | 78% | -0.23 [-0.69, 0.23] | 80% |
| Unseen domain: web | transformer deferrer (domain folds) | -0.04 [-0.35, 0.34] | 44% | -0.11 [-0.45, 0.23] | 62% |
| Unseen domains (mean) | stronger monitor, least confident | -0.01 [-0.20, 0.16] | 58% | -0.08 [-0.24, 0.07] | 82% |
| Unseen domains (mean) | LightGBM deferrer (task folds) | 0.14 [-0.03, 0.34] | 5% | 0.01 [-0.16, 0.18] | 50% |
| Unseen domains (mean) | LightGBM deferrer (domain folds) | 0.35 [0.18, 0.52] | 0% | 0.16 [0.02, 0.31] | 2% |
| Unseen domains (mean) | transformer deferrer (task folds) | 0.13 [-0.05, 0.29] | 12% | 0.18 [0.00, 0.40] | 2% |
| Unseen domains (mean) | transformer deferrer (domain folds) | 0.21 [0.02, 0.38] | 2% | 0.20 [0.04, 0.38] | 0% |

## Attack rate 1%: missed attacks left per 1,000 calls

Review budget 2% (no deferral in the first column):

| setting | no deferral (main monitor) | least confident | most novel | stronger monitor, least confident | LightGBM deferrer (task folds) | LightGBM deferrer (domain folds) | transformer deferrer (task folds) | transformer deferrer (domain folds) | perfect (upper bound) |
|---|---|---|---|---|---|---|---|---|---|
| Task-disjoint test | 0.13 [0.00, 0.56] | 0.03 [0.00, 0.19] | 0.12 [0.00, 0.39] | 0.02 [0.00, 0.19] | 0.04 [0.00, 0.19] | — | 0.05 [0.00, 0.37] | — | 0.00 [0.00, 0.00] |
| Tool coverage 100% | 0.30 [0.00, 1.18] | 0.12 [0.00, 0.93] | 0.25 [0.00, 1.15] | 0.10 [0.00, 0.59] | 0.13 [0.00, 0.93] | — | 0.16 [0.00, 0.99] | — | 0.00 [0.00, 0.00] |
| Tool coverage 75% | 0.36 [0.00, 1.35] | 0.17 [0.00, 1.04] | 0.31 [0.00, 1.35] | 0.14 [0.00, 1.04] | 0.23 [0.00, 1.06] | — | 0.24 [0.00, 1.07] | — | 0.00 [0.00, 0.00] |
| Tool coverage 50% | 0.40 [0.00, 1.49] | 0.19 [0.00, 0.98] | 0.38 [0.00, 1.47] | 0.20 [0.00, 0.98] | 0.27 [0.00, 1.35] | — | 0.29 [0.00, 1.44] | — | 0.00 [0.00, 0.00] |
| Tool coverage 25% | 0.41 [0.00, 1.43] | 0.20 [0.00, 0.95] | 0.38 [0.00, 1.43] | 0.25 [0.00, 0.97] | 0.26 [0.00, 1.00] | — | 0.28 [0.00, 1.35] | — | 0.00 [0.00, 0.00] |
| Unseen domain: banking | 0.58 [0.13, 1.17] | 0.32 [0.00, 0.90] | 0.56 [0.13, 1.16] | 0.35 [0.00, 0.90] | 0.39 [0.00, 1.03] | 0.45 [0.00, 1.03] | 0.39 [0.00, 0.91] | 0.38 [0.00, 0.91] | 0.00 [0.00, 0.00] |
| Unseen domain: coding | 0.70 [0.13, 1.40] | 0.36 [0.00, 0.87] | 0.64 [0.13, 1.27] | 0.31 [0.00, 0.87] | 0.47 [0.12, 1.01] | 0.45 [0.00, 1.01] | 0.44 [0.00, 0.90] | 0.46 [0.00, 0.89] | 0.00 [0.00, 0.00] |
| Unseen domain: email | 0.19 [0.00, 0.75] | 0.12 [0.00, 0.62] | 0.18 [0.00, 0.75] | 0.10 [0.00, 0.51] | 0.16 [0.00, 0.64] | 0.18 [0.00, 0.75] | 0.17 [0.00, 0.63] | 0.18 [0.00, 0.63] | 0.00 [0.00, 0.00] |
| Unseen domain: medical | 0.09 [0.00, 0.38] | 0.05 [0.00, 0.25] | 0.09 [0.00, 0.38] | 0.08 [0.00, 0.38] | 0.06 [0.00, 0.26] | 0.09 [0.00, 0.38] | 0.07 [0.00, 0.38] | 0.06 [0.00, 0.26] | 0.00 [0.00, 0.00] |
| Unseen domain: web | 0.30 [0.00, 0.76] | 0.27 [0.00, 0.65] | 0.29 [0.00, 0.76] | 0.28 [0.00, 0.64] | 0.24 [0.00, 0.64] | 0.29 [0.00, 0.76] | 0.22 [0.00, 0.64] | 0.25 [0.00, 0.65] | 0.00 [0.00, 0.00] |
| Unseen domains (mean) | 0.37 [0.18, 0.59] | 0.23 [0.08, 0.41] | 0.35 [0.18, 0.56] | 0.23 [0.08, 0.39] | 0.26 [0.13, 0.45] | 0.29 [0.13, 0.48] | 0.26 [0.10, 0.43] | 0.27 [0.13, 0.44] | 0.00 [0.00, 0.00] |

Review budget 5% (no deferral in the first column):

| setting | no deferral (main monitor) | least confident | most novel | stronger monitor, least confident | LightGBM deferrer (task folds) | LightGBM deferrer (domain folds) | transformer deferrer (task folds) | transformer deferrer (domain folds) | perfect (upper bound) |
|---|---|---|---|---|---|---|---|---|---|
| Task-disjoint test | 0.13 [0.00, 0.56] | 0.01 [0.00, 0.19] | 0.12 [0.00, 0.39] | 0.00 [0.00, 0.00] | 0.01 [0.00, 0.19] | — | 0.03 [0.00, 0.19] | — | 0.00 [0.00, 0.00] |
| Tool coverage 100% | 0.30 [0.00, 1.18] | 0.06 [0.00, 0.57] | 0.22 [0.00, 1.15] | 0.03 [0.00, 0.56] | 0.06 [0.00, 0.58] | — | 0.10 [0.00, 0.95] | — | 0.00 [0.00, 0.00] |
| Tool coverage 75% | 0.36 [0.00, 1.35] | 0.09 [0.00, 0.54] | 0.31 [0.00, 1.35] | 0.05 [0.00, 0.53] | 0.11 [0.00, 0.54] | — | 0.17 [0.00, 1.06] | — | 0.00 [0.00, 0.00] |
| Tool coverage 50% | 0.40 [0.00, 1.49] | 0.08 [0.00, 0.50] | 0.35 [0.00, 1.47] | 0.08 [0.00, 0.50] | 0.12 [0.00, 0.90] | — | 0.22 [0.00, 1.02] | — | 0.00 [0.00, 0.00] |
| Tool coverage 25% | 0.41 [0.00, 1.43] | 0.15 [0.00, 0.90] | 0.35 [0.00, 1.35] | 0.13 [0.00, 0.90] | 0.16 [0.00, 0.95] | — | 0.22 [0.00, 0.97] | — | 0.00 [0.00, 0.00] |
| Unseen domain: banking | 0.58 [0.13, 1.17] | 0.20 [0.00, 0.65] | 0.53 [0.13, 1.16] | 0.17 [0.00, 0.52] | 0.18 [0.00, 0.52] | 0.24 [0.00, 0.77] | 0.25 [0.00, 0.76] | 0.25 [0.00, 0.78] | 0.00 [0.00, 0.00] |
| Unseen domain: coding | 0.70 [0.13, 1.40] | 0.19 [0.00, 0.51] | 0.59 [0.13, 1.26] | 0.14 [0.00, 0.50] | 0.23 [0.00, 0.63] | 0.20 [0.00, 0.63] | 0.33 [0.00, 0.76] | 0.27 [0.00, 0.66] | 0.00 [0.00, 0.00] |
| Unseen domain: email | 0.19 [0.00, 0.75] | 0.10 [0.00, 0.50] | 0.18 [0.00, 0.75] | 0.08 [0.00, 0.50] | 0.14 [0.00, 0.63] | 0.15 [0.00, 0.63] | 0.15 [0.00, 0.63] | 0.17 [0.00, 0.63] | 0.00 [0.00, 0.00] |
| Unseen domain: medical | 0.09 [0.00, 0.38] | 0.04 [0.00, 0.25] | 0.08 [0.00, 0.26] | 0.06 [0.00, 0.26] | 0.04 [0.00, 0.25] | 0.07 [0.00, 0.25] | 0.05 [0.00, 0.25] | 0.04 [0.00, 0.25] | 0.00 [0.00, 0.00] |
| Unseen domain: web | 0.30 [0.00, 0.76] | 0.25 [0.00, 0.64] | 0.26 [0.00, 0.65] | 0.21 [0.00, 0.52] | 0.18 [0.00, 0.51] | 0.28 [0.00, 0.76] | 0.19 [0.00, 0.52] | 0.22 [0.00, 0.64] | 0.00 [0.00, 0.00] |
| Unseen domains (mean) | 0.37 [0.18, 0.59] | 0.16 [0.05, 0.29] | 0.33 [0.15, 0.54] | 0.13 [0.03, 0.26] | 0.15 [0.03, 0.28] | 0.19 [0.05, 0.33] | 0.19 [0.08, 0.38] | 0.19 [0.05, 0.36] | 0.00 [0.00, 0.00] |

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
| Unseen domain: banking | stronger monitor, least confident | 0.03 [-0.26, 0.26] | 18% | -0.03 [-0.26, 0.13] | 30% |
| Unseen domain: banking | LightGBM deferrer (task folds) | 0.07 [-0.13, 0.39] | 12% | -0.02 [-0.26, 0.13] | 30% |
| Unseen domain: banking | LightGBM deferrer (domain folds) | 0.13 [0.00, 0.39] | 0% | 0.04 [-0.13, 0.26] | 15% |
| Unseen domain: banking | transformer deferrer (task folds) | 0.07 [-0.13, 0.39] | 17% | 0.05 [-0.13, 0.39] | 20% |
| Unseen domain: banking | transformer deferrer (domain folds) | 0.06 [-0.13, 0.26] | 8% | 0.05 [-0.13, 0.39] | 20% |
| Unseen domain: coding | stronger monitor, least confident | -0.05 [-0.26, 0.25] | 40% | -0.05 [-0.25, 0.00] | 41% |
| Unseen domain: coding | LightGBM deferrer (task folds) | 0.11 [-0.13, 0.51] | 10% | 0.04 [-0.13, 0.25] | 5% |
| Unseen domain: coding | LightGBM deferrer (domain folds) | 0.09 [-0.13, 0.38] | 11% | 0.01 [-0.13, 0.25] | 18% |
| Unseen domain: coding | transformer deferrer (task folds) | 0.08 [-0.13, 0.38] | 14% | 0.14 [-0.13, 0.51] | 3% |
| Unseen domain: coding | transformer deferrer (domain folds) | 0.10 [-0.13, 0.38] | 10% | 0.08 [-0.13, 0.38] | 7% |
| Unseen domain: email | stronger monitor, least confident | -0.02 [-0.25, 0.13] | 22% | -0.02 [-0.25, 0.13] | 20% |
| Unseen domain: email | LightGBM deferrer (task folds) | 0.03 [-0.12, 0.25] | 3% | 0.04 [-0.12, 0.25] | 3% |
| Unseen domain: email | LightGBM deferrer (domain folds) | 0.06 [0.00, 0.25] | 2% | 0.04 [-0.12, 0.26] | 4% |
| Unseen domain: email | transformer deferrer (task folds) | 0.04 [-0.12, 0.25] | 4% | 0.05 [-0.12, 0.25] | 4% |
| Unseen domain: email | transformer deferrer (domain folds) | 0.06 [-0.00, 0.25] | 2% | 0.07 [-0.00, 0.25] | 2% |
| Unseen domain: medical | stronger monitor, least confident | 0.03 [0.00, 0.25] | 1% | 0.02 [-0.13, 0.25] | 4% |
| Unseen domain: medical | LightGBM deferrer (task folds) | 0.01 [-0.13, 0.13] | 7% | -0.01 [-0.13, 0.13] | 10% |
| Unseen domain: medical | LightGBM deferrer (domain folds) | 0.04 [0.00, 0.25] | 0% | 0.02 [0.00, 0.13] | 2% |
| Unseen domain: medical | transformer deferrer (task folds) | 0.02 [-0.13, 0.13] | 4% | 0.00 [-0.13, 0.13] | 9% |
| Unseen domain: medical | transformer deferrer (domain folds) | 0.01 [-0.13, 0.13] | 7% | -0.00 [-0.13, 0.13] | 8% |
| Unseen domain: web | stronger monitor, least confident | 0.01 [-0.26, 0.26] | 22% | -0.04 [-0.39, 0.26] | 38% |
| Unseen domain: web | LightGBM deferrer (task folds) | -0.03 [-0.13, 0.13] | 28% | -0.07 [-0.26, 0.13] | 47% |
| Unseen domain: web | LightGBM deferrer (domain folds) | 0.02 [0.00, 0.13] | 2% | 0.03 [-0.13, 0.26] | 3% |
| Unseen domain: web | transformer deferrer (task folds) | -0.06 [-0.26, 0.13] | 38% | -0.06 [-0.26, 0.13] | 44% |
| Unseen domain: web | transformer deferrer (domain folds) | -0.02 [-0.25, 0.13] | 16% | -0.03 [-0.25, 0.13] | 25% |
| Unseen domains (mean) | stronger monitor, least confident | 0.00 [-0.10, 0.10] | 44% | -0.02 [-0.13, 0.05] | 66% |
| Unseen domains (mean) | LightGBM deferrer (task folds) | 0.04 [-0.05, 0.13] | 20% | -0.00 [-0.08, 0.08] | 52% |
| Unseen domains (mean) | LightGBM deferrer (domain folds) | 0.07 [-0.00, 0.18] | 2% | 0.03 [-0.05, 0.13] | 19% |
| Unseen domains (mean) | transformer deferrer (task folds) | 0.03 [-0.05, 0.13] | 30% | 0.04 [-0.05, 0.15] | 26% |
| Unseen domains (mean) | transformer deferrer (domain folds) | 0.04 [-0.03, 0.15] | 18% | 0.04 [-0.05, 0.13] | 26% |

## Review burden at attack rate 2%

Share of reviews spent on calls the monitor already had right (a 5% budget):

| setting | least confident | most novel | stronger monitor, least confident | LightGBM deferrer (task folds) | LightGBM deferrer (domain folds) | transformer deferrer (task folds) | transformer deferrer (domain folds) | perfect (upper bound) |
|---|---|---|---|---|---|---|---|---|
| Task-disjoint test | 93% | 98% | 95% | 93% | — | 94% | — | 93% |
| Tool coverage 100% | 92% | 98% | 94% | 92% | — | 93% | — | 92% |
| Tool coverage 75% | 87% | 96% | 89% | 89% | — | 91% | — | 86% |
| Tool coverage 50% | 86% | 98% | 87% | 88% | — | 91% | — | 83% |
| Tool coverage 25% | 86% | 96% | 86% | 89% | — | 90% | — | 82% |
| Unseen domain: banking | 85% | 99% | 89% | 87% | 84% | 89% | 86% | 83% |
| Unseen domain: coding | 86% | 99% | 86% | 86% | 86% | 89% | 87% | 85% |
| Unseen domain: email | 64% | 95% | 67% | 88% | 50% | 77% | 64% | 38% |
| Unseen domain: medical | 62% | 94% | 64% | 78% | 33% | 60% | 58% | 24% |
| Unseen domain: web | 54% | 84% | 53% | 86% | 37% | 59% | 46% | 0% |
| Unseen domains (mean) | 70% | 94% | 72% | 85% | 58% | 75% | 68% | 46% |

Figures: `results/figures/09_risk_vs_escalation.png`, `results/figures/10_domain_fold_labels.png`. Every budget: `reports/week3/03_deferral_eval_full.md`.
