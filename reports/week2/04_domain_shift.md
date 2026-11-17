# Week 2 · Monitors under domain shift

Each cell: mean ± standard deviation over seeds 0, 1, 2. Temperature fit on each split's val set, verdict at 0.5. Confident false alarm = benign call scored ≥ 0.9; confident miss = hijacked call scored ≤ 0.1.

## Summary: average over the five held-out domains

| monitor | errors | dangerous | costly | confident false alarms | confident misses | AUROC | ECE | task-disjoint errors |
|---|---|---|---|---|---|---|---|---|
| lstm:hash | 347 ± 87 | 130 ± 13 | 217 ± 99 | 86 ± 54 | 76 ± 9 | 0.995 ± 0.000 | 0.016 ± 0.006 | 69 ± 2 |
| transformer:hash | 353 ± 75 | 111 ± 22 | 242 ± 71 | 85 ± 24 | 67 ± 15 | 0.996 ± 0.001 | 0.019 ± 0.007 | 74 ± 5 |
| transformer:hash-roles | 294 ± 44 | 114 ± 26 | 180 ± 68 | 70 ± 34 | 68 ± 17 | 0.996 ± 0.000 | 0.014 ± 0.003 | 67 ± 14 |
| transformer:hash-roles-noid | 325 ± 30 | 120 ± 3 | 205 ± 27 | 76 ± 11 | 68 ± 2 | 0.996 ± 0.000 | 0.016 ± 0.003 | 76 ± 8 |
| transformer:minilm | 429 ± 107 | 156 ± 35 | 273 ± 140 | 101 ± 65 | 83 ± 14 | 0.994 ± 0.000 | 0.024 ± 0.010 | 98 ± 37 |
| transformer:minilm-roles | 425 ± 55 | 167 ± 13 | 259 ± 55 | 105 ± 36 | 90 ± 12 | 0.994 ± 0.001 | 0.021 ± 0.004 | 109 ± 41 |

## Task-disjoint test (no shift)

9796 decision points, 1755 hijacked.

| monitor | seeds | errors | dangerous | costly | confident false alarms | AUROC | ECE |
|---|---|---|---|---|---|---|---|
| lstm:hash | 3 | 69 ± 2 | 35 ± 9 | 33 ± 11 | 14 ± 7 | 0.999 ± 0.000 | 0.003 ± 0.001 |
| transformer:hash | 3 | 74 ± 5 | 41 ± 11 | 33 ± 9 | 7 ± 5 | 0.999 ± 0.000 | 0.003 ± 0.001 |
| transformer:hash-roles | 3 | 67 ± 14 | 27 ± 6 | 40 ± 10 | 12 ± 3 | 0.999 ± 0.000 | 0.003 ± 0.001 |
| transformer:hash-roles-noid | 3 | 76 ± 8 | 23 ± 3 | 54 ± 9 | 23 ± 6 | 0.999 ± 0.000 | 0.006 ± 0.001 |
| transformer:minilm | 3 | 98 ± 37 | 35 ± 10 | 63 ± 40 | 20 ± 14 | 0.999 ± 0.000 | 0.006 ± 0.004 |
| transformer:minilm-roles | 3 | 109 ± 41 | 45 ± 7 | 64 ± 34 | 19 ± 7 | 0.999 ± 0.001 | 0.006 ± 0.003 |

## Held out: banking

11634 decision points, 1170 hijacked.

| monitor | seeds | errors | dangerous | costly | confident false alarms | AUROC | ECE |
|---|---|---|---|---|---|---|---|
| lstm:hash | 3 | 180 ± 19 | 91 ± 34 | 88 ± 43 | 23 ± 17 | 0.995 ± 0.001 | 0.009 ± 0.002 |
| transformer:hash | 3 | 165 ± 8 | 112 ± 25 | 53 ± 31 | 16 ± 13 | 0.995 ± 0.001 | 0.008 ± 0.002 |
| transformer:hash-roles | 3 | 146 ± 15 | 87 ± 22 | 59 ± 35 | 15 ± 12 | 0.996 ± 0.001 | 0.006 ± 0.000 |
| transformer:hash-roles-noid | 3 | 153 ± 3 | 121 ± 16 | 31 ± 15 | 6 ± 4 | 0.996 ± 0.000 | 0.007 ± 0.002 |
| transformer:minilm | 3 | 351 ± 136 | 88 ± 35 | 263 ± 167 | 88 ± 66 | 0.992 ± 0.000 | 0.024 ± 0.020 |
| transformer:minilm-roles | 3 | 346 ± 58 | 102 ± 20 | 244 ± 75 | 53 ± 18 | 0.991 ± 0.000 | 0.021 ± 0.006 |

## Held out: coding

14817 decision points, 2808 hijacked.

| monitor | seeds | errors | dangerous | costly | confident false alarms | AUROC | ECE |
|---|---|---|---|---|---|---|---|
| lstm:hash | 3 | 252 ± 11 | 147 ± 23 | 105 ± 26 | 36 ± 17 | 0.996 ± 0.001 | 0.010 ± 0.001 |
| transformer:hash | 3 | 319 ± 119 | 139 ± 110 | 180 ± 194 | 67 ± 71 | 0.996 ± 0.003 | 0.017 ± 0.010 |
| transformer:hash-roles | 3 | 257 ± 13 | 120 ± 11 | 137 ± 23 | 54 ± 16 | 0.996 ± 0.000 | 0.010 ± 0.002 |
| transformer:hash-roles-noid | 3 | 264 ± 18 | 139 ± 8 | 126 ± 25 | 51 ± 5 | 0.995 ± 0.001 | 0.010 ± 0.001 |
| transformer:minilm | 3 | 758 ± 309 | 92 ± 36 | 666 ± 344 | 265 ± 177 | 0.995 ± 0.001 | 0.043 ± 0.025 |
| transformer:minilm-roles | 3 | 881 ± 376 | 96 ± 49 | 785 ± 419 | 375 ± 230 | 0.993 ± 0.002 | 0.052 ± 0.029 |

## Held out: email

14986 decision points, 2862 hijacked.

| monitor | seeds | errors | dangerous | costly | confident false alarms | AUROC | ECE |
|---|---|---|---|---|---|---|---|
| lstm:hash | 3 | 367 ± 163 | 84 ± 58 | 284 ± 214 | 117 ± 90 | 0.997 ± 0.000 | 0.021 ± 0.014 |
| transformer:hash | 3 | 512 ± 115 | 37 ± 14 | 476 ± 128 | 157 ± 31 | 0.998 ± 0.000 | 0.036 ± 0.013 |
| transformer:hash-roles | 3 | 370 ± 79 | 48 ± 22 | 322 ± 101 | 125 ± 55 | 0.998 ± 0.000 | 0.023 ± 0.008 |
| transformer:hash-roles-noid | 3 | 421 ± 87 | 47 ± 10 | 374 ± 94 | 148 ± 45 | 0.998 ± 0.000 | 0.028 ± 0.009 |
| transformer:minilm | 3 | 329 ± 69 | 186 ± 80 | 143 ± 100 | 54 ± 40 | 0.996 ± 0.001 | 0.012 ± 0.004 |
| transformer:minilm-roles | 3 | 314 ± 47 | 174 ± 14 | 139 ± 54 | 45 ± 24 | 0.996 ± 0.001 | 0.008 ± 0.004 |

## Held out: medical

14795 decision points, 2758 hijacked.

| monitor | seeds | errors | dangerous | costly | confident false alarms | AUROC | ECE |
|---|---|---|---|---|---|---|---|
| lstm:hash | 3 | 235 ± 23 | 139 ± 46 | 96 ± 57 | 27 ± 27 | 0.997 ± 0.001 | 0.008 ± 0.003 |
| transformer:hash | 3 | 176 ± 22 | 93 ± 24 | 83 ± 44 | 23 ± 24 | 0.998 ± 0.000 | 0.005 ± 0.002 |
| transformer:hash-roles | 3 | 200 ± 38 | 120 ± 54 | 79 ± 82 | 20 ± 31 | 0.998 ± 0.000 | 0.008 ± 0.003 |
| transformer:hash-roles-noid | 3 | 235 ± 9 | 116 ± 38 | 120 ± 46 | 31 ± 17 | 0.997 ± 0.000 | 0.008 ± 0.001 |
| transformer:minilm | 3 | 328 ± 233 | 85 ± 60 | 243 ± 290 | 77 ± 110 | 0.998 ± 0.000 | 0.020 ± 0.027 |
| transformer:minilm-roles | 3 | 176 ± 21 | 152 ± 23 | 24 ± 2 | 6 ± 3 | 0.999 ± 0.000 | 0.006 ± 0.001 |

## Held out: web

14792 decision points, 2751 hijacked.

| monitor | seeds | errors | dangerous | costly | confident false alarms | AUROC | ECE |
|---|---|---|---|---|---|---|---|
| lstm:hash | 3 | 703 ± 327 | 190 ± 68 | 513 ± 386 | 229 ± 224 | 0.989 ± 0.002 | 0.034 ± 0.023 |
| transformer:hash | 3 | 592 ± 270 | 174 ± 58 | 418 ± 318 | 160 ± 137 | 0.992 ± 0.001 | 0.029 ± 0.021 |
| transformer:hash-roles | 3 | 498 ± 130 | 197 ± 88 | 302 ± 196 | 136 ± 118 | 0.992 ± 0.002 | 0.024 ± 0.010 |
| transformer:hash-roles-noid | 3 | 551 ± 151 | 176 ± 31 | 376 ± 182 | 145 ± 103 | 0.992 ± 0.001 | 0.025 ± 0.013 |
| transformer:minilm | 3 | 377 ± 4 | 329 ± 25 | 49 ± 22 | 21 ± 7 | 0.991 ± 0.001 | 0.018 ± 0.003 |
| transformer:minilm-roles | 3 | 409 ± 37 | 309 ± 71 | 101 ± 42 | 44 ± 24 | 0.992 ± 0.002 | 0.018 ± 0.004 |
