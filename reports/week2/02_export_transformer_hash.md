# Week 2 · ONNX export: transformer + hash

Files: `board/models/monitor_transformer_hash.onnx` (445.3 KB) and `board/models/monitor_transformer_hash_int8.onnx` (161.0 KB).

All 9796 task-disjoint test decision points, calibrated scores, threshold 0.5:

| version | AUROC | AUPRC | F1 | max abs diff vs torch |
|---|---|---|---|---|
| torch | 0.9994 | 0.9978 | 0.9797 | 0 |
| ONNX float | 0.9994 | 0.9978 | 0.9797 | 6.3e-07 |
| ONNX int8 | 0.9994 | 0.9978 | 0.9797 | 1.8e-01 |

int8 changes the verdict on 2 of 9796 decision points.

Laptop latency for one decision on the longest sample trace (8 steps, 15 events), single thread:

| stage | median ms | p95 ms |
|---|---|---|
| featurize | 0.166 | 0.172 |
| model (float) | 0.084 | 0.086 |
| model (int8) | 0.085 | 0.088 |

Board bundle: `board/models/monitor_bundle_transformer_hash.json` (featurizer config, tool vocabulary, expected outputs for 688 decisions from 120 test traces) plus `board/step_features.py`.
On the board: `python board/time_monitor.py --bundle models/monitor_bundle_transformer_hash.json`
