# Week 2 · ONNX export: transformer + hash-roles

Files: `board/models/monitor_transformer_hash-roles.onnx` (447.3 KB) and `board/models/monitor_transformer_hash-roles_int8.onnx` (161.5 KB).

All 9796 task-disjoint test decision points, calibrated scores, threshold 0.5:

| version | AUROC | AUPRC | F1 | max abs diff vs torch |
|---|---|---|---|---|
| torch | 0.9997 | 0.9987 | 0.9844 | 0 |
| ONNX float | 0.9997 | 0.9987 | 0.9844 | 5.7e-07 |
| ONNX int8 | 0.9997 | 0.9986 | 0.9841 | 1.1e-01 |

int8 changes the verdict on 3 of 9796 decision points.

Laptop latency for one decision on the longest sample trace (8 steps, 15 events), single thread:

| stage | median ms | p95 ms |
|---|---|---|
| featurize | 0.175 | 0.184 |
| model (float) | 0.084 | 0.089 |
| model (int8) | 0.086 | 0.091 |

Board bundle: `board/models/monitor_bundle_transformer_hash-roles.json` (featurizer config, tool vocabulary, expected outputs for 688 decisions from 120 test traces) plus `board/step_features.py`.
On the board: `python board/time_monitor.py --bundle models/monitor_bundle_transformer_hash-roles.json`
