# Week 2 · ONNX export: lstm + hash

Files: `board/models/monitor_lstm_hash.onnx` (276.1 KB) and `board/models/monitor_lstm_hash_int8.onnx` (77.3 KB).

All 9796 task-disjoint test decision points, calibrated scores, threshold 0.5:

| version | AUROC | AUPRC | F1 | max abs diff vs torch |
|---|---|---|---|---|
| torch | 0.9995 | 0.9981 | 0.9804 | 0 |
| ONNX float | 0.9995 | 0.9981 | 0.9804 | 5.4e-07 |
| ONNX int8 | 0.9995 | 0.9981 | 0.9804 | 5.7e-02 |

int8 changes the verdict on 2 of 9796 decision points.

Laptop latency for one decision on the longest sample trace (8 steps, 15 events), single thread:

| stage | median ms | p95 ms |
|---|---|---|
| featurize | 0.168 | 0.173 |
| model (float) | 0.036 | 0.036 |
| model (int8) | 0.031 | 0.031 |

Board bundle: `board/models/monitor_bundle_lstm_hash.json` (featurizer config, tool vocabulary, expected outputs for 688 decisions from 120 test traces) plus `board/step_features.py`.
On the board: `python board/time_monitor.py --bundle models/monitor_bundle_lstm_hash.json`
