#!/usr/bin/env bash
# Week 2: compact neural monitor. Run scripts/run_week1.sh first. Each step writes a report to reports/week2/.
# Times are for a laptop CPU; the out-of-fold step trains 6 models per split on 9 splits.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m l2e.train_monitor                           # 1. LSTM / GRU / transformer on hashed text vs. the LR   (~5 min)
python -m l2e.train_monitor --encoder hash-nothought  # 2. ablation: agent thoughts hidden                        (~5 min)
python -m l2e.export_onnx                             # 3. chosen monitor -> ONNX float + int8, board bundle
python -m l2e.export_onnx --arch transformer          #    the other int8 architecture, for the edge table
python -m l2e.oof                                     # 4. out-of-fold error labels, all 9 splits               (~30-60 min)
python -m pytest -q                                   # 5. tests

# Sentence-encoder versions. They download models from Hugging Face once. Run with WITH_ENCODERS=1.
if [ "${WITH_ENCODERS:-0}" = "1" ]; then
  python -m l2e.encoders embed minilm                                # MiniLM embeddings for every event text
  python -m l2e.train_monitor --encoder minilm --archs lstm transformer
  python -m l2e.encoders export-minilm                               # MiniLM itself -> ONNX for the board
  python -m l2e.export_onnx --encoder minilm
  python -m l2e.encoders embed mpnet                                 # DriftNet's encoder (laptop only)
  python -m l2e.train_monitor --encoder mpnet --archs transformer_big  # the "better monitor" baseline
fi
echo "Week 2 pipeline done. Reports are in reports/week2/"
