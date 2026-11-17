#!/usr/bin/env bash
# Fix the main monitor: transformer on hashed text + tool roles (config.MAIN_ARCH / MAIN_FEATURES).
# Run after scripts/run_week2.sh. About 15-20 minutes on a laptop.
set -euo pipefail
cd "$(dirname "$0")/.."

# 1. Seven more seeds (10 in total) for the main monitor and the same model without roles, on every
#    leave-one-domain-out split, so the choice doesn't rest on 3 lucky or unlucky runs.
python -m l2e.compare_shift --configs transformer:hash-roles transformer:hash --seeds 3 4 5 6 7 8 9
# 2. Train the main monitor on the task-disjoint split and export it for the board (float + int8).
python -m l2e.train_monitor --encoder hash-roles --archs transformer
python -m l2e.export_onnx --encoder hash-roles --arch transformer
# 3. Out-of-fold error labels from the main monitor: the deferrer's training data for week 3.
python -m l2e.oof --encoder hash-roles --arch transformer
python -m pytest -q
echo "Main monitor fixed. See reports/week2/04_domain_shift.md, 01_monitor_hash-roles.md, 03_errors_transformer_hash-roles.md"
