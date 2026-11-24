#!/usr/bin/env bash
# Deliverable 3 end-to-end check (a few minutes, after run_week3.sh): the tests, the report figures, and the
# training/evaluation notebook run top to bottom. Everything printed is also saved to results/logs/.
#
#   bash scripts/run_d3_check.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/logs
LOG="results/logs/d3_check_$(date +%Y%m%d_%H%M).log"
{
  echo "== $(date) | $(uname -sm) | $(python --version 2>&1)"
  echo "== 1. tests"
  python -m pytest -q
  echo "== 2. report figures (D2 vs D3, confusion matrices, ROC, examples)"
  python -m l2e.report_figures
  echo "== 3. notebook, executed in place (outputs saved in notebooks/train_eval.ipynb)"
  jupyter nbconvert --to notebook --execute --inplace notebooks/train_eval.ipynb \
    --ExecutePreprocessor.timeout=1800
  echo "== done $(date)"
} 2>&1 | tee "$LOG"
echo "Log: $LOG"
