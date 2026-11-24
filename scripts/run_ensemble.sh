#!/usr/bin/env bash
# Seed ensemble and seed disagreement (week 3). Run after scripts/run_week3.sh with EXTRA_SEEDS=1, which already
# trained seeds 1 and 2 on the unseen-domain splits. This adds the same two seeds on the other splits, full model
# only (no folds, ~5-10 min), then reruns the evaluation with the three seed policies (~3 min):
#   3-seed ensemble, least confident · seeds disagree most · ensemble, unsure or disagreeing
set -euo pipefail
cd "$(dirname "$0")/.."

M="--encoder hash-roles --arch transformer"
for s in 1 2; do
  python -m l2e.oof $M --seed "$s" --full-only --splits taskdisjoint tool_seed0 tool_seed1 tool_seed2
done
python -m l2e.evaluate_deferral
python -m pytest -q
echo "Done. Seed policies are in reports/week3/03_deferral_eval.md (columns '3-seed ensemble ...', 'seeds disagree most')"
