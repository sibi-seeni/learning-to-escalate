#!/usr/bin/env bash
# Week 3 (Deliverable 3): deferrer v2 with text novelty, the transformer deferrer, domain-fold labels, and the
# full evaluation (1-5% attack rates, review budgets 0-30%, intervals). Run after scripts/finalize_monitor.sh.
#
#   bash scripts/run_week3.sh                  about 1-1.5 hours on a laptop CPU
#   EXTRA_SEEDS=1 bash scripts/run_week3.sh    + monitor seeds 1 and 2 on the unseen-domain splits (about 2 hours more)
#   NEURAL_SEEDS=1 bash scripts/run_week3.sh   one transformer-deferrer seed per run instead of 3 (faster)
#
# Every step writes to its own folder and can be rerun on its own. Steps 5 and 6 run LightGBM and PyTorch in
# separate processes on purpose (they can crash together on macOS).
set -euo pipefail
cd "$(dirname "$0")/.."

M="--encoder hash-roles --arch transformer"
LODO="lodo_banking lodo_coding lodo_email lodo_medical lodo_web"
SEEDS=""
if [ "${EXTRA_SEEDS:-0}" = "1" ]; then SEEDS="1 2"; fi

# 1. Context rows for the tool-coverage test sets: the full model's scores on the earlier calls of each sampled
#    trace, so the deferrer's "earlier in this trace" features are exact. No training (~1 min).
python -m l2e.oof $M --context-only --splits tool_seed0 tool_seed1 tool_seed2

# 2. Domain-grouped out-of-fold labels: each fold model trains on 3 of the 4 training domains and scores the 4th.
#    The full model is reused from the week 2 run. 4 trainings per split (~15 min).
python -m l2e.oof $M --fold-by domain --splits $LODO

# 3. Optional: two more monitor seeds on the unseen-domain splits (task folds + full model, then domain folds).
for s in $SEEDS; do
  python -m l2e.oof $M --seed "$s" --splits $LODO
  python -m l2e.oof $M --seed "$s" --fold-by domain --splits $LODO
done

# 4. Text novelty (distance to the nearest training call / tool result), out of fold, for every run (~5 min).
python -m l2e.novelty
python -m l2e.novelty --fold-by domain --splits $LODO
for s in $SEEDS; do
  python -m l2e.novelty --seed "$s" --splits $LODO
  python -m l2e.novelty --seed "$s" --fold-by domain --splits $LODO
done

# 5. LightGBM deferrer v2: 5 seeds per run, every split and every out-of-fold run found (~10 min).
python -m l2e.deferrer

# 6. Transformer deferrer that reads the trace: 3 seeds per run (~30-60 min).
python -m l2e.neural_deferrer --seeds "${NEURAL_SEEDS:-3}"

# 7. Every policy at the same review budgets, 1-5% attack rates, 200 resamples per cell (~2 min).
python -m l2e.evaluate_deferral

python -m pytest -q
echo "Week 3 done. Main report: reports/week3/03_deferral_eval.md; figures 09 and 10 in results/figures/"
