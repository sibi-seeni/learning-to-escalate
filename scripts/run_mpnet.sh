#!/usr/bin/env bash
# The plan's "better monitor" (week 3, about 1 hour): a 4-layer, width-256 causal transformer on mpnet sentence
# embeddings + tool roles, the same encoder DriftNet uses. Laptop only: far too big for the board. It is compared
# with everything else as the "big mpnet monitor, least confident" policy in reports/week3/03_deferral_eval.md.
#
#   bash scripts/run_mpnet.sh                   trains on your Mac's GPU (MPS) if it has one, else the CPU
#   L2E_DEVICE=cpu bash scripts/run_mpnet.sh    force the CPU (slower, ~10+ min per split)
set -euo pipefail
cd "$(dirname "$0")/.."
export L2E_DEVICE="${L2E_DEVICE:-auto}"
M="--encoder mpnet-roles --arch transformer_big --full-only"

# 1. Embed every event text with all-mpnet-base-v2 (downloads ~420 MB once; ~10-15 min on the Mac's GPU).
if [ ! -f data/processed/emb/mpnet.npy ]; then
  python -m l2e.encoders embed mpnet
fi

# 2. Smoke test: 3 epochs on one split (a few minutes). If the big model fails, it fails here, not an hour in.
python -m l2e.oof $M --quick --splits taskdisjoint

# 3. The real run: the full model on all 9 splits (val, test and context scores).
python -m l2e.oof $M

# 4. Every policy again, now with the big mpnet monitor.
python -m l2e.evaluate_deferral
python -m pytest -q
echo "Done. See 'big mpnet monitor' in reports/week3/03_deferral_eval.md and reports/week3/03_errors_transformer_big_mpnet-roles_fullonly.md"
