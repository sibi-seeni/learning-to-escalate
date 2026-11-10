#!/usr/bin/env bash
# Run the whole week 1 pipeline in order. Each step also writes a short report to reports/week1/.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m l2e.data               # 1. load + check split sizes
python -m l2e.anonymize          # 2. anonymize worlds, mask targets   (~1 min)
python -m l2e.leakage            # 3. world-lookup shortcut, before vs after
python -m l2e.decision_points    # 4. decision points + labels
python -m l2e.baseline_lr        # 5. six-feature LR: paper sanity check + prefix monitor
python -m l2e.splits             # 6. tool-coverage + leave-one-domain-out splits
python -m pytest -q              # 7. tests
echo "Week 1 pipeline done. Reports are in reports/week1/"
