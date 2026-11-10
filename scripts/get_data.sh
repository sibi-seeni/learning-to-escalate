#!/usr/bin/env bash
# Download AgentDrift (CC BY 4.0) into data/raw/AgentDrift, pinned to the commit used in this project.
set -euo pipefail
cd "$(dirname "$0")/.."

URL="https://github.com/Asif-0209/AgentDrift.git"
COMMIT="014a514fa998b4ac4519579fceb8a5884b379bda"
DEST="data/raw/AgentDrift"

# (Re)clone if missing or broken (e.g. an interrupted clone).
if [ ! -d "$DEST/.git" ] || ! git -C "$DEST" rev-parse --verify --quiet HEAD >/dev/null 2>&1; then
  rm -rf "$DEST"
  mkdir -p data/raw
  git clone "$URL" "$DEST"
fi
git -C "$DEST" fetch --quiet origin
git -C "$DEST" checkout --quiet "$COMMIT"
echo "AgentDrift at $(git -C "$DEST" rev-parse --short HEAD) in $DEST"
ls "$DEST/data_taskdisjoint"
