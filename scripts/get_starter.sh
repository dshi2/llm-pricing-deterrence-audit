#!/usr/bin/env bash
# Fetch the pinned hackathon starter into ./starter (only the two folders we use).
# STARTER_REMOTE overrides the source, e.g. a local clone.
set -euo pipefail
COMMIT=d585a7b3be50366cc2bdf5789bf8add5890b2abd
REMOTE="${STARTER_REMOTE:-https://github.com/poseidon-research/collusion_hackathon.git}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/starter"

if [ -d "$DEST/.git" ]; then
  FOUND="$(git -C "$DEST" rev-parse HEAD)"
  [ "$FOUND" = "$COMMIT" ] && { echo "starter already at $COMMIT"; exit 0; }
  echo "starter/ is at $FOUND, expected $COMMIT; remove it and rerun" >&2; exit 1
fi
git clone --quiet --filter=blob:none --no-checkout --sparse "$REMOTE" "$DEST"
git -C "$DEST" sparse-checkout set algorithmic-pricing-llms util
git -C "$DEST" checkout --quiet "$COMMIT"
echo "starter at $(git -C "$DEST" rev-parse HEAD)"
