#!/usr/bin/env bash
# Run this yourself after reviewing Claude Code's local commit(s).
# Usage: scripts/sync.sh "commit message"
set -euo pipefail

if [ -z "${1:-}" ]; then
  echo "Usage: scripts/sync.sh \"commit message\"" >&2
  exit 1
fi

git add -A
git commit -m "$1"
git push
