#!/usr/bin/env bash
# Run one mission inside the container.  ./scripts/fly.sh missions/square.yaml [--tier fast]
set -euo pipefail
cd "$(dirname "$0")/.."
MISSION="${1:?usage: fly.sh missions/<name>.yaml [extra args]}"
shift || true
docker compose -f docker/compose.yml exec avians bash -lc "avians fly $MISSION $*"
