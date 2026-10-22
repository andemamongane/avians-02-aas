#!/usr/bin/env bash
# Linux machine: start avians next to a running aerial-autonomy-stack (see
# docker/compose.remote.yml). Start AAS first.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] || { cp .env.example .env; echo "Created .env; add your API keys to it before flying."; }
docker network inspect aas-sim-network-inst0 >/dev/null 2>&1 || {
  echo "AAS is not running (no aas-sim-network-inst0). Start it first:"
  echo "  cd external/aerial-autonomy-stack/tools_and_docs && NUM_QUADS=1 WORLD=impalpable_greyness ./sim_run.sh"
  exit 1; }
docker compose -f docker/compose.yml -f docker/compose.remote.yml up -d --build
docker compose -f docker/compose.yml -f docker/compose.remote.yml exec avians bash -lc "avians doctor"
