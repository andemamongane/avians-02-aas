#!/usr/bin/env bash
# Build (first time) and start the container.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] || { cp .env.example .env; echo "Created .env; add your ANTHROPIC_API_KEY to it before flying."; }
docker compose -f docker/compose.yml up -d --build
docker compose -f docker/compose.yml exec avians bash -lc "avians doctor"
