#!/usr/bin/env bash
# Dump the ROS graph to docs/ros_notes_raw.txt — the factual basis for topics.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."
docker compose -f docker/compose.yml exec avians bash -lc "avians probe" | tee docs/ros_notes_raw.txt
