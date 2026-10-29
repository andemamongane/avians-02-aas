#!/usr/bin/env bash
# Linux machine: add avians' plan-ahead control to AAS's aircraft image
# (docker/aas/aircraft.dockerfile). Run once after AAS's sim_build.sh, and again
# after changing docker/aas/avians_plan_ahead.patch. A minute or two: only two
# packages are rebuilt. AAS's own image is kept as aircraft-image:aas, and the
# patched one replaces aircraft-image:latest, which sim_run.sh starts.
set -euo pipefail
cd "$(dirname "$0")/.."
SRC=external/aerial-autonomy-stack/aircraft/aircraft_ws/src
PATCH=docker/aas/avians_plan_ahead.patch

docker image inspect aircraft-image:latest >/dev/null 2>&1 || {
  echo "no aircraft-image: build AAS first (external/aerial-autonomy-stack/tools_and_docs/sim_build.sh)"; exit 1; }
if [ "$(docker image inspect -f '{{index .Config.Labels "avians.plan_ahead"}}' aircraft-image:latest)" != "1" ]; then
  docker tag aircraft-image:latest aircraft-image:aas      # a fresh AAS build: the new base
elif ! docker image inspect aircraft-image:aas >/dev/null 2>&1; then
  echo "aircraft-image:latest is already patched and AAS's original (aircraft-image:aas) is gone:"
  echo "rebuild AAS (sim_build.sh), then run this again"; exit 1
fi

ctx=$(mktemp -d)
trap 'rm -rf "$ctx"' EXIT
cp -r "$SRC/autopilot_interface" "$SRC/offboard_control" "$ctx/"
patch -d "$ctx" -p1 --forward --quiet < "$PATCH"
docker build -t aircraft-image:latest -f docker/aas/aircraft.dockerfile "$ctx"
echo "aircraft-image:latest now has avians' plan-ahead control; restart AAS (sim_run.sh) to use it"
