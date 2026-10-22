#!/usr/bin/env bash
# Starts the XRCE-DDS agent, PX4 (SIH) and AAS's px4_interface the way AAS does
# (namespace /Drone1, ROS domain 1), waits until the takeoff action is served,
# then runs whatever command it was given.
#
# Skip the sim with AVIANS_START_SIM=0 (useful for `exec` into a running box).
# Logs: /tmp/xrce.log, /tmp/px4.log, /tmp/px4_interface.log
set -o pipefail

# PX4's startup script sources px4-alias.sh from PATH.
export PATH=/opt/px4/bin:$PATH

# No `set -u` around these: ROS setup scripts reference unset variables by
# design (AMENT_TRACE_SETUP_FILES and friends) and would kill the container.
set +u
source /opt/ros/jazzy/setup.bash
source /ws/install/setup.bash
set -u

: "${PX4_SIM_MODEL:=sihsim_quadx}"
: "${AVIANS_DRONE_NS:=Drone1}"
: "${AVIANS_START_SIM:=1}"

wait_for() {  # wait_for <label> <seconds> <command...>
  local label=$1 secs=$2; shift 2
  echo -n "[entrypoint] waiting for ${label} "
  for i in $(seq 1 "$secs"); do
    if "$@" >/dev/null 2>&1; then echo "ok (${i}s)"; return 0; fi
    echo -n "."; sleep 1
  done
  echo " TIMEOUT"; return 1
}

if [ "$AVIANS_START_SIM" = "1" ]; then
  echo "[entrypoint] starting XRCE-DDS agent on udp4:8888"
  MicroXRCEAgent udp4 -p 8888 >/tmp/xrce.log 2>&1 &
  sleep 1

  echo "[entrypoint] starting PX4 (${PX4_SIM_MODEL}, SIH, headless) as /${AVIANS_DRONE_NS}"
  mkdir -p /tmp/px4 && cd /tmp/px4
  PX4_SIM_MODEL="${PX4_SIM_MODEL}" PX4_UXRCE_DDS_NS="${AVIANS_DRONE_NS}" \
    PX4_PARAM_UXRCE_DDS_DOM_ID="${ROS_DOMAIN_ID}" \
    px4 -d /opt/px4/etc >/tmp/px4.log 2>&1 &   # data path: PX4 symlinks etc/ here
  cd - >/dev/null

  if wait_for "/${AVIANS_DRONE_NS}/fmu topics" 60 \
       bash -c "ros2 topic list | grep -q '^/${AVIANS_DRONE_NS}/fmu/out/'"; then
    # QGroundControl on the host (display only): an extra MAVLink stream to
    # host.docker.internal:14550, where QGC listens by default. PX4 wants an
    # IP, not a name. AVIANS_QGC_HOST= (empty) turns it off.
    QGC_HOST="${AVIANS_QGC_HOST-host.docker.internal}"
    if [ -n "$QGC_HOST" ]; then
      QGC_IP=$(getent ahostsv4 "$QGC_HOST" | awk 'NR==1 {print $1}')
      if [ -n "$QGC_IP" ]; then
        (cd /tmp/px4 && px4-mavlink start -x -u 14600 -r 4000000 -t "$QGC_IP" -o 14550) \
          >>/tmp/px4.log 2>&1 \
          && echo "[entrypoint] MAVLink to QGroundControl at ${QGC_HOST} (${QGC_IP}):14550"
      else
        echo "[entrypoint] ${QGC_HOST} does not resolve; no QGroundControl stream"
      fi
    fi
    echo "[entrypoint] starting AAS px4_interface"
    ros2 run autopilot_interface px4_interface --ros-args -r __ns:=/${AVIANS_DRONE_NS} \
      >/tmp/px4_interface.log 2>&1 &
    wait_for "/${AVIANS_DRONE_NS}/takeoff_action" 30 \
      bash -c "ros2 action list | grep -q '^/${AVIANS_DRONE_NS}/takeoff_action'" \
      || { echo "  tail of px4_interface.log:"; tail -20 /tmp/px4_interface.log; }
  else
    echo "  tail of px4.log:"; tail -20 /tmp/px4.log 2>/dev/null
    echo "  tail of xrce.log:"; tail -10 /tmp/xrce.log 2>/dev/null
  fi
fi

exec "$@"
