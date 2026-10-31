# avians' plan-ahead control on top of AAS's aircraft image (built by
# scripts/aas_plan_ahead.sh, which applies avians_plan_ahead.patch to AAS's
# autopilot_interface and offboard_control and passes them as the context).
#   px4_offboard   + controller "avians_trajectory": tracks the trajectory
#                    avians publishes on /DroneN/avians/trajectory
#   px4_interface  + topic /DroneN/offboard_end: ends the offboard action into
#                    loiter/hover, as at max_duration_sec (a cancel would abort)
# Only these two packages are rebuilt; everything else is AAS's image as built.
ARG BASE_IMAGE=aircraft-image:aas
FROM ${BASE_IMAGE}
LABEL avians.plan_ahead="1"
COPY autopilot_interface /aas/aircraft_ws/src/autopilot_interface
COPY offboard_control /aas/aircraft_ws/src/offboard_control
WORKDIR /aas/aircraft_ws
RUN bash -c "source /opt/ros/jazzy/setup.bash && source /aas/github_ws/install/setup.bash && source /aas/aircraft_ws/install/setup.bash && \
    colcon build --symlink-install --packages-select autopilot_interface offboard_control --cmake-args -DCMAKE_BUILD_TYPE=Release"
WORKDIR /aas
