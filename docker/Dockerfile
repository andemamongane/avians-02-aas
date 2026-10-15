# Local stack, one container: PX4 SIH + XRCE-DDS agent + AAS's px4_interface + avians.
#
# It runs the SAME PX4 (v1.17.0 + AAS's patch and DDS topic list) and the SAME
# autopilot_interface node as the aerial-autonomy-stack, built from the pinned
# submodule. Only the physics differs: PX4's built-in SIH instead of Gazebo, so
# it runs on a laptop (no GPU, no wind, no QGroundControl). The full AAS runs on
# the remote Linux machine; the avians code is identical in both.

# ---- stage 1: PX4 SITL, built the way AAS builds it (minus Gazebo)
FROM ubuntu:24.04 AS px4
ARG PX4_REF=v1.17.0
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      git sudo ca-certificates lsb-release tzdata \
 && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 --branch ${PX4_REF} --recursive --shallow-submodules \
      https://github.com/PX4/PX4-Autopilot.git /px4
WORKDIR /px4
RUN bash ./Tools/setup/ubuntu.sh --no-nuttx --no-sim-tools && rm -rf /var/lib/apt/lists/*
ARG AAS=external/aerial-autonomy-stack/simulation/simulation_resources/patches
COPY ${AAS}/px4-v1.17.0.patch /tmp/aas.patch
COPY ${AAS}/dds_topics.yaml src/modules/uxrce_dds_client/dds_topics.yaml
RUN git apply /tmp/aas.patch
# cmake + ninja directly so the job count is bounded: `make px4_sitl` uses every
# core and runs out of memory in a 4 GB Docker VM.
ARG JOBS=4
RUN cmake -S . -B build/px4_sitl_default -G Ninja -DCONFIG=px4_sitl_default \
 && ninja -C build/px4_sitl_default -j${JOBS} \
 && test -f build/px4_sitl_default/etc/init.d-posix/airframes/10040_sihsim_quadx

# ---- stage 2: the runtime image
FROM ros:jazzy-ros-base
ARG PX4_MSGS_REF=86d8239e962f6939e05c3737784f60c02fa884db
ENV PX4_MSGS_REF=${PX4_MSGS_REF} \
    DEBIAN_FRONTEND=noninteractive \
    ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
    ROS_DOMAIN_ID=1 \
    PYTHONUNBUFFERED=1

# No `|| true` here: a failed install must stop the build.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3-pip git curl \
      ros-${ROS_DISTRO}-rosbag2 \
      ros-${ROS_DISTRO}-mavros-msgs ros-${ROS_DISTRO}-geographic-msgs \
      libgeographiclib-dev \
 && rm -rf /var/lib/apt/lists/*

# Micro XRCE-DDS Agent: prefer the apt package; build from source only if absent
# (there is no arm64 package for jazzy). The source build is a CMake superbuild
# with no top-level `install` target, so the binary AND its shared libraries
# have to be placed by hand.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ros-${ROS_DISTRO}-micro-xrce-dds-agent \
    || ( \
      apt-get install -y --no-install-recommends build-essential cmake git; \
      git clone -b v2.4.3 --depth 1 https://github.com/eProsima/Micro-XRCE-DDS-Agent /tmp/agent; \
      cmake -S /tmp/agent -B /tmp/agent/build -DCMAKE_BUILD_TYPE=Release; \
      cmake --build /tmp/agent/build -j2; \
      find /tmp/agent/build -name 'MicroXRCEAgent' -type f -perm -u+x -exec install -m0755 {} /usr/local/bin/ \; ; \
      find /tmp/agent/build -name '*.so*' -exec cp -P {} /usr/local/lib/ \; ; \
    ); \
    echo /usr/local/lib > /etc/ld.so.conf.d/local.conf; ldconfig; \
    rm -rf /tmp/agent /var/lib/apt/lists/*; \
    command -v MicroXRCEAgent; \
    ! ldd "$(command -v MicroXRCEAgent)" | grep -q "not found"

COPY --from=px4 /px4/build/px4_sitl_default/bin /opt/px4/bin
COPY --from=px4 /px4/build/px4_sitl_default/etc /opt/px4/etc

# px4_msgs must MATCH the PX4 version (release/1.17, as AAS uses), then AAS's
# interface packages straight from the submodule.
RUN mkdir -p /ws/src/px4_msgs && cd /ws/src/px4_msgs \
 && git init -q && git remote add origin https://github.com/PX4/px4_msgs.git \
 && git fetch --depth 1 origin ${PX4_MSGS_REF} && git checkout -q FETCH_HEAD
ARG AAS_WS=external/aerial-autonomy-stack/aircraft/aircraft_ws/src
COPY ${AAS_WS}/autopilot_interface_msgs /ws/src/autopilot_interface_msgs
COPY ${AAS_WS}/autopilot_interface /ws/src/autopilot_interface
RUN . /opt/ros/jazzy/setup.sh && cd /ws \
 && MAKEFLAGS=-j2 colcon build --parallel-workers 2 --cmake-args -DCMAKE_BUILD_TYPE=Release \
 && test -x /ws/install/autopilot_interface/lib/autopilot_interface/px4_interface

# Python deps. No conda, no venv: rclpy is compiled against this interpreter.
# matplotlib from pip, NOT apt (apt's pulls python3-brotli 1.1, which breaks the
# anthropic SDK's HTTP stack). numpy pinned to the apt version rclpy uses.
RUN python3 -m pip install --break-system-packages --no-cache-dir \
      anthropic pyyaml pytest matplotlib \
      "numpy==$(python3 -c 'import numpy; print(numpy.__version__)')"

WORKDIR /workspace
COPY pyproject.toml /workspace/
COPY src /workspace/src
RUN python3 -m pip install --break-system-packages --no-cache-dir -e /workspace

# Other pilots' SDK (DeepSeek and Codex speak OpenAI's APIs) and ffmpeg for
# flight videos. A late layer, so adding it doesn't rebuild PX4 or ROS.
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
 && rm -rf /var/lib/apt/lists/* \
 && python3 -m pip install --break-system-packages --no-cache-dir openai

# The pinned AAS commit, recorded in every episode's meta.json. The submodule's
# git dir lives in the parent repo's .git/modules (clone with --recursive).
COPY .git/modules/external/aerial-autonomy-stack/HEAD /opt/aas_commit

# Source ROS for EVERY shell (`docker compose exec` bypasses the entrypoint).
RUN printf '%s\n' \
      'set +u' \
      'export PATH=/opt/px4/bin:$PATH' \
      'source /opt/ros/jazzy/setup.bash' \
      '[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash' \
      | tee /etc/profile.d/ros.sh >> /root/.bashrc

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["sleep", "infinity"]
