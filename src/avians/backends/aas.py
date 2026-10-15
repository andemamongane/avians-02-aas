"""aerial-autonomy-stack backend over ROS 2. The ONLY module that imports rclpy.

Everything above this file is plain Python and testable without ROS.

It is a client of AAS's px4_interface node (autopilot_interface package):
    takeoff_action   autopilot_interface_msgs/action/Takeoff
    land_action      autopilot_interface_msgs/action/Land
    orbit_action     autopilot_interface_msgs/action/Orbit
    set_reposition   autopilot_interface_msgs/srv/SetReposition
    set_speed        autopilot_interface_msgs/srv/SetSpeed
It never publishes PX4 setpoints itself: AAS owns the low-level control.

Facts this file exists to contain:
  1. Frames. AAS takes east/north metres from the home point it saves at
     takeoff, and altitude above home. PX4's local position is NED (x north,
     y east, z DOWN) from the EKF origin, which is NOT at ground level (in SIH
     a landed drone reads z = -0.9 m). So the backend measures everything from
     where the drone rests when it connects, which is where AAS will save home
     at takeoff. That origin goes into meta.json so the rosbag trace (raw NED)
     is converted the same way for plots and STL scoring.
  2. AAS refuses any command while another one is active, and accepts
     reposition/orbit/land only while hovering or orbiting. A refused action
     carries no message, so the reasons below are written from AAS's source.
  3. Never cancel an AAS action: cancelling repositions the drone at
     abort_reposition_alt (100 m by default). hold() repositions in place.
  4. PX4 publishes with BEST_EFFORT QoS. A reliable subscriber gets nothing,
     silently. If state never arrives, check this first.
"""
from __future__ import annotations

import math
import threading
import time

from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from autopilot_interface_msgs.action import Land, Orbit, Takeoff
from autopilot_interface_msgs.srv import SetReposition, SetSpeed
from px4_msgs.msg import BatteryStatus, VehicleLocalPosition, VehicleStatus

from .base import ActionHandle, VehicleState

PX4_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)

NAV_STATE_NAMES = {
    0: "manual", 1: "altctl", 2: "posctl", 3: "auto_mission", 4: "auto_loiter",
    5: "auto_rtl", 14: "offboard", 15: "stab", 17: "auto_takeoff", 18: "auto_land",
    19: "auto_follow_target", 20: "auto_precland", 21: "orbit",
}

REJECTED_BECAUSE = {
    "takeoff": "it accepts takeoff only on the ground with preflight checks passing",
    "land": "it accepts land only while hovering or orbiting",
    "orbit": "it accepts orbit only while hovering or orbiting",
}


class AasBackend(Node):
    name = "aas"

    def __init__(self, cfg: dict):
        super().__init__("avians_backend")
        ns = cfg.get("namespace", "/Drone1").rstrip("/")
        self.cfg = cfg
        self._lock = threading.Lock()
        self._state = VehicleState(ok=False)
        self._raw_ned = (0.0, 0.0, 0.0)
        self.frame_origin_ned: tuple[float, float, float] | None = None
        self.counts = {"position": 0, "status": 0, "battery": 0}

        def name(key: str) -> str:
            return f"{ns}/{cfg[key]}"

        self.create_subscription(VehicleLocalPosition, name("local_position"),
                                 self._on_position, PX4_QOS)
        self.create_subscription(VehicleStatus, name("status"), self._on_status, PX4_QOS)
        if cfg.get("battery"):
            self.create_subscription(BatteryStatus, name("battery"), self._on_battery, PX4_QOS)

        self.takeoff_client = ActionClient(self, Takeoff, name("takeoff_action"))
        self.land_client = ActionClient(self, Land, name("land_action"))
        self.orbit_client = ActionClient(self, Orbit, name("orbit_action"))
        self.reposition_client = self.create_client(SetReposition, name("set_reposition"))
        self.speed_client = self.create_client(SetSpeed, name("set_speed"))

    # ---------------------------------------------------------------- state
    def _on_position(self, msg: VehicleLocalPosition) -> None:
        self.counts["position"] += 1
        with self._lock:
            self._raw_ned = (float(msg.x), float(msg.y), float(msg.z))
            ox, oy, oz = self.frame_origin_ned or (0.0, 0.0, 0.0)
            s = self._state
            s.north, s.east = float(msg.x) - ox, float(msg.y) - oy
            s.alt = -(float(msg.z) - oz)             # NED down -> altitude up
            s.v_north, s.v_east, s.v_up = float(msg.vx), float(msg.vy), -float(msg.vz)
            if math.isfinite(msg.heading):
                s.heading_deg = (math.degrees(msg.heading) + 360.0) % 360.0
            s.t_sim = msg.timestamp / 1e6
            s.ok = True

    def _on_status(self, msg: VehicleStatus) -> None:
        self.counts["status"] += 1
        with self._lock:
            self._state.armed = msg.arming_state == VehicleStatus.ARMING_STATE_ARMED
            nav = int(msg.nav_state)
            self._state.mode = NAV_STATE_NAMES.get(nav, f"nav_{nav}") if self._state.armed \
                else "disarmed"

    def _on_battery(self, msg: BatteryStatus) -> None:
        self.counts["battery"] += 1
        with self._lock:
            if msg.remaining >= 0:
                self._state.battery_pct = float(msg.remaining) * 100.0

    def get_state(self) -> VehicleState:
        with self._lock:
            return VehicleState(**vars(self._state))

    def now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def connect(self) -> None:
        """Wait for telemetry and AAS's servers. Assumes something else spins
        this node (cli.py starts a thread); rclpy refuses two executors."""
        timeout = float(self.cfg.get("connect_timeout_s", 60))
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not self.get_state().ok:
            time.sleep(0.1)
        if not self.get_state().ok:
            raise RuntimeError(
                f"no telemetry on {self.cfg['namespace']}/{self.cfg['local_position']} "
                f"after {timeout:.0f}s. Check: PX4 running with PX4_UXRCE_DDS_NS, XRCE "
                f"agent up, same ROS_DOMAIN_ID, topic names in config/aas.yaml.")
        time.sleep(0.5)                              # let vehicle_status arrive too
        if self.get_state().armed:
            raise RuntimeError("the drone is already armed: land it (or restart the "
                               "simulation) before starting a mission, so the frame "
                               "origin is the landing pad")
        with self._lock:
            self.frame_origin_ned = self._raw_ned
        missing = [c for c, ok in (
            ("takeoff_action", self.takeoff_client.wait_for_server(timeout_sec=10)),
            ("land_action", self.land_client.wait_for_server(timeout_sec=2)),
            ("orbit_action", self.orbit_client.wait_for_server(timeout_sec=2)),
            ("set_reposition", self.reposition_client.wait_for_service(timeout_sec=2)),
            ("set_speed", self.speed_client.wait_for_service(timeout_sec=2)),
        ) if not ok]
        if missing:
            raise RuntimeError(f"AAS px4_interface is not serving: {', '.join(missing)}. "
                               f"Is it running under namespace {self.cfg['namespace']}?")

    def disconnect(self) -> None:
        try:
            self.destroy_node()
        except Exception:
            pass

    # ------------------------------------------------------------- commands
    def takeoff(self, altitude: float) -> ActionHandle:
        st = self.get_state()
        h = ActionHandle(kind="takeoff", target=(st.east, st.north, altitude))
        goal = Takeoff.Goal()
        goal.takeoff_altitude = float(altitude)
        return self._send(self.takeoff_client, goal, h)

    def land(self, alt: float) -> ActionHandle:
        h = ActionHandle(kind="land", target=(0.0, 0.0, 0.0))
        goal = Land.Goal()
        goal.landing_altitude = float(alt)
        return self._send(self.land_client, goal, h)

    def orbit(self, east: float, north: float, alt: float, radius: float) -> ActionHandle:
        h = ActionHandle(kind="orbit", target=(east, north, alt))
        goal = Orbit.Goal()
        goal.east, goal.north, goal.altitude, goal.radius = (
            float(east), float(north), float(alt), float(radius))
        return self._send(self.orbit_client, goal, h)

    def reposition(self, east: float, north: float, alt: float,
                   speed: float | None = None) -> ActionHandle:
        h = ActionHandle(kind="reposition", target=(east, north, alt))
        if speed:
            # AAS applies set_speed to the NEXT reposition, and refuses it while
            # any other command is active; wait for its reply before moving on.
            ok, msg = self._call_sync(self.speed_client, SetSpeed.Request(speed=float(speed)))
            if not ok:
                return h.fail(f"set_speed failed: {msg}")
        req = SetReposition.Request()
        req.east, req.north, req.altitude = float(east), float(north), float(alt)
        ok, msg = self._call_sync(self.reposition_client, req)
        return h.succeed(msg) if ok else h.fail(f"set_reposition failed: {msg}")

    def hold(self) -> ActionHandle:
        st = self.get_state()
        return self.reposition(st.east, st.north, st.alt)

    # -------------------------------------------------------------- helpers
    def _send(self, client: ActionClient, goal, h: ActionHandle) -> ActionHandle:
        if not client.server_is_ready():
            return h.fail(f"{h.kind} action server is not available")

        def on_feedback(fb) -> None:
            h.feedback["message"] = fb.feedback.message

        def on_goal(fut) -> None:
            gh = fut.result()
            if gh is None or not gh.accepted:
                why = REJECTED_BECAUSE.get(h.kind, "")
                h.fail(f"the autopilot interface rejected {h.kind}"
                       + (f" ({why}, and no other command active)" if why else ""))
                return
            gh.get_result_async().add_done_callback(on_result)

        def on_result(fut) -> None:
            res = fut.result()
            if res is not None and res.result.success:
                h.succeed(h.feedback.get("message", f"{h.kind} done"))
            else:
                h.fail(f"{h.kind} did not succeed: {h.feedback.get('message', 'no detail')}")

        client.send_goal_async(goal, feedback_callback=on_feedback).add_done_callback(on_goal)
        return h

    def _call_sync(self, client, request, timeout: float = 5.0) -> tuple[bool, str]:
        """Call a service and wait for the reply. The node is spun by another
        thread, so waiting on the future here is safe."""
        if not client.service_is_ready():
            return False, "service not available"
        fut = client.call_async(request)
        deadline = time.monotonic() + timeout
        while not fut.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not fut.done():
            return False, f"no reply within {timeout:.0f} s"
        res = fut.result()
        return bool(res.success), str(res.message)
