"""In-process fake backend. No ROS, no PX4, no container.

Purpose: unit-test tools/safety/skills/logging/runner/STL on a laptop in
seconds. It is NOT a simulator. Never collect training data from it: meta.json
records the backend name so fake episodes can always be excluded.

It does mimic the aerial-autonomy-stack's px4_interface state machine, so the
same commands are accepted and rejected in the same states:
  - takeoff only from the ground; done at 90% of the target altitude
  - reposition / orbit / land only while hovering or orbiting
  - land flies home at the given altitude, then descends and disarms
  - orbit keeps circling until another command replaces it
"""
from __future__ import annotations

import math
import time

from .base import ActionHandle, VehicleState

ORBIT_SPEED = 5.0           # AAS's mc_orbit_speed_ms default
TAKEOFF_DONE_RATIO = 0.9    # AAS's mc_takeoff_completed_ratio default


class FakeBackend:
    name = "fake"

    def __init__(self, speed: float = 3.0, time_fn=None):
        # Look time.monotonic up at call time (not as a default argument), so a
        # patched clock in tests reaches backends that the CLI creates.
        self._time = time_fn or (lambda: time.monotonic())
        self._t0 = self._time()
        self.state = VehicleState(mode="disarmed")
        self.cruise = speed
        self.fsm = "landed"              # landed | takeoff | hover | orbit | rtl | landing
        self._target: tuple[float, float, float] | None = None
        self._orbit: tuple[float, float, float, float] | None = None   # e, n, alt, r
        self._handle: ActionHandle | None = None
        self._last_update = self.now()
        self.connected = False

    # --- lifecycle
    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def now(self) -> float:
        return self._time() - self._t0

    # --- state
    def get_state(self) -> VehicleState:
        self._integrate()
        self.state.t_sim = self.now()
        self.state.battery_pct = max(0.0, 100.0 - self.now() * 0.5)
        # Return a COPY: skills.py holds snapshots to measure the interval
        # between samples. Handing out the live object makes every delta zero.
        return VehicleState(**vars(self.state))

    # --- commands (AAS semantics)
    def takeoff(self, altitude: float) -> ActionHandle:
        h = ActionHandle(kind="takeoff", target=(self.state.east, self.state.north, altitude))
        if self.fsm != "landed":
            return h.fail("takeoff rejected by the autopilot interface: already flying")
        self._integrate()
        self.state.armed = True
        self.state.mode = "auto_takeoff"
        self.fsm = "takeoff"
        self._orbit = None
        self._target = h.target
        self._handle = h
        return h

    def reposition(self, east: float, north: float, alt: float,
                   speed: float | None = None) -> ActionHandle:
        h = ActionHandle(kind="reposition", target=(east, north, alt))
        if self.fsm not in ("hover", "orbit"):
            return h.fail("set_reposition rejected: not in a hover/orbit state")
        self._integrate()
        if speed:
            self.cruise = speed
        self.fsm, self.state.mode = "hover", "auto_loiter"
        self._orbit = None
        self._target = (east, north, alt)
        return h.succeed("set_reposition request sent")

    def orbit(self, east: float, north: float, alt: float, radius: float) -> ActionHandle:
        h = ActionHandle(kind="orbit", target=(east, north, alt))
        if self.fsm not in ("hover", "orbit"):
            return h.fail("orbit rejected: not in a hover/orbit state")
        self._integrate()
        self.fsm, self.state.mode = "orbit", "orbit"
        self._target = None
        self._orbit = (east, north, alt, radius)
        return h.succeed("orbit sent")

    def hold(self) -> ActionHandle:
        s = self.state
        return self.reposition(s.east, s.north, s.alt)

    def land(self, alt: float) -> ActionHandle:
        h = ActionHandle(kind="land", target=(0.0, 0.0, 0.0))
        if self.fsm not in ("hover", "orbit"):
            return h.fail("land rejected: not in a hover/orbit state")
        self._integrate()
        self.fsm, self.state.mode = "rtl", "auto_rtl"
        self._orbit = None
        self._target = (0.0, 0.0, alt)
        self._handle = h
        return h

    # --- kinematics
    def _integrate(self) -> None:
        t = self.now()
        dt = max(0.0, t - self._last_update)
        self._last_update = t
        s = self.state
        if dt == 0:
            return
        if self._orbit is not None:
            self._integrate_orbit(dt)
            return
        if self._target is None:
            s.v_east = s.v_north = s.v_up = 0.0
            return
        if self._move_toward(self._target, self.cruise, dt):
            self._on_arrival()

    def _move_toward(self, target, speed: float, dt: float) -> bool:
        s = self.state
        te, tn, ta = target
        de, dn, da = te - s.east, tn - s.north, ta - s.alt
        dist = math.sqrt(de * de + dn * dn + da * da)
        step = speed * dt
        if dist <= step or dist == 0:
            s.east, s.north, s.alt = te, tn, ta
            s.v_east = s.v_north = s.v_up = 0.0
            return True
        f = step / dist
        s.east += de * f
        s.north += dn * f
        s.alt += da * f
        s.v_east, s.v_north, s.v_up = de / dist * speed, dn / dist * speed, da / dist * speed
        if self.fsm == "takeoff" and self._handle and s.alt >= TAKEOFF_DONE_RATIO * ta:
            self._handle.succeed("takeoff completed, hovering")
            self.fsm, s.mode = "hover", "auto_loiter"
        return False

    def _on_arrival(self) -> None:
        s = self.state
        if self.fsm == "takeoff":
            self._handle.succeed("takeoff completed, hovering")
            self.fsm, s.mode = "hover", "auto_loiter"
        elif self.fsm == "rtl":
            self._handle.succeed("final descent")
            self.fsm, s.mode = "landing", "auto_land"
            self._target = (0.0, 0.0, 0.0)
        elif self.fsm == "landing":
            self.fsm, s.mode, s.armed = "landed", "disarmed", False
            self._target = None

    def _integrate_orbit(self, dt: float) -> None:
        s = self.state
        ce, cn, ca, r = self._orbit
        de, dn = s.east - ce, s.north - cn
        rho = math.hypot(de, dn)
        if abs(rho - r) > 0.2 or abs(s.alt - ca) > 0.2:      # fly onto the circle first
            if rho < 1e-6:
                de, dn, rho = 1.0, 0.0, 1.0
            entry = (ce + de / rho * r, cn + dn / rho * r, ca)
            self._move_toward(entry, self.cruise, dt)
            return
        w = ORBIT_SPEED / r                                    # clockwise seen from above
        ang = math.atan2(dn, de) - w * dt
        s.east, s.north, s.alt = ce + r * math.cos(ang), cn + r * math.sin(ang), ca
        s.v_east, s.v_north, s.v_up = ORBIT_SPEED * math.sin(ang), -ORBIT_SPEED * math.cos(ang), 0.0
