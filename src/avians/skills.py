"""Skill layer: turns a one-shot command into a completed, summarized action.

This is policy, not transport, so it lives ABOVE the backend and behaves
identically for the fake and AAS backends. If this logic drifted into a
backend, data collected on one would not match the other.

AAS's reposition and orbit return as soon as the command is accepted, and its
takeoff finishes at 90% of the altitude. The agent needs "did I get there?",
so arrival is decided here from the vehicle state, for every backend.

Between two agent decisions thousands of messages pass. We compress that
interval into a few numbers (path length, peak speed, elapsed) rather than
dumping it, because that summary is what makes a trajectory learnable.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass

from .backends.base import ActionHandle, VehicleState
from .schema import ABORTED, ARRIVED, FAILED, TIMEOUT


@dataclass
class ActionOutcome:
    status: str
    reason: str
    state: VehicleState
    elapsed: float
    path_length: float
    max_speed: float
    distance_to_target: float | None
    target: tuple[float, float, float] | None


class Skills:
    def __init__(self, backend, arrive_radius: float = 1.0, poll_hz: float = 20.0,
                 default_timeout: float = 60.0, land_timeout: float = 120.0,
                 settle_speed: float = 0.5):
        self.backend = backend
        self.arrive_radius = arrive_radius
        self.poll_dt = 1.0 / poll_hz
        self.default_timeout = default_timeout
        self.land_timeout = land_timeout
        self.settle_speed = settle_speed
        # Called every poll while an action runs; a non-empty string stops the
        # action (ABORTED, with that reason). The online STL monitor sets it.
        self.interrupt = None

    # --- public actions -------------------------------------------------
    def takeoff(self, altitude: float, timeout: float | None = None) -> ActionOutcome:
        st = self.backend.get_state()
        h = self.backend.takeoff(altitude)
        return self._await(h, (st.east, st.north, altitude), timeout or self.default_timeout)

    def reposition(self, east: float, north: float, alt: float, speed: float | None = None,
                   timeout: float | None = None) -> ActionOutcome:
        h = self.backend.reposition(east, north, alt, speed)
        return self._await(h, (east, north, alt), timeout or self.default_timeout)

    def orbit(self, east: float, north: float, alt: float, radius: float,
              duration: float) -> ActionOutcome:
        h = self.backend.orbit(east, north, alt, radius)
        out = self._await(h, (east, north, alt), duration, hold_full_duration=True)
        if out.status == ARRIVED:
            self.backend.hold()                    # stop circling where we are
            out.reason = "orbit complete; holding position"
            out.distance_to_target = None          # the centre is not where we stop
        return out

    def hover(self, duration: float) -> ActionOutcome:
        st = self.backend.get_state()
        h = self.backend.hold()
        return self._await(h, st.pos, duration, hold_full_duration=True)

    def land(self, timeout: float | None = None, min_alt: float = 2.0) -> ActionOutcome:
        # AAS flies home at `landing_altitude` before descending; use the current
        # altitude so the return leg neither climbs nor drops unexpectedly.
        st = self.backend.get_state()
        h = self.backend.land(max(st.alt, min_alt))
        return self._await(h, None, timeout or self.land_timeout, landing=True)

    def abort(self) -> ActionOutcome:
        h = self.backend.hold()
        st = self.backend.get_state()
        reason = "holding position" if h.ok else f"could not hold: {h.result}"
        return ActionOutcome(ABORTED, reason, st, 0.0, 0.0, 0.0, None, None)

    # --- the polling loop -----------------------------------------------
    def _await(self, handle: ActionHandle, target, timeout: float,
               hold_full_duration: bool = False, landing: bool = False) -> ActionOutcome:
        t0 = self.backend.now()
        prev = self.backend.get_state()
        path = 0.0
        peak = 0.0
        status, reason = TIMEOUT, ""

        while True:
            if handle.done and not handle.ok:
                status, reason = FAILED, handle.result
                break
            self._tick()
            st = self.backend.get_state()
            path += math.dist(prev.pos, st.pos)
            peak = max(peak, st.speed)
            prev = st
            elapsed = self.backend.now() - t0

            if not st.ok:
                status, reason = FAILED, "no telemetry from the vehicle"
                break

            why = self.interrupt() if self.interrupt else None
            if why:
                status, reason = ABORTED, why
                break

            if landing:
                if handle.done and not st.armed:
                    status, reason = ARRIVED, "landed and disarmed"
                    break
            elif hold_full_duration:
                if elapsed >= timeout:
                    status, reason = ARRIVED, "hold complete"
                    break
            elif target is not None and handle.done:
                d = math.dist(st.pos, target)
                if d <= self.arrive_radius and st.speed < self.settle_speed:
                    status, reason = ARRIVED, ""
                    break

            if not hold_full_duration and elapsed >= timeout:
                status = TIMEOUT
                reason = f"did not {'land' if landing else 'arrive'} within {timeout:.0f} s"
                break

        st = self.backend.get_state()
        dist = math.dist(st.pos, target) if target else None
        return ActionOutcome(
            status=status, reason=reason, state=st,
            elapsed=self.backend.now() - t0, path_length=path,
            max_speed=peak, distance_to_target=dist, target=target,
        )

    def _tick(self) -> None:
        """Let time pass. The ROS backend is spun by its own thread."""
        time.sleep(self.poll_dt)
