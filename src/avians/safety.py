"""The arbiter. Every command passes through here before it reaches the vehicle.

The model is untrusted input. Rejections are returned to it as readable
observations so it can adapt, and so the recovery lands in the training data.

GLOBAL LIMITS ONLY: geofence, altitude band, speed, durations, battery. Mission
regions (no-fly zones and the like) are deliberately NOT enforced here: whether
the agent respects them is what the STL score measures. Enforcing them would
make that clause measure this file instead of the agent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .backends.base import VehicleState


@dataclass
class Limits:
    geofence_radius_m: float = 50.0
    max_altitude_m: float = 30.0
    min_altitude_m: float = 1.0
    max_speed_ms: float = 8.0
    min_battery_pct: float = 20.0
    max_hover_s: float = 60.0
    max_orbit_s: float = 120.0
    # plan-ahead flight (trajectory.Envelope)
    max_accel_ms2: float = 3.0
    max_climb_ms: float = 3.0
    max_descent_ms: float = 1.5
    corner_radius_m: float = 4.0
    max_plan_s: float = 600.0
    # handed to PX4 only (environment.envelope_params); None leaves PX4's value
    max_tilt_deg: float | None = None
    max_jerk_ms3: float | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Limits":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


def resolve(cfg: dict, envelope: str | None = None) -> dict:
    """limits.yaml with one envelope profile applied: a flat dict."""
    name = envelope or cfg.get("envelope")
    profiles = cfg.get("envelopes", {})
    if name is not None and name not in profiles:
        raise ValueError(f"unknown envelope '{name}'; known: {', '.join(profiles) or 'none'}")
    flat = {k: v for k, v in cfg.items() if k not in ("envelopes", "envelope")}
    flat.update(profiles.get(name, {}))
    flat["envelope"] = name
    return flat


AIRBORNE_TOOLS = ("reposition", "goto_region", "orbit", "hover", "land")


class SafetyArbiter:
    def __init__(self, limits: Limits):
        self.limits = limits

    def check(self, tool: str, args: dict, state: VehicleState) -> tuple[bool, str]:
        """Return (allowed, reason). reason is written for the model to read.

        goto_region arrives here already resolved to east/north (tools.py)."""
        L = self.limits

        if state.battery_pct is not None and state.battery_pct < L.min_battery_pct:
            if tool not in ("land", "get_state", "abort", "check_path"):
                return False, (
                    f"battery at {state.battery_pct:.0f}% is below the "
                    f"{L.min_battery_pct:.0f}% floor; land now"
                )

        if tool in AIRBORNE_TOOLS and not state.armed:
            return False, "the drone is not airborne; call takeoff first"

        if tool == "takeoff":
            if state.armed:
                return False, "already airborne; use reposition to change position"
            reason = self._altitude(args["altitude"])
            if reason:
                return False, reason

        if tool in ("reposition", "goto_region"):
            e, n = args["east"], args["north"]
            r = math.hypot(e, n)
            if r > L.geofence_radius_m:
                scale = L.geofence_radius_m / r
                return False, (
                    f"target is {r:.1f} m from launch, outside the "
                    f"{L.geofence_radius_m:.0f} m geofence; the nearest allowed point on "
                    f"that bearing is east={e * scale:.1f} north={n * scale:.1f}"
                )
            reason = self._altitude(args["altitude"]) or self._speed(args.get("speed"))
            if reason:
                return False, reason

        if tool == "orbit":
            far = math.hypot(args["east"], args["north"]) + args["radius"]
            if far > L.geofence_radius_m:
                return False, (
                    f"the orbit reaches {far:.1f} m from launch, outside the "
                    f"{L.geofence_radius_m:.0f} m geofence; move the centre closer or "
                    f"shrink the radius"
                )
            reason = self._altitude(args["altitude"])
            if reason:
                return False, reason
            if args["duration"] > L.max_orbit_s:
                return False, f"duration exceeds the {L.max_orbit_s:.0f} s maximum for one orbit"

        if tool == "hover" and args["duration"] > L.max_hover_s:
            return False, f"duration exceeds the {L.max_hover_s:.0f} s maximum for one hover"

        return True, ""

    def check_plan(self, traj, state: VehicleState) -> tuple[bool, str]:
        """A whole flight plan (trajectory.Trajectory), point by point. The
        planner already keeps speed and acceleration inside the envelope; this
        checks where the plan goes and for how long."""
        L = self.limits
        if not state.armed:
            return False, "the drone is not airborne; call takeoff first"
        if traj.duration > L.max_plan_s:
            return False, (f"the plan takes {traj.duration:.0f} s; the maximum is "
                           f"{L.max_plan_s:.0f} s, split it and append the rest later")
        for k in traj.knots:
            if k.dwell and k.dwell > L.max_hover_s:
                return False, f"a hold of {k.dwell:g} s exceeds the {L.max_hover_s:.0f} s maximum"
        for t, (e, n, a) in zip(traj.t, traj.pos):
            r = math.hypot(e, n)
            if r > L.geofence_radius_m:
                return False, (f"at t+{t:.0f} s the plan is {r:.1f} m from launch "
                               f"(east={e:.1f} north={n:.1f}), outside the "
                               f"{L.geofence_radius_m:.0f} m geofence")
            if a > L.max_altitude_m:
                return False, (f"at t+{t:.0f} s the plan climbs to {a:.1f} m, above the "
                               f"{L.max_altitude_m:.0f} m ceiling")
            if a < L.min_altitude_m - 0.3 and t > 0:
                return False, (f"at t+{t:.0f} s the plan is at {a:.1f} m, below the "
                               f"{L.min_altitude_m:g} m floor; end the plan with land "
                               f"to come down")
        return True, ""

    def _altitude(self, alt: float) -> str:
        L = self.limits
        if alt > L.max_altitude_m:
            return (f"altitude {alt:.1f} m exceeds the {L.max_altitude_m:.0f} m ceiling; "
                    f"choose {L.max_altitude_m:.0f} m or lower")
        if alt < L.min_altitude_m:
            return (f"altitude {alt:.1f} m is below the {L.min_altitude_m:g} m floor; "
                    f"use land to come down")
        return ""

    def _speed(self, speed: float | None) -> str:
        if speed is not None and speed > self.limits.max_speed_ms:
            return f"speed {speed:.1f} m/s exceeds the {self.limits.max_speed_ms:g} m/s limit"
        return ""
