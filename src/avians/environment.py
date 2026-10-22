"""The simulated environment a mission flies in: world, wind, failures, battery.

A mission declares it in an `environment:` block. Everything here is applied
by sim.py at the start of the flight, and recorded in meta.json, so a flight
can be reproduced:

    environment:
      world: impalpable_greyness   # checked, not switched: Gazebo loads it at start
      wind:
        east: 3.0                  # mean air velocity, m/s, TOWARD east (from the west)
        north: 0.0                 # ... toward north (from the south)
        gust_std: 1.5              # gusts: std of each horizontal component, m/s
        gust_tau: 4.0              # gust correlation time, s
        seed: 7                    # same seed, same gusts
      failures:
        - {at: 30, unit: gps, type: off, duration: 8}
      battery: {drain_s: 600, min_pct: 15}
      px4_params: {COM_DISARM_LAND: 1.0}

Gusts are a first-order Gauss-Markov process per axis (the discrete form of a
Dryden-like low-pass on white noise): realistic enough to disturb tracking,
simple enough to reproduce exactly from the seed.

The flight envelope also becomes PX4 parameters here (envelope_params), so the
autopilot's own limits match what the planner and the safety arbiter assume.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

FAILURE_UNITS = ("gps", "mag", "baro", "accel", "gyro", "battery", "motor")
FAILURE_TYPES = ("off", "stuck", "garbage", "wrong", "slow", "delayed", "intermittent")


class EnvironmentError(ValueError):
    pass


@dataclass
class Wind:
    east: float = 0.0
    north: float = 0.0
    gust_std: float = 0.0
    gust_tau: float = 4.0
    seed: int = 0

    @property
    def calm(self) -> bool:
        return self.east == 0 and self.north == 0 and self.gust_std == 0

    def schedule(self, duration: float, dt: float = 0.5) -> list[tuple[float, float, float]]:
        """(t, east, north) every dt seconds over the flight: mean plus gusts."""
        rng = random.Random(self.seed)
        a = math.exp(-dt / self.gust_tau) if self.gust_tau > 0 else 0.0
        b = self.gust_std * math.sqrt(1 - a * a)
        ge = gn = 0.0
        if self.gust_std > 0:                       # start in the stationary distribution
            ge, gn = rng.gauss(0, self.gust_std), rng.gauss(0, self.gust_std)
        out = []
        for k in range(int(math.ceil(duration / dt)) + 1):
            out.append((round(k * dt, 3), round(self.east + ge, 3), round(self.north + gn, 3)))
            ge, gn = a * ge + b * rng.gauss(0, 1), a * gn + b * rng.gauss(0, 1)
        return out


@dataclass
class Failure:
    at: float
    unit: str
    type: str = "off"
    duration: float | None = None              # None: until the end of the flight


@dataclass
class Environment:
    world: str | None = None
    wind: Wind = field(default_factory=Wind)
    failures: list[Failure] = field(default_factory=list)
    battery_drain_s: float | None = None
    battery_min_pct: float | None = None
    px4_params: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_mission(cls, mission: dict) -> "Environment":
        raw = mission.get("environment") or {}
        if not isinstance(raw, dict):
            raise EnvironmentError("environment must be a mapping")
        unknown = set(raw) - {"world", "wind", "failures", "battery", "px4_params"}
        if unknown:
            raise EnvironmentError(f"environment: unknown keys {sorted(unknown)}")
        w = raw.get("wind") or {}
        bad = set(w) - {"east", "north", "gust_std", "gust_tau", "seed"}
        if bad:
            raise EnvironmentError(f"environment.wind: unknown keys {sorted(bad)}")
        wind = Wind(**{k: (int(v) if k == "seed" else float(v)) for k, v in w.items()})
        if wind.gust_std < 0 or wind.gust_tau <= 0:
            raise EnvironmentError("environment.wind: gust_std must be >= 0 and gust_tau > 0")
        failures = []
        for i, f in enumerate(raw.get("failures") or [], 1):
            if f.get("unit") not in FAILURE_UNITS or f.get("type", "off") not in FAILURE_TYPES:
                raise EnvironmentError(
                    f"environment.failures[{i}]: unit must be one of {', '.join(FAILURE_UNITS)} "
                    f"and type one of {', '.join(FAILURE_TYPES)}")
            failures.append(Failure(at=float(f["at"]), unit=f["unit"], type=f.get("type", "off"),
                                    duration=float(f["duration"]) if f.get("duration") else None))
        bat = raw.get("battery") or {}
        return cls(world=raw.get("world"), wind=wind, failures=failures,
                   battery_drain_s=bat.get("drain_s"), battery_min_pct=bat.get("min_pct"),
                   px4_params={k: float(v) for k, v in (raw.get("px4_params") or {}).items()})

    def params(self) -> dict[str, float]:
        """PX4 parameters this environment needs (battery, failure injection)."""
        out: dict[str, float] = {}
        if self.battery_drain_s is not None:
            out["SIM_BAT_DRAIN"] = float(self.battery_drain_s)
        if self.battery_min_pct is not None:
            out["SIM_BAT_MIN_PCT"] = float(self.battery_min_pct)
        if self.failures:
            out["SYS_FAILURE_EN"] = 1.0
        out.update(self.px4_params)
        return out

    def summary(self) -> dict:
        """For meta.json."""
        return {
            "world": self.world,
            "wind": vars(self.wind),
            "failures": [vars(f) for f in self.failures],
            "battery": {"drain_s": self.battery_drain_s, "min_pct": self.battery_min_pct},
            "px4_params": self.px4_params,
        }


def envelope_params(limits) -> dict[str, float]:
    """The flight envelope as PX4 multicopter parameters, so the autopilot's own
    limits match what the planner and the arbiter assume."""
    out = {
        "MPC_XY_VEL_MAX": limits.max_speed_ms,
        "MPC_XY_CRUISE": min(limits.max_speed_ms, 12.0),
        "MPC_ACC_HOR_MAX": limits.max_accel_ms2,
        "MPC_ACC_HOR": limits.max_accel_ms2,
        "MPC_Z_VEL_MAX_UP": limits.max_climb_ms,
        "MPC_Z_VEL_MAX_DN": limits.max_descent_ms,
    }
    if getattr(limits, "max_tilt_deg", None):
        out["MPC_TILTMAX_AIR"] = limits.max_tilt_deg
    if getattr(limits, "max_jerk_ms3", None):
        out["MPC_JERK_MAX"] = limits.max_jerk_ms3
    return {k: float(v) for k, v in out.items()}
