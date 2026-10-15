"""The agent-facing layer: validate -> resolve map references -> safety -> skills
-> observation.

Returns an Observation for EVERY call, including failures. Nothing raises out
of dispatch(): an exception the model cannot see is a lost recovery example.
"""
from __future__ import annotations

import time
import traceback

from .backends.base import VehicleState
from .maps import Map
from .safety import SafetyArbiter
from .schema import (
    BLOCKED, FAILED, OK, REJECTED,
    Observation, ToolArgError, validate_tool_args,
)
from .skills import ActionOutcome, Skills


class ToolLayer:
    def __init__(self, skills: Skills, arbiter: SafetyArbiter, mission_map: Map | None = None):
        self.skills = skills
        self.arbiter = arbiter
        self.map = mission_map
        self.backend = skills.backend
        self.t_start = self.backend.now()
        self.call_count = 0

    # ------------------------------------------------------------------
    def dispatch(self, name: str, args: dict) -> Observation:
        self.call_count += 1
        try:
            clean = validate_tool_args(name, args)
            clean = self._resolve(name, clean)
        except ToolArgError as e:
            return self._observe(REJECTED, str(e))

        state = self.backend.get_state()
        allowed, reason = self.arbiter.check(name, clean, state)
        if not allowed:
            status = BLOCKED if "not airborne" in reason or "already airborne" in reason else REJECTED
            return self._observe(status, reason)

        try:
            return self._run(name, clean)
        except Exception as e:                      # never let it escape
            traceback.print_exc()
            return self._observe(FAILED, f"{type(e).__name__}: {e}")

    def _resolve(self, name: str, args: dict) -> dict:
        """goto_region -> a concrete point inside the region (checked by safety
        like any other reposition)."""
        if name != "goto_region":
            return args
        region = self.map.get(args["region"]) if self.map else None
        if region is None:
            known = ", ".join(self.map.regions) if self.map else "none (this mission has no map)"
            raise ToolArgError(f"unknown region '{args['region']}'; regions: {known}")
        e, n = region.interior_point()
        return {**args, "east": round(e, 2), "north": round(n, 2)}

    # ------------------------------------------------------------------
    def _run(self, name: str, args: dict) -> Observation:
        if name == "get_state":
            return self._observe(OK, "")
        if name == "check_path":
            return self._check_path(args["east"], args["north"])
        if name == "abort":
            out = self.skills.abort()
            return self._from_outcome(out, reason=args.get("reason", "aborted by agent"))

        if name == "takeoff":
            out = self.skills.takeoff(args["altitude"])
        elif name in ("reposition", "goto_region"):
            out = self.skills.reposition(args["east"], args["north"], args["altitude"],
                                         args.get("speed"))
            if name == "goto_region" and not out.reason:
                out.reason = (f"in {args['region']} at east={args['east']:g} "
                              f"north={args['north']:g}")
        elif name == "orbit":
            out = self.skills.orbit(args["east"], args["north"], args["altitude"],
                                    args["radius"], args["duration"])
        elif name == "hover":
            out = self.skills.hover(args["duration"])
        elif name == "land":
            out = self.skills.land()
        else:                                        # unreachable: validated above
            return self._observe(FAILED, f"unhandled tool {name}")

        return self._from_outcome(out)

    def _check_path(self, east: float, north: float) -> Observation:
        st = self.backend.get_state()
        if self.map is None or not self.map.regions:
            info = "this mission has no map regions"
        else:
            start = [rid for rid, r in self.map.regions.items()
                     if r.contains_xy(st.east, st.north)]
            hits = [rid for rid in self.map.crossings((st.east, st.north), (east, north))
                    if rid not in start]
            if hits:
                info = "the straight path crosses: " + ", ".join(
                    f"{rid} [{self.map.regions[rid].kind}]" for rid in hits)
            else:
                info = "the straight path crosses no map regions"
            if start:
                info += f" (it starts inside {', '.join(start)})"
            info = (f"path from east={st.east:.1f} north={st.north:.1f} to "
                    f"east={east:g} north={north:g}: {info}")
        obs = self._observe(OK, "")
        obs.info = info
        return obs

    # ------------------------------------------------------------------
    def _from_outcome(self, out: ActionOutcome, reason: str | None = None) -> Observation:
        obs = self._base(out.status, reason if reason is not None else out.reason, out.state)
        obs.target = out.target
        obs.distance_to_target = out.distance_to_target
        obs.action_elapsed_s = out.elapsed
        obs.path_length_m = out.path_length
        obs.max_speed_ms = out.max_speed
        return obs

    def _observe(self, status: str, reason: str) -> Observation:
        return self._base(status, reason, self.backend.get_state())

    def _base(self, status: str, reason: str, st: VehicleState) -> Observation:
        obs = Observation(
            status=status, reason=reason,
            east=st.east, north=st.north, alt=st.alt,
            v_east=st.v_east, v_north=st.v_north, v_up=st.v_up,
            heading_deg=st.heading_deg, mode=st.mode, armed=st.armed,
            battery_pct=st.battery_pct,
            mission_elapsed_s=self.backend.now() - self.t_start,
            t_sim=st.t_sim, t_wall=time.time(),
        )
        if self.map is not None and self.map.regions:
            obs.inside = self.map.inside(st.east, st.north, st.alt)
            obs.region_distance_m = {
                rid: round(r.robustness(st.east, st.north, st.alt), 1)
                for rid, r in self.map.regions.items()
            }
        return obs
