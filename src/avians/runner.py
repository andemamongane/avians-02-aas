"""The mission runner: a tool-use loop with any pilot (Claude, DeepSeek, Codex...).

Deliberately a plain loop, with no agent framework. The serialized messages ARE
the training data, so nothing may sit between the model and the log.

Three independent caps end an episode cleanly, each catching a different
failure: step cap (model loops), wall-clock cap (something hangs), token cap
(runaway cost). Whichever trips is recorded in meta.json as terminated_by.
The online STL monitor (monitor.py) can also end it once the outcome is
decided (stl_violated / stl_satisfied); the runner then holds and lands.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .maps import Map
from .schema import TOOL_SPECS
from .tools import ToolLayer

BASE_PROMPT = """You are piloting a quadrotor drone through a small set of tools that command its autopilot.

Frame and units: positions are metres relative to the launch point: east, north,
and altitude ABOVE the launch point (always positive). Times are seconds from the
start of the mission; every tool result reports the mission time.
There is no camera and no obstacle sensing: you know your own state and the map.

Rules:
- Call takeoff before anything else.
- One tool call at a time. Wait for the result before deciding the next step.
- Tool results tell you whether the action arrived, timed out, failed, or was
  rejected. A rejection includes the reason; adapt rather than repeating the call.
- land flies back to the launch point at the current altitude and lands there.
  Call it when the mission is complete, then stop.
- If you are unsure or something is wrong, call abort.
- The safety system enforces only a geofence and altitude/speed limits. It does
  NOT know about the map regions: respecting them is your responsibility."""

SPEC_GUIDE = """The mission's requirements in signal temporal logic (STL). Your flight is scored
against this formula; times are seconds from the start of the mission.
  in(r)       the drone is inside region r (and within its altitude band, if any)
  G[a,b] p    p holds at every moment from a to b seconds
  F[a,b] p    p holds at some moment between a and b seconds
  p U[a,b] q  q becomes true between a and b seconds, and p holds until then
  ! & | ->    not, and, or, implies
Requirements:
  {spec}"""


PLAN_PROMPT = """You are piloting a quadrotor drone by planning its flight. You command it with
flight plans, not single moves.

Frame and units: positions are metres relative to the launch point: east, north,
and altitude ABOVE the launch point (always positive). Times are seconds from the
start of the mission; every tool result reports the mission time.
There is no camera and no obstacle sensing: you know your own state and the map.

How control works:
- The drone flies your current plan continuously while you think. Every tool
  returns at once; what happened since your last call (segments completed,
  regions entered or left, the plan completing, landing) arrives as events.
- Think ahead. Send the whole route in one fly_plan. Waypoints are flown
  THROUGH, not stopped at: the planner rounds the corners and slows only where
  a turn demands it. Stop (hold) only where the mission needs the drone to be
  stationary.
- Fly decisively. The flight envelope allows up to {speed:g} m/s and
  {accel:g} m/s^2 of acceleration; the planner uses all of it unless you cap a
  leg's speed. Fast, smooth, efficient flight is the goal.
- To change course, send a new fly_plan: it replaces the old one from wherever
  the drone is, keeping its momentum. To extend a plan, use mode "append".
- While a plan runs, use wait to let time pass until the next event instead of
  polling get_state.
- Call takeoff first; you can send fly_plan right away. End the mission with a
  plan whose last segment is land, routed by you around regions you must avoid,
  then wait until the drone has landed, then stop.
- The safety system enforces only a {geofence:g} m geofence around the launch
  point, an altitude band of {floor:g}-{ceiling:g} m and the flight envelope. It
  does NOT know about the map regions: respecting them is your responsibility."""


def build_system_prompt(mission_map: Map | None, spec: str | None, control: str = "step",
                        limits=None) -> str:
    if control == "plan":
        from .safety import Limits
        L = limits or Limits()
        parts = [PLAN_PROMPT.format(speed=L.max_speed_ms, accel=L.max_accel_ms2,
                                    geofence=L.geofence_radius_m, floor=L.min_altitude_m,
                                    ceiling=L.max_altitude_m)]
    else:
        parts = [BASE_PROMPT]
    if mission_map is not None and mission_map.regions:
        parts.append("Mission map. Region vertices are (east, north) in metres:\n"
                     + mission_map.describe())
    if spec:
        parts.append(SPEC_GUIDE.format(spec=spec))
    if control != "plan":
        parts.append("Be efficient: every extra call costs time and battery.")
    return "\n\n".join(parts)


@dataclass
class RunCaps:
    max_steps: int = 25
    max_wall_s: float = 300.0
    max_tokens_total: int = 300_000


@dataclass
class RunResult:
    steps: int = 0
    terminated_by: str = "model_stop"
    usage: dict = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})
    error: str | None = None


class MissionRunner:
    """Flies one mission with any pilot (see pilots.py). The log is in one
    canonical format whichever provider flew, so episodes are comparable."""

    def __init__(self, pilot, tools: ToolLayer, logger, model: str,
                 caps: RunCaps | None = None, system_prompt: str | None = None,
                 monitor=None):
        self.pilot = pilot
        self.tools = tools
        self.log = logger
        self.model = model
        self.caps = caps or RunCaps()
        self.system_prompt = system_prompt or build_system_prompt(None, None)
        self.monitor = monitor

    # ------------------------------------------------------------------
    def run(self, mission_name: str, mission_prompt: str, mission_cfg: dict) -> RunResult:
        result = RunResult()
        self.log.system(self.system_prompt, self._specs())
        self.log.mission(mission_name, mission_prompt, mission_cfg)
        if self.monitor is not None:
            self.monitor.start()
            self.tools.set_interrupt(self.monitor.poll)
        try:
            self._loop(result, mission_prompt)
            if not (self.monitor and self.monitor.terminated_by):
                self._finish_plan(result)
        finally:
            if self.monitor is not None:
                self.tools.set_interrupt(None)
                if self.monitor.terminated_by:
                    self._land_after_stop()
                self.monitor.close()
        return result

    def _specs(self) -> list[dict]:
        return getattr(self.tools, "specs", TOOL_SPECS)

    def _finish_plan(self, result: RunResult) -> None:
        """Plan-ahead control: the pilot may stop talking while the drone still
        flies the plan it committed to (typically ending with a landing). Let it
        finish, still under the monitor; logged as notes."""
        finish = getattr(self.tools, "finish", None)
        if finish is None:
            return
        out = finish()
        if out is None:
            return
        status, reason, st = out
        if self.monitor is not None and self.monitor.terminated_by:
            result.terminated_by = self.monitor.terminated_by
        self.log.note("the drone finished the pilot's plan", status=status, reason=reason,
                      state=_state(st))

    def _stopped(self, result: RunResult) -> bool:
        if self.monitor is None or self.monitor.poll() is None:
            return False
        result.terminated_by = self.monitor.terminated_by
        return True

    def _land_after_stop(self) -> None:
        """The monitor ended the mission: hold, then land. Not a model action, so
        it is logged as notes, never as tool calls."""
        self.monitor.disarm()
        st = self.tools.backend.get_state()
        self.log.note("stl monitor ended the mission", reason=self.monitor.stop_reason,
                      report=self.monitor.last, state=_state(st))
        if not st.armed:
            return
        t0 = time.time()
        status, reason, final = self.tools.emergency_land()
        self.log.note("monitor landed the drone", status=status, reason=reason,
                      elapsed_s=round(time.time() - t0, 1), state=_state(final))

    def _loop(self, result: RunResult, mission_prompt: str) -> None:
        self.pilot.start(self.system_prompt, self._specs(), mission_prompt)

        messages: list[dict[str, Any]] = [{"role": "user", "content": mission_prompt}]
        t0 = time.time()

        while True:
            if self._stopped(result):
                break
            if result.steps >= self.caps.max_steps:
                result.terminated_by = "step_cap"
                break
            if time.time() - t0 > self.caps.max_wall_s:
                result.terminated_by = "wall_clock_cap"
                break
            total_tokens = result.usage["input_tokens"] + result.usage["output_tokens"]
            if total_tokens > self.caps.max_tokens_total:
                result.terminated_by = "token_cap"
                break

            self.tools.tick()                       # plan mode: events while it flew
            self.log.request(messages, self.model)
            try:
                reply = self.pilot.step()
            except Exception as e:                  # network/rate-limit/overload/bad request
                result.terminated_by = "api_error"
                result.error = f"{type(e).__name__}: {e}"
                self.log.note("api error", error=result.error)
                break

            result.usage["input_tokens"] += reply.usage["input_tokens"]
            result.usage["output_tokens"] += reply.usage["output_tokens"]
            self.log.response(reply.content, reply.stop_reason, reply.usage)

            messages.append({"role": "assistant", "content": reply.content})

            tool_uses = reply.tool_uses
            if not tool_uses:
                result.terminated_by = "model_stop"
                break

            tool_results, native_results = [], []
            for block in tool_uses:
                name, args, uid = block["name"], block.get("input", {}), block["id"]
                self.log.tool_call(name, args, uid)
                obs = self.tools.dispatch(name, args)
                text = obs.to_text()
                self.log.observation(obs.to_dict(), text, uid)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": uid, "content": text}
                )
                native_results.append((uid, text))
                result.steps += 1

            self.pilot.add_tool_results(native_results)
            messages.append({"role": "user", "content": tool_results})


def _state(st) -> dict:
    """Vehicle state for a note: the trace of actions the model did not take
    (no observation records them; plots without a rosbag need them)."""
    return {"east": st.east, "north": st.north, "alt": st.alt,
            "v_east": st.v_east, "v_north": st.v_north, "v_up": st.v_up}
