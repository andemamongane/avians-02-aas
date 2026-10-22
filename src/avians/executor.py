"""Plan-ahead control: flies the pilot's plans while the pilot keeps thinking.

Step-by-step control (skills.py) blocks every tool call until the drone has
arrived, so the drone hovers whenever the model thinks: stop-and-go flight.
Here the pilot commits to a whole plan (trajectory.py), the backend flies it
continuously, and every call returns at once. What happens in between reaches
the pilot as events on its next observation; `wait` lets time pass until the
next one.

Phases: ground -> takeoff -> flying -> landing -> landed. A plan sent during
the climb is queued and starts the moment the climb completes. A new plan
replaces the current one from the drone's reference state (position AND
velocity), so the drone never stops to change its mind. Plans are checked by
the safety arbiter before they fly; STL is never consulted.

tick() advances the state machine. With a ROS backend a thread ticks it; with
the fake backend (single-threaded, a patched clock in tests) the tool layer
ticks it during `wait`, and the runner between model turns.
"""
from __future__ import annotations

import math
import threading
import time

from .maps import Map
from .safety import SafetyArbiter
from .schema import ABORTED, BLOCKED, FAILED, OK, REJECTED, ToolArgError
from .trajectory import ALONG, Envelope, Knot, Trajectory, build, expand, preview


class PlanExecutor:
    def __init__(self, backend, arbiter: SafetyArbiter, mission_map: Map | None,
                 env: Envelope, poll_hz: float = 20.0, land_timeout: float = 120.0,
                 threaded: bool = False):
        self.backend = backend
        self.arbiter = arbiter
        self.map = mission_map
        self.env = env
        self.poll_dt = 1.0 / poll_hz
        self.land_timeout = land_timeout
        self.threaded = threaded
        self.phase = "ground"
        self.traj: Trajectory | None = None
        self.traj_t0 = 0.0
        self.pending: list[Knot] | None = None      # sent during the climb
        self.takeoff_alt = 0.0
        self.interrupt = None                       # online STL monitor hook
        self.events: list[str] = []                 # since the pilot's last observation
        self.t_start = backend.now()
        self._handle = None
        self._next_event = 0
        self._plan_done = False
        self._inside: set[str] = set()
        self._started: Trajectory | None = None    # started since the last observation
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._running = False

    # --- lifecycle -------------------------------------------------------
    def start(self) -> None:
        if self.threaded and self._thread is None:
            self._running = True
            self._thread = threading.Thread(target=self._loop, name="plan-executor", daemon=True)
            self._thread.start()

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _loop(self) -> None:
        while self._running:
            self.tick()
            time.sleep(self.poll_dt)

    # --- pilot commands: (status, reason, preview text) ---------------------
    def takeoff(self, altitude: float) -> tuple[str, str, str]:
        with self._lock:
            st = self.backend.get_state()
            if self.phase != "ground" or st.armed:
                return BLOCKED, "already airborne; send a plan with fly_plan", ""
            ok, reason = self.arbiter.check("takeoff", {"altitude": altitude}, st)
            if not ok:
                return REJECTED, reason, ""
            h = self.backend.takeoff(altitude)
            if h.done and not h.ok:
                return FAILED, h.result, ""
            self._handle, self.takeoff_alt, self.phase = h, altitude, "takeoff"
            self._inside = set(self.map.inside(st.east, st.north, st.alt)) if self.map else set()
            return OK, f"climbing to {altitude:g} m; send a plan, it starts when the climb completes", ""

    def fly_plan(self, segments: list, mode: str = "replace") -> tuple[str, str, str]:
        with self._lock:
            if self.phase in ("ground", "landed"):
                return BLOCKED, "the drone is not airborne; call takeoff first", ""
            if self.phase == "landing":
                return BLOCKED, "the drone is landing", ""
            try:
                if self.phase == "takeoff":
                    return self._queue(segments, mode)
                start, v0 = self._reference()
                knots = self._knots(segments, mode, start)
                return self._fly(build(start, v0, knots, self.env), "plan accepted, flying it now")
            except ToolArgError as e:
                return REJECTED, str(e), ""

    def hold(self) -> tuple[str, str, str]:
        with self._lock:
            if self.phase != "flying":
                return BLOCKED, f"nothing to hold: the drone is {self._phase_text()}", ""
            start, v0 = self._reference()
            speed = math.sqrt(sum(c * c for c in v0))
            stop = start
            if speed > 0.1:
                d = speed * speed / (2 * ALONG * self.env.max_accel) * 1.1
                stop = tuple(start[i] + v0[i] / speed * d for i in range(3))
            knots = [Knot(stop, self.env.max_speed, dwell=0.0, label="stopped, hovering")]
            return self._fly(build(start, v0, knots, self.env), "braking to a hover")

    def land(self) -> tuple[str, str, str]:
        return self.fly_plan([{"type": "land"}])

    def wait(self, seconds: float, until: str = "event") -> tuple[str, str, str]:
        """Let time pass while the drone flies (ticks on this thread)."""
        n0 = len(self.events)
        t_end = self.backend.now() + seconds
        while True:
            why = self.tick()
            if why:
                return ABORTED, why, ""
            if until == "event" and len(self.events) > n0:
                return OK, "an event occurred", ""
            if until == "plan_complete" and (self._plan_done or self.phase in ("landing", "landed")):
                return OK, "the plan is complete", ""
            if until == "landed" and self.phase == "landed":
                return OK, "landed", ""
            if self.phase == "ground":
                return BLOCKED, "the drone is on the ground; nothing to wait for", ""
            if self.backend.now() >= t_end:
                return OK, f"waited {seconds:g} s", ""
            time.sleep(self.poll_dt)

    def emergency_land(self):
        """The monitor ended the mission: land now (no interrupts), wait for it."""
        self.interrupt = None
        self.pending = None
        status, reason, _ = self.land() if self.phase != "takeoff" else (OK, "", "")
        t_end = self.backend.now() + self.land_timeout
        while self.phase not in ("landed", "ground") and self.backend.now() < t_end:
            if self.phase == "flying" and not self.traj:
                self.land()                         # the climb just completed
            self.tick()
            time.sleep(self.poll_dt)
        landed = self.phase == "landed"
        return ("arrived" if landed else "timeout",
                "landed and disarmed" if landed else reason or "did not land in time",
                self.backend.get_state())

    def finish(self, timeout: float = 300.0):
        """The pilot has stopped: fly out a running plan (and its landing).
        None when nothing was running."""
        busy = lambda: (self.phase == "landing" or self.phase == "takeoff"
                        or (self.phase == "flying" and self.traj is not None
                            and not self._plan_done))
        if not busy():
            return None
        t_end = self.backend.now() + timeout
        while busy() and self.backend.now() < t_end:
            if self.tick():
                break                               # the monitor stopped it
            time.sleep(self.poll_dt)
        return OK, f"plan finished; the drone is {self._phase_text()}", self.backend.get_state()

    # --- the state machine --------------------------------------------------
    def tick(self) -> str | None:
        """Advance phases, emit events. Returns the monitor's stop reason, if any."""
        with self._lock:
            st = self.backend.get_state()
            why = self.interrupt() if self.interrupt else None
            if self.map is not None and self.phase != "ground":
                now_in = set(self.map.inside(st.east, st.north, st.alt))
                for rid in sorted(now_in - self._inside):
                    self._event(f"entered {rid}")
                for rid in sorted(self._inside - now_in):
                    self._event(f"left {rid}")
                self._inside = now_in

            if self.phase == "takeoff" and self._handle.done:
                if not self._handle.ok:
                    self.phase = "ground" if not st.armed else "flying"
                    self._event(f"takeoff failed: {self._handle.result}")
                else:
                    self.phase = "flying"
                    self._event(f"takeoff complete at {st.alt:.1f} m")
                    if self.pending:
                        knots, self.pending = self.pending, None
                        status, reason, _ = self._fly(
                            build(st.pos, (0.0, 0.0, 0.0), knots, self.env), "")
                        if status != OK:
                            self._event(f"queued plan not flown: {reason}")

            elif self.phase == "flying" and self.traj is not None:
                elapsed = self.backend.now() - self.traj_t0
                events = self.traj.events
                while self._next_event < len(events) and events[self._next_event][0] <= elapsed:
                    self._event(events[self._next_event][1])
                    self._next_event += 1
                if not self._plan_done and elapsed >= self.traj.duration:
                    self._plan_done = True
                    if self.traj.lands:
                        self.backend.end_follow()
                        self._handle = self.backend.land(max(st.alt, self.env_min_alt()))
                        self.phase = "landing"
                        self._event("plan complete, landing")
                    else:
                        self._event("plan complete, holding at its last point")

            elif self.phase == "landing" and self._handle is not None:
                if self._handle.done and not self._handle.ok:
                    self.phase = "flying"
                    self.traj = None
                    self._event(f"landing failed: {self._handle.result}")
                elif not st.armed:
                    self.phase = "landed"
                    self._event("landed and disarmed")
            return why

    # --- internals ------------------------------------------------------------
    def _queue(self, segments: list, mode: str) -> tuple[str, str, str]:
        st = self.backend.get_state()
        start = (st.east, st.north, self.takeoff_alt)
        if mode == "append" and self.pending:
            here = self.pending[-1].pos
            knots = self.pending + expand(segments, here, self.env, self.map)
        else:
            knots = expand(segments, start, self.env, self.map)
        traj = build(start, (0.0, 0.0, 0.0), knots, self.env)
        ok, reason = self.arbiter.check_plan(traj, st)
        if not ok:
            return REJECTED, reason, ""
        self.pending = knots
        return OK, "plan queued: it starts when the climb completes", preview(traj, self.map)

    def _knots(self, segments: list, mode: str, start) -> list[Knot]:
        if mode == "append" and self.traj is not None and not self._plan_done:
            rest = self.traj.remaining_knots(self.backend.now() - self.traj_t0)
            if rest and rest[-1].land:
                raise ToolArgError("the current plan ends with land; nothing can be appended")
            here = rest[-1].pos if rest else start
            return rest + expand(segments, here, self.env, self.map)
        return expand(segments, start, self.env, self.map)

    def _fly(self, traj: Trajectory, message: str) -> tuple[str, str, str]:
        ok, reason = self.arbiter.check_plan(traj, self.backend.get_state())
        if not ok:
            return REJECTED, reason, ""
        h = self.backend.follow(traj)
        if h.done and not h.ok:
            return FAILED, h.result, ""
        self.traj, self.traj_t0 = traj, self.backend.now()
        self._next_event, self._plan_done = 0, False
        self._started = traj
        return OK, message, preview(traj, self.map)

    def _reference(self):
        """Where the plan is now: the reference (smooth) or, without a plan,
        the measured state."""
        if self.traj is not None:
            return self.traj.sample(self.backend.now() - self.traj_t0)
        st = self.backend.get_state()
        return st.pos, (st.v_east, st.v_north, st.v_up)

    def env_min_alt(self) -> float:
        return self.arbiter.limits.min_altitude_m

    def _event(self, text: str) -> None:
        self.events.append(f"t={self.backend.now() - self.t_start:.1f} s {text}")

    def started_plan(self) -> tuple[Trajectory, float] | None:
        """The plan that started since the last observation (for plots), and
        how long it has been flying."""
        with self._lock:
            traj, self._started = self._started, None
            return (traj, self.backend.now() - self.traj_t0) if traj is not None else None

    def drain(self) -> list[str]:
        with self._lock:
            out, self.events = self.events, []
            return out

    def _phase_text(self) -> str:
        return {"ground": "on the ground", "takeoff": "climbing", "flying": "flying",
                "landing": "landing", "landed": "landed"}[self.phase]

    def status(self) -> str:
        with self._lock:
            if self.phase == "takeoff":
                return f"climbing to {self.takeoff_alt:g} m" + (
                    "; a plan is queued" if self.pending else "; no plan yet")
            if self.phase == "flying":
                if self.traj is None:
                    return "hovering, no plan"
                left = self.traj.duration - (self.backend.now() - self.traj_t0)
                return ("complete, holding at its last point" if left <= 0
                        else f"flying, {left:.0f} s of plan left")
            return self._phase_text()
