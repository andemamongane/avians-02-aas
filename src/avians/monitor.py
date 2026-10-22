"""Online STL monitor: scores the flight WHILE it flies, and can end it early.

It samples the vehicle state on the mission clock (t = 0 at the `mission`
record, as the offline scorer uses) and periodically computes the robust
satisfaction interval of the spec (stl.score_online). The policy decides when
that ends the episode:

    off        never; the intervals are still logged
    violated   when the spec can no longer be satisfied (hi <= 0)
    decided    also when it is satisfied whatever happens next (lo > 0)

The pilot is never shown the live verdict: that would leak the score and
change what is being measured. When the monitor stops a mission, the action
in progress returns ABORTED with the reason, and the runner holds and lands.

Sampling: with a ROS backend a thread samples at `hz`, so the trace keeps
growing while the model thinks. With the fake backend (single-threaded, and
driven by a patched clock in tests) the skills' polling loop calls poll()
instead, as does the runner between model turns.

The online trace is the agent-side state stream, not the rosbag the offline
score uses, so the two can differ by a sample or so. FutureBounds.max_speed
carries a margin for that and for overshoot.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from .maps import Map
from .stl import FutureBounds, Trace, parse, score_online

POLICIES = ("off", "violated", "decided")


class OnlineMonitor:
    def __init__(self, spec: str, mission_map: Map | None, backend,
                 policy: str = "violated", future: FutureBounds | None = None,
                 eval_every_s: float = 1.0, dt: float = 0.1,
                 log_path: Path | None = None, threaded: bool = False, hz: float = 10.0):
        if policy not in POLICIES:
            raise ValueError(f"early_stop must be one of {', '.join(POLICIES)}, not '{policy}'")
        self.spec = parse(spec)
        self.map = mission_map
        self.backend = backend
        self.policy = policy
        self.future = future
        self.eval_every_s = eval_every_s
        self.dt = dt
        self.threaded = threaded
        self.period = 1.0 / hz
        self.trace = Trace([], [], [], [], [])
        self.last: dict | None = None            # latest score_online report
        self.stop_reason: str | None = None
        self.terminated_by: str | None = None
        self._t0: float | None = None
        self._last_eval = -float("inf")
        self._lock = threading.Lock()
        self._log = log_path.open("a", encoding="utf-8") if log_path else None
        self._thread: threading.Thread | None = None
        self._running = False
        self._armed = True

    # --- lifecycle -------------------------------------------------------
    def start(self) -> None:
        """Start the mission clock (call right after logging the mission)."""
        self._t0 = time.time()
        self.sample()
        if self.threaded:
            self._running = True
            self._thread = threading.Thread(target=self._loop, name="stl-monitor", daemon=True)
            self._thread.start()

    def close(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._log is not None and not self._log.closed:
            self._log.close()

    def disarm(self) -> None:
        """Keep scoring, but stop interrupting (the runner is landing)."""
        self._armed = False

    def _loop(self) -> None:
        while self._running:
            self.sample()
            time.sleep(self.period)

    # --- sampling and scoring ---------------------------------------------
    def poll(self) -> str | None:
        """Skills.interrupt hook: the reason to stop, or None to carry on."""
        if not self.threaded:
            self.sample()
        return self.stop_reason

    def sample(self) -> None:
        if self._t0 is None:
            return
        st = self.backend.get_state()
        if not st.ok:
            return
        with self._lock:
            t = time.time() - self._t0
            tr = self.trace
            if tr.t and t <= tr.t[-1]:
                return
            tr.t.append(t), tr.east.append(st.east), tr.north.append(st.north)
            tr.alt.append(st.alt), tr.speed.append(st.speed)
            if t - self._last_eval >= self.eval_every_s:
                self._last_eval = t
                self._evaluate()

    def _evaluate(self) -> None:
        rep = score_online(self.spec, self.trace, self.map, self.dt, self.future)
        self.last = rep
        if self._log is not None:
            self._log.write(json.dumps(rep) + "\n")
            self._log.flush()
        if self.stop_reason is not None or self.policy == "off" or not self._armed:
            return
        if rep["verdict"] == "violated":
            worst = min(rep["clauses"], key=lambda c: _num(c["hi"]))
            self._stop("stl_violated",
                       f"the mission specification can no longer be satisfied at "
                       f"t={rep['t']:.1f} s: '{worst['clause']}' is at best {worst['hi']}")
        elif rep["verdict"] == "satisfied" and self.policy == "decided":
            self._stop("stl_satisfied",
                       f"the mission specification is satisfied at t={rep['t']:.1f} s, "
                       f"whatever happens next")

    def _stop(self, terminated_by: str, reason: str) -> None:
        self.terminated_by = terminated_by
        self.stop_reason = "mission ended by the monitor: " + reason

    def summary(self) -> dict:
        return {
            "policy": self.policy,
            "terminated_by": self.terminated_by,
            "reason": self.stop_reason,
            "last": self.last,
        }


def _num(v) -> float:
    return float("inf") if v == "inf" else float("-inf") if v == "-inf" else v
