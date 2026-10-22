"""Plan-ahead control: the planner stays inside the envelope, plans fly while
the pilot thinks, and replacing a plan keeps the drone's momentum."""
from __future__ import annotations

import math
import time

import pytest

from avians.backends.fake import FakeBackend
from avians.executor import PlanExecutor
from avians.logging.adp import load_jsonl
from avians.logging.trajectory import TrajectoryLogger
from avians.runner import MissionRunner, RunCaps
from avians.safety import Limits, SafetyArbiter, resolve
from avians.schema import BLOCKED, OK, REJECTED, ToolArgError
from avians.testing_stub import StubPilot
from avians.tools import PlanToolLayer
from avians.trajectory import Envelope, build, expand

import yaml
from conftest import REPO

ENV = Envelope()                              # the aggressive envelope's numbers
AGGRESSIVE = Limits.from_dict(resolve(yaml.safe_load((REPO / "config/limits.yaml").read_text()),
                                      "aggressive"))
WP = lambda e, n, a=5, **kw: {"type": "waypoint", "east": e, "north": n, "altitude": a, **kw}


def plan(segments, m=None, start=(0.0, 0.0, 5.0), v0=(0.0, 0.0, 0.0)):
    return build(start, v0, expand(segments, start, ENV, m), ENV)


def peak_accel(tr) -> float:
    return max((math.dist(tr.vel[i], tr.vel[i - 1]) / (tr.t[i] - tr.t[i - 1])
                for i in range(1, len(tr.t)) if tr.t[i] - tr.t[i - 1] > 1e-6), default=0.0)


# --- the planner --------------------------------------------------------------
CASES = {
    "corner": [WP(60, 0), WP(60, 60)],
    "zigzag": [WP(20 * k, 15 if k % 2 else 0, 6) for k in range(1, 7)],
    "u-turn": [WP(40, 0), WP(0, 0.5)],
    "orbit": [{"type": "orbit", "east": 20, "north": 0, "altitude": 8, "radius": 10, "turns": 2}],
    "tight orbit": [{"type": "orbit", "east": 0, "north": 0, "altitude": 8, "radius": 3}],
    "survey": [WP(0, 16), {"type": "survey", "region": "survey_a", "altitude": 5, "spacing": 4},
               {"type": "hold", "duration": 3}, {"type": "land"}],
    "climb": [WP(10, 0, 35)],
}


@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("v0", [(0.0, 0.0, 0.0), (8.0, 0.0, 0.0)], ids=["at rest", "moving"])
def test_planner_stays_inside_the_envelope(field_map, name, v0):
    tr = plan(CASES[name], field_map, v0=v0)
    assert peak_accel(tr) <= ENV.max_accel * 1.02
    assert tr.top_speed <= ENV.max_speed + 1e-9
    climbs = [(b[2] - a[2]) / (t1 - t0) for a, b, t0, t1 in
              zip(tr.pos, tr.pos[1:], tr.t, tr.t[1:]) if t1 - t0 > 1e-6]
    assert max(climbs) <= ENV.max_climb * 1.05 and min(climbs) >= -ENV.max_descent * 1.05


def test_waypoints_are_flown_through_not_stopped_at():
    tr = plan([WP(60, 0), WP(60, 60)])
    i = min(range(len(tr.pos)), key=lambda k: math.dist(tr.pos[k], (60, 0, 5)))
    assert math.hypot(*tr.vel[i]) > 4.0                  # rounds the corner at speed
    assert math.dist(tr.pos[i], (60, 0, 5)) < ENV.corner_radius


def test_hold_stops_for_its_duration():
    tr = plan([WP(30, 0), {"type": "hold", "duration": 5}, WP(30, 30)])
    still = [t for t, p, v in zip(tr.t, tr.pos, tr.vel)
             if math.hypot(*v) < 1e-6 and math.dist(p, (30, 0, 5)) < 1e-6]
    assert max(still) - min(still) == pytest.approx(5.0, abs=0.3)
    (p1, _), (p2, _) = tr.sample(min(still) + 1.0), tr.sample(min(still) + 4.0)
    assert math.dist(p1, (30, 0, 5)) < 1e-6 and math.dist(p2, (30, 0, 5)) < 1e-6


def test_events_in_order_and_survey_inside_the_region(field_map):
    tr = plan(CASES["survey"], field_map)
    times = [t for t, _ in tr.events]
    assert times == sorted(times) and "survey of survey_a done" in tr.events[1][1]
    first, last = tr.knot_times[1], tr.knot_times[-2]          # the survey's own knots
    a = field_map.get("survey_a")
    rows = [p for t, p in zip(tr.t, tr.pos) if first <= t <= last]
    assert len(rows) > 100 and all(a.contains_xy(e, n) for e, n, _ in rows)


def test_momentum_is_kept_when_starting_while_moving():
    tr = plan([WP(0, 40)], v0=(10.0, 0.0, 0.0))        # moving east, told to go north
    assert math.hypot(*tr.vel[0]) == pytest.approx(10.0, rel=0.05)
    assert tr.pos[len(tr.pos) // 8][0] > 1.0           # carries on east before turning


@pytest.mark.parametrize("segments,match", [
    ([{"type": "teleport"}], "type"),
    ([{"type": "waypoint", "east": 1, "north": 2}], "altitude"),
    ([{"type": "land"}, WP(1, 1)], "nothing can follow"),
    ([{"type": "region", "region": "mars", "altitude": 5}], "unknown region"),
    ([{"type": "orbit", "east": 0, "north": 0, "altitude": 5, "radius": 0.2}], "radius"),
    ([], "non-empty"),
])
def test_bad_plans_explain_themselves(field_map, segments, match):
    with pytest.raises(ToolArgError, match=match):
        expand(segments, (0, 0, 5), ENV, field_map)


@pytest.mark.parametrize("segments,match", [
    ([WP(200, 0)], "geofence"),
    ([WP(10, 0, 60)], "ceiling"),
    ([WP(10, 0), {"type": "hold", "duration": 90}], "hold"),
])
def test_arbiter_checks_the_whole_plan(segments, match):
    from avians.backends.base import VehicleState
    ok, reason = SafetyArbiter(AGGRESSIVE).check_plan(plan(segments), VehicleState(armed=True))
    assert not ok and match in reason


# --- flying plans -------------------------------------------------------------
@pytest.fixture
def plan_stack(clock, field_map):
    backend = FakeBackend(time_fn=clock.now)
    backend.connect()
    ex = PlanExecutor(backend, SafetyArbiter(AGGRESSIVE), field_map,
                      Envelope.from_limits(AGGRESSIVE))
    return backend, ex, PlanToolLayer(ex, field_map)


def test_takeoff_returns_at_once_and_a_queued_plan_flies(plan_stack):
    backend, _, tools = plan_stack
    t0 = backend.now()
    assert tools.dispatch("takeoff", {"altitude": 5}).status == OK
    assert backend.now() - t0 < 0.5                      # no waiting for the climb
    obs = tools.dispatch("fly_plan", {"segments": [WP(20, 20), {"type": "land"}]})
    assert obs.status == OK and "queued" in obs.reason and "plan:" in obs.info
    obs = tools.dispatch("wait", {"seconds": 120, "until": "landed"})
    assert obs.status == OK and not backend.get_state().armed
    text = " | ".join(obs.events)
    for e in ("takeoff complete", "passed waypoint 1", "landing", "landed and disarmed"):
        assert e in text


def test_the_drone_flies_while_the_pilot_thinks(plan_stack, clock):
    backend, _, tools = plan_stack
    tools.dispatch("takeoff", {"altitude": 5})
    tools.dispatch("fly_plan", {"segments": [WP(40, 0), WP(40, 40)]})
    tools.dispatch("wait", {"seconds": 30, "until": "event"})          # climb done
    before = backend.get_state().pos
    time.sleep(3.0)                                      # the model thinking: no calls
    after = backend.get_state()
    assert math.dist(before, after.pos) > 10 and after.speed > 3


def test_replacing_a_plan_keeps_momentum(plan_stack, clock):
    backend, ex, tools = plan_stack
    tools.dispatch("takeoff", {"altitude": 5})
    tools.dispatch("fly_plan", {"segments": [WP(60, 0)]})
    tools.dispatch("wait", {"seconds": 30, "until": "event"})
    time.sleep(3.0)
    st = backend.get_state()
    assert st.speed > 5
    assert tools.dispatch("fly_plan", {"segments": [WP(st.east, 40)]}).status == OK
    nxt = backend.get_state()
    assert math.dist(st.pos, nxt.pos) < 0.5 and nxt.speed > 0.8 * st.speed
    time.sleep(0.5)
    assert backend.get_state().speed > 3                 # turning, not stopping


def test_append_continues_after_the_current_plan(plan_stack):
    backend, _, tools = plan_stack
    tools.dispatch("takeoff", {"altitude": 5})
    tools.dispatch("fly_plan", {"segments": [WP(20, 0)]})
    tools.dispatch("wait", {"seconds": 30, "until": "event"})
    obs = tools.dispatch("fly_plan", {"segments": [WP(20, 20)], "mode": "append"})
    assert obs.status == OK and "(20, 0, 5)" in obs.info and "(20, 20, 5)" in obs.info
    tools.dispatch("wait", {"seconds": 60, "until": "plan_complete"})
    assert math.dist(backend.get_state().pos, (20, 20, 5)) < 0.5


def test_hold_brakes_to_a_stop(plan_stack, clock):
    backend, _, tools = plan_stack
    tools.dispatch("takeoff", {"altitude": 5})
    tools.dispatch("fly_plan", {"segments": [WP(80, 0)]})
    tools.dispatch("wait", {"seconds": 30, "until": "event"})
    time.sleep(4.0)
    v = backend.get_state().speed
    x0 = backend.get_state().east
    assert tools.dispatch("hold", {}).status == OK
    tools.dispatch("wait", {"seconds": 20, "until": "plan_complete"})
    st = backend.get_state()
    assert st.speed < 0.1 and st.east - x0 < v * v / (2 * 0.6 * ENV.max_accel) * 1.5


def test_rejected_plan_leaves_the_current_one_flying(plan_stack):
    backend, ex, tools = plan_stack
    tools.dispatch("takeoff", {"altitude": 5})
    tools.dispatch("fly_plan", {"segments": [WP(30, 0)]})
    tools.dispatch("wait", {"seconds": 30, "until": "event"})
    current = ex.traj
    obs = tools.dispatch("fly_plan", {"segments": [WP(500, 0)]})
    assert obs.status == REJECTED and "geofence" in obs.reason and ex.traj is current


def test_blocked_before_takeoff_and_after_landing(plan_stack):
    _, _, tools = plan_stack
    assert tools.dispatch("fly_plan", {"segments": [WP(1, 1)]}).status == BLOCKED
    tools.dispatch("takeoff", {"altitude": 5})
    tools.dispatch("land", {})
    tools.dispatch("wait", {"seconds": 60, "until": "landed"})
    assert tools.dispatch("fly_plan", {"segments": [WP(1, 1)]}).status == BLOCKED


def test_a_committed_plan_finishes_after_the_pilot_stops(clock, tmp_path, field_map):
    """The pilot sends a plan ending in land and says it is done: the drone
    still flies it out and lands; the runner logs that as a note."""
    backend = FakeBackend(time_fn=clock.now)
    backend.connect()
    ex = PlanExecutor(backend, SafetyArbiter(AGGRESSIVE), field_map,
                      Envelope.from_limits(AGGRESSIVE))
    logger = TrajectoryLogger(tmp_path)
    script = [("takeoff", {"altitude": 5}),
              ("fly_plan", {"segments": [WP(-15, 15), {"type": "land"}]})]
    result = MissionRunner(StubPilot(script), PlanToolLayer(ex, field_map), logger, "stub",
                           caps=RunCaps(max_steps=10)).run("t", "fly", {})
    logger.close()
    assert result.terminated_by == "model_stop" and not backend.get_state().armed
    notes = [r for r in load_jsonl(tmp_path / "trajectory.jsonl") if r["kind"] == "note"]
    assert notes[-1]["message"] == "the drone finished the pilot's plan"


def test_plan_tools_are_what_the_pilot_gets(clock, tmp_path, field_map):
    backend = FakeBackend(time_fn=clock.now)
    ex = PlanExecutor(backend, SafetyArbiter(AGGRESSIVE), field_map, ENV)
    logger = TrajectoryLogger(tmp_path)
    MissionRunner(StubPilot([]), PlanToolLayer(ex, field_map), logger, "stub").run("t", "x", {})
    logger.close()
    system = load_jsonl(tmp_path / "trajectory.jsonl")[0]
    assert {t["name"] for t in system["tools"]} == {
        "get_state", "takeoff", "fly_plan", "wait", "hold", "land", "check_path"}
