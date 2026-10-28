"""Plan-ahead control as decisions: a receding horizon, decision points at
events or every wake_s, self-contained prompts with the pilot's own notes,
lockstep time, and a range-limited sensor that reveals regions in flight."""
from __future__ import annotations

import json
import time

import pytest
import yaml

from avians import cli, clock
from avians.backends.fake import FakeBackend
from avians.executor import PlanExecutor
from avians.logging.adp import load_jsonl
from avians.logging.trajectory import TrajectoryLogger
from avians.maps import Map
from avians.runner import MissionRunner, RunCaps, build_system_prompt
from avians.safety import Limits, SafetyArbiter, resolve
from avians.schema import OK, REJECTED
from avians.sensing import SensedMap
from avians.testing_stub import StubPilot
from avians.tools import DecisionToolLayer
from avians.trajectory import Envelope, build, expand

from conftest import REPO

AGGRESSIVE = Limits.from_dict(resolve(yaml.safe_load((REPO / "config/limits.yaml").read_text()),
                                      "aggressive"))
ENV = Envelope.from_limits(AGGRESSIVE)
WP = lambda e, n, a=5, **kw: {"type": "waypoint", "east": e, "north": n, "altitude": a, **kw}


@pytest.fixture
def stack(clock, field_map):
    """A fake drone with plan-ahead decisions: 15 s horizon, 3 s wake-ups."""
    backend = FakeBackend(time_fn=clock.now)
    backend.connect()
    ex = PlanExecutor(backend, SafetyArbiter(AGGRESSIVE), field_map, ENV, horizon_s=15.0,
                      plan_ending_s=7.0)
    return backend, ex, DecisionToolLayer(ex, field_map, wake_s=3.0)


def airborne(tools):
    tools.dispatch("takeoff", {"altitude": 5})
    tools.dispatch("keep", {})


# --- the clock ----------------------------------------------------------------------------------
def test_lockstep_time_only_moves_when_avians_moves_it():
    clock.use_lockstep(start=1000.0)
    try:
        t0 = clock.now()
        time.sleep(0.05)                                 # a model thinking: nothing moves
        assert clock.now() == t0 and clock.lockstep()
        clock.sleep(2.5)
        assert clock.now() == t0 + 2.5
    finally:
        clock.use_realtime()
    assert not clock.lockstep() and abs(clock.now() - time.time()) < 1


# --- the receding horizon --------------------------------------------------------------------
def test_long_plans_fly_while_the_pilot_keeps_deciding(stack):
    """Plans may be any length; every decision, keep included, renews the
    15 s the drone may fly on it."""
    backend, ex, tools = stack
    airborne(tools)
    assert tools.dispatch("fly_plan", {"segments": [WP(120, 0), WP(0, 0)]}).status == OK  # ~240 m
    for _ in range(4):                                   # ~12 s of decisions
        obs = tools.dispatch("keep", {})
        assert not any("no decision" in e for e in obs.events)
    assert backend.get_state().speed > 3                 # still flying the ~20 s plan
    ex.wait(14.0, "plan_complete")                       # and it completes in full
    assert abs(backend.get_state().east) < 0.5


def test_the_dead_mans_switch_holds_a_drone_nobody_is_deciding_for(stack):
    backend, ex, tools = stack
    airborne(tools)
    tools.dispatch("fly_plan", {"segments": [WP(120, 0), WP(0, 0)]})
    ex.wait(16.0, "time")                                # the pilot goes silent
    assert any("no decision for 15 s: braking to a hover" in e for e in ex.events)
    ex.wait(4.0, "time")
    assert backend.get_state().speed < 0.1


def test_a_landing_is_finished_even_without_decisions(stack):
    backend, ex, tools = stack
    airborne(tools)
    tools.dispatch("fly_plan", {"segments": [WP(60, 0), WP(60, 60), {"type": "land"}]})
    ex.wait(60.0, "landed")
    assert ex.phase == "landed" and not any("no decision" in e for e in ex.events)


def test_the_plan_ending_warning_comes_once_in_time_to_extend(stack):
    _, ex, tools = stack
    airborne(tools)
    tools.dispatch("fly_plan", {"segments": [WP(90, 0)]})
    events = []
    for _ in range(8):
        events += tools.dispatch("keep", {}).events
    warnings = [e for e in events if "plan ends in" in e]
    assert len(warnings) == 1 and "plan ends in 7 s" in warnings[0]


def test_extending_during_a_hold_keeps_the_rest_of_it(stack, clock):
    """Regression: an append in the middle of a hold dropped what was left of it."""
    backend, ex, tools = stack
    airborne(tools)
    ex.fly_plan([{"type": "hold", "duration": 10}])
    ex.wait(4.0, "time")                                 # 4 s into the 10 s hold
    assert ex.fly_plan([WP(20, 0)], mode="append")[0] == OK
    x0 = backend.get_state().east
    ex.wait(5.5, "time")                                 # still holding: 6 s were left
    assert abs(backend.get_state().east - x0) < 0.05
    ex.wait(3.0, "time")                                 # then on to the waypoint
    assert backend.get_state().east - x0 > 1.0


def test_hold_events_fire_when_the_hold_ends(field_map):
    tr = build((0, 0, 5), (0, 0, 0), expand([WP(10, 0), {"type": "hold", "duration": 6}],
                                            (0, 0, 5), ENV, field_map), ENV)
    (t_event, label), = tr.events
    assert "held 6 s" in label and t_event == pytest.approx(tr.knot_times[0] + 6.0)


def test_only_horizontal_momentum_is_carried(field_map):
    """Regression: right after takeoff a new plan kept climbing first."""
    knots = expand([{"type": "hold", "duration": 2}], (0, 0, 5), ENV, field_map)
    climbing = build((0, 0, 5), (0, 0, 3.0), knots, ENV)
    still = build((0, 0, 5), (0, 0, 0), knots, ENV)
    assert climbing.duration == pytest.approx(still.duration, abs=0.1)
    assert max(p[2] for p in climbing.pos) <= 5.0 + 1e-9


# --- decisions and their prompts ------------------------------------------------------------
def test_each_decision_flies_on_to_the_next_decision_point(stack, clock):
    backend, ex, tools = stack
    airborne(tools)
    tools.dispatch("fly_plan", {"segments": [WP(80, 0)]})
    t0 = ex.mission_time()
    obs = tools.dispatch("keep", {})
    assert 0 < ex.mission_time() - t0 <= 3.0 + 1e-6     # at an event, or wake_s at most
    assert obs.status == OK and "keeping" in obs.reason


def test_notes_persist_and_history_is_bounded(stack):
    _, ex, tools = stack
    tools.dispatch("takeoff", {"altitude": 5, "notes": "survey A, then home",
                               "reason": "start"})
    for _ in range(4):
        tools.dispatch("keep", {})
    assert tools.notes == "survey A, then home"
    assert len(tools.history) == 3
    tools.dispatch("keep", {"notes": "changed my mind"})
    assert tools.notes == "changed my mind"


def test_the_decision_prompt_is_self_contained(stack):
    _, ex, tools = stack
    airborne(tools)
    tools.dispatch("fly_plan", {"segments": [WP(0, 16), {"type": "region", "region": "survey_a",
                                                         "altitude": 5}],
                                "notes": "detour north of the lines"})
    prompt = tools.decision_prompt("Survey field A, then land.", lockstep=True)
    for section in ("MISSION", "NOW", "LAST RESULT AND CURRENT STATE", "PLAN",
                    "TIME IN REGIONS SO FAR", "MAP AS YOU KNOW IT", "YOUR NOTES",
                    "YOUR LAST DECISIONS"):
        assert section in prompt
    assert "survey_a" in prompt and "detour north of the lines" in prompt
    assert "paused while you decide" in prompt and "s of plan left" in prompt


def test_the_system_prompt_has_no_map_under_plan_control(field_map):
    plan = build_system_prompt(field_map, "F[0,10] in(survey_a)", "plan", AGGRESSIVE,
                               decisions={"horizon_s": 15, "wake_s": 3})
    assert "survey_a" not in plan.split("Requirements")[0]   # the map is per decision
    assert "the next 15 s of flight" in plan and "within 3 s" in plan
    assert "survey_a" in build_system_prompt(field_map, None, "step")


# --- the decision loop -------------------------------------------------------------------------
def run_decisions(clock, tmp_path, field_map, replies, max_wall_s=120):
    """Fly a scripted pilot through the decision loop; replies are raw tool-use lists."""
    class Scripted(StubPilot):
        def __init__(self):
            super().__init__([("keep", {})])
            self.turn, self.prompts = 0, []

        def start(self, system, tools, prompt):
            self.prompts.append(prompt)

        def step(self):
            from avians.pilots import Reply
            uses = replies[self.turn] if self.turn < len(replies) else []
            self.turn += 1
            content = [{"type": "tool_use", "id": f"u{self.turn}_{k}", "name": n, "input": a}
                       for k, (n, a) in enumerate(uses)]
            return Reply(content or [{"type": "text", "text": "ok"}],
                         "tool_use" if uses else "end_turn", {"input_tokens": 1, "output_tokens": 1})

        def add_tool_results(self, results):
            pass

    backend = FakeBackend(time_fn=clock.now)
    backend.connect()
    ex = PlanExecutor(backend, SafetyArbiter(AGGRESSIVE), field_map, ENV, horizon_s=15.0)
    tools = DecisionToolLayer(ex, field_map, wake_s=3.0)
    logger = TrajectoryLogger(tmp_path)
    pilot = Scripted()
    result = MissionRunner(pilot, tools, logger, "stub", caps=RunCaps(max_wall_s=max_wall_s)
                           ).run("t", "Fly to (10, 0) and land.", {})
    logger.close()
    return result, pilot, load_jsonl(tmp_path / "trajectory.jsonl"), backend


def test_the_loop_ends_when_the_drone_has_landed(clock, tmp_path, field_map):
    replies = [[("takeoff", {"altitude": 5})],
               [("fly_plan", {"segments": [WP(10, 0), {"type": "land"}]})]]
    result, pilot, records, backend = run_decisions(clock, tmp_path, field_map, replies)
    assert result.terminated_by == "landed" and not backend.get_state().armed
    # every request is one self-contained prompt, not a growing conversation
    requests = [r for r in records if r["kind"] == "request"]
    assert all(len(r["messages"]) == 1 for r in requests)
    assert any("no decision this turn" in r.get("message", "")
               for r in records if r["kind"] == "note")      # silence counts as keep


def test_lookups_are_free_and_only_one_decision_per_turn(clock, tmp_path, field_map):
    replies = [[("check_path", {"east": 10, "north": 0}), ("takeoff", {"altitude": 5}),
                ("keep", {})],
               [("fly_plan", {"segments": [WP(10, 0), {"type": "land"}]})]]
    result, pilot, records, _ = run_decisions(clock, tmp_path, field_map, replies)
    done = [r["tool"] for r in records if r["kind"] == "action"]
    assert done[:3] == ["check_path", "takeoff", "keep"]
    ignored = [r for r in records if r["kind"] == "note" and r.get("message") == "extra call ignored"]
    assert [n["tool"] for n in ignored] == ["keep"]
    assert result.terminated_by == "landed"


# --- the sensor ---------------------------------------------------------------------------------
def test_hidden_regions_appear_once_within_range(field_map):
    view = SensedMap(field_map, ["nofly_1"], range_m=5.0)
    assert "nofly_1" not in view.regions and view.get("nofly_1") is None
    assert view.crossings((10, 0), (10, 20)) == []        # unseen: not on the pilot's map
    assert view.sense(0, 0, t=1.0) == []                  # the pad is 8 m from the lines
    found = view.sense(5, 4, t=2.0)                       # 4 m below them
    assert [r.id for r in found] == ["nofly_1"] and view.discovered == {"nofly_1": 2.0}
    assert "nofly_1" in view.regions and "nofly_1" in view.describe()
    assert "nofly_1" in view.crossings((10, 0), (10, 20))


def test_the_executor_reports_discoveries_as_events(clock, field_map):
    backend = FakeBackend(time_fn=clock.now)
    backend.connect()
    view = SensedMap(field_map, ["nofly_1"], range_m=5.0)
    ex = PlanExecutor(backend, SafetyArbiter(AGGRESSIVE), view, ENV, horizon_s=15.0)
    tools = DecisionToolLayer(ex, view, wake_s=3.0)
    airborne(tools)
    with pytest.raises(Exception):
        expand([{"type": "region", "region": "nofly_1", "altitude": 5}], (0, 0, 5), ENV, view)
    events = []
    tools.dispatch("fly_plan", {"segments": [WP(0, 6)]})
    for _ in range(3):
        events += tools.dispatch("keep", {}).events
    assert any("discovered nofly_1 [no_fly]" in e for e in events)


# --- the CLI ---------------------------------------------------------------------------------------
@pytest.fixture
def fly(clock, tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "DATA", tmp_path / "episodes")
    return lambda *argv: cli.main(["fly", *argv, "--backend", "fake", "--dry-run", "--no-plots"])


def test_lockstep_is_the_default_on_the_fake_backend_and_recorded(fly, tmp_path):
    assert fly(str(REPO / "missions/survey_then_land.yaml")) == 0
    meta = json.loads((tmp_path / "episodes/ep_0001/meta.json").read_text())
    assert meta["clock"] == "lockstep" and meta["terminated_by"] == "landed"
    assert meta["decisions"] == {"horizon_s": 15.0, "wake_s": 3.0, "plan_ending_s": 7.0,
                                 "latency_s": 1.0, "history": 3}
    assert not clock.lockstep()                           # restored after the flight


def test_realtime_has_no_artificial_latency(fly, tmp_path):
    assert fly(str(REPO / "missions/survey_then_land.yaml"), "--clock", "realtime") == 0
    meta = json.loads((tmp_path / "episodes/ep_0001/meta.json").read_text())
    assert meta["clock"] == "realtime" and meta["decisions"]["latency_s"] == 0.0


def test_sensor_missions_need_plan_control_and_lockstep_needs_the_fake(fly, monkeypatch):
    assert fly(str(REPO / "missions/slalom_blind.yaml"), "--control", "step") == 2
    assert cli.main(["fly", str(REPO / "missions/hover.yaml"), "--backend", "aas",
                     "--control", "step", "--clock", "lockstep", "--dry-run"]) == 2


def test_the_blind_slalom_is_decided_from_discoveries(fly, tmp_path):
    pytest.importorskip("matplotlib")
    from avians.plots.flight import compute_metrics, load_episode
    assert fly(str(REPO / "missions/slalom_blind.yaml")) == 0
    ep = tmp_path / "episodes/ep_0001"
    assert json.loads((ep / "stl.json").read_text())["satisfied"]
    events = [e for r in load_jsonl(ep / "trajectory.jsonl") if r["kind"] == "observation"
              for e in r["observation"].get("events") or []]
    assert sum("discovered wall_" in e for e in events) == 3
    dec = compute_metrics(load_episode(ep))["decisions"]
    assert dec["plan_changes"] >= 3 and dec["changes_after_an_event"] == 1.0
    assert dec["rejected"] <= 2                      # retried once there was room
