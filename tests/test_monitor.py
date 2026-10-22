"""The online STL monitor ends a mission once its outcome is decided, holds and
lands the drone, and logs the intervals. Fake backend, simulated clock."""
from __future__ import annotations

import json

import pytest

from avians import cli
from avians.backends.fake import FakeBackend
from avians.logging.adp import load_jsonl
from avians.logging.trajectory import TrajectoryLogger
from avians.monitor import OnlineMonitor
from avians.runner import MissionRunner, RunCaps
from avians.safety import Limits, SafetyArbiter
from avians.schema import ABORTED
from avians.skills import Skills
from avians.stl import FutureBounds, Trace, robustness, parse
from avians.testing_stub import StubPilot
from avians.tools import ToolLayer

from conftest import REPO

SURVEY_SPEC = ("G[0,240] !in(nofly_1) & F[0,120] G[0,10] in(survey_a) "
               "& F[0,240] (in(home_pad) & alt <= 1)")
# Straight from the pad to field A: through the power lines.
THROUGH_NOFLY = [("takeoff", {"altitude": 5}),
                 ("reposition", {"east": 15, "north": 25, "altitude": 5}),
                 ("hover", {"duration": 12}),
                 ("land", {})]


def fly(clock, tmp_path, field_map, spec, policy, script):
    backend = FakeBackend(time_fn=clock.now)
    backend.connect()
    tools = ToolLayer(Skills(backend, poll_hz=20), SafetyArbiter(Limits()), field_map)
    monitor = OnlineMonitor(spec, field_map, backend, policy=policy,
                            future=FutureBounds(max_speed=12.0, alt_range=(-1.0, 31.0)),
                            log_path=tmp_path / "stl_online.jsonl")
    logger = TrajectoryLogger(tmp_path)
    result = MissionRunner(StubPilot(script), tools, logger, "stub",
                           caps=RunCaps(max_steps=10), monitor=monitor).run("t", "fly", {})
    logger.close()
    return result, monitor, backend, load_jsonl(tmp_path / "trajectory.jsonl")


def test_violation_stops_the_flight_and_lands(clock, tmp_path, field_map):
    result, monitor, backend, records = fly(clock, tmp_path, field_map, SURVEY_SPEC,
                                            "violated", THROUGH_NOFLY)
    assert result.terminated_by == "stl_violated" and result.steps == 2
    obs = [r["observation"] for r in records if r["kind"] == "observation"]
    assert obs[-1]["status"] == ABORTED and "nofly_1" in obs[-1]["reason"]
    # the monitor's land is a note, never a tool call the model made
    assert [r["tool"] for r in records if r["kind"] == "action"] == ["takeoff", "reposition"]
    notes = [r for r in records if r["kind"] == "note"]
    assert notes[-1]["message"] == "monitor landed the drone" and notes[-1]["status"] == "arrived"
    assert not backend.get_state().armed
    # stopped as soon as the drone was inside, not at the end of the window
    assert monitor.last["t"] < 20
    assert monitor.summary()["terminated_by"] == "stl_violated"


def test_stop_agrees_with_the_offline_score(clock, tmp_path, field_map):
    _, monitor, _, _ = fly(clock, tmp_path, field_map, SURVEY_SPEC, "violated", THROUGH_NOFLY)
    assert robustness(parse(SURVEY_SPEC), monitor.trace, field_map) <= 0


def test_policy_off_logs_but_never_stops(clock, tmp_path, field_map):
    result, monitor, _, _ = fly(clock, tmp_path, field_map, SURVEY_SPEC, "off", THROUGH_NOFLY)
    assert result.terminated_by == "model_stop" and result.steps == 4
    lines = [json.loads(s) for s in (tmp_path / "stl_online.jsonl").read_text().splitlines()]
    assert len(lines) > 10 and lines[-1]["verdict"] == "violated"
    assert [ln["t"] for ln in lines] == sorted(ln["t"] for ln in lines)


def test_decided_stops_once_satisfied(clock, tmp_path, field_map):
    script = [("takeoff", {"altitude": 5}), ("hover", {"duration": 30}), ("land", {})]
    result, monitor, backend, _ = fly(clock, tmp_path, field_map,
                                      "F[0,40] G[0,10] (alt >= 4 & alt <= 6)", "decided", script)
    assert result.terminated_by == "stl_satisfied" and result.steps == 2
    assert 10 < monitor.last["t"] < 20                # 10 s at altitude, then stopped
    assert not backend.get_state().armed


def test_correct_plan_is_never_stopped(clock, tmp_path, field_map):
    plan = [("takeoff", {"altitude": 5}), ("reposition", {"east": 0, "north": 16, "altitude": 5}),
            ("goto_region", {"region": "survey_a", "altitude": 5}), ("hover", {"duration": 12}),
            ("reposition", {"east": 0, "north": 20, "altitude": 5}), ("land", {})]
    result, monitor, _, _ = fly(clock, tmp_path, field_map, SURVEY_SPEC, "violated", plan)
    assert result.terminated_by == "model_stop" and monitor.terminated_by is None
    assert monitor.last["clauses"][1]["verdict"] == "satisfied"     # survey done


def test_sampling_ignores_repeated_timestamps(clock, field_map):
    backend = FakeBackend(time_fn=clock.now)
    mon = OnlineMonitor("G[0,5] alt <= 8", field_map, backend)
    mon.start()
    mon.sample()                                      # same instant as start()
    assert len(mon.trace.t) == 1


def test_bad_policy_fails_before_flying(tmp_path):
    bad = tmp_path / "m.yaml"
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "p.txt").write_text("fly")
    bad.write_text("name: x\nprompt_file: p.txt\nspec: alt <= 5\nearly_stop: sometimes\n")
    with pytest.raises(SystemExit, match="early_stop"):
        cli.load_mission(bad)
    with pytest.raises(ValueError):
        OnlineMonitor("alt <= 5", None, None, policy="sometimes")


def test_dry_run_records_early_stop_in_meta(clock, tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    monkeypatch.setattr(cli, "DATA", tmp_path / "episodes")
    assert cli.main(["fly", str(REPO / "missions" / "survey_then_land.yaml"),
                     "--backend", "fake", "--dry-run", "--no-plots"]) == 0
    ep = tmp_path / "episodes" / "ep_0001"
    meta = json.loads((ep / "meta.json").read_text())
    assert meta["early_stop"]["policy"] == "violated"
    assert meta["early_stop"]["terminated_by"] is None
    assert (ep / "stl_online.jsonl").exists()


def test_threaded_monitor_samples_while_the_model_thinks(field_map):
    """ROS backends: a thread keeps sampling between model turns (real clock)."""
    import time
    backend = FakeBackend()
    mon = OnlineMonitor("G[0,5] alt <= 8", field_map, backend, threaded=True, hz=50,
                        eval_every_s=0.1)
    mon.start()
    time.sleep(0.3)                                   # nobody calls poll()
    mon.close()
    assert len(mon.trace.t) >= 5 and mon.last is not None
    assert mon._thread is None
