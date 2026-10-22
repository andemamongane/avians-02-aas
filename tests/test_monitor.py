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

SURVEY_SPEC = ("G[0,240] !in(nofly_1) "
               "& F[0,120] (G[0,10] in(survey_a) & F[0,120] (in(home_pad) & alt <= 1))")
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
    assert monitor.last["clauses"][0]["verdict"] == "undecided"     # G runs to 240 s
    assert robustness(parse(SURVEY_SPEC), monitor.trace, field_map) > 0


def test_survey_without_landing_fails(clock, tmp_path, field_map):
    """Starting on the pad must not count as landing: the landing comes after
    the survey (the old spec's F (in(home_pad) & alt <= 1) held at t = 0)."""
    plan = [("takeoff", {"altitude": 5}), ("reposition", {"east": 0, "north": 16, "altitude": 5}),
            ("goto_region", {"region": "survey_a", "altitude": 5}), ("hover", {"duration": 12})]
    _, monitor, _, _ = fly(clock, tmp_path, field_map, SURVEY_SPEC, "off", plan)
    assert robustness(parse(SURVEY_SPEC), monitor.trace, field_map) < 0


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


def test_stopped_flight_is_plotted_with_its_landing(clock, tmp_path, monkeypatch):
    """A4: plots, summary and video see the stop, and the monitor's landing
    counts as landed even though the model never called land."""
    pytest.importorskip("matplotlib")
    import yaml
    from avians import video
    from avians.plots import plot_episode
    from avians.plots.flight import load_episode

    mission = yaml.safe_load((REPO / "missions" / "survey_then_land.yaml").read_text())
    mission["dry_run_script"] = [[n, a] for n, a in THROUGH_NOFLY]
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / mission["prompt_file"]).write_text("fly")
    path = tmp_path / "through.yaml"
    path.write_text(yaml.safe_dump(mission))
    monkeypatch.setattr(cli, "DATA", tmp_path / "episodes")
    assert cli.main(["fly", str(path), "--backend", "fake", "--dry-run", "--no-plots"]) == 0
    ep = tmp_path / "episodes" / "ep_0001"

    f = load_episode(ep)
    assert f.stop["verdict"] == "violated" and f.stop["landed"]
    assert f.z[-1] == pytest.approx(0.0, abs=0.1) and f.t[-1] > f.stop["t"]   # landing traced
    files, metrics = plot_episode(ep, 50.0, dpi=40)
    assert ep / "plots" / "07_stl_online.png" in files
    assert metrics["checks"]["landed"]
    assert metrics["online"]["decided_at"][0]["verdict"] == "violated"
    assert "## Online STL monitor" in (ep / "plots" / "summary.md").read_text()
    live = video._live(f, [r["t"] for r in f.online], f.stop["t"])
    assert live["verdict"] == "violated"
    assert video._live(f, [r["t"] for r in f.online], -1.0) is None
