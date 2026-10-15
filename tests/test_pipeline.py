"""Offline tests of everything above the AAS backend: schema, safety, skills,
tools, runner, logging, scoring. No ROS, no Docker, no API key."""
from __future__ import annotations

import json
import math

import pytest

from avians import cli
from avians.backends.fake import FakeBackend
from avians.logging.adp import to_adp, validate_episode
from avians.logging.trajectory import TrajectoryLogger
from avians.runner import MissionRunner, RunCaps, build_system_prompt
from avians.schema import (
    ABORTED, ARRIVED, BLOCKED, FAILED, OK, REJECTED, SCHEMA_VERSION,
    ToolArgError, validate_tool_args,
)
from avians.testing_stub import StubPilot

from conftest import REPO


# --- schema ---------------------------------------------------------------
def test_validate_accepts_good_args():
    out = validate_tool_args("reposition", {"east": 1, "north": 2, "altitude": 3})
    assert out == {"east": 1.0, "north": 2.0, "altitude": 3.0}


@pytest.mark.parametrize("name,args", [
    ("reposition", {"east": 0, "north": 0}),                    # missing altitude
    ("reposition", {"east": 0, "north": 0, "altitude": -3}),    # NED leak
    ("takeoff", {"altitude": 0}),
    ("hover", {"duration": -1}),
    ("orbit", {"east": 0, "north": 0, "altitude": 5, "radius": 0, "duration": 5}),
    ("goto_region", {"altitude": 5}),                           # missing region
    ("reposition", {"x": 1, "y": 2, "z": 3}),                   # schema v1 names
    ("nonsense", {}),
])
def test_validate_rejects_bad_args(name, args):
    with pytest.raises(ToolArgError):
        validate_tool_args(name, args)


def test_observation_text_uses_east_north_alt(stack):
    _, _, tools = stack
    text = tools.dispatch("get_state", {}).to_text()
    assert "east=" in text and "north=" in text and "alt=" in text
    assert "inside regions: home_pad" in text


# --- safety: global limits only -----------------------------------------
def test_reposition_before_takeoff_is_blocked(stack):
    _, _, tools = stack
    assert tools.dispatch("reposition", {"east": 1, "north": 1, "altitude": 3}).status == BLOCKED


def test_double_takeoff_is_blocked(stack):
    _, _, tools = stack
    assert tools.dispatch("takeoff", {"altitude": 3}).status == ARRIVED
    assert tools.dispatch("takeoff", {"altitude": 3}).status == BLOCKED


def test_geofence_rejection_suggests_alternative(stack):
    _, _, tools = stack
    tools.dispatch("takeoff", {"altitude": 3})
    obs = tools.dispatch("reposition", {"east": 0, "north": 100, "altitude": 3})
    assert obs.status == REJECTED
    assert "geofence" in obs.reason and "north=50.0" in obs.reason


def test_ceiling_and_orbit_geofence(stack):
    _, _, tools = stack
    tools.dispatch("takeoff", {"altitude": 3})
    assert tools.dispatch("reposition", {"east": 0, "north": 5, "altitude": 99}).status == REJECTED
    obs = tools.dispatch("orbit", {"east": 40, "north": 0, "altitude": 5,
                                   "radius": 15, "duration": 5})
    assert obs.status == REJECTED and "geofence" in obs.reason


def test_mission_no_fly_regions_are_not_enforced(stack):
    """Respecting regions is the agent's job, measured by STL, never the arbiter's."""
    _, _, tools = stack
    tools.dispatch("takeoff", {"altitude": 3})
    obs = tools.dispatch("reposition", {"east": 10, "north": 10, "altitude": 3})
    assert obs.status == ARRIVED
    assert "nofly_1" in obs.inside
    assert obs.region_distance_m["nofly_1"] > 0


# --- map tools --------------------------------------------------------------
def test_goto_region_flies_inside(stack, field_map):
    _, _, tools = stack
    tools.dispatch("takeoff", {"altitude": 5})
    obs = tools.dispatch("goto_region", {"region": "survey_a", "altitude": 5})
    assert obs.status == ARRIVED
    assert "survey_a" in obs.inside
    assert field_map.get("survey_a").contains(obs.east, obs.north, obs.alt)


def test_goto_region_unknown_lists_regions(stack):
    _, _, tools = stack
    tools.dispatch("takeoff", {"altitude": 5})
    obs = tools.dispatch("goto_region", {"region": "atlantis", "altitude": 5})
    assert obs.status == REJECTED and "survey_a" in obs.reason


def test_check_path_reports_crossings_without_moving(stack):
    _, _, tools = stack
    tools.dispatch("takeoff", {"altitude": 5})
    before = tools.dispatch("get_state", {})
    obs = tools.dispatch("check_path", {"east": 19, "north": 25})
    assert obs.status == OK and "nofly_1 [no_fly]" in obs.info
    assert math.isclose(obs.east, before.east) and math.isclose(obs.north, before.north)
    clear = tools.dispatch("check_path", {"east": 0, "north": 16})
    assert "crosses no map regions" in clear.info


# --- AAS semantics (mimicked by the fake) ----------------------------------
def test_orbit_circles_then_holds(stack):
    backend, _, tools = stack
    tools.dispatch("takeoff", {"altitude": 5})
    obs = tools.dispatch("orbit", {"east": 0, "north": 0, "altitude": 5,
                                   "radius": 4, "duration": 15})
    assert obs.status == ARRIVED and "holding" in obs.reason
    assert math.isclose(math.hypot(obs.east, obs.north), 4, abs_tol=0.3)
    assert backend.fsm == "hover"                       # no longer circling


def test_land_returns_home_and_disarms(stack):
    _, _, tools = stack
    tools.dispatch("takeoff", {"altitude": 5})
    tools.dispatch("reposition", {"east": 6, "north": -4, "altitude": 5})
    obs = tools.dispatch("land", {})
    assert obs.status == ARRIVED
    assert not obs.armed and abs(obs.east) < 0.1 and abs(obs.north) < 0.1
    assert obs.inside == ["home_pad"]


def test_interface_refusal_is_a_failed_observation(clock):
    """AAS refuses reposition unless hovering; the agent sees FAILED, not a crash."""
    backend = FakeBackend(time_fn=clock.now)
    h = backend.reposition(1, 1, 3)
    assert h.done and not h.ok and "hover" in h.result


def test_abort_holds_position(stack):
    _, _, tools = stack
    tools.dispatch("takeoff", {"altitude": 5})
    obs = tools.dispatch("abort", {"reason": "testing"})
    assert obs.status == ABORTED and obs.reason == "testing"


# --- prompt -------------------------------------------------------------------
def test_system_prompt_has_map_and_spec(field_map):
    p = build_system_prompt(field_map, "G[0,10] !in(nofly_1)")
    assert "nofly_1 [no_fly]" in p and "G[0,10] !in(nofly_1)" in p
    assert "NOT know about the map regions" in p


# --- whole pipeline -----------------------------------------------------------
def _fly_dry_run(tmp_path, monkeypatch, mission: str) -> tuple:
    monkeypatch.setattr(cli, "DATA", tmp_path / "episodes")
    rc = cli.main(["fly", str(REPO / "missions" / mission), "--backend", "fake",
                   "--dry-run", "--no-plots"])
    episode = tmp_path / "episodes" / "ep_0001"
    return rc, episode


@pytest.mark.parametrize("mission", ["hover.yaml", "survey_then_land.yaml", "two_fields.yaml"])
def test_dry_run_missions_satisfy_their_specs(clock, tmp_path, monkeypatch, mission):
    """Each mission's dry_run_script is a correct plan: flown on the fake
    backend it must satisfy the mission's own STL spec."""
    pytest.importorskip("numpy")
    rc, episode = _fly_dry_run(tmp_path, monkeypatch, mission)
    assert rc == 0
    assert validate_episode(episode) == []
    meta = json.loads((episode / "meta.json").read_text())
    assert meta["schema_version"] == SCHEMA_VERSION and meta["dry_run"]
    stl = json.loads((episode / "stl.json").read_text())
    assert stl["satisfied"], stl


def test_runner_logs_valid_adp(clock, tmp_path, field_map):
    backend = FakeBackend(time_fn=clock.now)
    from avians.safety import Limits, SafetyArbiter
    from avians.skills import Skills
    from avians.tools import ToolLayer
    tools = ToolLayer(Skills(backend, poll_hz=20), SafetyArbiter(Limits()), field_map)
    logger = TrajectoryLogger(tmp_path)
    runner = MissionRunner(StubPilot(), tools, logger, "stub", caps=RunCaps(max_steps=10))
    result = runner.run("t", "fly", {})
    logger.close()
    assert result.terminated_by == "model_stop" and result.steps == 5
    from avians.logging.adp import load_jsonl
    adp = to_adp(load_jsonl(tmp_path / "trajectory.jsonl"))
    kinds = [c["type"] for c in adp["content"]]
    assert kinds.count("action") == kinds.count("observation") == 5


def test_runner_stops_at_step_cap(clock, tmp_path):
    backend = FakeBackend(time_fn=clock.now)
    from avians.safety import Limits, SafetyArbiter
    from avians.skills import Skills
    from avians.tools import ToolLayer
    tools = ToolLayer(Skills(backend, poll_hz=20), SafetyArbiter(Limits()))
    logger = TrajectoryLogger(tmp_path)
    script = [("get_state", {})] * 50
    result = MissionRunner(StubPilot(script), tools, logger, "stub",
                           caps=RunCaps(max_steps=3)).run("t", "loop", {})
    logger.close()
    assert result.terminated_by == "step_cap" and result.steps == 3


def test_failed_status_reaches_the_model(stack):
    """An exception inside a skill becomes a FAILED observation, never a crash."""
    _, skills, tools = stack
    tools.dispatch("takeoff", {"altitude": 5})

    def boom(*a, **k):
        raise RuntimeError("link lost")
    skills.reposition = boom
    obs = tools.dispatch("reposition", {"east": 1, "north": 1, "altitude": 5})
    assert obs.status == FAILED and "link lost" in obs.reason
