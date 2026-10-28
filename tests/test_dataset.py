"""The fine-tuning export: one example per model call, the selection rule,
and a manifest that says what was left out and why."""
from __future__ import annotations

import json

import pytest

from avians import cli
from avians.dataset import Selection, export

from conftest import REPO


@pytest.fixture
def episodes(clock, tmp_path, monkeypatch):
    """Two scripted plan-ahead flights (one through the power lines) and one
    step-by-step flight, on the fake backend."""
    pytest.importorskip("matplotlib")
    import yaml
    monkeypatch.setattr(cli, "DATA", tmp_path / "episodes")
    fly = lambda path, *a: cli.main(["fly", str(path), "--backend", "fake", "--dry-run", *a])
    assert fly(REPO / "missions/survey_then_land.yaml") == 0
    bad = yaml.safe_load((REPO / "missions/survey_then_land.yaml").read_text())
    bad["dry_run_plan"] = [["takeoff", {"altitude": 5}],
                           ["fly_plan", {"segments": [{"type": "waypoint", "east": 15,
                                                       "north": 25, "altitude": 5}]}]]
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / bad["prompt_file"]).write_text("fly")
    (tmp_path / "bad.yaml").write_text(yaml.safe_dump(bad))
    assert fly(tmp_path / "bad.yaml") == 0
    assert fly(REPO / "missions/hover.yaml", "--control", "step") == 0
    return tmp_path / "episodes"


def mark_real(ep_dir):
    """Pretend an episode was a real (non-scripted) flight, for the selection rule."""
    p = ep_dir / "meta.json"
    m = json.loads(p.read_text())
    m.update(dry_run=False, pilot="claude", model="claude-sonnet-5-5")
    p.write_text(json.dumps(m))


def test_one_example_per_model_call_with_its_outcome(episodes, tmp_path):
    for ep in sorted(episodes.glob("ep_*")):
        mark_real(ep)
    man = export(episodes, tmp_path / "out.jsonl", Selection(include_fake=True))
    rows = [json.loads(l) for l in open(tmp_path / "out.jsonl")]
    assert [e["episode"] for e in man["included"]] == ["ep_0001"]
    assert man["excluded"] == {"ep_0002": "spec not satisfied",
                               "ep_0003": "not plan-ahead decisions"}
    assert len(rows) == man["examples"] > 5
    first = rows[0]
    assert len(first["messages"]) == 1 and "MISSION" in first["messages"][0]["content"]
    assert first["target"][0]["type"] == "tool_use" and first["target"][0]["name"] == "takeoff"
    assert first["outcome"]["satisfied"] and first["setting"]["clock"] == "lockstep"
    assert {t["name"] for t in first["tools"]} >= {"keep", "fly_plan"}
    assert [r["turn"] for r in rows] == list(range(len(rows)))


def test_the_defaults_leave_out_what_a_student_should_not_copy(episodes, tmp_path):
    man = export(episodes, tmp_path / "out.jsonl")
    assert man["examples"] == 0
    assert set(man["excluded"].values()) == {"scripted dry run"}
    for ep in sorted(episodes.glob("ep_*")):
        mark_real(ep)
    man = export(episodes, tmp_path / "out.jsonl")
    assert man["excluded"]["ep_0001"] == "fake backend (noise-free physics)"
    loose = export(episodes, tmp_path / "out.jsonl",
                   Selection(include_fake=True, require_satisfied=False))
    assert {e["episode"] for e in loose["included"]} == {"ep_0001", "ep_0002"}
    assert json.loads((tmp_path / "out.manifest.json").read_text())["examples"] == loose["examples"]
