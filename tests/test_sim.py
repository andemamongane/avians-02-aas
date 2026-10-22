"""Environment, geo exports, and the simulator controller's commands (against a
recording stand-in for docker: nothing here needs Gazebo or PX4)."""
from __future__ import annotations

import math
import statistics
import subprocess
import xml.etree.ElementTree as ET

import pytest
import yaml

from avians import geo
from avians.environment import Environment, EnvironmentError, Wind, envelope_params
from avians.maps import Map
from avians.safety import Limits, resolve
from avians.sim import SimControl, SimError

from conftest import REPO

AGGRESSIVE = Limits.from_dict(resolve(yaml.safe_load((REPO / "config/limits.yaml").read_text()),
                                      "aggressive"))
WORLD = REPO / "external/aerial-autonomy-stack/simulation/simulation_resources/simulation_worlds"


# --- environment ----------------------------------------------------------------
def test_environment_parses_and_summarizes():
    env = Environment.from_mission({"environment": {
        "world": "swiss_town",
        "wind": {"east": 3, "north": -1, "gust_std": 1.5, "gust_tau": 4, "seed": 7},
        "failures": [{"at": 30, "unit": "gps", "duration": 8}],
        "battery": {"drain_s": 600, "min_pct": 15},
        "px4_params": {"COM_DISARM_LAND": 1}}})
    assert env.world == "swiss_town" and env.failures[0].type == "off"
    assert env.params() == {"SIM_BAT_DRAIN": 600.0, "SIM_BAT_MIN_PCT": 15.0,
                            "SYS_FAILURE_EN": 1.0, "COM_DISARM_LAND": 1.0}
    assert env.summary()["wind"]["seed"] == 7
    assert Environment.from_mission({}).wind.calm


@pytest.mark.parametrize("raw,match", [
    ({"weather": 1}, "unknown keys"),
    ({"wind": {"speed": 3}}, "unknown keys"),
    ({"wind": {"gust_std": -1}}, "gust_std"),
    ({"failures": [{"at": 3, "unit": "flux"}]}, "unit"),
])
def test_environment_rejects_bad_blocks(raw, match):
    with pytest.raises(EnvironmentError, match=match):
        Environment.from_mission({"environment": raw})


def test_gusts_are_reproducible_and_have_the_requested_spread():
    w = Wind(east=3.0, north=0.0, gust_std=1.5, gust_tau=4.0, seed=7)
    a, b = w.schedule(2000), w.schedule(2000)
    assert a == b and a != Wind(east=3.0, gust_std=1.5, seed=8).schedule(2000)
    east = [e for _, e, _ in a]
    assert statistics.mean(east) == pytest.approx(3.0, abs=0.3)
    assert statistics.pstdev(east) == pytest.approx(1.5, rel=0.2)
    # correlated over gust_tau, not white noise: neighbours 0.5 s apart are close
    d = [abs(x - y) for x, y in zip(east, east[1:])]
    assert statistics.mean(d) < 0.6
    assert Wind().schedule(10) == [(k * 0.5, 0.0, 0.0) for k in range(21)]


def test_envelope_becomes_px4_parameters():
    p = envelope_params(AGGRESSIVE)
    assert p["MPC_XY_VEL_MAX"] == 15 and p["MPC_ACC_HOR_MAX"] == 7
    assert p["MPC_TILTMAX_AIR"] == 45 and p["MPC_JERK_MAX"] == 20
    assert "MPC_TILTMAX_AIR" not in envelope_params(Limits())       # standard: PX4's own


# --- geo ---------------------------------------------------------------------------
def test_world_origin_and_latlon():
    lat, lon, elev = geo.world_origin(WORLD / "impalpable_greyness.sdf")
    assert lat == pytest.approx(45.5477, abs=1e-3) and elev == 179
    la, lo = geo.to_latlon(100.0, 100.0, (lat, lon, elev))
    assert (la - lat) * 111_320 == pytest.approx(100, rel=1e-2)
    assert (lo - lon) * 111_320 * math.cos(math.radians(lat)) == pytest.approx(100, rel=1e-2)


def test_kml_and_region_models_are_valid(field_map):
    k = ET.fromstring(geo.kml(field_map, geo.PX4_SIH_HOME))
    ns = {"k": "http://www.opengis.net/kml/2.2"}
    assert len(k.findall(".//k:Placemark", ns)) == len(field_map.regions)
    models = geo.region_models(field_map, anchor=(2.0, -1.0))
    assert set(models) == {f"avians_region_{r}" for r in field_map.regions}
    nofly = ET.fromstring(models["avians_region_nofly_1"].split("?>", 1)[1])
    assert nofly.find(".//pose").text.startswith("2.000 -1.000")
    assert nofly.find(".//collision") is None                      # visual only
    course = Map.load(REPO / "maps/course_01.yaml")
    wall = ET.fromstring(geo.region_models(course)["avians_region_wall_1"].split("?>", 1)[1])
    assert wall.find(".//collision") is not None                   # solid: a real obstacle
    assert float(wall.find(".//height").text) == 40
    ET.fromstring(geo.chase_camera_model().split("?>", 1)[1])


# --- sim control ---------------------------------------------------------------------
class Docker:
    """Records commands; answers the few whose output matters."""

    def __init__(self, world="impalpable_greyness"):
        self.calls: list[list[str]] = []
        self.world = world

    def run(self, cmd, **kw):
        self.calls.append(cmd)
        text = " ".join(cmd)
        out = ""
        if "gz topic -l" in text:
            out = f"/clock\n/world/{self.world}/pose/info\n/world/{self.world}/wind/\n"
        elif "avians_gz.py pose" in text:
            out = "1.5 -2.0 0.5 0.0\n"
        elif "*.ulg" in text:
            out = "1700000000.0 /logs/2026-10-22/12_00_00.ulg\n"
        return subprocess.CompletedProcess(cmd, 0, out, "")

    def popen(self, cmd, **kw):
        self.calls.append(cmd)

        class P:
            stdout = None
            def wait(self, timeout=None): return 0
            def kill(self): pass
        return P()

    def joined(self) -> list[str]:
        return [" ".join(c) for c in self.calls]


def test_remote_prepare_start_stop(tmp_path, field_map):
    d = Docker()
    sim = SimControl({}, tmp_path, "remote", run=d.run, popen=d.popen)
    env = Environment.from_mission({"environment": {
        "wind": {"east": 4, "gust_std": 1, "seed": 1},
        "failures": [{"at": 1000, "unit": "gps", "duration": 5}]}})
    sim.prepare(env, AGGRESSIVE, field_map)
    sim.start(horizon_s=60)
    sim.stop()
    calls = d.joined()
    assert any("px4-param --instance 0 set MPC_XY_VEL_MAX 15" in c for c in calls)
    assert any("set SYS_FAILURE_EN 1" in c for c in calls)
    spawns = [c for c in calls if "avians_gz.py spawn" in c]
    assert len(spawns) == len(field_map.regions)
    assert (tmp_path / "sim" / "avians_region_nofly_1.sdf").read_text().count("1.500 -2.000") == 1
    assert any("avians_gz.py wind" in c for c in calls)
    assert any("avians_gz.py record" in c for c in calls)
    assert any("kill -TERM $(cat /tmp/avians/record.pid)" in c for c in calls)
    assert sum("avians_gz.py remove" in c for c in calls) == len(field_map.regions)
    assert any("px4-failure --instance 0 gps ok" in c for c in calls)   # cleared at the end
    assert any(c.endswith("px4.ulg") and "docker cp" in c for c in calls)
    assert (tmp_path / "wind.json").exists()
    assert sim.summary()["world"] == "impalpable_greyness"


def test_wrong_world_stops_before_flying(tmp_path):
    sim = SimControl({}, tmp_path, "remote", run=Docker("swiss_town").run)
    with pytest.raises(SimError, match="WORLD=impalpable_greyness"):
        sim.prepare(Environment(world="impalpable_greyness"), AGGRESSIVE, None)


def test_local_profile_sets_params_and_notes_what_sih_cannot_do(tmp_path):
    d = Docker()
    sim = SimControl({}, tmp_path, "local", run=d.run, popen=d.popen)
    sim.prepare(Environment(world="swiss_town", wind=Wind(east=3)), AGGRESSIVE, None)
    calls = d.joined()
    assert calls[0].startswith("/opt/px4/bin/px4-param set ")
    assert not any("docker" in c for c in calls)
    assert any("no world" in n for n in sim.notes) and any("no wind" in n for n in sim.notes)


def test_helper_script_compiles():
    import py_compile
    from avians.sim import HELPER
    py_compile.compile(str(HELPER), doraise=True)
