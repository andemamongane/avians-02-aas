"""Load one episode into plain arrays for plotting.

Vehicle state comes from one of two sources, best first:
  1. bag/  rosbag2 flight recorder (AAS backend): PX4 local position at
     ~100 Hz. This is also the trace the STL score is computed from.
  2. trajectory.jsonl observations: one sample per agent step. Always present,
     so fake/dry-run episodes still plot and score, just coarsely.
Commanded positions always come from the agent's tool calls: under AAS the
autopilot interface, not avians, generates the low-level setpoints.

The agent's side (tool calls, their outcomes, model thinking time) always
comes from trajectory.jsonl. Both use host wall-clock time, so they align:
t = 0 is the moment the mission was handed to the model.

Internally x is NORTH, y is EAST and z is altitude up (the figures were drawn
that way first); the tool API's (east, north, alt) is converted on load. The
NED -> up flip for bag data mirrors backends/aas.py; rosbag2_py is imported
lazily so this module stays importable without ROS.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..logging.adp import load_jsonl
from ..schema import FLIGHT_TOOLS

ARRIVAL_TOOLS = ("takeoff", "reposition", "goto_region")


@dataclass
class ToolCall:
    tool: str
    args: dict[str, Any]
    t_start: float
    t_end: float
    status: str
    reason: str
    target: tuple[float, float, float] | None      # (north, east, alt), like x/y/z
    distance_to_target: float | None


@dataclass
class FlightData:
    episode: str
    mission: str
    meta: dict[str, Any]
    source: str                                   # "rosbag2" | "agent observations"
    # vehicle state
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    speed: np.ndarray
    # commanded position (NaN where nothing was commanded)
    sp_t: np.ndarray
    sp_x: np.ndarray
    sp_y: np.ndarray
    sp_z: np.ndarray
    # battery
    bat_t: np.ndarray
    bat_pct: np.ndarray
    # agent side
    calls: list[ToolCall] = field(default_factory=list)
    thinking: list[tuple[float, float]] = field(default_factory=list)   # model latency spans
    final_text: str = ""
    # mission map (episode's map.yaml) and STL score (stl.json), when present
    map: Any = None
    stl: dict[str, Any] | None = None

    @property
    def dense(self) -> bool:
        return self.source == "rosbag2"


# ---------------------------------------------------------------- loading
def load_episode(episode_dir: Path) -> FlightData:
    d = Path(episode_dir)
    records = load_jsonl(d / "trajectory.jsonl")
    meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}

    mission = next((r for r in records if r["kind"] == "mission"), None)
    t0 = mission["t_wall"] if mission else records[0]["t_wall"]
    calls = _tool_calls(records, t0)
    thinking, final_text = _thinking_spans(records, t0)

    state, setpoint, battery = _from_observations(records, calls, t0)
    source = "agent observations"
    origin = meta.get("frame_origin_ned") or (0.0, 0.0, 0.0)
    bag = _read_bag(d / "bag", t0, origin) if (d / "bag" / "metadata.yaml").exists() else None
    if bag is not None:
        state, bag_battery = bag
        if len(bag_battery[0]):
            battery = bag_battery
        source = "rosbag2"

    t, x, y, z, speed = state
    sp_t, sp_x, sp_y, sp_z = setpoint
    bat_t, bat_pct = battery
    return FlightData(
        episode=meta.get("episode", d.name),
        mission=meta.get("mission", mission["name"] if mission else "unknown"),
        meta=meta, source=source,
        t=t, x=x, y=y, z=z, speed=speed,
        sp_t=sp_t, sp_x=sp_x, sp_y=sp_y, sp_z=sp_z,
        bat_t=bat_t, bat_pct=bat_pct,
        calls=calls, thinking=thinking, final_text=final_text,
        map=_load_map(d / "map.yaml"),
        stl=json.loads((d / "stl.json").read_text()) if (d / "stl.json").exists() else None,
    )


def _load_map(path: Path):
    if not path.exists():
        return None
    from ..maps import Map
    return Map.load(path)


def _tool_calls(records: list[dict], t0: float) -> list[ToolCall]:
    ends = {r["tool_use_id"]: r for r in records if r["kind"] == "observation"}
    calls = []
    for r in records:
        if r["kind"] != "action":
            continue
        obs_rec = ends.get(r["tool_use_id"])
        obs = obs_rec["observation"] if obs_rec else {}
        target = obs.get("target")                     # (east, north, alt)
        a = r.get("args", {})
        if target is None and r["tool"] == "reposition" and "east" in a:
            target = (a.get("east"), a.get("north"), a.get("altitude"))
        calls.append(ToolCall(
            tool=r["tool"], args=a,
            t_start=r["t_wall"] - t0,
            t_end=(obs_rec["t_wall"] if obs_rec else r["t_wall"]) - t0,
            status=obs.get("status", "missing"), reason=obs.get("reason", ""),
            target=(target[1], target[0], target[2]) if target else None,
            distance_to_target=obs.get("distance_to_target"),
        ))
    return calls


def _thinking_spans(records: list[dict], t0: float) -> tuple[list[tuple[float, float]], str]:
    spans, start, final_text = [], None, ""
    for r in records:
        if r["kind"] == "request":
            start = r["t_wall"] - t0
        elif r["kind"] == "response" and start is not None:
            spans.append((start, r["t_wall"] - t0))
            start = None
            texts = [b.get("text", "") for b in r.get("content", []) if b.get("type") == "text"]
            if any(texts):
                final_text = "\n".join(t for t in texts if t)
    return spans, final_text


def _read_bag(bag_dir: Path, t0: float, origin=(0.0, 0.0, 0.0)):
    """Decode the bag, or return None if ROS isn't importable here. Positions
    are made relative to the episode's frame origin (the bag holds raw NED)."""
    try:
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
    except ImportError:
        return None

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id=""),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    types = {t.name: get_message(t.type) for t in reader.get_all_topics_and_types()}

    pos, bat = [], []
    while reader.has_next():
        topic, raw, t_ns = reader.read_next()
        t = t_ns / 1e9 - t0
        if "vehicle_local_position" in topic:
            m = deserialize_message(raw, types[topic])
            ox, oy, oz = origin
            pos.append((t, m.x - ox, m.y - oy, -(m.z - oz),
                        math.sqrt(m.vx ** 2 + m.vy ** 2 + m.vz ** 2)))
        elif "battery_status" in topic:
            m = deserialize_message(raw, types[topic])
            if m.remaining >= 0:
                bat.append((t, m.remaining * 100.0))

    if not pos:
        return None
    p = np.array(pos, dtype=float)
    b = np.array(bat, dtype=float).reshape(-1, 2)
    return (p[:, 0], p[:, 1], p[:, 2], p[:, 3], p[:, 4]), (b[:, 0], b[:, 1])


def _from_observations(records: list[dict], calls: list[ToolCall], t0: float):
    obs = [r for r in records if r["kind"] == "observation"]
    rows = [(0.0, 0.0, 0.0, 0.0, 0.0)] + [
        (r["t_wall"] - t0, o["north"], o["east"], o["alt"],
         math.sqrt(o["v_east"] ** 2 + o["v_north"] ** 2 + o["v_up"] ** 2))
        for r in obs for o in [r["observation"]]
    ]
    p = np.array(rows, dtype=float)

    # commanded = the target of each flight action, held until the next one
    sp = []
    for c in calls:
        if c.tool in ("takeoff", "reposition", "goto_region", "hover") and c.target and c.status not in (
                "rejected", "blocked"):
            sp.append((c.t_start, *c.target))
            sp.append((c.t_end, *c.target))
        elif c.tool == "land":
            sp.append((c.t_start, np.nan, np.nan, np.nan))
    s = np.array(sp, dtype=float).reshape(-1, 4)

    b = np.array([(r["t_wall"] - t0, r["observation"]["battery_pct"]) for r in obs
                  if r["observation"].get("battery_pct") is not None], dtype=float).reshape(-1, 2)
    return (p[:, 0], p[:, 1], p[:, 2], p[:, 3], p[:, 4]), \
           (s[:, 0], s[:, 1], s[:, 2], s[:, 3]), (b[:, 0], b[:, 1])


# ---------------------------------------------------------------- metrics
def compute_metrics(f: FlightData) -> dict[str, Any]:
    """Headline numbers for the slide and for summary.json."""
    by_status: dict[str, int] = {}
    for c in f.calls:
        by_status[c.status] = by_status.get(c.status, 0) + 1

    steps = np.sqrt(np.diff(f.x) ** 2 + np.diff(f.y) ** 2 + np.diff(f.z) ** 2)
    airborne = f.z > 0.3
    air_t = f.t[airborne]

    land = next((c for c in reversed(f.calls) if c.tool == "land"), None)
    landed = bool(land and land.status == "arrived")
    flight_calls = [c for c in f.calls if c.tool in FLIGHT_TOOLS]
    safety_rejections = [c for c in f.calls if c.status in ("rejected", "blocked")]

    arrivals = [
        {"tool": c.tool,
         "target": [round(v, 2) for v in (c.target[1], c.target[0], c.target[2])],
         "error_m": round(c.distance_to_target, 2)}
        for c in f.calls
        if c.tool in ARRIVAL_TOOLS and c.status == "arrived"
        and c.target and c.distance_to_target is not None
    ]

    holds = []
    for c in f.calls:
        if c.tool == "hover" and c.status == "arrived":
            m = (f.t >= c.t_start) & (f.t <= c.t_end)
            if m.sum() >= 2:
                cx, cy, cz = f.x[m].mean(), f.y[m].mean(), f.z[m].mean()
                drift = np.sqrt((f.x[m] - cx) ** 2 + (f.y[m] - cy) ** 2 + (f.z[m] - cz) ** 2)
                holds.append({"duration_s": round(c.t_end - c.t_start, 1),
                              "max_drift_m": round(float(drift.max()), 3),
                              "rms_drift_m": round(float(np.sqrt((drift ** 2).mean())), 3)})

    usage = f.meta.get("usage", {})
    checks = {
        "ended by the model (not a cap)": f.meta.get("terminated_by") == "model_stop",
        "landed": landed,
        "no failed or timed-out actions": not any(
            c.status in ("failed", "timeout") for c in flight_calls),
    }
    if f.stl is not None:
        checks["STL spec satisfied"] = bool(f.stl["satisfied"])
    return {
        "episode": f.episode,
        "mission": f.mission,
        "model": f.meta.get("model"),
        "backend": f.meta.get("backend"),
        "data_source": f.source,
        "terminated_by": f.meta.get("terminated_by"),
        "mission_duration_s": round(float(f.meta.get("duration_s", f.t[-1])), 1),
        "airborne_s": round(float(air_t[-1] - air_t[0]), 1) if len(air_t) > 1 else 0.0,
        "distance_flown_m": round(float(np.nansum(steps)), 1),
        "max_altitude_m": round(float(np.nanmax(f.z)), 2),
        "max_speed_ms": round(float(np.nanmax(f.speed)), 2),
        "landing_offset_m": round(float(math.hypot(f.x[-1], f.y[-1])), 2),
        "tool_calls": len(f.calls),
        "tool_calls_by_status": by_status,
        "safety_rejections": [{"tool": c.tool, "args": c.args, "reason": c.reason}
                              for c in safety_rejections],
        "arrivals": arrivals,
        "holds": holds,
        "model_time_s": round(sum(b - a for a, b in f.thinking), 1),
        "tokens": {"input": usage.get("input_tokens", 0),
                   "output": usage.get("output_tokens", 0)},
        "stl": f.stl,
        "checks": checks,
        "success": all(checks.values()),
    }
