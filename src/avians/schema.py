"""The contract. Every other module imports its definitions from here.

FRAME AND UNITS (the rule that prevents the most common bug class):
    The tool API uses the aerial-autonomy-stack's frame: metres EAST and NORTH
    of the launch point, and ALTITUDE above it (always positive). That is what
    AAS's set_reposition and orbit_action take, so tool arguments pass through
    unchanged. PX4's local frame is NED (x north, y east, z DOWN); that
    conversion happens ONLY inside the backends.

Tools mirror AAS one-to-one where AAS has a command (takeoff, reposition,
orbit, land). The rest (hover, abort, get_state, goto_region, check_path) are
thin helpers built on those commands and on the mission map.

Bump SCHEMA_VERSION whenever a tool signature or observation field changes, and
re-record rather than mixing versions in one dataset.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

SCHEMA_VERSION = 3

# --- status codes ----------------------------------------------------------
# Failures are OBSERVATIONS, never exceptions: the model must be able to see a
# rejection and recover from it, and that recovery becomes training data.
ARRIVED = "arrived"      # command completed, vehicle at target
OK = "ok"                # informational call succeeded (get_state, check_path)
TIMEOUT = "timeout"      # ran out of time before arriving
REJECTED = "rejected"    # safety arbiter or argument check refused the command
ABORTED = "aborted"      # stopped (by abort, or episode teardown)
BLOCKED = "blocked"      # vehicle state forbids it (e.g. reposition before takeoff)
FAILED = "failed"        # the autopilot interface refused it, or transport error

ALL_STATUSES = (ARRIVED, OK, TIMEOUT, REJECTED, ABORTED, BLOCKED, FAILED)


class ToolArgError(ValueError):
    """Raised when arguments fail validation; caught and returned as REJECTED."""


# --- observation -----------------------------------------------------------
@dataclass
class Observation:
    """What the agent sees. This is the model's entire view of the world.

    Only include information a REAL drone would have. Never simulator ground
    truth: training on it produces a model that fails silently on hardware.
    Map regions are fine: an operator has the map before the flight.
    """
    status: str
    reason: str = ""

    # pose / motion (metres east/north of launch, altitude up; m/s)
    east: float = 0.0
    north: float = 0.0
    alt: float = 0.0
    v_east: float = 0.0
    v_north: float = 0.0
    v_up: float = 0.0
    heading_deg: float = 0.0

    # vehicle
    mode: str = "unknown"
    armed: bool = False
    battery_pct: float | None = None       # None when the autopilot doesn't report it

    # mission context
    target: tuple[float, float, float] | None = None     # (east, north, alt)
    distance_to_target: float | None = None
    action_elapsed_s: float = 0.0
    mission_elapsed_s: float = 0.0

    # interval summary produced by skills.py (what happened while flying)
    path_length_m: float | None = None
    max_speed_ms: float | None = None

    # map context: which regions the drone is in, and how far from each
    # (signed: + inside, - outside, metres to the boundary)
    inside: list[str] = field(default_factory=list)
    region_distance_m: dict[str, float] = field(default_factory=dict)

    # tool-specific result text (e.g. what check_path found, a plan preview)
    info: str = ""

    # plan-ahead control: what the plan is doing, what happened since the
    # pilot's last call, and the accepted plan's path for plots (never sent
    # to the model: [seconds into the plan, east, north, alt] every ~2 m)
    plan_status: str = ""
    events: list[str] = field(default_factory=list)
    plan_path: list[list[float]] = field(default_factory=list)
    plan_age_s: float = 0.0                # how long that plan had been flying

    # perception: always empty in this repo. Kept so perception episodes later
    # stay schema-compatible.
    detections: list[dict[str, Any]] = field(default_factory=list)

    # clocks: t_sim is PX4's timestamp, t_wall is host wall-clock. Log both.
    t_sim: float | None = None
    t_wall: float | None = None

    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_text(self) -> str:
        """The exact string handed to the model. Keep it terse: every episode
        re-sends the whole history, so verbosity costs tokens quadratically."""
        lines = [f"status: {self.status}" + (f" ({self.reason})" if self.reason else "")]
        if self.events:
            lines.append("events since your last call: " + "; ".join(self.events))
        if self.plan_status:
            lines.append(f"plan: {self.plan_status}")
        if self.info:
            lines.append(self.info)
        lines.append(
            f"position: east={self.east:.2f} north={self.north:.2f} alt={self.alt:.2f} m"
        )
        lines.append(
            f"velocity: {self._speed():.2f} m/s  heading: {self.heading_deg:.0f} deg"
        )
        state = f"mode: {self.mode}  armed: {self.armed}"
        if self.battery_pct is not None:
            state += f"  battery: {self.battery_pct:.0f}%"
        lines.append(state)
        if self.target is not None:
            te, tn, ta = self.target
            d = self.distance_to_target
            lines.append(
                f"target: east={te:.1f} north={tn:.1f} alt={ta:.1f}"
                + (f"  distance: {d:.2f} m" if d is not None else "")
            )
        if self.path_length_m is not None:
            lines.append(
                f"last action: {self.action_elapsed_s:.1f} s, "
                f"travelled {self.path_length_m:.1f} m, "
                f"peak {self.max_speed_ms:.1f} m/s"
            )
        if self.region_distance_m:
            lines.append("inside regions: " + (", ".join(self.inside) or "none"))
            lines.append("region boundaries (+ inside, - outside): " + ", ".join(
                f"{rid} {d:+.1f} m" for rid, d in self.region_distance_m.items()))
        lines.append(f"mission time: {self.mission_elapsed_s:.0f} s")
        for d in self.detections:
            lines.append(f"detected: {d}")
        return "\n".join(lines)

    def _speed(self) -> float:
        return (self.v_east ** 2 + self.v_north ** 2 + self.v_up ** 2) ** 0.5


# --- tool definitions ------------------------------------------------------
# Sent verbatim to the Messages API. Descriptions are part of the prompt, so
# they are written for the model, not for us.
_EAST = {"type": "number", "description": "Metres east of the launch point (negative = west)."}
_NORTH = {"type": "number", "description": "Metres north of the launch point (negative = south)."}
_ALT = {"type": "number", "description": "Altitude in metres above the launch point (positive)."}
_SPEED = {"type": "number", "description": "Optional cruise speed in m/s for this leg."}

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "get_state",
        "description": (
            "Read the drone's current state without commanding anything: position, "
            "velocity, mode, which map regions it is in, and the mission time."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "takeoff",
        "description": (
            "Arm the drone and climb vertically to the given altitude above the "
            "launch point. Must be called first. Returns when the altitude is "
            "reached or the attempt times out."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"altitude": _ALT},
            "required": ["altitude"],
        },
    },
    {
        "name": "reposition",
        "description": (
            "Fly in a straight line to a point and hold there. Returns when the "
            "drone arrives or the attempt times out. The straight path may cross "
            "map regions; use check_path first if that matters."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"east": _EAST, "north": _NORTH, "altitude": _ALT, "speed": _SPEED},
            "required": ["east", "north", "altitude"],
        },
    },
    {
        "name": "goto_region",
        "description": (
            "Fly in a straight line to a point well inside a map region and hold "
            "there. Use this instead of computing a point inside a polygon yourself. "
            "Returns when the drone arrives or the attempt times out."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "region": {"type": "string", "description": "Region id from the mission map."},
                "altitude": _ALT,
                "speed": _SPEED,
            },
            "required": ["region", "altitude"],
        },
    },
    {
        "name": "orbit",
        "description": (
            "Circle a centre point at a fixed radius and altitude for a number of "
            "seconds, then hold position wherever the circle left the drone. The "
            "orbit speed is fixed by the autopilot (about 5 m/s)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "east": _EAST, "north": _NORTH, "altitude": _ALT,
                "radius": {"type": "number", "description": "Circle radius in metres."},
                "duration": {"type": "number", "description": "Seconds to keep circling."},
            },
            "required": ["east", "north", "altitude", "radius", "duration"],
        },
    },
    {
        "name": "hover",
        "description": "Hold the current position for a number of seconds.",
        "input_schema": {
            "type": "object",
            "properties": {"duration": {"type": "number", "description": "Seconds to hold."}},
            "required": ["duration"],
        },
    },
    {
        "name": "check_path",
        "description": (
            "Without flying, list the map regions that a straight flight from the "
            "current position to (east, north) would cross."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"east": _EAST, "north": _NORTH},
            "required": ["east", "north"],
        },
    },
    {
        "name": "land",
        "description": (
            "Fly straight back to the launch point at the current altitude, land "
            "there and disarm. The drone always lands at the launch point. Call "
            "this to end the mission."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "abort",
        "description": (
            "Stop and hold the current position. Use this if something is wrong or "
            "you are unsure how to proceed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string", "description": "Why you are aborting."}},
            "required": [],
        },
    },
]

TOOL_NAMES = tuple(t["name"] for t in TOOL_SPECS)

# --- plan-ahead tools ---------------------------------------------------------
# The pilot commits to a whole flight plan; the drone flies it continuously
# while the pilot keeps thinking. Calls about the plan return at once.
_SEGMENT = {
    "type": "object",
    "description": (
        "One part of the plan. type is one of: "
        "waypoint (east, north, altitude: fly THROUGH this point without stopping), "
        "region (region, altitude: fly through a point well inside a map region), "
        "orbit (east, north, altitude, radius, turns, direction cw|ccw: circle a centre), "
        "survey (region, altitude, spacing: lawnmower rows covering a region), "
        "hold (duration: stop at the previous point and hover that many seconds), "
        "land (fly straight to the launch point at the current altitude and land; must be last). "
        "speed (m/s) optionally caps the leg arriving at the segment; otherwise "
        "the drone flies as fast as the flight envelope and the turns allow."
    ),
    "properties": {
        "type": {"type": "string",
                 "enum": ["waypoint", "region", "orbit", "survey", "hold", "land"]},
        "east": _EAST, "north": _NORTH, "altitude": _ALT, "speed": _SPEED,
        "region": {"type": "string", "description": "Region id from the mission map."},
        "radius": {"type": "number", "description": "Orbit radius in metres."},
        "turns": {"type": "number", "description": "Orbit turns (default 1)."},
        "direction": {"type": "string", "enum": ["cw", "ccw"]},
        "spacing": {"type": "number", "description": "Survey row spacing in metres (default 5)."},
        "duration": {"type": "number", "description": "Hold time in seconds."},
    },
    "required": ["type"],
}

PLAN_TOOL_SPECS: list[dict[str, Any]] = [
    TOOL_SPECS[0],                                   # get_state
    {
        "name": "takeoff",
        "description": (
            "Arm the drone and climb vertically to the given altitude above the "
            "launch point. Returns at once; the climb takes a few seconds. Send "
            "fly_plan straight away: it starts the moment the climb completes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"altitude": _ALT},
            "required": ["altitude"],
        },
    },
    {
        "name": "fly_plan",
        "description": (
            "Commit to a flight plan: an ordered list of segments that the drone "
            "flies continuously, rounding corners and slowing only where turns "
            "require it. Returns at once with a preview (length, duration, when "
            "each segment completes, which map regions the path passes through); "
            "the drone is already flying it. mode 'replace' (default) abandons "
            "the current plan and starts this one from where the drone is, "
            "keeping its momentum; 'append' adds these segments after the rest "
            "of the current plan."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "segments": {"type": "array", "items": _SEGMENT,
                             "description": "The plan, in order."},
                "mode": {"type": "string", "enum": ["replace", "append"]},
            },
            "required": ["segments"],
        },
    },
    {
        "name": "wait",
        "description": (
            "Let time pass while the drone flies, without commanding anything. "
            "Returns at the first event (a segment completed, a region entered or "
            "left), when the plan completes, or when the drone has landed, "
            "depending on 'until' (default 'event'), and after 'seconds' at most."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "seconds": {"type": "number", "description": "Longest wait in seconds."},
                "until": {"type": "string", "enum": ["event", "plan_complete", "landed"]},
            },
            "required": ["seconds"],
        },
    },
    {
        "name": "hold",
        "description": (
            "Abandon the plan: brake along the current direction and hover where "
            "the drone comes to a stop."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "land",
        "description": (
            "Abandon the plan, fly straight to the launch point at the current "
            "altitude and land. If that straight line must avoid a region, plan "
            "the route home yourself with fly_plan, ending with a land segment."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    next(t for t in TOOL_SPECS if t["name"] == "check_path"),
]
FLIGHT_TOOLS = ("takeoff", "reposition", "goto_region", "orbit", "hover", "land",
                "fly_plan", "hold", "wait")


def validate_tool_args(name: str, args: dict[str, Any],
                       specs: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Type/range check arguments before anything touches the vehicle.

    Returns normalized args. Raises ToolArgError with a message written for the
    model to read and act on. Region ids are checked against the map in tools.py;
    plan segments in trajectory.expand.
    """
    specs = specs or TOOL_SPECS
    names = [t["name"] for t in specs]
    if name not in names:
        raise ToolArgError(f"unknown tool '{name}'; available: {', '.join(names)}")

    spec = next(t for t in specs if t["name"] == name)
    props = spec["input_schema"]["properties"]
    required = spec["input_schema"].get("required", [])

    for key in required:
        if key not in args or args[key] is None:
            raise ToolArgError(f"missing required argument '{key}' for {name}")

    out: dict[str, Any] = {}
    for key, value in args.items():
        if key not in props:
            raise ToolArgError(f"unexpected argument '{key}' for {name}")
        if value is None:
            continue                                 # optional argument left out
        expected = props[key]["type"]
        if expected == "number":
            try:
                out[key] = float(value)
            except (TypeError, ValueError):
                raise ToolArgError(f"argument '{key}' must be a number, got {value!r}")
        elif expected == "string":
            out[key] = str(value)
            enum = props[key].get("enum")
            if enum and out[key] not in enum:
                raise ToolArgError(f"argument '{key}' must be one of {', '.join(enum)}")
        elif expected == "array" and not isinstance(value, list):
            raise ToolArgError(f"argument '{key}' must be a list")
        else:
            out[key] = value

    if "altitude" in out and out["altitude"] <= 0:
        raise ToolArgError("altitude must be positive: it is metres above the launch point")
    for key in ("duration", "radius", "speed", "seconds"):
        if key in out and out[key] <= 0:
            raise ToolArgError(f"{key} must be positive")

    return out
