"""The transport interface. Deliberately tiny.

A backend sends commands to an autopilot interface and reports vehicle state
back. It contains NO policy: no arrival detection, no timeouts, no limits.
Those live in skills.py and safety.py so that every backend behaves identically.

The commands mirror the aerial-autonomy-stack's autopilot_interface:
takeoff_action, set_reposition (+ set_speed), orbit_action, land_action.
`hold` is a reposition to the current position: AAS has no hold command, and
cancelling an AAS action makes it climb to its abort altitude (100 m by default).

All coordinates crossing this interface are metres east/north of the launch
point and altitude above it (see schema.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass
class VehicleState:
    east: float = 0.0
    north: float = 0.0
    alt: float = 0.0            # above launch, positive up
    v_east: float = 0.0
    v_north: float = 0.0
    v_up: float = 0.0
    heading_deg: float = 0.0
    mode: str = "unknown"
    armed: bool = False
    battery_pct: float | None = None
    t_sim: float | None = None   # PX4 timestamp, seconds
    ok: bool = True              # False when no telemetry has arrived yet

    @property
    def speed(self) -> float:
        return (self.v_east ** 2 + self.v_north ** 2 + self.v_up ** 2) ** 0.5

    @property
    def pos(self) -> tuple[float, float, float]:
        return (self.east, self.north, self.alt)


@dataclass
class ActionHandle:
    """Handle to a command sent to the autopilot interface. skills.py polls it.

    done:  the interface has finished with the command (action result arrived,
           or the service replied). For reposition/orbit this means "accepted",
           not "arrived": skills.py decides arrival from the vehicle state.
    ok:    False when the interface rejected or failed the command.
    """
    kind: str
    target: tuple[float, float, float] | None = None
    done: bool = False
    ok: bool = True
    result: str = ""
    feedback: dict = field(default_factory=dict)

    def fail(self, message: str) -> "ActionHandle":
        self.done, self.ok, self.result = True, False, message
        return self

    def succeed(self, message: str = "") -> "ActionHandle":
        self.done, self.ok, self.result = True, True, message
        return self


@runtime_checkable
class Backend(Protocol):
    name: str

    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def get_state(self) -> VehicleState: ...
    def now(self) -> float: ...
    def takeoff(self, altitude: float) -> ActionHandle: ...
    def reposition(self, east: float, north: float, alt: float,
                   speed: float | None = None) -> ActionHandle: ...
    def orbit(self, east: float, north: float, alt: float, radius: float) -> ActionHandle: ...
    def hold(self) -> ActionHandle: ...
    def land(self, alt: float) -> ActionHandle: ...
