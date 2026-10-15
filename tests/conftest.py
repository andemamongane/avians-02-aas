"""Shared fixtures. Everything here runs offline: no ROS, no Docker, no API key."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from avians.backends.fake import FakeBackend
from avians.maps import Map
from avians.safety import Limits, SafetyArbiter
from avians.skills import Skills
from avians.tools import ToolLayer

REPO = Path(__file__).resolve().parents[1]


class SimClock:
    """Simulated time. sleep() advances the clock instead of waiting, so a
    four-minute mission runs in well under a second and is deterministic."""

    def __init__(self, start: float = 1_000_000.0):
        self.t = start

    def now(self) -> float:
        return self.t

    def sleep(self, dt: float) -> None:
        self.t += max(0.0, dt)


@pytest.fixture
def clock(monkeypatch) -> SimClock:
    c = SimClock()
    monkeypatch.setattr(time, "time", c.now)
    monkeypatch.setattr(time, "monotonic", c.now)
    monkeypatch.setattr(time, "sleep", c.sleep)
    return c


@pytest.fixture
def field_map() -> Map:
    return Map.load(REPO / "maps" / "field_01.yaml")


@pytest.fixture
def stack(clock, field_map):
    backend = FakeBackend(time_fn=clock.now)
    backend.connect()
    skills = Skills(backend, arrive_radius=0.5, poll_hz=20, default_timeout=60)
    tools = ToolLayer(skills, SafetyArbiter(Limits()), field_map)
    return backend, skills, tools
