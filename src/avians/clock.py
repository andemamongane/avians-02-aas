"""The mission clock: real time, or lockstep.

Real time (demos, AAS): now() is the wall clock and sleep() waits. The drone
keeps flying while a model thinks, so how long it thinks shapes what it sees.

Lockstep (data collection, fake backend): the simulation stands still unless
avians advances it. sleep(dt) moves virtual time forward at once, and nothing
moves while a model thinks; each decision then takes effect after a fixed,
recorded latency (executor.py). The data no longer depends on how long the
teacher happened to think, so a student with a different latency is trained
on the states it will actually meet.

Everything that measures mission time reads it here (logger, monitor,
executor, skills, runner, fake backend). In real time the time module is
looked up at call time, so tests that patch it still drive everything.
"""
from __future__ import annotations

import time

_virtual: float | None = None


def use_lockstep(start: float | None = None) -> None:
    """Switch to virtual time, starting now (or at `start`)."""
    global _virtual
    _virtual = time.time() if start is None else start


def use_realtime() -> None:
    global _virtual
    _virtual = None


def lockstep() -> bool:
    return _virtual is not None


def now() -> float:
    """Seconds, epoch-like: the wall clock, or virtual time in lockstep."""
    return time.time() if _virtual is None else _virtual


def monotonic() -> float:
    return time.monotonic() if _virtual is None else _virtual


def sleep(dt: float) -> None:
    """Let dt seconds of mission time pass."""
    global _virtual
    if _virtual is None:
        time.sleep(dt)
    else:
        _virtual += max(0.0, dt)
