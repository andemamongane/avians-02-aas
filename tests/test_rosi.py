"""Robust satisfaction intervals: sound, narrowing, and equal to the offline
score once the spec's horizon has passed."""
from __future__ import annotations

import math
import random

import pytest

from avians.maps import Map
from avians.stl import FutureBounds, Trace, interval, parse, robustness, score_online

SQUARE = {"id": "a", "kind": "goal", "polygon": [[10, 10], [20, 10], [20, 20], [10, 20]]}
STRIP = {"id": "nf", "kind": "no_fly", "polygon": [[0, 5], [30, 5], [30, 7], [0, 7]],
         "altitude": [0, 10]}
SPECS = [
    "G[0,20] !in(nf)",
    "F[0,15] G[0,3] in(a)",
    "G[0,20] !in(nf) & F[0,20] (in(a) & alt <= 6)",
    "!in(a) U[0,20] alt >= 8",
    "F[0,10] speed > 2 -> G[5,15] east >= 0",
]
BOUNDS = FutureBounds(max_speed=3.0, alt_range=(0.0, 12.0))


@pytest.fixture
def m() -> Map:
    return Map.from_dict({"regions": [SQUARE, STRIP]})


def walk(seed: int, n: int = 250, dt: float = 0.1, v: float = 3.0) -> Trace:
    """A random flight that respects BOUNDS: speed <= v, altitude in range."""
    rng = random.Random(seed)
    e, nn, a = 0.0, 0.0, 2.0
    tr = Trace([], [], [], [], [])
    he = rng.uniform(0, 2 * math.pi)
    for k in range(n):
        he += rng.gauss(0, 0.3)
        sp = rng.uniform(0, v)
        climb = rng.uniform(-0.3, 0.3) * sp
        horiz = math.sqrt(max(0.0, sp * sp - climb * climb))
        e += horiz * math.cos(he) * dt
        nn += horiz * math.sin(he) * dt
        a = min(12.0, max(0.0, a + climb * dt))
        tr.t.append(k * dt), tr.east.append(e), tr.north.append(nn)
        tr.alt.append(a), tr.speed.append(sp)
    return tr


def prefix(tr: Trace, k: int) -> Trace:
    return Trace(tr.t[:k], tr.east[:k], tr.north[:k], tr.alt[:k], tr.speed[:k])


@pytest.mark.parametrize("spec", SPECS)
@pytest.mark.parametrize("future", [None, BOUNDS], ids=["unbounded", "bounded"])
def test_interval_contains_final_and_narrows(m, spec, future):
    s = parse(spec)
    for seed in range(4):
        tr = walk(seed)
        final = robustness(s, tr, m)
        prev = (-math.inf, math.inf)
        for k in range(1, len(tr.t) + 1, 7):
            lo, hi = interval(s, prefix(tr, k), m, future=future)
            assert lo - 1e-9 <= final <= hi + 1e-9, (spec, seed, k, lo, final, hi)
            assert lo >= prev[0] - 1e-9 and hi <= prev[1] + 1e-9   # nested
            prev = (lo, hi)


@pytest.mark.parametrize("spec", SPECS)
def test_collapses_to_offline_after_horizon(m, spec):
    s = parse(spec)
    tr = walk(7, n=260)                              # 26 s > every horizon above
    lo, hi = interval(s, tr, m, future=BOUNDS)
    assert lo == pytest.approx(hi) == pytest.approx(robustness(s, tr, m))


def test_bounds_decide_before_the_deadline(m):
    # Hovering at the origin: 10+ m from the square, at most 3 m/s. "Reach a
    # within 3 s" is lost from the start, not only once the 3 s have passed.
    tr = Trace([0.0, 0.5], [0.0, 0.0], [0.0, 0.0], [2.0, 2.0], [0.0, 0.0])
    s = parse("F[0,3] in(a)")
    assert score_online(s, tr, m, future=BOUNDS)["verdict"] == "violated"
    assert score_online(s, tr, m)["verdict"] == "undecided"
    assert score_online(parse("F[0,30] in(a)"), tr, m, future=BOUNDS)["verdict"] == "undecided"


def test_satisfied_early_and_per_clause(m):
    tr = Trace([0.0, 1.0], [15.0, 15.0], [15.0, 15.0], [2.0, 2.0], [0.0, 0.0])
    rep = score_online("F[0,10] in(a) & G[0,10] !in(nf)", tr, m, future=BOUNDS)
    assert [c["verdict"] for c in rep["clauses"]] == ["satisfied", "undecided"]
    assert rep["verdict"] == "undecided"


def test_true_is_never_uncertain(m):
    tr = Trace([0.0], [0.0], [0.0], [0.0], [0.0])
    assert interval(parse("G[0,10] true"), tr, m) == (math.inf, math.inf)
