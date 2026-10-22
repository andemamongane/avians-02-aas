"""SFO: a superset of STL with the same scores on STL specs, quantifiers equal
to their STL expansions, cross-time comparisons, and sound online intervals."""
from __future__ import annotations

import math
import time

import pytest

from avians import sfo, stl
from avians.maps import Map
from avians.stl import STLSyntaxError, Trace

from test_rosi import BOUNDS, SPECS, SQUARE, STRIP, prefix, walk

EXTRA = {"id": "b", "kind": "goal", "polygon": [[-20, 10], [-10, 10], [-10, 20], [-20, 20]]}
STRIP2 = {"id": "nf2", "kind": "no_fly", "polygon": [[-30, -10], [-25, -10], [-25, 30], [-30, 30]]}


@pytest.fixture
def m() -> Map:
    return Map.from_dict({"regions": [SQUARE, STRIP, EXTRA, STRIP2]})


def climb(points) -> Trace:
    """(east, north, alt) one sample per second."""
    e, n, a = (list(c) for c in zip(*points))
    return Trace([float(i) for i in range(len(points))], e, n, a, [0.0] * len(points))


@pytest.mark.parametrize("spec", SPECS)
def test_stl_specs_score_identically(m, spec):
    for seed in range(3):
        tr = walk(seed)
        assert sfo.robustness(sfo.parse(spec), tr, m) == pytest.approx(
            stl.robustness(stl.parse(spec), tr, m))
        assert sfo.horizon(sfo.parse(spec)) == stl.parse(spec).horizon()


@pytest.mark.parametrize("q,expanded", [
    ("forall r : no_fly. G[0,20] !in(r)", "G[0,20] !in(nf) & G[0,20] !in(nf2)"),
    ("exists r : goal. F[0,20] in(r)", "F[0,20] in(a) | F[0,20] in(b)"),
    ("exists r in {a, nf}. F[0,10] in(r)", "F[0,10] in(a) | F[0,10] in(nf)"),
    ("exists t in [0, 10]. (alt >= 5) @ t", "F[0,10] alt >= 5"),
    ("forall t in [2, 12]. (!in(nf)) @ t", "G[2,12] !in(nf)"),
    ("exists t1 in [0, 10]. ((G[0,3] in(a)) @ t1 & exists t2 in [t1 + 1, t1 + 6]. in(nf) @ t2)",
     "F[0,10] (G[0,3] in(a) & F[1,6] in(nf))"),
])
def test_quantifiers_equal_their_stl_expansion(m, q, expanded):
    for seed in range(3):
        tr = walk(seed)
        assert sfo.robustness(sfo.parse(q), tr, m) == pytest.approx(
            stl.robustness(stl.parse(expanded), tr, m))


def test_signals_compared_across_times(m):
    # altitude 10, 9, ... 0: falls 1 m/s
    tr = climb([(0, 0, 10.0 - k) for k in range(11)])
    # never more than 3 m below the altitude at any earlier moment within 5 s
    s = sfo.parse("forall t1 in [0, 5]. forall t2 in [t1, t1 + 5]. alt@t2 >= alt@t1 - 3")
    assert sfo.robustness(s, tr, m, dt=1.0) == pytest.approx(-2.0)   # drops 5 m in 5 s
    s = sfo.parse("forall t1 in [0, 5]. forall t2 in [t1, t1 + 2]. alt@t2 >= alt@t1 - 3")
    assert sfo.robustness(s, tr, m, dt=1.0) == pytest.approx(1.0)
    # time arithmetic: the first moment below 4 m comes 7 s after... any moment above 9 m
    s = sfo.parse("exists t1 in [0, 10]. exists t2 in [t1, 10]. "
                  "(alt >= 9) @ t1 & (alt < 4) @ t2 & t2 - t1 <= 7")
    assert sfo.robustness(s, tr, m, dt=1.0) == pytest.approx(0.0)    # 1 m and 0 s of margin
    assert sfo.robustness(sfo.parse("exists t in [0, 10]. 2 * t - 3 >= 15"), tr, m, dt=1.0) == 2


@pytest.mark.parametrize("spec", [
    "forall r : no_fly. G[0,20] !in(r)",
    "forall t1 in [0, 15]. (in(a) @ t1 -> forall t2 in [t1, t1 + 5]. alt@t2 >= alt@t1 - 1)",
    "exists t in [0, 20]. (alt >= 6) @ t & speed@t <= 2.5",
])
def test_online_intervals_are_sound(m, spec):
    s = sfo.parse(spec)
    for seed in range(3):
        tr = walk(seed)
        final = sfo.robustness(s, tr, m)
        prev = (-math.inf, math.inf)
        for k in range(1, len(tr.t) + 1, 25):
            rep = sfo.score_online(s, prefix(tr, k), m, future=BOUNDS)
            lo, hi = (float(rep[x]) if isinstance(rep[x], str) else rep[x] for x in ("lo", "hi"))
            assert lo - 1e-3 <= final <= hi + 1e-3            # reports round to 1 mm
            assert lo >= prev[0] - 1e-3 and hi <= prev[1] + 1e-3
            prev = (lo, hi)


def test_reports_match_stl_format(m):
    tr = walk(1)
    # a quantifier's body extends as far right as it can: parenthesize one clause
    rep = sfo.score("(forall r : no_fly. G[0,20] !in(r)) & F[0,20] in(a)", tr, m)
    assert rep["logic"] == "sfo" and len(rep["clauses"]) == 2
    assert set(rep) >= set(stl.score("F[0,20] in(a)", tr, m))
    assert sfo.parse(rep["clauses"][0]["clause"]) == sfo.parse("forall r : no_fly. G[0,20] !in(r)")


def test_regions_and_horizon(m):
    s = sfo.parse("forall r : no_fly. G[0,20] !in(r) & exists r in {a, zz}. F[0,5] in(r) "
                  "& forall t in [0, 30]. (G[0,4] in(b)) @ t")
    assert sfo.regions(s) == {"a", "zz", "b"}
    assert sfo.check_regions(s, m) == ["zz"]
    assert sfo.horizon(s) == 34
    assert sfo.horizon(sfo.parse("forall t1 in [0, 10]. exists t2 in [t1, t1 + 5]. alt@t2 > 1")) == 15


@pytest.mark.parametrize("bad,match", [
    ("G[0,5] in(a) @ t", "not a time variable"),
    ("forall t in [0, 5]. G[0,5] (in(a) @ t)", "STL only"),
    ("alt@t >= 3", "not a time variable"),
    ("forall t in [0, s]. in(a) @ t", "not a time variable"),
    ("forall a in [0,1]. forall b in [0,1]. forall c in [0,1]. alt@c > 1", "at most 2"),
    ("forall alt in [0, 5]. in(a) @ alt", "bad variable"),
])
def test_parse_errors(bad, match):
    with pytest.raises(STLSyntaxError, match=match):
        sfo.parse(bad)


def test_two_time_variables_over_a_long_flight_are_fast_enough(m):
    tr = walk(2, n=2400)                              # 240 s at 0.1 s
    s = sfo.parse("forall t1 in [0, 200]. (in(a) @ t1 -> "
                  "forall t2 in [t1, t1 + 30]. alt@t2 >= alt@t1 - 3)")
    t0 = time.time()
    sfo.robustness(s, tr, m)
    assert time.time() - t0 < 10
