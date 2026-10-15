"""Map geometry and STL semantics, checked against hand-computed values."""
from __future__ import annotations

import math

import pytest

from avians.maps import Map
from avians.stl import STLSyntaxError, Trace, check_regions, parse, robustness, score

SQUARE = {"id": "a", "kind": "goal", "polygon": [[10, 10], [20, 10], [20, 20], [10, 20]]}
STRIP = {"id": "nf", "kind": "no_fly", "polygon": [[0, 5], [30, 5], [30, 7], [0, 7]]}
ELL = {"id": "L", "kind": "other", "polygon": [[0, 0], [10, 0], [10, 2], [2, 2], [2, 10], [0, 10]]}
PRISM = {"id": "p", "kind": "goal", "polygon": [[-5, -5], [5, -5], [5, 5], [-5, 5]],
         "altitude": [2, 8]}


@pytest.fixture
def m() -> Map:
    return Map.from_dict({"regions": [SQUARE, STRIP, ELL, PRISM]})


def trace(points, dt=1.0) -> Trace:
    """points: list of (east, north, alt) sampled every dt seconds."""
    t = [i * dt for i in range(len(points))]
    e, n, a = (list(c) for c in zip(*points))
    return Trace(t, e, n, a, [0.0] * len(points))


# --- maps ---------------------------------------------------------------------
def test_signed_distance(m):
    a = m.get("a")
    assert a.signed_distance_xy(15, 15) == 5
    assert a.signed_distance_xy(25, 15) == -5
    assert a.signed_distance_xy(20, 15) == 0
    assert a.signed_distance_xy(23, 24) == -5            # corner: Euclidean, 3-4-5


def test_prism_robustness_is_min_of_margins(m):
    p = m.get("p")
    assert p.robustness(0, 0, 5) == 3                    # 5 to the walls, 3 to floor/ceiling
    assert p.robustness(0, 0, 9) == -1                   # above the ceiling
    assert not p.contains(0, 0, 1)


def test_concave_interior_point_is_inside(m):
    L = m.get("L")
    assert not L.contains_xy(*L.centroid()) or L.signed_distance_xy(*L.centroid()) < 0.5
    e, n = L.interior_point()
    assert L.signed_distance_xy(e, n) >= 0.9


def test_crossings(m):
    assert set(m.crossings((6, 0), (6, 30))) == {"nf", "L"}
    assert m.crossings((40, 0), (40, 30)) == []
    assert "a" in m.crossings((0, 15), (30, 15))         # passes straight through


@pytest.mark.parametrize("bad,match", [
    ({"regions": [SQUARE, SQUARE]}, "duplicate"),
    ({"regions": [{**SQUARE, "kind": "lava"}]}, "kind"),
    ({"regions": [{**SQUARE, "polygon": [[0, 0], [1, 1]]}]}, "3 vertices"),
])
def test_map_rejects_bad_input(bad, match):
    with pytest.raises(ValueError, match=match):
        Map.from_dict(bad)


# --- parser -------------------------------------------------------------------
def test_parse_round_trip_and_precedence():
    s = parse("G[0,40] !in(nf) & F[0,30] G[0,10] in(a) | alt <= 30")
    assert s.op == "or"                                  # & binds tighter than |
    assert str(s) == "(G[0,40] !in(nf) & F[0,30] G[0,10] in(a)) | alt <= 30"
    assert parse(str(s)) == s
    assert parse("in(a) -> F[0,5] in(nf)").op == "or"


def test_horizon_and_regions():
    s = parse("G[0,40] !in(nf) & F[10,30] G[0,10] in(a)")
    assert s.horizon() == 40
    assert s.regions() == {"nf", "a"}
    assert check_regions(s, Map.from_dict({"regions": [SQUARE]})) == ["nf"]


@pytest.mark.parametrize("bad", ["", "in(a", "G[5,1] in(a)", "alt", "foo(a)", "in(a) &"])
def test_parse_errors(bad):
    with pytest.raises(STLSyntaxError):
        parse(bad)


# --- semantics ----------------------------------------------------------------
def test_atoms_and_boolean_ops(m):
    tr = trace([(15, 15, 5)])
    assert robustness(parse("in(a)"), tr, m) == 5
    assert robustness(parse("!in(a)"), tr, m) == -5
    assert robustness(parse("alt <= 8"), tr, m) == 3
    assert robustness(parse("alt >= 8"), tr, m) == -3
    assert robustness(parse("in(a) & alt >= 8"), tr, m) == -3
    assert robustness(parse("in(a) | alt >= 8"), tr, m) == 5


def test_always_and_eventually_windows(m):
    # altitude 0,1,2,...,10 one sample per second
    tr = trace([(0, 0, float(k)) for k in range(11)])
    assert robustness(parse("F[0,10] alt >= 9"), tr, m, dt=1.0) == 1
    assert robustness(parse("G[0,10] alt >= 0"), tr, m, dt=1.0) == 0
    assert robustness(parse("G[2,4] alt >= 1"), tr, m, dt=1.0) == 1      # min over 2..4
    assert robustness(parse("F[0,3] alt >= 5"), tr, m, dt=1.0) == -2     # best is 3


def test_until(m):
    # alt climbs 0..10; "alt <= 6 until alt >= 5": q first at t=5, p still holds
    tr = trace([(0, 0, float(k)) for k in range(11)])
    # alt = t. p: alt <= 6 (rho 6 - t), q: alt >= 5 (rho t - 5). Choosing t' = 6:
    # q(6) = 1, and p on [0, 6) has minimum p(5) = 1, so rho = 1 (the best t').
    r = robustness(parse("alt <= 6 U[0,10] alt >= 5"), tr, m, dt=1.0)
    assert r == 1
    # q never happens inside the window -> violated
    assert robustness(parse("true U[0,3] alt >= 5"), tr, m, dt=1.0) < 0


def test_short_flight_is_padded_with_last_state(m):
    # 3 s flight ending inside a; "always in a for the first 10 s" needs padding
    tr = trace([(15, 15, 5), (15, 15, 5), (15, 15, 5), (15, 15, 5)])
    rep = score("G[0,10] in(a)", tr, m)
    assert rep["satisfied"] and rep["padded_s"] == 7.0 and rep["flight_end_s"] == 3.0


def test_score_reports_each_clause(m):
    # cut 1 m into the no-fly strip on the way to a, then hold in a for 10 s
    pts = [(0, 0, 5), (3, 6, 5), (15, 15, 5)] + [(15, 15, 5)] * 12
    rep = score("G[0,15] !in(nf) & F[0,5] G[0,10] in(a)", trace(pts), m)
    assert not rep["satisfied"] and rep["robustness"] == -1.0
    assert [c["satisfied"] for c in rep["clauses"]] == [False, True]
    assert rep["clauses"][1]["robustness"] == 5.0


def test_true_reports_infinity_as_text(m):
    rep = score("true", trace([(0, 0, 0)]), m)
    assert rep["robustness"] == "inf" and rep["satisfied"]
    assert not math.isnan(robustness(parse("true"), trace([(0, 0, 0)]), m))
