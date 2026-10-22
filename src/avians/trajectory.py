"""Plan-ahead flight: turn a pilot's flight plan into one smooth, timed trajectory.

The pilot commits to a sequence of segments (fly-through waypoints, region
centres, orbits, survey patterns, holds, a landing). This module turns them
into a reference the drone can fly continuously, without stopping at every
point, within the flight envelope:

  1. segments -> knots: points with a speed cap, an optional stop, an event label
  2. knots -> path: every fly-through corner is rounded (a quadratic Bezier
     fillet no tighter than corner_radius); stops stay sharp
  3. speed caps along the path: the segment's speed, the turn (lateral
     acceleration v^2 * curvature <= LATERAL * max_accel), climb/descent rates
  4. forward and backward passes keep acceleration and braking within
     ALONG * max_accel, so the drone slows down only where the geometry
     demands it. The two shares add up (as vectors) to under max_accel.

The result is sampled every `ds` metres: time, position and velocity. The
backend flies it (the fake follows it exactly; on AAS a trajectory controller
in AAS's offboard framework tracks it). Nothing here knows about STL: a plan is
checked against the envelope and the map's geometry, never against the score.

Pure Python, no numpy: it runs on every turn of a flight.
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

from .maps import Map
from .schema import ToolArgError

Vec = tuple[float, float, float]
LATERAL, ALONG = 0.7, 0.6            # shares of max_accel: |(0.7, 0.6)| = 0.92
SEGMENT_TYPES = ("waypoint", "region", "orbit", "survey", "hold", "land")
MAX_SEGMENTS = 60


@dataclass
class Envelope:
    max_speed: float = 15.0          # m/s along the path
    max_accel: float = 7.0           # m/s^2, along the path and across it in turns
    max_climb: float = 4.0           # m/s
    max_descent: float = 3.0         # m/s
    corner_radius: float = 6.0       # m, the widest fillet a fly-through corner gets
    ds: float = 0.25                 # m, path resolution

    @classmethod
    def from_limits(cls, limits) -> "Envelope":
        return cls(max_speed=limits.max_speed_ms, max_accel=limits.max_accel_ms2,
                   max_climb=limits.max_climb_ms, max_descent=limits.max_descent_ms,
                   corner_radius=limits.corner_radius_m)


@dataclass
class Knot:
    pos: Vec
    speed: float                     # cap on the leg arriving here (m/s)
    dwell: float | None = None       # stop here for this long (None: fly through)
    label: str = ""                  # event when passed ("" = silent)
    land: bool = False               # the plan ends with a landing here
    smooth: bool = False             # part of a curve (orbit): never filleted


@dataclass
class Trajectory:
    t: list[float]
    pos: list[Vec]
    vel: list[Vec]
    events: list[tuple[float, str]] = field(default_factory=list)
    knots: list[Knot] = field(default_factory=list)
    knot_times: list[float] = field(default_factory=list)
    lands: bool = False

    @property
    def duration(self) -> float:
        return self.t[-1]

    @property
    def length(self) -> float:
        return sum(math.dist(a, b) for a, b in zip(self.pos, self.pos[1:]))

    @property
    def top_speed(self) -> float:
        return max((_norm(v) for v in self.vel), default=0.0)

    def sample(self, t: float) -> tuple[Vec, Vec]:
        """Reference position and velocity at time t (held at the ends)."""
        if t <= 0:
            return self.pos[0], self.vel[0]
        if t >= self.t[-1]:
            return self.pos[-1], (0.0, 0.0, 0.0)
        k = bisect.bisect_right(self.t, t)
        t0, t1 = self.t[k - 1], self.t[k]
        u = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        return _lerp(self.pos[k - 1], self.pos[k], u), _lerp(self.vel[k - 1], self.vel[k], u)

    def remaining_knots(self, t: float) -> list[Knot]:
        """Knots not reached by time t: what an appended plan continues from."""
        return [k for k, tk in zip(self.knots, self.knot_times) if tk > t]


# ---------------------------------------------------------------- segments -> knots
def expand(segments: list, start: Vec, env: Envelope, m: Map | None) -> list[Knot]:
    """The pilot's plan as knots. Raises ToolArgError with a message for the pilot."""
    if not isinstance(segments, list) or not segments:
        raise ToolArgError("segments must be a non-empty list")
    if len(segments) > MAX_SEGMENTS:
        raise ToolArgError(f"at most {MAX_SEGMENTS} segments per plan")
    knots: list[Knot] = []
    here = start
    for i, seg in enumerate(segments, 1):
        if not isinstance(seg, dict) or seg.get("type") not in SEGMENT_TYPES:
            raise ToolArgError(f"segment {i}: 'type' must be one of {', '.join(SEGMENT_TYPES)}")
        if knots and knots[-1].land:
            raise ToolArgError(f"segment {i}: nothing can follow 'land'")
        kind = seg["type"]
        speed = _speed(seg, env, i)
        if kind == "waypoint":
            p = (_num(seg, "east", i), _num(seg, "north", i), _alt(seg, i))
            knots.append(Knot(p, speed, label=f"passed waypoint {i} ({p[0]:g}, {p[1]:g}, {p[2]:g})"))
        elif kind == "region":
            r = _region(seg, m, i)
            e, n = r.interior_point()
            knots.append(Knot((e, n, _alt(seg, i)), speed, label=f"reached {r.id} (segment {i})"))
        elif kind == "orbit":
            knots += _orbit(seg, here, speed, env, i)
        elif kind == "survey":
            knots += _survey(seg, here, speed, m, i)
        elif kind == "hold":
            dur = _num(seg, "duration", i)
            if dur < 0:
                raise ToolArgError(f"segment {i}: duration must be >= 0")
            stop = knots[-1] if knots else None
            if stop is None:                     # hold where the plan starts
                stop = Knot(here, env.max_speed)
                knots.append(stop)
            stop.dwell = (stop.dwell or 0.0) + dur
            stop.label = (stop.label + "; " if stop.label else "") + f"held {dur:g} s (segment {i})"
            continue
        elif kind == "land":
            alt = knots[-1].pos[2] if knots else here[2]
            knots.append(Knot((0.0, 0.0, alt), speed, label="over the launch point, landing",
                              land=True))
        here = knots[-1].pos
    return knots


def _num(seg: dict, key: str, i: int) -> float:
    if seg.get(key) is None:
        raise ToolArgError(f"segment {i} ({seg.get('type')}): missing '{key}'")
    try:
        return float(seg[key])
    except (TypeError, ValueError):
        raise ToolArgError(f"segment {i}: '{key}' must be a number, got {seg[key]!r}")


def _alt(seg: dict, i: int) -> float:
    a = _num(seg, "altitude", i)
    if a <= 0:
        raise ToolArgError(f"segment {i}: altitude must be positive (metres above launch)")
    return a


def _speed(seg: dict, env: Envelope, i: int) -> float:
    if seg.get("speed") is None:
        return env.max_speed
    s = _num(seg, "speed", i)
    if s <= 0:
        raise ToolArgError(f"segment {i}: speed must be positive")
    return min(s, env.max_speed)


def _region(seg: dict, m: Map | None, i: int):
    rid = seg.get("region")
    r = m.get(rid) if (m is not None and rid is not None) else None
    if r is None:
        known = ", ".join(m.regions) if m else "none (this mission has no map)"
        raise ToolArgError(f"segment {i}: unknown region {rid!r}; regions: {known}")
    return r


def _orbit(seg: dict, here: Vec, speed: float, env: Envelope, i: int) -> list[Knot]:
    ce, cn, alt = _num(seg, "east", i), _num(seg, "north", i), _alt(seg, i)
    r = _num(seg, "radius", i)
    turns = float(seg.get("turns", 1.0))
    if r < 1.0 or not 0 < turns <= 10:
        raise ToolArgError(f"segment {i}: orbit needs radius >= 1 m and 0 < turns <= 10")
    sign = 1.0 if seg.get("direction", "cw") == "ccw" else -1.0
    a0 = math.atan2(here[1] - cn, here[0] - ce) if math.dist(here[:2], (ce, cn)) > 1e-6 else 0.0
    speed = min(speed, math.sqrt(LATERAL * env.max_accel * r))
    steps = max(8, int(math.ceil(turns * 360 / 2)))          # a knot every ~2 degrees
    out = []
    for k in range(steps + 1):
        a = a0 + sign * 2 * math.pi * turns * k / steps
        out.append(Knot((ce + r * math.cos(a), cn + r * math.sin(a), alt), speed, smooth=k > 0))
    out[-1].label = f"orbit done (segment {i})"
    out[-1].smooth = False
    return out


def _survey(seg: dict, here: Vec, speed: float, m: Map | None, i: int) -> list[Knot]:
    """Boustrophedon rows over the region's footprint, along its longer side,
    starting from the corner nearest the drone."""
    r = _region(seg, m, i)
    alt = _alt(seg, i)
    spacing = float(seg.get("spacing", 5.0))
    if spacing < 1.0:
        raise ToolArgError(f"segment {i}: survey spacing must be >= 1 m")
    xs, ys = zip(*r.polygon)
    swap = (max(ys) - min(ys)) > (max(xs) - min(xs))          # rows run along the long side
    poly = [(y, x) for x, y in r.polygon] if swap else list(r.polygon)
    lo, hi = min(p[1] for p in poly), max(p[1] for p in poly)
    n = max(1, int((hi - lo) // spacing))
    rows = []
    for k in range(n):
        c = lo + (hi - lo) * (k + 0.5) / n
        xs_ = _crossings(poly, c)
        if len(xs_) >= 2:
            a, b = min(xs_) + spacing / 2, max(xs_) - spacing / 2
            if b <= a:
                a = b = (min(xs_) + max(xs_)) / 2
            rows.append((a, b, c))
    if not rows:
        raise ToolArgError(f"segment {i}: region {r.id} is too small to survey")
    ends = [(rows[0][0], rows[0][2]), (rows[0][1], rows[0][2]),
            (rows[-1][0], rows[-1][2]), (rows[-1][1], rows[-1][2])]
    h = (here[1], here[0]) if swap else (here[0], here[1])
    start = min(range(4), key=lambda k: math.dist(ends[k], h))
    if start >= 2:
        rows.reverse()
    flip = start % 2 == 1
    pts = []
    for a, b, c in rows:
        pts += [(b, c), (a, c)] if flip else [(a, c), (b, c)]
        flip = not flip
    out = [Knot(((y, x) if swap else (x, y)) + (alt,), speed) for x, y in pts]
    out[-1].label = f"survey of {r.id} done (segment {i})"
    return out


def _crossings(poly, c: float) -> list[float]:
    """x where the horizontal line y = c crosses the polygon's edges."""
    out = []
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        if (y1 <= c < y2) or (y2 <= c < y1):
            out.append(x1 + (c - y1) * (x2 - x1) / (y2 - y1))
    return out


# ---------------------------------------------------------------- knots -> trajectory
def build(start: Vec, v0: Vec, knots: list[Knot], env: Envelope) -> Trajectory:
    """Fly from `start` (moving at v0) through the knots."""
    verts: list[Knot] = [Knot(start, env.max_speed)]
    s0 = _norm(v0)
    if s0 > 0.3:
        # Carry the momentum: the path first continues along the current
        # velocity for as far as it takes to brake (plus the corner's fillet),
        # then turns, instead of starting with a velocity jump.
        lead = min(s0 * s0 / (2 * ALONG * env.max_accel) + env.corner_radius, 40.0)
        u = _scale(v0, 1 / s0)
        verts.append(Knot(_add(start, _scale(u, lead)), env.max_speed))
    verts += knots
    if len(verts) == 1:
        verts.append(Knot(start, env.max_speed, dwell=0.0))

    pts: list[Vec] = [start]
    caps: list[float] = [env.max_speed]
    stops: dict[int, float] = {}
    knot_idx: list[int] = []
    cur = start
    for j in range(1, len(verts)):
        k = verts[j]
        fillet = None
        if j < len(verts) - 1 and k.dwell is None and not k.smooth:
            fillet = _fillet(cur, k.pos, verts[j + 1].pos, env.corner_radius)
        if fillet is None:
            _line(pts, caps, cur, k.pos, k.speed, env.ds)
            cur = k.pos
            mid = len(pts) - 1
        else:
            enter, leave = fillet
            _line(pts, caps, cur, enter, k.speed, env.ds)
            first = len(pts)
            _bezier(pts, caps, enter, k.pos, leave, min(k.speed, verts[j + 1].speed), env.ds)
            mid = (first + len(pts) - 1) // 2
            cur = leave
        if j >= len(verts) - len(knots):
            knot_idx.append(mid)
        if k.dwell is not None:
            stops[mid] = stops.get(mid, 0.0) + k.dwell

    n = len(pts)
    ds = [0.0] + [math.dist(pts[i - 1], pts[i]) for i in range(1, n)]
    v = list(caps)
    w = max(1, round(1.0 / env.ds))          # 1 m stencil: curvature without chord noise
    for i in range(n):
        kappa = _curvature(pts[max(0, i - w)], pts[i], pts[min(n - 1, i + w)])
        if kappa > 1e-9:
            v[i] = min(v[i], math.sqrt(LATERAL * env.max_accel / kappa))
        if i > 0 and ds[i] > 1e-9:
            dz = (pts[i][2] - pts[i - 1][2]) / ds[i]
            if dz > 1e-6:
                v[i] = min(v[i], env.max_climb / dz)
            elif dz < -1e-6:
                v[i] = min(v[i], env.max_descent / -dz)
    for i in stops:
        v[i] = 0.0
    v[0] = min(v[0], s0)
    v[-1] = 0.0
    tans = [_tangent(pts, i, max(1, w // 2)) for i in range(n)]
    # The caps above use a smoothed curvature and treat turning and braking
    # separately; where the path is tighter than that (an orbit entered at a
    # right angle), the real acceleration still overshoots. Measure it and
    # lower the speed there until the whole trajectory is inside the envelope.
    for _ in range(30):
        _passes(v, ds, ALONG * env.max_accel)
        times = _times(v, ds, env.max_accel)
        over = [i for i in range(1, n) if times[i] - times[i - 1] > 1e-9 and _norm(_sub(
            _scale(tans[i], v[i]), _scale(tans[i - 1], v[i - 1]))) / (times[i] - times[i - 1])
            > env.max_accel * 1.02]
        if not over:
            break
        for i in over:
            v[i] *= 0.85
            v[i - 1] *= 0.85
    _passes(v, ds, ALONG * env.max_accel)
    times = _times(v, ds, env.max_accel)

    t_out, p_out, v_out = [], [], []
    shift = 0.0
    for i in range(n):
        t = times[i] + shift
        t_out.append(t), p_out.append(pts[i]), v_out.append(_scale(tans[i], v[i]))
        if stops.get(i):
            shift += stops[i]
            t_out.append(t + stops[i]), p_out.append(pts[i]), v_out.append((0.0, 0.0, 0.0))
    times = [times[i] + sum(d for j, d in stops.items() if j < i) for i in range(n)]

    knot_times = [times[i] for i in knot_idx]
    events = [(tk, k.label) for k, tk in zip(knots, knot_times) if k.label]
    return Trajectory(t_out, p_out, v_out, events=events, knots=list(knots),
                      knot_times=knot_times, lands=bool(knots and knots[-1].land))


def _passes(v: list[float], ds: list[float], a: float) -> None:
    """Forward (accelerate) and backward (brake) passes, in place."""
    n = len(v)
    for i in range(1, n):
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * a * ds[i]))
    for i in range(n - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * a * ds[i + 1]))


def _times(v: list[float], ds: list[float], a: float) -> list[float]:
    """Arrival time at each point (dwells not included)."""
    t, out = 0.0, [0.0]
    for i in range(1, len(v)):
        if ds[i] > 1e-9:
            t += ds[i] / max((v[i - 1] + v[i]) / 2, math.sqrt(a * ds[i]) / 2)
        out.append(t)
    return out


def _fillet(a: Vec, b: Vec, c: Vec, radius: float) -> tuple[Vec, Vec] | None:
    """Where a fillet at corner b starts and ends, or None for a straight corner."""
    l1, l2 = math.dist(a, b), math.dist(b, c)
    if l1 < 1e-6 or l2 < 1e-6:
        return None
    u1, u2 = _scale(_sub(b, a), 1 / l1), _scale(_sub(c, b), 1 / l2)
    cosang = max(-1.0, min(1.0, _dot(u1, u2)))
    turn = math.acos(cosang)
    if turn < 1e-3:
        return None
    d = min(radius * math.tan(turn / 2) if turn < math.pi - 1e-3 else math.inf,
            0.45 * min(l1, l2))
    return _sub(b, _scale(u1, d)), _add(b, _scale(u2, d))


def _line(pts, caps, a: Vec, b: Vec, cap: float, ds: float) -> None:
    n = max(1, int(math.ceil(math.dist(a, b) / ds)))
    if math.dist(a, b) < 1e-9:
        return
    for k in range(1, n + 1):
        pts.append(_lerp(a, b, k / n))
        caps.append(cap)


def _bezier(pts, caps, e: Vec, c: Vec, x: Vec, cap: float, ds: float) -> None:
    n = max(2, int(math.ceil((math.dist(e, c) + math.dist(c, x)) / ds)))
    for k in range(1, n + 1):
        u = k / n
        pts.append(tuple((1 - u) ** 2 * e[j] + 2 * u * (1 - u) * c[j] + u ** 2 * x[j]
                         for j in range(3)))
        caps.append(cap)


def _curvature(a: Vec, b: Vec, c: Vec) -> float:
    """1 / circumradius of the three points (0 when collinear)."""
    ab, bc, ca = math.dist(a, b), math.dist(b, c), math.dist(c, a)
    if ab * bc * ca < 1e-12:
        return 0.0
    cross = _cross(_sub(b, a), _sub(c, a))
    return 2 * _norm(cross) / (ab * bc * ca)


def _tangent(pts: list[Vec], i: int, h: int = 1) -> Vec:
    """Direction of travel at point i, over +-h points: a curve drawn as short
    chords would otherwise turn in jumps at every vertex."""
    for j, k in ((i - h, i + h), (i, i + 1), (i - 1, i)):
        j, k = max(0, j), min(len(pts) - 1, k)
        d = _sub(pts[k], pts[j])
        nd = _norm(d)
        if nd > 1e-9:
            return _scale(d, 1 / nd)
    return (0.0, 0.0, 0.0)


# ---------------------------------------------------------------- preview
def preview(traj: Trajectory, m: Map | None) -> str:
    """What the pilot is told about an accepted plan: geometry only."""
    lines = [f"plan: {traj.length:.0f} m, {traj.duration:.0f} s, "
             f"top speed {traj.top_speed:.1f} m/s"
             + (", ends with a landing at the launch point" if traj.lands else "")]
    for t, label in traj.events:
        lines.append(f"  t+{t:5.1f} s  {label}")
    if m is not None and m.regions:
        spans = region_spans(traj, m)
        lines.append("path inside regions: " + ("; ".join(
            f"{rid} t+{a:.0f}..{b:.0f} s" for rid, a, b in spans) or "none"))
    return "\n".join(lines)


def region_spans(traj: Trajectory, m: Map) -> list[tuple[str, float, float]]:
    """(region, enter time, leave time) along the planned path."""
    spans, open_ = [], {}
    for t, p in zip(traj.t, traj.pos):
        now = set(m.inside(*p))
        for rid in now - open_.keys():
            open_[rid] = t
        for rid in list(open_):
            if rid not in now:
                spans.append((rid, open_.pop(rid), t))
    spans += [(rid, a, traj.t[-1]) for rid, a in open_.items()]
    return sorted(spans, key=lambda s: s[1])


# ---------------------------------------------------------------- vectors
def _add(a, b): return (a[0] + b[0], a[1] + b[1], a[2] + b[2])
def _sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def _scale(a, s): return (a[0] * s, a[1] * s, a[2] * s)
def _dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def _norm(a): return math.sqrt(_dot(a, a))
def _lerp(a, b, u): return (a[0] + (b[0] - a[0]) * u, a[1] + (b[1] - a[1]) * u,
                            a[2] + (b[2] - a[2]) * u)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
