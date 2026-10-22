"""Signal temporal logic (STL): parse a mission spec, score a flight against it.

Syntax (whitespace-insensitive, times in seconds from mission start):

    in(survey_a)            inside a map region (footprint, and altitude band if any)
    alt <= 30               comparisons on signals: alt, speed, east, north
    !phi   phi & psi   phi | psi   phi -> psi
    G[a,b] phi              always, for every time in [t+a, t+b]
    F[a,b] phi              eventually, at some time in [t+a, t+b]
    phi U[a,b] psi          psi at some time in [t+a, t+b], and phi until then
    true   (parentheses)

    example:  G[0,180] !in(nofly_1) & F[0,90] G[0,10] in(survey_a)

Quantitative (robustness) semantics, the standard ones (Donze & Maler 2010):
    in(R)       signed distance to R's boundary, metres (+ inside)
    alt <= c    c - alt          alt >= c    alt - c
    !phi        -rho(phi)
    &  / |      min / max
    G / F       min / max over the window
    U           max over t' in window of min(rho(psi, t'), min of rho(phi) on [t, t'))
                (phi must hold BEFORE psi becomes true, not necessarily at that instant)

rho > 0 means satisfied, with that much margin; rho < 0 means violated by that
much. The units are mixed (metres for regions and altitude, m/s for speed), as
usual in STL: compare robustness of the same spec across episodes, not across
specs.

Finite traces: a flight ends (the drone lands) before a long window does. The
trace is padded by holding the last sample, i.e. the drone stays where it
stopped. That keeps G[0,180] meaningful for a 120 s flight. The amount of
padding is reported so nobody is surprised by it.

The trace is resampled to a fixed step (default 0.1 s) before evaluation.
"""
from __future__ import annotations

import math
import re
from collections import deque
from dataclasses import dataclass

from .maps import Map

SIGNALS = ("alt", "speed", "east", "north")


# ---------------------------------------------------------------- AST
@dataclass
class Node:
    op: str                                   # in, cmp, true, not, and, or, G, F, U
    args: tuple = ()
    lo: float = 0.0
    hi: float = 0.0
    region: str = ""
    signal: str = ""
    cmp: str = ""
    value: float = 0.0

    def __str__(self) -> str:
        if self.op == "true":
            return "true"
        if self.op == "in":
            return f"in({self.region})"
        if self.op == "cmp":
            return f"{self.signal} {self.cmp} {self.value:g}"
        if self.op == "not":
            return f"!{_paren(self.args[0])}"
        if self.op in ("and", "or"):
            sym = " & " if self.op == "and" else " | "
            return sym.join(_paren(a) for a in self.args)
        if self.op in ("G", "F"):
            return f"{self.op}[{self.lo:g},{self.hi:g}] {_paren(self.args[0])}"
        if self.op == "U":
            return f"{_paren(self.args[0])} U[{self.lo:g},{self.hi:g}] {_paren(self.args[1])}"
        return self.op

    def horizon(self) -> float:
        """How far past t this formula looks, in seconds."""
        if self.op in ("G", "F", "U"):
            return self.hi + max((a.horizon() for a in self.args), default=0.0)
        return max((a.horizon() for a in self.args), default=0.0)

    def regions(self) -> set[str]:
        out = {self.region} if self.op == "in" else set()
        for a in self.args:
            out |= a.regions()
        return out


def _paren(n: Node) -> str:
    s = str(n)
    return f"({s})" if n.op in ("and", "or", "U") else s


# ---------------------------------------------------------------- parser
class STLSyntaxError(ValueError):
    pass


_TOKEN = re.compile(r"\s*(->|<=|>=|[()\[\],!&|<>~]|[A-Za-z_][A-Za-z0-9_]*|-?\d+(?:\.\d*)?|-?\.\d+)")


def _tokenize(text: str) -> list[str]:
    tokens, pos = [], 0
    text = text.strip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if not m:
            raise STLSyntaxError(f"unexpected character at {pos}: {text[pos:pos + 10]!r}")
        tokens.append(m.group(1))
        pos = m.end()
        while pos < len(text) and text[pos].isspace():
            pos += 1
    return tokens


class _Parser:
    # precedence, loosest first:  ->  |  &  U  unary(! G F)
    def __init__(self, text: str):
        self.toks = _tokenize(text)
        self.i = 0

    def parse(self) -> Node:
        if not self.toks:
            raise STLSyntaxError("empty spec")
        node = self.implies()
        if self.i != len(self.toks):
            raise STLSyntaxError(f"unexpected '{self.toks[self.i]}'")
        return node

    def peek(self) -> str | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def take(self, expected: str | None = None) -> str:
        tok = self.peek()
        if tok is None:
            raise STLSyntaxError(f"spec ended early (expected {expected or 'more'})")
        if expected is not None and tok != expected:
            raise STLSyntaxError(f"expected '{expected}', got '{tok}'")
        self.i += 1
        return tok

    def implies(self) -> Node:
        left = self.or_()
        if self.peek() == "->":
            self.take()
            right = self.implies()
            return Node("or", (Node("not", (left,)), right))
        return left

    def or_(self) -> Node:
        parts = [self.and_()]
        while self.peek() == "|":
            self.take()
            parts.append(self.and_())
        return parts[0] if len(parts) == 1 else Node("or", tuple(parts))

    def and_(self) -> Node:
        parts = [self.until()]
        while self.peek() == "&":
            self.take()
            parts.append(self.until())
        return parts[0] if len(parts) == 1 else Node("and", tuple(parts))

    def until(self) -> Node:
        left = self.unary()
        if self.peek() == "U":
            self.take()
            lo, hi = self.interval()
            right = self.unary()
            return Node("U", (left, right), lo=lo, hi=hi)
        return left

    def unary(self) -> Node:
        tok = self.peek()
        if tok in ("!", "~"):
            self.take()
            return Node("not", (self.unary(),))
        if tok in ("G", "F"):
            self.take()
            lo, hi = self.interval()
            return Node(tok, (self.unary(),), lo=lo, hi=hi)
        return self.atom()

    def interval(self) -> tuple[float, float]:
        self.take("[")
        lo = self.number()
        self.take(",")
        hi = self.number()
        self.take("]")
        if not 0 <= lo <= hi:
            raise STLSyntaxError(f"interval [{lo:g},{hi:g}] must satisfy 0 <= a <= b")
        return lo, hi

    def number(self) -> float:
        tok = self.take()
        try:
            return float(tok)
        except ValueError:
            raise STLSyntaxError(f"expected a number, got '{tok}'")

    def atom(self) -> Node:
        tok = self.take()
        if tok == "(":
            node = self.implies()
            self.take(")")
            return node
        if tok == "true":
            return Node("true")
        if tok == "in":
            self.take("(")
            region = self.take()
            self.take(")")
            return Node("in", region=region)
        if tok in SIGNALS:
            cmp = self.take()
            if cmp not in ("<", "<=", ">", ">="):
                raise STLSyntaxError(f"expected a comparison after '{tok}', got '{cmp}'")
            return Node("cmp", signal=tok, cmp=cmp, value=self.number())
        raise STLSyntaxError(
            f"unexpected '{tok}' (atoms are in(<region>), true, or "
            f"{'/'.join(SIGNALS)} compared to a number)")


def parse(text: str) -> Node:
    return _Parser(text).parse()


def check_regions(spec: Node, m: Map | None) -> list[str]:
    """Region ids the spec uses that the map does not define."""
    known = set(m.regions) if m else set()
    return sorted(spec.regions() - known)


# ---------------------------------------------------------------- traces
@dataclass
class Trace:
    """A flight as signals over time. t is seconds from mission start."""
    t: list[float]
    east: list[float]
    north: list[float]
    alt: list[float]
    speed: list[float]

    def resampled(self, dt: float, until: float) -> "Trace":
        """Zero-order hold onto a fixed grid from 0 to `until` (inclusive),
        holding the last sample past the end of the flight."""
        if not self.t:
            raise ValueError("empty trace")
        n = int(math.floor(until / dt + 1e-9)) + 1
        out = Trace([], [], [], [], [])
        j = 0
        for k in range(n):
            tk = k * dt
            while j + 1 < len(self.t) and self.t[j + 1] <= tk:
                j += 1
            out.t.append(tk)
            out.east.append(self.east[j])
            out.north.append(self.north[j])
            out.alt.append(self.alt[j])
            out.speed.append(self.speed[j] if tk <= self.t[-1] else 0.0)
        return out


# ---------------------------------------------------------------- semantics
def _rho(node: Node, tr: Trace, m: Map | None, dt: float, need: int = 1) -> list[float]:
    """Robustness per sample. Only the first `need` entries are guaranteed:
    the top of a formula needs t = 0 alone, and each temporal operator widens
    that by its window. This keeps U (quadratic) to O(window) at the top level."""
    n = len(tr.t)
    op = node.op
    if op == "true":
        return [math.inf] * n
    if op == "in":
        region = m.get(node.region) if m else None
        if region is None:
            raise ValueError(f"spec uses region '{node.region}', which the map does not define")
        return [region.robustness(e, nn, a) for e, nn, a in zip(tr.east, tr.north, tr.alt)]
    if op == "cmp":
        sig = getattr(tr, node.signal)
        if node.cmp in ("<", "<="):
            return [node.value - v for v in sig]
        return [v - node.value for v in sig]
    if op == "not":
        return [-v for v in _rho(node.args[0], tr, m, dt, need)]
    if op in ("and", "or"):
        parts = [_rho(a, tr, m, dt, need) for a in node.args]
        f = min if op == "and" else max
        return [f(vals) for vals in zip(*parts)]
    lo, hi = int(round(node.lo / dt)), int(round(node.hi / dt))
    sub = min(n, need + hi)
    if op in ("G", "F"):
        return _window(_rho(node.args[0], tr, m, dt, sub), lo, hi, use_min=(op == "G"))
    if op == "U":
        return _until(_rho(node.args[0], tr, m, dt, sub), _rho(node.args[1], tr, m, dt, sub),
                      lo, hi, need)
    raise ValueError(f"unknown operator {op}")


def _until(a: list[float], b: list[float], lo: int, hi: int, need: int) -> list[float]:
    """The first `need` entries of p U[lo,hi] q (indices in samples)."""
    n = min(len(a), len(b))
    out = []
    for i in range(min(n, need)):
        best, run_min = -math.inf, math.inf
        for j in range(i, min(n, i + hi + 1)):
            if j >= i + lo:
                best = max(best, min(b[j], run_min))
            run_min = min(run_min, a[j])
        out.append(best)
    return out


def _window(x: list[float], lo: int, hi: int, use_min: bool) -> list[float]:
    """out[i] = min (or max) of x[i+lo .. i+hi], clipped to the trace.
    Monotonic deque, O(n). Empty windows give +inf for G and -inf for F; the
    padding in score() makes them impossible for the spec's own horizon."""
    n = len(x)            # x may be shorter than the trace: only its needed prefix
    better = (lambda a, b: a <= b) if use_min else (lambda a, b: a >= b)
    empty = math.inf if use_min else -math.inf
    out, dq, nxt = [], deque(), 0
    for i in range(n):
        while nxt < n and nxt <= i + hi:
            while dq and better(x[nxt], x[dq[-1]]):
                dq.pop()
            dq.append(nxt)
            nxt += 1
        while dq and dq[0] < i + lo:
            dq.popleft()
        out.append(x[dq[0]] if dq else empty)
    return out


def robustness(spec: Node, trace: Trace, m: Map | None = None, dt: float = 0.1) -> float:
    """Robustness of the spec at mission start (t = 0)."""
    tr = trace.resampled(dt, max(trace.t[-1], spec.horizon()))
    return _rho(spec, tr, m, dt)[0]


def score(spec_text: str, trace: Trace, m: Map | None = None, dt: float = 0.1) -> dict:
    """Overall and per-clause robustness, for summary.json and the CLI."""
    spec = parse(spec_text)
    end = trace.t[-1]
    horizon = spec.horizon()
    tr = trace.resampled(dt, max(end, horizon))
    clauses = spec.args if spec.op == "and" else (spec,)

    def entry(node: Node) -> dict:
        r = _rho(node, tr, m, dt)[0]
        return {"clause": str(node), "robustness": _round(r), "satisfied": r > 0}

    overall = _rho(spec, tr, m, dt)[0]
    return {
        "spec": str(spec),
        "robustness": _round(overall),
        "satisfied": overall > 0,
        "clauses": [entry(c) for c in clauses],
        "horizon_s": horizon,
        "flight_end_s": round(end, 1),
        "padded_s": round(max(0.0, horizon - end), 1),
        "dt_s": dt,
    }


def _round(v: float) -> float | str:
    """JSON has no infinity; an unconstrained clause (e.g. `true`) reports "inf"."""
    return ("inf" if v > 0 else "-inf") if math.isinf(v) else round(v, 3)


# ---------------------------------------------------------------- online (ROSI)
# Robust satisfaction intervals (Deshmukh et al. 2017): during a flight the
# trace is known only up to now, so each formula's robustness is an interval
# [lo, hi] that contains every value the finished flight could still score.
# hi <= 0: the spec can no longer be satisfied, whatever the pilot does next.
# lo > 0: it is satisfied already, whatever happens next.
#
# Every operator is monotone in its arguments (! is antitone), so intervals
# propagate bound by bound through the same _window and _until as above.

@dataclass
class FutureBounds:
    """What the unobserved rest of the flight can do. Without it the future is
    (-inf, +inf) and a clause is decided only when its window has fully passed.

    Region robustness is a signed distance and altitude margins are distances,
    so every atom is 1-Lipschitz in position: a drone moving at most max_speed
    cannot change an atom faster than max_speed per second. max_speed must be a
    true bound on the vehicle (with margin over the commanded limit), or the
    monitor can declare a violation that the flight would have avoided."""
    max_speed: float = 12.0
    alt_range: tuple[float, float] = (-1.0, 31.0)


def _rho_iv(node: Node, tr: Trace, m: Map | None, dt: float, known: int,
            future: FutureBounds | None, need: int = 1) -> tuple[list[float], list[float]]:
    """(lo, hi) per sample. Samples [0, known) are observed; the rest are bounds.
    tr holds the last observed state past `known` (from resampled())."""
    n = len(tr.t)
    op = node.op
    if op == "true":
        return [math.inf] * n, [math.inf] * n
    if op in ("in", "cmp"):
        head = Trace(tr.t[:known], tr.east[:known], tr.north[:known], tr.alt[:known],
                     tr.speed[:known])
        now = _rho(node, head, m, dt)
        now += [now[-1]] * (n - known)       # past `known`: the value at t_now
        lo, hi = list(now), list(now)
        t_now = tr.t[known - 1]
        for k in range(known, n):
            if future is None:
                lo[k], hi[k] = -math.inf, math.inf
                continue
            reach = future.max_speed * (tr.t[k] - t_now)
            if op == "cmp" and node.signal == "speed":
                s_lo, s_hi = 0.0, future.max_speed
                lo[k], hi[k] = ((node.value - s_hi, node.value - s_lo) if node.cmp in ("<", "<=")
                                else (s_lo - node.value, s_hi - node.value))
            elif op == "cmp" and node.signal == "alt":
                a0, (a_min, a_max) = tr.alt[k], future.alt_range
                a_lo, a_hi = max(a_min, a0 - reach), min(a_max, a0 + reach)
                lo[k], hi[k] = ((node.value - a_hi, node.value - a_lo) if node.cmp in ("<", "<=")
                                else (a_lo - node.value, a_hi - node.value))
            else:
                lo[k], hi[k] = now[k] - reach, now[k] + reach
        return lo, hi
    if op == "not":
        lo, hi = _rho_iv(node.args[0], tr, m, dt, known, future, need)
        return [-v for v in hi], [-v for v in lo]
    if op in ("and", "or"):
        parts = [_rho_iv(a, tr, m, dt, known, future, need) for a in node.args]
        f = min if op == "and" else max
        return ([f(v) for v in zip(*(p[0] for p in parts))],
                [f(v) for v in zip(*(p[1] for p in parts))])
    w_lo, w_hi = int(round(node.lo / dt)), int(round(node.hi / dt))
    sub = min(n, need + w_hi)
    if op in ("G", "F"):
        lo, hi = _rho_iv(node.args[0], tr, m, dt, known, future, sub)
        g = op == "G"
        return _window(lo, w_lo, w_hi, g), _window(hi, w_lo, w_hi, g)
    if op == "U":
        a_lo, a_hi = _rho_iv(node.args[0], tr, m, dt, known, future, sub)
        b_lo, b_hi = _rho_iv(node.args[1], tr, m, dt, known, future, sub)
        return _until(a_lo, b_lo, w_lo, w_hi, need), _until(a_hi, b_hi, w_lo, w_hi, need)
    raise ValueError(f"unknown operator {op}")


def verdict(lo: float, hi: float) -> str:
    if lo > 0:
        return "satisfied"
    if hi <= 0:
        return "violated"
    return "undecided"


def interval(spec: Node, trace: Trace, m: Map | None = None, dt: float = 0.1,
             future: FutureBounds | None = None) -> tuple[float, float]:
    """Robust satisfaction interval of the spec at t = 0, given the flight so far
    (trace ends at "now")."""
    tr, known = _online_grid(spec, trace, dt)
    lo, hi = _rho_iv(spec, tr, m, dt, known, future)
    return lo[0], hi[0]


def _online_grid(spec: Node, trace: Trace, dt: float) -> tuple[Trace, int]:
    t_now = trace.t[-1]
    tr = trace.resampled(dt, max(t_now, spec.horizon()))
    return tr, int(math.floor(t_now / dt + 1e-9)) + 1


def score_online(spec_text: str | Node, trace: Trace, m: Map | None = None, dt: float = 0.1,
                 future: FutureBounds | None = None) -> dict:
    """Overall and per-clause intervals for the flight so far. Same clause split
    as score(), so the two reports line up."""
    spec = parse(spec_text) if isinstance(spec_text, str) else spec_text
    tr, known = _online_grid(spec, trace, dt)
    clauses = spec.args if spec.op == "and" else (spec,)

    def entry(node: Node) -> dict:
        lo, hi = _rho_iv(node, tr, m, dt, known, future)
        return {"clause": str(node), "lo": _round(lo[0]), "hi": _round(hi[0]),
                "verdict": verdict(lo[0], hi[0])}

    lo, hi = _rho_iv(spec, tr, m, dt, known, future)
    return {
        "t": round(trace.t[-1], 2),
        "lo": _round(lo[0]),
        "hi": _round(hi[0]),
        "verdict": verdict(lo[0], hi[0]),
        "clauses": [entry(c) for c in clauses],
    }
