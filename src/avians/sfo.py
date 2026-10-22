"""Signal first-order logic (SFO): STL plus quantifiers over regions and times.

SFO (Bakhirkin, Ferrere, Henzinger, Nickovic 2018) adds first-order
quantification to STL: over time variables, so formulas can name the moments
they relate, and over data, so they can compare signal values taken at
different moments. This module implements the fragment the missions need, as
a superset of stl.py (every STL spec is an SFO spec, with the same score):

    forall r : no_fly. G[0,240] !in(r)              every region of a kind
    exists r in {survey_a, survey_b}. F[0,60] in(r) a listed set of regions
    forall t in [0, 240]. phi                       every moment in a window
    exists t2 in [t1, t1 + 30]. phi                 bounds: a number, or an
                                                    outer variable +- a number
    (G[0,5] in(a)) @ t                              an STL formula at time t
    alt@t2 >= alt@t1 - 3                            signals compared ACROSS
    t2 - t1 <= 30                                   times; linear terms in
                                                    alt speed east north, time
                                                    variables and numbers

    example: "after reaching field A, never drop more than 3 m below the
    altitude it was reached at":
      forall t1 in [0, 240]. (in(survey_a) @ t1 ->
          forall t2 in [t1, 240]. alt@t2 >= alt@t1 - 3)

A quantifier's body extends as far right as it can, as usual in logic:
`forall r : no_fly. A & B` quantifies both; write `(forall r : no_fly. A) & B`
to make A its own clause. A bare STL formula is evaluated at mission start
(t = 0), as in stl.py. STL
operators (G, F, U) take STL only; to quantify inside, put the quantifier
outside and use @. At most MAX_TIME_VARS nested time variables.

Robustness: quantifiers are min (forall) / max (exists) over their range,
the empty range giving +inf / -inf; a comparison `x <= y` scores y - x, in its
own units (metres, m/s, seconds), as STL atoms do.

Everything is evaluated in interval form, as stl.py's online semantics: with
the whole flight known the interval collapses to the offline score; during a
flight, unknown future samples are bounded (FutureBounds), so SFO specs get
the same early stop as STL ones.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

from .maps import Map
from .stl import (SIGNALS, FutureBounds, Node, STLSyntaxError, Trace, _Parser, _rho_iv,
                  _round, verdict)
import re

MAX_TIME_VARS = 2
SFO_OPS = ("at", "q_time", "q_region", "lin")
_TOKEN = re.compile(r"\s*(->|<=|>=|[()\[\],!&|<>~.@:{}+*\-]|[A-Za-z_][A-Za-z0-9_]*"
                    r"|\d+(?:\.\d*)?|\.\d+)")


# ---------------------------------------------------------------- AST
# SFO nodes reuse stl.Node:
#   at        args=(stl_formula,), signal=<time var>
#   q_time    args=(body,), cmp="forall"|"exists", signal=<var>, value=(lo, hi)
#             where lo/hi are (var | None, offset seconds)
#   q_region  args=(body,), cmp="forall"|"exists", signal=<var>,
#             region=<kind> or value=(id, id, ...) for an explicit set
#   lin       value=((term, coef), ...), cmp="<="|"<"|">="|">"  : expr cmp 0
#             term: ("sig", name, var) | ("time", var) | ("const",)
def to_text(n: Node) -> str:
    if n.op == "at":
        return f"({to_text(n.args[0])}) @ {n.signal}"
    if n.op == "q_time":
        (lv, lo), (hv, hi) = n.value
        b = lambda v, o: (f"{v} {'+' if o >= 0 else '-'} {abs(o):g}" if v and o else v or f"{o:g}")
        return f"{n.cmp} {n.signal} in [{b(lv, lo)}, {b(hv, hi)}]. {to_text(n.args[0])}"
    if n.op == "q_region":
        where = f": {n.region}" if n.region else " in {" + ", ".join(n.value) + "}"
        return f"{n.cmp} {n.signal}{where}. {to_text(n.args[0])}"
    if n.op == "lin":
        parts = []
        for term, c in n.value:
            name = {"sig": lambda t: f"{t[1]}@{t[2]}", "time": lambda t: t[1],
                    "const": lambda t: ""}[term[0]](term)
            parts.append(f"{c:g}" if not name else
                         (name if c == 1 else f"-{name}" if c == -1 else f"{c:g}*{name}"))
        return " + ".join(parts).replace("+ -", "- ") + f" {n.cmp} 0"
    if n.op == "not":
        return "!" + _paren(n.args[0])
    if n.op in ("and", "or"):
        return (" & " if n.op == "and" else " | ").join(_paren(a) for a in n.args)
    return str(n)


def _paren(n: Node) -> str:
    s = to_text(n)
    return f"({s})" if n.op in ("and", "or", "U", "q_time", "q_region", "lin") else s


def is_stl(n: Node) -> bool:
    return n.op not in SFO_OPS and all(is_stl(a) for a in n.args)


# ---------------------------------------------------------------- parser
class _SFOParser(_Parser):
    def __init__(self, text: str):
        self.toks = _tokenize(text)
        self.i = 0
        self.tvars: list[str] = []
        self.rvars: list[str] = []

    def number(self) -> float:
        if self.peek() == "-":
            self.take()
            return -super().number()
        return super().number()

    def unary(self) -> Node:
        if self.peek() in ("forall", "exists"):
            return self.quantifier()
        node = super().unary()
        if node.op in ("G", "F", "U") and not is_stl(node):
            raise STLSyntaxError("G, F and U take STL only; quantify outside them and use "
                                 "@: e.g. (G[0,5] in(a)) @ t")
        return node

    def quantifier(self) -> Node:
        q = self.take()
        var = self.take()
        if not var.isidentifier() or var in SIGNALS or var in ("in", "true", "G", "F", "U"):
            raise STLSyntaxError(f"bad variable name '{var}'")
        if self.peek() == ":":
            self.take()
            kind = self.take()
            self.take(".")
            return Node("q_region", (self._body(var, region=True),), cmp=q, signal=var,
                        region=kind)
        self.take("in")
        if self.peek() == "{":
            self.take()
            ids = [self.take()]
            while self.peek() == ",":
                self.take()
                ids.append(self.take())
            self.take("}")
            self.take(".")
            return Node("q_region", (self._body(var, region=True),), cmp=q, signal=var,
                        value=tuple(ids))
        if len(self.tvars) >= MAX_TIME_VARS:
            raise STLSyntaxError(f"at most {MAX_TIME_VARS} nested time variables")
        self.take("[")
        lo = self.time_bound()
        self.take(",")
        hi = self.time_bound()
        self.take("]")
        self.take(".")
        return Node("q_time", (self._body(var, region=False),), cmp=q, signal=var,
                    value=(lo, hi))

    def _body(self, var: str, region: bool) -> Node:
        scope = self.rvars if region else self.tvars
        scope.append(var)
        try:
            return self.implies()
        finally:
            scope.pop()

    def time_bound(self) -> tuple[str | None, float]:
        tok = self.peek()
        if tok in self.tvars:
            self.take()
            off = 0.0
            if self.peek() in ("+", "-"):
                sign = 1.0 if self.take() == "+" else -1.0
                off = sign * super().number()
            return tok, off
        if tok is not None and tok.isidentifier():
            raise STLSyntaxError(f"'{tok}' is not a time variable in scope")
        return None, self.number()

    def atom(self) -> Node:
        tok = self.peek()
        nxt = self.toks[self.i + 1] if self.i + 1 < len(self.toks) else None
        if (tok in SIGNALS and nxt == "@") or tok in self.tvars or (
                tok is not None and (tok[0].isdigit() or tok[0] == ".") and nxt in ("*", "+", "-")):
            return self.comparison()
        if tok == "in" and nxt == "(":
            self.take(), self.take("(")
            region = self.take()
            self.take(")")
            node = Node("in", region=region)
        else:
            node = super().atom()
        if self.peek() == "@":
            self.take()
            var = self.take()
            if var not in self.tvars:
                raise STLSyntaxError(f"'@ {var}': {var} is not a time variable in scope")
            node = Node("at", (node,), signal=var)
        return node

    def comparison(self) -> Node:
        lhs = self.linear()
        cmp = self.take()
        if cmp not in ("<", "<=", ">", ">="):
            raise STLSyntaxError(f"expected a comparison, got '{cmp}'")
        rhs = self.linear()
        terms: dict = {}
        for t, c in lhs:
            terms[t] = terms.get(t, 0.0) + c
        for t, c in rhs:
            terms[t] = terms.get(t, 0.0) - c
        return Node("lin", cmp=cmp, value=tuple((t, c) for t, c in terms.items() if c != 0))

    def linear(self) -> list:
        out = [self.term(1.0)]
        while self.peek() in ("+", "-"):
            sign = 1.0 if self.take() == "+" else -1.0
            out.append(self.term(sign))
        return out

    def term(self, sign: float):
        tok = self.peek()
        if tok == "-":
            self.take()
            return self.term(-sign)
        coef = 1.0
        if tok is not None and (tok[0].isdigit() or tok[0] == "."):
            coef = float(self.take())
            if self.peek() != "*":
                return ("const",), sign * coef
            self.take()
            tok = self.peek()
        if tok in SIGNALS:
            self.take()
            self.take("@")
            var = self.take()
            if var not in self.tvars:
                raise STLSyntaxError(f"'{tok}@{var}': {var} is not a time variable in scope")
            return ("sig", tok, var), sign * coef
        if tok in self.tvars:
            self.take()
            return ("time", tok), sign * coef
        raise STLSyntaxError(f"expected a signal@time, a time variable or a number, got '{tok}'")


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


def parse(text: str) -> Node:
    return _SFOParser(text).parse()


def regions(n: Node, bound: frozenset = frozenset()) -> set[str]:
    """Region ids the spec names (not region variables), for the map check."""
    if n.op == "in":
        return set() if n.region in bound else {n.region}
    if n.op == "q_region":
        listed = set(n.value) if n.value else set()
        return listed | regions(n.args[0], bound | {n.signal})
    out: set[str] = set()
    for a in n.args:
        out |= regions(a, bound)
    return out


def check_regions(spec: Node, m: Map | None) -> list[str]:
    known = set(m.regions) if m else set()
    return sorted(regions(spec) - known)


def horizon(n: Node, tmax: dict | None = None) -> float:
    """How far past t = 0 the spec looks, in seconds."""
    tmax = tmax or {}
    if n.op == "at":
        return tmax[n.signal] + n.args[0].horizon()
    if n.op == "q_time":
        var, off = n.value[1]
        return horizon(n.args[0], {**tmax, n.signal: (tmax[var] if var else 0.0) + off})
    if n.op == "lin":
        return max([tmax[t[2]] for t, _ in n.value if t[0] == "sig"]
                   + [tmax[t[1]] for t, _ in n.value if t[0] == "time"] + [0.0])
    if n.op == "q_region":
        return horizon(n.args[0], tmax)
    if is_stl(n):
        return n.horizon()
    return max(horizon(a, tmax) for a in n.args)


# ---------------------------------------------------------------- evaluation
@dataclass
class _Eval:
    tr: Trace
    m: Map | None
    dt: float
    known: int
    future: FutureBounds | None
    cache: dict = field(default_factory=dict)

    def arr(self, node: Node, renv: dict) -> tuple[list[float], list[float]]:
        """An STL formula's (lo, hi) robustness at every sample, region
        variables substituted."""
        key = (id(node), tuple(sorted(renv.items())))
        if key not in self.cache:
            self.cache[key] = _rho_iv(_subst(node, renv), self.tr, self.m, self.dt,
                                      self.known, self.future, need=len(self.tr.t))
        return self.cache[key]

    def signal(self, name: str, i: int) -> tuple[float, float]:
        lo, hi = self.arr(_SIG[name], {})
        return lo[i], hi[i]

    def idx(self, bound, tenv: dict) -> int:
        var, off = bound
        t = (tenv[var] * self.dt if var else 0.0) + off
        return int(round(t / self.dt))

    def ev(self, n: Node, tenv: dict, renv: dict) -> tuple[float, float]:
        op = n.op
        if op == "at":
            lo, hi = self.arr(n.args[0], renv)
            i = tenv[n.signal]
            return lo[i], hi[i]
        if op == "q_time":
            a = max(0, self.idx(n.value[0], tenv))
            b = min(len(self.tr.t) - 1, self.idx(n.value[1], tenv))
            body = n.args[0]
            if body.op == "at" and body.signal == n.signal:   # fast path: a slice
                if a > b:
                    return _agg(n.cmp, [])
                lo, hi = self.arr(body.args[0], renv)
                f = min if n.cmp == "forall" else max
                return f(lo[a:b + 1]), f(hi[a:b + 1])
            if body.op == "lin":                          # fast path: scan the signals
                return _agg(n.cmp, self._lin_range(body, n.signal, a, b, tenv))
            vals = [self.ev(body, {**tenv, n.signal: i}, renv) for i in range(a, b + 1)]
            return _agg(n.cmp, vals)
        if op == "q_region":
            ids = (list(n.value) if n.value else
                   [rid for rid, r in (self.m.regions.items() if self.m else []) if r.kind == n.region])
            vals = [self.ev(n.args[0], tenv, {**renv, n.signal: rid}) for rid in ids]
            return _agg(n.cmp, vals)
        if op == "lin":
            lo, hi = self._lin(n.value, tenv)
            # rho(expr <= 0) = -expr ; rho(expr >= 0) = expr
            return (-hi, -lo) if n.cmp in ("<", "<=") else (lo, hi)
        if op == "not":
            lo, hi = self.ev(n.args[0], tenv, renv)
            return -hi, -lo
        if op in ("and", "or") and not is_stl(n):
            parts = [self.ev(a, tenv, renv) for a in n.args]
            f = min if op == "and" else max
            return f(p[0] for p in parts), f(p[1] for p in parts)
        lo, hi = self.arr(n, renv)                     # plain STL: at mission start
        return lo[0], hi[0]

    def _lin(self, terms, tenv: dict) -> tuple[float, float]:
        """Interval of a linear expression (interval arithmetic)."""
        lo = hi = 0.0
        for term, c in terms:
            if term[0] == "const":
                tl = th = 1.0
            elif term[0] == "time":
                tl = th = tenv[term[1]] * self.dt
            else:
                tl, th = self.signal(term[1], tenv[term[2]])
            a, b = c * tl, c * th
            lo, hi = lo + min(a, b), hi + max(a, b)
        return lo, hi

    def _lin_range(self, n: Node, var: str, a: int, b: int, tenv: dict) -> list:
        """A comparison at every index of var in [a, b]: the terms that do not
        involve var are evaluated once."""
        fixed = [(t, c) for t, c in n.value if var not in t]
        moving = [(t, c) for t, c in n.value if var in t]
        f_lo, f_hi = self._lin(fixed, tenv)
        sig = {t[1]: self.arr(_SIG[t[1]], {}) for t, _ in moving if t[0] == "sig"}
        out = []
        for i in range(a, b + 1):
            lo, hi = f_lo, f_hi
            for t, c in moving:
                if t[0] == "time":
                    tl = th = i * self.dt
                else:
                    arr = sig[t[1]]
                    tl, th = arr[0][i], arr[1][i]
                x, y = c * tl, c * th
                lo, hi = lo + min(x, y), hi + max(x, y)
            out.append((-hi, -lo) if n.cmp in ("<", "<=") else (lo, hi))
        return out


_SIG = {s: Node("cmp", signal=s, cmp=">=", value=0.0) for s in SIGNALS}


def _agg(q: str, vals: list) -> tuple[float, float]:
    if not vals:
        return (math.inf, math.inf) if q == "forall" else (-math.inf, -math.inf)
    f = min if q == "forall" else max
    return f(v[0] for v in vals), f(v[1] for v in vals)


def _subst(n: Node, renv: dict) -> Node:
    if not renv:
        return n
    if n.op == "in" and n.region in renv:
        return replace(n, region=renv[n.region])
    if not n.args:
        return n
    return replace(n, args=tuple(_subst(a, renv) for a in n.args))


def _evaluate(spec: Node, trace: Trace, m, dt, future, online: bool):
    end = trace.t[-1]
    tr = trace.resampled(dt, max(end, horizon(spec)))
    known = int(math.floor(end / dt + 1e-9)) + 1 if online else len(tr.t)
    return _Eval(tr, m, dt, min(known, len(tr.t)), future if online else None)


def _clauses(spec: Node):
    return spec.args if spec.op == "and" else (spec,)


def robustness(spec: Node, trace: Trace, m: Map | None = None, dt: float = 0.1) -> float:
    return _evaluate(spec, trace, m, dt, None, online=False).ev(spec, {}, {})[0]


def score(spec_text: str, trace: Trace, m: Map | None = None, dt: float = 0.1) -> dict:
    """Same report as stl.score."""
    spec = parse(spec_text)
    e = _evaluate(spec, trace, m, dt, None, online=False)
    hz, end = horizon(spec), trace.t[-1]
    per = [(c, e.ev(c, {}, {})[0]) for c in _clauses(spec)]
    overall = min(r for _, r in per)                  # a top-level & is the min
    return {
        "spec": to_text(spec), "logic": "sfo",
        "robustness": _round(overall), "satisfied": overall > 0,
        "clauses": [{"clause": to_text(c), "robustness": _round(r), "satisfied": r > 0}
                    for c, r in per],
        "horizon_s": hz, "flight_end_s": round(end, 1),
        "padded_s": round(max(0.0, hz - end), 1), "dt_s": dt,
    }


def score_online(spec_text: str | Node, trace: Trace, m: Map | None = None, dt: float = 0.1,
                 future: FutureBounds | None = None) -> dict:
    """Same report as stl.score_online."""
    spec = parse(spec_text) if isinstance(spec_text, str) else spec_text
    e = _evaluate(spec, trace, m, dt, future, online=True)

    per = [(c, *e.ev(c, {}, {})) for c in _clauses(spec)]
    lo, hi = min(p[1] for p in per), min(p[2] for p in per)   # a top-level & is the min
    return {"t": round(trace.t[-1], 2), "lo": _round(lo), "hi": _round(hi),
            "verdict": verdict(lo, hi),
            "clauses": [{"clause": to_text(c), "lo": _round(l), "hi": _round(h),
                         "verdict": verdict(l, h)} for c, l, h in per]}
