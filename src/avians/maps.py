"""Mission maps: named polygon regions the agent and the STL spec refer to.

A map is a YAML file of regions. Coordinates are metres EAST and NORTH of the
launch point (the same frame as the tool API and AAS's set_reposition), so a
vertex [20, 30] is 20 m east and 30 m north of where the drone took off.

    regions:
      - id: survey_a
        name: Survey field A
        kind: goal                  # goal | no_fly | corridor | landing | other
        description: Open field. Inspect it from 15 m or lower.
        polygon: [[20, 30], [45, 30], [45, 55], [20, 55]]
        altitude: [5, 15]           # optional floor/ceiling in metres

Regions are INFORMATION for the agent, never enforced: the safety layer knows
nothing about them. Whether the agent stays out of a no-fly region is exactly
what the STL score measures.

Pure Python on purpose (no shapely): polygons here are small and simple, and
the tests must run on a laptop without extra dependencies.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import yaml

Point = tuple[float, float]          # (east, north)

KINDS = ("goal", "no_fly", "corridor", "landing", "other")


@dataclass
class Region:
    id: str
    name: str
    kind: str
    description: str
    polygon: list[Point]
    altitude: tuple[float, float] | None = None      # (floor, ceiling)

    # ------------------------------------------------------------ geometry
    def contains_xy(self, e: float, n: float) -> bool:
        """Ray casting. Points exactly on an edge count as inside."""
        if _on_boundary(self.polygon, (e, n)):
            return True
        inside = False
        pts = self.polygon
        for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
            if (y1 > n) != (y2 > n):
                x_cross = x1 + (n - y1) * (x2 - x1) / (y2 - y1)
                if e < x_cross:
                    inside = not inside
        return inside

    def signed_distance_xy(self, e: float, n: float) -> float:
        """Distance to the boundary in metres: positive inside, negative outside."""
        d = min(_dist_to_segment((e, n), a, b) for a, b in self._edges())
        return d if self.contains_xy(e, n) else -d

    def robustness(self, e: float, n: float, alt: float | None = None) -> float:
        """STL robustness of `in(region)`: how far inside (+) or outside (-).

        With an altitude band the region is a prism, and being inside means
        inside the footprint AND between floor and ceiling, so the robustness
        is the minimum of the three margins (the usual STL conjunction)."""
        r = self.signed_distance_xy(e, n)
        if self.altitude is not None and alt is not None:
            floor, ceiling = self.altitude
            r = min(r, alt - floor, ceiling - alt)
        return r

    def contains(self, e: float, n: float, alt: float | None = None) -> bool:
        return self.robustness(e, n, alt) >= 0

    def centroid(self) -> Point:
        """Area centroid (falls back to the vertex mean for degenerate shapes)."""
        a = cx = cy = 0.0
        for (x1, y1), (x2, y2) in self._edges():
            cross = x1 * y2 - x2 * y1
            a += cross
            cx += (x1 + x2) * cross
            cy += (y1 + y2) * cross
        if abs(a) < 1e-9:
            xs, ys = zip(*self.polygon)
            return sum(xs) / len(xs), sum(ys) / len(ys)
        return cx / (3 * a), cy / (3 * a)

    def area(self) -> float:
        return abs(sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in self._edges())) / 2

    def interior_point(self) -> Point:
        """A point well inside the region, for goto_region.

        The centroid when it is comfortably inside (convex shapes); otherwise
        the deepest point of a grid over the bounding box (L- or U-shapes,
        whose centroid can fall outside)."""
        c = self.centroid()
        xs, ys = zip(*self.polygon)
        span = min(max(xs) - min(xs), max(ys) - min(ys))
        if self.signed_distance_xy(*c) >= 0.25 * span:
            return c
        best, best_d = c, self.signed_distance_xy(*c)
        steps = 40
        for i in range(steps + 1):
            for j in range(steps + 1):
                p = (min(xs) + (max(xs) - min(xs)) * i / steps,
                     min(ys) + (max(ys) - min(ys)) * j / steps)
                d = self.signed_distance_xy(*p)
                if d > best_d:
                    best, best_d = p, d
        return best

    def crossed_by(self, a: Point, b: Point) -> bool:
        """Does the straight segment a->b touch this region's footprint?"""
        if self.contains_xy(*a) or self.contains_xy(*b):
            return True
        return any(_segments_intersect(a, b, p, q) for p, q in self._edges())

    def _edges(self):
        pts = self.polygon
        return list(zip(pts, pts[1:] + pts[:1]))

    # ------------------------------------------------------------ text
    def describe(self) -> str:
        """One region as the agent reads it in the system prompt."""
        ce, cn = self.centroid()
        verts = ", ".join(f"({e:g}, {n:g})" for e, n in self.polygon)
        alt = (f"; altitude {self.altitude[0]:g}-{self.altitude[1]:g} m"
               if self.altitude else "")
        return (f"- {self.id} [{self.kind}] \"{self.name}\": {self.description}\n"
                f"  vertices (east, north): {verts}\n"
                f"  centre ({ce:.1f}, {cn:.1f}), area {self.area():.0f} m^2{alt}")


@dataclass
class Map:
    name: str
    regions: dict[str, Region]
    description: str = ""

    @classmethod
    def load(cls, path: str | Path) -> "Map":
        path = Path(path)
        return cls.from_dict(yaml.safe_load(path.read_text()), default_name=path.stem)

    @classmethod
    def from_dict(cls, d: dict, default_name: str = "map") -> "Map":
        regions: dict[str, Region] = {}
        for i, r in enumerate(d.get("regions", [])):
            rid = str(r.get("id") or f"region_{i}")
            if rid in regions:
                raise ValueError(f"duplicate region id '{rid}'")
            poly = [(float(p[0]), float(p[1])) for p in r["polygon"]]
            if len(poly) >= 2 and poly[0] == poly[-1]:
                poly = poly[:-1]                     # accept closed rings
            if len(poly) < 3:
                raise ValueError(f"region '{rid}' needs at least 3 vertices")
            kind = r.get("kind", "other")
            if kind not in KINDS:
                raise ValueError(f"region '{rid}': kind must be one of {KINDS}")
            alt = r.get("altitude")
            regions[rid] = Region(
                id=rid, name=r.get("name", rid), kind=kind,
                description=r.get("description", ""), polygon=poly,
                altitude=(float(alt[0]), float(alt[1])) if alt else None,
            )
        return cls(name=d.get("name", default_name), regions=regions,
                   description=d.get("description", ""))

    def get(self, region_id: str) -> Region | None:
        return self.regions.get(region_id)

    def inside(self, e: float, n: float, alt: float | None = None) -> list[str]:
        return [r.id for r in self.regions.values() if r.contains(e, n, alt)]

    def crossings(self, a: Point, b: Point) -> list[str]:
        return [r.id for r in self.regions.values() if r.crossed_by(a, b)]

    def describe(self) -> str:
        head = f"Map \"{self.name}\"" + (f": {self.description}" if self.description else "")
        return "\n".join([head] + [r.describe() for r in self.regions.values()])


# ---------------------------------------------------------------- helpers
def _dist_to_segment(p: Point, a: Point, b: Point) -> float:
    (px, py), (ax, ay), (bx, by) = p, a, b
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _on_boundary(poly: list[Point], p: Point, eps: float = 1e-9) -> bool:
    return any(_dist_to_segment(p, a, b) <= eps for a, b in zip(poly, poly[1:] + poly[:1]))


def _orient(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> bool:
    d1, d2 = _orient(q1, q2, p1), _orient(q1, q2, p2)
    d3, d4 = _orient(p1, p2, q1), _orient(p1, p2, q2)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)) and d1 and d2 and d3 and d4:
        return True
    # collinear / touching cases
    return any(abs(d) < 1e-12 and _dist_to_segment(pt, s, e) < 1e-9
               for d, pt, s, e in ((d1, p1, q1, q2), (d2, p2, q1, q2),
                                   (d3, q1, p1, p2), (d4, q2, p1, p2)))
