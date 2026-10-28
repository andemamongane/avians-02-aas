"""What the pilot knows of the map: a range-limited sensor reveals regions.

A mission can hide some of its map's regions:

    sensor:
      range_m: 30
      hidden: [wall_1, wall_2, wall_3]

Hidden regions are not in the pilot's map at the start. Once the drone comes
within range_m of one (horizontally: its distance to the polygon), it is
discovered: it appears on the pilot's map from then on, and the next
observation reports the discovery as an event. Information then arrives
through observations during the flight, which is what makes deciding from the
current observation worthwhile. It is a stand-in for camera and lidar.

SensedMap quacks like maps.Map for everything the pilot side uses (tools,
planner, executor). Scoring always uses the full map: the spec does not
change because the pilot has not seen a region yet.
"""
from __future__ import annotations

from .maps import Map, Region


class SensedMap:
    def __init__(self, full: Map, hidden: list[str], range_m: float):
        unknown = set(hidden) - set(full.regions)
        if unknown:
            raise ValueError(f"sensor.hidden names regions the map lacks: {sorted(unknown)}")
        if range_m <= 0:
            raise ValueError("sensor.range_m must be positive")
        self.full = full
        self.name, self.description = full.name, full.description
        self.hidden = set(hidden)
        self.range_m = float(range_m)
        self.discovered: dict[str, float] = {}       # region id -> mission time

    @property
    def regions(self) -> dict[str, Region]:
        return {k: r for k, r in self.full.regions.items()
                if k not in self.hidden or k in self.discovered}

    def get(self, region_id: str) -> Region | None:
        return self.regions.get(region_id)

    def inside(self, e: float, n: float, alt: float | None = None) -> list[str]:
        return [r.id for r in self.regions.values() if r.contains(e, n, alt)]

    def crossings(self, a, b) -> list[str]:
        known = self.regions
        return [rid for rid in self.full.crossings(a, b) if rid in known]

    def describe(self) -> str:
        return Map(name=self.name, regions=self.regions, description=self.description).describe()

    def sense(self, e: float, n: float, t: float) -> list[Region]:
        """Regions newly within range of (e, n) at mission time t."""
        found = []
        for rid in sorted(self.hidden - set(self.discovered)):
            r = self.full.regions[rid]
            if -r.signed_distance_xy(e, n) <= self.range_m:
                self.discovered[rid] = t
                found.append(r)
        return found


def sensed_map(mission: dict, full: Map | None):
    """The pilot's map for a mission: the full map, or a SensedMap."""
    cfg = mission.get("sensor")
    if not cfg or full is None:
        return full
    return SensedMap(full, list(cfg.get("hidden", [])), float(cfg.get("range_m", 30.0)))
