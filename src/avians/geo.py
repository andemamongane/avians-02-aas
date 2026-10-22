"""Map regions in other tools' coordinates: Gazebo models and QGroundControl KML.

The map is metres east/north of the launch point. Gazebo's world frame is ENU
too, offset by where the drone spawns (`anchor`, the model's x, y). QGC wants
latitude/longitude: the launch point's (`origin`), from the Gazebo world's
<spherical_coordinates> or, for PX4 SIH, PX4's default home. Over the few
hundred metres of a mission a local flat-earth conversion is exact to
centimetres.

Region volumes in Gazebo are visual only (translucent, no collision), so the
recording shows them and the drone flies through them as it would through
lines on a map. `solid: true` on a region adds a collision as well: a real
obstacle the drone can hit.
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from xml.sax.saxutils import escape

from .maps import Map

EARTH_R = 6378137.0
PX4_SIH_HOME = (47.397742, 8.545594, 488.0)      # PX4 SITL default home (lat, lon, alt)
UNBOUNDED_HEIGHT = 30.0                          # m, drawn height of a region with no ceiling
RGBA = {"goal": (0.05, 0.64, 0.05), "no_fly": (0.82, 0.23, 0.23), "landing": (0.3, 0.3, 0.3),
        "corridor": (0.54, 0.36, 0.84), "other": (0.54, 0.53, 0.51)}


def world_origin(world_sdf: Path) -> tuple[float, float, float]:
    """(lat, lon, elevation) of a Gazebo world's origin."""
    text = Path(world_sdf).read_text()
    block = re.search(r"<spherical_coordinates>(.*?)</spherical_coordinates>", text, re.S)
    if not block:
        raise ValueError(f"{world_sdf} has no <spherical_coordinates>")
    get = lambda tag: float(re.search(rf"<{tag}>\s*([-\d.eE+]+)\s*</{tag}>", block.group(1)).group(1))
    return get("latitude_deg"), get("longitude_deg"), get("elevation")


def to_latlon(east: float, north: float, origin: tuple[float, float, float]) -> tuple[float, float]:
    lat0, lon0, _ = origin
    lat = lat0 + math.degrees(north / EARTH_R)
    lon = lon0 + math.degrees(east / (EARTH_R * math.cos(math.radians(lat0))))
    return lat, lon


# ---------------------------------------------------------------- Gazebo
def region_models(m: Map, anchor: tuple[float, float] = (0.0, 0.0)) -> dict[str, str]:
    """{model name: SDF} for every region, placed relative to the drone's spawn."""
    out = {}
    for rid, r in m.regions.items():
        floor, ceiling = r.altitude if r.altitude is not None else (0.0, UNBOUNDED_HEIGHT)
        red, green, blue = RGBA.get(r.kind, RGBA["other"])
        pts = "".join(f"<point>{e:.3f} {n:.3f}</point>" for e, n in r.polygon)
        geometry = f"<geometry><polyline>{pts}<height>{ceiling - floor:.3f}</height></polyline></geometry>"
        solid = r.solid
        collision = f"<collision name='collision'>{geometry}</collision>" if solid else ""
        alpha = 0.85 if solid else 0.3
        name = f"avians_region_{rid}"
        out[name] = (
            f"<?xml version='1.0'?><sdf version='1.9'><model name='{escape(name)}'>"
            f"<static>true</static><pose>{anchor[0]:.3f} {anchor[1]:.3f} {floor:.3f} 0 0 0</pose>"
            f"<link name='link'><visual name='visual'>{geometry}"
            f"<material><ambient>{red} {green} {blue} {alpha}</ambient>"
            f"<diffuse>{red} {green} {blue} {alpha}</diffuse></material>"
            f"<transparency>{1 - alpha:.2f}</transparency><cast_shadows>false</cast_shadows>"
            f"</visual>{collision}</link></model></sdf>")
    return out


def chase_camera_model(name: str = "avians_chase_cam", width: int = 1280, height: int = 720,
                       fps: int = 15, topic: str = "/avians/chase/image") -> str:
    """A free-floating camera the recorder moves behind the drone every frame."""
    return (
        f"<?xml version='1.0'?><sdf version='1.9'><model name='{name}'><static>true</static>"
        f"<pose>-8 0 6 0 0.45 0</pose><link name='link'><sensor name='camera' type='camera'>"
        f"<topic>{topic}</topic><update_rate>{fps}</update_rate><always_on>true</always_on>"
        f"<camera><horizontal_fov>1.2</horizontal_fov>"
        f"<image><width>{width}</width><height>{height}</height><format>R8G8B8</format></image>"
        f"<clip><near>0.1</near><far>500</far></clip></camera></sensor></link></model></sdf>")


# ---------------------------------------------------------------- QGroundControl
def kml(m: Map, origin: tuple[float, float, float], name: str = "avians map") -> str:
    """Regions as KML polygons (QGC: Plan view, import a KML/SHP overlay)."""
    style = "".join(
        f"<Style id='{kind}'><LineStyle><color>ff{_kml_hex(rgb)}</color><width>2</width>"
        f"</LineStyle><PolyStyle><color>55{_kml_hex(rgb)}</color></PolyStyle></Style>"
        for kind, rgb in RGBA.items())
    marks = []
    for rid, r in m.regions.items():
        ring = r.polygon + r.polygon[:1]
        coords = " ".join(f"{lon:.8f},{lat:.8f},0" for lat, lon in
                          (to_latlon(e, n, origin) for e, n in ring))
        marks.append(
            f"<Placemark><name>{escape(rid)}</name>"
            f"<description>{escape(f'{r.kind}: {r.name}. {r.description}'.strip())}</description>"
            f"<styleUrl>#{r.kind}</styleUrl><Polygon><outerBoundaryIs><LinearRing>"
            f"<coordinates>{coords}</coordinates></LinearRing></outerBoundaryIs></Polygon>"
            f"</Placemark>")
    return ("<?xml version='1.0' encoding='UTF-8'?>"
            "<kml xmlns='http://www.opengis.net/kml/2.2'><Document>"
            f"<name>{escape(name)}</name>{style}{''.join(marks)}</Document></kml>")


def _kml_hex(rgb) -> str:
    """KML colours are aabbggrr."""
    r, g, b = (int(round(c * 255)) for c in rgb)
    return f"{b:02x}{g:02x}{r:02x}"
