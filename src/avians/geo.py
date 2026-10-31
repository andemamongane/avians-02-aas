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
GROUND_RGB = (0.86, 0.86, 0.86)                 # avians' plain ground
CAMERA_HFOV = 1.2                                # rad, both recording cameras


def _material(rgb: tuple[float, float, float], alpha: float = 1.0) -> str:
    r, g, b = rgb
    return (f"<material><ambient>{r:.3f} {g:.3f} {b:.3f} {alpha}</ambient>"
            f"<diffuse>{r:.3f} {g:.3f} {b:.3f} {alpha}</diffuse></material>"
            + (f"<transparency>{1 - alpha:.2f}</transparency>" if alpha < 1 else ""))


def plain_ground_model(name: str = "avians_ground", size_m: float = 400.0) -> str:
    """A flat, empty ground lit by a sun: the same collision and light as AAS's
    impalpable_greyness (both live in the scenery model it replaces), drawn
    plain so regions stand out."""
    return (
        f"<?xml version='1.0'?><sdf version='1.9'><model name='{name}'><static>true</static>"
        f"<link name='link'><collision name='collision'><pose>0 0 -5 0 0 0</pose>"
        f"<geometry><box><size>100000 100000 10</size></box></geometry></collision>"
        f"<visual name='visual'><geometry><plane><normal>0 0 1</normal>"
        f"<size>{size_m:g} {size_m:g}</size></plane></geometry>{_material(GROUND_RGB)}"
        f"</visual></link><link name='lighting_link'><light name='sun' type='directional'>"
        f"<pose>0 0 500 0 0 0</pose><cast_shadows>true</cast_shadows><intensity>1</intensity>"
        f"<direction>0.001 0.625 -0.78</direction><diffuse>0.904 0.904 0.904 1</diffuse>"
        f"<specular>0.271 0.271 0.271 1</specular></light></link></model></sdf>")


def region_models(m: Map, anchor: tuple[float, float] = (0.0, 0.0)) -> dict[str, str]:
    """{model name: SDF} for every region, placed relative to the drone's spawn.

    Drawn like the 2D map: a tinted patch on the ground with an outline in the
    region's colour, plus a faint volume between its floor and ceiling (so a
    survey field's altitude band shows as a box floating above its patch)."""
    out = {}
    for rid, r in m.regions.items():
        floor, ceiling = r.altitude if r.altitude is not None else (0.0, UNBOUNDED_HEIGHT)
        rgb = RGBA.get(r.kind, RGBA["other"])
        tint = tuple(0.45 * c + 0.55 * g for c, g in zip(rgb, GROUND_RGB))
        pts = "".join(f"<point>{e:.3f} {n:.3f}</point>" for e, n in r.polygon)
        poly = lambda h: f"<geometry><polyline>{pts}<height>{h:.3f}</height></polyline></geometry>"
        visuals = [
            f"<visual name='patch'><pose>0 0 0.01 0 0 0</pose>{poly(0.02)}{_material(tint)}"
            f"<cast_shadows>false</cast_shadows></visual>"]
        for k, ((e1, n1), (e2, n2)) in enumerate(zip(r.polygon, r.polygon[1:] + r.polygon[:1])):
            length, yaw = math.hypot(e2 - e1, n2 - n1), math.atan2(n2 - n1, e2 - e1)
            visuals.append(
                f"<visual name='edge_{k}'><pose>{(e1 + e2) / 2:.3f} {(n1 + n2) / 2:.3f} 0.035 "
                f"0 0 {yaw:.4f}</pose><geometry><box><size>{length + 0.25:.3f} 0.25 0.05</size>"
                f"</box></geometry>{_material(rgb)}<cast_shadows>false</cast_shadows></visual>")
        visuals.append(
            f"<visual name='volume'><pose>0 0 {floor:.3f} 0 0 0</pose>{poly(ceiling - floor)}"
            f"{_material(rgb, 0.85 if r.solid else 0.12)}<cast_shadows>false</cast_shadows></visual>")
        collision = (f"<collision name='collision'><pose>0 0 {floor:.3f} 0 0 0</pose>"
                     f"{poly(ceiling - floor)}</collision>" if r.solid else "")
        name = f"avians_region_{rid}"
        out[name] = (
            f"<?xml version='1.0'?><sdf version='1.9'><model name='{escape(name)}'>"
            f"<static>true</static><pose>{anchor[0]:.3f} {anchor[1]:.3f} 0 0 0 0</pose>"
            f"<link name='link'>{''.join(visuals)}{collision}</link></model></sdf>")
    return out


def _camera_model(name: str, pose: str, width: int, height: int, fps: int, topic: str) -> str:
    return (
        f"<?xml version='1.0'?><sdf version='1.9'><model name='{name}'><static>true</static>"
        f"<pose>{pose}</pose><link name='link'><sensor name='camera' type='camera'>"
        f"<topic>{topic}</topic><update_rate>{fps}</update_rate><always_on>true</always_on>"
        f"<camera><horizontal_fov>{CAMERA_HFOV}</horizontal_fov>"
        f"<image><width>{width}</width><height>{height}</height><format>R8G8B8</format></image>"
        f"<clip><near>0.1</near><far>500</far></clip></camera></sensor></link></model></sdf>")


def chase_camera_model(name: str = "avians_chase_cam", width: int = 1280, height: int = 720,
                       fps: int = 15, topic: str = "/avians/chase/image") -> str:
    """A free-floating camera the recorder moves behind the drone every frame."""
    return _camera_model(name, "-8 0 6 0 0.45 0", width, height, fps, topic)


def overview_camera_model(m: Map | None, anchor: tuple[float, float] = (0.0, 0.0),
                          name: str = "avians_overview_cam", width: int = 1280,
                          height: int = 720, fps: int = 15, topic: str = "/avians/overview/image",
                          pitch: float = 1.05) -> str:
    """A fixed camera south of the map looking north and down (`pitch` rad), far
    enough back that every region and the launch point are in frame: the
    ground-track panel of `avians video`, in 3D."""
    pts = [(0.0, 0.0)] + [p for r in (m.regions.values() if m else []) for p in r.polygon]
    es, ns = [p[0] for p in pts], [p[1] for p in pts]
    if max(es) - min(es) < 1 and max(ns) - min(ns) < 1:   # no map: 40 m around the pad
        es, ns = [-20.0, 20.0], [-20.0, 20.0]
    ce, cn = (min(es) + max(es)) / 2, (min(ns) + max(ns)) / 2
    w, h = 1.15 * (max(es) - min(es)) + 4, 1.15 * (max(ns) - min(ns)) + 4
    half_h = math.tan(CAMERA_HFOV / 2)
    half_v = half_h * height / width
    dist = max(w / 2 / half_h, h * math.sin(pitch) / 2 / half_v)
    x = anchor[0] + ce
    y = anchor[1] + cn - dist * math.cos(pitch)
    z = dist * math.sin(pitch)
    return _camera_model(name, f"{x:.3f} {y:.3f} {z:.3f} 0 {pitch:.4f} {math.pi / 2:.4f}",
                         width, height, fps, topic)


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
