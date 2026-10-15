"""Slide-ready matplotlib figures for one episode.

Every figure is 16:9 (13.33 x 7.5 in) so it drops onto a slide unscaled; at the
default 200 dpi that is 2667 x 1500 px. Styling is deliberately plain: one
colour for what the drone did, one for what it was told to do, status colours
only for tool-call outcomes, and hairline grids.

Tool calls are numbered in order (#1, #2, ...) and the same number is used on
every figure, so a waypoint on the ground track can be matched to its bar on
the agent timeline.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                     # headless: the container has no display
import matplotlib.pyplot as plt           # noqa: E402
import numpy as np                        # noqa: E402
from matplotlib.lines import Line2D       # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

from .flight import FlightData, ToolCall  # noqa: E402

# --- palette (validated reference palette, light mode) ---------------------
SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
ACTUAL = "#2a78d6"        # series 1: what the drone did
COMMANDED = "#eb6834"     # series 2: what it was told to do
STATUS = {                # reserved status colours, always paired with a text label
    "arrived": "#0ca30c", "ok": "#0ca30c",
    "timeout": "#fab219",
    "aborted": "#ec835a",
    "rejected": "#d03b3b", "blocked": "#d03b3b", "failed": "#d03b3b",
}
THINKING = AXIS
REGION = {                # map regions, by kind (fill is drawn translucent)
    "goal": "#0ca30c", "landing": INK_2, "no_fly": "#d03b3b",
    "corridor": "#8a5cd6", "other": MUTED,
}

FIGSIZE = (13.333, 7.5)
MIN_SPAN = 6.0            # metres: smallest horizontal window on map views

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 14,
    "axes.titlesize": 17,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.titlepad": 12,
    "axes.labelsize": 14,
    "axes.labelcolor": INK_2,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 1.0,
    "axes.facecolor": SURFACE,
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.frameon": False,
    "legend.fontsize": 13,
    "figure.facecolor": SURFACE,
    "text.color": INK,
    "lines.linewidth": 2.0,
    "savefig.facecolor": SURFACE,
})


# --- helpers ---------------------------------------------------------------
def call_label(i: int, c: ToolCall) -> str:
    a = c.args
    if c.tool == "takeoff":
        body = f"takeoff to {a.get('altitude', 0):g} m"
    elif c.tool == "reposition":
        body = (f"reposition (E {a.get('east', 0):g}, N {a.get('north', 0):g}, "
                f"{a.get('altitude', 0):g} m)")
    elif c.tool == "goto_region":
        body = f"goto_region {a.get('region', '?')} at {a.get('altitude', 0):g} m"
    elif c.tool == "orbit":
        body = f"orbit r={a.get('radius', 0):g} m for {a.get('duration', 0):g} s"
    elif c.tool == "check_path":
        body = f"check_path to (E {a.get('east', 0):g}, N {a.get('north', 0):g})"
    elif c.tool == "hover":
        body = f"hover {a.get('duration', 0):g} s"
    else:
        body = c.tool
    return f"#{i}  {body}"


def _caption(fig, f: FlightData) -> None:
    m = f.meta
    parts = [f.mission, f.episode, m.get("model", ""),
             f"backend: {m.get('backend', '?')}", f"data: {f.source}"]
    fig.text(0.012, 0.012, "  ·  ".join(p for p in parts if p),
             color=MUTED, fontsize=11, ha="left", va="bottom")


def _line_kw(f: FlightData) -> dict:
    """Dense bag data draws as a line; sparse per-step samples get markers too."""
    return {} if f.dense else {"marker": "o", "markersize": 6}


def _mark_calls(ax, f: FlightData, label: bool = True) -> None:
    """A hairline at the start of every flight action, numbered at the top."""
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    t_end = max(f.t[-1], f.calls[-1].t_end if f.calls else 0.0, 1.0)
    last_t, raised = -np.inf, False
    for i, c in enumerate(f.calls, 1):
        if c.tool == "get_state":
            continue
        colour = STATUS["rejected"] if c.status in ("rejected", "blocked") else AXIS
        ax.axvline(c.t_start, color=colour, lw=1.0, zorder=0)
        if label:
            # Neighbours closer than ~3% of the axis alternate between two rows.
            raised = (not raised) if c.t_start - last_t < 0.03 * t_end else False
            last_t = c.t_start
            ax.text(c.t_start, 1.07 if raised else 1.0, f"#{i}", transform=trans,
                    fontsize=11, color=INK_2, ha="center", va="bottom", clip_on=False)


def _commanded_targets(f: FlightData) -> list[tuple[int, ToolCall]]:
    return [(i, c) for i, c in enumerate(f.calls, 1)
            if c.tool in ("takeoff", "reposition", "goto_region") and c.target is not None
            and c.status not in ("rejected", "blocked")]


def _rejected_targets(f: FlightData) -> list[tuple[int, ToolCall]]:
    return [(i, c) for i, c in enumerate(f.calls, 1)
            if c.tool in ("reposition", "goto_region", "orbit") and c.target is not None
            and c.status in ("rejected", "blocked")]


def _orbits(f: FlightData) -> list[tuple[int, ToolCall]]:
    return [(i, c) for i, c in enumerate(f.calls, 1)
            if c.tool == "orbit" and c.target is not None
            and c.status not in ("rejected", "blocked")]


# --- individual plots (draw into a given axes) ------------------------------
def draw_regions(ax, f: FlightData) -> tuple[list[float], list[float]]:
    """Map regions as translucent polygons, labelled by id. Returns their extent."""
    xs, ys, seen = [], [], set()
    if f.map is None:
        return xs, ys
    for r in f.map.regions.values():
        colour = REGION.get(r.kind, MUTED)
        e = [p[0] for p in r.polygon] + [r.polygon[0][0]]
        n = [p[1] for p in r.polygon] + [r.polygon[0][1]]
        label = r.kind.replace("_", "-") + " region" if r.kind not in seen else None
        seen.add(r.kind)
        ax.fill(e, n, color=colour, alpha=0.12, zorder=0, lw=0)
        ax.plot(e, n, color=colour, lw=1.4, zorder=1, label=label,
                linestyle="--" if r.kind == "no_fly" else "-")
        # Label just above the top edge, clear of waypoints placed inside.
        ax.text(sum(e[:-1]) / len(r.polygon), max(n), r.id, color=colour, fontsize=12,
                ha="center", va="bottom", zorder=2)
        xs += e
        ys += n
    return xs, ys


def draw_ground_track(ax, f: FlightData, geofence_m: float | None) -> None:
    # East on the horizontal axis and north up, like a map.
    rx, ry = draw_regions(ax, f)
    for n, (i, c) in enumerate(_orbits(f)):
        cn, ce, _ = c.target
        r = c.args.get("radius", 0.0)
        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(ce + r * np.cos(theta), cn + r * np.sin(theta), color=COMMANDED, lw=1.2,
                linestyle=":", zorder=2, label="Commanded orbit" if n == 0 else None)
        ax.annotate(f"#{i}", (ce, cn), fontsize=12, color=INK_2, ha="center")
    ax.plot(f.y, f.x, color=ACTUAL, label="Flown path", zorder=3, **_line_kw(f))
    ax.plot(0, 0, marker="s", color=INK, markersize=10, zorder=5, linestyle="none",
            label="Launch")
    ax.plot(f.y[-1], f.x[-1], marker="v", color=ACTUAL, markersize=11, zorder=5,
            markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none", label="Landed")

    # Labels at a repeated (x, y), e.g. takeoff and return-to-launch, share one
    # line ("#1, #9") instead of printing on top of each other.
    labels: list[tuple[float, float, list[str]]] = []
    for n, (i, c) in enumerate(_commanded_targets(f)):
        tx, ty, _ = c.target
        ax.plot(ty, tx, marker="o", markersize=11, color=COMMANDED, zorder=4,
                markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none",
                label="Commanded waypoint" if n == 0 else None)
        near = next((l for l in labels if np.hypot(l[0] - tx, l[1] - ty) < 1.0), None)
        if near:
            near[2].append(f"#{i}")
        else:
            labels.append((tx, ty, [f"#{i}"]))
    for tx, ty, names in labels:
        ax.annotate(", ".join(names), (ty, tx), xytext=(10, 6), textcoords="offset points",
                    fontsize=12, color=INK_2)

    rejected = _rejected_targets(f)
    for n, (i, c) in enumerate(rejected):
        tx, ty, _ = c.target
        ax.plot(ty, tx, marker="X", markersize=13, color=STATUS["rejected"], zorder=4,
                markeredgecolor=SURFACE, linestyle="none",
                label="Rejected by safety layer" if n == 0 else None)
        ax.annotate(f"#{i} rejected", (ty, tx), xytext=(10, 6), textcoords="offset points",
                    fontsize=12, color=INK_2)
    if rejected and geofence_m:
        theta = np.linspace(0, 2 * np.pi, 400)
        ax.plot(geofence_m * np.sin(theta), geofence_m * np.cos(theta), color=MUTED,
                lw=1.2, zorder=1, label=f"Geofence ({geofence_m:g} m)")

    # Frame on the flight (plus rejected targets), not the whole geofence.
    xs = np.concatenate([f.y[np.isfinite(f.y)], [0.0], rx] + [[c.target[1]] for _, c in rejected]
                        + [[c.target[1]] for _, c in _commanded_targets(f)])
    ys = np.concatenate([f.x[np.isfinite(f.x)], [0.0], ry] + [[c.target[0]] for _, c in rejected]
                        + [[c.target[0]] for _, c in _commanded_targets(f)])
    # Square window, at least MIN_SPAN wide, so a hover doesn't zoom to centimetres.
    half = 0.56 * max(np.ptp(xs), np.ptp(ys), MIN_SPAN)
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")


def draw_altitude(ax, f: FlightData) -> None:
    ax.plot(f.sp_t, f.sp_z, color=COMMANDED, lw=1.6, label="Commanded", drawstyle="steps-post")
    ax.plot(f.t, f.z, color=ACTUAL, label="Actual", **_line_kw(f))
    _mark_calls(ax, f)
    ax.set_xlabel("Mission time (s)")
    ax.set_ylabel("Altitude above launch (m)")
    ax.set_ylim(bottom=min(0.0, np.nanmin(f.z)))
    ax.set_xlim(0, max(f.t[-1], f.calls[-1].t_end if f.calls else 0))


def draw_timeline(ax, f: FlightData, reasons: bool = True) -> None:
    """Gantt of the agent's decisions: model thinking on top, one row per tool call."""
    rows = [call_label(i, c) for i, c in enumerate(f.calls, 1)]
    n = len(rows)
    h = 0.62

    for a, b in f.thinking:
        ax.barh(0, b - a, left=a, height=h, color=THINKING, edgecolor=SURFACE, linewidth=2)
    for k, c in enumerate(f.calls, 1):
        colour = STATUS.get(c.status, MUTED)
        width = max(c.t_end - c.t_start, 0.25)       # instant calls still get a visible tick
        ax.barh(k, width, left=c.t_start, height=h, color=colour,
                edgecolor=SURFACE, linewidth=2)
        reason = c.reason.split(";")[0]          # "outside the 50 m geofence; nearest..."
        note = c.status + (f": {textwrap.shorten(reason, 64)}" if reasons and reason
                           and c.status not in ("arrived", "ok") else "")
        ax.text(c.t_start + width + 0.6, k, note, va="center", fontsize=11, color=INK_2)

    ax.set_yticks(range(n + 1), ["model thinking"] + rows)
    ax.set_ylim(n + 0.7, -0.7)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Mission time (s)")
    end = max([c.t_end for c in f.calls] + [b for _, b in f.thinking] + [1.0])
    ax.set_xlim(0, end * 1.18)                   # room for the status labels

    handles = [Line2D([], [], color=THINKING, lw=8, label="Model thinking")]
    statuses = list(dict.fromkeys(c.status for c in f.calls))
    for status in statuses:
        if status == "ok" and "arrived" in statuses:
            continue                             # same colour: one entry
        name = "arrived / ok" if status == "arrived" and "ok" in statuses else status
        handles.append(Line2D([], [], color=STATUS.get(status, MUTED), lw=8, label=name))
    ax.legend(handles=handles, loc="lower right", ncols=len(handles), fontsize=12,
              bbox_to_anchor=(1.0, 1.0), borderaxespad=0.3)


# --- whole figures -----------------------------------------------------------
def fig_overview(f: FlightData, metrics: dict, geofence_m: float | None):
    fig = plt.figure(figsize=FIGSIZE, layout="constrained")
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.35])
    ax_track = fig.add_subplot(gs[:, 0])
    ax_alt = fig.add_subplot(gs[0, 1])
    ax_tl = fig.add_subplot(gs[1, 1])

    draw_ground_track(ax_track, f, geofence_m)
    ax_track.set_title("Ground track")
    ax_track.legend(loc="upper left", bbox_to_anchor=(0.0, -0.13), ncols=3, fontsize=11,
                    columnspacing=1.2, handletextpad=0.5)
    draw_altitude(ax_alt, f)
    ax_alt.set_title("Altitude", pad=22)
    ax_alt.legend(loc="upper right", fontsize=11)
    draw_timeline(ax_tl, f, reasons=False)      # narrow panel: full reasons are on 06
    ax_tl.set_title("Agent decisions", pad=26)
    ax_tl.tick_params(axis="y", labelsize=10)
    ax_tl.get_legend().remove()

    verdict = "PASS" if metrics["success"] else "CHECK"
    fig.suptitle(f"{f.mission}: mission {verdict}", x=0.012, ha="left",
                 fontsize=22, fontweight="bold")
    fig.text(0.012, 0.935, headline(metrics), ha="left", va="top", fontsize=13, color=INK_2,
             transform=fig.transFigure)
    fig.get_layout_engine().set(rect=(0, 0.045, 1, 0.9))
    _caption(fig, f)
    return fig


def fig_ground_track(f: FlightData, geofence_m: float | None):
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    draw_ground_track(ax, f, geofence_m)
    ax.set_title(f"Ground track: {f.mission}")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.get_layout_engine().set(rect=(0, 0.04, 1, 0.96))
    _caption(fig, f)
    return fig


def fig_trajectory_3d(f: FlightData):
    fig = plt.figure(figsize=FIGSIZE, layout="constrained")
    ax = fig.add_subplot(projection="3d")
    ax.plot(f.y, f.x, f.z, color=ACTUAL, label="Flown path", **_line_kw(f))
    ax.plot(f.y, f.x, np.zeros_like(f.z), color=AXIS, lw=1.2, label="Shadow on ground")
    for i, c in _commanded_targets(f):
        tx, ty, tz = c.target
        ax.scatter(ty, tx, tz, s=90, color=COMMANDED, edgecolors=SURFACE, linewidths=1.5,
                   depthshade=False)
        ax.text(ty, tx, tz + 0.4, f"#{i}", fontsize=11, color=INK_2)
    ax.scatter(0, 0, 0, s=90, marker="s", color=INK, label="Launch", depthshade=False)
    ax.plot([], [], marker="o", color=COMMANDED, linestyle="none", label="Commanded waypoint")
    ax.set_xlabel("East (m)", labelpad=12)
    ax.set_ylabel("North (m)", labelpad=12)
    ax.set_zlabel("Altitude (m)", labelpad=10)
    # Same metres-per-unit horizontally, at least MIN_SPAN wide; altitude gets its
    # own scale (stretched up to half the width) so climbs stay visible.
    e, n = f.y[np.isfinite(f.y)], f.x[np.isfinite(f.x)]
    half = 0.55 * max(np.ptp(e), np.ptp(n), MIN_SPAN)
    ce, cn = (e.min() + e.max()) / 2, (n.min() + n.max()) / 2
    ax.set_xlim(ce - half, ce + half)
    ax.set_ylim(cn - half, cn + half)
    top = max(np.nanmax(f.z), 1.0) * 1.15
    ax.set_zlim(0, top)
    ax.set_box_aspect((1, 1, float(np.clip(top / (2 * half), 0.35, 0.8))))
    ax.view_init(elev=24, azim=-60)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor(SURFACE)
        pane.set_edgecolor(GRID)
    fig.suptitle(f"3D trajectory: {f.mission}", x=0.012, ha="left",
                 fontsize=17, fontweight="bold")
    ax.legend(loc="upper left", bbox_to_anchor=(-0.15, 0.95))
    fig.get_layout_engine().set(rect=(0, 0.04, 1, 0.96))
    _caption(fig, f)
    return fig


def fig_position_tracking(f: FlightData):
    fig, axes = plt.subplots(3, 1, figsize=FIGSIZE, sharex=True, layout="constrained")
    series = [("North, x (m)", f.x, f.sp_x), ("East, y (m)", f.y, f.sp_y),
              ("Altitude, z (m)", f.z, f.sp_z)]
    for k, (ax, (label, actual, cmd)) in enumerate(zip(axes, series)):
        ax.plot(f.sp_t, cmd, color=COMMANDED, lw=1.6, label="Commanded",
                drawstyle="steps-post")
        ax.plot(f.t, actual, color=ACTUAL, label="Actual", **_line_kw(f))
        _mark_calls(ax, f, label=(k == 0))
        ax.set_ylabel(label)
    axes[0].set_title("Commanded vs actual position", pad=22)
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    axes[-1].set_xlabel("Mission time (s)")
    axes[-1].set_xlim(0, f.t[-1])
    fig.get_layout_engine().set(rect=(0, 0.04, 1, 0.96))
    _caption(fig, f)
    return fig


def fig_speed_battery(f: FlightData):
    # Two panels, one y-scale each: never a dual axis.
    fig, (ax_s, ax_b) = plt.subplots(2, 1, figsize=FIGSIZE, sharex=True, layout="constrained")
    ax_s.plot(f.t, f.speed, color=ACTUAL, **_line_kw(f))
    _mark_calls(ax_s, f)
    ax_s.set_ylabel("Speed (m/s)")
    ax_s.set_ylim(bottom=0)
    ax_s.set_title("Ground speed", pad=22)
    i_max = int(np.nanargmax(f.speed))
    ax_s.annotate(f"peak {f.speed[i_max]:.1f} m/s", (f.t[i_max], f.speed[i_max]),
                  xytext=(10, -2), textcoords="offset points", va="top",
                  fontsize=12, color=INK_2)

    if len(f.bat_t):
        ax_b.plot(f.bat_t, f.bat_pct, color=ACTUAL, **_line_kw(f))
        ax_b.annotate(f"{f.bat_pct[-1]:.0f}%", (f.bat_t[-1], f.bat_pct[-1]),
                      xytext=(6, 0), textcoords="offset points", va="center",
                      fontsize=12, color=INK_2)
    else:
        ax_b.text(0.5, 0.5, "no battery telemetry", transform=ax_b.transAxes,
                  ha="center", color=MUTED)
    _mark_calls(ax_b, f, label=False)
    ax_b.set_ylabel("Battery (%)")
    ax_b.set_ylim(0, 105)
    ax_b.set_title("Battery remaining (simulated)")
    ax_b.set_xlabel("Mission time (s)")
    ax_b.set_xlim(0, f.t[-1])
    fig.get_layout_engine().set(rect=(0, 0.04, 1, 0.96))
    _caption(fig, f)
    return fig


def fig_timeline(f: FlightData):
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    draw_timeline(ax, f)
    ax.set_title("Agent decisions and outcomes", pad=30)
    fig.get_layout_engine().set(rect=(0, 0.04, 1, 0.96))
    _caption(fig, f)
    return fig


def headline(m: dict) -> str:
    parts = [
        f"{m['distance_flown_m']:g} m flown",
        f"peak {m['max_speed_ms']:g} m/s",
        f"max altitude {m['max_altitude_m']:g} m",
        f"{m['airborne_s']:g} s airborne",
        f"{m['tool_calls']} tool calls",
    ]
    if m["safety_rejections"]:
        n = len(m["safety_rejections"])
        parts.append(f"{n} unsafe command{'s' if n > 1 else ''} rejected")
    parts.append(f"landed {m['landing_offset_m']:g} m from launch")
    if m.get("stl"):
        stl = m["stl"]
        parts.append(f"STL {'satisfied' if stl['satisfied'] else 'violated'} "
                     f"(robustness {stl['robustness']})")
    return "  ·  ".join(parts)


FIGURES = {
    "01_overview": lambda f, m, g: fig_overview(f, m, g),
    "02_ground_track": lambda f, m, g: fig_ground_track(f, g),
    "03_trajectory_3d": lambda f, m, g: fig_trajectory_3d(f),
    "04_position_tracking": lambda f, m, g: fig_position_tracking(f),
    "05_speed_battery": lambda f, m, g: fig_speed_battery(f),
    "06_agent_timeline": lambda f, m, g: fig_timeline(f),
}


def save_all(f: FlightData, metrics: dict, out_dir: Path, geofence_m: float | None,
             dpi: int = 200, formats: tuple[str, ...] = ("png",)) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, build in FIGURES.items():
        fig = build(f, metrics, geofence_m)
        for ext in formats:
            path = out_dir / f"{name}.{ext}"
            fig.savefig(path, dpi=dpi)
            written.append(path)
        plt.close(fig)
    return written
