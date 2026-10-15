"""Flight videos: one episode as an animation, or several pilots side by side.

    render_episode(ep_dir, out.mp4)          map + altitude + agent log + STL verdict
    render_comparison([ep1, ep2, ep3], out)  one map per pilot, one shared clock

Everything is drawn from what the episode recorded (rosbag trace, tool calls,
model latency, stl.json), so a video is a faithful replay, not a mock-up. The
clock is mission time; videos play faster than real time (`speed`) and say so.

matplotlib + ffmpeg, both inside the container. 1920x1080, H.264.
"""
from __future__ import annotations

import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
import numpy as np                                  # noqa: E402
from matplotlib.animation import FFMpegWriter       # noqa: E402

from .plots.figures import (                        # noqa: E402
    ACTUAL, AXIS, COMMANDED, GRID, INK, INK_2, MUTED, STATUS, SURFACE,
    call_label, draw_regions,
)
from .plots.flight import FlightData, load_episode  # noqa: E402

W, H, DPI = 19.2, 10.8, 100                         # 1920 x 1080
PASS, FAIL = STATUS["arrived"], STATUS["failed"]


# ---------------------------------------------------------------- helpers
class Replay:
    """Interpolated state of one episode at any mission time."""

    def __init__(self, f: FlightData):
        self.f = f
        t = np.asarray(f.t, dtype=float)
        keep = np.concatenate([[True], np.diff(t) > 0])       # np.interp needs increasing t
        self.t, self.n, self.e, self.a = t[keep], f.x[keep], f.y[keep], f.z[keep]
        self.end = max(float(self.t[-1]), max((c.t_end for c in f.calls), default=0.0))

    def at(self, t: float) -> tuple[float, float, float]:
        return (float(np.interp(t, self.t, self.e)), float(np.interp(t, self.t, self.n)),
                float(np.interp(t, self.t, self.a)))

    def heading(self, t: float, dt: float = 0.5) -> float | None:
        e0, n0, _ = self.at(t - dt)
        e1, n1, _ = self.at(t)
        if math.hypot(e1 - e0, n1 - n0) < 0.05:
            return None
        return math.degrees(math.atan2(n1 - n0, e1 - e0))   # 0 = east, CCW

    def upto(self, t: float) -> slice:
        return slice(0, int(np.searchsorted(self.t, t, side="right")))

    def thinking(self, t: float) -> bool:
        return any(a <= t < b for a, b in self.f.thinking)

    def calls_so_far(self, t: float):
        return [(i, c) for i, c in enumerate(self.f.calls, 1) if c.t_start <= t]


def _label(f: FlightData) -> str:
    """Pilot name for titles. Episodes flown before pilots.yaml existed only
    recorded a model id, so look that up there."""
    if f.meta.get("pilot_label"):
        return f.meta["pilot_label"]
    if f.meta.get("dry_run"):
        return "Scripted plan"
    try:
        import yaml
        from .cli import CONFIG
        pilots = yaml.safe_load((CONFIG / "pilots.yaml").read_text()).get("pilots", {})
        for p in pilots.values():
            if p.get("model") == f.meta.get("model"):
                return p.get("label", p["model"])
    except Exception:
        pass
    return f.meta.get("model") or f.meta.get("pilot") or "?"


def _extent(replays: list[Replay]) -> tuple[float, float, float, float]:
    """One square map window that fits every flight and the whole map."""
    xs, ys = [0.0], [0.0]
    for r in replays:
        xs += list(r.e[np.isfinite(r.e)])
        ys += list(r.n[np.isfinite(r.n)])
        if r.f.map is not None:
            for reg in r.f.map.regions.values():
                xs += [p[0] for p in reg.polygon]
                ys += [p[1] for p in reg.polygon]
    half = 0.55 * max(max(xs) - min(xs), max(ys) - min(ys), 8.0)
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    return cx - half, cx + half, cy - half, cy + half


def _setup_map(ax, f: FlightData, extent) -> dict:
    ax.set_facecolor(SURFACE)
    draw_regions(ax, f)
    ax.plot(0, 0, marker="s", color=INK, markersize=9, zorder=5)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("East (m)")
    ax.set_ylabel("North (m)")
    ax.grid(True, color=GRID, lw=0.8)
    trail, = ax.plot([], [], color=ACTUAL, lw=2.4, zorder=3)
    drone, = ax.plot([], [], marker="o", markersize=15, color=ACTUAL, zorder=6,
                     markeredgecolor=SURFACE, markeredgewidth=2.5, linestyle="none")
    nose, = ax.plot([], [], color=INK, lw=2.5, zorder=7)
    targets, = ax.plot([], [], marker="o", markersize=11, color=COMMANDED, zorder=4,
                       markeredgecolor=SURFACE, markeredgewidth=2, linestyle="none")
    return {"trail": trail, "drone": drone, "nose": nose, "targets": targets,
            "span": extent[1] - extent[0]}


def _update_map(art: dict, r: Replay, t: float) -> None:
    s = r.upto(t)
    art["trail"].set_data(r.e[s], r.n[s])
    e, n, _ = r.at(t)
    art["drone"].set_data([e], [n])
    hd = r.heading(t)
    if hd is None:
        art["nose"].set_data([], [])
    else:
        L = 0.045 * art["span"]
        art["nose"].set_data([e, e + L * math.cos(math.radians(hd))],
                             [n, n + L * math.sin(math.radians(hd))])
    pts = [(c.target[1], c.target[0]) for _, c in r.calls_so_far(t)
           if c.target is not None and c.tool in ("takeoff", "reposition", "goto_region")
           and c.status not in ("rejected", "blocked")]
    art["targets"].set_data([p[0] for p in pts], [p[1] for p in pts])


def _action_line(r: Replay, t: float) -> tuple[str, str]:
    """What the agent is doing right now, and the colour to show it in."""
    calls = r.calls_so_far(t)
    if calls:
        i, c = calls[-1]
        if t < c.t_end:
            return f"{call_label(i, c)}  ...", INK
    if r.thinking(t):
        return "model is deciding ...", MUTED
    if calls:
        i, c = calls[-1]
        return f"{call_label(i, c)}  -> {c.status}", STATUS.get(c.status, INK_2)
    return "waiting for the model ...", MUTED


def _verdict(f: FlightData) -> tuple[str, str]:
    if not f.stl:
        return "no STL spec", MUTED
    ok = f.stl["satisfied"]
    return (f"STL {'SATISFIED' if ok else 'VIOLATED'}   robustness {f.stl['robustness']}",
            PASS if ok else FAIL)


def _frames(end: float, speed: float, fps: int, hold_s: float):
    n = int(math.ceil(end / speed * fps)) + 1
    for k in range(n):
        yield min(end, k * speed / fps), False
    for _ in range(int(hold_s * fps)):
        yield end, True


def _writer(fps: int, title: str) -> FFMpegWriter:
    return FFMpegWriter(fps=fps, codec="libx264", bitrate=6000,
                        extra_args=["-pix_fmt", "yuv420p"], metadata={"title": title})


# ---------------------------------------------------------------- one episode
def render_episode(episode_dir: Path, out: Path, speed: float = 4.0, fps: int = 30,
                   hold_s: float = 4.0) -> Path:
    f = load_episode(Path(episode_dir))
    r = Replay(f)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=SURFACE)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.05, 1], height_ratios=[1, 1.15, 0.95],
                          left=0.05, right=0.97, top=0.86, bottom=0.07, wspace=0.12, hspace=0.42)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_alt = fig.add_subplot(gs[0, 1])
    ax_log = fig.add_subplot(gs[1, 1])
    ax_stl = fig.add_subplot(gs[2, 1])

    fig.text(0.05, 0.95, f"{f.mission}  ·  {_label(f)}", fontsize=26, fontweight="bold",
             color=INK, va="center")
    spec = f.stl["spec"] if f.stl else (f.meta.get("spec") or "")
    fig.text(0.05, 0.905, textwrap.shorten(f"spec: {spec}", 150), fontsize=13, color=INK_2,
             va="center", family="monospace")
    clock = fig.text(0.97, 0.95, "", fontsize=22, color=INK, ha="right", va="center")
    fig.text(0.97, 0.905, f"{speed:g}x real time  ·  {f.episode}  ·  {f.source}",
             fontsize=12, color=MUTED, ha="right", va="center")

    art = _setup_map(ax_map, f, _extent([r]))
    ax_map.set_title("Ground track", loc="left", fontsize=17, fontweight="bold")

    ax_alt.plot(r.t, r.a, color=AXIS, lw=1.2)                 # the whole flight, faint
    alt_line, = ax_alt.plot([], [], color=ACTUAL, lw=2.4)
    cursor = ax_alt.axvline(0, color=COMMANDED, lw=1.5)
    ax_alt.set_xlim(0, r.end)
    ax_alt.set_ylim(min(0.0, float(np.nanmin(r.a))) - 0.5, float(np.nanmax(r.a)) * 1.25 + 0.5)
    ax_alt.set_title("Altitude (m)", loc="left", fontsize=17, fontweight="bold")
    ax_alt.set_xlabel("Mission time (s)")
    ax_alt.grid(True, color=GRID)

    for ax in (ax_log, ax_stl):
        ax.set_axis_off()
    ax_log.set_title("Agent", loc="left", fontsize=17, fontweight="bold")
    now = ax_log.text(0.0, 0.93, "", fontsize=16, transform=ax_log.transAxes, va="top")
    history = [ax_log.text(0.0, 0.76 - 0.12 * k, "", fontsize=13, transform=ax_log.transAxes,
                           va="top", color=INK_2, family="monospace") for k in range(5)]
    inside = ax_log.text(0.0, 0.0, "", fontsize=13, transform=ax_log.transAxes, color=MUTED)

    ax_stl.set_title("Specification", loc="left", fontsize=17, fontweight="bold")
    clauses = f.stl["clauses"] if f.stl else []
    clause_txt = [ax_stl.text(0.0, 0.88 - 0.17 * k, textwrap.shorten(c["clause"], 70),
                              fontsize=14, family="monospace", color=INK_2,
                              transform=ax_stl.transAxes, va="top")
                  for k, c in enumerate(clauses[:4])]
    verdict = ax_stl.text(0.0, 0.05, "", fontsize=19, fontweight="bold",
                          transform=ax_stl.transAxes)
    vtext, vcol = _verdict(f)

    m = f.map
    writer = _writer(fps, f"{f.mission} {_label(f)}")
    with writer.saving(fig, str(out), dpi=DPI):
        for t, final in _frames(r.end, speed, fps, hold_s):
            clock.set_text(f"t = {t:5.1f} s")
            _update_map(art, r, t)
            s = r.upto(t)
            alt_line.set_data(r.t[s], r.a[s])
            cursor.set_xdata([t, t])
            line, colour = _action_line(r, t)
            now.set_text(line)
            now.set_color(colour)
            done = [(i, c) for i, c in r.calls_so_far(t) if c.t_end <= t][-5:]
            for k, h in enumerate(history):
                if k < len(done):
                    i, c = done[-1 - k]
                    h.set_text(f"{call_label(i, c):<48.48} {c.status}")
                    h.set_color(STATUS.get(c.status, INK_2))
                else:
                    h.set_text("")
            if m is not None:
                e, n, a = r.at(t)
                inside.set_text("inside: " + (", ".join(m.inside(e, n, a)) or "no region"))
            if final:
                for c, txt in zip(clauses, clause_txt):
                    txt.set_text(f"{'+' if c['satisfied'] else '-'} {c['robustness']!s:>7}  "
                                 + textwrap.shorten(c["clause"], 60))
                    txt.set_color(PASS if c["satisfied"] else FAIL)
                verdict.set_text(vtext)
                verdict.set_color(vcol)
            writer.grab_frame()
    plt.close(fig)
    return out


# ---------------------------------------------------------------- comparison
def render_comparison(episode_dirs: list[Path], out: Path, speed: float = 4.0,
                      fps: int = 30, hold_s: float = 5.0) -> Path:
    flights = [load_episode(Path(d)) for d in episode_dirs]
    reps = [Replay(f) for f in flights]
    end = max(r.end for r in reps)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = len(flights)

    fig = plt.figure(figsize=(W, H), dpi=DPI, facecolor=SURFACE)
    gs = fig.add_gridspec(2, n, height_ratios=[1, 0.32], left=0.04, right=0.98,
                          top=0.84, bottom=0.03, wspace=0.18, hspace=0.12)
    f0 = flights[0]
    fig.text(0.04, 0.95, f"{f0.mission}: {n} pilots, same mission", fontsize=26,
             fontweight="bold", va="center")
    spec = f0.stl["spec"] if f0.stl else (f0.meta.get("spec") or "")
    fig.text(0.04, 0.905, textwrap.shorten(f"spec: {spec}", 170), fontsize=13, color=INK_2,
             family="monospace", va="center")
    clock = fig.text(0.98, 0.95, "", fontsize=22, ha="right", va="center")
    fig.text(0.98, 0.905, f"{speed:g}x real time", fontsize=12, color=MUTED, ha="right",
             va="center")

    extent = _extent(reps)
    panels = []
    for k, (f, r) in enumerate(zip(flights, reps)):
        ax = fig.add_subplot(gs[0, k])
        art = _setup_map(ax, f, extent)
        ax.set_title(_label(f), loc="left", fontsize=18, fontweight="bold")
        if k:
            ax.set_ylabel("")
        info = fig.add_subplot(gs[1, k])
        info.set_axis_off()
        act = info.text(0.0, 0.92, "", fontsize=14, transform=info.transAxes, va="top",
                        family="monospace")
        stats = info.text(0.0, 0.55, "", fontsize=13, transform=info.transAxes, va="top",
                          color=INK_2)
        verdict = info.text(0.0, 0.15, "", fontsize=17, fontweight="bold",
                            transform=info.transAxes)
        panels.append((f, r, art, act, stats, verdict))

    writer = _writer(fps, f"{f0.mission} comparison")
    with writer.saving(fig, str(out), dpi=DPI):
        for t, final in _frames(end, speed, fps, hold_s):
            clock.set_text(f"t = {t:5.1f} s")
            for f, r, art, act, stats, verdict in panels:
                tt = min(t, r.end)
                _update_map(art, r, tt)
                line, colour = ("finished", MUTED) if t > r.end else _action_line(r, tt)
                act.set_text(textwrap.shorten(line, 46))
                act.set_color(colour)
                ncalls = len(r.calls_so_far(tt))
                think = sum(min(b, tt) - a for a, b in f.thinking if a < tt)
                stats.set_text(f"{ncalls} tool calls  ·  {think:.0f} s deciding")
                if final or t >= r.end:
                    v, c = _verdict(f)
                    tok = {} if f.meta.get("dry_run") else f.meta.get("usage", {})
                    verdict.set_text(v + (f"\n{tok.get('input_tokens', 0) + tok.get('output_tokens', 0):,} tokens"
                                          if tok else ""))
                    verdict.set_color(c)
            writer.grab_frame()
    plt.close(fig)
    return out
