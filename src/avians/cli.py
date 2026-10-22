"""Command line entry point:  avians fly | score | plot | validate | doctor | probe"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from .logging.adp import convert_episode, validate_episode
from .logging.bag import BagRecorder
from .logging.trajectory import TrajectoryLogger
from .maps import Map
from .pilots import PilotConfigError, make_pilot, pilot_label
from .runner import MissionRunner, RunCaps, build_system_prompt
from .safety import Limits, SafetyArbiter
from .schema import SCHEMA_VERSION
from .skills import Skills
from .tools import ToolLayer

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "config"
DATA = Path(os.environ.get("AVIANS_DATA", REPO / "data")) / "episodes"


def _load(name: str) -> dict:
    return yaml.safe_load((CONFIG / name).read_text())


def _next_episode_dir() -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    n = 1 + max(
        (int(p.name.split("_")[1]) for p in DATA.glob("ep_*") if p.name.split("_")[1].isdigit()),
        default=0,
    )
    d = DATA / f"ep_{n:04d}"
    d.mkdir()
    return d


def _make_backend(kind: str, aas_cfg: dict):
    """Create a backend and, for ROS, start the thread that spins its node.

    Nothing else spins the node, so without this thread no callbacks fire,
    no telemetry arrives, and connect() waits out its timeout.
    """
    if kind == "fake":
        from .backends.fake import FakeBackend
        return FakeBackend(), None

    import threading

    import rclpy
    from rclpy.executors import MultiThreadedExecutor

    from .backends.aas import AasBackend

    rclpy.init()
    backend = AasBackend(aas_cfg)
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(backend)
    threading.Thread(target=executor.spin, daemon=True).start()
    return backend, rclpy


def load_mission(path: str | Path) -> tuple[dict, str, Map | None]:
    """Mission yaml -> (config, prompt text, map). Paths in the yaml are
    relative to the repo root (map) and to missions/prompts/ (prompt_file)."""
    path = Path(path)
    mission = yaml.safe_load(path.read_text())
    prompt = (path.parent / "prompts" / mission["prompt_file"]).read_text().strip()
    m = Map.load(REPO / mission["map"]) if mission.get("map") else None
    if mission.get("spec"):
        from .stl import check_regions, parse
        parse(mission["spec"])                       # fail before flying, not after
        unknown = check_regions(parse(mission["spec"]), m)
        if unknown:
            raise SystemExit(f"{path}: spec uses regions the map does not define: {unknown}")
    from .monitor import POLICIES
    if mission.get("early_stop", "violated") not in POLICIES:
        raise SystemExit(f"{path}: early_stop must be one of {', '.join(POLICIES)}")
    return mission, prompt, m


# ---------------------------------------------------------------- commands
def cmd_fly(args: argparse.Namespace) -> int:
    mission, prompt, mission_map = load_mission(args.mission)

    pilots_cfg = _load("pilots.yaml")
    pilot_name = args.pilot or pilots_cfg.get("default", "claude")
    if args.dry_run:
        from .testing_stub import StubPilot
        script = [(s[0], s[1]) for s in mission.get("dry_run_script", [])] or None
        pilot = StubPilot(script)
        pilot_name, model_id = "scripted", "scripted"
    else:
        # Before anything flies: a missing key or SDK must not strand a drone.
        try:
            pilot = make_pilot(pilot_name, pilots_cfg)
        except PilotConfigError as e:
            print(f"[avians] {e}")
            return 2
        model_id = pilot.model

    limits_cfg = _load("limits.yaml")
    aas_cfg = _load("aas.yaml")

    backend, ros = _make_backend(args.backend, aas_cfg)
    episode = _next_episode_dir()
    if mission.get("map"):
        shutil.copy(REPO / mission["map"], episode / "map.yaml")
    print(f"[avians] episode {episode.name}  backend={args.backend}  "
          f"pilot={pilot_name} ({pilot.provider}: {model_id})"
          + ("  (dry run: scripted, no API calls)" if args.dry_run else ""))

    logger = TrajectoryLogger(episode)
    recorder = BagRecorder(episode, aas_cfg.get("record", []),
                           enabled=(args.backend != "fake"))
    started = time.time()
    result = None
    monitor = None
    try:
        backend.connect()
        recorder.start()
        skills = Skills(
            backend,
            arrive_radius=limits_cfg.get("arrive_radius_m", 1.0),
            default_timeout=limits_cfg.get("action_timeout_s", 60),
            land_timeout=limits_cfg.get("land_timeout_s", 120),
        )
        tools = ToolLayer(skills, SafetyArbiter(Limits.from_dict(limits_cfg)), mission_map)
        monitor = _make_monitor(mission, mission_map, backend, limits_cfg, episode,
                                threaded=(args.backend != "fake"))

        spec_for_agent = mission.get("spec") if mission.get("show_spec", True) else None
        runner = MissionRunner(
            pilot, tools, logger, model_id,
            caps=RunCaps(
                max_steps=mission.get("max_steps", 25),
                max_wall_s=mission.get("max_wall_s", 300),
                max_tokens_total=mission.get("max_tokens_total", 300_000),
            ),
            system_prompt=build_system_prompt(mission_map, spec_for_agent),
            monitor=monitor,
        )
        result = runner.run(mission["name"], prompt, mission)
        print(f"[avians] {result.steps} steps, ended by {result.terminated_by}")
    finally:
        # The flight recorder keeps running a little after the last action so
        # the trace includes the drone settling (and landing, if it did).
        time.sleep(1.0 if args.backend != "fake" else 0.0)
        recorder.stop()
        logger.close()
        backend.disconnect()
        if ros is not None and ros.ok():
            ros.shutdown()

    meta = {
        "episode": episode.name,
        "schema_version": SCHEMA_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
        "mission": mission["name"],
        "spec": mission.get("spec"),
        "spec_shown_to_agent": bool(mission.get("spec") and mission.get("show_spec", True)),
        "map": mission.get("map"),
        "backend": args.backend,
        "dry_run": args.dry_run,
        "pilot": pilot_name,
        "pilot_label": "Scripted plan" if args.dry_run else pilot_label(pilot_name, pilots_cfg),
        "provider": pilot.provider,
        "model": None if args.dry_run else model_id,
        "duration_s": round(time.time() - started, 1),
        "steps": getattr(result, "steps", 0),
        "terminated_by": getattr(result, "terminated_by", "crashed"),
        "early_stop": monitor.summary() if monitor else None,
        "usage": getattr(result, "usage", {}),
        "error": getattr(result, "error", None),
        "versions": _versions(),
        # NED point the tool frame is measured from (AAS backend); the bag is raw NED
        "frame_origin_ned": getattr(backend, "frame_origin_ned", None),
    }
    (episode / "meta.json").write_text(json.dumps(meta, indent=2))
    convert_episode(episode)

    problems = validate_episode(episode)
    for p in problems:
        print(f"[warn] {p}")
    print(f"[avians] wrote {episode}")
    _score(episode)
    if not args.no_plots:
        _plot(episode, limits_cfg.get("geofence_radius_m"))
    return 0


def _make_monitor(mission: dict, mission_map: Map | None, backend, limits_cfg: dict,
                  episode: Path, threaded: bool):
    """The online STL monitor for a mission with a spec (early_stop: off still
    logs the intervals to stl_online.jsonl)."""
    if not mission.get("spec"):
        return None
    from .monitor import OnlineMonitor
    from .stl import FutureBounds
    limits = Limits.from_dict(limits_cfg)
    future = FutureBounds(
        max_speed=limits.max_speed_ms * limits_cfg.get("monitor_speed_margin", 1.5),
        alt_range=(-1.0, limits.max_altitude_m + 1.0),   # landed drift; ceiling overshoot
    )
    return OnlineMonitor(
        mission["spec"], mission_map, backend,
        policy=mission.get("early_stop", "violated"), future=future,
        eval_every_s=limits_cfg.get("monitor_eval_s", 1.0),
        log_path=episode / "stl_online.jsonl", threaded=threaded,
    )


def _score(episode: Path) -> dict | None:
    """STL-score one episode into stl.json. Never raises: the episode is already
    safely on disk, and a scoring problem must not look like a failed flight."""
    try:
        from .plots.flight import load_episode
        from .stl import Trace, score
        meta = json.loads((episode / "meta.json").read_text())
        if not meta.get("spec"):
            return None
        m = Map.load(episode / "map.yaml") if (episode / "map.yaml").exists() else None
        f = load_episode(episode)
        trace = Trace(t=[max(0.0, float(v)) for v in f.t], east=list(map(float, f.y)),
                      north=list(map(float, f.x)), alt=list(map(float, f.z)),
                      speed=list(map(float, f.speed)))
        report = score(meta["spec"], trace, m)
        report["trace_source"] = f.source
        (episode / "stl.json").write_text(json.dumps(report, indent=2))
    except Exception as e:
        print(f"[warn] STL scoring for {episode.name} failed ({type(e).__name__}: {e}); "
              f"retry with: avians score {episode.name}")
        return None
    verdict = "SATISFIED" if report["satisfied"] else "VIOLATED"
    print(f"[avians] STL {verdict}  robustness {report['robustness']}  ({report['trace_source']})")
    for c in report["clauses"]:
        print(f"           {'+' if c['satisfied'] else '-'} {c['robustness']:>8}  {c['clause']}")
    if report["padded_s"]:
        print(f"           (flight ended at {report['flight_end_s']} s; held the last state "
              f"{report['padded_s']} s to cover the spec's {report['horizon_s']:g} s horizon)")
    return report


def _plot(episode: Path, geofence_m: float | None, **kw) -> bool:
    """Plot one episode. Never raises, for the same reason as _score."""
    try:
        from .plots import plot_episode
        written, metrics = plot_episode(episode, geofence_m=geofence_m, **kw)
    except Exception as e:
        print(f"[warn] plots for {episode.name} failed ({type(e).__name__}: {e}); "
              f"retry with: avians plot {episode.name}")
        return False
    verdict = "PASS" if metrics["success"] else "CHECK"
    failed = [name for name, ok in metrics["checks"].items() if not ok]
    print(f"[avians] {episode.name} {metrics['mission']}: {verdict}"
          + (f"  (failed: {'; '.join(failed)})" if failed else ""))
    print(f"[avians] {len(written)} files in {episode / 'plots'}")
    return True


def _episodes(args: argparse.Namespace) -> list[Path]:
    if getattr(args, "all", False):
        return sorted(d for d in DATA.glob("ep_*") if (d / "trajectory.jsonl").exists())
    if args.episodes:
        return [_resolve_episode(e) for e in args.episodes]
    return sorted(d for d in DATA.glob("ep_*") if (d / "trajectory.jsonl").exists())[-1:]


def cmd_score(args: argparse.Namespace) -> int:
    dirs = _episodes(args)
    if not dirs:
        print(f"no episodes under {DATA}")
        return 1
    results = [_score(d) for d in dirs]
    return 0 if all(r is not None for r in results) else 1


def cmd_plot(args: argparse.Namespace) -> int:
    dirs = _episodes(args)
    if not dirs:
        print(f"no episodes under {DATA}")
        return 1
    geofence = _load("limits.yaml").get("geofence_radius_m")
    formats = ("png", "pdf") if args.pdf else ("png",)
    ok = [_plot(d, geofence, dpi=args.dpi, formats=formats) for d in dirs]
    return 0 if all(ok) else 1


def _resolve_episode(name: str) -> Path:
    """Accept a path, 'ep_0004', or just '4'."""
    p = Path(name)
    if p.is_dir():
        return p
    if name.isdigit():
        name = f"ep_{int(name):04d}"
    return DATA / name


def cmd_validate(args: argparse.Namespace) -> int:
    root = Path(args.path) if args.path else DATA
    dirs = sorted(d for d in root.glob("ep_*") if d.is_dir())
    if not dirs:
        print(f"no episodes under {root}")
        return 1
    bad = 0
    steps = 0
    for d in dirs:
        problems = validate_episode(d)
        meta = d / "meta.json"
        if meta.exists():
            steps += json.loads(meta.read_text()).get("steps", 0)
        if problems:
            bad += 1
            for p in problems:
                print(f"FAIL {p}")
        else:
            print(f"ok   {d.name}")
    print(f"\n{len(dirs)} episodes, {len(dirs) - bad} valid, {steps} total steps")
    return 1 if bad else 0


def cmd_video(args: argparse.Namespace) -> int:
    from .video import render_episode
    dirs = _episodes(args)
    if not dirs:
        print(f"no episodes under {DATA}")
        return 1
    for d in dirs:
        out = render_episode(d, d / "plots" / "flight.mp4", speed=args.speed)
        print(f"[avians] {out}")
    return 0


def _latest_flights(pilots: list[str], include_dry: bool) -> dict[tuple[str, str], Path]:
    """(mission, pilot) -> the most recent real flight of that pair."""
    found: dict[tuple[str, str], Path] = {}
    for d in sorted(DATA.glob("ep_*")):
        meta_p = d / "meta.json"
        if not meta_p.exists() or not (d / "trajectory.jsonl").exists():
            continue
        meta = json.loads(meta_p.read_text())
        if meta.get("backend") == "fake" or (meta.get("dry_run") and not include_dry):
            continue
        if meta.get("pilot") in pilots and meta.get("terminated_by") != "crashed":
            found[(meta["mission"], meta["pilot"])] = d
    return found


def cmd_showcase(args: argparse.Namespace) -> int:
    """Fly (optionally) and film every mission x pilot pair, plus a side-by-side
    comparison per mission, into data/showcase/."""
    pilots_cfg = _load("pilots.yaml")
    pilots = args.pilots or pilots_cfg.get("showcase", [pilots_cfg.get("default", "claude")])
    mission_files = [Path(m) for m in args.missions] if args.missions else \
        sorted((REPO / "missions").glob("*.yaml"))
    names = {yaml.safe_load(m.read_text())["name"]: m for m in mission_files}

    if args.fly:
        cfg = pilots_cfg.get("pilots", {})
        keyless = [p for p in pilots if cfg.get(p, {}).get("key_env")
                   and not os.environ.get(cfg[p]["key_env"])]
        if keyless:
            print("[showcase] not flying " + ", ".join(
                f"{p} (no {cfg[p]['key_env']} in .env)" for p in keyless)
                + "; their existing flights are still filmed")
        have = _latest_flights(pilots, include_dry=False)
        for name, path in names.items():
            for p in pilots:
                if p in keyless or ((name, p) in have and not args.refly):
                    continue
                print(f"\n[showcase] flying {name} with {p}")
                cmd = [sys.executable, "-m", "avians.cli", "fly", str(path), "--pilot", p]
                if subprocess.run(cmd).returncode != 0:
                    print(f"[showcase] {p} could not fly {name}; skipping")

    from .video import render_comparison, render_episode
    flights = _latest_flights(pilots, include_dry=args.include_dry_runs)
    out_root = DATA.parent / "showcase"
    rows, missing = [], []
    for name in names:
        eps = [(p, flights[(name, p)]) for p in pilots if (name, p) in flights]
        missing += [f"{name} / {p}" for p in pilots if (name, p) not in flights]
        for p, ep in eps:
            out = render_episode(ep, out_root / name / f"{p}.mp4", speed=args.speed)
            print(f"[showcase] {out}")
            rows.append((name, p, ep))
        if len(eps) >= 2:
            out = render_comparison([ep for _, ep in eps], out_root / name / "compare.mp4",
                                    speed=args.speed)
            print(f"[showcase] {out}")
    (out_root).mkdir(parents=True, exist_ok=True)
    (out_root / "README.md").write_text(_showcase_md(rows, missing, pilots_cfg))
    print(f"[showcase] {len(rows)} flights filmed; index: {out_root / 'README.md'}")
    if missing:
        print("[showcase] no flight yet for: " + ", ".join(missing))
    return 0


def _showcase_md(rows, missing, pilots_cfg) -> str:
    lines = ["# avians-02 showcase", "",
             "Each video replays one recorded flight (rosbag trace, the agent's tool calls, ",
             "its thinking time, the STL score). `compare.mp4` puts every pilot side by side.",
             "", "| mission | pilot | STL | robustness | ended by | tool calls | tokens | video "
             "| episode |",
             "|---|---|---|---|---|---|---|---|---|"]
    for name, p, ep in rows:
        meta = json.loads((ep / "meta.json").read_text())
        stl = json.loads((ep / "stl.json").read_text()) if (ep / "stl.json").exists() else {}
        tok = meta.get("usage", {})
        lines.append(
            f"| {name} | {pilot_label(p, pilots_cfg)} | "
            f"{'satisfied' if stl.get('satisfied') else ('violated' if stl else 'n/a')} | "
            f"{stl.get('robustness', '')} | {meta.get('terminated_by', '')} | "
            f"{meta.get('steps', '')} | "
            f"{tok.get('input_tokens', 0) + tok.get('output_tokens', 0):,} | "
            f"[{p}.mp4]({name}/{p}.mp4) | {ep.name} |")
    if missing:
        lines += ["", "Not flown yet: " + ", ".join(missing)]
    return "\n".join(lines) + "\n"


def cmd_check(args: argparse.Namespace) -> int:
    """Parse a mission's map and spec without flying; print what the agent sees."""
    mission, prompt, m = load_mission(args.mission)
    spec = mission.get("spec")
    print(build_system_prompt(m, spec if mission.get("show_spec", True) else None))
    print("\n--- mission prompt ---\n" + prompt)
    if spec:
        from .stl import parse
        node = parse(spec)
        print(f"\n--- spec parsed as ---\n{node}\nhorizon {node.horizon():g} s")
    return 0


def cmd_doctor(_: argparse.Namespace) -> int:
    ns = _load("aas.yaml").get("namespace", "/Drone1")
    print("=== avians doctor ===")
    print(f"python            {sys.version.split()[0]}")
    print(f"ROS_DISTRO        {os.environ.get('ROS_DISTRO', '(unset)')}")
    print(f"ROS_DOMAIN_ID     {os.environ.get('ROS_DOMAIN_ID', '(unset)')}")
    pilots = _load("pilots.yaml").get("pilots", {})
    for key in sorted({p.get("key_env") for p in pilots.values() if p.get("key_env")}):
        who = ", ".join(n for n, p in pilots.items() if p.get("key_env") == key)
        print(f"{key:<18}{'set' if os.environ.get(key) else 'MISSING':<8} (pilots: {who})")
    for mod in ("rclpy", "px4_msgs", "autopilot_interface_msgs", "anthropic", "openai", "yaml"):
        try:
            __import__(mod)
            print(f"import {mod:<24} ok")
        except Exception as e:
            print(f"import {mod:<24} FAILED ({type(e).__name__})")
    try:
        out = subprocess.run(["ros2", "topic", "list"], capture_output=True,
                             text=True, timeout=15).stdout
        fmu = [t for t in out.splitlines() if t.startswith(f"{ns}/fmu")]
        print(f"ros2 topics       {len(out.splitlines())} total, {len(fmu)} {ns}/fmu/*")
        if not fmu:
            print("  !! no PX4 topics: is PX4 running and the XRCE agent up?")
        acts = subprocess.run(["ros2", "action", "list"], capture_output=True,
                              text=True, timeout=15).stdout.split()
        mine = [a for a in acts if a.startswith(f"{ns}/")]
        print(f"AAS actions       {', '.join(a.split('/')[-1] for a in mine) or 'NONE'}")
        if f"{ns}/takeoff_action" not in mine:
            print("  !! AAS px4_interface is not serving takeoff_action")
    except Exception as e:
        print(f"ros2 CLI          unavailable ({type(e).__name__})")
    print(f"data dir          {DATA}")
    return 0


def cmd_probe(_: argparse.Namespace) -> int:
    for cmd in (["ros2", "node", "list"], ["ros2", "topic", "list", "-t"],
                ["ros2", "action", "list", "-t"], ["ros2", "service", "list", "-t"]):
        print(f"\n$ {' '.join(cmd)}")
        try:
            print(subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout)
        except Exception as e:
            print(f"({type(e).__name__})")
    return 0


def _versions() -> dict:
    aas = Path("/opt/aas_commit")
    return {
        "aas_commit": aas.read_text().strip() if aas.exists() else _git(
            "rev-parse", "HEAD", cwd=REPO / "external" / "aerial-autonomy-stack"),
        "px4": os.environ.get("PX4_VERSION", "unknown"),
        "px4_msgs": os.environ.get("PX4_MSGS_REF", "unknown"),
        "ros_distro": os.environ.get("ROS_DISTRO", "unknown"),
        "git_sha": _git("rev-parse", "--short", "HEAD", cwd=REPO),
    }


def _git(*args: str, cwd: Path) -> str:
    try:
        return subprocess.run(["git", *args], cwd=cwd, capture_output=True,
                              text=True, timeout=5).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="avians")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fly", help="run one mission and record one episode")
    f.add_argument("mission", help="path to a mission yaml")
    f.add_argument("--pilot", help="AI agent from config/pilots.yaml "
                   "(claude, deepseek, codex, ...); default: its `default`")
    f.add_argument("--backend", default="aas", choices=["aas", "fake"])
    f.add_argument("--dry-run", action="store_true",
                   help="scripted client (the mission's dry_run_script), no API calls")
    f.add_argument("--no-plots", action="store_true",
                   help="skip generating plots after the flight")
    f.set_defaults(func=cmd_fly)

    c = sub.add_parser("check", help="parse a mission's map and spec; print the agent's prompt")
    c.add_argument("mission")
    c.set_defaults(func=cmd_check)

    for name, helptext, func in (
            ("score", "STL-score recorded episodes against their spec", cmd_score),
            ("plot", "make slide-ready plots for recorded episodes", cmd_plot)):
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("episodes", nargs="*",
                        help="episode dirs or names (ep_0004 or 4); default: the latest")
        sp.add_argument("--all", action="store_true", help="every episode")
        if name == "plot":
            sp.add_argument("--dpi", type=int, default=200, help="PNG resolution (default 200)")
            sp.add_argument("--pdf", action="store_true", help="also write vector PDFs")
        sp.set_defaults(func=func)

    vid = sub.add_parser("video", help="render flight videos (plots/flight.mp4)")
    vid.add_argument("episodes", nargs="*", help="episodes (ep_0004 or 4); default: the latest")
    vid.add_argument("--all", action="store_true", help="every episode")
    vid.add_argument("--speed", type=float, default=4.0, help="playback speed (default 4x)")
    vid.set_defaults(func=cmd_video)

    sc = sub.add_parser("showcase", help="film every mission x pilot into data/showcase/")
    sc.add_argument("--pilots", nargs="+", help="default: `showcase` in config/pilots.yaml")
    sc.add_argument("--missions", nargs="+", help="mission yamls; default: all")
    sc.add_argument("--fly", action="store_true", help="first fly the pairs not flown yet")
    sc.add_argument("--refly", action="store_true", help="with --fly: fly every pair again")
    sc.add_argument("--include-dry-runs", action="store_true",
                    help="also film scripted (--dry-run) flights")
    sc.add_argument("--speed", type=float, default=4.0, help="playback speed (default 4x)")
    sc.set_defaults(func=cmd_showcase)

    v = sub.add_parser("validate", help="check recorded episodes")
    v.add_argument("path", nargs="?")
    v.set_defaults(func=cmd_validate)

    sub.add_parser("doctor", help="environment health check").set_defaults(func=cmd_doctor)
    sub.add_parser("probe", help="dump the ROS graph").set_defaults(func=cmd_probe)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
