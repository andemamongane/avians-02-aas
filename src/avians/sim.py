"""The simulator around one flight: environment in, recordings out.

    prepare()  before takeoff: check the world, swap its scenery for a plain
               ground (sim.plain_world), set PX4 parameters (the flight
               envelope, battery, failure injection), draw the map's regions in
               Gazebo
    start()    at mission start: play the wind, schedule failures, start the
               Gazebo recordings (chase and overview cameras, every model's pose)
    stop()     after the flight: stop all of that, clear failures, remove the
               overlay, and copy into the episode: gazebo.mp4,
               gazebo_overview.mp4, gazebo_poses.jsonl, px4.ulg

Two profiles (config/aas.yaml, sim.profile):
    local    PX4 SIH in the avians container: parameters, failures and the
             .ulg work; there is no Gazebo, so no wind, overlay or video
    remote   AAS's simulation container on the Linux machine, reached with
             `docker exec` (the avians container needs the docker socket);
             Gazebo work goes through simside/avians_gz.py, copied there

With the fake backend there is no simulator: the environment is only recorded.

Commands that run in the background inside the container (wind, recording)
write a pid file and are stopped with `kill` inside the container: killing a
`docker exec` client does not stop what it started.
"""
from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import threading
import time
from pathlib import Path

from .environment import Environment, envelope_params
from .geo import chase_camera_model, overview_camera_model, plain_ground_model, region_models
from .maps import Map

HELPER = Path(__file__).parent / "simside" / "avians_gz.py"
WORK = "/tmp/avians"                         # scratch directory in the sim container
VIDEOS = {"chase": "gazebo.mp4", "overview": "gazebo_overview.mp4"}   # camera view: file


class SimError(RuntimeError):
    pass


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


class SimControl:
    def __init__(self, cfg: dict, episode: Path, profile: str, run=_run, popen=subprocess.Popen):
        if profile not in ("local", "remote"):
            raise SimError(f"sim profile must be local or remote, not {profile!r}")
        self.cfg = cfg
        self.episode = Path(episode)
        self.profile = profile
        self._run = run
        self._popen = popen
        self.container = cfg.get("container", "simulation-container-inst0")
        self.world: str | None = None
        self.model = cfg.get("model", "x500_0")
        self.notes: list[str] = []                   # what was (not) applied, for meta.json
        self.applied_params: dict[str, float] = {}
        self._env: Environment | None = None
        self._overlay: list[str] = []
        self._timers: list[threading.Timer] = []
        self._recordings: list[tuple[str, str, object, object]] = []   # view, job, recorder, ffmpeg
        self._map: Map | None = None
        self._anchor_xy = (0.0, 0.0)
        self._plain_world: list[str] = []
        self._t_start: float | None = None

    # --- plumbing --------------------------------------------------------------
    def _cmd(self, args: list[str]) -> list[str]:
        if self.profile == "remote":
            return ["docker", "exec", self.container, "bash", "-lc", shlex.join(args)]
        return args

    def sh(self, args: list[str], check: bool = True) -> str:
        res = self._run(self._cmd(args))
        if check and res.returncode != 0:
            raise SimError(f"{' '.join(args)} failed: {(res.stderr or res.stdout).strip()}")
        return res.stdout

    def put(self, text: str, name: str) -> str:
        """Write a file where the simulator can read it; returns its path there."""
        local = self.episode / "sim" / name
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(text)
        if self.profile == "local":
            return str(local)
        self._run(["docker", "exec", self.container, "mkdir", "-p", WORK])
        res = self._run(["docker", "cp", str(local), f"{self.container}:{WORK}/{name}"])
        if res.returncode != 0:
            raise SimError(f"docker cp {name} failed: {res.stderr.strip()}")
        return f"{WORK}/{name}"

    def _px4(self, tool: str) -> list[str]:
        if self.profile == "local":
            return [str(Path(self.cfg.get("local_px4_bin", "/opt/px4/bin")) / f"px4-{tool}")]
        bin_ = self.cfg.get("px4_bin", "/aas/github_apps/PX4-Autopilot/build/px4_sitl_default/bin")
        return [f"{bin_}/px4-{tool}", "--instance", str(self.cfg.get("px4_instance", 0))]

    def _helper(self, *args: str) -> list[str]:
        return ["python3", f"{WORK}/avians_gz.py", *args]

    def _background(self, name: str, args: list[str], stdout=None):
        """Start args inside the container, recording its pid for stop. Its
        stderr goes to sim/<name>.log in the episode."""
        inner = f"echo $$ > {WORK}/{name}.pid; exec {shlex.join(args)}"
        log = self.episode / "sim" / f"{name}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "w") as err:
            return self._popen(["docker", "exec", "-i", self.container, "bash", "-lc", inner],
                               stdout=stdout or subprocess.DEVNULL, stderr=err)

    def _kill(self, name: str) -> None:
        self._run(["docker", "exec", self.container, "bash", "-lc",
                   f"kill -TERM $(cat {WORK}/{name}.pid) 2>/dev/null; rm -f {WORK}/{name}.pid"])

    # --- before takeoff --------------------------------------------------------
    def prepare(self, env: Environment, limits, mission_map: Map | None) -> None:
        self._env = env
        if self.profile == "remote":
            self._check_world(env.world or self.cfg.get("world"))
            res = self._run(["docker", "cp", str(HELPER), f"{self.container}:{WORK}/avians_gz.py"])
            if res.returncode != 0:
                self._run(["docker", "exec", self.container, "mkdir", "-p", WORK])
                res = self._run(["docker", "cp", str(HELPER), f"{self.container}:{WORK}/avians_gz.py"])
                if res.returncode != 0:
                    raise SimError(f"could not copy the Gazebo helper: {res.stderr.strip()}")
            self._make_plain_world()
        elif env.world:
            self.notes.append(f"world {env.world} not used: PX4 SIH has no world")

        params = {**envelope_params(limits), **env.params()}
        for name, value in params.items():
            self.sh(self._px4("param") + ["set", name, f"{value:g}"])
        self.applied_params = params

        if self.profile == "remote":
            self._map = mission_map
            self._anchor_xy = self._anchor()
        if self.profile == "remote" and mission_map is not None and mission_map.regions:
            for name, sdf in region_models(mission_map, self._anchor_xy).items():
                path = self.put(sdf, f"{name}.sdf")
                self.sh(self._helper("spawn", "--world", self.world, "--sdf", path))
                self._overlay.append(name)
        if self.profile == "local" and not env.wind.calm:
            self.notes.append("wind not simulated: PX4 SIH has no wind model")

    def _check_world(self, world: str | None) -> None:
        topics = self.sh(["gz", "topic", "-l"])
        running = sorted({t.split("/")[2] for t in topics.split() if t.startswith("/world/")})
        if not running:
            raise SimError("Gazebo is not running in the simulation container")
        if world and world not in running:
            raise SimError(f"the mission needs world {world!r} but Gazebo runs {running}; "
                           f"restart AAS with WORLD={world}")
        self.world = world or running[0]

    def _make_plain_world(self) -> None:
        """Swap the world's scenery (sim.plain_world: models per world) for a
        plain ground, so the recordings show the map's regions and nothing else.
        The world's own ground goes too, so the new one must exist first: the
        drone is standing on it."""
        scenery = self.cfg.get("plain_world", {}).get(self.world, [])
        if not scenery:
            return
        path = self.put(plain_ground_model(), "avians_ground.sdf")
        self.sh(self._helper("spawn", "--world", self.world, "--sdf", path), check=False)
        if not self.sh(self._helper("pose", "--world", self.world, "--model", "avians_ground"),
                       check=False).strip():
            self.notes.append("kept the world's scenery: could not add a plain ground")
            return
        for name in scenery:          # already gone after the first flight of an AAS session
            self.sh(self._helper("remove", "--world", self.world, "--name", name), check=False)
        self._plain_world = list(scenery)

    def _anchor(self) -> tuple[float, float]:
        """Where the drone spawned in the world: the map's origin."""
        out = self.sh(self._helper("pose", "--world", self.world, "--model", self.model),
                      check=False).split()
        if len(out) >= 2:
            return float(out[0]), float(out[1])
        self.notes.append(f"pose of {self.model} unknown; regions drawn at the world origin")
        return 0.0, 0.0

    # --- during the flight -------------------------------------------------------
    def start(self, horizon_s: float) -> None:
        env = self._env or Environment()
        self._t_start = time.time()
        schedule = env.wind.schedule(horizon_s)
        (self.episode / "wind.json").write_text(json.dumps(schedule))
        if self.profile == "remote":
            if not env.wind.calm:
                path = self.put(json.dumps(schedule), "wind.json")
                self._background("wind", self._helper("wind", "--world", self.world,
                                                      "--schedule", path))
            self._start_recording()
        for f in env.failures:
            self._at(f.at, self._px4("failure") + [f.unit, f.type])
            if f.duration:
                self._at(f.at + f.duration, self._px4("failure") + [f.unit, "ok"])

    def _at(self, t: float, args: list[str]) -> None:
        timer = threading.Timer(t, lambda: self.sh(args, check=False))
        timer.daemon = True
        timer.start()
        self._timers.append(timer)

    def _start_recording(self) -> None:
        """One recorder per camera view (sim.record_video.views): `chase` follows
        the drone and logs every model's pose, `overview` frames the whole map."""
        rec = self.cfg.get("record_video", {})
        w, h, fps = rec.get("width", 1280), rec.get("height", 720), rec.get("fps", 15)
        if not shutil.which("ffmpeg"):
            self.notes.append("no ffmpeg: the Gazebo videos were not recorded")
            return
        stamp = int(time.time())
        for view in rec.get("views", list(VIDEOS)):
            if view not in VIDEOS:
                self.notes.append(f"unknown video view {view!r} (known: {', '.join(VIDEOS)})")
                continue
            # A fresh camera name and topic every flight: Gazebo keeps a removed
            # camera's sensor rendering, and silently skips a new camera whose
            # name matches it.
            name = f"avians_{view}_cam_{self.episode.name}_{stamp}"
            topic = f"/avians/{view}/{name}/image"
            if view == "chase":
                sdf = chase_camera_model(name=name, width=w, height=h, fps=fps, topic=topic)
                extra = ["--poses", f"{WORK}/poses.jsonl"]
            else:
                sdf = overview_camera_model(self._map, self._anchor_xy, name=name, width=w,
                                            height=h, fps=fps, topic=topic)
                extra = ["--fixed", "--track"]
            cam = self.put(sdf, f"{view}_cam.sdf")
            job = "record" if view == "chase" else f"record_{view}"
            recorder = self._background(job, self._helper(
                "record", "--world", self.world, "--model", self.model, "--camera-sdf", cam,
                "--camera-name", name, "--topic", topic, *extra, "--width", str(w),
                "--height", str(h), "--fps", str(fps)), stdout=subprocess.PIPE)
            ffmpeg = self._popen(
                ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                 "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264",
                 "-pix_fmt", "yuv420p", str(self.episode / VIDEOS[view])],
                stdin=recorder.stdout)
            self._recordings.append((view, job, recorder, ffmpeg))

    # --- after the flight ------------------------------------------------------------
    def stop(self) -> None:
        for t in self._timers:
            t.cancel()
        env = self._env or Environment()
        for unit in {f.unit for f in env.failures}:
            self.sh(self._px4("failure") + [unit, "ok"], check=False)
        if self.profile == "remote":
            self._kill("wind")
            for _, job, _, _ in self._recordings:
                self._kill(job)
            for view, job, recorder, ffmpeg in self._recordings:
                for p in (recorder, ffmpeg):
                    try:
                        p.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        p.kill()
                self._check_video(view, job)
            if any(view == "chase" for view, *_ in self._recordings):
                self._run(["docker", "cp", f"{self.container}:{WORK}/poses.jsonl",
                           str(self.episode / "gazebo_poses.jsonl")])
            for name in self._overlay:
                self.sh(self._helper("remove", "--world", self.world, "--name", name), check=False)
        self._copy_ulog()

    def _check_video(self, view: str, job: str) -> None:
        """An MP4 with no frames is ~260 bytes: say so, with the recorder's error."""
        video = self.episode / VIDEOS[view]
        if video.exists() and video.stat().st_size > 1024:
            return
        log = self.episode / "sim" / f"{job}.log"
        err = log.read_text().strip().splitlines() if log.exists() else []
        self.notes.append(f"the Gazebo {view} video has no frames"
                          + (f": {err[0]}" if err else "") + f" (see sim/{job}.log)")

    def _copy_ulog(self) -> None:
        """PX4's own flight log (.ulg) written since the mission started."""
        if self._t_start is None:
            return
        root = self.cfg.get("local_log_root" if self.profile == "local" else "px4_log_root",
                            "/tmp/px4" if self.profile == "local"
                            else "/aas/github_apps/PX4-Autopilot/build/px4_sitl_default")
        since = int(self._t_start) - 5
        found = self.sh(["bash", "-c", f"find {shlex.quote(root)} -name '*.ulg' "
                                       f"-newermt @{since} -printf '%T@ %p\\n' | sort -n | tail -1"],
                        check=False).split()
        if len(found) < 2:
            self.notes.append("no PX4 .ulg log found for this flight")
            return
        src, dst = found[1], self.episode / "px4.ulg"
        if self.profile == "local":
            shutil.copy(src, dst)
        else:
            self._run(["docker", "cp", f"{self.container}:{src}", str(dst)])

    def summary(self) -> dict:
        return {"profile": self.profile, "world": self.world,
                "px4_params": self.applied_params, "overlay": self._overlay,
                "plain_world": self._plain_world, "notes": self.notes}
