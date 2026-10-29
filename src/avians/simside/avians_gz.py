#!/usr/bin/env python3
"""avians' hands inside AAS's simulation container (Gazebo Harmonic).

Copied there and run by avians (sim.py) with `docker exec`. Standalone on
purpose: it imports only Gazebo's Python bindings, exactly as AAS's own
simulation_resources/scripts/gz_wind.py does, never avians.

    avians_gz.py pose   --world W --model x500_0                 print "x y z yaw"
    avians_gz.py spawn  --world W --sdf FILE                      add a model
    avians_gz.py remove --world W --name NAME                     remove a model
    avians_gz.py move   --world W                                 set poses, spawn, remove: lines on stdin
    avians_gz.py wind   --world W --schedule FILE                 play [[t, east, north], ...]
    avians_gz.py record --world W --model x500_0 --camera-sdf FILE
                        --poses OUT.jsonl --width 1280 --height 720 --fps 15
        Spawns the chase camera, keeps it behind and above the drone (or, with
        --fixed, where its SDF puts it; --track adds a ground track), logs every
        model pose to OUT.jsonl (optional), and writes raw RGB24 frames to
        stdout (avians pipes them into ffmpeg). Runs until SIGTERM/SIGINT, then
        cleans up.

Wind convention (as gz_wind.py): linear_velocity.x blows toward +X (east),
.y toward +Y (north), in m/s.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time

import gz.transport13 as transport
from gz.msgs10.boolean_pb2 import Boolean
from gz.msgs10.entity_factory_pb2 import EntityFactory
from gz.msgs10.entity_pb2 import Entity
from gz.msgs10.image_pb2 import Image
from gz.msgs10.pose_pb2 import Pose
from gz.msgs10.pose_v_pb2 import Pose_V
from gz.msgs10.wind_pb2 import Wind

TIMEOUT_MS = 5000
class _Flag:
    """A stop flag safe to set from a signal handler. threading.Event is not:
    a second signal arriving inside Event.set() deadlocks on its lock."""

    def __init__(self) -> None:
        self._set = False

    def set(self) -> None:
        self._set = True

    def is_set(self) -> bool:
        return self._set


_stop = _Flag()


def _on_signal(*_):
    _stop.set()


def _yaw(q) -> float:
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class PoseCache:
    """Latest pose of every model, from /world/W/pose/info."""

    def __init__(self, node, world: str, log=None):
        self.poses: dict[str, Pose] = {}
        self.lock = threading.Lock()
        self.log = log
        self.t0 = time.time()
        node.subscribe(Pose_V, f"/world/{world}/pose/info", self._cb)

    def _cb(self, msg: Pose_V) -> None:
        with self.lock:
            for p in msg.pose:
                self.poses[p.name] = p
            if self.log is not None:
                t = time.time() - self.t0
                rows = [{"t": round(t, 3), "name": p.name, "x": round(p.position.x, 3),
                         "y": round(p.position.y, 3), "z": round(p.position.z, 3),
                         "yaw": round(_yaw(p.orientation), 4)}
                        for p in msg.pose if not p.name.startswith("avians_")]
                self.log.write("".join(json.dumps(r) + "\n" for r in rows))

    def get(self, name: str, wait_s: float = 10.0) -> Pose | None:
        end = time.time() + wait_s
        while True:
            with self.lock:
                if name in self.poses:
                    return self.poses[name]
            if time.time() >= end:
                return None
            time.sleep(0.05)


def cmd_pose(node, a) -> int:
    p = PoseCache(node, a.world).get(a.model)
    if p is None:
        print(f"no pose for model {a.model} in world {a.world}", file=sys.stderr)
        return 1
    print(f"{p.position.x} {p.position.y} {p.position.z} {_yaw(p.orientation)}")
    return 0


def spawn(node, world: str, sdf: str) -> bool:
    req = EntityFactory()
    req.sdf = sdf
    ok, rep = node.request(f"/world/{world}/create", req, EntityFactory, Boolean, TIMEOUT_MS)
    return bool(ok and rep.data)


def remove(node, world: str, name: str) -> bool:
    req = Entity()
    req.name = name
    req.type = Entity.MODEL
    ok, rep = node.request(f"/world/{world}/remove", req, Entity, Boolean, TIMEOUT_MS)
    return bool(ok and rep.data)


def _self(*args: str) -> list[str]:
    """This script as a separate process. `record` makes every service request
    through one: in a process that also receives busy subscriptions (camera
    images, poses), Gazebo's Python bindings stall requests until they time out."""
    return [sys.executable, os.path.abspath(__file__), *args]


def spawn_confirmed(world: str, sdf_path: str, name: str, poses: PoseCache,
                    tries: int = 3) -> bool:
    """Spawn a model and wait until Gazebo reports its pose: the pose stream,
    not the reply to /create, says whether it exists."""
    for _ in range(tries):
        if poses.get(name, wait_s=0.0) is None:
            subprocess.run(_self("spawn", "--world", world, "--sdf", sdf_path),
                           capture_output=True, timeout=30)
        if poses.get(name, wait_s=3.0) is not None:
            return True
    return False


def cmd_spawn(node, a) -> int:
    with open(a.sdf) as f:
        return 0 if spawn(node, a.world, f.read()) else 1


def cmd_remove(node, a) -> int:
    return 0 if remove(node, a.world, a.name) else 1


def cmd_move(node, a) -> int:
    """Gazebo requests read from stdin until it closes, one per line:
    `name x y z qw qx qy qz` sets a pose, `spawn <one-line SDF>` adds a model,
    `remove <name>` removes one."""
    set_pose = f"/world/{a.world}/set_pose"
    for line in sys.stdin:
        verb, _, rest = line.strip().partition(" ")
        if verb == "spawn":
            spawn(node, a.world, rest)
        elif verb == "remove":
            remove(node, a.world, rest)
        elif len(f := line.split()) == 8:
            req = Pose()
            req.name = f[0]
            req.position.x, req.position.y, req.position.z = map(float, f[1:4])
            (req.orientation.w, req.orientation.x,
             req.orientation.y, req.orientation.z) = map(float, f[4:8])
            node.request(set_pose, req, Pose, Boolean, 1000)
    return 0


TRACK_RGB = "0.165 0.471 0.839"                  # avians' ACTUAL blue (plots/figures.py)


def _disc(name: str, x: float, y: float, z: float, radius: float) -> str:
    """A flat blue disc on the ground, visual only, as one line of SDF."""
    return (f"<?xml version='1.0'?><sdf version='1.9'><model name='{name}'><static>true</static>"
            f"<pose>{x:.3f} {y:.3f} {z:.3f} 0 0 0</pose><link name='link'><visual name='visual'>"
            f"<geometry><cylinder><radius>{radius}</radius><length>0.02</length></cylinder>"
            f"</geometry><material><ambient>{TRACK_RGB} 1</ambient><diffuse>{TRACK_RGB} 1"
            f"</diffuse></material><cast_shadows>false</cast_shadows></visual></link></model></sdf>")


def cmd_wind(node, a) -> int:
    with open(a.schedule) as f:
        schedule = json.load(f)
    pub = node.advertise(f"/world/{a.world}/wind/", Wind)
    for _ in range(50):                               # wait for WindEffects to subscribe
        if pub.has_connections():
            break
        time.sleep(0.1)
    t0 = time.time()
    try:
        for t, east, north in schedule:
            while not _stop.is_set() and time.time() - t0 < t:
                time.sleep(0.01)
            if _stop.is_set():
                break
            msg = Wind()
            msg.linear_velocity.x, msg.linear_velocity.y = east, north
            msg.enable_wind = True
            pub.publish(msg)
    finally:
        msg = Wind()
        msg.enable_wind = False
        pub.publish(msg)
        time.sleep(0.3)
    return 0


def cmd_record(node, a) -> int:
    log = open(a.poses, "w", buffering=1) if a.poses else None
    poses = PoseCache(node, a.world, log)
    remove_cam = _self("remove", "--world", a.world, "--name", a.camera_name)
    if not spawn_confirmed(a.world, a.camera_sdf, a.camera_name, poses):
        print(f"could not spawn the camera {a.camera_name}", file=sys.stderr)
        subprocess.run(remove_cam, capture_output=True, timeout=30)
        return 1
    requests = None
    if not a.fixed or a.track:
        requests = subprocess.Popen(_self("move", "--world", a.world), stdin=subprocess.PIPE,
                                    stdout=subprocess.DEVNULL, text=True, bufsize=1)
    frame_bytes = a.width * a.height * 3
    out = sys.stdout.buffer
    out_lock = threading.Lock()

    def on_image(msg: Image) -> None:
        if len(msg.data) == frame_bytes and not _stop.is_set():
            with out_lock:
                try:
                    out.write(msg.data)
                except BrokenPipeError:
                    _stop.set()

    node.subscribe(Image, a.topic, on_image)
    pitch = math.atan2(a.height_above, a.distance)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    yaw_s = None
    # --track: a disc under the drone and a breadcrumb every metre, like the
    # dot and ground track of `avians video`; removed when the recording ends
    marker, crumbs, last_crumb = f"{a.camera_name}_marker", [], None
    try:
        if a.track:
            requests.stdin.write(f"spawn {_disc(marker, 0, 0, -1, 0.8)}\n")
        while not _stop.is_set():
            p = None if requests is None else poses.get(a.model, wait_s=1.0)
            if p is not None and not a.fixed:
                yaw = _yaw(p.orientation)
                # smooth the camera's heading so it does not whip round with the drone
                yaw_s = yaw if yaw_s is None else yaw_s + 0.15 * math.atan2(
                    math.sin(yaw - yaw_s), math.cos(yaw - yaw_s))
                x = p.position.x - a.distance * math.cos(yaw_s)
                y = p.position.y - a.distance * math.sin(yaw_s)
                z = p.position.z + a.height_above
                cy, sy = math.cos(yaw_s / 2), math.sin(yaw_s / 2)
                requests.stdin.write(f"{a.camera_name} {x:.3f} {y:.3f} {z:.3f} "
                                     f"{cy * cp:.5f} {-sy * sp:.5f} {cy * sp:.5f} {sy * cp:.5f}\n")
            if p is not None and a.track:
                x, y = p.position.x, p.position.y
                requests.stdin.write(f"{marker} {x:.3f} {y:.3f} 0.09 1 0 0 0\n")
                if last_crumb is None or math.hypot(x - last_crumb[0], y - last_crumb[1]) >= 1.0:
                    crumbs.append(f"{a.camera_name}_crumb_{len(crumbs)}")
                    requests.stdin.write(f"spawn {_disc(crumbs[-1], x, y, 0.07, 0.25)}\n")
                    last_crumb = (x, y)
            time.sleep(1.0 / 30)
    except BrokenPipeError:
        print("the request helper stopped", file=sys.stderr)
    finally:
        if requests is not None:
            try:
                if a.track:
                    for name in [marker, *crumbs]:
                        requests.stdin.write(f"remove {name}\n")
                requests.stdin.close()
            except BrokenPipeError:
                pass
            try:
                requests.wait(timeout=60)
            except subprocess.TimeoutExpired:
                requests.kill()
        subprocess.run(remove_cam, capture_output=True, timeout=30)
        if log is not None:
            log.close()
        with out_lock:
            try:
                out.flush()
            except BrokenPipeError:
                pass
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pose", "spawn", "remove", "move", "wind", "record"):
        p = sub.add_parser(name)
        p.add_argument("--world", required=True)
    sub.choices["pose"].add_argument("--model", required=True)
    sub.choices["spawn"].add_argument("--sdf", required=True)
    sub.choices["remove"].add_argument("--name", required=True)
    sub.choices["wind"].add_argument("--schedule", required=True)
    r = sub.choices["record"]
    r.add_argument("--model", required=True)
    r.add_argument("--camera-sdf", required=True)
    r.add_argument("--camera-name", default="avians_chase_cam")
    r.add_argument("--topic", default="/avians/chase/image")
    r.add_argument("--poses", help="log every model's pose to this JSONL file")
    r.add_argument("--fixed", action="store_true", help="leave the camera where its SDF puts it")
    r.add_argument("--track", action="store_true",
                   help="mark the drone and its ground track with blue discs")
    r.add_argument("--width", type=int, default=1280)
    r.add_argument("--height", type=int, default=720)
    r.add_argument("--fps", type=int, default=15)
    r.add_argument("--distance", type=float, default=8.0, help="metres behind the drone")
    r.add_argument("--height-above", type=float, default=4.0, help="metres above the drone")
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)
    node = transport.Node()
    return {"pose": cmd_pose, "spawn": cmd_spawn, "remove": cmd_remove,
            "move": cmd_move, "wind": cmd_wind, "record": cmd_record}[a.cmd](node, a)


if __name__ == "__main__":
    code = main()
    try:
        sys.stdout.flush()
    except BrokenPipeError:
        pass
    os._exit(code)  # skip interpreter teardown: Gazebo's Python bindings abort in it
