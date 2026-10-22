#!/usr/bin/env python3
"""avians' hands inside AAS's simulation container (Gazebo Harmonic).

Copied there and run by avians (sim.py) with `docker exec`. Standalone on
purpose: it imports only Gazebo's Python bindings, exactly as AAS's own
simulation_resources/scripts/gz_wind.py does, never avians.

    avians_gz.py pose   --world W --model x500_0                 print "x y z yaw"
    avians_gz.py spawn  --world W --sdf FILE                      add a model
    avians_gz.py remove --world W --name NAME                     remove a model
    avians_gz.py wind   --world W --schedule FILE                 play [[t, east, north], ...]
    avians_gz.py record --world W --model x500_0 --camera-sdf FILE
                        --poses OUT.jsonl --width 1280 --height 720 --fps 15
        Spawns the chase camera, keeps it behind and above the drone, logs every
        model pose to OUT.jsonl, and writes raw RGB24 frames to stdout (avians
        pipes them into ffmpeg). Runs until SIGTERM/SIGINT, then cleans up.

Wind convention (as gz_wind.py): linear_velocity.x blows toward +X (east),
.y toward +Y (north), in m/s.
"""
from __future__ import annotations

import argparse
import json
import math
import signal
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
_stop = threading.Event()


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
        while time.time() < end:
            with self.lock:
                if name in self.poses:
                    return self.poses[name]
            time.sleep(0.05)
        return None


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


def cmd_spawn(node, a) -> int:
    with open(a.sdf) as f:
        return 0 if spawn(node, a.world, f.read()) else 1


def cmd_remove(node, a) -> int:
    return 0 if remove(node, a.world, a.name) else 1


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
    with open(a.camera_sdf) as f:
        cam_sdf = f.read()
    log = open(a.poses, "w", buffering=1)
    poses = PoseCache(node, a.world, log)
    if not spawn(node, a.world, cam_sdf):
        print("could not spawn the chase camera", file=sys.stderr)
        return 1
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
    set_pose = f"/world/{a.world}/set_pose"
    yaw_s = None
    try:
        while not _stop.is_set():
            p = poses.get(a.model, wait_s=1.0)
            if p is not None:
                yaw = _yaw(p.orientation)
                # smooth the camera's heading so it does not whip round with the drone
                yaw_s = yaw if yaw_s is None else yaw_s + 0.15 * math.atan2(
                    math.sin(yaw - yaw_s), math.cos(yaw - yaw_s))
                req = Pose()
                req.name = a.camera_name
                req.position.x = p.position.x - a.distance * math.cos(yaw_s)
                req.position.y = p.position.y - a.distance * math.sin(yaw_s)
                req.position.z = p.position.z + a.height_above
                pitch = math.atan2(a.height_above, a.distance)
                cy, sy = math.cos(yaw_s / 2), math.sin(yaw_s / 2)
                cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
                req.orientation.w, req.orientation.x = cy * cp, -sy * sp
                req.orientation.y, req.orientation.z = cy * sp, sy * cp
                node.request(set_pose, req, Pose, Boolean, 200)
            time.sleep(1.0 / 30)
    finally:
        remove(node, a.world, a.camera_name)
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
    for name in ("pose", "spawn", "remove", "wind", "record"):
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
    r.add_argument("--poses", required=True)
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
            "wind": cmd_wind, "record": cmd_record}[a.cmd](node, a)


if __name__ == "__main__":
    sys.exit(main())
