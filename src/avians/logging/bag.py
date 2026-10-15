"""rosbag2 ground truth, recorded alongside the agent trajectory.

The bag is NOT what the agent saw. It is the flight recorder: used for replay,
validation, debugging, and computing reward signals later.

Records an explicit topic list (from config/topics.yaml), never --all.
"""
from __future__ import annotations

import shutil
import signal
import subprocess
from pathlib import Path


class BagRecorder:
    def __init__(self, episode_dir: Path, topics: list[str], enabled: bool = True):
        self.dir = Path(episode_dir) / "bag"
        self.topics = topics
        self.enabled = enabled and shutil.which("ros2") is not None
        self.proc: subprocess.Popen | None = None

    def start(self) -> bool:
        if not self.enabled or not self.topics:
            return False
        self.dir.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ros2", "bag", "record", "-o", str(self.dir), *self.topics]
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return True

    def stop(self, timeout: float = 10.0) -> None:
        if self.proc is None:
            return
        # SIGINT, not kill: rosbag2 needs to finalize its metadata file.
        self.proc.send_signal(signal.SIGINT)
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.proc = None

    def __enter__(self) -> "BagRecorder":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
