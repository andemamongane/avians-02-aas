"""Verbatim episode log.

Every API request and response is written AS SENT/RECEIVED, before anything is
parsed. The serialized conversation is the training data; a reconstruction of
it is not.

One JSON object per line. Never write credentials here (see tests).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from ..schema import SCHEMA_VERSION


class TrajectoryLogger:
    def __init__(self, episode_dir: Path):
        self.dir = Path(episode_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "trajectory.jsonl"
        self._f = self.path.open("a", encoding="utf-8")
        self._seq = 0

    # ------------------------------------------------------------------
    def _write(self, kind: str, **payload: Any) -> None:
        self._seq += 1
        record = {
            "seq": self._seq,
            "kind": kind,
            "t_wall": time.time(),
            "schema_version": SCHEMA_VERSION,
            **payload,
        }
        self._f.write(json.dumps(record, default=str) + "\n")
        self._f.flush()          # flush every line: a crashed episode is still data

    # --- event types ---------------------------------------------------
    def system(self, prompt: str, tools: list[dict]) -> None:
        self._write("system", prompt=prompt, tools=tools)

    def mission(self, name: str, prompt: str, config: dict) -> None:
        self._write("mission", name=name, prompt=prompt, config=config)

    def request(self, messages: list[dict], model: str) -> None:
        self._write("request", model=model, messages=messages)

    def response(self, content: list[dict], stop_reason: str | None, usage: dict) -> None:
        self._write("response", content=content, stop_reason=stop_reason, usage=usage)

    def tool_call(self, name: str, args: dict, tool_use_id: str) -> None:
        self._write("action", tool=name, args=args, tool_use_id=tool_use_id)

    def observation(self, obs_dict: dict, text: str, tool_use_id: str) -> None:
        self._write("observation", observation=obs_dict, text=text, tool_use_id=tool_use_id)

    def note(self, message: str, **extra: Any) -> None:
        self._write("note", message=message, **extra)

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()
