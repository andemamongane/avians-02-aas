"""Convert an episode's JSONL into an Agent Data Protocol style trajectory.

ADP represents an episode as an alternating sequence of actions and
observations, plus task-level metadata. We emit that shape here.

NOTE: verify field names against the reference implementation
(github.com/neulab/agent-data-protocol) before training on the output; this
writes our best reading of the format and keeps the raw JSONL alongside it so
nothing is lost if the mapping needs to change.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..schema import SCHEMA_VERSION

ADP_FORMAT_NOTE = "adp-style v0 (verify against neulab/agent-data-protocol)"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def to_adp(records: list[dict[str, Any]], meta: dict | None = None) -> dict[str, Any]:
    """Build one ADP-style trajectory from one episode's records."""
    system = next((r for r in records if r["kind"] == "system"), None)
    mission = next((r for r in records if r["kind"] == "mission"), None)

    content: list[dict[str, Any]] = []
    pending: dict[str, dict[str, Any]] = {}

    for r in records:
        if r["kind"] == "action":
            entry = {
                "type": "action",
                "action_type": "tool",
                "tool": r["tool"],
                "arguments": r["args"],
                "timestamp": r["t_wall"],
            }
            pending[r["tool_use_id"]] = entry
            content.append(entry)
        elif r["kind"] == "observation":
            obs = r["observation"]
            content.append({
                "type": "observation",
                "observation_type": "text",
                "content": r["text"],
                "status": obs.get("status"),
                "state": {k: obs.get(k) for k in
                          ("east", "north", "alt", "v_east", "v_north", "v_up",
                           "heading_deg", "mode", "armed", "battery_pct", "inside")},
                "timestamp": r["t_wall"],
            })

    return {
        "format": ADP_FORMAT_NOTE,
        "schema_version": SCHEMA_VERSION,
        "task": {
            "name": mission["name"] if mission else "unknown",
            "instruction": mission["prompt"] if mission else "",
            "system_prompt": system["prompt"] if system else "",
        },
        "content": content,
        "metadata": meta or {},
    }


def convert_episode(episode_dir: Path) -> Path:
    episode_dir = Path(episode_dir)
    records = load_jsonl(episode_dir / "trajectory.jsonl")
    meta_path = episode_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    adp = to_adp(records, meta)
    out = episode_dir / "adp.json"
    out.write_text(json.dumps(adp, indent=2))
    return out


# --- validation ------------------------------------------------------------
def validate_episode(episode_dir: Path) -> list[str]:
    """Return a list of problems. Empty list means the episode is usable."""
    d = Path(episode_dir)
    problems: list[str] = []

    traj = d / "trajectory.jsonl"
    if not traj.exists():
        return [f"{d.name}: no trajectory.jsonl"]

    records = load_jsonl(traj)
    if not records:
        return [f"{d.name}: trajectory.jsonl is empty"]

    versions = {r.get("schema_version") for r in records}
    if versions != {SCHEMA_VERSION}:
        problems.append(f"{d.name}: mixed/unknown schema versions {versions}")

    times = [r["t_wall"] for r in records]
    if times != sorted(times):
        problems.append(f"{d.name}: timestamps are not monotonic")

    actions = [r for r in records if r["kind"] == "action"]
    observations = [r for r in records if r["kind"] == "observation"]
    if len(actions) != len(observations):
        problems.append(
            f"{d.name}: {len(actions)} actions but {len(observations)} observations"
        )
    ids_a = {r["tool_use_id"] for r in actions}
    ids_o = {r["tool_use_id"] for r in observations}
    if ids_a != ids_o:
        problems.append(f"{d.name}: action/observation ids do not match")

    if not (d / "meta.json").exists():
        problems.append(f"{d.name}: no meta.json")

    text = traj.read_text()
    if "sk-ant" in text:
        problems.append(f"{d.name}: CREDENTIAL LEAK — an API key appears in the log")

    return problems
