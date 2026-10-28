"""Episodes -> a supervised fine-tuning dataset of pilot decisions.

Under plan-ahead control every model call is self-contained (runner.py's
decision loop): its request holds the whole decision prompt, so each
(request, reply) pair is one training example on its own: the system prompt,
the tool definitions, the prompt the teacher saw, and what it answered (its
decision, with the reason and notes it gave). A turn that started with free
check_path lookups contributes one example per call, each with the lookups so
far in its messages.

Which episodes count is the selection rule. The defaults keep only what a
student should copy:

    plan-ahead decision episodes (schema 4, control: plan)
    the spec satisfied (and, optionally, by a margin)
    the host never suspended mid-flight
    not the fake backend (noise-free physics) unless include_fake
    not scripted dry runs

Every example carries its episode's outcome (robustness, flow and decision
metrics), so finer selection can happen later without re-exporting. The
manifest records what was included, what was excluded, and why.

The format is provider-neutral: messages in the canonical content-block form
the trajectory log uses (text, tool_use, tool_result blocks).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .logging.adp import load_jsonl


@dataclass
class Selection:
    include_fake: bool = False
    require_satisfied: bool = True
    min_robustness: float | None = None
    missions: tuple[str, ...] = ()           # empty: every mission
    pilots: tuple[str, ...] = ()             # empty: every pilot


def _why_excluded(meta: dict, stl: dict, sel: Selection) -> str | None:
    if meta.get("dry_run"):
        return "scripted dry run"
    if meta.get("control") != "plan" or not meta.get("decisions"):
        return "not plan-ahead decisions"
    if meta.get("schema_version", 0) < 4:
        return f"schema {meta.get('schema_version')} (decision format needs 4+)"
    if meta.get("backend") == "fake" and not sel.include_fake:
        return "fake backend (noise-free physics)"
    if meta.get("host_suspended_s"):
        return f"host suspended {meta['host_suspended_s']} s mid-flight"
    if meta.get("terminated_by") in ("crashed", "api_error"):
        return f"ended by {meta['terminated_by']}"
    if sel.missions and meta.get("mission") not in sel.missions:
        return "mission not selected"
    if sel.pilots and meta.get("pilot") not in sel.pilots:
        return "pilot not selected"
    if sel.require_satisfied and not stl.get("satisfied"):
        return "spec not satisfied"
    rob = stl.get("robustness")
    if sel.min_robustness is not None and (not isinstance(rob, (int, float))
                                           or rob < sel.min_robustness):
        return f"robustness {rob} below {sel.min_robustness}"
    return None


def episode_examples(ep: Path) -> list[dict]:
    """Every model call of one episode as a training example."""
    records = load_jsonl(ep / "trajectory.jsonl")
    meta = json.loads((ep / "meta.json").read_text())
    stl = json.loads((ep / "stl.json").read_text()) if (ep / "stl.json").exists() else {}
    summary = ep / "plots" / "summary.json"
    metrics = json.loads(summary.read_text()) if summary.exists() else {}
    system = next((r for r in records if r["kind"] == "system"), {})
    t0 = next((r["t_wall"] for r in records if r["kind"] == "mission"), records[0]["t_wall"])
    outcome = {
        "satisfied": stl.get("satisfied"), "robustness": stl.get("robustness"),
        "terminated_by": meta.get("terminated_by"),
        "flow": metrics.get("flow"), "decisions": metrics.get("decisions"),
    }
    out, request = [], None
    for r in records:
        if r["kind"] == "request":
            request = r
        elif r["kind"] == "response" and request is not None:
            out.append({
                "episode": meta.get("episode", ep.name),
                "mission": meta.get("mission"),
                "turn": len(out),
                "t_mission": round(request["t_wall"] - t0, 2),     # when it was asked
                "system": system.get("prompt"),
                "tools": system.get("tools"),
                "messages": request["messages"],
                "target": r["content"],
                "teacher": {"pilot": meta.get("pilot"), "model": meta.get("model"),
                            "provider": meta.get("provider"),
                            "latency_s": r.get("latency_s")},
                "setting": {"backend": meta.get("backend"), "clock": meta.get("clock"),
                            "decisions": meta.get("decisions"), "envelope": meta.get("envelope"),
                            "environment": meta.get("environment"), "sensor": meta.get("sensor"),
                            "logic": meta.get("logic"), "schema_version": meta.get("schema_version")},
                "outcome": outcome,
            })
            request = None
    return out


def export(episodes_dir: Path, out: Path, sel: Selection | None = None) -> dict:
    """Write the selected episodes' examples to out (JSONL) and a manifest
    next to it; return the manifest."""
    sel = sel or Selection()
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    included, excluded, n = [], {}, 0
    with out.open("w", encoding="utf-8") as f:
        for ep in sorted(Path(episodes_dir).glob("ep_*")):
            if not (ep / "meta.json").exists() or not (ep / "trajectory.jsonl").exists():
                excluded[ep.name] = "incomplete episode"
                continue
            meta = json.loads((ep / "meta.json").read_text())
            stl = json.loads((ep / "stl.json").read_text()) if (ep / "stl.json").exists() else {}
            why = _why_excluded(meta, stl, sel)
            if why:
                excluded[ep.name] = why
                continue
            examples = episode_examples(ep)
            for ex in examples:
                f.write(json.dumps(ex, default=str) + "\n")
            n += len(examples)
            included.append({"episode": ep.name, "mission": meta.get("mission"),
                             "pilot": meta.get("pilot"), "examples": len(examples),
                             "robustness": stl.get("robustness")})
    manifest = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "file": out.name, "examples": n,
        "selection": vars(sel) | {"missions": list(sel.missions), "pilots": list(sel.pilots)},
        "included": included, "excluded": excluded,
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
