"""Post-flight plots: slide-ready figures + summary metrics for one episode.

    plot_episode(data/episodes/ep_0004)  ->  ep_0004/plots/*.png, summary.json, summary.md

matplotlib is imported only when plotting, so the rest of the package (and the
flight itself) never depends on it.
"""
from __future__ import annotations

import json
from pathlib import Path


def plot_episode(episode_dir: Path, geofence_m: float | None = None, dpi: int = 200,
                 formats: tuple[str, ...] = ("png",)) -> tuple[list[Path], dict]:
    from .figures import headline, save_all
    from .flight import compute_metrics, load_episode

    episode_dir = Path(episode_dir)
    f = load_episode(episode_dir)
    metrics = compute_metrics(f)
    out = episode_dir / "plots"
    written = save_all(f, metrics, out, geofence_m, dpi=dpi, formats=formats)

    (out / "summary.json").write_text(json.dumps(metrics, indent=2))
    (out / "summary.md").write_text(_summary_md(f, metrics, headline(metrics)))
    written += [out / "summary.json", out / "summary.md"]
    return written, metrics


def _summary_md(f, m: dict, headline: str) -> str:
    lines = [
        f"# {m['mission']} ({m['episode']}): {'PASS' if m['success'] else 'CHECK'}",
        "",
        headline,
        "",
        f"- Model: {m['model']} · backend: {m['backend']} · data: {m['data_source']}",
        f"- Mission time {m['mission_duration_s']} s, of which the model spent "
        f"{m['model_time_s']} s thinking; {m['tokens']['input']} input / "
        f"{m['tokens']['output']} output tokens",
        "",
        "## Checks",
        *[f"- [{'x' if ok else ' '}] {name}" for name, ok in m["checks"].items()],
    ]
    if m["arrivals"]:
        lines += ["", "## Arrival error", "", "| action | target (east, north, alt) | error (m) |",
                  "|---|---|---|"]
        lines += [f"| {a['tool']} | {tuple(a['target'])} | {a['error_m']} |"
                  for a in m["arrivals"]]
    if m.get("stl"):
        stl = m["stl"]
        lines += ["", "## STL specification", "", f"`{stl['spec']}`", "",
                  f"**{'Satisfied' if stl['satisfied'] else 'Violated'}**, robustness "
                  f"{stl['robustness']} (scored on {stl.get('trace_source', '?')})", "",
                  "| clause | robustness | satisfied |", "|---|---|---|"]
        lines += [f"| `{c['clause']}` | {c['robustness']} | {'yes' if c['satisfied'] else 'no'} |"
                  for c in stl["clauses"]]
        if stl.get("padded_s"):
            lines += ["", f"The flight ended at {stl['flight_end_s']} s; the last state was "
                          f"held for {stl['padded_s']} s to cover the spec's "
                          f"{stl['horizon_s']:g} s horizon."]
    if m["holds"]:
        lines += ["", "## Position hold", ""]
        lines += [f"- {h['duration_s']} s hold: max drift {h['max_drift_m']} m, "
                  f"RMS {h['rms_drift_m']} m" for h in m["holds"]]
    if m["safety_rejections"]:
        lines += ["", "## Safety rejections", ""]
        lines += [f"- `{r['tool']} {r['args']}`: {r['reason']}" for r in m["safety_rejections"]]
    if f.final_text:
        lines += ["", "## Agent's final report", "", f.final_text]
    return "\n".join(lines) + "\n"
