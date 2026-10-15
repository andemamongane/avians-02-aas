"""A scripted stand-in pilot.

Lets you exercise the entire pipeline (runner, tools, safety, skills, logging,
ADP conversion, STL scoring) with no API key and no network:
`avians fly <mission> --dry-run` flies the mission's dry_run_script.
"""
from __future__ import annotations

from .pilots import Reply

DEFAULT_SCRIPT = [
    ("takeoff", {"altitude": 5.0}),
    ("reposition", {"east": 8.0, "north": 0.0, "altitude": 5.0}),
    ("reposition", {"east": 8.0, "north": 8.0, "altitude": 5.0}),
    ("hover", {"duration": 3.0}),
    ("land", {}),
]


class StubPilot:
    provider = "stub"
    model = "scripted"

    def __init__(self, script=None):
        self.script = list(script or DEFAULT_SCRIPT)
        self.i = 0

    def start(self, system: str, tools: list[dict], prompt: str) -> None:
        self.i = 0

    def step(self) -> Reply:
        usage = {"input_tokens": 100, "output_tokens": 20}
        if self.i >= len(self.script):
            return Reply([{"type": "text", "text": "Mission complete."}], "end_turn", usage)
        name, args = self.script[self.i]
        self.i += 1
        return Reply([{"type": "tool_use", "id": f"stub_{self.i:03d}", "name": name,
                       "input": args}], "tool_use", usage)

    def add_tool_results(self, results: list[tuple[str, str]]) -> None:
        pass
