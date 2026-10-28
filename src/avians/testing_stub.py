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
    """Replays a script of (tool, args) calls. Under plan-ahead decisions a
    step may carry a condition, (tool, args, {"when": "text"}): until "text"
    appears in the decision prompt (an event, say "plan ends in"), the stub
    answers keep instead. A scripted decision the last result reports as
    rejected (say, a plan longer than the room left in the horizon) is tried
    again at the next turn, as a pilot would. Past the end of the script it
    answers in text, which the decision loop treats as keep (and the step
    loop as done)."""
    provider = "stub"
    model = "scripted"

    def __init__(self, script=None):
        self.script = list(script or DEFAULT_SCRIPT)
        self.i = 0
        self.prompt = ""
        self._calls = 0
        self._scripted = False           # was the last answer a scripted (not a waiting) call?

    def start(self, system: str, tools: list[dict], prompt: str) -> None:
        # a decision-mode turn restarts the conversation: keep our place
        self.prompt = prompt
        if "keep" not in {t["name"] for t in tools}:
            self.i = 0
        elif self._scripted and "\nstatus: rejected" in prompt and self.i > 0:
            self.i -= 1                  # rejected: try the same decision again
            self._retry = True

    def _events(self) -> str:
        """The events since the last decision, from the prompt (not the plan's
        upcoming ones, which the prompt lists too)."""
        for line in self.prompt.splitlines():
            if line.startswith("events since your last call:"):
                return line
        return ""

    def step(self) -> Reply:
        usage = {"input_tokens": 100, "output_tokens": 20}
        self._calls += 1
        if self.i >= len(self.script):
            return Reply([{"type": "text", "text": "Mission complete."}], "end_turn", usage)
        entry = self.script[self.i]
        name, args = entry[0], entry[1]
        when = entry[2].get("when") if len(entry) > 2 else None
        retry, self._retry = getattr(self, "_retry", False), False
        self._scripted = False
        if when and not retry and when not in self._events():
            return Reply([{"type": "tool_use", "id": f"stub_{self._calls:03d}", "name": "keep",
                           "input": {}}], "tool_use", usage)
        self.i += 1
        self._scripted = True
        return Reply([{"type": "tool_use", "id": f"stub_{self._calls:03d}", "name": name,
                       "input": args}], "tool_use", usage)

    def add_tool_results(self, results: list[tuple[str, str]]) -> None:
        pass
