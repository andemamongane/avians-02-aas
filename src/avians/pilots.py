"""Pilots: the AI agents that can fly a mission, behind one small interface.

    pilot.start(system, tools, prompt)    begin a conversation
    pilot.step() -> Reply                  one model call
    pilot.add_tool_results([(id, text)])   feed back what the tools returned

Each adapter keeps the conversation in its provider's NATIVE format (some
providers need things passed back verbatim: DeepSeek's reasoning_content,
OpenAI's reasoning items), and returns a Reply in one CANONICAL format that the
runner logs for every pilot alike:

    {"type": "text", "text": ...}
    {"type": "tool_use", "id": ..., "name": ..., "input": {...}}
    {"type": "thinking", "text": ...}        reasoning, when the provider exposes it

So episodes from different pilots have the same shape and can be compared or
mixed in one dataset; meta.json records which pilot flew.

Providers:
    anthropic          Claude, Messages API (prompt caching on system + tools)
    openai_chat        any OpenAI-compatible Chat Completions API (DeepSeek)
    openai_responses   OpenAI Responses API (Codex's models; Codex is Responses-only)

SDKs are imported lazily, so the rest of the package works without them.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Reply:
    content: list[dict[str, Any]]
    stop_reason: str
    usage: dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0})

    @property
    def tool_uses(self) -> list[dict[str, Any]]:
        return [b for b in self.content if b.get("type") == "tool_use"]


class PilotConfigError(RuntimeError):
    """A pilot can't be created (unknown name, missing key or SDK)."""


def _parse_args(raw: str | None) -> dict[str, Any]:
    """Tool arguments arrive as a JSON string from OpenAI-style APIs. Bad JSON
    becomes an argument the validator rejects, so the model sees why."""
    try:
        val = json.loads(raw or "{}")
        return val if isinstance(val, dict) else {"_invalid_arguments": raw}
    except json.JSONDecodeError:
        return {"_invalid_arguments": raw}


# ------------------------------------------------------------------ Anthropic
class AnthropicPilot:
    provider = "anthropic"

    def __init__(self, model: str, api_key: str, max_tokens: int = 1024, client=None, **_):
        if client is None:
            import anthropic
            client = anthropic.Anthropic(api_key=api_key)
        self.client, self.model, self.max_tokens = client, model, max_tokens

    def start(self, system: str, tools: list[dict], prompt: str) -> None:
        self.system, self.tools = system, tools
        self.messages: list[dict] = [{"role": "user", "content": prompt}]

    def step(self) -> Reply:
        r = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            # identical for every call in an episode, so cached
            system=[{"type": "text", "text": self.system, "cache_control": {"type": "ephemeral"}}],
            tools=self.tools, messages=self.messages,
        )
        native = [b.model_dump() if hasattr(b, "model_dump") else b for b in r.content]
        self.messages.append({"role": "assistant", "content": native})
        content = []
        for b in native:
            if b.get("type") == "text":
                content.append({"type": "text", "text": b.get("text", "")})
            elif b.get("type") == "tool_use":
                content.append({"type": "tool_use", "id": b["id"], "name": b["name"],
                                "input": b.get("input") or {}})
            elif b.get("type") == "thinking":
                content.append({"type": "thinking", "text": b.get("thinking", "")})
        u = r.usage
        return Reply(content, r.stop_reason or "", {
            "input_tokens": getattr(u, "input_tokens", 0) or 0,
            "output_tokens": getattr(u, "output_tokens", 0) or 0,
        })

    def add_tool_results(self, results: list[tuple[str, str]]) -> None:
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": uid, "content": text} for uid, text in results]})


# ------------------------------------------------- OpenAI-compatible chat (DeepSeek)
class OpenAIChatPilot:
    provider = "openai_chat"

    def __init__(self, model: str, api_key: str, base_url: str | None = None,
                 max_tokens: int = 4096, extra_body: dict | None = None, client=None, **_):
        if client is None:
            import openai
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.client, self.model, self.max_tokens = client, model, max_tokens
        self.extra_body = extra_body or {}

    def start(self, system: str, tools: list[dict], prompt: str) -> None:
        self.tools = [{"type": "function", "function": {
            "name": t["name"], "description": t["description"], "parameters": t["input_schema"],
        }} for t in tools]
        self.messages: list[dict] = [{"role": "system", "content": system},
                                     {"role": "user", "content": prompt}]

    def step(self) -> Reply:
        r = self.client.chat.completions.create(
            model=self.model, messages=self.messages, tools=self.tools,
            max_tokens=self.max_tokens, extra_body=self.extra_body or None,
        )
        msg = r.choices[0].message
        native: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        reasoning = getattr(msg, "reasoning_content", None)
        if reasoning:
            # DeepSeek: must be passed back on every later request that has tools
            native["reasoning_content"] = reasoning
        calls = msg.tool_calls or []
        if calls:
            native["tool_calls"] = [{"id": c.id, "type": "function", "function": {
                "name": c.function.name, "arguments": c.function.arguments}} for c in calls]
        self.messages.append(native)

        content: list[dict] = []
        if reasoning:
            content.append({"type": "thinking", "text": reasoning})
        if msg.content:
            content.append({"type": "text", "text": msg.content})
        for c in calls:
            content.append({"type": "tool_use", "id": c.id, "name": c.function.name,
                            "input": _parse_args(c.function.arguments)})
        u = r.usage
        return Reply(content, "tool_use" if calls else (r.choices[0].finish_reason or ""), {
            "input_tokens": getattr(u, "prompt_tokens", 0) or 0,
            "output_tokens": getattr(u, "completion_tokens", 0) or 0,
        })

    def add_tool_results(self, results: list[tuple[str, str]]) -> None:
        for uid, text in results:
            self.messages.append({"role": "tool", "tool_call_id": uid, "content": text})


# ------------------------------------------------------- OpenAI Responses (Codex)
class OpenAIResponsesPilot:
    provider = "openai_responses"

    def __init__(self, model: str, api_key: str, base_url: str | None = None,
                 max_tokens: int = 8192, reasoning_effort: str | None = None,
                 client=None, **_):
        if client is None:
            import openai
            client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.client, self.model, self.max_tokens = client, model, max_tokens
        self.reasoning_effort = reasoning_effort

    def start(self, system: str, tools: list[dict], prompt: str) -> None:
        self.system = system
        self.tools = [{"type": "function", "name": t["name"], "description": t["description"],
                       "parameters": t["input_schema"]} for t in tools]
        self.input: list[dict] = [{"role": "user", "content": prompt}]

    def step(self) -> Reply:
        kw: dict[str, Any] = {}
        if self.reasoning_effort:
            kw["reasoning"] = {"effort": self.reasoning_effort, "summary": "auto"}
        r = self.client.responses.create(
            model=self.model, instructions=self.system, input=self.input, tools=self.tools,
            parallel_tool_calls=False, max_output_tokens=self.max_tokens,
            # stateless: reasoning comes back encrypted and is passed back verbatim
            store=False, include=["reasoning.encrypted_content"], **kw,
        )
        content: list[dict] = []
        for item in r.output:
            native = item.model_dump(exclude_none=True) if hasattr(item, "model_dump") else item
            self.input.append(native)
            kind = native.get("type")
            if kind == "message":
                text = "".join(p.get("text", "") for p in native.get("content", [])
                               if p.get("type") == "output_text")
                if text:
                    content.append({"type": "text", "text": text})
            elif kind == "function_call":
                content.append({"type": "tool_use", "id": native["call_id"], "name": native["name"],
                                "input": _parse_args(native.get("arguments"))})
            elif kind == "reasoning":
                summary = " ".join(s.get("text", "") for s in native.get("summary", []))
                if summary:
                    content.append({"type": "thinking", "text": summary})
        u = r.usage
        called = any(b["type"] == "tool_use" for b in content)
        return Reply(content, "tool_use" if called else (getattr(r, "status", "") or ""), {
            "input_tokens": getattr(u, "input_tokens", 0) or 0,
            "output_tokens": getattr(u, "output_tokens", 0) or 0,
        })

    def add_tool_results(self, results: list[tuple[str, str]]) -> None:
        for uid, text in results:
            self.input.append({"type": "function_call_output", "call_id": uid, "output": text})


PROVIDERS = {
    "anthropic": AnthropicPilot,
    "openai_chat": OpenAIChatPilot,
    "openai_responses": OpenAIResponsesPilot,
}


def make_pilot(name: str, pilots_cfg: dict, client=None):
    """Create a pilot from config/pilots.yaml. Fails early, with a fix, when the
    key or the SDK is missing, so nothing takes off without a pilot."""
    pilots = pilots_cfg.get("pilots", {})
    if name not in pilots:
        raise PilotConfigError(f"unknown pilot '{name}'; choose from: {', '.join(pilots)}")
    cfg = dict(pilots[name])
    provider = cfg.pop("provider")
    key_env = cfg.pop("key_env", None)
    cfg.pop("label", None)
    api_key = os.environ.get(key_env, "") if key_env else ""
    if client is None and key_env and not api_key:
        raise PilotConfigError(f"pilot '{name}' needs {key_env}: add it to .env, then "
                               f"recreate the container (Rebuild Container, or ./scripts/up.sh)")
    try:
        return PROVIDERS[provider](api_key=api_key, client=client, **cfg)
    except ImportError as e:
        raise PilotConfigError(f"pilot '{name}' needs the {e.name} package (pip install {e.name})")


def pilot_label(name: str, pilots_cfg: dict) -> str:
    """Human name for figures and videos, e.g. 'Claude Sonnet 5.5'."""
    p = pilots_cfg.get("pilots", {}).get(name, {})
    return p.get("label") or name
