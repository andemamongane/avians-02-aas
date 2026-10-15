"""Each pilot adapter against a fake SDK client: the request it builds, how it
reads the reply, and what it sends back on the next turn. No network."""
from __future__ import annotations

import json
from types import SimpleNamespace as NS

import pytest

from avians.pilots import (
    AnthropicPilot, OpenAIChatPilot, OpenAIResponsesPilot, PilotConfigError, make_pilot,
)
from avians.schema import TOOL_SPECS


class Recorder:
    """Fake SDK endpoint: records each request, returns queued replies."""

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def create(self, **kw):
        self.calls.append(json.loads(json.dumps(kw, default=lambda o: o.__dict__)))
        return self.replies.pop(0)


class Item(NS):
    """An SDK object with model_dump, like openai's pydantic models."""

    def model_dump(self, exclude_none=False):
        return {k: v for k, v in vars(self).items() if not (exclude_none and v is None)}


# --- Anthropic ----------------------------------------------------------------
def test_anthropic_round_trip():
    rec = Recorder([NS(content=[Item(type="text", text="Taking off."),
                                Item(type="tool_use", id="tu1", name="takeoff",
                                     input={"altitude": 5})],
                       stop_reason="tool_use", usage=NS(input_tokens=10, output_tokens=5))])
    p = AnthropicPilot("claude-x", "k", client=NS(messages=rec))
    p.start("sys", TOOL_SPECS, "fly")
    r = p.step()
    assert r.tool_uses == [{"type": "tool_use", "id": "tu1", "name": "takeoff",
                            "input": {"altitude": 5}}]
    assert rec.calls[0]["system"][0]["cache_control"] == {"type": "ephemeral"}
    p.add_tool_results([("tu1", "status: arrived")])
    assert p.messages[-1]["content"][0] == {"type": "tool_result", "tool_use_id": "tu1",
                                            "content": "status: arrived"}


# --- OpenAI-compatible chat (DeepSeek) ------------------------------------------
def _chat_reply(content="", reasoning=None, calls=(), finish="tool_calls"):
    msg = NS(content=content, reasoning_content=reasoning, tool_calls=[
        NS(id=i, function=NS(name=n, arguments=a)) for i, n, a in calls] or None)
    return NS(choices=[NS(message=msg, finish_reason=finish)],
              usage=NS(prompt_tokens=20, completion_tokens=7))


def test_chat_tools_reasoning_and_results():
    rec = Recorder([
        _chat_reply(reasoning="Need altitude first.",
                    calls=[("c1", "takeoff", '{"altitude": 5}')]),
        _chat_reply(content="Done.", finish="stop"),
    ])
    p = OpenAIChatPilot("deepseek-x", "k", client=NS(chat=NS(completions=rec)),
                        extra_body={"thinking": {"type": "enabled"}})
    p.start("sys", TOOL_SPECS, "fly")
    r = p.step()
    req = rec.calls[0]
    assert req["messages"][0] == {"role": "system", "content": "sys"}
    assert req["tools"][0]["type"] == "function"
    assert req["tools"][0]["function"]["parameters"] == TOOL_SPECS[0]["input_schema"]
    assert req["extra_body"] == {"thinking": {"type": "enabled"}}
    assert [b["type"] for b in r.content] == ["thinking", "tool_use"]
    assert r.tool_uses[0]["input"] == {"altitude": 5}
    assert r.usage == {"input_tokens": 20, "output_tokens": 7}

    p.add_tool_results([("c1", "status: arrived")])
    p.step()
    sent = rec.calls[1]["messages"]
    # DeepSeek returns 400 unless reasoning_content goes back with the turn
    assert sent[2]["reasoning_content"] == "Need altitude first."
    assert sent[2]["tool_calls"][0]["function"]["arguments"] == '{"altitude": 5}'
    assert sent[3] == {"role": "tool", "tool_call_id": "c1", "content": "status: arrived"}


def test_chat_bad_json_becomes_a_rejectable_argument():
    rec = Recorder([_chat_reply(calls=[("c1", "takeoff", "{altitude: 5")])])
    p = OpenAIChatPilot("m", "k", client=NS(chat=NS(completions=rec)))
    p.start("sys", TOOL_SPECS, "fly")
    assert "_invalid_arguments" in p.step().tool_uses[0]["input"]


# --- OpenAI Responses (Codex) ---------------------------------------------------
def test_responses_round_trip_passes_reasoning_back():
    out = [Item(type="reasoning", id="rs1", encrypted_content="xyz",
                summary=[{"type": "summary_text", "text": "Climb first."}]),
           Item(type="function_call", id="fc1", call_id="call1", name="takeoff",
                arguments='{"altitude": 5}', status="completed")]
    rec = Recorder([
        NS(output=out, usage=NS(input_tokens=30, output_tokens=9), status="completed"),
        NS(output=[Item(type="message", role="assistant",
                        content=[{"type": "output_text", "text": "Landed."}])],
           usage=NS(input_tokens=40, output_tokens=3), status="completed"),
    ])
    p = OpenAIResponsesPilot("gpt-x", "k", client=NS(responses=rec), reasoning_effort="medium")
    p.start("sys", TOOL_SPECS, "fly")
    r = p.step()
    req = rec.calls[0]
    assert req["instructions"] == "sys" and req["store"] is False
    assert req["parallel_tool_calls"] is False
    assert req["tools"][0] == {"type": "function", "name": "get_state",
                               "description": TOOL_SPECS[0]["description"],
                               "parameters": TOOL_SPECS[0]["input_schema"]}
    assert req["reasoning"]["effort"] == "medium"
    assert [b["type"] for b in r.content] == ["thinking", "tool_use"]
    assert r.tool_uses[0]["id"] == "call1"

    p.add_tool_results([("call1", "status: arrived")])
    r2 = p.step()
    sent = rec.calls[1]["input"]
    assert sent[1]["type"] == "reasoning" and sent[1]["encrypted_content"] == "xyz"
    assert sent[3] == {"type": "function_call_output", "call_id": "call1",
                       "output": "status: arrived"}
    assert r2.content == [{"type": "text", "text": "Landed."}] and r2.tool_uses == []


# --- config -----------------------------------------------------------------------
def test_make_pilot_needs_its_key(monkeypatch):
    cfg = {"pilots": {"ds": {"provider": "openai_chat", "model": "m", "key_env": "DS_KEY"}}}
    monkeypatch.delenv("DS_KEY", raising=False)
    with pytest.raises(PilotConfigError, match="DS_KEY"):
        make_pilot("ds", cfg)
    with pytest.raises(PilotConfigError, match="unknown pilot"):
        make_pilot("nope", cfg)


def test_shipped_pilot_config_is_complete():
    import yaml
    from conftest import REPO
    cfg = yaml.safe_load((REPO / "config" / "pilots.yaml").read_text())
    assert cfg["default"] in cfg["pilots"]
    assert set(cfg["showcase"]) <= set(cfg["pilots"])
    for name, p in cfg["pilots"].items():
        assert p["provider"] in ("anthropic", "openai_chat", "openai_responses"), name
        assert p.get("model") and p.get("key_env") and p.get("label"), name
