"""The live demo's endpoint, tested offline with a scripted stand-in for the model client."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("demo_ask", ROOT / "demo" / "api" / "ask.py")
assert spec and spec.loader
ask = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ask)


class FakeClient:
    """Replays scripted responses and records each request it was sent."""

    def __init__(self, *responses):
        self.responses, self.requests = list(responses), []
        self.messages = NS(create=self.create)

    def create(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


def reply(*blocks, stop="end_turn", tokens=(1000, 100)):
    return NS(
        content=list(blocks),
        stop_reason=stop,
        usage=NS(input_tokens=tokens[0], output_tokens=tokens[1]),
    )


def search(query, call_id="t1"):
    return NS(type="tool_use", id=call_id, name="search_docs", input={"query": query})


def text(value):
    return NS(type="text", text=value)


@pytest.fixture(autouse=True)
def fresh_limits():
    ask._recent.clear()
    ask._day.update(date="", count=0)


def test_answer_runs_the_search_and_reports_sources_and_cost():
    client = FakeClient(
        reply(NS(type="thinking", thinking=""), search("E4291"), stop="tool_use"),
        reply(text("You exceeded the limit [rate-limits#0].")),
    )
    out = ask.answer("What does error E4291 mean?", client)
    assert out["answer"] == "You exceeded the limit [rate-limits#0]."
    assert out["searches"][0]["query"] == "E4291" and "rate-limits#0" in out["searches"][0]["hits"]
    assert [s["id"] for s in out["sources"]] == ["rate-limits#0"]  # only cited passages
    assert (out["input_tokens"], out["output_tokens"]) == (2000, 200)
    assert out["cost_usd"] == pytest.approx(2000 * 0.10 / 1e6 + 200 * 0.50 / 1e6)
    # the second request carries the first turn unchanged, then the passages as a tool result
    second = client.requests[1]["messages"]
    assert second[1]["content"][0].type == "thinking"
    result = second[2]["content"][0]
    assert result["tool_use_id"] == "t1" and "[rate-limits#0]" in result["content"]
    assert client.requests[0]["model"] == "claude-haiku-5-5"


def test_answer_stops_after_the_turn_limit_and_handles_a_refusal():
    looping = FakeClient(*[reply(search("zzzz"), stop="tool_use") for _ in range(ask.MAX_TURNS)])
    out = ask.answer("loop forever", looping)
    assert len(looping.requests) == ask.MAX_TURNS and "did not finish" in out["answer"]
    assert out["searches"][0]["hits"] == []
    assert "declined" in ask.answer("x", FakeClient(reply(stop="refusal")))["answer"]


def test_handle_validates_input_before_any_model_call():
    client = FakeClient()
    assert ask.handle(b"not json", "1.1.1.1", client)[0] == 400
    assert ask.handle(b"[1]", "1.1.1.1", client)[0] == 400
    assert ask.handle(b'{"question": "  "}', "1.1.1.1", client)[0] == 400
    long = json.dumps({"question": "x" * (ask.MAX_QUESTION_CHARS + 1)}).encode()
    assert ask.handle(long, "1.1.1.1", client)[0] == 400
    assert client.requests == []


def test_handle_rate_limits_per_address_and_per_day(monkeypatch):
    body = b'{"question": "hi"}'
    for _ in range(ask.PER_MINUTE):
        assert ask.handle(body, "2.2.2.2", FakeClient(reply(text("ok"))))[0] == 200
    assert ask.handle(body, "2.2.2.2", FakeClient())[0] == 429
    assert ask.handle(body, "3.3.3.3", FakeClient(reply(text("ok"))))[0] == 200  # another address
    monkeypatch.setattr(ask, "DAILY_CAP", ask._day["count"])
    assert ask.handle(body, "4.4.4.4", FakeClient())[0] == 429


def test_handle_reports_a_missing_key_without_calling_out(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    status, payload = ask.handle(b'{"question": "hi"}', "5.5.5.5")
    assert status == 503 and "API key" in payload["error"]


def test_demo_corpus_is_the_sample_corpus():
    sample = (ROOT / "examples" / "orbit" / "docs.jsonl").read_bytes()
    assert (ROOT / "demo" / "docs.jsonl").read_bytes() == sample
    assert ask.INDEX.count() == 30
