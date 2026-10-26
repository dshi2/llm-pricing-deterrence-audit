"""Failure injection for the LLM client: every error class takes its specified path, every attempt is
counted and logged (fatal and cancelled ones included), retry state survives a restart, and nothing is
substituted for a missing decision."""

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace

import openai
import pytest

try:  # the transport library openai uses: httpx2 in recent releases, httpx before
    import httpx2 as httpx
except ImportError:
    import httpx

from sprint.client import Budget, Declined, Fatal, Interrupted, LLMPolicy, ModelConfig, Unusable, derived_seed
from sprint.engine import Request
from sprint.market import Market
from sprint.prompt import messages

CONFIG = ModelConfig("deepseek/deepseek-chat-v3.1", "DeepInfra")
REQUEST = Request(seller=0, round_number=1, seed=12345, history=(), plans=None, insights=None)
URL = "https://openrouter.ai/api/v1/chat/completions"


VALID = '{"observations": "o", "plans": "p", "insights": "i", "chosen_price": 5.6}'


def ok(content=VALID, finish="stop", provider="DeepInfra", reasoning=None):
    message = SimpleNamespace(content=content, reasoning=reasoning)
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish, message=message)], provider=provider,
                           model=CONFIG.model, error=None,
                           usage=SimpleNamespace(prompt_tokens=900, completion_tokens=120, cost=0.0004))


def status(code, headers=None):
    response = httpx.Response(code, headers=headers or {}, request=httpx.Request("POST", URL))
    cls = openai.RateLimitError if code == 429 else openai.APIStatusError
    return cls(f"HTTP {code}", response=response, body=None)


class FakeClient:
    """Replays a script of responses or exceptions and records every request sent."""

    def __init__(self, script):
        self.script, self.sent = list(script), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    async def create(self, **kwargs):
        self.sent.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def make(script, tmp_path, budget=100, **kw):
    """A policy on a fake client and clock; a second make() on the same tmp_path is a restart."""
    clock, slept = Clock(), []

    async def sleep(seconds):
        slept.append(seconds)
        clock.now += seconds

    client = FakeClient(script)
    policy = LLMPolicy(CONFIG, Market(3), Budget(budget), tmp_path / "calls.jsonl", client=client,
                       sleep=sleep, clock=clock, **kw)
    return policy, client, slept


def call(policy):
    return asyncio.run(policy(REQUEST, "parent:1:0"))


def logged(tmp_path):
    return [json.loads(line) for line in (tmp_path / "calls.jsonl").read_text().splitlines()]


def test_request_is_pinned_and_complete(tmp_path):
    policy, client, _ = make([ok()], tmp_path)
    assert call(policy).price == pytest.approx(5.6 / 3.2)
    sent = client.sent[0]
    assert sent["extra_body"]["provider"] == {"order": ["DeepInfra"], "allow_fallbacks": False,
                                              "require_parameters": True}
    assert (sent["seed"], sent["temperature"], sent["response_format"]) == (12345, 0.7, {"type": "json_object"})
    row = logged(tmp_path)[0]
    assert (row["outcome"], row["provider_served"], row["prompt_tokens"], row["cost"]) == ("ok", "DeepInfra", 900, 0.0004)


def test_legacy_request_serialization_is_unchanged(tmp_path):
    policy, client, _ = make([ok()], tmp_path)
    call(policy)
    # Exact pre-change argument order and values, including no optional keys set to null.
    expected = {"model": CONFIG.model, "messages": messages(REQUEST, Market(3)), "temperature": 0.7,
                "seed": 12345, "max_tokens": 2000, "response_format": {"type": "json_object"},
                "extra_body": {"provider": {"order": ["DeepInfra"], "allow_fallbacks": False,
                                            "require_parameters": True}, "usage": {"include": True}}}
    assert json.dumps(client.sent[0]).encode() == json.dumps(expected).encode()


def test_reasoning_and_endpoint_pin_are_explicit_but_provider_check_uses_company(tmp_path):
    config = replace(CONFIG, model="openai/gpt-oss-120b", provider_route="deepinfra/bf16",
                     quantization="bf16", reasoning_effort="medium", max_tokens=8192)
    response = ok()
    response.usage.completion_tokens_details = SimpleNamespace(reasoning_tokens=77)
    client = FakeClient([response])
    policy = LLMPolicy(config, Market(3), Budget(1), tmp_path / "calls.jsonl", client=client)
    call(policy)
    assert client.sent[0]["extra_body"] == {
        "provider": {"order": ["deepinfra/bf16"], "allow_fallbacks": False, "require_parameters": True,
                     "quantizations": ["bf16"]},
        "usage": {"include": True}, "reasoning": {"effort": "medium"}}
    row = logged(tmp_path)[0]
    assert row["provider_served"] == row["provider_requested"] == "DeepInfra"
    assert row["provider_route_requested"] == "deepinfra/bf16"
    assert row["quantization_requested"] == "bf16"
    assert row["reasoning_tokens"] == 77


def test_invalid_reasoning_effort_is_rejected():
    with pytest.raises(ValueError, match="reasoning_effort"):
        replace(CONFIG, reasoning_effort="guess")


def test_qwen_thinking_budget_is_not_sent_as_reasoning_effort(tmp_path):
    config = replace(CONFIG, model="qwen/qwen3.8-flash", provider="Alibaba", provider_route="alibaba/us-east-1",
                     max_tokens=8192, reasoning_max_tokens=4096)
    client = FakeClient([ok(provider="Alibaba")])
    call(LLMPolicy(config, Market(3), Budget(1), tmp_path / "calls.jsonl", client=client))
    assert client.sent[0]["extra_body"]["reasoning"] == {"enabled": True, "max_tokens": 4096}
    assert client.sent[0]["extra_body"]["provider"]["order"] == ["alibaba/us-east-1"]
    with pytest.raises(ValueError, match="do not combine"):
        replace(config, reasoning_effort="medium")
    with pytest.raises(ValueError, match="below max_tokens"):
        replace(config, reasoning_max_tokens=8192)


def test_empty_then_valid_uses_a_fresh_seed(tmp_path):
    policy, client, _ = make([ok(content=""), ok()], tmp_path)
    call(policy)
    assert [s["seed"] for s in client.sent] == [12345, derived_seed(12345, 1)]
    assert [r["outcome"] for r in logged(tmp_path)] == ["unusable", "ok"]


@pytest.mark.parametrize("bad", [ok(content=""), ok(content="   "), ok(finish="length"),
                                 ok(content=None, reasoning="thinking..."), ok(content='{"chosen_price": 5.6}'),
                                 ok(content=VALID.replace("5.6", "-3"))])
def test_repeated_unusable_replies_fail_the_game(tmp_path, bad):
    policy, client, _ = make([bad] * 6, tmp_path, content_attempts=6)
    with pytest.raises(Unusable):
        call(policy)
    assert len(client.sent) == 6 and len(logged(tmp_path)) == 6
    assert len({s["seed"] for s in client.sent}) == 6


def test_reasoning_only_and_truncation_are_labelled(tmp_path):
    policy, _, _ = make([ok(content=None, reasoning="..."), ok(finish="length"), ok()], tmp_path)
    call(policy)
    assert [r["reason"] for r in logged(tmp_path)[:2]] == ["reasoning_only", "truncated"]


@pytest.mark.parametrize("declined", [ok(content="I won't set prices."), ok(finish="content_filter")])
def test_non_participation_stops_without_a_retry(tmp_path, declined):
    policy, client, _ = make([declined, ok()], tmp_path)
    with pytest.raises(Declined):
        call(policy)
    assert len(client.sent) == 1 and logged(tmp_path)[0]["outcome"] == "declined"
    policy, client, _ = make([ok()], tmp_path)  # a restart does not resample it either
    with pytest.raises(Declined):
        call(policy)
    assert client.sent == []


def test_retry_count_and_seeds_continue_after_a_restart(tmp_path):
    policy, _, _ = make([ok(content=""), ok(content="")], tmp_path, budget=2)
    with pytest.raises(Interrupted, match="budget"):
        call(policy)
    policy, client, _ = make([ok(content="")] * 4, tmp_path, content_attempts=6)
    with pytest.raises(Unusable):
        call(policy)
    assert [r["content_attempt"] for r in logged(tmp_path)] == [1, 2, 3, 4, 5, 6]
    assert [s["seed"] for s in client.sent] == [derived_seed(12345, k) for k in (2, 3, 4, 5)]
    policy, client, _ = make([ok()], tmp_path)  # the limit holds across any number of restarts
    with pytest.raises(Unusable):
        call(policy)
    assert client.sent == []


def test_reply_received_before_a_crash_is_used_not_requested_again(tmp_path):
    policy, _, _ = make([ok(content=VALID.replace("5.6", "6.1"))], tmp_path)
    call(policy)  # logged, but suppose the process died before the journal saved the decision
    policy, client, _ = make([ok()], tmp_path)
    assert call(policy).price == pytest.approx(6.1 / 3.2) and client.sent == []


def test_cancelled_attempt_is_logged(tmp_path):
    started = asyncio.Event()

    class Hanging(FakeClient):
        async def create(self, **kwargs):
            started.set()
            await asyncio.Event().wait()

    async def cancel_mid_request():
        policy = LLMPolicy(CONFIG, Market(3), Budget(5), tmp_path / "calls.jsonl", client=Hanging([]))
        task = asyncio.create_task(policy(REQUEST, "parent:1:0"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(cancel_mid_request())
    assert [(r["outcome"], r["billed"]) for r in logged(tmp_path)] == [("cancelled", "unknown")]


def test_transport_errors_retry_the_same_request(tmp_path):
    request = httpx.Request("POST", URL)
    script = [openai.APITimeoutError(request=request), openai.APIConnectionError(request=request), status(503), ok()]
    policy, client, slept = make(script, tmp_path)
    call(policy)
    assert {s["seed"] for s in client.sent} == {12345}
    rows = logged(tmp_path)
    assert [r["outcome"] for r in rows] == ["transport_error"] * 3 + ["ok"]
    assert rows[0]["billed"] == "unknown"  # a timed-out request may still have been charged
    assert len(slept) == 3


def test_rate_limit_honours_retry_after(tmp_path):
    policy, _, slept = make([status(429, {"retry-after": "7"}), ok()], tmp_path)
    call(policy)
    assert slept == [7.0]


def test_error_inside_200_body_is_retried(tmp_path):
    body_error = ok()
    body_error.error = {"code": 502, "message": "upstream error"}
    policy, client, _ = make([body_error, ok()], tmp_path)
    call(policy)
    assert len(client.sent) == 2


def test_persistent_transport_errors_interrupt_the_game(tmp_path):
    policy, _, _ = make([status(503)] * 50, tmp_path, transport_window=100)
    with pytest.raises(Interrupted):
        call(policy)


@pytest.mark.parametrize("code", [400, 401, 402, 403, 404])
def test_fatal_errors_stop_immediately(tmp_path, code):
    policy, client, _ = make([status(code), ok()], tmp_path)
    with pytest.raises(Fatal):
        call(policy)
    assert len(client.sent) == 1
    assert [(r["outcome"], r["http_status"]) for r in logged(tmp_path)] == [("fatal", code)]


def test_wrong_provider_is_fatal(tmp_path):
    policy, _, _ = make([ok(provider="SiliconFlow")], tmp_path)
    with pytest.raises(Fatal, match="only DeepInfra"):
        call(policy)
    row = logged(tmp_path)[0]
    assert (row["outcome"], row["provider_served"], row["cost"]) == ("fatal", "SiliconFlow", 0.0004)


def test_missing_provider_identity_stops_collection(tmp_path):
    policy, client, _ = make([ok(provider=None), ok()], tmp_path)
    with pytest.raises(Fatal, match="provider identity missing"):
        call(policy)
    assert len(client.sent) == 1 and logged(tmp_path)[0]["outcome"] == "fatal"


def test_fatal_error_in_response_body_is_logged(tmp_path):
    body_error = ok()
    body_error.error = {"code": 402, "message": "insufficient credits"}
    policy, _, _ = make([body_error], tmp_path)
    with pytest.raises(Fatal, match="insufficient credits"):
        call(policy)
    assert logged(tmp_path)[0]["outcome"] == "fatal"


def test_budget_counts_every_attempt(tmp_path):
    policy, client, _ = make([ok(content=""), ok(content=""), ok()], tmp_path, budget=2)
    with pytest.raises(Interrupted, match="budget"):
        call(policy)
    assert len(client.sent) == 2


def test_key_never_logged(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-secret")
    real = LLMPolicy(CONFIG, Market(3), Budget(1), tmp_path / "calls.jsonl")
    assert real.client.max_retries == 0  # no hidden SDK retries
    policy, _, _ = make([ok(content=""), ok()], tmp_path)
    call(policy)
    assert "sk-or-test-secret" not in (tmp_path / "calls.jsonl").read_text()
