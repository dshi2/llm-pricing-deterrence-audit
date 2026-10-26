"""LLM sellers through OpenRouter: one pinned provider, one retry controller, a log row per attempt.

- Transport problems (timeout, connection, 408/409/429, 5xx, error inside a 200 body, no choices) are
  retried with the same request and seed, with backoff, for up to `transport_window` seconds; then the
  game is Interrupted and can be resumed.
- Formatting failures (empty, reasoning only, cut off, invalid JSON, missing or invalid fields) are
  retried with a fresh derived seed up to `content_attempts` times; then the game fails (Unusable).
- Non-participation (a prose reply with no JSON object, or a provider content filter) is not retried:
  resampling until the model takes part would select on participation. The game stops (Declined).
- Key, credit, model, request/context and pinning problems raise Fatal and stop the run.
Every attempt counts against the Budget and is logged before anything is raised, including attempts
cut off by cancellation; missing fields are logged as null, and a request that may have been charged
without a reply is logged as billed: "unknown". The log is also the retry state: on resume a reply
that reached the log is used, never requested again, and the attempt count and seed sequence continue.
A crash before that write (mid-request) costs one repeated request.
SDK-level retries are disabled and the API key is never written anywhere.
"""

import asyncio
import datetime
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import openai

from sprint.engine import Decision, Request
from sprint.market import Market
from sprint.prompt import UNITS, BadReply, NoDecision, messages, parse

FATAL_STATUS = {400, 401, 402, 403, 404, 413, 422}
TRANSPORT_STATUS = {408, 409, 425, 429}


class Fatal(Exception):
    """Stop the whole run: retrying cannot help."""


class Interrupted(Exception):
    """Stop this game for now; its journal can be resumed later."""


class Unusable(Exception):
    """The model gave no usable reply within the content-attempt limit; the game fails."""


class Declined(Unusable):
    """The model (or its provider's filter) did not take part; the game stops without a retry."""


class Budget:
    """Attempts allowed for one run directory in total, shared by its games. `used` starts from the
    attempts already in the directory's call logs, so a rerun does not get a fresh allowance."""

    def __init__(self, limit: int, used: int = 0):
        self.limit, self.used = limit, used

    def take(self):
        if self.used >= self.limit:
            raise Interrupted(f"call budget of {self.limit} attempts used up")
        self.used += 1


@dataclass(frozen=True)
class ModelConfig:
    model: str  # OpenRouter model id
    provider: str  # the only provider allowed to serve it
    temperature: float = 0.7
    max_tokens: int = 2000
    prefix: str = "P1"
    units: str = "consistent"  # "consistent": all money in the price currency; "thesis": the starter's mix
    provider_route: str | None = None  # optional endpoint slug; provider remains the returned company name
    quantization: str | None = None  # advertised provider filter, not independent attestation of weights
    reasoning_effort: str | None = None  # omitted entirely for legacy configurations
    reasoning_max_tokens: int | None = None  # for providers that expose a thinking budget instead

    def __post_init__(self):
        if self.prefix not in ("P1", "P2") or self.units not in UNITS:
            raise ValueError(f"unknown prefix {self.prefix!r} or units {self.units!r}")
        if self.reasoning_effort not in (None, "low", "medium", "high"):
            raise ValueError("reasoning_effort must be low, medium, high, or omitted")
        if self.provider_route is not None and not self.provider_route.strip():
            raise ValueError("provider_route cannot be empty")
        if self.reasoning_max_tokens is not None and (
            type(self.reasoning_max_tokens) is not int
            or not 0 < self.reasoning_max_tokens < self.max_tokens
            or self.reasoning_effort is not None
        ):
            raise ValueError("reasoning_max_tokens must be below max_tokens; do not combine with effort")


BLANK_ATTEMPT = {"outcome": None, "reason": None, "provider_served": None, "model_served": None,
                 "http_status": None, "finish_reason": None, "prompt_tokens": None, "completion_tokens": None,
                 "cost": None, "billed": True, "retry_after": None, "latency_s": None, "text": None}


def derived_seed(seed: int, attempt: int) -> int:
    return (seed + 7919 * attempt) & 0x7FFFFFFF


class LLMPolicy:
    """A Policy: Request -> thesis prompt -> pinned model call -> validated Decision."""

    def __init__(self, config: ModelConfig, market: Market, budget: Budget, log_path, client=None,
                 content_attempts: int = 6, transport_window: float = 600.0, timeout: float = 120.0,
                 sleep=asyncio.sleep, clock=time.monotonic):
        self.config, self.market, self.budget = config, market, budget
        self.log_path = Path(log_path)
        self.client = client or openai.AsyncOpenAI(base_url="https://openrouter.ai/api/v1",
                                                   api_key=os.environ["OPENROUTER_API_KEY"],
                                                   max_retries=0, timeout=timeout)
        self.content_attempts, self.transport_window = content_attempts, transport_window
        self.sleep, self.clock = sleep, clock
        self._jitter = random.Random()  # timing only; never touches experimental randomness
        self._attempts: dict[str, list[dict]] = {}  # label -> logged attempts, restored on resume
        if self.log_path.exists():
            for line in self.log_path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:  # a torn final line: that attempt is simply made again
                    continue
                self._attempts.setdefault(row["label"], []).append(row)

    async def __call__(self, request: Request, label: str) -> Decision:
        msgs = messages(request, self.market, self.config.prefix, self.config.units)
        return await self.complete(msgs, request.seed, label, lambda text: parse(text, self.market))

    async def complete(self, msgs: list[dict], seed: int, label: str, parser):
        """Shared transport for pricing and separate diagnostics; parser validates format, not accuracy."""
        earlier = self._attempts.get(label, [])
        received = [r for r in earlier if r["outcome"] == "ok"]
        if received:  # answered before an interruption but not yet in the journal: never ask again
            return parser(received[-1]["text"])
        if any(r["outcome"] == "declined" for r in earlier):
            raise Declined(f"declined at {label} before the interruption")
        failures = sum(r["outcome"] == "unusable" for r in earlier)
        last_reason = earlier[-1]["reason"] if earlier else None
        original_seed = seed
        transport_since, waits = None, 0
        while failures < self.content_attempts:
            seed = derived_seed(original_seed, failures) if failures else original_seed
            self.budget.take()
            record = {"label": label, "seed": seed, "content_attempt": failures + 1, **BLANK_ATTEMPT,
                      "provider_requested": self.config.provider,
                      "provider_route_requested": self.config.provider_route,
                      "quantization_requested": self.config.quantization,
                      "reasoning_effort_requested": self.config.reasoning_effort,
                      "reasoning_max_tokens_requested": self.config.reasoning_max_tokens}
            started = self.clock()
            try:
                await self._send(record, msgs, seed)
            except BaseException as e:  # cancelled or unexpected: keep the attempt on record, then stop
                cancelled = isinstance(e, asyncio.CancelledError)
                self._log(record | {"outcome": "cancelled" if cancelled else "error", "billed": "unknown",
                                    "reason": record["reason"] or repr(e)})
                raise
            decision = None
            if record["outcome"] in ("empty", "reasoning_only", "truncated"):
                record |= {"reason": record["outcome"], "outcome": "unusable"}
            elif record["outcome"] == "ok":
                try:
                    decision = parser(record["text"])
                except NoDecision as e:
                    record |= {"outcome": "declined", "reason": str(e)}
                except BadReply as e:
                    record |= {"outcome": "unusable", "reason": str(e)}
            self._log(record)
            match record["outcome"]:
                case "ok":
                    return decision
                case "fatal":
                    raise Fatal(f"{label}: {record['reason']}")
                case "declined":
                    raise Declined(f"declined at {label}: {record['reason']}")
                case "unusable":
                    failures, last_reason, transport_since, waits = failures + 1, record["reason"], None, 0
                case "transport_error":
                    transport_since = transport_since or started
                    if self.clock() - transport_since > self.transport_window:
                        raise Interrupted(f"{label}: transport errors for over {self.transport_window:.0f} s")
                    waits += 1
                    await self.sleep(record["retry_after"] or min(60.0, 2.0**waits) + self._jitter.random())
        raise Unusable(f"{label}: no usable reply in {failures} attempts ({last_reason})")

    async def _send(self, record: dict, msgs: list[dict], seed: int):
        """Make one request and fill in `record` with what came back; the caller decides what to do."""
        started = self.clock()
        extra = {"provider": {"order": [self.config.provider_route or self.config.provider],
                              "allow_fallbacks": False, "require_parameters": True},
                 "usage": {"include": True}}
        if self.config.quantization is not None:
            extra["provider"]["quantizations"] = [self.config.quantization]
        if self.config.reasoning_effort is not None:
            extra["reasoning"] = {"effort": self.config.reasoning_effort}
        elif self.config.reasoning_max_tokens is not None:
            extra["reasoning"] = {"enabled": True, "max_tokens": self.config.reasoning_max_tokens}
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model, messages=msgs, temperature=self.config.temperature, seed=seed,
                max_tokens=self.config.max_tokens, response_format={"type": "json_object"},
                extra_body=extra)
        except (openai.APITimeoutError, openai.APIConnectionError) as e:
            record |= {"outcome": "transport_error", "reason": type(e).__name__, "billed": "unknown"}
            return
        except openai.APIStatusError as e:
            status = e.status_code
            record |= {"http_status": status, "reason": f"HTTP {status}: {e.message}", "billed": False}
            if status in TRANSPORT_STATUS or status >= 500:
                retry_after = e.response.headers.get("retry-after") if e.response is not None else None
                record |= {"outcome": "transport_error", "retry_after": _seconds(retry_after)}
            else:  # 400/401/402/403/404/413/422 and anything unexpected: retrying cannot help
                record["outcome"] = "fatal"
            return
        finally:
            record["latency_s"] = self.clock() - started
        record["http_status"] = 200
        served = getattr(response, "provider", None)
        record |= {"provider_served": served, "model_served": getattr(response, "model", None),
                   "response_id": getattr(response, "id", None),
                   "system_fingerprint": getattr(response, "system_fingerprint", None)}
        usage = getattr(response, "usage", None)
        if usage is not None:
            record |= {"prompt_tokens": getattr(usage, "prompt_tokens", None),
                       "completion_tokens": getattr(usage, "completion_tokens", None),
                       "cost": getattr(usage, "cost", None),
                       "reasoning_tokens": getattr(getattr(usage, "completion_tokens_details", None),
                                                   "reasoning_tokens", None)}
        error = getattr(response, "error", None)
        if error:  # OpenRouter can report errors inside a 200 body
            code = error.get("code") if isinstance(error, dict) else getattr(error, "code", None)
            record |= {"outcome": "fatal" if code in FATAL_STATUS else "transport_error",
                       "reason": f"error in body: {error}"}
            return
        if served is not None and served.lower() != self.config.provider.lower():
            record |= {"outcome": "fatal", "reason": f"served by {served}, but only {self.config.provider} is allowed"}
            return
        if not getattr(response, "choices", None):
            record |= {"outcome": "transport_error", "reason": "no choices in response"}
            return
        if served is None:
            record |= {"outcome": "fatal", "reason": "provider identity missing from a model reply"}
            return
        choice = response.choices[0]
        content = getattr(choice.message, "content", None)
        record |= {"finish_reason": choice.finish_reason, "text": content}
        if choice.finish_reason == "length":
            record["outcome"] = "truncated"
        elif choice.finish_reason == "content_filter":
            record |= {"outcome": "declined", "reason": "content_filter"}
        elif not content or not content.strip():
            record["outcome"] = "reasoning_only" if getattr(choice.message, "reasoning", None) else "empty"
        else:
            record["outcome"] = "ok"

    def _log(self, record: dict):
        row = {"time": datetime.datetime.now(datetime.UTC).isoformat(timespec="milliseconds"),
               "model_requested": self.config.model, **record}
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())


def _seconds(value) -> float | None:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None
