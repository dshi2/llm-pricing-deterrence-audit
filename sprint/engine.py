"""Rounds of simultaneous pricing recorded in an append-only journal, with exact resume.

Live sellers are called through a stateless policy that sees only a Request: its visible public
history, its own quantity and profit, and its own previous notes. Programmed sellers (scripted rivals,
replayed paths) are deterministic functions of the public record, so they need no saved state.
A validated decision is journaled as soon as it arrives; a round is committed only when every live
seller has a validated decision. Reopening a journal replays committed rounds and saved decisions
instead of calling the model again.
"""

import asyncio
import hashlib
import json
import math
import os
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from sprint.market import Market


@dataclass(frozen=True)
class Decision:
    price: float  # normalised
    plans: str = ""
    insights: str = ""
    observations: str = ""

    def __post_init__(self):
        if isinstance(self.price, bool) or not isinstance(self.price, (int, float)):
            raise ValueError(f"price must be a number, got {self.price!r}")
        if not math.isfinite(self.price) or self.price < 0:
            raise ValueError(f"price must be finite and non-negative, got {self.price!r}")
        for name in ("plans", "insights", "observations"):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be a string")


@dataclass(frozen=True)
class Round:
    number: int
    prices: tuple[float, ...]
    quantities: tuple[float, ...]
    profits: tuple[float, ...]
    decisions: tuple[Decision, ...]

    @classmethod
    def from_dict(cls, d: dict) -> "Round":
        return cls(d["number"], tuple(d["prices"]), tuple(d["quantities"]), tuple(d["profits"]),
                   tuple(Decision(**x) for x in d["decisions"]))


@dataclass(frozen=True)
class Observation:
    """One past round as a single seller saw it."""
    number: int
    prices: tuple[float, ...]
    quantity: float
    profit: float


@dataclass(frozen=True)
class View:
    """A live seller sees rounds from memory_from on, its notes written from notes_from on, and a notice."""
    memory_from: int = 1
    notes_from: int = 1
    notice: str = ""


@dataclass(frozen=True)
class Request:
    seller: int
    round_number: int
    seed: int
    history: tuple[Observation, ...]
    plans: str | None  # None: no notes visible (first round, or cleared by a memory reset)
    insights: str | None
    notice: str = ""


Policy = Callable[[Request, str], Awaitable[Decision]]  # str: a log label, never shown to the model
Program = Callable[[tuple[Round, ...]], float]


def sampling_seed(game_seed: int, seller: int, round_number: int, replicate: int = 0) -> int:
    # Branch identity is deliberately excluded: matched branches get common random numbers.
    key = f"{game_seed}:{seller}:{round_number}:{replicate}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big") & 0x7FFFFFFF


def make_request(rows: Sequence[Round], seller: int, seed: int, view: View, window: int) -> Request:
    visible = [r for r in rows if r.number >= view.memory_from][-window:] if window > 0 else []
    history = tuple(Observation(r.number, r.prices, r.quantities[seller], r.profits[seller]) for r in visible)
    last = rows[-1] if rows else None
    notes = last.decisions[seller] if last is not None and last.number >= view.notes_from else None
    return Request(seller, len(rows) + 1, seed, history,
                   notes.plans if notes else None, notes.insights if notes else None, view.notice)


class Journal:
    """Append-only JSONL record of one game. Reopening it with the same header resumes the game."""

    def __init__(self, path, header: dict):
        self.path = Path(path)
        self.header = json.loads(json.dumps(header))
        self.rounds: dict[str, dict[int, Round]] = {}
        self.decisions: dict[tuple[str, int, int], Decision] = {}
        self.result: dict | None = None
        self.failure: str | None = None
        if self.path.exists():
            self._load()
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.write({"event": "header", **self.header})

    def _load(self):
        lines = self.path.read_text().splitlines()
        events = []
        for i, line in enumerate(lines):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                if i < len(lines) - 1:
                    raise ValueError(f"{self.path}: corrupt line {i + 1}") from None
                # A write cut off mid-line: keep the valid prefix, re-request what it held.
                self.path.write_text("".join(json.dumps(e) + "\n" for e in events))
                events.append({"event": "torn_line_dropped", "line": i + 1})
                self.write(events[-1])
        if not events:  # created but nothing written before a crash
            self.write({"event": "header", **self.header})
            return
        saved = {k: v for k, v in events[0].items() if k != "event"}
        if events[0].get("event") != "header" or saved != self.header:
            raise ValueError(f"{self.path} belongs to a different configuration or code version")
        for e in events[1:]:
            kind = e["event"]
            if kind == "decision":
                self.decisions[(e["branch"], e["round"], e["seller"])] = Decision(**e["decision"])
            elif kind == "round":
                self.rounds.setdefault(e["branch"], {})[e["row"]["number"]] = Round.from_dict(e["row"])
            elif kind == "complete":
                self.result = e["result"]
            elif kind == "failed":
                self.failure = e["reason"]
        self.write({"event": "resumed", "committed_rounds": sum(map(len, self.rounds.values()))})

    def write(self, event: dict):
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, allow_nan=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def record_decision(self, branch: str, number: int, seller: int, decision: Decision):
        self.write({"event": "decision", "branch": branch, "round": number, "seller": seller,
                    "decision": asdict(decision)})
        self.decisions[(branch, number, seller)] = decision

    def record_round(self, branch: str, row: Round):
        self.write({"event": "round", "branch": branch, "row": asdict(row)})
        self.rounds.setdefault(branch, {})[row.number] = row

    def finish(self, result: dict) -> dict:
        self.write({"event": "complete", "result": result})
        self.result = result
        return result

    def fail(self, reason: str):
        self.write({"event": "failed", "reason": reason})
        self.failure = reason


async def play(journal: Journal, branch: str, start: Sequence[Round], rounds: int, *, market: Market,
               game_seed: int, policy: Policy | None, live: Sequence[int], programs: Mapping[int, Program],
               views: Mapping[int, View] | None = None, window: int = 100, replicate: int = 0,
               label: str = "") -> tuple[Round, ...]:
    """Continue `start` for `rounds` rounds in `branch`; committed rounds are replayed, not re-requested."""
    if set(live) & set(programs) or sorted([*live, *programs]) != list(range(market.n)):
        raise ValueError("every seller must be either live or programmed")
    rows = list(start)
    for _ in range(rounds):
        row = journal.rounds.get(branch, {}).get(len(rows) + 1)
        if row is None:
            row = await _play_round(journal, branch, tuple(rows), market, game_seed, policy, live, programs,
                                    views or {}, window, replicate, label)
        rows.append(row)
    return tuple(rows[len(start):])


async def _play_round(journal, branch, rows, market, game_seed, policy, live, programs, views, window,
                      replicate, label) -> Round:
    number = len(rows) + 1
    decisions = {i: Decision(program(rows)) for i, program in programs.items()}

    async def decide(seller: int):
        request = make_request(rows, seller, sampling_seed(game_seed, seller, number, replicate),
                               views.get(seller, View()), window)
        answer = await policy(request, f"{label}{branch}:{number}:{seller}")
        if not isinstance(answer, Decision):
            raise TypeError(f"policy returned {type(answer).__name__}, not a Decision")
        journal.record_decision(branch, number, seller, answer)  # saved now, even if this round is cancelled

    pending = [i for i in live if (branch, number, i) not in journal.decisions]
    for outcome in await asyncio.gather(*map(decide, pending), return_exceptions=True):
        if isinstance(outcome, BaseException):
            raise outcome  # the other sellers' decisions are already saved for resume
    decisions |= {i: journal.decisions[(branch, number, i)] for i in live}
    ordered = tuple(decisions[i] for i in range(market.n))
    quantities, profits = market.outcomes([d.price for d in ordered])
    row = Round(number, tuple(d.price for d in ordered), quantities, profits, ordered)
    journal.record_round(branch, row)
    return row
