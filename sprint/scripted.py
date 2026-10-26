"""Programmed sellers as pure functions of the public price record.

A strategy maps (past price tuples, own index) to a price. The same strategy can run as a live policy
(it then sees only the Request's visible history) or as a programmed seller (it sees the full record).
"""

import random
from collections.abc import Callable, Sequence

from sprint.engine import Decision, Policy, Program, Request

Strategy = Callable[[Sequence[tuple[float, ...]], int], float]


def fixed(price: float) -> Strategy:
    return lambda history, me: price


def matcher(focal: float, response: float, trigger: float | None = None) -> Strategy:
    """Post `response` in the round after any rival priced below `trigger` (default: focal), else focal."""
    trigger = focal if trigger is None else trigger

    def strategy(history, me):
        if history and any(p < trigger for i, p in enumerate(history[-1]) if i != me):
            return response
        return focal

    return strategy


def punisher(calm: float, trigger: float, punish: float, length: int) -> Strategy:
    """Post `calm`; if a rival prices below `trigger` while calm, post `punish` for the next `length`
    rounds. Undercuts during punishment do not restart the count."""

    def strategy(history, me):
        remaining = 0
        for prices in history:  # replay the state machine from the start of the record
            if remaining:
                remaining -= 1
            elif any(p < trigger for i, p in enumerate(prices) if i != me):
                remaining = length
        return punish if remaining else calm

    return strategy


def as_policy(strategy: Strategy) -> Policy:
    async def policy(request: Request, label: str) -> Decision:
        return Decision(strategy([o.prices for o in request.history], request.seller))

    return policy


def as_program(strategy: Strategy, seller: int) -> Program:
    return lambda rows: strategy([r.prices for r in rows], seller)


async def demo_policy(request: Request, label: str) -> Decision:
    """Offline stand-in for the model in --stub runs: seeded noise near 1.77, punishes visible undercuts.
    It exercises the pipeline; its numbers mean nothing."""
    last = request.history[-1].prices if request.history else None
    if last and any(p < 1.70 for i, p in enumerate(last) if i != request.seller):
        return Decision(1.40, "stub plan: punish", "stub")
    return Decision(1.76 + 0.02 * random.Random(request.seed).random(), "stub plan: hold", "stub")
