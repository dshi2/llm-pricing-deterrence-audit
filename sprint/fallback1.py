"""Controlled removal of a scripted punisher; all prices/profits here are normalised.

Every completed parent gets an independent fixed-rival competence baseline. Six paired interventions and an
A/A repeat run only at calm checkpoints, so replacement preserves the first rival price. Restraint
selects the persistence subgroup, not whether the fresh competence baseline is collected.
"""

import math
import random
from dataclasses import asdict, dataclass

from sprint.engine import Journal, Policy, View, play
from sprint.experiments import discounted
from sprint.market import Market
from sprint.scripted import as_program, fixed, punisher

# (rival replaced, memory reset, truthful information). Not a full factorial.
CELLS = {
    "kept_memory_neutral": (False, False, False),
    "kept_reset_neutral": (False, True, False),
    "replaced_memory_neutral": (True, False, False),
    "replaced_reset_neutral": (True, True, False),
    "replaced_memory_truth": (True, False, True),
    "replaced_reset_truth": (True, True, True),
}


@dataclass(frozen=True)
class ReplacementRules:
    parent_rounds: int = 100
    horizon: int = 20
    window: int = 100
    restraint_window: int = 20
    restraint_share: float = 0.80
    calm: float = 1.75
    trigger: float = 1.70
    punishment_rounds: int = 5
    discount: float = 0.95

    def __post_init__(self):
        for name in ("parent_rounds", "horizon", "window", "restraint_window", "punishment_rounds"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.restraint_window > self.parent_rounds:
            raise ValueError("restraint window cannot exceed parent history")
        if not 0 < self.discount <= 1 or not 0 <= self.restraint_share <= 1:
            raise ValueError("invalid discount or restraint share")
        if not (math.isfinite(self.calm) and math.isfinite(self.trigger) and 0 < self.trigger < self.calm):
            raise ValueError("require finite 0 < trigger < calm")

    def planned_calls(self):
        return {"maximum": self.parent_rounds + 8 * self.horizon,
                "active_checkpoint": self.parent_rounds + self.horizon}


def next_state(state: int, price: float, rules: ReplacementRules) -> int:
    return state - 1 if state else (rules.punishment_rounds if price < rules.trigger else 0)


def state_after(rows, rules: ReplacementRules) -> int:
    state = 0
    for row in rows:
        state = next_state(state, row.prices[0], rules)
    return state


def notices(market: Market, rules: ReplacementRules) -> tuple[str, str]:
    # Same word count and placement; not claimed to have identical tokenizer length or semantics.
    truth = (f"Your competitor now posts {rules.calm * market.alpha:.2f} every round regardless of your price. "
             "Prices use the same currency as your price.")
    neutral = ("This market continues with another sequence of rounds following saved history. "
               "Prices use the same currency as your price.")
    return neutral, truth


def dynamic_benchmark(market: Market, rules: ReplacementRules, initial_state: int = 0) -> dict:
    """Finite-horizon optimum, zero terminal value, against the known punishment state machine.

In a calm state only crossing the trigger changes future state. Logit profit is unimodal, so the
optimum in each trigger region is its static best response clipped to that region. During punishment
current price does not affect the countdown. No price-grid approximation is used for this benchmark.
"""
    if market.n != 2 or not market.cost < rules.trigger:
        raise ValueError("benchmark requires two sellers and trigger above cost")
    if not 0 <= initial_state <= rules.punishment_rounds:
        raise ValueError("invalid initial punishment state")
    punishment = market.nash()
    br_calm = market.best_response(0, (market.cost, rules.calm))
    br_punish = market.best_response(0, (market.cost, punishment))
    under = min(br_calm, math.nextafter(rules.trigger, -math.inf))
    hold = max(br_calm, rules.trigger)
    value = [0.0] * (rules.punishment_rounds + 1)
    policies = []
    for _ in range(rules.horizon):
        updated, policy = [], []
        for state in range(rules.punishment_rounds + 1):
            rival = punishment if state else rules.calm
            candidates = (br_punish,) if state else (under, hold)
            best_value, best_price = max(
                (market.outcomes((price, rival))[1][0] + rules.discount * value[next_state(state, price, rules)],
                 price) for price in candidates)
            updated.append(best_value)
            policy.append(best_price)
        value = updated
        policies.append(policy)
    return {"value": value[initial_state], "initial_price": policies[-1][initial_state],
            "punishment_price": punishment, "fixed_best_response": br_calm,
            "policies_by_remaining_horizon": policies, "terminal_value": 0.0}


def checkpoint(rows, rules):
    tail = rows[-rules.restraint_window:]
    mean = math.fsum(r.prices[0] for r in tail) / len(tail)
    share = sum(r.prices[0] >= rules.trigger for r in tail) / len(tail)
    state = state_after(rows, rules)
    states, current = [], 0
    for row in rows:
        states.append(current)
        current = next_state(current, row.prices[0], rules)
    return {"mean_price": mean, "share_at_or_above_trigger": share,
            "undercut_frequency": 1 - share,
            "punishment_occupancy": sum(s > 0 for s in states[-rules.restraint_window:]) / len(tail),
            "next_punishment_remaining": state, "calm": state == 0,
            "restrained": mean >= rules.trigger and share >= rules.restraint_share and state == 0}


def score_path(rows, market, rules, benchmark, replaced):
    price = benchmark["fixed_best_response"]
    fixed_profit = market.outcomes((price, rules.calm))[1][0]
    optimum = (math.fsum(rules.discount**t * fixed_profit for t in range(len(rows)))
               if replaced else benchmark["value"])
    value = discounted(rows, 0, rules.discount)
    undercuts = [i + 1 for i, row in enumerate(rows) if row.prices[0] < rules.trigger]
    return {"first_price": rows[0].prices[0],
            "mean_price": math.fsum(r.prices[0] for r in rows) / len(rows),
            "undercut_rate": len(undercuts) / len(rows),
            "first_undercut_round": undercuts[0] if undercuts else None,
            "no_undercut_in_window": not undercuts,
            "discounted_profit": value, "benchmark_value": optimum,
            "discounted_profit_gap": optimum - value,
            "mean_price_gap_fixed_br": math.fsum(r.prices[0] - price for r in rows) / len(rows)
            if replaced else None}


async def replacement_game(policy: Policy, game_seed: int, journal: Journal,
                           market=None, rules=None, label="") -> dict:
    market, rules = market or Market(2), rules or ReplacementRules()
    if market.n != 2:
        raise ValueError("fallback 1 requires two sellers")
    if journal.result is not None:
        return journal.result
    if journal.failure is not None:
        return {"status": "failed", "game_seed": game_seed, "reason": journal.failure}
    benchmark = dynamic_benchmark(market, rules)
    punishment = as_program(punisher(rules.calm, rules.trigger, benchmark["punishment_price"],
                                    rules.punishment_rounds), 1)
    replacement = as_program(fixed(rules.calm), 1)
    neutral, truth = notices(market, rules)

    async def run(branch, start, rounds, program, view=None, replicate=0):
        return await play(journal, branch, start, rounds, market=market, game_seed=game_seed,
                          policy=policy, live=[0], programs={1: program}, views={0: view or View()},
                          window=rules.window, replicate=replicate, label=label)

    parent = await run("parent", (), rules.parent_rounds, punishment)
    gate = checkpoint(parent, rules)
    # Independent memory, round numbering and random stream; never fed back to intervention branches.
    fresh = await run("fresh_fixed", (), rules.horizon, replacement, View(notice=truth), replicate=1)
    paths = {"fresh_fixed": score_path(fresh, market, rules, benchmark, True)}
    order = list(CELLS)
    random.Random(f"replacement_order:{game_seed}").shuffle(order)
    if gate["calm"]:
        for cell in order:
            replaced, reset, informed = CELLS[cell]
            view = View(memory_from=len(parent) if reset else 1,
                        notes_from=len(parent) + 1 if reset else 1,
                        notice=truth if informed else neutral)
            rows = await run(cell, parent, rules.horizon, replacement if replaced else punishment, view)
            paths[cell] = score_path(rows, market, rules, benchmark, replaced)
        repeat = await run("kept_memory_neutral_repeat", parent, rules.horizon, punishment, View(notice=neutral))
        paths["kept_memory_neutral_repeat"] = score_path(repeat, market, rules, benchmark, False)
    return journal.finish({"status": "complete", "game_seed": game_seed, "rules": asdict(rules),
                           "checkpoint": gate, "branches_run": gate["calm"], "branch_order": order,
                           "benchmark": benchmark, "paths": paths})
