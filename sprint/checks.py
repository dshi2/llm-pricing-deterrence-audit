"""Scripted validation of the main study against independently computed answers (no API calls).

Each scenario runs through the real engine (parent, reference, eligibility, eight paired branches,
A/A repeat, journal). The expected gains are recomputed here from scratch: own branch simulation and a
separate logit formula, so an error in branching, replay timing, eligibility or the starter-based
economics shows up as a mismatch.
"""

import asyncio
import math
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from sprint.engine import Journal
from sprint.experiments import MASKS, MainRules, main_game
from sprint.market import Market
from sprint.scripted import as_policy, fixed, matcher

FOCAL = 1.75
NASH3 = 1.3702  # strong response; any price well below the cut works
SCENARIOS = {  # seller -> strategy; which seller becomes A is drawn by the engine
    "no_response": {i: fixed(FOCAL) for i in range(3)},
    "one_responder": {0: matcher(FOCAL, NASH3), 1: fixed(FOCAL), 2: fixed(FOCAL)},
    "all_respond_strong": {i: matcher(FOCAL, NASH3) for i in range(3)},
    "all_respond_mild": {i: matcher(FOCAL, 1.62, trigger=1.60) for i in range(3)},
}


def logit_profits(prices, quality=2.0, mu=0.25, cost=1.0, size=100.0) -> list[float]:
    weights = [math.exp((quality - x) / mu) for x in prices]
    total = 1.0 + sum(weights)
    return [size * w / total * (x - cost) for w, x in zip(weights, prices)]


def expected_gains(strategies: dict, roles: dict, rules: MainRules) -> dict:
    """A's discounted cut-minus-no-cut gain per condition, simulated without the engine."""
    a, b, c = roles["A"], roles["B"], roles["C"]
    parent = [(FOCAL,) * 3] * rules.parent_rounds  # every scenario holds the focal price until a cut

    def branch_return(mask: str, cut: bool) -> float:
        history, total = list(parent), 0.0
        for s in range(rules.horizon):
            row = [FOCAL] * 3  # replayed sellers post their reference price, the focal price
            row[a] = FOCAL * (1 - rules.cut_fraction) if cut and s < rules.cut_rounds else FOCAL
            for seller, live in ((b, mask[0] == "1"), (c, mask[1] == "1")):
                if live:
                    row[seller] = strategies[seller](history, seller)
            history.append(tuple(row))
            total += rules.discount**s * logit_profits(row)[a]
        return total

    return {m: branch_return(m, True) - branch_return(m, False) for m in MASKS}


def scripted_check(out_dir, rules: MainRules = MainRules(), game_seed: int = 7) -> pd.DataFrame:
    """Run every scenario; one row per scenario with engine and independent values."""
    out_dir = Path(out_dir)
    rows = []
    for name, strategies in SCENARIOS.items():
        async def policy(request, label, strategies=strategies):
            return await as_policy(strategies[request.seller])(request, label)

        journal = Journal(out_dir / f"{name}.jsonl", {"check": name, "rules": asdict(rules), "seed": game_seed})
        result = asyncio.run(main_game(policy, game_seed, journal, Market(3), rules))
        expected = expected_gains(strategies, result["roles"], rules)
        r1 = expected["00"] - (expected["10"] + expected["01"]) / 2
        rows.append({
            "scenario": name, "A": result["roles"]["A"],
            **{f"G{m}": result["gains"][m] for m in MASKS}, "R1": result["R1"], "R2": result["R2"],
            "max_error": max([abs(result["gains"][m] - expected[m]) for m in MASKS]
                             + [abs(result["R1"] - r1), abs(result["R2"] - (expected["00"] - expected["11"]))]),
            "aa_difference": result["aa_difference"],
        })
    return pd.DataFrame(rows)
