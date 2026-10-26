"""Offline controls for policy removal, with independent payoff and punishment calculations."""

import argparse
import asyncio
import csv
import math
from pathlib import Path

from sprint.client import Budget
from sprint.engine import Decision
from sprint.fallback1 import ReplacementRules
from sprint.fallback1_analyze import tables
from sprint.fallback1_run import load_config, run_batch

CONTROLS = ("constant", "price_responsive", "reset_sensitive")


def control_policy(kind, rules):
    if kind not in CONTROLS:
        raise ValueError(kind)

    async def policy(request, _label):
        if kind == "constant" or request.round_number <= rules.parent_rounds:
            return Decision(1.75, "hold")
        # Only observed market history and the agent's own saved note affect the decision.
        punished = any(o.number > rules.parent_rounds and o.prices[1] < rules.calm
                       for o in request.history)
        if kind == "reset_sensitive" and request.plans == "hold":
            return Decision(1.75, "hold")
        if punished or request.plans == "punishment_seen":
            return Decision(1.70, "punishment_seen")
        return Decision(1.60, "undercutting")

    return policy


def independent_paths(kind, removed, reset, horizon=20):
    # Fixed default market: solve the symmetric Nash FOC by bisection independently of Market.
    low, high = 1.0, 2.0
    for _ in range(80):
        p = (low + high) / 2
        exp_value = math.exp((2 - p) / .25)
        share = exp_value / (1 + 2 * exp_value)
        if 1 - (p - 1) * (1 - share) / .25 > 0:
            low = p
        else:
            high = p
    nash = (low + high) / 2
    prices, profits, countdown, seen_punishment = [], [], 0, False
    probes = kind == "price_responsive" or (kind == "reset_sensitive" and reset)
    for _ in range(horizon):
        a = (1.70 if seen_punishment else 1.60) if probes else 1.75
        b = 1.75 if removed or countdown == 0 else nash
        ea, eb = math.exp((2 - a) / .25), math.exp((2 - b) / .25)
        prices.append(a)
        profits.append(100 * (a - 1) * ea / (1 + ea + eb))
        seen_punishment |= b < 1.75
        countdown = countdown - 1 if countdown else (5 if a < 1.70 else 0)
    return {"mean_price": sum(prices) / horizon,
            "discounted_profit": sum(.95**t * p for t, p in enumerate(profits))}


def run_controls(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    config = load_config("configs/fallback1_pilot_deepseek41.toml")
    config |= {"rules": ReplacementRules(), "seeds": [1], "concurrency": 1}
    records = []
    for kind in CONTROLS:
        directory = out / kind
        directory.mkdir(exist_ok=True)
        asyncio.run(run_batch(config, directory, Budget(0), True,
                              lambda _, kind=kind: control_policy(kind, config["rules"])))
        _, paths, contrasts = tables([directory])
        expected = {}
        for removed in (False, True):
            for reset in (False, True):
                branch = f"{'replaced' if removed else 'kept'}_{'reset' if reset else 'memory'}_neutral"
                expected[branch] = independent_paths(kind, removed, reset)
                actual = paths[paths.branch == branch].iloc[0]
                for metric, value in expected[branch].items():
                    # Independent Nash root and the engine's numerical best-response fixed point
                    # differ slightly; allow 1e-5 profit units, while prices must match to 1e-12.
                    tolerance = 1e-5 if metric == "discounted_profit" else 1e-12
                    if not math.isclose(actual[metric], value, abs_tol=tolerance, rel_tol=0):
                        raise AssertionError((kind, branch, metric, actual[metric], value))
        for metric in ("mean_price", "discounted_profit"):
            memory = expected["replaced_memory_neutral"][metric] - expected["kept_memory_neutral"][metric]
            reset = expected["replaced_reset_neutral"][metric] - expected["kept_reset_neutral"][metric]
            for name, value in (("replacement_memory", memory), ("replacement_reset", reset),
                                ("removal_context_interaction", reset - memory)):
                actual = float(contrasts[(contrasts.contrast == name) &
                                         (contrasts.outcome == metric)].difference.iloc[0])
                tolerance = 2e-5 if metric == "discounted_profit" else 1e-12
                if not math.isclose(actual, value, abs_tol=tolerance, rel_tol=0):
                    raise AssertionError((kind, name, metric, actual, value))
                records.append({"control": kind, "contrast": name, "outcome": metric,
                                "expected": value, "observed": actual, "absolute_error": abs(actual-value)})
    with (out / "controls.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"{len(records)} control contrasts verified; largest error {max(r['absolute_error'] for r in records):.3g}")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    run_controls(parser.parse_args().out)
