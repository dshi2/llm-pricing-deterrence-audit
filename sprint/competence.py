"""Separate diagnostics: 20 arithmetic questions and two 10-round fixed-rival markets.

No answers or feedback enter main-study histories. Wrong numeric answers are retained, not retried.
Pricing uses the main prompt plus a truthful fixed-rival notice; it measures learning/optimisation,
not arithmetic alone. These small checks are descriptive, not a validated model-selection threshold.
"""

import argparse
import asyncio
import hashlib
import json
import math
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

import pandas as pd

from sprint.client import Budget, LLMPolicy, Unusable
from sprint.engine import Journal, View, play
from sprint.market import Market
from sprint.prompt import BadReply, parse_object
from sprint.run import attempts_logged, code_sha, load_config, provenance

ROUNDS = 10
RIVALS = ((1.75, 1.75), (1.60, 1.925))  # normalised prices, fixed before any live results


def arithmetic_cases() -> list[dict]:
    cases = []
    for i, (p, q) in enumerate((("5.60", "31.5"), ("6.40", "24"), ("4.40", "42.75"), ("3.10", "50"))):
        price, quantity, cost, discount = map(Decimal, (p, q, "3.20", "0.95"))
        profit = (price - cost) * quantity
        questions = (
            ("profit", f"Price is {p}, unit cost is 3.20 and quantity sold is {q}. What is profit?", profit),
            ("cut", f"A price of {p} is reduced by 10%. What is the new price?", price * Decimal("0.90")),
            ("units", f"Raw price is {p} and alpha is 3.20. What is raw price divided by alpha?", price / cost),
            ("discount", (f"Profit is {profit} in each of three rounds. Discount at 0.95 per round, "
             "with the first round undiscounted. What is the total?"), profit * (1 + discount + discount**2)),
            ("gain", (f"Old price is {p}, old quantity is {q}, and unit cost is 3.20. "
             f"New price is {price * Decimal('0.90')} and new quantity is {quantity + 5}. "
             "What is new profit minus old profit?"),
             (price * Decimal("0.90") - cost) * (quantity + 5) - profit),
        )
        cases.extend({"id": f"{kind}_{i}", "question": question, "expected": float(expected)}
                     for kind, question, expected in questions)
    return cases


def parse_number(text: str) -> float:
    obj = parse_object(text)
    value = obj.get("answer")
    if set(obj) != {"answer"} or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BadReply('expected exactly {"answer": number}')
    if not math.isfinite(value):
        raise BadReply("answer must be finite")
    return float(value)


def score_answer(answer: float, expected: float) -> dict:
    # Frozen tolerance: one millionth in the requested numeric units, plus float rounding.
    error = answer - expected
    return {"answer": answer, "expected": expected, "error": error,
            "correct": math.isclose(answer, expected, abs_tol=1e-6, rel_tol=1e-9)}


async def run_checks(config: dict, out: Path, budget: Budget, client=None) -> dict:
    market = Market(3)
    head = {"experiment": "competence_v1", "model": asdict(config["model"]), "market": asdict(market),
            "code_sha": code_sha(), "diagnostic_sha": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    arithmetic, pricing = [], []
    for i, case in enumerate(arithmetic_cases()):
        journal = Journal(out / f"arithmetic_{case['id']}.jsonl", head | {"case": case, "seed": 20000 + i})
        if journal.result is None and journal.failure is None:
            policy = LLMPolicy(config["model"], market, budget,
                               out / f"arithmetic_{case['id']}.calls.jsonl", client=client)
            msgs = [{"role": "system", "content": 'Return only JSON: {"answer": number}. Do not round.'},
                    {"role": "user", "content": case["question"]}]
            try:
                answer = await policy.complete(msgs, 20000 + i, case["id"], parse_number)
                journal.finish({"id": case["id"], "status": "answered", **score_answer(answer, case["expected"])})
            except Unusable as error:
                journal.fail(str(error))
        arithmetic.append(journal.result or {"id": case["id"], "status": "failed", "reason": journal.failure})

    for i, rivals in enumerate(RIVALS):
        seed = 21000 + i
        journal = Journal(out / f"pricing_{i}.jsonl", head | {"rivals": rivals, "rounds": ROUNDS, "seed": seed})
        if journal.result is None and journal.failure is None:
            policy = LLMPolicy(config["model"], market, budget, out / f"pricing_{i}.calls.jsonl", client=client)
            notice = (f"In this diagnostic, Firm B always charges {rivals[0] * market.alpha:.2f} and "
                      f"Firm C always charges {rivals[1] * market.alpha:.2f}, regardless of your actions. "
                      "These are raw prices in the same currency as your price. They remain fixed for all rounds.")
            try:
                rows = await play(journal, "fixed_rivals", (), ROUNDS, market=market, game_seed=seed,
                                  policy=policy, live=[0], programs={1: lambda rows, p=rivals[0]: p,
                                                                   2: lambda rows, p=rivals[1]: p},
                                  views={0: View(notice=notice)}, window=config["rules"].window)
                best = market.best_response(0, (market.cost, *rivals))
                optimum = market.outcomes((best, *rivals))[1][0]
                journal.finish({"episode": i, "status": "complete", "best_response": best,
                                "optimal_profit": optimum, "rounds": [
                                    {"round": r.number, "price": r.prices[0], "profit": r.profits[0],
                                     "profit_gap": optimum - r.profits[0]} for r in rows]})
            except Unusable as error:
                journal.fail(str(error))
        pricing.append(journal.result or {"episode": i, "status": "failed", "reason": journal.failure})
    return {"arithmetic": arithmetic, "pricing": pricing}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("config", type=Path)
    ap.add_argument("--out", type=Path, default=Path("runs"))
    ap.add_argument("--budget", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    config = load_config(args.config)
    print("Competence checks: 20 independent arithmetic calls + 20 pricing calls, before retries.")
    if args.dry_run:
        return
    if args.budget <= 0:
        ap.error("set --budget (total attempts for this diagnostic directory)")
    out = args.out / (config["name"] + "_competence")
    budget = Budget(args.budget, used=attempts_logged(out))
    result = asyncio.run(run_checks(config, out, budget))
    result |= {"provenance": provenance("competence"), "attempts_used": budget.used}
    (out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(result["arithmetic"]).to_csv(out / "arithmetic.csv", index=False)
    pd.DataFrame([{**{k: v for k, v in episode.items() if k != "rounds"}, **r}
                  for episode in result["pricing"] for r in episode.get("rounds", [{}])]).to_csv(
        out / "pricing.csv", index=False)
    answered = sum(r["status"] == "answered" for r in result["arithmetic"])
    correct = sum(r.get("correct", False) for r in result["arithmetic"])
    print(f"Arithmetic: {answered}/20 answered, {correct}/20 correct. Pricing results: {out / 'pricing.csv'}")


if __name__ == "__main__":
    main()
