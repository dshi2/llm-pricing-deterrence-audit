"""Main study: does either rival alone, or only both together, make a prescribed price cut unprofitable?

Parent game -> checkpoint -> all-live reference continuation -> full-window eligibility -> deviator A
drawn among eligible sellers -> four rival conditions (B, C live or replayed), each with a cut branch
and a matched no-cut branch; A's whole path is fixed in every branch -> A/A repeat of no-cut 11.
"""

import math
import random
from dataclasses import asdict, dataclass

from sprint.engine import Journal, Policy, Round, play
from sprint.market import Market

MASKS = ("00", "10", "01", "11")  # (B live?, C live?)


@dataclass(frozen=True)
class MainRules:
    parent_rounds: int = 150
    horizon: int = 20
    cut_rounds: int = 10
    cut_fraction: float = 0.10
    discount: float = 0.95
    eligibility_margin: float = 0.01
    window: int = 100

    def __post_init__(self):
        for name in ("parent_rounds", "horizon", "cut_rounds", "window"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.cut_rounds > self.horizon:
            raise ValueError("the cut cannot outlast the continuation")
        if not 0 < self.cut_fraction < 1 or not 0 < self.discount <= 1 or self.eligibility_margin < 0:
            raise ValueError("invalid cut fraction, discount or eligibility margin")

    def planned_calls(self) -> dict:
        """Decision calls per game before retries: parent + reference for three sellers, then live rivals
        in the response branches (0 + 1 + 1 + 2 per cut/no-cut) and the two-rival A/A repeat."""
        ineligible = 3 * (self.parent_rounds + self.horizon)
        return {"eligible": ineligible + 2 * 4 * self.horizon + 2 * self.horizon, "ineligible": ineligible}


def discounted(rows, seller: int, discount: float) -> float:
    return math.fsum(discount**t * row.profits[seller] for t, row in enumerate(rows))


def cut_path(reference: tuple[Round, ...], seller: int, rules: MainRules) -> list[tuple[float, ...]]:
    """Reference prices with `seller` at (1 - cut) x its own reference price for the first cut rounds."""
    return [tuple(p * (1 - rules.cut_fraction) if i == seller and t < rules.cut_rounds else p
                  for i, p in enumerate(row.prices)) for t, row in enumerate(reference)]


def eligibility(reference: tuple[Round, ...], market: Market, rules: MainRules) -> list[dict]:
    """Each seller's discounted gain from the prescribed cut with both rivals held to the reference."""
    candidates = []
    for i in range(market.n):
        baseline = discounted(reference, i, rules.discount)
        cut = math.fsum(rules.discount**t * market.outcomes(prices)[1][i]
                        for t, prices in enumerate(cut_path(reference, i, rules)))
        gain = cut - baseline
        candidates.append({"seller": i, "baseline": baseline, "unopposed_gain": gain,
                           "eligible": baseline > 0 and gain > 0 and gain >= rules.eligibility_margin * baseline})
    return candidates


def replay(prices: list[float], start: int):
    """Programmed seller that posts prices[k] in the k-th round after `start` completed rounds."""
    return lambda rows: prices[len(rows) - start]


async def main_game(policy: Policy, game_seed: int, journal: Journal, market: Market = Market(3),
                    rules: MainRules = MainRules(), label: str = "") -> dict:
    if market.n != 3:
        raise ValueError("the main study has three sellers")
    if journal.result is not None:
        return journal.result
    if journal.failure is not None:
        return {"status": "failed", "game_seed": game_seed, "reason": journal.failure}

    def run(branch, start, rounds, live, programs):
        return play(journal, branch, start, rounds, market=market, game_seed=game_seed, policy=policy,
                    live=live, programs=programs, window=rules.window, label=label)

    everyone = list(range(market.n))
    parent = await run("parent", (), rules.parent_rounds, everyone, {})
    reference = await run("reference", parent, rules.horizon, everyone, {})
    candidates = eligibility(reference, market, rules)
    pool = [c["seller"] for c in candidates if c["eligible"]]
    selected = random.Random(f"selection:{game_seed}").choice(pool) if pool else None  # before any response
    selection = {"candidates": candidates, "selected": selected,
                 "probabilities": [1 / len(pool) if i in pool else 0.0 for i in everyone]}
    base = {"game_seed": game_seed, "rules": asdict(rules), "selection": selection}
    if selected is None:
        return journal.finish({"status": "ineligible", **base})

    rivals = [i for i in everyone if i != selected]
    random.Random(f"labels:{game_seed}").shuffle(rivals)  # rivals[0] is B, rivals[1] is C
    t = len(parent)
    paths = {"no_cut": [row.prices for row in reference], "cut": cut_path(reference, selected, rules)}
    cells = [(mask, kind) for mask in MASKS for kind in ("no_cut", "cut")]
    random.Random(f"order:{game_seed}").shuffle(cells)

    returns = {}
    for mask, kind in cells:
        live = [r for r, on in zip(rivals, mask) if on == "1"]
        programs = {i: replay([p[i] for p in paths[kind]], t) for i in everyone if i not in live}
        rows = await run(f"{mask}:{kind}", parent, rules.horizon, live, programs)
        returns[mask, kind] = discounted(rows, selected, rules.discount)
    gains = {mask: returns[mask, "cut"] - returns[mask, "no_cut"] for mask in MASKS}
    if not math.isclose(gains["00"], candidates[selected]["unopposed_gain"], rel_tol=1e-9, abs_tol=1e-9):
        raise AssertionError("replayed 00 branch disagrees with the eligibility calculation")

    repeat = await run("11:no_cut:repeat", parent, rules.horizon, rivals,
                       {selected: replay([p[selected] for p in paths["no_cut"]], t)})
    r1 = gains["00"] - (gains["10"] + gains["01"]) / 2
    r2 = gains["00"] - gains["11"]
    return journal.finish({
        "status": "eligible", **base, "roles": {"A": selected, "B": rivals[0], "C": rivals[1]},
        "gains": gains, "R1": r1, "R2": r2, "nonadditivity": r2 - 2 * r1,
        "aa_difference": discounted(repeat, selected, rules.discount) - returns["11", "no_cut"],
    })
