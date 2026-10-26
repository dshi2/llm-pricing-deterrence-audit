"""A/A repeatability from saved main-game journals; no paid calls.

The repeat `11:no_cut:repeat` replays `11:no_cut` from the same checkpoint and requested seeds, so the live
rivals' first prompts are identical. Replies are compared round by round up to the first round in which any
rival's reply differs (price or notes); after it the prompts differ and matching is no longer expected.
Only completed games are used.
"""

import argparse
from pathlib import Path

import pandas as pd

from sprint.covert_analyze import events

ORIGINAL, REPEAT = "11:no_cut", "11:no_cut:repeat"


def game_agreement(path):
    records = events(path)
    if not any(e["event"] == "complete" for e in records):
        return None  # unfinished, e.g. stopped mid-repeat by credit: a partial repeat would overstate matching
    decisions = {}
    for e in records[1:]:
        if e["event"] == "decision" and e["branch"] in (ORIGINAL, REPEAT):
            decisions.setdefault(e["branch"], {})[e["round"], e["seller"]] = e["decision"]
    original, repeat = decisions.get(ORIGINAL, {}), decisions.get(REPEAT, {})
    rounds = sorted({r for r, _ in original} & {r for r, _ in repeat})
    if not rounds:
        return None
    sellers = sorted({s for r, s in original if r == rounds[0]})
    matched, first_difference = 0, None
    for r in rounds:
        if any(original[r, s] != repeat[r, s] for s in sellers):
            first_difference = r
            break
        matched += 1
    first = rounds[0]
    return {"seed": records[0]["game_seed"], "rounds_compared": len(rounds),
            "identical_rounds_before_first_difference": matched,
            "first_difference_round": None if first_difference is None else first_difference - first + 1,
            "first_round_identical_replies": sum(original[first, s] == repeat[first, s] for s in sellers),
            "first_round_identical_prices": sum(original[first, s]["price"] == repeat[first, s]["price"]
                                                for s in sellers),
            "rivals": len(sellers)}


def agreement(run_dirs):
    rows = []
    for run_dir in map(Path, run_dirs):
        for path in sorted(run_dir.glob("game_*.jsonl")):
            if path.name.endswith(".calls.jsonl"):
                continue
            row = game_agreement(path)
            if row:
                rows.append({"run": run_dir.name, **row})
    return pd.DataFrame(rows)


def summary(table):
    """Per run: share of first-round rival replies (identical prompts by construction) that repeat exactly."""
    if table.empty:
        return table
    return table.groupby("run").apply(lambda g: pd.Series({
        "games": len(g),
        "first_round_identical_reply_share": g.first_round_identical_replies.sum() / g.rivals.sum(),
        "first_round_identical_price_share": g.first_round_identical_prices.sum() / g.rivals.sum(),
        "games_identical_throughout": int(g.first_difference_round.isna().sum()),
        "median_identical_rounds": g.identical_rounds_before_first_difference.median()}),
        include_groups=False).reset_index()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    table = agreement(args.runs)
    table.to_csv(args.out / "aa_agreement_games.csv", index=False)
    summary(table).to_csv(args.out / "aa_agreement.csv", index=False)
    print(summary(table).to_string(index=False))
