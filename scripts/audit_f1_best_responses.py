"""Descriptive, hindsight static best responses to recorded F1 rival prices.

This is not a feasible-information policy or a dynamic optimum, and agreement with it does not
identify the seller's decision rule. No API calls.
"""

import argparse
import csv
import hashlib
import json
from functools import lru_cache
from pathlib import Path

from sprint.market import Market

BRANCHES = ("kept_memory_neutral", "replaced_memory_neutral", "kept_memory_neutral_repeat")


def audit(inputs, out):
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    rows, accounting, hashes = [], [], {}
    for path in sorted(inputs.glob("game_*.jsonl")):
        if ".calls." in path.name:
            continue
        raw = path.read_bytes()
        hashes[path.name] = hashlib.sha256(raw).hexdigest()
        events = [json.loads(line) for line in raw.splitlines()]
        head = events[0]
        if head["experiment"] != "replacement" or head["market"]["n"] != 2:
            raise ValueError("Expected two-seller F1 journals")
        market, rules = Market(**head["market"]), head["rules"]
        best_response = lru_cache(None)(lambda rival, m=market: m.best_response(0, (m.cost, rival)))
        branches = {}
        for e in events:
            if e["event"] == "round":
                branches.setdefault(e["branch"], []).append(e["row"])
        model = head["model"] if isinstance(head["model"], str) else head["model"]["model"]
        terminal = [e for e in events if e["event"] in ("complete", "failed")]
        status = (terminal[-1]["result"]["status"] if terminal and terminal[-1]["event"] == "complete"
                  else "unfinished_or_failed")
        accounting.append({"model": model, "seed": head["game_seed"], "status": status})
        expected = list(range(rules["parent_rounds"] + 1, rules["parent_rounds"] + rules["horizon"] + 1))
        for branch in BRANCHES:
            path_rows = branches.get(branch, [])
            complete = [r["number"] for r in path_rows] == expected
            for row in path_rows:
                price, rival = row["prices"]
                br = best_response(rival)
                profit = market.outcomes((price, rival))[1][0]
                rows.append({"model": model, "seed": head["game_seed"], "branch": branch,
                             "complete_path": complete, "round": row["number"],
                             "price": price, "rival_price": rival, "static_best_response": br,
                             "signed_price_gap": price - br, "absolute_price_gap": abs(price - br),
                             "static_profit_gap": market.outcomes((br, rival))[1][0] - profit})
    out.mkdir(parents=True)
    for name, records in [("round_comparison", rows), ("game_accounting", accounting)]:
        with (out / f"{name}.csv").open("w") as f:
            if records:
                writer = csv.DictWriter(f, fieldnames=list(records[0]))
                writer.writeheader()
                writer.writerows(records)
    (out / "manifest.json").write_text(json.dumps({
        "inputs": str(inputs.resolve()), "input_sha256": hashes,
        "interpretation": "Post-hoc descriptive benchmark against realised simultaneous rival prices; "
                          "not an identified decision rule or dynamic optimum.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, help="F1 model run folder with game_*.jsonl journals")
    parser.add_argument("--out", required=True, type=Path, help="New output folder; never overwrites")
    args = parser.parse_args()
    audit(args.inputs, args.out)
