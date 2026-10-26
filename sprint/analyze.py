"""From journals and call logs to tables and figures. Games are the unit; families are kept apart.

Every started game appears in games(): eligible, ineligible, failed, interrupted or incomplete. Effects
are summarised only over complete eligible games, with the other statuses reported beside them.
"""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sprint.experiments import MASKS
from sprint.market import Market

UNSTABLE_BELOW = 10  # fewer eligible games than this: intervals are labelled unstable
OUTCOMES = ("unusable", "transport_error", "declined", "fatal", "cancelled", "error")  # non-ok attempts


def _events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def games(run_dirs) -> pd.DataFrame:
    rows = []
    for run_dir in map(Path, run_dirs):
        for path in sorted(run_dir.glob("game_*.jsonl")):
            if path.name.endswith(".calls.jsonl"):
                continue
            events = _events(path)
            head = events[0]
            model = head["model"] if isinstance(head["model"], str) else head["model"]["model"]
            setup = {k: head[k] for k in ("experiment", "rules", "market", "model", "code_sha")}
            row = {"run": run_dir.name, "model": model,
                   "setup": hashlib.sha256(json.dumps(setup, sort_keys=True).encode()).hexdigest()[:8],
                   "seed": head["game_seed"], "status": "incomplete"}
            for e in events:
                if e["event"] == "complete":
                    r = e["result"]
                    row["status"] = r["status"]
                    if r["status"] == "eligible":
                        row |= {"A": r["roles"]["A"], **{f"G{m}": r["gains"][m] for m in MASKS},
                                "R1": r["R1"], "R2": r["R2"], "nonadditivity": r["nonadditivity"],
                                "aa_difference": r["aa_difference"]}
                elif e["event"] == "failed":
                    row |= {"status": "failed", "reason": e["reason"]}
                elif e["event"] == "interrupted" and row["status"] == "incomplete":
                    row["reason"] = e["reason"]
            row |= _call_stats(path.with_name(path.stem + ".calls.jsonl"))
            rows.append(row)
    return pd.DataFrame(rows)


def _call_stats(path: Path) -> dict:
    if not path.exists():
        return {}
    calls = pd.DataFrame(_events(path))
    counts = calls["outcome"].value_counts()
    return {"attempts": len(calls), **{k: int(counts.get(k, 0)) for k in OUTCOMES},
            "providers_served": ",".join(sorted(calls["provider_served"].dropna().unique())),
            "prompt_tokens": calls["prompt_tokens"].sum(min_count=1),
            "completion_tokens": calls["completion_tokens"].sum(min_count=1),
            "cost": calls["cost"].sum(min_count=1)}


def _attempts(run_dir) -> list[dict]:
    return [a for path in sorted(Path(run_dir).glob("*.calls.jsonl")) for a in _events(path)]


def launch_checks(canary_dir, probe_dir, max_unusable: int = 2) -> dict[str, bool]:
    """Integration criteria for launching a pilot, read from the canary's and the probe's logs.
    All must hold; a transport error that was retried successfully does not block."""
    table, attempts, probe = games([canary_dir]), _attempts(canary_dir), _attempts(probe_dir)
    outcomes = Counter(a["outcome"] for a in attempts + probe)
    replies = [a for a in attempts + probe if a["http_status"] == 200]
    return {
        "canary game completed": len(table) > 0 and table.status.isin(["eligible", "ineligible"]).all(),
        "no fatal, declined or error attempts": not (outcomes["fatal"] or outcomes["declined"] or outcomes["error"]),
        f"at most {max_unusable} unusable attempts": outcomes["unusable"] <= max_unusable,
        "context probe answered and parsed": any(a["outcome"] == "ok" for a in probe),
        "every reply served by the pinned provider": bool(replies) and all(
            (a["provider_served"] or "").lower() == a["provider_requested"].lower() for a in replies),
        "cost recorded for every reply": bool(replies) and all(a["cost"] is not None for a in replies),
    }


def bootstrap_ci(values, n_boot: int = 10_000, seed: int = 0) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return (float("nan"), float("nan"))
    draws = np.random.default_rng(seed).choice(values, size=(n_boot, len(values))).mean(axis=1)
    return tuple(np.percentile(draws, [2.5, 97.5]))


def verdict(ci: tuple[float, float]) -> str:
    """Describe an unadjusted interval; make no significance or equivalence decision."""
    lo, hi = ci
    if np.isnan(lo):
        return "too few games"
    if lo > 0 or hi < 0:
        return "excludes 0 (unadjusted)"
    return "includes 0"


def summary(table: pd.DataFrame) -> pd.DataFrame:
    """Per run and setup (experiment, rules, market, model incl. prompt units, code version), never
    pooled across them: status counts, eligibility, and mean effects with game-level 95% bootstrap
    intervals. These are descriptive: precision, equivalence and multiplicity rules are not implemented.
    """
    rows = []
    for (run, setup, model), g in table.groupby(["run", "setup", "model"]):
        done = g[g.status == "eligible"]
        reasons = g.get("reason", pd.Series(dtype=str)).fillna("")
        classified_games = int(g.status.isin(["eligible", "ineligible"]).sum())
        eligibility_rate = len(done) / classified_games if classified_games else float("nan")
        row = {"run": run, "setup": setup, "model": model, "started": len(g), **g.status.value_counts().to_dict(),
               "declined_games": int(reasons.str.startswith("declined").sum()),
               "eligibility_classified_games": classified_games, "eligibility_rate": eligibility_rate}
        for col in [f"G{m}" for m in MASKS] + ["R1", "R2", "nonadditivity", "aa_difference"]:
            ci = bootstrap_ci(done[col]) if len(done) else (float("nan"), float("nan"))
            row |= {col: done[col].mean() if len(done) else float("nan"), f"{col}_lo": ci[0], f"{col}_hi": ci[1]}
        for col in ("R1", "R2"):
            row[f"{col}_verdict"] = verdict((row[f"{col}_lo"], row[f"{col}_hi"]))
        row["G11_negative_share"] = (done["G11"] < 0).mean() if len(done) else float("nan")
        row["intervals"] = "descriptive; small sample" if len(done) < UNSTABLE_BELOW else "descriptive"
        rows.append(row)
    return pd.DataFrame(rows)


def rival_responses(journal_path, market: Market = Market(3)) -> pd.DataFrame:
    """Describe next-round price responses to observed cuts, without inferring punitive intent.

    Compare response rounds 2..cut_rounds+1, clipped to the recorded horizon. For a response in round r,
    the reference benchmark uses prices from r-1, with only A's prescribed cut applied. The other
    prices are held at that lagged reference, not at their endogenous live-branch values. Thus
    `myopic_response` is an isolated, lagged best-response change, not the agent's optimal dynamic policy.
    """
    events = _events(Path(journal_path))
    result = next((e["result"] for e in events if e["event"] == "complete"), None)
    if not result or result["status"] != "eligible":
        return pd.DataFrame()
    rules, roles = result["rules"], result["roles"]
    branches = {}
    for e in events:
        if e["event"] == "round":
            branches.setdefault(e["branch"], {})[e["row"]["number"]] = e["row"]["prices"]
    t = rules["parent_rounds"]
    last = t + min(rules["horizon"], rules["cut_rounds"] + 1)
    window = range(t + 2, last + 1)
    if not window:  # no recorded decision after the first cut was observable
        return pd.DataFrame(columns=["mask", "rival", "response", "myopic_response"])
    reference = branches["reference"]
    rows = []
    for mask in MASKS:
        for name, rival in (("B", roles["B"]), ("C", roles["C"])):
            if mask[0 if name == "B" else 1] != "1":
                continue
            cut, base = branches[f"{mask}:cut"], branches[f"{mask}:no_cut"]
            myopic = []
            for r in window:
                previous = list(reference[r - 1])
                prices = previous.copy()
                prices[roles["A"]] *= 1 - rules["cut_fraction"]
                myopic.append(market.best_response(rival, prices) - market.best_response(rival, previous))
            rows.append({"mask": mask, "rival": name,
                         "response": float(np.mean([cut[r][rival] - base[r][rival] for r in window])),
                         "myopic_response": float(np.mean(myopic))})
    return pd.DataFrame(rows)


def best_response_gaps(journal_path) -> pd.DataFrame:
    """Ex-post static profit gaps in parent/reference rounds only, where every seller acts freely.

    Uses realised simultaneous rival prices, which were not known when the seller acted. A gap can
    reflect uncertainty or dynamic incentives; it is not an arithmetic-error or collusion label.
    Price and profit columns use the journal's normalised units. No additional model calls.
    """
    events = _events(Path(journal_path))
    head = events[0]
    if head.get("experiment") != "main":
        raise ValueError("best_response_gaps requires a main-study journal")
    market = Market(**head["market"])
    rows = []
    for e in events:
        if e["event"] != "round" or e["branch"] not in ("parent", "reference"):
            continue
        r = e["row"]
        for seller, price in enumerate(r["prices"]):
            best = market.best_response(seller, r["prices"])
            alternative = list(r["prices"])
            alternative[seller] = best
            optimal_profit = market.outcomes(alternative)[1][seller]
            rows.append({"seed": head["game_seed"], "branch": e["branch"], "round": r["number"],
                         "seller": seller, "price": price, "best_response": best,
                         "profit": r["profits"][seller], "optimal_profit": optimal_profit,
                         "profit_gap": optimal_profit - r["profits"][seller]})
    return pd.DataFrame(rows)


def competence_summary(result: dict) -> dict:
    """Secondary summaries from saved diagnostics; no calls or changes to primary scoring."""
    cases = result["arithmetic"]
    answered = [r for r in cases if r["status"] == "answered"]
    arithmetic = {"recorded": len(cases), "answered": len(answered),
                  "failed": sum(r["status"] == "failed" for r in cases),
                  "correct_strict": sum(r["correct"] for r in answered),
                  "within_0_01": sum(math.isclose(r["answer"], r["expected"], rel_tol=0, abs_tol=0.01)
                                     for r in answered)}
    pricing = []
    for i, episode in enumerate(result["pricing"]):
        for start, stop in ((1, 10), (6, 10)):
            rows = [r for r in episode.get("rounds", []) if start <= r["round"] <= stop]
            pricing.append({"episode": episode.get("episode", i), "status": episode["status"],
                            "window": f"{start}-{stop}", "observed_rounds": len(rows),
                            "mean_profit_gap": math.fsum(r["profit_gap"] for r in rows) / len(rows)
                            if rows else None})
    return {"arithmetic": arithmetic, "pricing": pricing}


def plot_gains(table: pd.DataFrame, ax=None):
    """A's deviation gain by rival condition, per model: mean and 95% game-level interval.
    Pass games from one run and setup only."""
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    done = table[table.status == "eligible"]
    if done[["run", "setup"]].drop_duplicates().shape[0] > 1:
        raise ValueError("plot_gains got games from more than one run or setup")
    models = sorted(done.model.unique())
    width = 0.8 / max(1, len(models))
    for k, model in enumerate(models):
        g = done[done.model == model]
        means = [g[f"G{m}"].mean() for m in MASKS]
        cis = [bootstrap_ci(g[f"G{m}"]) for m in MASKS]
        err = [[m - lo for m, (lo, hi) in zip(means, cis)], [hi - m for m, (lo, hi) in zip(means, cis)]]
        ax.bar(np.arange(4) + k * width, means, width, yerr=err, capsize=3, label=f"{model} (n={len(g)})")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(4) + width * (len(models) - 1) / 2,
                  ["neither responds", "B only", "C only", "both respond"])
    ax.set_ylabel("A's discounted gain from the cut")
    ax.legend(fontsize=8)
    return ax


def plot_prices(journal_path, ax=None, alpha: float = 1.0):
    """Parent-game price paths (normalised) with the three-firm Nash and monopoly benchmarks."""
    ax = ax or plt.subplots(figsize=(7, 3))[1]
    rows = [e["row"] for e in _events(Path(journal_path)) if e["event"] == "round" and e["branch"] == "parent"]
    prices = np.array([r["prices"] for r in rows])
    for i in range(prices.shape[1]):
        ax.plot([r["number"] for r in rows], prices[:, i], lw=1, label=f"Firm {'ABC'[i]}")
    for level, name in ((Market(3).nash(), "Nash"), (Market(3).monopoly(), "monopoly")):
        ax.axhline(level, ls="--", lw=0.8, color="grey")
        ax.text(1, level, name, fontsize=7, va="bottom")
    ax.set_xlabel("round")
    ax.set_ylabel("price / alpha")
    ax.legend(fontsize=7)
    return ax
