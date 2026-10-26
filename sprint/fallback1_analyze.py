"""Descriptive paired estimates for fallback 1. Families, configurations and run directories stay apart."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

from sprint.analyze import _call_stats
from sprint.analyze import launch_checks as shared_launch_checks
from sprint.engine import Round
from sprint.fallback1 import CELLS, ReplacementRules, checkpoint, dynamic_benchmark, score_path
from sprint.market import Market

# All differences are first minus second; pair within the same parent, never across games.
CONTRASTS = {
    "replacement_memory": ("replaced_memory_neutral", "kept_memory_neutral"),
    "replacement_reset": ("replaced_reset_neutral", "kept_reset_neutral"),
    "memory_under_replacement": ("replaced_reset_neutral", "replaced_memory_neutral"),
    "information_memory": ("replaced_memory_truth", "replaced_memory_neutral"),
    "information_reset": ("replaced_reset_truth", "replaced_reset_neutral"),
    "aa_repeat": ("kept_memory_neutral_repeat", "kept_memory_neutral"),
}

# (Removal effect with context reset) minus (removal effect with context retained).
REMOVAL_CONTEXT_WEIGHTS = {
    "replaced_reset_neutral": 1, "kept_reset_neutral": -1,
    "replaced_memory_neutral": -1, "kept_memory_neutral": 1,
}


def tables(run_dirs):
    games, paths, contrasts = [], [], []
    for directory in map(Path, run_dirs):
        for path in sorted(directory.glob("game_*.jsonl")):
            if path.name.endswith(".calls.jsonl"):
                continue
            events = []
            lines = path.read_text().splitlines()
            for i, line in enumerate(lines):
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    if i != len(lines) - 1:
                        raise
                    # Report the last complete prefix without mutating a live journal.
            head = events[0]
            if head["experiment"] != "replacement":
                raise ValueError("fallback analysis received a different experiment")
            market, rules = Market(**head["market"]), ReplacementRules(**head["rules"])
            setup = {k: head[k] for k in ("experiment", "model", "market", "rules", "code_sha")}
            key = {"run": str(directory.resolve()), "setup": hashlib.sha256(
                json.dumps(setup, sort_keys=True).encode()).hexdigest()[:16],
                "model": head["model"] if isinstance(head["model"], str) else head["model"]["model"],
                "seed": head["game_seed"]}
            branches, status, reason = {}, "incomplete", None
            for event in events[1:]:
                if event["event"] == "round":
                    branches.setdefault(event["branch"], []).append(Round.from_dict(event["row"]))
                elif event["event"] == "complete":
                    status = event["result"]["status"]
                elif event["event"] == "failed":
                    status, reason = "failed", event["reason"]
                elif event["event"] == "interrupted":
                    reason = event["reason"]
            parent = branches.get("parent", [])
            gate = checkpoint(parent, rules) if len(parent) == rules.parent_rounds else {}
            games.append(key | {"status": status, "reason": reason, "parent_rounds": len(parent)} |
                         gate | _call_stats(path.with_name(path.stem + ".calls.jsonl")))
            benchmark = dynamic_benchmark(market, rules)
            complete = {}
            for branch, rows in branches.items():
                if branch == "parent":
                    continue
                replaced = branch == "fresh_fixed" or CELLS.get(branch, (False,))[0]
                metrics = score_path(rows, market, rules, benchmark, replaced) if len(rows) == rules.horizon else {}
                paths.append(key | {"branch": branch, "observed_rounds": len(rows),
                                    "complete": len(rows) == rules.horizon,
                                    "restrained": gate.get("restrained")} | metrics)
                if metrics:
                    complete[branch] = metrics
            for name, (left, right) in CONTRASTS.items():
                if left in complete and right in complete:
                    # Actual profits, not regret against two different benchmarks.
                    for metric in ("first_price", "mean_price", "discounted_profit"):
                        contrasts.append(key | {"contrast": name, "outcome": metric,
                                                "restrained": gate.get("restrained", False),
                                                "difference": complete[left][metric] - complete[right][metric]})
            if REMOVAL_CONTEXT_WEIGHTS.keys() <= complete.keys():
                for metric in ("mean_price", "discounted_profit"):
                    difference = sum(weight * complete[branch][metric]
                                     for branch, weight in REMOVAL_CONTEXT_WEIGHTS.items())
                    contrasts.append(key | {"contrast": "removal_context_interaction", "outcome": metric,
                                            "restrained": gate.get("restrained", False),
                                            "difference": difference})
    return pd.DataFrame(games), pd.DataFrame(paths), pd.DataFrame(contrasts)


# Primary D: removal effect on mean price with context kept, over all calm checkpoints. Fixed before any
# F1 data: Bonferroni over the four models; fewer than ten pairs keeps the exploratory label.
PRIMARY, PRIMARY_MODELS, MIN_PAIRS = ("replacement_memory", "mean_price"), 4, 10


def summary(contrasts):
    output = []
    if contrasts.empty:
        return pd.DataFrame()
    for subgroup in ("all_calm_checkpoints", "restrained_checkpoints"):
        frame = contrasts if subgroup == "all_calm_checkpoints" else contrasts[contrasts.restrained]
        for keys, group in frame.groupby(["run", "setup", "model", "contrast", "outcome"]):
            values = group.difference.to_numpy()
            n, mean = len(values), float(values.mean())
            primary = subgroup == "all_calm_checkpoints" and keys[3:] == PRIMARY
            alpha = 0.05 / PRIMARY_MODELS if primary else 0.05
            se = float(np.std(values, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
            half = float(t.ppf(1 - alpha / 2, n - 1) * se) if n > 1 and se > 0 else float("nan")
            interval = "descriptive unadjusted 95% t; exploratory"
            if primary:
                interval = f"primary: t, Bonferroni over {PRIMARY_MODELS} models" + (
                    "" if n >= MIN_PAIRS else f"; exploratory (< {MIN_PAIRS} pairs)")
            output.append(dict(zip(["run", "setup", "model", "contrast", "outcome"], keys)) |
                          {"subgroup": subgroup, "n_pairs": n, "mean": mean, "low": mean - half,
                           "high": mean + half, "level": 1 - alpha, "interval": interval})
    return pd.DataFrame(output)


def export_tables(run_dirs, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    games, paths, contrasts = tables(run_dirs)
    for name, table in (("games", games), ("paths", paths), ("contrasts", contrasts),
                        ("summary", summary(contrasts))):
        table.to_csv(out / f"replacement_{name}.csv", index=False)
    return games, paths, contrasts


def launch_checks(canary_dir, probe_dir):
    checks = shared_launch_checks(canary_dir, probe_dir)
    games, _, _ = tables([canary_dir])
    checks["canary game completed"] = not games.empty and games.status.eq("complete").all()
    return checks
