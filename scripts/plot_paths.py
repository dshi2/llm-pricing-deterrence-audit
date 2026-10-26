"""Offline intervention paths, parent profit benchmarks and F1 paired cells; no paid calls."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.report_figures import groups, plt, save
from sprint.covert_analyze import events
from sprint.fallback1_analyze import tables
from sprint.market import Market


def read_game(path):
    records = events(path)
    if not records:
        raise ValueError(f"No complete journal header: {path}")
    head, branches, status, result = records[0], {}, "incomplete", None
    for event in records[1:]:
        if event["event"] == "round":
            branch, row = event["branch"], event["row"]
            branch_rows = branches.setdefault(branch, {})
            if row["number"] in branch_rows:
                raise ValueError(f"Duplicate round in {path}: {branch}")
            branch_rows[row["number"]] = row
        elif event["event"] == "complete":
            result, status = event["result"], event["result"]["status"]
        elif event["event"] == "failed":
            status = "failed"
    return head, branches, status, result


def late_profit_gain(head, branches):
    """Fixed 100–150 window, including ineligible games; never clip the index to [0,1]."""
    parent = branches.get("parent", {})
    base = {"seed": head["game_seed"], "window_start": 100, "window_end": 150}
    if not all(r in parent for r in range(100, 151)):
        return [base | {"status": "window incomplete"}]
    market = Market(**head["market"])
    nash = np.mean(market.outcomes([market.nash()] * market.n)[1])
    monopoly = np.mean(market.outcomes([market.monopoly()] * market.n)[1])
    if monopoly <= nash:
        raise ValueError("Profit-index denominator must be positive")
    means = np.mean([parent[r]["profits"] for r in range(100, 151)], axis=0)
    return [base | {"status": "complete window", "series": name, "mean_profit": float(value),
                    "nash_profit": float(nash), "monopoly_profit": float(monopoly),
                    "profit_gain": float((value - nash) / (monopoly - nash))}
            for name, value in [(f"seller_{i}", v) for i, v in enumerate(means)] + [("market_mean", means.mean())]]


def removal_opportunity(head, branches):
    """Access to missing retaliation before a completed decision, not psychological awareness."""
    start, trigger = head["rules"]["parent_rounds"], head["rules"]["trigger"]
    rows = []
    for name in ("replaced_memory_neutral", "replaced_reset_neutral"):
        branch = branches.get(name, {})
        cuts = sorted(r for r, row in branch.items() if row["prices"][0] < trigger)
        first = cuts[0] if cuts else None
        # A cuts in t; B's absent retaliation is observed after t+1. Only the decision
        # in t+2 can use that observation. A cut in the penultimate round is too late.
        reveal = first + 1 if first is not None and first + 1 in branch else None
        available = reveal is not None and first + 2 in branch
        rows.append({"branch": name, "observed_rounds": len(branch),
                     "first_undercut_round": first - start if first is not None else None,
                     "first_revealing_outcome_round": reveal - start if reveal is not None else None,
                     "first_completed_decision_with_observation": first + 2 - start if available else None,
                     "opportunity_in_completed_path": available,
                     "scope": "descriptive; not an awareness measure or causal subgroup"})
    return rows


def main_paths(head, branches, result, out, tag, label):
    if not result or result["status"] != "eligible":
        return []
    start, horizon = head["rules"]["parent_rounds"], head["rules"]["horizon"]
    roles, rows = result["roles"], []
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax, mask in zip(axes.flat, ("00", "10", "01", "11")):
        for role, color in zip(("A", "B", "C"), ("tab:blue", "tab:orange", "tab:green")):
            seller = roles[role]
            for kind, style in (("cut", "-"), ("no_cut", "--")):
                name = f"{mask}:{kind}"
                combined = {r: row for r, row in branches.get("parent", {}).items()
                            if start - 9 <= r <= start} | branches.get(name, {})
                points = sorted(combined.items())
                ax.plot([r - start for r, _ in points], [row["prices"][seller] for _, row in points],
                        color=color, ls=style, lw=1.4, label=f"{role} {kind}")
                rows.extend({"mask": mask, "kind": kind, "role": role, "round": r - start,
                             "price": row["prices"][seller]} for r, row in points)
            reference = sorted(branches.get("reference", {}).items())
            ax.plot([r - start for r, _ in reference], [row["prices"][seller] for _, row in reference],
                    color="grey", ls=":", alpha=.65, label="reference" if role == "A" else None)
        ax.axvspan(.5, head["rules"]["cut_rounds"] + .5, color="grey", alpha=.10)
        ax.set(title=f"Live rivals {mask} — seed {head['game_seed']}", xlabel="Round relative to checkpoint",
               ylabel="Normalised price", xlim=(-9, horizon))
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"{label}\nPaired cut/no-cut paths; shaded = prescribed-cut window")
    save(fig, out, f"main_paths_{tag}")
    return rows


F1_PANELS = (("Context retained", ["kept_memory_neutral", "replaced_memory_neutral", "replaced_memory_truth",
                                    "kept_memory_neutral_repeat"]),
             ("Context reset", ["kept_reset_neutral", "replaced_reset_neutral", "replaced_reset_truth"]),
             ("Fresh fixed-rival baseline", ["fresh_fixed"]))


def f1_levels(head):
    """Reference prices: where restraint, punishment and full adjustment to the fixed rival would sit."""
    market, rules = Market(**head["market"]), head["rules"]
    return {"calm rival price": rules["calm"], "punishment trigger": rules["trigger"],
            "punishment price (Nash)": market.nash(),
            "best response to fixed rival": market.best_response(0, (market.cost, rules["calm"]))}


def reference_lines(ax, levels):
    for (name, level), style in zip(levels.items(), ("-", ":", "-.", "--")):
        ax.axhline(level, color="grey", ls=style, lw=.8, label=name)


def f1_paths(head, branches, out, tag, label):
    start = head["rules"]["parent_rounds"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    rows = []
    for ax, (title, names) in zip(axes, F1_PANELS):
        for name in names:
            points = sorted(branches.get(name, {}).items())
            offset = 0 if name == "fresh_fixed" else start
            rounds = [r - offset for r, _ in points]
            complete = rounds == list(range(1, head["rules"]["horizon"] + 1))
            ax.plot(rounds, [row["prices"][0] for _, row in points], label=name.replace("_", " "), lw=1.4)
            rows.extend({"branch": name, "round": r, "price": row["prices"][0], "complete": complete}
                        for r, (_, row) in zip(rounds, points))
        reference_lines(ax, f1_levels(head))
        ax.set(title=title, xlabel="Continuation round", ylabel="Seller A normalised price")
        ax.legend(fontsize=6)
    fig.suptitle(f"{label}\nF1 seed {head['game_seed']}; partial paths remain visible")
    save(fig, out, f"f1_paths_{tag}")
    return rows


def f1_mean_paths(prices, levels, out, label):
    """Mean continuation path per arm over complete paths (all `horizon` rounds), one thin line per
    parent: the treatment time series used in punishment and post-cartel experiments. Interrupted paths
    are left out here, and a model with none complete gets no figure; f1_paths shows them."""
    if prices.empty or not prices["complete"].any():
        return
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for tag, identity, group in groups(prices[prices["complete"]]):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
        for ax, (title, names) in zip(axes, F1_PANELS):
            for name, color in zip(names, colors):
                wide = group[group.branch == name].pivot(index="round", columns="seed", values="price")
                if wide.empty:
                    continue
                ax.plot(wide.index, wide.to_numpy(), color=color, lw=.6, alpha=.3)
                ax.plot(wide.index, wide.mean(axis=1), color=color, lw=2,
                        label=f"{name.replace('_', ' ')} (n={wide.shape[1]})")
            reference_lines(ax, levels[identity["setup"]])
            ax.set(title=title, xlabel="Continuation round", ylabel="Seller A normalised price")
            ax.legend(fontsize=6)
        fig.suptitle(f"{label}\n{identity['model']} — mean of complete paths (bold); one thin line per parent")
        save(fig, out, f"f1_mean_paths_{tag}")


def f1_cells(paths, out, label):
    if paths.empty or "mean_price" not in paths:
        return
    cells = [f"{rival}_{context}_neutral" for context in ("memory", "reset") for rival in ("kept", "replaced")]
    for tag, identity, group in groups(paths):
        complete = group[group.branch.isin(cells) & group.complete]
        wide = complete.pivot(index="seed", columns="branch", values="mean_price").reindex(columns=cells).dropna()
        if wide.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharey=True)
        for ax, context in zip(axes, ("memory", "reset")):
            pair = wide[[f"kept_{context}_neutral", f"replaced_{context}_neutral"]]
            for seed, row in pair.iterrows():
                ax.plot([0, 1], row, "o-", alpha=.55, label=f"seed {seed}")
            ax.set_xticks([0, 1], ["Punisher retained", "Punisher removed"])
            ax.set(title=f"Context {context}; n={len(wide)} complete parents", ylabel="Mean normalised price")
        fig.suptitle(f"{label}\n{identity['model']} — four matched neutral cells")
        save(fig, out, f"f1_cells_{tag}")


def export(kind, run_dirs, out, label):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    prices, profits, opportunities, sources, levels = [], [], [], [], {}
    for directory in map(Path, run_dirs):
        for path in sorted(directory.glob("game_*.jsonl")):
            if path.name.endswith(".calls.jsonl"):
                continue
            head, branches, status, result = read_game(path)
            expected = "main" if kind == "main" else "replacement"
            if head.get("experiment") != expected:
                raise ValueError(f"Expected {expected} journal: {path}")
            setup = {key: head[key] for key in ("experiment", "market", "rules", "model", "code_sha")}
            common = {"run": str(directory.resolve()), "setup": hashlib.sha256(
                json.dumps(setup, sort_keys=True).encode()).hexdigest()[:12],
                "model": head["model"] if isinstance(head["model"], str) else head["model"]["model"],
                "seed": head["game_seed"], "game_status": status, "journal": str(path.resolve())}
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            tag = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:12]
            sources.append({"path": str(path.resolve()), "sha256": digest})
            if kind == "main":
                profits.extend(common | row for row in late_profit_gain(head, branches))
                data = main_paths(head, branches, result, out, tag, label + " — " + common["model"])
            else:
                data = f1_paths(head, branches, out, tag, label + " — " + common["model"])
                levels[common["setup"]] = f1_levels(head)
                opportunities.extend(common | row for row in removal_opportunity(head, branches))
            prices.extend(common | row for row in data)
    pd.DataFrame(prices).to_csv(out / "price_paths.csv", index=False)
    if kind == "main":
        frame = pd.DataFrame(profits)
        frame.to_csv(out / "late_profit_gain.csv", index=False)
        if not frame.empty and "series" in frame:
            for tag, identity, group in groups(frame[frame.series == "market_mean"]):
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.scatter(group.seed, group.profit_gain)
                ax.axhline(0, color="grey", ls="--", label="Nash")
                ax.axhline(1, color="grey", ls=":", label="Joint-profit optimum")
                ax.set(title=f"{label}\n{identity['model']} — complete parent windows 100–150",
                       xlabel="Game seed (one point per game)", ylabel="Normalised profit gain Δ")
                ax.legend()
                save(fig, out, f"main_profit_gain_{tag}")
    else:
        pd.DataFrame(opportunities).to_csv(out / "observation_opportunity.csv", index=False)
        f1_mean_paths(pd.DataFrame(prices), levels, out, label)
        _, paths, _ = tables(run_dirs)
        f1_cells(paths, out, label)
    (out / "sources.json").write_text(json.dumps({"sources": sources, "label": label}, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("main", "f1"))
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="Descriptive results")
    args = parser.parse_args()
    export(args.kind, args.runs, args.out, args.label)
