"""The same saved-data reports for pilot and expansion; no model calls."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t

from sprint import analyze

ANALYSIS = {
    "version": 2, "primary": ["S", "J", "M", "S_minus_J"], "planned_families": 4,
    "familywise_alpha": 0.05, "primary_intervals": 16,
    # Pilot and continuation games share one run folder and one unchanged protocol, so they pool.
    "pooled_stages": True, "minimum_eligible_for_nonpilot_label": 10,
    "relative_gain_scale": 0.5, "equivalence_classification": False,
}

# What an interval wholly above / wholly below zero supports. M is read only in the positive direction.
SIGNS = {
    "S": ("cut pays on average", "cut unprofitable on average"),
    "J": ("cut pays on average", "cut unprofitable on average"),
    "M": ("smaller realised single-rival gain positive on average",
          "negative M does not establish individual deterrence"),
    "S_minus_J": ("two live rivals reduce the cut gain more than one on average",
                  "two live rivals reduce the cut gain less than one on average"),
}


def primary(table, stage):
    """Game-level t intervals, Bonferroni over the planned family of intervals."""
    if stage not in ("pilot", "expansion", "pooled"):
        raise ValueError("stage must be pilot, expansion or pooled")
    rows = []
    if table.empty:
        return pd.DataFrame()
    k, alpha = ANALYSIS["primary_intervals"], ANALYSIS["familywise_alpha"]
    for (run, setup, model), group in table.groupby(["run", "setup", "model"]):
        done = group[group.status == "eligible"]
        for outcome in ANALYSIS["primary"]:
            values = np.array([], dtype=float)
            if len(done):
                values = {"S": (done.G10 + done.G01) / 2, "J": done.G11,
                          "M": np.minimum(done.G10, done.G01),
                          "S_minus_J": (done.G10 + done.G01) / 2 - done.G11}[outcome].to_numpy(float)
            n = len(values)
            mean = float(values.mean()) if n else np.nan
            lo = hi = np.nan
            interpretation = "uncertainty unavailable"
            if n >= 2 and np.isfinite(values).all() and values.std(ddof=1) > 0:
                radius = t.ppf(1 - alpha / (2 * k), n - 1) * values.std(ddof=1) / np.sqrt(n)
                lo, hi = mean - radius, mean + radius
                interpretation = "sign unresolved"
                if lo > 0:
                    interpretation = SIGNS[outcome][0]
                elif hi < 0:
                    interpretation = SIGNS[outcome][1]
            leave_one_out = (values.sum() - values) / (n - 1) if n > 1 else np.array([np.nan])
            rows.append({"run": run, "setup": setup, "model": model, "stage": stage,
                         "outcome": outcome, "eligible": n, "mean": mean, "lo": lo, "hi": hi,
                         "method": f"Student-t, Bonferroni over {k} planned intervals",
                         "level": 1 - alpha / k, "interpretation": interpretation,
                         "scope": "exploratory" if stage == "pilot" or n < ANALYSIS["minimum_eligible_for_nonpilot_label"]
                         else "prespecified; assumptions apply",
                         "leave_one_out_min": leave_one_out.min(), "leave_one_out_max": leave_one_out.max()})
    return pd.DataFrame(rows)


def late_price_trends(events):
    """Descriptive fixed-window diagnostics in normalised prices, never a convergence test."""
    window = [e["row"] for e in events if e.get("event") == "round"
              and e.get("branch") == "parent" and 100 <= e["row"]["number"] <= 150]
    window.sort(key=lambda row: row["number"])
    base = {"seed": events[0]["game_seed"], "window_start": 100, "window_end": 150,
            "rounds_observed": len(window), "units": "normalised price"}
    if [row["number"] for row in window] != list(range(100, 151)):
        return pd.DataFrame([{**base, "status": "window incomplete"}])
    prices = np.array([row["prices"] for row in window], dtype=float)
    x = np.arange(51, dtype=float) - 25
    rows = []
    series = [(f"seller_{i}", prices[:, i]) for i in range(prices.shape[1])]
    series.append(("market_mean", prices.mean(axis=1)))
    for seller, values in series:
        slope = float(x @ (values - values.mean()) / (x @ x))
        rows.append({**base, "status": "complete window", "series": seller,
                     "mean": float(values.mean()), "slope_per_round": slope,
                     "fitted_change_50_rounds": 50 * slope,
                     "late_minus_early_mean": float(values[26:].mean() - values[:25].mean()),
                     "sd": float(values.std(ddof=1)),
                     "mean_absolute_step": float(np.abs(np.diff(values)).mean())})
    return pd.DataFrame(rows)


def export(run_dirs, out, stage, planned_seeds=None):
    """Export incomplete runs too. Each model/configuration remains a separate analysis group.
    `planned_seeds` is one list for every run, or a {run folder name: seeds} mapping."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    table = analyze.games(run_dirs)
    if planned_seeds is not None:
        missing = []
        for directory in map(Path, run_dirs):
            planned = planned_seeds.get(directory.name, []) if isinstance(planned_seeds, dict) else planned_seeds
            seen = set(table.loc[table.run == directory.name, "seed"]) if not table.empty else set()
            missing += [{"run": directory.name, "seed": seed, "status": "not_started"}
                        for seed in planned if seed not in seen]
        table = pd.concat([table, pd.DataFrame(missing)], ignore_index=True)
    table.to_csv(out / "games.csv", index=False)
    if not table.empty:
        table.groupby(["run", "status"]).size().rename("games").to_csv(out / "accounting.csv")
    observed = table.dropna(subset=["setup"]) if "setup" in table else pd.DataFrame()
    descriptive = analyze.summary(observed) if not observed.empty else pd.DataFrame()
    descriptive.to_csv(out / "descriptive.csv", index=False)
    result = primary(observed, stage)
    result.to_csv(out / "primary.csv", index=False)
    effects = observed[observed.status == "eligible"].copy() if not observed.empty else pd.DataFrame()
    if not effects.empty:
        effects["D1"] = effects.R1 - 0.5 * effects.G00
        effects["D2"] = effects.R2 - 0.5 * effects.G00
        effects["S_minus_J"] = (effects.G10 + effects.G01) / 2 - effects.G11
        effects["observed_joint_only"] = (effects.G10 > 0) & (effects.G01 > 0) & (effects.G11 < 0)
    effects.to_csv(out / "game_effects.csv", index=False)
    for directory in map(Path, run_dirs):
        family = out / directory.name
        family.mkdir(exist_ok=True)
        paths = [p for p in sorted(directory.glob("game_*.jsonl")) if not p.name.endswith(".calls.jsonl")]
        gaps, responses, trends = [], [], []
        for path in paths:
            gaps.append(analyze.best_response_gaps(path))
            response = analyze.rival_responses(path)
            response["journal"] = path.name
            responses.append(response)
            events = analyze._events(path)
            trend = late_price_trends(events)
            trend["journal"] = path.name
            trends.append(trend)
            if any(e.get("branch") == "parent" and e["event"] == "round" for e in events):
                ax = analyze.plot_prices(path)
                ax.figure.tight_layout()
                ax.figure.savefig(family / f"{path.stem}_prices.png", dpi=160)
                plt.close(ax.figure)
        for name, frames in (("best_response_gaps", gaps), ("rival_responses", responses),
                             ("late_price_trends", trends)):
            if frames:
                pd.concat(frames, ignore_index=True).to_csv(family / f"{name}.csv", index=False)
        if not effects.empty:
            selected = effects[effects.run == directory.name]
            if not selected.empty:
                ax = analyze.plot_gains(selected)
                ax.set_title("Descriptive 95% bootstrap intervals; not primary intervals")
                ax.figure.tight_layout()
                ax.figure.savefig(family / "gains.png", dpi=160)
                plt.close(ax.figure)
                ax = selected.set_index("seed")[["G10", "G01", "G11"]].plot(marker="o", linestyle="none")
                ax.axhline(0, color="black", linewidth=0.8)
                ax.set_ylabel("Prescribed-cut gain (each eligible game)")
                ax.figure.tight_layout()
                ax.figure.savefig(family / "individual_games.png", dpi=160)
                plt.close(ax.figure)
    (out / "analysis.json").write_text(json.dumps({"stage": stage, **ANALYSIS}, indent=2) + "\n")
    return table, result
