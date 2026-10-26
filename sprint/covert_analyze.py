"""Evidence-specific F2 scores; base-case blocks, not repeated views, define uncertainty."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from scripts.check_covert_controls import baseline
from sprint.analyze import bootstrap_ci
from sprint.covert import QUESTIONS, VIEWS


def events(path):
    """Read an interrupted journal without mistaking its unfinished final line for a result."""
    if not path.exists():
        return []
    lines = path.read_text().splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines.pop()
    return [json.loads(line) for line in lines]


def judgments(run_dirs):
    rows = []
    for directory in map(Path, run_dirs):
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        setup = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:12]
        for case in manifest["cases"]:
            path = directory / f"case_{case['id']}.jsonl"
            status, reply, reason = "not_started", {}, ""
            for event in events(path):
                if event["event"] == "header":
                    status = "incomplete"
                elif event["event"] == "complete":
                    status, reply = "answered", event["result"]["reply"]
                elif event["event"] == "failed":
                    status, reason = "failed", event["reason"]
            calls = events(path.with_name(path.stem + ".calls.jsonl"))
            stats = {"attempts": len(calls), "providers_served": ",".join(sorted(
                {a["provider_served"] for a in calls if a.get("provider_served")}))}
            for metric in ("cost", "prompt_tokens", "completion_tokens"):
                values = [a[metric] for a in calls if a.get(metric) is not None]
                stats[metric] = sum(values) if values else None
            common = {"run": str(directory.resolve()), "setup": setup,
                      "model": manifest["model"]["model"], "case_id": case["id"],
                      **{k: case[k] for k in ("block", "scheme", "kind", "view")},
                      "status": status, "reason": reason, **stats}
            for question in QUESTIONS:
                expected = case["expected"][question]
                predicted = reply.get(question)
                rows.append(common | {"question": question, "expected": expected,
                    "hidden_truth": case["truth"][question], "predicted": predicted,
                    "answered": status == "answered", "correct": predicted == expected,
                    **{name + "_correct": baseline(name, case["record"])[question] == expected
                       for name in ("evidence_level_only", "parallel_rise", "always_abstain", "always_accuse")}})
    return pd.DataFrame(rows)


def summary(table, by_kind=False):
    if table.empty:
        return pd.DataFrame()
    keys = ["run", "setup", "model", "scheme", "view", "question"] + (["kind"] if by_kind else [])
    rows = []
    for key, group in table.groupby(keys, dropna=False):
        ambiguous = group.expected == "insufficient_evidence"
        answered = group.answered
        metrics = {
            "answer_coverage": (pd.Series(True, index=group.index), answered),
            "correct_per_planned_judgment": (pd.Series(True, index=group.index), group.correct),
            "unsupported_positive": (ambiguous & answered, group.predicted == "supported"),
            "unsupported_negative": (ambiguous & answered, group.predicted == "contradicted"),
            "resolved_accuracy": (~ambiguous & answered, group.correct),
            "unnecessary_abstention": (~ambiguous & answered, group.predicted == "insufficient_evidence"),
            "false_positive": ((group.expected == "contradicted") & answered, group.predicted == "supported"),
        }
        for metric, (denominator, success) in metrics.items():
            subset = group.loc[denominator, ["block"]].assign(value=success[denominator].astype(int))
            block_rates = subset.groupby("block").value.mean()
            lo, hi = bootstrap_ci(block_rates)
            rows.append(dict(zip(keys, key)) | {"metric": metric, "numerator": int(subset.value.sum()),
                "denominator": len(subset), "blocks": len(block_rates),
                "mean_block_rate": block_rates.mean(), "ci_low": lo, "ci_high": hi,
                "interval": "descriptive block bootstrap; degenerate samples do not establish certainty"})
    return pd.DataFrame(rows)


def launch_checks(directory):
    table = judgments([directory])
    attempts = [a for path in Path(directory).glob("*.calls.jsonl") for a in events(path)]
    successful = [a for a in attempts if a["outcome"] == "ok"]
    return {
        "all canary cases answered": not table.empty and bool(table.answered.all()),
        "no fatal or refused calls": not any(a["outcome"] in ("fatal", "declined", "error") for a in attempts),
        "pinned provider recorded": bool(successful) and all(
            (a["provider_served"] or "").lower() == a["provider_requested"].lower() for a in successful),
        "usage and cost recorded": bool(successful) and all(
            all(a.get(k) is not None for k in ("cost", "prompt_tokens", "completion_tokens"))
            for a in successful),
    }  # Correctness is an outcome, never a reason to exclude a model or retry a judgment.


def export_tables(run_dirs, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    table = judgments(run_dirs)
    table.to_csv(out / "judgments.csv", index=False)
    if table.empty:
        return table
    scores = summary(table)
    scores.to_csv(out / "summary.csv", index=False)
    summary(table, by_kind=True).to_csv(out / "case_type_scores.csv", index=False)
    cases = table.drop_duplicates(["run", "setup", "case_id"]).drop(
        columns=["question", "expected", "hidden_truth", "predicted", "correct"])
    cases.to_csv(out / "cases.csv", index=False)  # token/cost totals counted once, not per question
    cases.groupby(["run", "setup", "model", "status"]).size().rename("cases").to_csv(out / "accounting.csv")
    paired_comparisons(table).to_csv(out / "paired_comparisons.csv", index=False)
    return table


def paired_comparisons(table):
    """Paired accuracy differences, averaged within blocks before intervals; failures score incorrect."""
    rows = []
    keys = ["run", "setup", "model", "scheme", "question"]
    for key, group in table.groupby(keys):
        wide = group.pivot(index=["block", "kind"], columns="view", values="correct").astype(float)
        contrasts = {f"{later}_minus_{earlier}": (wide[later] - wide[earlier]).groupby("block").mean()
                     for earlier, later in (("public", "decoder"), ("decoder", "complete"))}
        for view, subset in group.groupby("view"):
            for name in ("evidence_level_only", "parallel_rise", "always_abstain", "always_accuse"):
                differences = subset.correct.astype(float) - subset[name + "_correct"].astype(float)
                contrasts[f"{view}_minus_{name}"] = differences.groupby(subset.block).mean()
        for contrast, values in contrasts.items():
            lo, hi = bootstrap_ci(values)
            rows.append(dict(zip(keys, key)) | {"contrast": contrast, "mean_accuracy_difference": values.mean(),
                        "blocks": len(values), "ci_low": lo, "ci_high": hi, "interval": "descriptive, unadjusted"})
    return pd.DataFrame(rows)


def plot(table, out):
    if table.empty:
        return
    for (run, setup), group in table.groupby(["run", "setup"]):
        scores = summary(group)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
        for ax, question in zip(axes, QUESTIONS):
            subset = scores[(scores.question == question) & (scores.metric == "correct_per_planned_judgment")]
            values = subset.set_index("view").reindex(VIEWS).mean_block_rate
            ax.bar(range(len(VIEWS)), values)
            ax.set_xticks(range(len(VIEWS)), ["Prices", "+ decoder", "+ full table", "+ partial table"])
            ax.tick_params(axis="x", rotation=25)
            ax.set(title=question, ylim=(0, 1), ylabel="Correct / planned judgments")
        fig.suptitle(group.model.iloc[0] + " — " + group.scheme.iloc[0])
        fig.tight_layout()
        fig.savefig(Path(out) / f"accuracy_{Path(run).name}_{setup}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plot(export_tables(args.runs, args.out), args.out)
