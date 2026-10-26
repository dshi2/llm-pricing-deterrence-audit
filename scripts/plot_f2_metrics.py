"""Plot every F2 metric from the canonical summary CSV; no hardcoded models or results."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scripts.report_figures import groups


def plot_metrics(source, out, label="Descriptive snapshot", selected_metrics=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(source)
    views = ["public", "decoder", "partial_execution", "complete"]
    for tag, identity, frame in groups(table, ("scheme", "question")):
        metrics = list(frame.metric.unique()) if selected_metrics is None else selected_metrics
        if not metrics or not set(metrics).issubset(set(frame.metric)):
            raise ValueError("Requested metrics must exist in summary.csv")
        cols = min(4, len(metrics))
        rows_count = (len(metrics) + cols - 1) // cols
        fig, axes = plt.subplots(rows_count, cols, figsize=(3.75 * cols, 3.5 * rows_count),
                                 sharey=True, squeeze=False)
        for ax, metric in zip(axes.flat, metrics):
            rows = frame.loc[frame.metric == metric].set_index("view").reindex(views)
            for i, row in enumerate(rows.itertuples()):
                if np.isfinite(row.mean_block_rate):
                    ax.plot(i, row.mean_block_rate, "o", color="#235b83")
                if np.isfinite(row.ci_low) and np.isfinite(row.ci_high):
                    ax.vlines(i, row.ci_low, row.ci_high, color="#235b83")
            ax.set(title=metric.replace("_", " "), ylim=(-.04, 1.04), xticks=range(4),
                   xticklabels=[f"{v.replace('_execution', '')}\nn={int(n) if pd.notna(n) else 'NA'}"
                                for v, n in zip(views, rows.denominator)])
            ax.set_xlim(-.4, 3.4)
            ax.tick_params(axis="x", rotation=25, labelsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=.15)
        for ax in list(axes.flat)[len(metrics):]:
            ax.axis("off")
        fig.suptitle(f"{label}\n{identity['model']} — {identity['scheme']} — {identity['question']}")
        fig.text(.5, .01, "Means and descriptive block-bootstrap intervals from summary.csv. n = applicable judgments.\n"
                 "Blank estimates are undefined, not zero. Conditional metrics exclude unanswered cases; coverage is separate.",
                 ha="center", fontsize=9)
        fig.tight_layout(rect=(0, .08, 1, .92))
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(out / f"f2_metrics_{tag}.{suffix}", dpi=200, bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="Descriptive snapshot")
    parser.add_argument("--metrics", nargs="+", help="Only these canonical metrics, in this order")
    args = parser.parse_args()
    plot_metrics(args.source, args.out, args.label, args.metrics)
