"""Rebuild tables and figures from saved journals, without API calls."""

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")


def hashes(runs):
    return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for run in runs for path in sorted(run.iterdir())
            if path.is_file() and (path.suffix == ".jsonl" or path.name == "manifest.json")}


def environment(root):
    """Commit, uncommitted-change flag and library versions; None where git is unavailable."""
    def git(*args):
        try:
            return subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = git("status", "--porcelain")
    return {"git_commit": git("rev-parse", "HEAD"),
            "git_uncommitted_changes": None if status is None else bool(status),
            "python": platform.python_version(),
            "packages": {name: metadata.version(name)
                         for name in ("numpy", "pandas", "scipy", "matplotlib")}}


def run_report(kind, runs, out, stage="pilot", label="Descriptive snapshot", planned_seeds=None):
    from scripts import plot_paths, report_figures
    from sprint import covert_analyze, fallback1_analyze, study

    runs = [Path(p).resolve() for p in runs]
    out = Path(out).resolve()
    if len({p.name for p in runs}) != len(runs):
        raise ValueError("Run folder names must be unique; Main analysis identifies runs by folder name")
    for directory in runs:
        if not directory.is_dir():
            raise ValueError(f"Missing run folder: {directory}")
        if out == directory or out.is_relative_to(directory) or directory.is_relative_to(out):
            raise ValueError("Report output must be separate from input run folders")
        if kind == "f2" and not (directory / "manifest.json").is_file():
            raise ValueError(f"F2 manifest missing: {directory}")
        if kind != "f2" and not any(
                not p.name.endswith(".calls.jsonl") for p in directory.glob("game_*.jsonl")):
            raise ValueError(f"No game journals: {directory}")
    out.mkdir(parents=True, exist_ok=False)
    before = hashes(runs)
    manifest = {"experiment": kind, "stage": stage if kind == "main" else None,
                "label": label, "planned_seeds": planned_seeds,
                "created_utc": datetime.now(UTC).isoformat(),
                "inputs": before, "status": "started"}
    # Hash the implementation too, including staged edits absent from a git commit.
    root = Path(study.__file__).resolve().parents[1]
    manifest["analysis_source"] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for subdir in ("sprint", "scripts") for p in sorted((root / subdir).glob("*.py"))}
    manifest["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest["environment"] = environment(root)
    manifest_path = out / "report_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    tables, figures = out / "tables", out / "figures"
    try:
        if kind == "main":
            study.export(runs, tables, stage, planned_seeds=planned_seeds)
        elif kind == "f1":
            fallback1_analyze.export_tables(runs, tables)
        else:
            covert_analyze.export_tables(runs, tables)
        report_figures.run(kind, tables, figures, label)
        if kind in ("main", "f1"):
            plot_paths.export(kind, runs, figures / "paths", label)
        else:
            from scripts.plot_f2_metrics import plot_metrics
            plot_metrics(tables / "summary.csv", figures / "metrics", label)
        if hashes(runs) != before:
            raise RuntimeError("Input files changed during analysis. Use a stable snapshot and rerun to a new output folder")
        manifest["status"] = "complete"
        manifest["outputs"] = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(out.rglob("*")) if p.is_file() and p != manifest_path}
    except Exception as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("main", "f1", "f2"))
    parser.add_argument("runs", nargs="+", type=Path, help="Model run folders, not the parent directory")
    parser.add_argument("--out", required=True, type=Path, help="New report folder; never overwrites")
    parser.add_argument("--stage", choices=("pilot", "pooled", "expansion"), default="pilot")
    parser.add_argument("--planned-seeds", nargs="+", type=int, help="Main accounting only; use actual saved plan")
    parser.add_argument("--label", default="Descriptive snapshot")
    args = parser.parse_args()
    if args.planned_seeds and args.kind != "main":
        parser.error("--planned-seeds applies only to Main")
    print(run_report(args.kind, args.runs, args.out, args.stage, args.label, args.planned_seeds))


if __name__ == "__main__":
    main()
