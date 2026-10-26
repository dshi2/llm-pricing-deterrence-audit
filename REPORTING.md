# Reproducing this paper

The README gives installation and complete command examples. Analysis and plotting run offline from saved records. Inputs must be stable, model-specific folders; output must be a fresh directory outside the input. The raw dataset is not yet bundled in this code release.

## Table and figure map

Manuscript numbering can change during layout; the content and source columns identify each artifact.

| Paper item | Command / source | Output / assembly |
|---|---|---|
| Table 1: model settings | Saved journal headers, launch configs and collection logs | Manually typeset settings; distinguish planned configurations from those contributing data. The report runner does not generate this formatted table. |
| Table 2: scripted controls | Experiment 1 `sprint.checks.scripted_check`; `scripts.check_fallback1_controls` | Experiment 1 control results and Experiment 2 `controls.csv`; see offline commands below. Select the reported contrasts without replacing controls with live-model results. |
| Table 3: Experiment 1 game accounting | `scripts.run_saved_report main …` | `tables/games.csv`, `tables/accounting.csv`; planned seeds include not-started games. Launch exclusions need the launch records separately. |
| Table 4: per-game gains | Same Experiment 1 command | `tables/game_effects.csv`: G00/G10/G01/G11 and paired S−J; calculate S=(G10+G01)/2, J=G11, M=min(G10,G01) for presentation if needed. |
| Experiment 1 aggregate estimates / gain figure | Same Experiment 1 command | `tables/primary.csv`, `figures/main_primary_{tag}.png` and `.svg`. These are the protocol intervals; `tables/descriptive.csv` and per-run `gains.png` are descriptive, not substitutes. For n=1 keep the point and label the interval not estimable. |
| §4.4 Experiment 2 primary/secondary results and removal-effect figure | `scripts.run_saved_report f1 …` | `tables/replacement_games.csv`, `replacement_contrasts.csv`, `replacement_summary.csv`; `figures/f1_mean_price_{tag}.png/.svg`. Plot observations/means; take intervals and complete-pair counts from the summary. Reconcile never-started seeds with the plan. |
| §4.4 best-response comparison | `scripts.audit_f1_best_responses …` | `round_comparison.csv`; summarise complete paths by seed/branch before comparing models. Use canonical Experiment 2 contrasts for D and its interval. |
| §4.5 / diagnostics | Experiment 1/Experiment 2 report commands | Experiment 1 `tables/games.csv` for recorded attempts/costs, `tables/{run}/late_price_trends.csv`, `figures/paths/late_profit_gain.csv`; Experiment 2 `replacement_games.csv` and `figures/paths/observation_opportunity.csv`. Missing logs do not establish zero spend. Trends do not prove convergence. |
| Figure 1: experimental design | `python -m scripts.build_paper_diagrams --out reports/designs` | `report1_pricing_design.png`, PDF and SVG; select this paper’s design. |

## Verification and provenance

`run_saved_report.py` records input/output hashes, git revision, dirty-tree status and Python/library versions in `report_manifest.json`. Keep the manifest alongside tables and plots. Original collection source is archived under `provenance/`; the reporting release has its own commit. Auxiliary audit and table scripts have their own manifests with different fields; the diagram generator does not emit a manifest.

The review notebook displays a short, explicit selection of figures. All generated diagnostic plots remain in the report folder; they need not all appear in the paper. A plot generated from scripted fixtures validates the pipeline, not the empirical result.

A complete paper reproduction additionally requires the frozen journals and reconciliation against the reported numbers. Passing unit tests does not establish that this final rebuild has occurred.

## README results

`python -m scripts.build_readme_results` verifies the bundled export hashes, regenerates `results/README_TABLE.md`, replaces the marked README table and creates `figures/results_summary.png` and SVG. No API calls are made. This is an export-to-figure check, separate from the raw-journal reporting commands.
