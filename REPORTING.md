# Reproducing the report

All commands run offline from saved journals. Each input is one model's run folder; each output must be a new folder. The raw run archive is not bundled yet.

| Report item | Command | Output |
|---|---|---|
| Scripted controls | `sprint.checks.scripted_check`; `scripts.check_retaliation_controls` | Control results; `controls.csv` |
| Experiment 1 game accounting | `scripts.run_saved_report deterrence …` | `tables/games.csv`, `tables/accounting.csv` |
| Per-game gains | Same | `tables/game_effects.csv` (G00, G10, G01, G11 and S−J) |
| Experiment 1 estimates and figure | Same | `tables/primary.csv`, `figures/deterrence_primary_{tag}.png` |
| Experiment 2 results and figure | `scripts.run_saved_report retaliation …` | `tables/replacement_summary.csv`, `replacement_contrasts.csv`, `figures/retaliation_mean_price_{tag}.png` |
| Best-response comparison | `scripts.audit_static_best_responses …` | `round_comparison.csv` |
| Appendix diagnostics | Both report commands | `tables/{run}/late_price_trends.csv`, `figures/paths/*.csv` |
| Figure 1: design | `scripts.build_paper_diagrams --out reports/designs` | `report1_pricing_design.png`, PDF, SVG |
| Figure 2: results | `scripts.build_readme_results` | `figures/results_summary.png`, SVG |

For a model with one eligible game, report the estimate without an interval.

`run_saved_report` writes `report_manifest.json` with input and output hashes, the git revision and library versions. Keep it with the tables. The original collection source is archived under `provenance/`.

`build_readme_results` checks the bundled export hashes and regenerates the README table and results figure.
