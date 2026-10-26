# Which Rival Makes a Price Cut Unprofitable?

Two experiments with LLM pricing agents:

1. **Which rivals deter a price cut.** Three LLM sellers. A prescribed price cut is compared with no cut while neither, either, or both rivals respond.
2. **Pricing after retaliation is removed.** One LLM seller faces a scripted rival whose retaliation is kept or removed, with the seller's context kept or reset.

[Report (PDF)](paper/report.pdf) · [Tables and figures](REPORTING.md)

![Study design](figures/report1_pricing_design.png)

**Figure 1.** *Design of the two experiments. In Experiment 1, three LLM sellers play 150 rounds. If seller A's 10% price cut would pay against rivals that don't react, the game restarts from that point in eight 20-round branches: A cuts or keeps its price while neither, one or both rivals respond. A rival that doesn't respond repeats its earlier prices. In Experiment 2, one LLM seller plays 100 rounds against a scripted rival that punishes undercutting. At calm points, the game restarts with the punishment kept or removed and the seller's memory kept or reset. Branches from the same game are compared with each other, not counted as independent samples.*

## Results

<!-- RESULTS:START -->
| Model | Eligible games / completed | Price change after punishment removed | 98.75% interval | Paired games |
|---|---:|---:|---:|---:|
| DeepSeek | 4/4 | +0.0818 | [0.0783, 0.0854] | 8 |
| Qwen | 1/4 | +0.0600 | [0.0151, 0.1049] | 7 |
<!-- RESULTS:END -->

- In four of the five eligible games, one responding rival was enough to make the cut unprofitable. In no game did the cut become unprofitable only when both rivals responded.
- Three Qwen games were not eligible: the cut would not have gained at least 1% of profit even with no rival responding. Their gains are undefined, not zero.
- When the rival stopped punishing, both models raised their prices. This does not show that the models had learned to fear punishment or noticed that it had stopped.

![Results](figures/results_summary.png)

**Figure 2.** *Left: the gain from seller A's price cut in each of the five eligible games (four DeepSeek, one Qwen), by which rivals were allowed to respond. Lines join the conditions within one game, and points below zero mean the cut lost money. One responding rival was enough to make the cut unprofitable in four games; in DeepSeek game 4 the cut still paid when both rivals responded. Right: the rise in the seller's mean price when the rival's punishment was removed, with memory kept, shown with 98.75% intervals (8 DeepSeek and 7 Qwen paired games).*

Per-game and summary numbers are in [results/exports](results/exports/) ([what these files are](results/README.md)). Rebuild the table and figure with `python -m scripts.build_readme_results` (no API calls).

## Install and test

Requires Python 3.12 or newer.

```bash
git clone https://github.com/dshi2/llm-pricing-deterrence-audit.git
cd llm-pricing-deterrence-audit
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -c requirements-tested.txt -e '.[dev]'
bash scripts/get_starter.sh
python -m pip install -c requirements-tested.txt ./starter/util ./starter/algorithmic-pricing-llms
python -m pytest -q
```

## Rebuild from raw runs

```bash
python -m scripts.run_saved_report deterrence data/experiment1/qwen38 data/experiment1/deepseek41 --out reports/experiment1 --planned-seeds 1 2 3 4
python -m scripts.run_saved_report retaliation data/experiment2/qwen38 data/experiment2/deepseek41 --out reports/experiment2
python -m scripts.audit_static_best_responses data/experiment2/deepseek41 --out reports/deepseek-best-response
python -m scripts.check_retaliation_controls --out reports/experiment2-controls
python -m scripts.build_paper_diagrams --out reports/designs
```

Each `data/...` path is one model's saved run folder with its `game_*.jsonl` journals. The raw journals are not included in this repository, so these commands need a copy of them. They make no model calls.

## Layout

```text
paper/      Report, protocols and scripted-control outputs
notebooks/  Colab launchers used for collection (paid model calls)
sprint/     Market, game engine, experiments and analysis
scripts/    Controls, report builders and figures
configs/    Model and experiment settings
results/    Bundled numerical exports with hashes
tests/      Tests
```

## What is new and when it was done

This project was carried out entirely within the Apart AI Collusion Research Sprint (23–26 October 2026). No code or data predates the sprint; the first commit is from 25 October 2026.

It builds on the sprint starter kit ([poseidon-research/collusion_hackathon](https://github.com/poseidon-research/collusion_hackathon), commit `d585a7b`), which packages the pricing harness of Romero, Sauer and Peist:

- **Used unchanged:** the market functions and prompt texts. They are downloaded at install time, not copied here.
- **Adapted:** `sprint/prompt.py` rebuilds the starter's prompt construction and reuses its response-format text; a test checks that the output matches the starter exactly.
- **New:** everything else, including the experiments, game engine, model client, analysis, controls and figures.

The version submitted with the report is tagged [`pre-maintenance-2026-10-26`](https://github.com/dshi2/llm-pricing-deterrence-audit/tree/pre-maintenance-2026-10-26), committed on 26 October at 11:19 UTC, before the 11:59 UTC deadline. Later commits only tidy documentation and code layout; they add no experiments or results.

## Source

Author: **Drishti Sharma, Independent**. `provenance/source.json` records the sprint source commit and the hash of every file in this release. The companion paper's code is in [covert-channel-evidence-audit](https://github.com/dshi2/covert-channel-evidence-audit).

The market model and prompts come from [the Apart starter](https://github.com/poseidon-research/collusion_hackathon), based on [algorithmic_pricing_llms](https://github.com/luciasauer/algorithmic_pricing_llms); it is fetched, not bundled, and keeps its own license. The code in this repository is released under the [MIT License](LICENSE).

Each run file records Experiment 1 as `main` and Experiment 2 as `replacement`. The collection notebook saved Experiment 2 runs to `runs/fallback1_v2`, one folder per model named `fallback1_<model>`. These names are kept so the code matches the saved runs.
