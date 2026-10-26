# Release validation

- Fresh clone from GitHub, installed with the README steps (2026-10-26): 210 tests passed on Python 3.12 and 3.13; Ruff passed; every script's `--help` runs. Python 3.11 is not supported because the pinned NumPy needs 3.12.
- After removing evidence-audit code and renaming modules (2026-10-26): 210 tests passed and Ruff passed in the original research environment (Python 3.13.3).

Before that cleanup:

- Original research environment: 233 tests passed after removing unrelated launch notebooks and their notebook-specific tests.
- Fresh Python 3.13 environment: 233 tests passed; Ruff passed for scripts, sprint and tests with the repository configuration. The documented install includes the pinned starter packages and scikit-learn for upstream prompt-equivalence tests.
- The README result figure and table regenerate from hash-checked included exports without model calls. Both result figures were visually inspected.
- Notebook outputs are cleared and their ordinary Python cells compile. Live Colab execution of this release has not been claimed.
- Credential-pattern scan covers release files and archived collection source; raw journals and credentials are not included.
- Full raw-journal-to-paper reproduction remains pending the complete frozen raw dataset. This release provides export-to-table/figure reproduction, not a claim that missing raw data is bundled.

Remote CI results will appear in the repository Actions tab after upload.
