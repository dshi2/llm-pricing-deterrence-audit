# Release validation

- Original research environment: 233 tests passed after removing unrelated launch notebooks and their notebook-specific tests.
- Fresh Python 3.13 environment: 233 tests passed; Ruff passed for scripts, sprint and tests with the repository configuration. The documented install includes the pinned starter packages and scikit-learn for upstream prompt-equivalence tests.
- The README result figure and table regenerate from hash-checked included exports without model calls. Both result figures were visually inspected.
- Notebook outputs are cleared and their ordinary Python cells compile. Live Colab execution of this release has not been claimed.
- Credential-pattern scan covers release files and archived collection source; raw journals and credentials are not included.
- Full raw-journal-to-paper reproduction remains pending the complete frozen raw dataset. This release provides export-to-table/figure reproduction, not a claim that missing raw data is bundled.

Remote CI results will appear in the repository Actions tab after upload.
