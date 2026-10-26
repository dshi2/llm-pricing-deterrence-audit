# Release validation

Checks run on this release:

- Fresh clone from GitHub, installed with the README steps (26 October 2026): 210 tests pass on Python 3.12 and 3.13, Ruff passes, and every script's `--help` runs. Python 3.11 is not supported because the pinned NumPy needs 3.12.
- GitHub Actions installs the repository and runs the tests, Ruff and the README results rebuild on every push.
- The README table and results figure rebuild from the hash-checked files in `results/exports/` without model calls.
- Notebook outputs are cleared and their Python cells compile. The notebooks have not been rerun in Colab for this release.
- A scan for credential patterns found none in the release files or the archived collection source.
- The report has not been rebuilt here from raw journals, because the raw journals are not included.

Before the evidence-audit code was moved to its own repository, the combined code passed 233 tests.
