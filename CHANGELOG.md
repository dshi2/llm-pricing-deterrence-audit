# Change log

All changes below were made after the submission deadline (26 October 2026, 11:59 UTC). The submitted version is tagged `pre-maintenance-2026-10-26` (commit `bc0a0eb`). None of them changes experiment settings, saved results, statistical definitions or the report.

## 26 October 2026

- Added the final report PDF and its checksum.
- Renamed modules, scripts, configs and report commands to the experiment names used in the report (`deterrence`, `retaliation`). Names stored in the run data are unchanged.
- Removed the evidence-audit code, which has its own repository, including from the collection snapshots and the source file list. Removed an unused review notebook.
- Rewrote the README and the reproduction guide, matched the guide to the report's tables and figures, and added figure captions and a statement of what is new.
- Matched the citation title to the report, set the minimum Python version to 3.12, removed figure-drafting notes and added the MIT License.
- Made the tests runnable from any folder and gave `kill_and_resume.py` a `--help` page.
