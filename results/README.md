# Included results

The CSV files in `exports/` are the analysis outputs behind the report's tables, copied from the final analysis run. `export_manifest.json` records each file's hash before and after copying; the only change was replacing absolute run-folder paths with folder names.

`python -m scripts.build_readme_results` checks those hashes and rebuilds the README table and results figure from these files, without model calls. They are processed outputs, not raw run data; the raw journals are not included in this repository.
