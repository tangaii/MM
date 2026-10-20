# D0/D4 Replay Drift Audit

This is an offline audit of the historical PCDCA_20261019 D0/D4 folds against SCPDA_20261020 E0/E1. It does not retrain any official fold.

Run the artifact/source audit and GPU-only diagnostic stages with `diagnose.py` and `config.json`. The required environment is `~/venvs/mm`; `gpu`, `hooks`, and `val-rng` refuse CPU model execution. `analyze` rebuilds `report.md`, `summary.json`, and `run.log` from the verified CSV/JSON outputs.

Primary outputs are `fold_drift.csv`, `trajectory_summary.csv`, `state_diff.csv`, `crossover_effects.csv`, `fixed_rng_summary.csv`, `validation_rng.csv`, and `report.md`. Historical checkpoints and experiment files are read-only.
