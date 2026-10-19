# Experiment 9: LCRA Dual-Path Attribution

Read-only diagnosis of 30 existing L3 checkpoints. No neural training,
optimizer, backward, VAE pretraining or RFE fitting. Native stochastic
inference runs only on PPU. All new outputs are local to this directory;
only the final accepted handoff is appended to the root.

```bash
source ~/venvs/mm/bin/activate
PYTHONDONTWRITEBYTECODE=1 python -B -u experiments/LCRA_Attribution_20261019/run.py all
```

`run.py` performs storage/hash/input audits, 5 fixed two-draw preflights,
then resumes those results into all 30 five-draw diagnostics. Physical
device ownership follows E8, 15 anchors per PPU, up to 2 workers per card.
Single per-fold files in `runs/` contain probabilities and compact graph,
propagation, pooling, relation score and state/RNG statistics. They never
contain copies of historical model weights. Completion skips hash-valid
work; task locks prevent duplicate workers and atomic writes preserve
completed draws. Only transient FileNotFoundError is retried on ossfs.

Frozen math is implemented here; E8's `lcra.py` is read only for explicit
deterministic equivalence, never its launcher/trainer. A0/A1 forwards
double as historical off/true replay. Native shuffled replay requires
150 additional forwards; it is distinct from calibrated A4. Formal
A0-A6 use 1050 forwards, deterministic math checks another 30.

A2-A6 are frozen artificial graph interventions, not method scores.
Stage B votes only with current training-anchor labels and fits no model.
Historical transductive graph and RFE leakage remain limitations.
Final Chinese interpretation is in `report.md`; complete hierarchical
statistics and fingerprints are in `summary.json` / `audit.json`.
No automatic GitHub push.
