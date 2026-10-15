# DRPA Mechanism Diagnosis

Post-hoc diagnosis of the 240 completed fits. No training, model/source fixes,
test tuning, checkpoint writes, or GitHub push.

- Local and remote reference commit: `a6471c54477d663b6321c706b21173e1db979b7e`.
- Formal training SHA256: `9186fe8b76142e6ef114d850409cb7d8e9ac5f2691d8e132647c389ee60e5572`.
- Read-only inputs: `../DRPA_20261014/outputs/` and `../DRPA_20261014/artifacts/`.
- D1: actual alignment spaces, subset-specific retrieval, pairing controls.
- D2: logged training gates and separately reconstructed post-hoc gates.
- D3: logged gradients, saved representation statistics, 30 frozen-checkpoint
  gradient diagnostics (E1/E3/E4/E5/E6, seeds 911-913, folds 0/5).
- D4: 12 frozen E0/E6 checkpoints, three repeats per condition, RNG and backend
  controls, plus isolated VAE/dropout interventions. Six checkpoints per PPU.

```bash
source ~/venvs/mm/bin/activate
PYTHONDONTWRITEBYTECODE=1 python -B experiments/DRPA_Mechanism_20261015/diagnose.py --self-test
PYTHONDONTWRITEBYTECODE=1 python -B experiments/DRPA_Mechanism_20261015/diagnose.py
```

CPU array statistics precede one read-only diagnostic worker per PPU. The parent
alone writes atomic `summary.json`, `report.md`, and the mechanism section of
root `handoff.md`. Resuming checks input/code hashes and skips completed work.
`replay.py` communicates through pipes and writes no files. Static reward/penalty
input tables are restored in memory from saved splits/phenotypes for the required
forward; learned AMRS parameters are read from checkpoints, never refitted.

`--render-only` refreshes the report without GPU execution. Report-only revisions
retain executed code hashes; resume rejects changes to numerical-analysis AST or
GPU replay code. A full resume also revalidates the historical input manifest.

`summary.json` holds per-fit numbers, provenance, missing-statistic notices,
progress and checks. `report.md` is the Chinese research report. No embeddings,
predictions or model tensors are duplicated. Frozen gradient diagnostics are
not historical training gradients; deterministic execution is diagnostic only.
