# Experiment 8: LCRA

Independent, historical-behavior paired experiment. No root model, utility,
dataset, previous experiment, checkpoint, RFE or VAE training is modified.
All generated files remain here. Only the final accepted handoff is appended
atomically to the root `handoff.md`. No automatic GitHub push.

L1 uses image-only local graph features; all three phenotype-dependent MLP
channels (K, image*K, abs(image-K)) are zero. L2 uses true age/sex kernels.
L3 adds the registered soft true-vs-within-site-null JS margin. L4 uses
partition/site-restricted joint age/sex permutations in the new branch only.
All groups have the same projection, zero-initialized MLP, optimizer and RNG.

Fixed train-fitted image Top64 candidates precede original dual GTUNet
propagation. A bounded, centered continuous edge correction is inserted only
after native `fused_sim * rp_graph`. Native VAE, RP, edge dropout, GTUNet,
ACMGL, single classifier call, BN and graph regularization are preserved.
The native transductive 871-node graph and historical cross-fold RFE leakage
remain limitations. This is not leakage-free clinical generalization.

## Execution

```bash
source ~/venvs/mm/bin/activate
PYTHONDONTWRITEBYTECODE=1 python -u -B experiments/LCRA_20261018/launch.py
```

The launcher verifies source/input hashes, performs five exact three-step
diagnostics, then schedules five anchors x L1-L4 x 80 development epochs.
It does not compute any new outer-test performance during development.
Only a registered Go decision admits 120 formal tasks, restarted from the
historical training-start states. The final report is explicit when stopped.
Completed checksum-valid work is skipped. Failed incomplete work restarts
from its original anchor, never from a score-selected state.

CPU is used only for metadata, fixed candidate preprocessing and synthetic
unit tests. Actual neural optimization is PPU-only, with one CPU thread per
worker. Two workers per card initially; resource admission can raise to four
without interrupting healthy processes. Fixed paired fold device ownership.

State/diagnostics use `summary.json`, `audit.json`, `run.log` and necessary
single per-task artifacts in `runs/`. No per-epoch model copies or plots.
Baseline R0 reuses historical C0. Per-seed ten-fold means precede three-seed
sample SD. Official single stochastic testing is not replaced by averaging.

Generic serialization, historical loading, RNG and parity utilities are
adapted as independent code from experiment7 after direct read-only audit;
no old runner is imported or executed. LCRA, development gating, scheduling
and reporting are local implementations with frozen source fingerprints.

Sources: [paper v3](https://arxiv.org/html/2406.14455v3),
[official repository](https://github.com/NZWANG/MM-GTUNets),
[current repository](https://github.com/tangaii/MM).
