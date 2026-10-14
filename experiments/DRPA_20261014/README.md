# DRPA: Paired Official-Behavior Experiments

Only this directory and the root `handoff.md` are written. Original model, data,
RFE overwrite, AMRS return, label encoding, VAE sampling, dropout, graph losses,
and the existing PPU compatibility path are preserved.

```bash
source ~/venvs/mm/bin/activate
PYTHONDONTWRITEBYTECODE=1 python experiments/DRPA_20261014/launch.py
```

The launcher audits inputs, tests correctness, sequentially replays the three E0
seeds, checks seed 911 against history, then schedules 210 paired variant folds
on physical PPU 0/1. Restarting the same command verifies and skips complete folds.
No CPU training, DDP, test tuning, root source changes, or GitHub push.
The strict raw historical check is never overwritten or silently relaxed. When
mean ACC/AUC differences satisfy the 0.5-point limit but diagnosed fold-level
drift triggers the stricter check, `review_history.py` records native-repeat
evidence, fold differences, and result hashes. An explicit reviewed resume uses
`launch.py --diagnosed-history-review --workers-per-gpu 4`; all 30 actual-anchor numerical checks
are still required before variants. This is not exact historical replication.
Host thread counts are recorded and fixed to 1 for each worker. Trusted original
NumPy caches use `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` to retain torch 2.0 loading
behavior on torch 2.9; no serialized input content is changed.

E1-E7 use mixed round-robin queues, with up to four independent processes on
each PPU (eight total). Each method assigns 15 folds to each card, 105 variant
tasks per card in total. Completed folds keep their original physical card;
all paired variants of a seed/fold share that card. The immutable assignment
table is `artifacts/variant_plan.json`. `scheduling_policy.json` records the
concurrency and `queue.json` exposes live slots, PIDs, counts, and free memory.
A new worker needs a 14 GiB budget with 8 GiB held back per card; the guard
also reserves expected allocation growth for warming workers and respects
unrelated jobs. Only scheduling changes: no DDP, CPU fallback, repeated VAE
pretraining, altered RNG/backend flags, or training-hyperparameter changes.
Memory telemetry is queried only when a pending task can fill a free slot.
A query timeout or temporary telemetry error pauses new dispatch only; active
training and completion checks continue. Launches never use stale free-memory
data. The initial fatal monitoring timeout and its seven interrupted workers
are retained in the run log and `artifacts/scheduler_timeout_20261014T125655.json`;
incomplete folds restart from their unchanged anchors, not from a chosen score.
The current launcher runs in an independent background session. Its PID and
command are in `artifacts/launcher_service.json`; full launcher stdout/stderr
is in `launcher_service.log`. It continues through group summaries and final
analysis without waiting for another instruction. The launcher and per-fold
file locks prevent a second launcher or duplicate worker from taking ownership.

The E1 summary initially stopped on a NumPy `int64` improvement count that the
JSON encoder rejected. Analysis now converts that count to a Python `int`, with
three CPU-only regression tests checked before dispatch. Metric values and the
training-code hash are unchanged. All 130 verified completed folds were retained;
interrupted folds restart from the original anchors. The original stop state and
verified per-group counts are in `artifacts/analysis_serialization_recovery.json`.

Pre-registered constants are in `config.json`. Site reliability is the geometric
mean of `sigmoid(2 * class_margin)` from both modalities. Each prototype excludes
the query site. Missing other-site classes (<2 subjects) or zero-norm prototypes
receive zero site reliability. Gate weights sum to the training sample count;
zero total support skips alignment. InfoNCE is divided by `log(N_train)`.
E7 permutations use private RNG, within class/site, with a same-class singleton
fallback. New heads also initialize with private RNG.

Training labels remain ASD=0 / HC=1. `official_metrics` retain the original
positive-class-1 convention for historical comparisons. Main research metrics
use ASD as positive, without relabeling training or selecting a new checkpoint.
Both conventions and continuous ASD probabilities are saved.

Preflight tests observed the same small PPU reduction jitter even when repeating
the official model without hooks (near-zero gradients around 4e-6). Equivalence
checks use forward rtol=1e-5/atol=1e-6, gradients rtol=1e-4/atol=1e-5, and Adam
step rtol=1e-5/atol=1e-5; measured errors and an official-repeat control are saved.
These are numerical test tolerances, not training hyperparameter changes.

Actual 871-subject default-backend repeats also exposed a larger, discontinuous
error: VAE latents, RP graph, and fused input graph were bitwise equal, but
GTUNet/TopK choices could differ, producing about 0.005 logit error. The original
model against itself showed this too. `torch.use_deterministic_algorithms(True)`
with `CUBLAS_WORKSPACE_CONFIG=:4096:8` made the source-repeat and unused-head
forward, parameter gradients, and Adam step bitwise equal in the diagnostic.
The 30-anchor numerical-equivalence tests therefore use that same controlled
backend for BOTH source and candidate. It is test-only: formal E0-E7 training
retains the original backend settings. The test is not a claim of bitwise
production repeatability, and native instability limits small-gain attribution.
Evidence: `artifacts/anchor_equivalence_diagnosis*.json`.
PyTorch's [deterministic-algorithms documentation](https://docs.pytorch.org/docs/2.9/generated/torch.use_deterministic_algorithms.html)
describes the distinction from the cuDNN-only deterministic flag.

Three full fold-0 controls call the original `train()` and `evaluate()` functions
verbatim, without hooks or extra backward diagnostics. With identical core/VAE
initialization, backend flags, and pre/post-test RNG fingerprints, ACC was
73.86%, 68.18%, and 68.18% (physical cards 0, 1, and 0). Thus even the same
physical card did not repeat bitwise. The canonical replay matches the first
original-function replay on fold 0. These controls are diagnostic only, not
additional candidates from which a high-scoring baseline is selected. Verified
evidence is in `artifacts/baseline_diagnostics/verification.json`. The exact
responsible kernel has not been isolated. All historical differences remain
reported; fresh paired anchors do not imply exact historical replication.

An earlier instrumented replay was archived in `artifacts/diagnostic_replay_v1`
before any variant was trained. The canonical replay uses the original NumPy
indexing and only one backward through the original model; backend flags are
also captured and restored. Neither version changes RFE, graphs, labels,
dropout, checkpoint selection, or any registered research hyperparameter.

Each anchor contains initial core weights, the unregistered RP weights, VAE
pretraining state, pre-initialization/training-start/end RNG, and exact splits.
All variants share these states. Hooks observe the two 16-D outputs and do not
replace the official attention or classifier. Checkpoints are selected only by
the original validation CE + graph loss, with its original random evaluation.

`outputs/E*/seed_*/fold_*/` holds best checkpoint, OOF rows, epoch diagnostics,
fold metrics, and completion status. `analyze.py` verifies OOF coverage, builds
paired summaries and the final Chinese report, and atomically updates handoff
after each complete group. Statistics are descriptive repeated-CV comparisons;
folds are not treated as independent clinical cohorts.

The preserved cross-fold RFE leakage limits clinical generalization claims.
These experiments test mechanisms under the official implementation protocol,
not a leakage-free or leave-site-out benchmark. No architectural private/shared
disentanglement is claimed from projection alignment alone.

## GitHub Archive

The GitHub copy contains experiment code, configuration, training/audit logs,
per-fold metrics, continuous OOF predictions, diagnostics, and final summaries.
Model checkpoints, RNG/initialization anchors, tensor caches, binary embedding
arrays, downloaded PDFs, and runtime locks/temporary files are excluded by this
directory's `.gitignore`. They remain unchanged in the original PPU workspace.
Original input data is also excluded by the repository's root ignore rules.
This lightweight archive supports inspecting the completed results; resuming
the existing paired training requires restoring the omitted binary artifacts
and original data. A saved `complete` status alone is not a usable checkpoint.

Sources: [paper v3](https://arxiv.org/pdf/2406.14455v3),
[official source](https://github.com/NZWANG/MM-GTUNets/tree/bb9351b931003535503694ad2f777b6cee1c151f).
