# GBSA-Diagnostic: Experiment 11 Frozen Mechanism Diagnosis

This is a frozen audit of Experiment 11, not a new neural-network experiment.
Historical files are read-only. New files stay in this directory; only the final
idempotent handoff append may write outside it. No automatic Git operations.

The predeclared rules are in `config.json`. A1 measures training-text geometry;
A2 uses 32 deterministic training subjects per fold and at most 3,600 controlled
sentences; A3 uses label-free inner KFold Ridge reconstruction. B uses historical
outer-training subjects only, inner StratifiedKFold, and fixed logistic probes.
Caption reference statistics and feature scalers are fitted on each inner
training split, never its holdout. B2 moves only text rows by private label-blind
permutations, independently within the inner training and holdout partitions.

C restores the already-trained semantic head and projects cached 871x16 image
features. It does not run GTUNet. Three ID-based retrieval settings each receive
64 fixed identity-permutation controls. Fold 6/8 auditing compares cached core,
BN, optimizer, RNG and native predictions, without another model forward.

Run with `/root/venvs/mm/bin/python -B diagnose.py all`, then
`env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /root/venvs/mm/bin/python -B analyze.py`.
The explicit environment also limits NumPy's BLAS initialization in the analyzer.
The second command validates all
results, writes the Chinese report, and finally appends `handoff.md` safely.
Validated stages and frozen text encodings are reusable; content hashes and an
exclusive coordinator lock prevent silent stale-cache reuse or duplicate work.
Both PPUs were tested for frozen text encoding, but their FP32 reference error
exceeded the unchanged numerical tolerance. All diagnostic text rows are therefore
encoded uniformly on CPU in two one-thread shards, before any diagnostic probe
results. Historical A1/C embeddings remain unchanged. Ridge, LR and retrieval also
use one-thread CPU arithmetic. All neural training and GTUNet forwards remain 0.
The execution-only backend revision and the failed preflight are retained in audit.

The historical `effective_rank` is entropy of normalized singular values. A1
also reports the requested covariance-eigenvalue entropy rank; these are not
interchangeable. Ten overlapping folds and 64 nulls are not independent cohorts.
FC-derived text cannot add new patient disease information by itself.
