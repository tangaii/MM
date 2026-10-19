# Experiment 11: Global Brain Network Semantic Alignment Pilot

Only C0 and S1, seed 911, ten historical folds each. Root source and historical
artifacts remain read-only. Historical cross-fold RFE leakage, transductive
population graphs, VAE sampling, eval functional Dropout, RP projection and
unregistered weights are preserved. These are exploratory protocol-matched
results, not leakage-free clinical estimates.

Each subject's original AAL116 correlation matrix produces deterministic
functional-connectivity and weighted graph statistics. A positive top-10%
graph has exactly 667 undirected edges; lengths are inverse correlations,
modularity uses deterministic greedy communities, and weighted clustering uses
NetworkX. Captions use one-based matrix ROI indices, with no invented atlas
names, diagnoses or demographics. Fold reference means/SD use training only.

Frozen `sentence-transformers/all-MiniLM-L6-v2` uses pinned revision in config,
official masked mean pooling and normalization, implemented through existing
Transformers without installing or changing the environment. Downloaded model
files and precalculated captions/embeddings live under `runs/`. Transformer
inference happens once, never inside neural training. These captions summarize
the same FC data, so this pilot cannot establish an independent clinical modality.

The single observational img_unet hook captures the native 16D output without
additional propagation. S1 adds a 16->128->384 head and symmetric InfoNCE over
training subjects only, divided by log(N_train). Weight ramps from zero at
epoch0 to .05 at epoch9; validation selects native CE+Graph only. Test needs
no text or head. Both groups start from identical historical core/BN/RNG/VAE.

Run with the unchanged mm environment:

```bash
source ~/venvs/mm/bin/activate
PYTHONDONTWRITEBYTECODE=1 python -B -u experiments/GBSA_Pilot_20261019/run.py all
```

The independent runner audits storage and protected SHA, prepares one cache,
checks three-step deterministic inert-hook gradients/Adam/BN/RNG and semantic
gradient routing on both PPUs, then schedules the 20 official tasks. Each fold
uses a unique lock and one internally SHA256-checked `runs/{group}_f{fold}.bin`.
Completed valid tasks are skipped; incomplete tasks restart at the historical
origin without overwriting completed results. Checkpoints include native RP
weights, BN, head and RNG. Formal stochastic single testing is not replaced
by averaging. Full epoch records stay inside each fold's single artifact.

`analyze.py` independently verifies probabilities, metrics, OOF, paired input
fingerprints and all protected files, writes Chinese report/summary, then
safely appends a unique experiment11 section to root handoff. No git push.

### ossfs Coordination Recovery

The initial coordinator encountered transient ENOENT while observing an active
checkpoint's atomic replacement. Its four workers completed normally, with
valid artifact SHA, and were not restarted. The frozen scientific code, data
cache and original preflights remain unchanged. The necessary `launch.py`
operational supervisor never reads active checkpoints and funnels worker
stdout through a single log writer. Its own SHA and the coordinator failure
are recorded separately in `audit.execution`; no model or protocol changes.
Use this recovery-safe queue for remaining jobs and completed-job resume:

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -u experiments/GBSA_Pilot_20261019/launch.py all
```

References: [fMRI-LM paper](https://arxiv.org/html/2511.21760v3),
[CVPR publication](https://openaccess.thecvf.com/content/CVPR2026/html/Wei_fMRI-LM_Towards_a_Universal_Foundation_Model_for_Language-Aligned_fMRI_Understanding_CVPR_2026_paper.html),
[fMRI-LM source](https://github.com/yuxiangwei0808/fMRI-LM),
[MM-GTUNets paper](https://arxiv.org/html/2406.14455v3).
