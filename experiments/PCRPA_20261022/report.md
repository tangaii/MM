# Experiment 23: PC-RPA

Status: COMPLETE_WITH_DEVIATION. Candidate decision: NO-GO for the current first-innovation claim.
Formal tasks: 40/40; GPU0 20/20; GPU1 20/20; OOF rows: 3484/3484.

| Group | ACC % | SEN-ASD % | SPE-TD % | Probability AUC % |
|---|---:|---:|---:|---:|
| historical_D4 | 82.56 +/- 6.24 | 79.93 +/- 9.97 | 84.84 +/- 5.05 | 88.69 +/- 6.43 |
| historical_M3 | 83.13 +/- 6.59 | 79.92 +/- 9.71 | 85.92 +/- 6.02 | 88.60 +/- 6.34 |
| historical_P1 | 83.02 +/- 6.88 | 79.92 +/- 9.56 | 85.70 +/- 6.49 | 88.79 +/- 6.31 |
| R1 | 82.79 +/- 6.15 | 80.68 +/- 8.95 | 84.63 +/- 5.56 | 88.52 +/- 6.47 |
| R2 | 82.44 +/- 6.25 | 79.93 +/- 8.62 | 84.63 +/- 5.56 | 88.51 +/- 6.39 |
| R3 | 82.33 +/- 6.18 | 80.43 +/- 7.94 | 83.99 +/- 6.43 | 88.79 +/- 6.49 |
| R4 | 82.44 +/- 6.34 | 81.68 +/- 9.71 | 83.13 +/- 4.62 | 88.41 +/- 6.34 |

Values are 10-fold mean +/- sample SD (ddof=1). Historical D4, M3, and P1 metrics were recomputed from their saved read-only OOF artifacts.

## Paired Fold Differences

Entries are mean delta in percentage points +/- sample SD of the ten paired fold deltas; `wins` counts folds where the left method is higher. Per-fold values and leave-one-fold-out ranges are in `summary.json`.

| Contrast | ACC | SEN-ASD | SPE-TD | AUC |
|---|---:|---:|---:|---:|
| R2 - R1 | -0.345 +/- 0.776; 1/10 | -0.750 +/- 1.208; 0/10 | -0.005 +/- 1.426; 2/10 | -0.011 +/- 0.429; 5/10 |
| R3 - R2 | -0.114 +/- 1.478; 3/10 | +0.500 +/- 1.972; 2/10 | -0.634 +/- 3.338; 2/10 | +0.276 +/- 0.673; 4/10 |
| R3 - R4 | -0.114 +/- 1.911; 3/10 | -1.244 +/- 2.121; 1/10 | +0.865 +/- 3.070; 2/10 | +0.377 +/- 0.707; 7/10 |
| R3 - historical_M3 | -0.803 +/- 1.537; 2/10 | +0.512 +/- 3.678; 4/10 | -1.924 +/- 1.864; 0/10 | +0.186 +/- 0.641; 7/10 |
| R3 - historical_P1 | -0.690 +/- 1.454; 1/10 | +0.512 +/- 3.678; 3/10 | -1.711 +/- 2.417; 1/10 | +0.000 +/- 0.704; 5/10 |
| R3 - historical_D4 | -0.229 +/- 1.186; 2/10 | +0.506 +/- 3.285; 4/10 | -0.851 +/- 2.497; 2/10 | +0.096 +/- 0.561; 5/10 |

R3's AUC advantage over R4 remains positive after omitting any one fold (+0.212 to +0.460 pp), but SEN-ASD is 1.244 pp lower and ACC is 0.114 pp lower. Against P1, R3's AUC is practically unchanged; against D4, the leave-one-fold-out AUC delta changes direction (-0.065 to +0.165 pp). The increases against M3/D4 are small and do not outweigh the ACC/SPE-TD reductions.

## Mechanism Checks

The same-forward fold0 preflight produced identical D4 CE (0.694661), true phCE (0.691890), and true P1 pair weights for all groups. R1/R2/R3/R4 residual CE values were 0.693381/0.693353/0.693456/0.693238; the raw P1 pair CE was 0.693478. R3 residual prototypes differ from raw P1 prototypes (mean prototype L2 0.1631), but residual-vs-P1 CE differs by only -0.000022 and their embedding-gradient cosine is 0.99918. Residual-vs-D4 gradient cosine is -0.00798. The residual signal differs from D4 but is almost collinear with P1, so this preflight does not establish a distinct P1-independent training direction.

The nested reference cross-fit was checked on CPU with an independent Python loop oracle for the P1 triplet marginals. For each private fold, query common components used only the other two private folds; each reference patient's common component excluded itself from that shared reference pool. All such rows remained within outer-train. Query residuals retain image-embedding gradients; reference residuals, condition kernels, and b(v) are detached. The native model still saw the full transductive graph and its private fold assignment uses train labels, so this is not complete label-independence.

Training-only diagnostics averaged per outer fold and then across folds found:

| Quantity | Raw image representation | R3 common component | R3 residual |
|---|---:|---:|---:|
| Mean absolute age correlation per dimension | 0.0410 | 0.3952 | 0.0129 |
| Sex-conditioned mean gap (L2) | 0.4872 | 0.0776 | 0.4352 |
| Site between-variance fraction | 0.0434 | 0.2610 | 0.0377 |
| ASD/TD mean gap (L2) | 2.4593 | 0.0162 | 2.4577 |

These descriptive checks suggest age association decreased and simple ASD/TD mean separation remained, while sex association fell only modestly. They do not identify b(v) as a pure nuisance factor and do not establish clinical generalization. R1/R2 residuals show similar pattern; these are training-internal summaries, not extra performance metrics.

## Site And Negative Control

Across the 10 outer folds and three private query folds, the minimum same-site ASD/TD ESS had mean 10.45, SD 8.10, median 7.39, p10 3.29, p90 26.38, and maximum 34.40. Missing same-site class support occurred for 0.04% of query rows; beta_site was zero for 54.8% and averaged 0.127. R3 therefore used the age-sex fallback for over half of patients and had limited site-specific contribution. At the same fold0 preflight embedding, mean b differences were R1-R2 0.0203, R2-R3 0.0649, and R3-R4 0.2666.

The R4 fixed within-site joint permutation changed age-sex assignment for 98.95% of training rows on average, changed sex for 25.8%, and shifted age by 5.28 years on average. Its b(v) was materially different from R3 in preflight, but residual CE remained close. On OOF predictions, comparing R3 (candidate) to R4 (reference), R3 corrected 16 R4 errors and introduced 17 new errors: net -1 correct subject, ASD net -5, TD net +4. The AUC contrast favors R3, while the operating-point change is not uniformly beneficial.

R2 class balancing did not improve R1: AUC was -0.011 pp and ACC -0.345 pp. R3 adds a modest AUC increase over R2, but specificity and accuracy decreased; site ESS and beta support do not support a strong site-control claim.

## Fold And Checkpoint Sensitivity

| Fold | D4 AUC % | M3 AUC % | P1 AUC % | R1 AUC % | R2 AUC % | R3 AUC % | R4 AUC % |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 75.039 | 74.935 | 75.195 | 74.987 | 74.987 | 75.039 | 74.987 |
| 1 | 91.516 | 91.251 | 91.410 | 91.410 | 91.357 | 91.304 | 90.933 |
| 2 | 92.895 | 93.213 | 93.001 | 93.160 | 93.107 | 93.107 | 91.251 |
| 3 | 93.351 | 92.819 | 93.351 | 93.617 | 93.830 | 93.457 | 93.511 |
| 4 | 85.904 | 85.957 | 85.957 | 85.798 | 85.957 | 85.957 | 86.117 |
| 5 | 93.883 | 93.457 | 93.989 | 93.723 | 93.777 | 93.777 | 93.617 |
| 6 | 80.691 | 80.957 | 81.436 | 80.372 | 80.532 | 80.372 | 80.319 |
| 7 | 88.457 | 88.457 | 88.351 | 88.245 | 87.819 | 88.670 | 88.191 |
| 8 | 94.734 | 94.681 | 94.840 | 94.574 | 93.670 | 94.202 | 94.574 |
| 9 | 90.426 | 90.266 | 90.319 | 89.309 | 90.053 | 91.968 | 90.585 |

Fold 9 is the largest positive AUC delta for R3 against R2 and all three historical groups; fold 8 and fold 6 are negative against P1/M3/D4. R3-R2 AUC is +0.000 pp at fold2, -0.160 pp at fold6, +0.532 pp at fold8, and +1.915 pp at fold9. R3-R4 is +1.856 pp at fold2, +0.053 pp at fold6, -0.372 pp at fold8, and +1.383 pp at fold9. Leave-one-fold-out R3-R4 AUC stays positive, but the historical comparisons are small and not uniformly robust.

Mean best-checkpoint epoch was inherited from native validation selection; R3/R4 best epochs by fold are in `summary.json`. The paired checkpoint relative L2 distance averaged 0.0211 for R3 vs R2 and 0.0552 for R3 vs R4. The latter is concentrated in folds 2 (0.135), 8 (0.058), and 9 (0.345); most other fold checkpoints are close. Maximum CPU-vs-PPU softmax absolute difference across 40 artifacts was 1.19e-7, below the fixed 2e-6 audit tolerance. Formal probabilities remain the PPU probabilities.

## Execution And Audit

Training wall time was 940.9 seconds (15m 41s); cumulative worker time was 7060.1 seconds. Peak concurrency was four workers on each PPU. Failures and formal-task retries: 0. Each task used the historical train-start and RNG; each epoch executed one train and one validation forward; each task performed one grad-enabled formal test forward. All four groups used zero new trainable parameters and no KL.

The accepted D4 `prototype.py` was copied byte-for-byte (SHA256 `ded24a70e72fc1f990d644f88b4445ea0ace750f01629330a9d540853dd44feb`). Frozen training code SHA256: `bf1335fbf7d0b2fd457bbc8733eef5b97bb7918529746b90c00cc87c7251a15c`; manifest SHA256: `274dc79a4b43798cc5f0848581289480e4242ab99497e7669d8dfcc66339bacf`. Original-source, RFE, split, VAE, train-start, RNG, and read-only D4/M3/P1 SHA256 inventories are in `audit.json` and `manifest.json`. No history, core source, VAE, RFE, or split was modified; no Git commit/push was performed.

Deviation: `/data` returned EIO when creating the registered experiment directory and attempting writes, so all experiment files and outputs are under `/tmp/PCRPA_20261022`. The fold0 preflight assertion initially compared distinct private-fold matrices across folds; after fixing the assertion, the fixed code passed. Two one-forward preflight calls passed at different pre-freeze source hashes and one call failed after its one forward; the final frozen-code preflight passed with one forward. Three preflight forward calls occurred in total, although no formal task started before the final accepted preflight; this violates the requested single preflight call and is reported explicitly. The append-only `/data/handoff.md` attempt also failed with `Failed to write file`; the file remained byte-identical, and the full pending body is preserved in `handoff_pending.md`.

## Decision

NO-GO for claiming PC-RPA as the confirmed MICCAI 2027 first innovation from this run. R3 has positive but small AUC deltas over M3/D4 and a larger AUC delta over R4, yet it does not improve the full four-metric profile over P1/M3, has reduced ACC/SPE-TD, and its residual gradient is almost collinear with P1. Treat the current result as a mechanism pilot only. Do not add seeds or tune from this test set; any future confirmation requires a separately authorized, preregistered multi-seed study and independent-site evaluation under corrected feature-selection leakage controls.
