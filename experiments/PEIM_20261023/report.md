# PEIM Experiment 26

Status: `PARTIAL`; scientific decision: `NO_GO_STAGE_A_INCOMPLETE`.

The registered scientific evidence audit did not complete, so PEIM did not pass its pre-training integrity gate. Formal training is `0/40`; GPU0 and GPU1 formal tasks are both `0/20`. No historical model was retrained and no formal test metrics or OOF predictions were generated.

## What Was Verified

- Local MM-GTUNets source preserves the registered 16-dimensional image and phenotype GTUNet outputs, 871-node transductive graph, stochastic VAE sampling in eval, active GTUNet dropout in eval, RP special state, early return in graph construction, and attention-score gradient truncation.
- Masking non-training diagnosis labels produced a byte-identical fold0 graph, so the Stage A diagnostic could preserve the original graph while withholding held-out labels.
- The exact historical D4 prototype objective and gradient scaler were located in PCDCA; D4 best checkpoints and train-start anchors were fingerprinted. Historical D0/D4/M1/M3 and Experiment 25 were not retrained.
- Synthetic math checks passed for rank-limited projectors, ASD=0/TD=1 target ordering, train-only encoding, within-site/private-fold shuffling, the 5% modulation cap, and the interaction gradient finite difference. The asymmetric patient-transition test also passed.
- Experiment 25's saved report showed nonzero FiLM modulation without a stable correct-pairing AUC benefit: the reported F3-F4 AUC delta was concentrated in fold2 and nearly vanished after removing that fold.

## Stage A Stop

The read-only historical D4 fold0 best checkpoint was loaded and exactly one native eval forward was executed with the registered private RNG. While assembling its training-only probe and null summaries, the audit code raised `KeyError: 'rank'` in direction-stability aggregation before fold0 outputs were persisted. The lookup was corrected after the failure. Fold0 was not repeated; the remaining folds each received exactly one read-only forward with their registered private RNG.

Stage A is incomplete because fold0 has no persisted result, so the ten-fold integrity/evidence gate cannot be evaluated and no formal training is authorized. The nine completed folds do provide a partial diagnostic: mean `Delta_MSE` -0.018217, median -0.017515; positive folds 0/9; true values above the within-fold 32-null P95 in 1/9. All nine completed folds had nonpositive `Delta_MSE`; the interaction projector had rank 2 in all three private folds. The mean principal angle between private-reference interaction subspaces was 58.94 degrees (fold means 50.57–66.09); this reflects substantial direction differences, not stability. Their mean angle to the ordinary disease main-effect direction was 83.51 degrees (range 73.93–88.00). These are descriptive partial results, not a completed ten-fold test or formal GO criterion.

| Fold | Delta_MSE | Null mean | Null P95 | True > P95 | Interaction rank (Q0,Q1,Q2) | Private-reference subspace angle |
|---:|---:|---:|---:|:---:|:---:|---:|
| 1 | -0.017515 | -0.023106 | -0.014244 | no | 2,2,2 | 58.38 |
| 2 | -0.034608 | -0.024577 | -0.011455 | no | 2,2,2 | 63.13 |
| 3 | -0.018200 | -0.020382 | -0.014283 | no | 2,2,2 | 66.02 |
| 4 | -0.023598 | -0.016962 | -0.007759 | no | 2,2,2 | 66.09 |
| 5 | -0.013433 | -0.020700 | -0.010440 | no | 2,2,2 | 50.57 |
| 6 | -0.020171 | -0.021960 | -0.008626 | no | 2,2,2 | 53.39 |
| 7 | -0.012367 | -0.016991 | -0.008338 | no | 2,2,2 | 63.74 |
| 8 | -0.011746 | -0.014564 | -0.008964 | no | 2,2,2 | 54.33 |
| 9 | -0.012318 | -0.020870 | -0.013747 | yes | 2,2,2 | 54.82 |
| 0 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |

The registered E1-E4 method remains unchanged: 32 zero-initialized parameters `W_d`; cross-fitted rank-at-most-two ordinary disease (`P_main`) or phenotype-by-image interaction (`P_int`) projection; multiplicative `Z * (W_d d)` modulation; rho warmup to 0.1; per-patient 5% relative L2 cap; E3/E4 train-reference prototype-distance gate; E4 within-split/site phenotype derangement; native CE+Graph plus the original D4 image prototype CE only. These methods were not trained.

No GO claim is made. A complete training-internal evidence audit is a prerequisite before any formal fold training; the registered limit does not allow this failed fold to be silently rerun. An independent future rerun would need a new experiment registration with a pre-forward persistence boundary for diagnostic outputs.

The append-only `handoff.md` attempt returned a write error. Its original byte count and SHA256 were unchanged after verification. The proposed append is in `handoff_pending.md`; its SHA256 and the failed attempt are recorded in `audit.json`.

Paper/source review references: [MM-GTUNets](https://arxiv.org/html/2406.14455v3), [RTGMFF](https://arxiv.org/abs/2509.03214), and [FiLM](https://arxiv.org/abs/1709.07871). FiLM and low-rank bilinear interaction are established tools; PEIM's candidate contribution would be task-controlled interaction evidence and confidence-selective modulation, which this run did not validate.
