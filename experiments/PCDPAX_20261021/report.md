# Experiment 20: PCDPA-X

Status: **COMPLETE_WITH_DEVIATION**; formal tasks: **40/40**.

PCDPA-X preserves the historical image D4 conditional prototype CE and adds a phenotype-branch prototype CE. M2/M3/M4 add one stopped-gradient, image-teacher to phenotype-student KL term; M2 uses global phenotype prototypes, M3 real age-sex conditioned prototypes, and M4 the same-site joint-shuffled condition. No direct feature-vector matching or trainable parameters are added.

## Four formal metrics

|Group|Method|ACC (%)|SEN-ASD (%)|SPE-TD (%)|Probability AUC (%)|
|---|---|---:|---:|---:|---:|
|historical_D4|PCDCA seed911 read-only|82.56 +/- 6.24|79.93 +/- 9.97|84.84 +/- 5.05|88.69 +/- 6.43|
|M1|D4 + Phenotype Prototype CE|82.56 +/- 5.65|79.67 +/- 8.27|85.06 +/- 5.30|88.53 +/- 6.26|
|M2|M1 + Global-Prototype Semantic Alignment|82.67 +/- 6.28|79.67 +/- 9.45|85.27 +/- 5.62|88.57 +/- 6.37|
|M3|M1 + Phenotype-Conditioned Semantic Alignment|83.13 +/- 6.59|79.92 +/- 9.71|85.92 +/- 6.02|88.60 +/- 6.34|
|M4|M1 + Shuffled-Condition Semantic Alignment|82.44 +/- 6.06|80.17 +/- 9.88|84.41 +/- 4.59|88.66 +/- 6.45|

Metrics are calculated per fold and summarized over 10 folds with sample SD (ddof=1). Main results contain only ACC, SEN-ASD, SPE-TD, and probability AUC. Historical D4 is read-only and its mean matches the registered reference. New-group comparisons are paired by fold; historical comparisons are cross-run and remain subject to accelerator trajectory drift.

## Paired changes

|Comparison|Metric|Mean change (pp)|Positive folds|LOO range (pp)|
|---|---|---:|---:|---:|
|M1-historical_D4|ACC|+0.000 +/- 1.084|3/10|-0.128 to +0.255|
|M1-historical_D4|SEN-ASD|-0.256 +/- 2.479|3/10|-0.562 to +0.271|
|M1-historical_D4|SPE-TD|+0.213 +/- 0.673|1/10|+0.000 to +0.236|
|M1-historical_D4|probability_AUC|-0.159 +/- 0.426|3/10|-0.218 to -0.082|
|M2-M1|ACC|+0.115 +/- 0.848|3/10|+0.000 to +0.255|
|M2-M1|SEN-ASD|-0.000 +/- 2.041|1/10|-0.556 to +0.278|
|M2-M1|SPE-TD|+0.213 +/- 1.208|2/10|+0.000 to +0.473|
|M2-M1|probability_AUC|+0.043 +/- 0.236|5/10|-0.006 to +0.083|
|M3-M2|ACC|+0.460 +/- 0.969|4/10|+0.255 to +0.639|
|M3-M2|SEN-ASD|+0.250 +/- 1.419|2/10|+0.000 to +0.556|
|M3-M2|SPE-TD|+0.648 +/- 1.760|3/10|+0.247 to +0.956|
|M3-M2|probability_AUC|+0.026 +/- 0.291|7/10|-0.006 to +0.088|
|M3-M4|ACC|+0.690 +/- 1.110|5/10|+0.511 to +0.894|
|M3-M4|SEN-ASD|-0.250 +/- 2.189|2/10|-0.556 to +0.278|
|M3-M4|SPE-TD|+1.503 +/- 2.034|4/10|+1.187 to +1.670|
|M3-M4|probability_AUC|-0.059 +/- 0.327|6/10|-0.101 to +0.005|
|M3-M1|ACC|+0.575 +/- 1.242|4/10|+0.255 to +0.766|
|M3-M1|SEN-ASD|+0.250 +/- 2.189|2/10|-0.278 to +0.556|
|M3-M1|SPE-TD|+0.860 +/- 1.111|4/10|+0.714 to +0.956|
|M3-M1|probability_AUC|+0.069 +/- 0.389|4/10|+0.006 to +0.154|
|M3-historical_D4|ACC|+0.575 +/- 0.813|4/10|+0.383 to +0.639|
|M3-historical_D4|SEN-ASD|-0.006 +/- 2.033|2/10|-0.285 to +0.549|
|M3-historical_D4|SPE-TD|+1.073 +/- 1.512|4/10|+0.719 to +1.192|
|M3-historical_D4|probability_AUC|-0.090 +/- 0.272|3/10|-0.136 to -0.041|

## Mechanism checks

Mechanism values are training-only diagnostics and are not additional paper metrics. Prototype/gradient values are summarized over sampled epochs (first three epochs and every 25 epochs).

|Group|KL mean|KL first|KL last|Teacher-student agreement|Teacher train accuracy|Student train accuracy|Student entropy|Student prototype distance|Ph CE start|Ph CE end|Ph min ESS|Ph beta|Actual ph output-gradient ratio|M4 condition changed fraction|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|M1|NA|NA|NA|NA|NA|NA|NA|NA|0.6936|0.6942|119.3530|0.9502|0.0490|NA|
|M2|0.1057|0.0009|0.2607|0.5003|0.8083|0.5008|0.6929|0.3076|0.6936|0.6942|119.3689|0.9502|0.0491|NA|
|M3|0.1059|0.0012|0.2601|0.5015|0.8085|0.5018|0.6926|0.4249|0.6936|0.6942|119.3689|0.9502|0.0491|NA|
|M4|0.1061|0.0012|0.2610|0.5016|0.8082|0.5006|0.6926|0.4229|0.6936|0.6943|119.3689|0.9502|0.0491|0.9895|

### Interpretation

Decision: **MIXED**. M3 AUC minus historical D4 is -0.09018298323593688 pp; M3-M1/M2/M4 AUC changes are 0.06904740416506794, 0.02623474199589637, and -0.05912547099568899 pp. M1-historical D4 AUC change is -0.1592303874010048 pp. M2-M1 changes the objective by adding KL and is not a zero-information alignment test. M3-M4 isolates the condition used by the new KL branch only; the shared phenotype CE remains real-conditioned. KL reduction alone is not evidence of diagnostic value. Sampled KL first-to-last was 0.001->0.261 for M2, 0.001->0.260 for M3, and 0.001->0.261 for M4; it did not decrease. Teacher-student agreement and student train accuracy stayed near 0.50, student entropy near the two-class maximum, and the shared phenotype prototype CE near chance, despite nonzero prototype geometry. KL gradients were zero at z_img and img_unet and nonzero at z_ph and ph_unet; a small RP-attention gradient shows an indirect path through the shared graph. Do not select PCDPA-X as the first MICCAI 2027 innovation candidate. A future study should first justify richer non-imaging inputs, then preregister multi-seed and independent-site confirmation; prior multi-seed D4 evidence also does not establish a stable gain or a high-quality teacher.

### Execution deviations

- The first launcher invocation stopped before any task spawned because `ROOT` was missing from its imports; the import was fixed, mathematical tests and the one-forward preflight were rerun, and a new manifest was frozen.
- The initial validator rejected 8 completed artifacts because PPU and CPU float32 softmax differed by at most 1.1921e-7, just above its 1e-7 tolerance. Retry attempts stopped at artifact validation without retraining; ASD probabilities and four metrics were recomputed from the saved logits on CPU, then all 40 artifacts passed full validation.
- The experiment JSONL log had 1346 NUL padding bytes removed; all 381 remaining lines parse as JSON.

### Fold, checkpoint, and patient sensitivity

M3 AUC fold deltas versus M1/M2/M4 and historical D4, including leave-one-fold-out ranges, are in `summary.json`. Fold 2/6/9 checkpoint differences: `{"2": {"M3-M1": {"best_core_state_delta": {"absolute_l2": 0.4944992841389396, "relative_l2_to_right": 0.0026643102115714626}, "best_epoch_delta": 0, "best_val_loss_delta": 0.005997180938720703}, "M3-M2": {"best_core_state_delta": {"absolute_l2": 0.301017001785296, "relative_l2_to_right": 0.0016218758324588}, "best_epoch_delta": 0, "best_val_loss_delta": -0.003171265125274658}, "M3-M4": {"best_core_state_delta": {"absolute_l2": 0.3404726940059471, "relative_l2_to_right": 0.0018344168706638582}, "best_epoch_delta": 0, "best_val_loss_delta": 0.005246102809906006}, "M3-historical_D4": {"best_core_state_delta": {"absolute_l2": 0.35816419922211196, "relative_l2_to_right": 0.0019297541098759448}, "best_epoch_delta": 0, "best_val_loss_delta": 0.0010197162628173828}}, "6": {"M3-M1": {"best_core_state_delta": {"absolute_l2": 0.21738880399683946, "relative_l2_to_right": 0.0013137147954458894}, "best_epoch_delta": 0, "best_val_loss_delta": -0.013252675533294678}, "M3-M2": {"best_core_state_delta": {"absolute_l2": 0.13469124398042406, "relative_l2_to_right": 0.0008139620187933457}, "best_epoch_delta": 0, "best_val_loss_delta": -0.0003179311752319336}, "M3-M4": {"best_core_state_delta": {"absolute_l2": 0.12061900256821062, "relative_l2_to_right": 0.0007289225836177722}, "best_epoch_delta": 0, "best_val_loss_delta": -0.0008829236030578613}, "M3-historical_D4": {"best_core_state_delta": {"absolute_l2": 0.2376770932311439, "relative_l2_to_right": 0.0014363241401935485}, "best_epoch_delta": 0, "best_val_loss_delta": -7.998943328857422e-05}}, "9": {"M3-M1": {"best_core_state_delta": {"absolute_l2": 0.34408360706999036, "relative_l2_to_right": 0.001880702048076594}, "best_epoch_delta": 0, "best_val_loss_delta": -0.010472774505615234}, "M3-M2": {"best_core_state_delta": {"absolute_l2": 0.34621009600288505, "relative_l2_to_right": 0.0018923225492947205}, "best_epoch_delta": 0, "best_val_loss_delta": -0.0016043782234191895}, "M3-M4": {"best_core_state_delta": {"absolute_l2": 0.2845532508419293, "relative_l2_to_right": 0.0015553227109753021}, "best_epoch_delta": 0, "best_val_loss_delta": 8.189678192138672e-05}, "M3-historical_D4": {"best_core_state_delta": {"absolute_l2": 0.46951991292803885, "relative_l2_to_right": 0.0025662852091192967}, "best_epoch_delta": 0, "best_val_loss_delta": -0.0065580010414123535}}}`.

Patient-level OOF transitions: `{"M3-M1": {"ASD_correct_net": 1, "TD_correct_net": 4, "correct_to_error": 4, "correct_to_error_ids": ["50025", "50390", "50527", "51568"], "error_to_correct": 9, "error_to_correct_ids": ["50145", "50358", "50575", "50772", "50970", "50978", "51046", "51048", "51251"], "n": 871, "net_correct_change": 5}, "M3-M2": {"ASD_correct_net": 1, "TD_correct_net": 3, "correct_to_error": 2, "correct_to_error_ids": ["50025", "51047"], "error_to_correct": 6, "error_to_correct_ids": ["50271", "50526", "50575", "50970", "51251", "51333"], "n": 871, "net_correct_change": 4}, "M3-M4": {"ASD_correct_net": -1, "TD_correct_net": 7, "correct_to_error": 4, "correct_to_error_ids": ["50152", "50295", "50669", "51291"], "error_to_correct": 10, "error_to_correct_ids": ["50247", "50271", "50363", "50526", "50575", "50772", "51048", "51100", "51251", "51333"], "n": 871, "net_correct_change": 6}, "M3-historical_D4": {"ASD_correct_net": 0, "TD_correct_net": 5, "correct_to_error": 3, "correct_to_error_ids": ["50295", "50669", "51291"], "error_to_correct": 8, "error_to_correct_ids": ["50271", "50363", "50526", "50575", "50772", "51001", "51100", "51251"], "n": 871, "net_correct_change": 5}}`.

### Audit and limitations

Historical train_start/split/RFE/VAE hashes are recorded in `audit.json`; all 40 tasks use seed911 and the original ten folds. Original model source hashes: `{"model/gtunet.py": "30539f14be3b69128610a7bf2a03df4f9ef36a9552002e739ce45558c0d6496c", "model/mm_gtunets.py": "cb0c27fdc29426e55746a3dc4926ffd2a316d15042fd33a3bd8f207a10e31158", "model/rp_graph.py": "d1c79f3319f9569c8caf98737f3ade96c7c5d334bb97f5c56255ee4ddc3a1783", "opt.py": "068a2fac00ce756eab4a0528a1d5ecc6a4e6e8f4fdd6af87e64fcedb2c18de3a", "train_mm_gtunets.py": "539b4af31189e2215a5a17adcafe78fe2cf4d1d9fafcb5a62aaa8061ce938cba", "utils/metrics.py": "59405483453a068143de753301910250f2609fa148fd4a42d4cf56ebcc432056", "utils/mydataloader.py": "7993dc3321c2a6bd2469313371793f1e6210a503a348a6a5d390dc654913ccf5", "utils/tools.py": "b08dfd36935b506b90712c5588488956c4a91b670b3bf8ba1186dd14602c3192"}`.

The study retains the transductive 871-node population graph, its historical cross-fold RFE behavior, stochastic VAE sampling, RP in-place constraints, and grad-enabled stochastic single test forward. The non-imaging branch receives only site, sex, and age; prototype evidence cannot be described as an independently identified disease factor. Historical D4 results are read-only and no historical model is retrained.

The 24 grouped mathematical assertions and real-data single-forward preflight passed. Preflight captured logits `[871, 2]`, z_img `[871, 16]`, and z_ph `[871, 16]` with exactly one core forward; M1-M4 had identical image D4 loss, phenotype CE, and teacher logits on the shared inputs. Training, manifest, analysis, input, and original-source SHA-256 values are recorded in `audit.json` and `manifest.json`.
