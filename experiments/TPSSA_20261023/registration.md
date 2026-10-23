# Experiment 24 TPSSA — preregistration

User-authorized output directory: `/tmp/TPSSA_20261023`. All original model/data/history paths are read-only. Only `handoff.md` may receive an idempotent append. Seed 911 and the original ten outer folds are fixed. This is a feasibility study for a candidate MICCAI 2027 first contribution, not optimization of this seed.

Stage A has no neural-network training, optimizer steps, validation/test scoring, or heldout diagnosis use. Each historical D4 best core is restored with its explicit RP state, followed by one fixed private-RNG native PPU eval forward. Both encoders still see the 871-node transductive graph. Only outer-train image embeddings are retained. Input CSV age/sex/site and historical private folds are used verbatim; the historical class-balanced private partition does not become label-independent through reuse.

`config.json` fixes the original scientific thresholds before any embeddings are extracted. Nonzero shared rank and protectability are assessed in all three query-excluded reference pools; an outer fold counts only when all three references meet the relevant validity condition. This conservative aggregation resolves the otherwise unspecified per-reference aggregation without consulting results. Delta R2 uses raw image-embedding squared error, so per-coordinate target scaling cannot silently change the gate's meaning. Ridge predictors have reference-only centering/scaling and alpha=1, with an unpenalized intercept. No search over alpha, rank, direction thresholds or losses is permitted.

Let g denote a historical private query fold and r={pf != g}. For every group,

`H_I=(z_I-mean_r(z_I))/std_r(z_I)`, `H_P=(z_P-mean_r(z_P))/std_r(z_P)`;
`W_g=stopgrad[(H_Pr^T H_Pr+I)^(-1)H_Pr^T H_Ir]`.

S1: `L_full=mean_{i,d}(H_Ii,d-(H_Pi W_g)_d)^2` over outer-train queries.

S2: site-residualize detached reference `H_Ir` and real raw-age/sex encoding with fixed ridge. `C_g=R_I^T R_age,sex/n_r`; `u_g=stopgrad[leading left singular vector(C_g)]`, k=1. Let `c_g=max(std_r(H_Ir u_g),0.1)`. `L_shared=mean_i[((H_Ii u_g)-(H_Pi W_g u_g))/c_g]^2`. Do not per-patient L2-normalize a scalar.

S3: `d_g=stopgrad[normalize(mean_ASD,r(H_I)-mean_TD,r(H_I))]`, `P_g=I-d_g d_g^T`. Only the added image alignment path uses `H_I_safe=stopgrad(H_I)+(H_I-stopgrad(H_I))P_g`. S2 and S3 values are identical. Direct image-output alignment derivatives in standardized coordinates are projected; native and historical D4 derivatives are unchanged. This does not claim parameter-wide orthogonality or protection of all nonlinear disease information.

S4: same as S3, with only query phenotype alignment correspondence changed by a fixed label-blind within-private-fold/site derangement. Original input, native graph, ridge references, W, u, d and P remain true-corresponded.

`L_native=CE_native+Graph_native`. Historical D4 CE remains byte-equivalent to the accepted `PCDCA_20261019/prototype.py`, with its own 5% image budget, epoch/9 warmup and cap 0.5. No old phenotype prototype CE or disease-probability KL remains. The added alignment paths use independent detached output-gradient scales for image and phenotype, each with a 5% native output-gradient target. A numerical surrogate may preserve the forward alignment value while composing the two separately scaled derivatives; optimization is explicitly a branch-separated gradient combination, not plain differentiation of one scalar lambda. Zero/constant directions give graph-connected zero; NaN/Inf blocks the task.

Any failed Stage A gate sets `NO_GO_STAGE_A`, permanently disables this registration's formal launcher, and leaves formal task count at 0/40. The fold0 train-start training-mode one-forward preflight and all 40 formal training tasks are conditional on passing Stage A. No historical model is retrained. No checkpoint is created on a NO-GO path.

Formal inference, if authorized by passing Stage A: retain train-end RNG, restore best core and RP state without restoring best RNG, one native grad-enabled eval forward, PPU probability `softmax(logits)[:,0]`, class ASD=0. Only four classification metrics, per-fold then mean±sample SD. Patient transitions are always candidate minus reference: corrected reference errors minus newly introduced errors, checked with synthetic data.

Limitations retained: supervised cross-fold RFE contamination, all-subject transductive graph, native stochastic eval VAE/dropout, RP forward constraints and unregistered weights, detached fusion-score tensors, PPU trajectory nondeterminism. Stage A is exploratory, not a formal significance test; it cannot prove age/sex never carries biological information or establish unbiased clinical generalization.
