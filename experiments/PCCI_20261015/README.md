# PCCI Experiment 3

Test whether correctly paired age/sex provides incremental diagnostic information
through a bounded conditional residual on the existing MM-GTUNets joint embedding.
Reference commit: `8e694537cc197520de0be36d3a18ce6c27accd31`.

The runner imports only the protected root model/utilities. Historical experiment
code is not a runtime dependency. Data, E0 results, splits, RFE features, VAE states,
initial core states and training RNG anchors are read-only references to
`../DRPA_20261014/`. C0 reuses the 30 verified E0 fits. C1-C6 train 180 new fits.

```bash
source ~/venvs/mm/bin/activate
python -B experiments/PCCI_20261015/launch.py
```

The latest scheduling instruction overrides the original serial-group schedule:
C1-C6 share one queue, with up to six independent workers per physical PPU.
Admission checks live device memory/utilization and host CPU/RAM, including
startup memory reservations. Healthy workers are never preempted for pressure.
Fold parity is preferred across methods; work stealing is recorded explicitly.
Each completed group is summarized immediately, without waiting for other groups.
Audit/equivalence and C0 validation precede training; five common-RNG frozen draws
and C2-C5 interventions follow all formal training. One atomic `.pt` per new fit;
summaries, failure/resume records and input/code hashes are in `summary.json`.
Only the parent writes `summary.json`, `report.md`, the global `run.log` and the
PCCI section of root `handoff.md`.
The final handoff update follows all prompt tasks, not just formal training.
`--adopt-launcher PID` transfers an existing coordinator's stdout readers before
retiring that coordinator, retaining its healthy training workers and fixed jobs.
Training source/configuration hashes and validated results remain unchanged.
The launcher retries only transient `summary.json` startup reads for up to 30s:
the `/data` ossfs mount can briefly hide a target during replacement. This wrapper
does not alter the training files, RNG, model, saved results, or retry budget.

Age uses train-only moments; sex is one-hot (original 1/2); site uses a train-only
vocabulary plus unknown slot. All groups share 24-D inputs and the same module;
site is masked outside C4. C6 jointly deranges age/sex within site over all 871
subjects using a private RNG. Output projection starts at zero. C5 adds 0.005
times training-row gate mean; checkpoint choice remains validation CE+graph.

Scientific comparisons and fixed draw policy are preregistered in `config.json`.
Single stochastic official scores and five-draw probability averages are reported
separately. Residual-off uses each newly trained core, not historical C0.
The inherited RFE leakage and transductive same-site protocol remain limitations.
Conditional modulation/gates are established techniques; this study tests paired,
non-additive predictive complementarity rather than claiming FiLM as a novelty.
