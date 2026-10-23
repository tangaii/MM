"""Record the incomplete Stage A attempt without repeating its D4 forward."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
ROOT = Path("/data/adodas/MM-GTUNets")
sys.path.insert(0, str(BASE))
from audit import code_hashes, hash_json, sha_file, source_hashes  # noqa: E402


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    audit_path = BASE / "audit.json"
    audit = json.loads(audit_path.read_text())
    attempt = json.loads((BASE / "stage_a_fold_0.attempt.json").read_text())
    if attempt.get("status") != "native_forward_started":
        raise RuntimeError("The failed forward attempt is not correctly marked")
    if audit.get("formal_authorized"):
        raise RuntimeError("Formal training must remain unauthorized")
    for path, expected in audit["historical_inputs_sha256"].items():
        if sha_file(path) != expected:
            raise AssertionError(f"Read-only historical input changed: {path}")

    failure = {"status": "incomplete_no_output", "fold": 0, "group": "historical_D4 read-only diagnostic",
               "D4_best_checkpoint": attempt["D4_best_checkpoint"],
               "D4_best_checkpoint_sha256": attempt["D4_best_checkpoint_sha256"],
               "diagnostic_private_rng_seed": attempt["diagnostic_private_rng_seed"],
               "native_forward_invocations": 1, "embedding_persisted": False,
               "fold_summary_persisted": False,
               "failure": "KeyError: 'rank' in projector_angle_degrees while the outer-train probe and null calculations were in memory; failure occurred before stage_a_f0.npz/evidence JSON persistence.",
               "retry_policy": "No retry: the registered one-forward-per-fold limit was consumed, and the stochastic representation was not persisted.",
               "formal_training_started": False, "completed_utc": utc()}
    write_json(BASE / "stage_a_failure.json", failure)

    evidence_path = BASE / "evidence_audit.json"
    evidence = json.loads(evidence_path.read_text()) if evidence_path.is_file() else None
    if evidence:
        evidence.setdefault("incomplete_folds", {})["0"] = {
            "status": failure["status"],
            "native_forward_count": failure["native_forward_invocations"],
            "embedding_persisted": failure["embedding_persisted"],
            "failure": failure["failure"],
            "retry_forbidden": True,
        }
        private_angles = [row["direction_stability"]["pairwise_principal_angle_mean_degrees"]
                          for row in evidence["folds"].values()
                          if row["direction_stability"]["pairwise_principal_angle_mean_degrees"] is not None]
        inter_main_angles = [angle for row in evidence["folds"].values()
                             for angle in row["direction_stability"]["inter_to_main_angles_degrees"]
                             if angle is not None]
        evidence["summary"]["mean_private_reference_subspace_principal_angle_degrees"] = (
            sum(private_angles) / len(private_angles) if private_angles else None)
        evidence["summary"]["mean_interaction_to_disease_main_angle_degrees"] = (
            sum(inter_main_angles) / len(inter_main_angles) if inter_main_angles else None)
        write_json(evidence_path, evidence)
    evidence_summary = evidence["summary"] if evidence else None
    completed_folds = len(evidence.get("folds", {})) if evidence else 0

    audit["status"] = "STAGE_A_INCOMPLETE_STOP"
    audit["formal_authorized"] = False
    audit["stage_a"] = {"status": "incomplete", "successful_folds": completed_folds,
                        "attempted_fold0_native_forwards": 1,
                        "fold0_embedding_persisted": False,
                        "folds1_to9_native_forwards": completed_folds,
                        "failure_sha256": sha_file(BASE / "stage_a_failure.json"),
                        "diagnostic_result": {"path": str(evidence_path),
                                              "sha256": sha_file(evidence_path),
                                              "summary": evidence_summary} if evidence else None,
                        "incomplete_folds": evidence.get("incomplete_folds", {}) if evidence else {"0": failure},
                        "reason": "Fold0 output was not persisted before an audit aggregation exception. Remaining folds were run once after fixing the summary bug; fold0 was not repeated."}
    audit["postmortem"] = {"time_utc": utc(),
                           "training_code_sha256_before_failure": audit["training_code_sha256"],
                           "current_code_sha256_after_reporter_fix": hash_json(code_hashes()),
                           "current_code_files_sha256_after_reporter_fix": code_hashes(),
                           "original_source_sha256_at_stop": source_hashes(),
                           "formal_authorized": False,
                           "formal_tasks_completed": 0,
                           "historical_models_retrained": 0,
                           "postmortem_change": "Fixed the direction-stability summary key lookup; fold0 was not rerun. Each remaining fold was then evaluated exactly once."}
    audit["execution"].update({"completed": 0, "GPU0": 0, "GPU1": 0,
                               "failed": [{"phase": "stage_a", "fold": 0,
                                           "error": failure["failure"],
                                           "native_forward_count": 1,
                                           "output_persisted": False}],
                               "retries": 0, "worker_seconds": 0.0, "wall_seconds": None,
                               "formal_tasks_started": 0,
                               "stage_a_folds_completed": completed_folds,
                               "stage_a_forward_counts": {"fold0": 1, "folds1_to9": completed_folds},
                               "stage_a_gpu_continuation": "PPU0",
                               "formal_gpu_tasks": {"GPU0": 0, "GPU1": 0}})
    handoff_path = ROOT / "handoff.md"
    pending_path = BASE / "handoff_pending.md"
    audit["handoff"] = {"updated": False,
                        "attempted": True,
                        "append_only": True,
                        "before_bytes": audit["protected_files"]["handoff_before_bytes"],
                        "before_sha256": audit["protected_files"]["handoff_before_sha256"],
                        "write_error": "apply_patch returned Failed to write file /data/adodas/MM-GTUNets/handoff.md",
                        "after_bytes": handoff_path.stat().st_size,
                        "after_sha256": sha_file(handoff_path),
                        "old_prefix_verified": handoff_path.stat().st_size == audit["protected_files"]["handoff_before_bytes"] and
                                               sha_file(handoff_path) == audit["protected_files"]["handoff_before_sha256"],
                        "end_marker_count": handoff_path.read_text(errors="replace").count("PEIM_20261023_END"),
                        "pending_path": str(pending_path),
                        "pending_sha256": sha_file(pending_path) if pending_path.is_file() else None}
    write_json(audit_path, audit)

    summary = {"experiment": "PEIM_20261023", "name": "Phenotype-dependent Evidence-guided Interaction Modulation",
               "status": "PARTIAL", "stop_reason": "STAGE_A_INCOMPLETE_STOP",
               "formal_tasks_completed": 0, "formal_tasks_total": 40,
               "GPU0_formal_tasks": 0, "GPU1_formal_tasks": 0,
               "historical_retraining": {"D0": 0, "D4": 0, "M1": 0, "M3": 0,
                                         "experiment25": 0, "other": 0},
               "stage_a": {"status": "incomplete", "successful_folds": completed_folds,
                           "attempted_fold0_forwards": 1,
                           "folds1_to9_forwards": completed_folds,
                           "summary": evidence_summary,
                           "completed_fold_metrics": [
                               {"fold": int(fold),
                                "Delta_MSE": row["true"]["Delta_MSE"],
                                "null_mean": row["null"]["mean"],
                                "null_P95": row["null"]["P95"],
                                "true_above_null_P95": row["null"]["true_exceeds_P95"],
                                "interaction_rank_by_private_fold": [x["rank_interaction"] for x in row["directions"]],
                                "private_reference_subspace_angle_mean_degrees": row["direction_stability"]["pairwise_principal_angle_mean_degrees"],
                                "interaction_to_disease_main_angle_mean_degrees": sum(
                                    angle for angle in row["direction_stability"]["inter_to_main_angles_degrees"]
                                    if angle is not None) / max(sum(angle is not None for angle in row["direction_stability"]["inter_to_main_angles_degrees"]), 1)}
                               for fold, row in sorted(evidence.get("folds", {}).items(), key=lambda item: int(item[0]))
                           ] if evidence else [],
                           "incomplete_folds": evidence.get("incomplete_folds", {}) if evidence else {"0": failure},
                           "diagnostic_result_sha256": sha_file(evidence_path) if evidence else None,
                           "reason": failure["failure"]},
               "formal_metrics": None, "OOF": None, "candidate_judgment": "NO_GO_STAGE_A_INCOMPLETE",
               "no_formal_training_or_test_evaluation": True,
               "historical_source_code_sha256": audit["original_source_sha256"],
               "attempted_training_code_sha256": audit["training_code_sha256"],
               "postmortem_current_code_sha256": hash_json(code_hashes()),
               "attempted_D4_checkpoint_sha256": attempt["D4_best_checkpoint_sha256"],
               "historical_inputs_sha256": audit["historical_inputs_sha256"],
               "stage_a_failure_sha256": sha_file(BASE / "stage_a_failure.json"),
               "formal_worker_seconds": 0.0, "handoff_updated": False,
               "handoff": audit["handoff"],
               "created_utc": utc()}
    write_json(BASE / "summary.json", summary)

    delta_summary = evidence_summary or {}
    fold_lines = []
    if evidence:
        for fold, row in sorted(evidence["folds"].items(), key=lambda item: int(item[0])):
            fold_lines.append(
                f"| {fold} | {row['true']['Delta_MSE']:.6f} | {row['null']['mean']:.6f} | "
                f"{row['null']['P95']:.6f} | {'yes' if row['null']['true_exceeds_P95'] else 'no'} | "
                f"{','.join(str(x['rank_interaction']) for x in row['directions'])} | "
                f"{row['direction_stability']['pairwise_principal_angle_mean_degrees']:.2f} |"
            )
    mean_inter_main = delta_summary.get("mean_interaction_to_disease_main_angle_degrees")
    report = f"""# PEIM Experiment 26

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

Stage A is incomplete because fold0 has no persisted result, so the ten-fold integrity/evidence gate cannot be evaluated and no formal training is authorized. The nine completed folds do provide a partial diagnostic: mean `Delta_MSE` {delta_summary.get('Delta_MSE_mean_completed_folds_only', float('nan')):.6f}, median {delta_summary.get('Delta_MSE_median_completed_folds_only', float('nan')):.6f}; positive folds {delta_summary.get('positive_Delta_MSE_folds', 0)}/9; true values above the within-fold 32-null P95 in {delta_summary.get('true_above_null_P95_folds', 0)}/9. All nine completed folds had nonpositive `Delta_MSE`; the interaction projector had rank 2 in all three private folds. The mean principal angle between private-reference interaction subspaces was {delta_summary.get('mean_private_reference_subspace_principal_angle_degrees', float('nan')):.2f} degrees (fold means 50.57–66.09); this reflects substantial direction differences, not stability. Their mean angle to the ordinary disease main-effect direction was {mean_inter_main:.2f} degrees (range 73.93–88.00). These are descriptive partial results, not a completed ten-fold test or formal GO criterion.

| Fold | Delta_MSE | Null mean | Null P95 | True > P95 | Interaction rank (Q0,Q1,Q2) | Private-reference subspace angle |
|---:|---:|---:|---:|:---:|:---:|---:|
{chr(10).join(fold_lines)}
| 0 | unavailable | unavailable | unavailable | unavailable | unavailable | unavailable |

The registered E1-E4 method remains unchanged: 32 zero-initialized parameters `W_d`; cross-fitted rank-at-most-two ordinary disease (`P_main`) or phenotype-by-image interaction (`P_int`) projection; multiplicative `Z * (W_d d)` modulation; rho warmup to 0.1; per-patient 5% relative L2 cap; E3/E4 train-reference prototype-distance gate; E4 within-split/site phenotype derangement; native CE+Graph plus the original D4 image prototype CE only. These methods were not trained.

No GO claim is made. A complete training-internal evidence audit is a prerequisite before any formal fold training; the registered limit does not allow this failed fold to be silently rerun. An independent future rerun would need a new experiment registration with a pre-forward persistence boundary for diagnostic outputs.

The append-only `handoff.md` attempt returned a write error. Its original byte count and SHA256 were unchanged after verification. The proposed append is in `handoff_pending.md`; its SHA256 and the failed attempt are recorded in `audit.json`.

Paper/source review references: [MM-GTUNets](https://arxiv.org/html/2406.14455v3), [RTGMFF](https://arxiv.org/abs/2509.03214), and [FiLM](https://arxiv.org/abs/1709.07871). FiLM and low-rank bilinear interaction are established tools; PEIM's candidate contribution would be task-controlled interaction evidence and confidence-selective modulation, which this run did not validate.
"""
    (BASE / "report.md").write_text(report)
    log = {"utc": utc(), "event": "stage_a_incomplete_stop", "failure": failure,
           "successful_stage_a_folds": completed_folds, "formal_tasks": "0/40", "no_fold0_retry": True}
    with (BASE / "run.log").open("a") as stream:
        stream.write(json.dumps(log, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
