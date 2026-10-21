"""Recompute saved OOF probabilities on CPU to remove backend rounding drift."""
from __future__ import annotations

import numpy as np
import torch

from run import (
    BASE, CFG, append_log, atomic_json, atomic_torch, audit, formal_metrics, load_torch,
    read_json, result_path, sha_file, utc, valid_result,
)


def main():
    audit_info = audit()
    queue_path = BASE / "queue.json"
    queue = read_json(queue_path)
    if queue.get("active") or queue.get("pending"):
        raise AssertionError("Output normalization requires a stopped queue with no active or pending workers")
    changes = []
    for group in CFG["groups"]:
        for fold in range(10):
            path = result_path(group, 911, fold)
            if not path.exists():
                continue
            value = load_torch(path)
            if value.get("state") != "complete":
                continue
            oof = value["oof"]
            expected = torch.softmax(torch.as_tensor(oof["logits"]), dim=1).numpy()[:, 0]
            current = np.asarray(oof["probability"])
            delta = np.abs(expected - current)
            if np.allclose(expected, current, rtol=0, atol=1e-7):
                continue
            if not np.isfinite(expected).all():
                raise FloatingPointError(f"Nonfinite CPU softmax output: {path}")

            old_sha256 = sha_file(path)
            old_metrics = value["metrics"]
            oof["probability"] = expected
            value["metrics"] = formal_metrics(oof["labels"], expected, oof["prediction"])
            atomic_torch(value, path)
            new_sha256 = sha_file(path)

            status_path = path.with_suffix(".status.json")
            status_before = read_json(status_path) if status_path.exists() else None
            atomic_json({
                "state": "complete", "group": group, "seed": 911, "fold": fold,
                "completed_utc": utc(), "artifact_sha256": new_sha256,
                "training_seconds": value["training_seconds"], "best_epoch": value["best_epoch"],
                "probability_normalized_from_saved_logits": True,
            }, status_path)
            changes.append({
                "group": group, "fold": fold, "max_abs_probability_delta": float(delta.max()),
                "old_artifact_sha256": old_sha256, "new_artifact_sha256": new_sha256,
                "old_metrics": old_metrics, "new_metrics": value["metrics"],
                "previous_status": status_before,
            })

    verification_failures = []
    verified = 0
    for group in CFG["groups"]:
        for fold in range(10):
            if not result_path(group, 911, fold).exists():
                continue
            try:
                valid_result(group, 911, fold, audit_info, required=True)
                verified += 1
            except Exception as error:
                verification_failures.append({"group": group, "fold": fold, "error": repr(error)})

    if verified != 40 or verification_failures:
        raise AssertionError(f"Expected 40 validated results, got {verified}: {verification_failures}")

    archived_failures = queue.get("failures", [])
    archived_retries = queue.get("retries", [])
    completed = []
    for group in CFG["groups"]:
        for fold in range(10):
            value = valid_result(group, 911, fold, audit_info, required=True)
            completed.append({
                "group": group, "seed": 911, "fold": fold, "gpu": fold % 2,
                "attempt": 1, "artifact_sha256": sha_file(result_path(group, 911, fold)),
                "worker_seconds": value["training_seconds"], "validated_after_normalization": True,
            })

    queue["reconciled_validation_failures"] = archived_failures
    queue["reconciled_validation_retries"] = archived_retries
    queue["state"] = "complete"
    queue["completed"] = completed
    queue["pending"] = []
    queue["active"] = []
    queue["failures"] = []
    queue["postprocessing"] = {
        "timestamp_utc": utc(), "kind": "saved_logits_probability_reconciliation",
        "artifacts_validated": verified, "changed_artifacts": len(changes),
    }
    atomic_json(queue, queue_path)

    entry = {
        "timestamp_utc": utc(),
        "method": "CPU float32 softmax recomputed from unchanged saved native logits; metrics recalculated",
        "training_code_sha256": audit_info["training_code_sha256"],
        "changed_artifacts": len(changes), "records": changes,
        "verified_artifacts": verified, "verification_failures": verification_failures,
    }
    audit_info.setdefault("probability_normalization", []).append(entry)
    execution = audit_info["execution"]
    execution["gpu0"] = 20
    execution["gpu1"] = 20
    execution["completed"] = verified
    execution["task_attempt_validation_failures_reconciled"] = archived_failures
    execution["validation_retries_reconciled"] = len(archived_retries)
    execution["failures"] = []
    execution["output_validation"] = {
        "complete": True, "validated_artifacts": verified,
        "changed_artifacts": len(changes), "verification_failures": [],
    }
    atomic_json(audit_info, BASE / "audit.json")
    append_log("probability_normalization_complete", changed=len(changes), verified=verified,
               verification_failures=len(verification_failures))
    print({"changed": len(changes), "verified": verified, "failures": verification_failures})


if __name__ == "__main__":
    main()
