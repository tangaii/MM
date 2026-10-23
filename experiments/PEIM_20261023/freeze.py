"""Freeze PEIM's already-registered source and authorize the 40 formal folds."""
from __future__ import annotations

import json
from pathlib import Path

from audit import (BASE, CFG, CODE_FILES, hash_json, code_hashes, make_anchor_records,
                   sha_file, source_hashes, utc)


def main():
    audit_path = BASE / "audit.json"
    audit = json.loads(audit_path.read_text())
    if audit.get("formal_authorized"):
        raise RuntimeError("PEIM source is already frozen")
    math_tests = audit.get("math_tests")
    if not math_tests or not math_tests.get("passed"):
        raise RuntimeError("Synthetic mathematical acceptance is incomplete")
    evidence_path = BASE / "evidence_audit.json"
    evidence = json.loads(evidence_path.read_text())
    if evidence.get("status") != "completed" or len(evidence.get("folds", {})) != 10:
        raise RuntimeError("Stage A must finish all ten historical folds before formal work")
    preflight_path = BASE / "preflight.json"
    preflight = json.loads(preflight_path.read_text())
    if preflight.get("status") != "passed" or preflight.get("native_forward_count") != 1:
        raise RuntimeError("Fold-0 single-forward preflight did not pass")

    current_code = code_hashes()
    current_source = source_hashes()
    if current_code != audit["training_code_files_sha256"]:
        raise RuntimeError("Experiment source changed after Stage A")
    if current_source != audit["original_source_sha256"]:
        raise RuntimeError("Original repository source changed during PEIM preparation")
    anchors, _ = make_anchor_records()
    if anchors != audit["anchors"]:
        raise RuntimeError("Historical anchors changed after registration")
    for path, expected in audit["historical_inputs_sha256"].items():
        if sha_file(path) != expected:
            raise RuntimeError(f"Historical input changed: {path}")

    manifest = {"experiment": CFG["experiment"], "frozen_utc": utc(),
                "requested_output": str(BASE), "config_sha256": sha_file(BASE / "config.json"),
                "code_files": current_code, "code_sha256": hash_json(current_code),
                "original_model_source_sha256": current_source,
                "historical_readonly_input_sha256": audit["historical_inputs_sha256"],
                "anchors": {fold: {"train_start_sha256": row["train_start_sha256"],
                                   "train_start_rng_sha256": row["train_start_rng_sha256"],
                                   "split_sha256": row["split_sha256"],
                                   "rfe_support_sha256": row["rfe_support_sha256"],
                                   "vae_sha256": row["vae_sha256"],
                                   "d4_best_sha256": row["d4_best_sha256"]}
                            for fold, row in anchors.items()},
                "stage_a_sha256": sha_file(evidence_path),
                "preflight_sha256": sha_file(preflight_path),
                "design": {"seed": 911, "folds": list(range(10)), "groups": CFG["groups"],
                           "tasks": 40, "GPU0_even_folds": 20, "GPU1_odd_folds": 20,
                           "rank_cap": 2, "new_trainable_parameters": 32,
                           "new_loss": False, "D4_source": "historical PCDCA PrototypeObjective/gradient_scale",
                           "selection": "native validation CE+Graph only",
                           "formal_probability": "single PPU test forward softmax(logits)[:,0]"},
                "historical_retraining": {"D0": 0, "D4": 0, "M1": 0, "M3": 0,
                                          "other_historical_groups": 0},
                "outer_test_used_for_design": False,
                "method_values_changed_after_stage_a": False}
    manifest_path = BASE / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    manifest_sha = sha_file(manifest_path)
    audit["status"] = "FORMAL_TRAINING_AUTHORIZED"
    audit["formal_authorized"] = True
    audit["manifest_sha256"] = manifest_sha
    audit["frozen_utc"] = manifest["frozen_utc"]
    audit["code_hash_rechecked_before_authorization"] = hash_json(current_code)
    audit["source_hash_rechecked_before_authorization"] = current_source
    audit["preflight"] = {"path": str(preflight_path), "sha256": sha_file(preflight_path),
                          "status": preflight["status"],
                          "native_forward_count": preflight["native_forward_count"]}
    audit["execution"]["manifest_sha256"] = manifest_sha
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"formal_authorized": True, "manifest_sha256": manifest_sha,
                      "code_sha256": hash_json(current_code)}, indent=2))


if __name__ == "__main__":
    main()
