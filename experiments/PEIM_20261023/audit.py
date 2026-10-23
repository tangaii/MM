"""Input integrity and training-only OOF interaction-evidence diagnostic."""
from __future__ import annotations

import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

from evidence import (diagnosis_target, fit_reference, interaction_projector, main_effect_projector,
                      predict_reference_delta, private_seed, shuffle_conditions)
from runtime import (BASE, CFG, HISTORY, PCDCA, ROOT, Loaded, array_hash, capture_rng, encode_conditions,
                     export_core, historical_prototype_module, import_core, load_torch,
                     payload_hash, sha_file, state_hash, utc)


CODE_FILES = ("config.json", "evidence.py", "interaction.py", "runtime.py", "run.py",
              "audit.py", "preflight.py", "launch.py", "analyze.py", "test_peim.py", "freeze.py",
              "history_review.json")
SOURCE_FILES = ("opt.py", "train_mm_gtunets.py", "model/mm_gtunets.py", "model/gtunet.py",
                "model/rp_graph.py", "utils/mydataloader.py", "utils/tools.py", "utils/metrics.py")


def hash_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def code_hashes():
    return {name: sha_file(BASE / name) for name in CODE_FILES}


def source_hashes():
    return {name: sha_file(ROOT / name) for name in SOURCE_FILES}


def private_seed_rng(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed % (2 ** 32))
    torch.cuda.manual_seed(seed % (2 ** 32))


def make_anchor_records():
    old_audit = json.loads((PCDCA / "audit.json").read_text())
    records = {}
    for fold in range(10):
        directory = HISTORY / "artifacts" / "seed_911" / f"fold_{fold}"
        files = {
            "splits": directory / "splits.pt",
            "train_start": directory / "train_start.pt",
            "fingerprints": directory / "fingerprints.json",
            "selected_features": directory.parent / "selected_features.pt",
            "rfe_support": directory.parent / "rfe_support.npy",
            "vae": directory.parent / "vae" / f"fold{fold}_pretrain.pth",
            "d4_best": PCDCA / "runs" / "D4" / f"s911_f{fold}.pt",
        }
        expected = old_audit["anchors"][f"911/{fold}"]["input_sha256"]
        hashes = {}
        for name, path in files.items():
            if not path.is_file():
                raise FileNotFoundError(path)
            value = sha_file(path)
            if name != "d4_best" and expected[str(path)] != value:
                raise AssertionError(f"Historical anchor fingerprint mismatch: {path}")
            hashes[str(path)] = value
        split = load_torch(files["splits"])
        start = load_torch(files["train_start"])
        fingerprint = json.loads(files["fingerprints"].read_text())
        if not np.array_equal(np.sort(np.concatenate([split[k] for k in ("train", "val", "test")])), np.arange(871)):
            raise AssertionError(f"Fold {fold} does not partition all 871 subjects")
        if len(split["test"]) == 0 or len(split["val"]) == 0:
            raise AssertionError(f"Fold {fold} has an empty held-out partition")
        if not np.array_equal(start["core"]["state"]["clf.2.num_batches_tracked"].numpy(), np.asarray(0, dtype=np.int64)):
            raise AssertionError(f"Unexpected train-start BatchNorm state: fold {fold}")
        if start["vae_sha256"] != hashes[str(files["vae"])]:
            raise AssertionError(f"Fold {fold} VAE SHA mismatch")
        records[str(fold)] = {"paths": {name: str(path) for name, path in files.items()},
                              "sha256": hashes,
                              "train_start_sha256": hashes[str(files["train_start"])],
                              "train_start_rng_sha256": payload_hash(start["rng"]),
                              "split_sha256": hashes[str(files["splits"])],
                              "rfe_support_sha256": hashes[str(files["rfe_support"])],
                              "vae_sha256": hashes[str(files["vae"])],
                              "d4_best_sha256": hashes[str(files["d4_best"])],
                              "split_fingerprints": fingerprint["split_fingerprints"],
                              "selected_features_fingerprint": fingerprint["features_fingerprint"],
                              "initial_core_fingerprint": fingerprint["train_start_core_fingerprint"],
                              "GPU": fold % 2}
    return records, old_audit


def make_audit_record():
    anchors, old_audit = make_anchor_records()
    protected = [ROOT / CFG["data_folder"] / "Phenotypic_V1_0b_preprocessed1.csv",
                 ROOT / CFG["data_folder"] / "id.txt",
                 ROOT / CFG["data_folder"] / "save_tensor/labels.pt",
                 ROOT / CFG["data_folder"] / "save_tensor/ph_data_SITE_ID_SEX_AGE_AT_SCAN.pt",
                 ROOT / CFG["data_folder"] / "save_tensor/ph_dict_SITE_ID_SEX_AGE_AT_SCAN.pt",
                 ROOT / CFG["data_folder"] / "save_tensor/up_triangles.pt",
                 HISTORY / "artifacts/context.pt", PCDCA / "config.json", PCDCA / "prototype.py",
                 PCDCA / "audit.json", PCDCA / "run.py", PCDCA / "runs/D4"]
    protected.extend([ROOT / "experiments/PCDCA_20261019/report.md",
                      ROOT / "experiments/PCDPAX_20261021/report.md",
                      ROOT / "experiments/PCCI_20261015/report.md",
                      Path("/tmp/IAFiLM_20261023/report.md"),
                      Path("/tmp/IAFiLM_20261023/summary.json"),
                      Path("/tmp/IAFiLM_20261023/oof.csv"),
                      BASE / "history_review.json"])
    for path in protected:
        if path.is_dir():
            continue
        if not path.is_file():
            raise FileNotFoundError(path)
    files = {}
    for path in protected:
        if path.is_file():
            files[str(path)] = sha_file(path)
        else:
            for checkpoint in sorted(path.glob("s911_f*.pt")):
                files[str(checkpoint)] = sha_file(checkpoint)
    handoff = ROOT / "handoff.md"
    return {"experiment": CFG["experiment"], "status": "STAGE_A_RUNNING",
            "created_utc": utc(), "requested_output": str(BASE), "actual_output": str(BASE),
            "output_location_user_requested": True,
            "training_code_sha256": hash_json(code_hashes()), "training_code_files_sha256": code_hashes(),
            "original_source_sha256": source_hashes(),
            "historical_pcdca_prototype_sha256": sha_file(PCDCA / "prototype.py"),
            "historical_pcdca_config_sha256": sha_file(PCDCA / "config.json"),
            "historical_pcdca_audit_sha256": sha_file(PCDCA / "audit.json"),
            "historical_inputs_sha256": files, "context_sha256": sha_file(HISTORY / "artifacts/context.pt"),
            "anchors": anchors, "folds": 10, "seed": 911,
            "cohort": {"n": 871, "ASD_label": 0, "TD_label": 1,
                       "classes": [403, 468], "sites": 20, "image_representation_dim": 16},
            "protected_files": {"handoff_path": str(handoff), "handoff_before_bytes": handoff.stat().st_size,
                                 "handoff_before_sha256": sha_file(handoff),
                                 "PEIM_20261023_END_count_before": handoff.read_text(errors="replace").count("PEIM_20261023_END")},
            "history_policy": {"D0_retrained": False, "D4_retrained": False, "M1_retrained": False,
                               "M3_retrained": False, "other_history_retrained": False,
                               "RFE_refit": False, "VAE_pretrained": False,
                               "folds_regenerated": False, "historical_checkpoints_read_only": True},
            "math_tests": None, "stage_a": None, "preflight": None,
            "formal_authorized": False, "manifest_sha256": None,
            "execution": {"tasks_total": 40, "completed": 0, "GPU0": 0, "GPU1": 0,
                          "failed": [], "retries": 0, "worker_seconds": 0.0,
                          "wall_seconds": None}}


def private_rng(seed, fold):
    phrase = f"PEIM-stage-a/{seed}/{fold}/one-forward".encode()
    return int(hashlib.sha256(phrase).hexdigest()[:15], 16) % (2 ** 32)


def projector_angle_degrees(a, b):
    if a["rank"] == 0 or b["rank"] == 0:
        return None
    ua, ub = a.get("u"), b.get("u")
    if ua is None:
        eig, vec = np.linalg.eigh(a["P"])
        ua = vec[:, eig > 0.5]
    if ub is None:
        eig, vec = np.linalg.eigh(b["P"])
        ub = vec[:, eig > 0.5]
    singular = np.linalg.svd(ua.T @ ub, compute_uv=False)
    return [float(math.degrees(math.acos(float(np.clip(v, -1.0, 1.0))))) for v in singular]


def training_evidence_audit(audit):
    historical = historical_prototype_module()
    pcdca_cfg = json.loads((PCDCA / "config.json").read_text())
    per_fold = {}
    incomplete_folds = {}
    all_source = source_hashes()
    for fold in range(10):
        fold_json = BASE / f"stage_a_fold_{fold}.json"
        attempt_json = BASE / f"stage_a_fold_{fold}.attempt.json"
        if fold_json.exists():
            row = json.loads(fold_json.read_text())
            expected_sha = audit["anchors"][str(fold)]["d4_best_sha256"]
            if row.get("D4_best_checkpoint_sha256") != expected_sha or row.get("native_forward_count") != 1:
                raise AssertionError(f"Invalid persisted Stage A fold result: {fold}")
            per_fold[str(fold)] = row
            continue
        if attempt_json.exists():
            attempt = json.loads(attempt_json.read_text())
            incomplete_folds[str(fold)] = {
                "status": attempt.get("status", "incomplete"),
                "native_forward_count": attempt.get("native_forward_count", 1),
                "embedding_persisted": attempt.get("embedding_persisted", False),
                "failure": attempt.get("failure"),
                "retry_forbidden": True,
            }
            continue
        loaded = Loaded(fold, 0, audit, train_only_labels=True)
        train = np.asarray(loaded.split["train"], dtype=np.int64)
        captures = {"z_img_train": None}
        hook = loaded.core.img_unet.register_forward_hook(
            lambda _module, _args, output: captures.__setitem__(
                "z_img_train", output[train].detach().clone()))
        try:
            path = PCDCA / "runs" / "D4" / f"s911_f{fold}.pt"
            best = load_torch(path, artifact=True)
            if (best["group"], best["seed"], best["fold"], best["state"]) != ("D4", 911, fold, "complete"):
                raise AssertionError(f"Unexpected historical D4 checkpoint: {path}")
            import_core(loaded.core, best["core"])
            loaded.core.eval()
            diag_seed = private_rng(911, fold)
            private_seed_rng(diag_seed)
            diag_rng_start_sha = payload_hash(capture_rng())
            attempt = {"fold": fold, "status": "native_forward_started", "utc": utc(),
                       "D4_best_checkpoint": str(path),
                       "D4_best_checkpoint_sha256": audit["anchors"][str(fold)]["d4_best_sha256"],
                       "diagnostic_private_rng_seed": diag_seed}
            attempt_json.write_text(json.dumps(attempt, indent=2) + "\n")
            with torch.set_grad_enabled(False):
                _logits, _graph = loaded.forward()
            diag_rng_after_sha = payload_hash(capture_rng())
            if loaded.forward_count != 1 or captures["z_img_train"] is None:
                raise AssertionError("Stage A must run exactly one D4 native forward")
            z = captures["z_img_train"].cpu().numpy().astype(np.float64)
            labels = loaded.diagnostic_labels
            labels_train = labels[train]
            meta_train = {key: value[train] for key, value in loaded.meta.items()}
            d, cond_stats = encode_conditions(meta_train, np.arange(len(train), dtype=np.int64))
            prepared = historical.prepare(labels, loaded.meta, train, 911, fold, pcdca_cfg)
            if not np.array_equal(prepared["train"], train):
                raise AssertionError("D4 private fold order changed")
            private = np.asarray(prepared["folds"], dtype=np.int64)
            local = np.arange(len(train), dtype=np.int64)
            site_train = meta_train["site"]
            age_train = meta_train["age"]
            sex_train = meta_train["sex"]
            obs_rows, null_values = [], []
            fold_info = []
            directions = []
            null_condition_stats = []

            def evaluate_condition(condition):
                rows, sub_info = [], []
                for group in range(3):
                    q_local = np.flatnonzero(private == group)
                    r_local = np.flatnonzero(private != group)
                    if len(np.unique(labels_train[r_local])) != 2:
                        raise ValueError(f"D4 private reference missing class, fold={fold}, private={group}")
                    result = predict_reference_delta(z, r_local, q_local, labels_train,
                                                     condition, site_train, CFG)
                    rows.append({"query": q_local, "target": diagnosis_target(labels_train, q_local),
                                 "base": result["pred_base"], "interaction": result["pred_interaction"],
                                 "mse_base": result["mse_base"], "mse_interaction": result["mse_interaction"],
                                 "delta_mse": result["delta_mse"]})
                    sub_info.append({"private_fold": group, "n_query": int(len(q_local)),
                                     "n_reference": int(len(r_local)), "MSE_Base": result["mse_base"],
                                     "MSE_Interaction": result["mse_interaction"],
                                     "Delta_MSE": result["delta_mse"], "rank": result["rank"],
                                     "singular_values": result["singular_values"]})
                y_all = np.concatenate([row["target"] for row in rows])
                base_all = np.concatenate([row["base"] for row in rows])
                inter_all = np.concatenate([row["interaction"] for row in rows])
                return {"MSE_Base": float(np.mean((y_all - base_all) ** 2)),
                        "MSE_Interaction": float(np.mean((y_all - inter_all) ** 2)),
                        "Delta_MSE": float(np.mean((y_all - base_all) ** 2) - np.mean((y_all - inter_all) ** 2)),
                        "subfolds": sub_info, "rows": rows}

            observed = evaluate_condition(d)
            obs_rows = observed["rows"]
            for group in range(3):
                ref = local[private != group]
                main = main_effect_projector(z[ref], d[ref], labels_train[ref], site_train[ref], CFG)
                inter = interaction_projector(z[ref], d[ref], labels_train[ref], site_train[ref], CFG)
                angle = None
                if main["rank"] and inter["rank"]:
                    u1 = np.asarray(main["u1"])
                    amount = float(np.linalg.norm(inter["P"] @ u1))
                    angle = float(math.degrees(math.acos(float(np.clip(amount, 0.0, 1.0)))))
                directions.append({"private_fold": group, "rank_interaction": inter["rank"],
                                   "rank_main": main["rank"], "interaction_singular_values": inter["singular_values"],
                                   "angle_interaction_to_disease_main_degrees": angle,
                                   "P_interaction": inter["P"], "P_main": main["P"],
                                   "B_interaction": inter["B_int"], "u_interaction": inter["u"]})

            for replicate in range(CFG["audit_null_replicates"]):
                null_seed = private_seed(911 + 1009 * (replicate + 1), fold, "stage-a-null")
                perm, changed = shuffle_conditions(local, site_train, private,
                                                   age_train, sex_train, null_seed)
                d_null = d[perm]
                null_result = evaluate_condition(d_null)
                null_values.append(null_result["Delta_MSE"])
                null_condition_stats.append(changed)

            direction_angles = []
            for left in range(3):
                for right in range(left + 1, 3):
                    a = {"rank": directions[left]["rank_interaction"],
                         "P": directions[left]["P_interaction"],
                         "u": directions[left]["u_interaction"]}
                    b = {"rank": directions[right]["rank_interaction"],
                         "P": directions[right]["P_interaction"],
                         "u": directions[right]["u_interaction"]}
                    angles = projector_angle_degrees(a, b)
                    if angles is not None:
                        direction_angles.extend(angles)
            fold_info = {"fold": fold, "D4_best_checkpoint": str(path),
                         "D4_best_checkpoint_sha256": audit["anchors"][str(fold)]["d4_best_sha256"],
                         "diagnostic_private_rng_seed": diag_seed,
                         "diagnostic_private_rng_start_sha256": diag_rng_start_sha,
                         "diagnostic_private_rng_after_forward_sha256": diag_rng_after_sha,
                         "native_forward_count": loaded.forward_count,
                         "z_img_shape_extracted": list(captures["z_img_train"].shape),
                         "outer_train_only": True, "outer_val_test_embeddings_excluded": True,
                         "outer_val_test_labels_excluded": True,
                         "condition_stats": cond_stats,
                         "true": {key: observed[key] for key in ("MSE_Base", "MSE_Interaction", "Delta_MSE", "subfolds")},
                         "null": {"replicates": len(null_values), "mean": float(np.mean(null_values)),
                                  "P95": float(np.quantile(null_values, 0.95)),
                                  "min": float(np.min(null_values)), "max": float(np.max(null_values)),
                                  "values": null_values,
                                  "true_exceeds_P95": bool(observed["Delta_MSE"] > np.quantile(null_values, .95)),
                                  "true_minus_null_mean": float(observed["Delta_MSE"] - np.mean(null_values)),
                                  "condition_change_mean": float(np.mean([x["joint_value_change_fraction"] for x in null_condition_stats])),
                                  "condition_change_min": float(np.min([x["joint_value_change_fraction"] for x in null_condition_stats])),
                                  "condition_change_by_replicate": null_condition_stats},
                         "directions": [{k: v for k, v in row.items() if not isinstance(v, np.ndarray)}
                                        for row in directions],
                         "direction_stability": {"pairwise_principal_angle_mean_degrees":
                                                  float(np.mean(direction_angles)) if direction_angles else None,
                                                  "pairwise_principal_angle_max_degrees":
                                                  float(np.max(direction_angles)) if direction_angles else None,
                                                  "inter_to_main_angles_degrees": [x["angle_interaction_to_disease_main_degrees"] for x in directions]},
                         "private_fold_site_null_changed": null_condition_stats[0]}
            # Persist numeric bases and sufficient statistics, not participant embeddings.
            np.savez_compressed(BASE / f"stage_a_f{fold}.npz",
                                P_interaction=np.stack([x["P_interaction"] for x in directions]),
                                P_main=np.stack([x["P_main"] for x in directions]),
                                B_interaction=np.stack([x["B_interaction"] for x in directions]),
                                true_delta_mse=np.asarray(observed["Delta_MSE"]))
            fold_json.write_text(json.dumps(fold_info, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
            per_fold[str(fold)] = fold_info
        finally:
            hook.remove()
            loaded.close()
        for name, old_hash in all_source.items():
            if sha_file(ROOT / name) != old_hash:
                raise AssertionError(f"Original source changed during Stage A: {name}")
    for path, expected in audit["historical_inputs_sha256"].items():
        if sha_file(path) != expected:
            raise AssertionError(f"Historical read-only input changed during Stage A: {path}")

    valid_folds = sorted(per_fold, key=int)
    delta = np.asarray([per_fold[f]["true"]["Delta_MSE"] for f in valid_folds], dtype=np.float64)
    above = np.asarray([per_fold[f]["null"]["true_exceeds_P95"] for f in valid_folds], dtype=bool)
    complete = len(valid_folds) == 10 and not incomplete_folds
    return {"status": "completed" if complete else "incomplete", "diagnostic_only": True,
            "folds": per_fold, "incomplete_folds": incomplete_folds,
            "summary": {"successful_folds": len(valid_folds),
                        "incomplete_folds": sorted(incomplete_folds, key=int),
                        "positive_Delta_MSE_folds": int(np.sum(delta > 0)),
                        "true_above_null_P95_folds": int(np.sum(above)),
                        "Delta_MSE_median_completed_folds_only": float(np.median(delta)) if len(delta) else None,
                        "Delta_MSE_mean_completed_folds_only": float(np.mean(delta)) if len(delta) else None,
                        "null_replicates_per_fold": CFG["audit_null_replicates"],
                        "ten_fold_scientific_gate_evaluable": bool(complete),
                        "does_not_gate_training_by_effect_size": True,
                        "does_gate_on_integrity_and_legal_reference_support": True},
            "uses_outer_test_metrics_for_design": False, "historical_D4_retrained": False}


def main():
    base_path = BASE / "audit.json"
    if base_path.exists():
        audit = json.loads(base_path.read_text())
        if audit.get("formal_authorized"):
            raise RuntimeError("Audit is frozen; do not rerun bootstrap")
    else:
        audit = make_audit_record()
        audit["math_tests"] = __import__("test_peim").run_tests()
        if not audit["math_tests"]["passed"]:
            raise AssertionError("Synthetic PEIM mathematical tests failed")
        (BASE / "math_tests.json").write_text(json.dumps(audit["math_tests"], indent=2) + "\n")
        base_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    evidence_path = BASE / "evidence_audit.json"
    if evidence_path.exists():
        raise RuntimeError("Stage A evidence audit already exists; refusing a second historical forward")
    evidence = training_evidence_audit(audit)
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    audit["stage_a"] = {"path": str(evidence_path), "sha256": sha_file(evidence_path),
                        "summary": evidence["summary"], "one_read_only_D4_forward_per_fold": True,
                        "incomplete_folds": evidence["incomplete_folds"]}
    audit["status"] = "STAGE_A_COMPLETE" if evidence["status"] == "completed" else "STAGE_A_INCOMPLETE_STOP"
    audit["formal_authorized"] = False
    base_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(evidence["summary"], indent=2))


if __name__ == "__main__":
    main()
