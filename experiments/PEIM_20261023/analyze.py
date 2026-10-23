"""Independent metric, fold, OOF, mechanism, and stability audit from saved outputs."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

from runtime import (BASE, CFG, HISTORY, PCDCA, ROOT, array_hash, load_torch,
                     metrics, result_path, sha_file, utc)

PCDPAX = ROOT / "experiments/PCDPAX_20261021"
IAFILM = Path("/tmp/IAFiLM_20261023")
METRICS = ("ACC", "SEN-ASD", "SPE-TD", "probability_AUC")


def summarize(folds):
    return {metric: {"mean": float(np.mean([row[metric] for row in folds])),
                     "sample_sd": float(np.std([row[metric] for row in folds], ddof=1)),
                     "formatted_percent": f"{100*np.mean([row[metric] for row in folds]):.2f} +/- "
                                         f"{100*np.std([row[metric] for row in folds], ddof=1):.2f}%"}
            for metric in METRICS}


def hash_payload(value):
    from runtime import payload_hash
    return payload_hash(value)


def load_peim(group, fold, audit):
    path = result_path(group, fold)
    saved = load_torch(path, artifact=True)
    if saved.get("state") != "complete" or saved.get("group") != group or saved.get("fold") != fold:
        raise AssertionError(f"Invalid PEIM task artifact: {path}")
    if saved["code_sha256"] != audit["training_code_sha256"]:
        raise AssertionError(f"PEIM task source SHA mismatch: {path}")
    if saved["formal_test_forwards"] != 1:
        raise AssertionError(f"Formal task used an unexpected test forward count: {path}")
    oof = saved["oof"]
    logits = np.asarray(oof["logits"], dtype=np.float64)
    p_cpu = torch.softmax(torch.as_tensor(logits, dtype=torch.float64), dim=1).numpy()[:, 0]
    max_probability_difference = float(np.max(np.abs(p_cpu - oof["probability"])))
    if max_probability_difference > 2e-6:
        raise AssertionError(f"Saved PPU/CPU probability discrepancy too large: {path}")
    computed = metrics(oof["labels"], oof["probability"], oof["prediction"])
    if max(abs(computed[key] - saved["metrics"][key]) for key in METRICS) > 1e-12:
        raise AssertionError(f"Saved official metric does not recompute: {path}")
    row = {"group": group, "seed": 911, "fold": fold, **computed,
           "best_epoch": int(saved["best_epoch"]), "best_val_loss": float(saved["best_val_loss"]),
           "training_seconds": float(saved["training_seconds"]),
           "probability_cpu_ppu_max_abs": max_probability_difference,
           "artifact_sha256": sha_file(path)}
    return saved, row


def load_old_checkpoint(path, expected_sha, group, fold, wrapper):
    if sha_file(path) != expected_sha:
        raise AssertionError(f"Historical artifact SHA mismatch: {path}")
    value = torch.load(path, map_location="cpu", weights_only=False)
    if wrapper:
        if hash_payload(value["payload"]) != value["content_sha256"]:
            raise AssertionError(f"Historical payload hash mismatch: {path}")
        value = value["payload"]
    if value.get("state") != "complete" or value.get("group") != group or value.get("fold") != fold:
        raise AssertionError(f"Unexpected historical checkpoint: {path}")
    return value


def oof_from_checkpoint(saved, group):
    oof = saved["oof"]
    idx = np.asarray(oof["indices"], dtype=np.int64)
    labels = np.asarray(oof["labels"], dtype=np.int64)
    probability = np.asarray(oof["probability"], dtype=np.float64)
    prediction = np.asarray(oof["prediction"], dtype=np.int64)
    logits = np.asarray(oof["logits"], dtype=np.float64)
    ids = [str(item) for item in oof["ids"]]
    return {"group": group, "subject_id": ids, "index": idx, "fold": np.full(len(idx), saved["fold"]),
            "label": labels, "probability_asd": probability, "prediction": prediction,
            "logit_asd": logits[:, 0], "logit_td": logits[:, 1]}


def combine_rows(rows):
    names = rows[0].keys()
    return {key: np.concatenate([np.asarray(row[key]) for row in rows]) for key in names}


def validate_oof(row, ctx, splits):
    n = len(ctx["ids"])
    if len(row["index"]) != n or len(set(row["subject_id"])) != n:
        raise AssertionError(f"OOF does not contain each subject once: {row['group']}")
    if not np.array_equal(np.sort(row["index"]), np.arange(n)):
        raise AssertionError(f"OOF index coverage failed: {row['group']}")
    labels_by_index = np.empty(n, dtype=np.int64)
    ids_by_index = np.empty(n, dtype=object)
    labels_by_index[row["index"]] = row["label"]
    ids_by_index[row["index"]] = row["subject_id"]
    if not np.array_equal(labels_by_index, ctx["labels"]):
        raise AssertionError(f"OOF labels do not match ABIDE context: {row['group']}")
    if not np.array_equal(ids_by_index.astype(str), np.asarray(ctx["ids"]).astype(str)):
        raise AssertionError(f"OOF subject IDs do not match ABIDE context: {row['group']}")
    if np.bincount(row["label"], minlength=2).tolist() != [403, 468]:
        raise AssertionError(f"OOF label balance failed: {row['group']}")
    if not np.isfinite(row["probability_asd"]).all() or np.any((row["probability_asd"] < 0) | (row["probability_asd"] > 1)):
        raise AssertionError(f"OOF probability invalid: {row['group']}")
    for fold in range(10):
        if not np.array_equal(np.sort(row["index"][row["fold"] == fold]), np.sort(splits[fold]["test"])):
            raise AssertionError(f"OOF fold/test membership mismatch: {row['group']} fold={fold}")


def csv_oof(path, group):
    with open(path, newline="") as stream:
        records = [row for row in csv.DictReader(stream) if row["group"] == group]
    if len(records) != 871:
        raise AssertionError(f"Expected 871 historical OOF rows in {path} for {group}")
    values = {"group": group,
              "subject_id": np.asarray([row["subject_id"] for row in records]),
              "index": np.asarray([int(row["index"]) for row in records]),
              "fold": np.asarray([int(row["fold"]) for row in records]),
              "label": np.asarray([int(row["label"]) for row in records]),
              "probability_asd": np.asarray([float(row["probability_asd"]) for row in records]),
              "prediction": np.asarray([int(row["prediction"]) for row in records])}
    values["logit_asd"] = np.full(871, np.nan)
    values["logit_td"] = np.full(871, np.nan)
    return values


def row_metrics(row):
    return metrics(row["label"], row["probability_asd"], row["prediction"])


def comparisons(fold_rows):
    pairs = {"E2-E1": ("E2", "E1"), "E3-E2": ("E3", "E2"), "E3-E4": ("E3", "E4"),
             "E3-historical_F3": ("E3", "historical_F3"),
             "historical_F3-historical_F4": ("historical_F3", "historical_F4"),
             "E3-historical_D4": ("E3", "historical_D4"),
             "E3-historical_M1": ("E3", "historical_M1"),
             "E3-historical_M3": ("E3", "historical_M3"),
             "E4-historical_F4": ("E4", "historical_F4")}
    out = {}
    for name, (left, right) in pairs.items():
        left_rows, right_rows = fold_rows[left], fold_rows[right]
        result = {}
        for metric in METRICS:
            delta = np.asarray([a[metric] - b[metric] for a, b in zip(left_rows, right_rows)]) * 100
            loo = [float(np.mean(np.delete(delta, index))) for index in range(10)]
            result[metric] = {"delta_mean_pp": float(delta.mean()),
                              "delta_sample_sd_pp": float(delta.std(ddof=1)),
                              "positive_folds": int(np.sum(delta > 0)),
                              "fold_deltas_pp": delta.tolist(),
                              "leave_one_fold_out_mean_range_pp": [float(min(loo)), float(max(loo))],
                              "fold2_delta_pp": float(delta[2]),
                              "mean_without_fold2_pp": float(np.delete(delta, 2).mean()),
                              "fold2_share_of_sum_signed_deltas": float(delta[2] / delta.sum()) if abs(delta.sum()) > 1e-12 else None}
        out[name] = result
    return out


def patient_changes(left, right):
    # Meaning is directional: count when the left group fixes or introduces an error.
    lookup = {sid: i for i, sid in enumerate(right["subject_id"])}
    if set(lookup) != set(left["subject_id"]):
        raise AssertionError("Patient IDs differ across OOF comparison")
    order = np.asarray([lookup[sid] for sid in left["subject_id"]])
    if not np.array_equal(left["label"], right["label"][order]):
        raise AssertionError("Patient labels differ across OOF comparison")
    if not np.array_equal(left["fold"], right["fold"][order]):
        raise AssertionError("Patient folds differ across OOF comparison")
    left_correct = left["prediction"] == left["label"]
    right_correct = right["prediction"][order] == right["label"][order]
    fixed = (~right_correct) & left_correct
    newly_wrong = right_correct & (~left_correct)
    labels = left["label"]
    return {"left_group": left["group"], "right_group": right["group"],
            "left_corrects_right_error": int(fixed.sum()),
            "left_introduces_error_vs_right": int(newly_wrong.sum()),
            "net_correct_change": int(fixed.sum() - newly_wrong.sum()),
            "ASD_net_correct_change": int(np.sum(fixed & (labels == 0)) - np.sum(newly_wrong & (labels == 0))),
            "TD_net_correct_change": int(np.sum(fixed & (labels == 1)) - np.sum(newly_wrong & (labels == 1))),
            "prediction_disagreement": int(np.sum(left["prediction"] != right["prediction"][order]))}


def synthetic_patient_change_test():
    left = {"group": "E3", "subject_id": np.asarray(["a", "b", "c", "d", "e", "f"]),
            "fold": np.asarray([0, 0, 1, 1, 2, 2]), "label": np.asarray([0, 0, 1, 1, 0, 1]),
            "prediction": np.asarray([0, 1, 1, 0, 1, 1])}
    right = {"group": "E4", "subject_id": np.asarray(["a", "b", "c", "d", "e", "f"]),
             "fold": np.asarray([0, 0, 1, 1, 2, 2]), "label": np.asarray([0, 0, 1, 1, 0, 1]),
             "prediction": np.asarray([1, 0, 1, 1, 0, 0])}
    result = patient_changes(left, right)
    assert result["left_corrects_right_error"] == 2
    assert result["left_introduces_error_vs_right"] == 3
    assert result["net_correct_change"] == -1
    assert result["ASD_net_correct_change"] == -1 and result["TD_net_correct_change"] == 0
    return {"passed": True, **result}


def csv_rows(rows, path, columns):
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_history(group, ctx):
    result = []
    if group == "historical_D4":
        summary = json.loads((PCDCA / "summary.json").read_text())
        for fold in range(10):
            path = PCDCA / "runs/D4" / f"s911_f{fold}.pt"
            expected = summary["artifact_sha256"][f"D4/911/{fold}"]
            saved = load_old_checkpoint(path, expected, "D4", fold, wrapper=True)
            result.append(oof_from_checkpoint(saved, group))
    elif group in ("historical_M1", "historical_M3"):
        old_group = group.removeprefix("historical_")
        summary = json.loads((PCDPAX / "summary.json").read_text())
        for fold in range(10):
            path = PCDPAX / "runs" / old_group / f"s911_f{fold}.pt"
            expected = summary["artifact_sha256"][f"{old_group}/f{fold}"]
            saved = load_old_checkpoint(path, expected, old_group, fold, wrapper=False)
            result.append(oof_from_checkpoint(saved, group))
    elif group in ("historical_F3", "historical_F4"):
        ia = csv_oof(IAFILM / "oof.csv", group.removeprefix("historical_"))
        # The experiment-25 CSV does not contain logits; its stored ASD probabilities are authoritative.
        by_fold = []
        for fold in range(10):
            mask = ia["fold"] == fold
            by_fold.append({key: np.asarray(value)[mask] for key, value in ia.items()
                            if key not in ("group",)})
        return ia, by_fold
    else:
        raise ValueError(group)
    combined = combine_rows(result)
    fold_rows = []
    for fold in range(10):
        mask = combined["fold"] == fold
        one = {key: np.asarray(value)[mask] for key, value in combined.items() if key != "group"}
        fold_rows.append({"fold": fold, **metrics(one["label"], one["probability_asd"], one["prediction"])})
    return combined, fold_rows


def performance_mechanisms(peim_payloads, stage_a):
    groups = {}
    starts = {fold: {"train_start_sha256": set(), "train_start_rng_sha256": set(),
                     "input_digest": set(), "initial_W_d_sha256": set(),
                     "epoch0": {"train_ce": [], "graph": [], "d4": []}}
               for fold in range(10)}
    max_memory = {"0": 0, "1": 0}
    for group in CFG["groups"]:
        fold_mechs = []
        for fold in range(10):
            saved = peim_payloads[(group, fold)]
            history = saved["epochs"]
            if not history:
                raise AssertionError("PEIM checkpoint lacks training audit trail")
            best = saved["best_record"]
            best_module = saved["interaction"]["W_d"].detach().cpu().numpy()
            train_mod = float(best.get("modulation_relative_l2_mean", 0.0))
            train_gate_hi = best.get("train_high_confidence_relative_l2")
            train_gate_lo = best.get("train_low_confidence_relative_l2")
            fold_mechs.append({"fold": fold, "best_epoch": saved["best_epoch"],
                               "epochs_run": saved["epochs_run"], "best_val_loss": saved["best_val_loss"],
                               "W_d_norm": float(np.linalg.norm(best_module)),
                               "training_modulation_relative_l2": train_mod,
                               "training_gate_mean": best.get("train_gate_mean"),
                               "train_img_weight": best.get("train_img_weight"),
                               "train_ph_weight": best.get("train_ph_weight"),
                               "train_graph_at_best": best.get("train_graph"),
                               "val_graph_at_best": best.get("val_graph"),
                               "high_confidence_modulation_relative_l2": train_gate_hi,
                               "low_confidence_modulation_relative_l2": train_gate_lo,
                               "high_confidence_count": best.get("train_high_confidence_count"),
                               "low_confidence_count": best.get("train_low_confidence_count"),
                               "w_gradient_norm_mean": float(np.mean([x["w_gradient_norm"] for x in history])),
                               "w_gradient_norm_at_best": float(best.get("w_gradient_norm", 0.0)),
                               "D4_gradient_budget_ratio_mean": float(np.mean([x["d4_weighted_gradient_ratio"] for x in history])),
                               "inactive_evidence_fraction_mean": float(np.mean([x["evidence_inactive_fraction"] for x in history])),
                               "mean_evidence_fit_seconds_per_epoch": float(np.mean([x["evidence_seconds"] for x in history])),
                               "peak_allocated_bytes": saved["peak_allocated_bytes"],
                               "peak_reserved_bytes": saved["peak_reserved_bytes"],
                               "physical_gpu": saved["physical_gpu"],
                               "initial_W_d_sha256": saved["initial_W_d_sha256"],
                               "test_modulation_relative_l2": saved["test_modulation"]["relative_l2_mean"],
                               "test_gate_mean": saved["test_modulation"]["gate_mean"],
                               "test_img_weight": saved["test_img_weight"],
                               "test_ph_weight": saved["test_ph_weight"],
                               "test_cap_fraction": saved["test_modulation"]["cap_fraction"],
                               "training_seconds": saved["training_seconds"],
                               "input_digest": saved["input_digest"],
                               "train_start_sha256": saved["train_start_sha256"],
                               "train_start_rng_sha256": saved["train_start_rng_sha256"],
                               "raw_z_img_test_sha256": saved["test_raw_image_sha256"],
                               "raw_z_ph_test_sha256": saved["test_raw_ph_sha256"]})
            starts[fold]["train_start_sha256"].add(saved["train_start_sha256"])
            starts[fold]["train_start_rng_sha256"].add(saved["train_start_rng_sha256"])
            starts[fold]["input_digest"].add(saved["input_digest"])
            starts[fold]["initial_W_d_sha256"].add(saved["initial_W_d_sha256"])
            starts[fold]["epoch0"]["train_ce"].append(history[0]["train_native_ce"])
            starts[fold]["epoch0"]["graph"].append(history[0]["train_graph"])
            starts[fold]["epoch0"]["d4"].append(history[0]["train_d4"])
            max_memory[str(saved["physical_gpu"])] = max(max_memory[str(saved["physical_gpu"])],
                                                          int(saved["peak_reserved_bytes"]))
        groups[group] = fold_mechs

    stage_fold = stage_a["folds"]
    delta = [row["true"]["Delta_MSE"] for row in stage_fold.values()]
    true_above = [row["null"]["true_exceeds_P95"] for row in stage_fold.values()]
    evidence_summary = {"true_Delta_MSE_positive_folds": int(sum(x > 0 for x in delta)),
                        "true_Delta_MSE_exceeds_null_P95_folds": int(sum(true_above)),
                        "true_Delta_MSE_mean": float(np.mean(delta)),
                        "true_Delta_MSE_median": float(np.median(delta)),
                        "true_Delta_MSE_by_fold": delta,
                        "null_mean_by_fold": [stage_fold[str(i)]["null"]["mean"] for i in range(10)],
                        "null_P95_by_fold": [stage_fold[str(i)]["null"]["P95"] for i in range(10)],
                        "interaction_rank_by_private_reference": {
                            str(fold): [item["rank_interaction"] for item in stage_fold[str(fold)]["directions"]]
                            for fold in range(10)},
                        "direction_stability_by_fold": {
                            str(fold): stage_fold[str(fold)]["direction_stability"] for fold in range(10)}}

    # Check same-fold fairness across all four complete restarts.
    epoch0_spread = {}
    for fold in range(10):
        for key in ("train_start_sha256", "train_start_rng_sha256", "input_digest", "initial_W_d_sha256"):
            if len(starts[fold][key]) != 1:
                raise AssertionError(f"Same-fold initialization mismatch for {key}: fold {fold}")
        epoch0_spread[str(fold)] = {key: float(np.ptp(values))
                                    for key, values in starts[fold]["epoch0"].items()}
    return {"groups": groups, "stage_a_evidence": evidence_summary,
            "epoch0_same_fold_group_spread_by_fold": epoch0_spread,
            "all_W_d_initial_sha_identical_per_fold": True,
            "maximum_peak_reserved_bytes_by_gpu": max_memory}


def candidate_judgment(stage, comparison, mechanisms):
    criteria = CFG["candidate_go_criteria_frozen_before_training"]
    e2e1 = comparison["E2-E1"]
    e3e2 = comparison["E3-E2"]
    e3e4 = comparison["E3-E4"]
    e3d4 = comparison["E3-historical_D4"]
    e3m3 = comparison["E3-historical_M3"]
    stage_pass = (stage["summary"]["Delta_MSE_median"] > 0 and
                  stage["summary"]["true_above_null_P95_folds"] >= 6)
    e2e1_pass = (e2e1["probability_AUC"]["delta_mean_pp"] > 0 and
                 e2e1["probability_AUC"]["positive_folds"] >= 6 and
                 e2e1["probability_AUC"]["leave_one_fold_out_mean_range_pp"][0] > 0)
    e3e2_pass = (e3e2["SEN-ASD"]["delta_mean_pp"] >= -1.0 and
                 e3e2["probability_AUC"]["delta_mean_pp"] >= -0.25 and
                 max(e3e2["ACC"]["positive_folds"], e3e2["probability_AUC"]["positive_folds"]) >= 6)
    e3e4_auc = e3e4["probability_AUC"]
    e3e4_pass = (e3e4_auc["delta_mean_pp"] > 0 and e3e4_auc["positive_folds"] >= 6 and
                 e3e4_auc["leave_one_fold_out_mean_range_pp"][0] > 0 and
                 e3e4_auc["mean_without_fold2_pp"] > 0)
    history_pass = (e3d4["ACC"]["delta_mean_pp"] >= 0 and e3d4["probability_AUC"]["delta_mean_pp"] >= 0 and
                    e3m3["ACC"]["delta_mean_pp"] >= 0 and e3m3["probability_AUC"]["delta_mean_pp"] >= 0 and
                    e3m3["SEN-ASD"]["delta_mean_pp"] >= -1.0)
    e3_mechs = mechanisms["groups"]["E3"]
    nontrivial_folds = sum(row["W_d_norm"] > 1e-6 and row["training_modulation_relative_l2"] > 1e-5
                           for row in e3_mechs)
    mechanism_pass = nontrivial_folds >= 8
    checks = {"stage_a_interaction_evidence": stage_pass,
              "E2_vs_E1_condition_interaction_direction": e2e1_pass,
              "E3_vs_E2_selective_protection": e3e2_pass,
              "E3_vs_E4_correct_patient_condition": e3e4_pass,
              "E3_vs_D4_and_M3_context": history_pass,
              "E3_nontrivial_learned_modulation": mechanism_pass,
              "frozen_criteria": criteria}
    return {"status": "CANDIDATE_GO_FOR_CONFIRMATION" if all(v for k, v in checks.items()
                                                               if k != "frozen_criteria") else "NO_GO",
            "checks": checks,
            "interpretation": "Seed 911 has been used for design development; even a GO outcome only supports an independent confirmation study."}


def main():
    audit = json.loads((BASE / "audit.json").read_text())
    if not audit.get("formal_authorized"):
        raise RuntimeError("Formal training is not authorized")
    launch = json.loads((BASE / "launch.json").read_text())
    if launch["tasks_completed"] != 40:
        raise RuntimeError("Cannot report complete PEIM metrics before all 40 tasks complete")
    ctx = load_torch(HISTORY / "artifacts/context.pt")
    splits = [load_torch(HISTORY / "artifacts/seed_911" / f"fold_{fold}" / "splits.pt") for fold in range(10)]
    from audit import code_hashes, hash_json, source_hashes
    if hash_json(code_hashes()) != audit["training_code_sha256"]:
        raise AssertionError("Frozen experiment source SHA changed before analysis")
    if source_hashes() != audit["original_source_sha256"]:
        raise AssertionError("Original model source changed during PEIM training")
    if sha_file(BASE / "manifest.json") != audit["manifest_sha256"]:
        raise AssertionError("Frozen manifest SHA mismatch during analysis")
    for path, expected in audit["historical_inputs_sha256"].items():
        if sha_file(path) != expected:
            raise AssertionError(f"Historical read-only artifact changed during analysis: {path}")

    all_oof, fold_rows, peim_payloads = {}, {}, {}
    ppu_cpu_probability_differences = []
    for group in CFG["groups"]:
        payload_rows, metrics_rows = [], []
        for fold in range(10):
            saved, metric_row = load_peim(group, fold, audit)
            ppu_cpu_probability_differences.append(metric_row["probability_cpu_ppu_max_abs"])
            peim_payloads[(group, fold)] = saved
            payload_rows.append(oof_from_checkpoint(saved, group))
            metrics_rows.append({"fold": fold, **{key: metric_row[key] for key in METRICS},
                                 "best_epoch": metric_row["best_epoch"],
                                 "best_val_loss": metric_row["best_val_loss"],
                                 "training_seconds": metric_row["training_seconds"]})
        all_oof[group] = combine_rows(payload_rows)
        validate_oof(all_oof[group], ctx, splits)
        fold_rows[group] = metrics_rows

    history_fold_rows = {}
    for group in ("historical_D4", "historical_M1", "historical_M3"):
        all_oof[group], fold_rows[group] = load_history(group, ctx)
        validate_oof(all_oof[group], ctx, splits)
    for group in ("historical_F3", "historical_F4"):
        all_oof[group], fold_rows[group] = load_history(group, ctx)
        validate_oof(all_oof[group], ctx, splits)
    for group in all_oof:
        recomputed = row_metrics(all_oof[group])
        if group not in fold_rows or len(fold_rows[group]) != 10:
            raise AssertionError(f"Missing fold metrics for {group}")
    historical_summary = {group: summarize(rows) for group, rows in fold_rows.items()}
    comparison = comparisons(fold_rows)

    synthetic_changes = synthetic_patient_change_test()
    patient = {f"E3_vs_{group}": patient_changes(all_oof["E3"], all_oof[group])
               for group in ("E1", "E2", "E4", "historical_D4", "historical_M3", "historical_F3")}
    mechanisms = performance_mechanisms(peim_payloads, json.loads((BASE / "evidence_audit.json").read_text()))
    judgment = candidate_judgment(json.loads((BASE / "evidence_audit.json").read_text()), comparison, mechanisms)

    metrics_rows_csv = []
    for group, rows in fold_rows.items():
        for row in rows:
            metrics_rows_csv.append({"group": group, "seed": 911, **row})
    csv_rows(metrics_rows_csv, BASE / "fold_metrics.csv",
             ["group", "seed", "fold", *METRICS, "best_epoch", "best_val_loss", "training_seconds"])

    oof_rows = []
    for group in CFG["groups"]:
        values = all_oof[group]
        order = np.argsort(values["index"])
        for index in order:
            oof_rows.append({"group": group, "subject_id": values["subject_id"][index],
                             "index": int(values["index"][index]), "fold": int(values["fold"][index]),
                             "label": int(values["label"][index]),
                             "probability_asd": float(values["probability_asd"][index]),
                             "prediction": int(values["prediction"][index])})
    csv_rows(oof_rows, BASE / "oof.csv",
             ["group", "subject_id", "index", "fold", "label", "probability_asd", "prediction"])

    stage_a = json.loads((BASE / "evidence_audit.json").read_text())
    exec = {"launch": launch, "historical_retraining": {"D0": 0, "D4": 0, "M1": 0,
             "M3": 0, "experiment25": 0, "other": 0},
            "completed_tasks": 40, "gpu0_tasks": launch["tasks_by_gpu"]["0"],
            "gpu1_tasks": launch["tasks_by_gpu"]["1"],
            "training_seconds_sum": float(sum(payload["training_seconds"] for payload in peim_payloads.values())),
            "failure_count": len(launch["failures"]), "retry_count": launch["retries"],
            "maximum_peak_reserved_bytes_by_gpu": mechanisms["maximum_peak_reserved_bytes_by_gpu"]}
    summary = {"experiment": CFG["experiment"], "name": CFG["name"], "status": "COMPLETE",
               "completed_tasks": 40, "gpu0_tasks": exec["gpu0_tasks"], "gpu1_tasks": exec["gpu1_tasks"],
               "historical_retraining": exec["historical_retraining"],
               "performance": historical_summary, "comparisons": comparison,
               "patient_oof_changes": patient,
               "patient_change_synthetic_test": synthetic_changes,
               "mechanisms": mechanisms, "stage_a": stage_a["summary"],
               "candidate_judgment": judgment, "execution": exec,
               "formal_metric_definition": "ACC, SEN-ASD (label 0), SPE-TD (label 1), probability AUC from saved single-forward PPU p_ASD; fold mean +/- sample SD (ddof=1)",
               "PPU_CPU_probability_max_abs_difference": float(max(ppu_cpu_probability_differences)),
               "audit_sha256_before_finalize": sha_file(BASE / "audit.json"),
               "manifest_sha256": sha_file(BASE / "manifest.json"),
               "created_utc": utc()}
    (BASE / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n")

    report = render_report(summary, fold_rows, comparison, patient, mechanisms, stage_a)
    (BASE / "report.md").write_text(report)
    final_audit = json.loads((BASE / "audit.json").read_text())
    final_audit["status"] = "COMPLETE"
    final_audit["execution"].update({"completed": 40, "GPU0": exec["gpu0_tasks"],
                                    "GPU1": exec["gpu1_tasks"], "failed": launch["failures"],
                                    "retries": launch["retries"],
                                    "worker_seconds": exec["training_seconds_sum"],
                                    "wall_seconds": launch["wall_seconds"]})
    final_audit["analysis"] = {"summary_sha256": sha_file(BASE / "summary.json"),
                               "report_sha256": sha_file(BASE / "report.md"),
                               "fold_metrics_sha256": sha_file(BASE / "fold_metrics.csv"),
                               "oof_sha256": sha_file(BASE / "oof.csv"),
                               "task_checkpoint_sha256": {f"{group}/f{fold}": sha_file(result_path(group, fold))
                                                          for group in CFG["groups"] for fold in range(10)}}
    (BASE / "audit.json").write_text(json.dumps(final_audit, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": summary["status"], "candidate_judgment": judgment["status"],
                      "performance": {key: {m: value for m, value in stats.items()}
                                      for key, stats in historical_summary.items()},
                      "wall_seconds": launch["wall_seconds"], "worker_seconds": exec["training_seconds_sum"]},
                     ensure_ascii=False, indent=2))


def render_report(summary, fold_rows, comparison, patient, mechanisms, stage_a):
    lines = ["# PEIM Experiment 26", "",
             f"Status: `{summary['status']}`; candidate decision: `{summary['candidate_judgment']['status']}`.",
             "",
             "PEIM estimates a phenotype-by-image disease interaction after controlling image, phenotype, and site main effects. E1 uses the ordinary disease main-effect subspace; E2-E4 use the cross-fitted interaction subspace. E3 gates bounded multiplicative modulation by train-reference class-prototype distance ambiguity; E4 shuffles only the added age/sex pairing.",
             "",
             "## Stage A", "",
             "The diagnostic uses one read-only D4 best-checkpoint forward per fold and only outer-train image embeddings, labels, and age/sex/site for the fitted probes. It is diagnostic evidence, not a forced GO gate.", "",
             "| Fold | Delta MSE | Null mean | Null P95 | interaction rank (three private refs) | direction principal-angle mean |"]
    lines.append("|---:|---:|---:|---:|---|---:|")
    for fold in range(10):
        row = stage_a["folds"][str(fold)]
        ranks = [item["rank"] for item in row["true"]["subfolds"]]
        angle = row["direction_stability"]["pairwise_principal_angle_mean_degrees"]
        lines.append(f"| {fold} | {row['true']['Delta_MSE']:.6f} | {row['null']['mean']:.6f} | {row['null']['P95']:.6f} | {ranks} | {angle if angle is not None else 'NA'} |")
    lines += ["", f"Positive Delta_MSE folds: {stage_a['summary']['positive_Delta_MSE_folds']}/10; true above null P95: {stage_a['summary']['true_above_null_P95_folds']}/10; median: {stage_a['summary']['Delta_MSE_median']:.6f}.",
              "", "## Formal Metrics", "", "Mean +/- sample SD across the ten historical folds; only the four registered metrics are reported.",
              "", "| Group | ACC | SEN-ASD | SPE-TD | probability AUC |", "|---|---:|---:|---:|---:|"]
    for group in ("historical_D4", "historical_M1", "historical_M3", "historical_F3", "E1", "E2", "E3", "E4"):
        values = summary["performance"][group]
        lines.append(f"| {group} | {values['ACC']['formatted_percent']} | {values['SEN-ASD']['formatted_percent']} | {values['SPE-TD']['formatted_percent']} | {values['probability_AUC']['formatted_percent']} |")
    lines += ["", "## Paired Ablations", "", "Delta is left minus right in percentage points. Positive-fold counts and leave-one-fold-out mean ranges expose fold concentration.",
              "", "| Contrast | Metric | Mean delta pp | SD pp | Positive folds | LOO mean range pp | Fold2 delta pp | Without fold2 pp |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("E2-E1", "E3-E2", "E3-E4", "historical_F3-historical_F4",
                 "E3-historical_F3", "E3-historical_D4", "E3-historical_M1", "E3-historical_M3"):
        for metric in METRICS:
            value = comparison[name][metric]
            lines.append(f"| {name} | {metric} | {value['delta_mean_pp']:.3f} | {value['delta_sample_sd_pp']:.3f} | {value['positive_folds']}/10 | [{value['leave_one_fold_out_mean_range_pp'][0]:.3f}, {value['leave_one_fold_out_mean_range_pp'][1]:.3f}] | {value['fold2_delta_pp']:.3f} | {value['mean_without_fold2_pp']:.3f} |")
    lines += ["", "## Mechanism and Integrity", "", f"The non-symmetric synthetic patient-change audit passed: `{summary['patient_change_synthetic_test']['passed']}`. OOF transition counts are directional: E3 correcting a comparator error versus E3 introducing an error.", ""]
    for name, change in patient.items():
        lines.append(f"- {name}: fixes {change['left_corrects_right_error']}, introduces {change['left_introduces_error_vs_right']}, net correct {change['net_correct_change']} (ASD {change['ASD_net_correct_change']}, TD {change['TD_net_correct_change']}).")
    lines += ["", f"Formal tasks: {summary['completed_tasks']}/40, GPU0 {summary['gpu0_tasks']}/20, GPU1 {summary['gpu1_tasks']}/20. Historical retraining: 0. Wall time {summary['execution']['launch']['wall_seconds']:.1f}s; summed worker training time {summary['execution']['training_seconds_sum']:.1f}s; retries {summary['execution']['retry_count']}.",
              "", "The original graph, transductive cohort, cross-fold RFE history, VAE eval sampling, active eval dropout, RP special state and attention score gradient truncation were preserved. The graph loss can change indirectly because attention modality weights depend on the modified fusion input.",
              "", "## Frozen Candidate Checks", ""]
    for name, value in summary["candidate_judgment"]["checks"].items():
        if name != "frozen_criteria":
            lines.append(f"- {name}: {value}")
    lines += ["", summary["candidate_judgment"]["interpretation"], "",
              "Source references: [MM-GTUNets](https://arxiv.org/html/2406.14455v3), [RTGMFF](https://arxiv.org/abs/2509.03214), [FiLM](https://arxiv.org/abs/1709.07871).",
              ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
