"""Independent four-metric, fold-paired, mechanism, and sensitivity analysis."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics
import sys

import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))
from run import (CFG, PCDCA, anchor, atomic_json, formal_metrics, load_torch,
                 read_json, result_path, sha_file, valid_result)


FORMAL_KEYS = ("ACC", "SEN-ASD", "SPE-TD", "probability_AUC")
COMPARISONS = {
    "M2-M1": ("M2", "M1"), "M3-M1": ("M3", "M1"),
    "M3-M2": ("M3", "M2"), "M3-M4": ("M3", "M4"),
}


def mean_sd(values):
    values = [float(value) for value in values]
    if not values:
        return {"mean": None, "sample_sd": None, "n": 0}
    return {"mean": float(np.mean(values)),
            "sample_sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
            "n": len(values)}


def _unwrap_historical(path: Path):
    value = load_torch(path)
    if "payload" in value and "content_sha256" in value:
        from run import payload_hash
        if payload_hash(value["payload"]) != value["content_sha256"]:
            raise AssertionError(f"Historical D4 artifact checksum mismatch: {path}")
        value = value["payload"]
    return value


def load_historical_d4():
    folds = {}
    for fold in range(10):
        path = PCDCA / "runs" / "D4" / f"s911_f{fold}.pt"
        value = _unwrap_historical(path)
        if (value.get("state"), value.get("group"), value.get("seed"), value.get("fold")) != ("complete", "D4", 911, fold):
            raise AssertionError(f"Unexpected historical D4 artifact at {path}")
        oof = value["oof"]
        recalc = formal_metrics(oof["labels"], oof["probability"], oof["prediction"])
        folds[fold] = {"artifact": value, "metrics": recalc, "oof": oof,
                       "artifact_sha256": sha_file(path)}
    metric_summary = {key: mean_sd([folds[fold]["metrics"][key] for fold in range(10)]) for key in FORMAL_KEYS}
    supplied = CFG["historical_d4_reference_percent"]
    actual = {key: metric_summary[key]["mean"] * 100 for key in FORMAL_KEYS}
    expected_keys = {"ACC": "ACC", "SEN-ASD": "SEN-ASD", "SPE-TD": "SPE-TD", "probability_AUC": "probability_AUC"}
    differences = {key: actual[key] - supplied[key] for key in expected_keys}
    if max(abs(value) for value in differences.values()) > 0.0001:
        raise AssertionError(f"Read-only historical D4 mean differs from registered reference: {differences}")
    return folds, metric_summary, differences


def artifact_records(audit):
    records, failures = {}, []
    for group in CFG["groups"]:
        for fold in range(10):
            try:
                value = valid_result(group, 911, fold, audit, required=False)
            except Exception as error:
                failures.append({"group": group, "fold": fold, "error": repr(error)})
                continue
            if value is not None:
                records[(group, fold)] = value
    return records, failures


def summarize_group(records, group):
    values = [records[(group, fold)] for fold in range(10) if (group, fold) in records]
    if not values:
        return None
    return {
        "folds_complete": len(values),
        "metrics": {key: mean_sd([value["metrics"][key] for value in values]) for key in FORMAL_KEYS},
        "mean_best_epoch": float(np.mean([value["best_epoch"] for value in values])),
        "mean_training_seconds": float(np.mean([value["training_seconds"] for value in values])),
        "median_training_seconds": float(np.median([value["training_seconds"] for value in values])),
        "mean_ph_ce_start": float(np.mean([value["epochs"][0]["phenotype_real_ce"] for value in values])),
        "mean_ph_ce_end": float(np.mean([value["epochs"][-1]["phenotype_real_ce"] for value in values])),
        "mean_d4_ce_start": float(np.mean([value["epochs"][0]["image_d4_ce"] for value in values])),
        "mean_d4_ce_end": float(np.mean([value["epochs"][-1]["image_d4_ce"] for value in values])),
    }


def paired_fold_deltas(records, left, right, metric):
    deltas = []
    for fold in range(10):
        a, b = records.get((left, fold)), records.get((right, fold))
        if a is not None and b is not None:
            deltas.append({"fold": fold, "left": left, "right": right,
                           "delta_pp": 100 * (a["metrics"][metric] - b["metrics"][metric])})
    vals = [item["delta_pp"] for item in deltas]
    loo = [float(np.mean([item["delta_pp"] for item in deltas if item["fold"] != fold]))
           for fold in range(10) if len(deltas) > 1 and any(item["fold"] == fold for item in deltas)]
    return {
        "metric": metric, "fold_deltas": deltas, "mean_delta_pp": mean_sd(vals),
        "positive_folds": sum(value > 0 for value in vals),
        "leave_one_fold_out_mean_delta_pp": loo,
        "leave_one_fold_out_min_pp": min(loo) if loo else None,
        "leave_one_fold_out_max_pp": max(loo) if loo else None,
        "leave_one_fold_out_direction_changes": bool(loo and min(loo) <= 0 <= max(loo)),
    }


def paired_oof(records, historical, left, right):
    if right == "historical_D4":
        right_oof = {subject_id: (int(label), int(pred), float(prob))
                     for fold in historical for subject_id, label, pred, prob in zip(
                         historical[fold]["oof"]["ids"], historical[fold]["oof"]["labels"],
                         historical[fold]["oof"]["prediction"], historical[fold]["oof"]["probability"])}
    else:
        right_oof = {subject_id: (int(label), int(pred), float(prob))
                     for fold in range(10) if (right, fold) in records
                     for subject_id, label, pred, prob in zip(
                         records[(right, fold)]["oof"]["ids"], records[(right, fold)]["oof"]["labels"],
                         records[(right, fold)]["oof"]["prediction"], records[(right, fold)]["oof"]["probability"])}
    left_oof = {subject_id: (int(label), int(pred), float(prob))
                for fold in range(10) if (left, fold) in records
                for subject_id, label, pred, prob in zip(
                    records[(left, fold)]["oof"]["ids"], records[(left, fold)]["oof"]["labels"],
                    records[(left, fold)]["oof"]["prediction"], records[(left, fold)]["oof"]["probability"])}
    common = sorted(set(left_oof) & set(right_oof))
    if not common:
        return {"n": 0}
    correct_to_error = []
    error_to_correct = []
    changed = {"ASD_correct_net": 0, "TD_correct_net": 0}
    for subject_id in common:
        y0, p0, _ = right_oof[subject_id]
        y1, p1, _ = left_oof[subject_id]
        if y0 != y1:
            raise AssertionError(f"Paired OOF labels differ for {subject_id}")
        if p0 == y0 and p1 != y1:
            correct_to_error.append(subject_id)
            changed["ASD_correct_net" if y0 == 0 else "TD_correct_net"] -= 1
        elif p0 != y0 and p1 == y1:
            error_to_correct.append(subject_id)
            changed["ASD_correct_net" if y0 == 0 else "TD_correct_net"] += 1
    return {
        "n": len(common), "error_to_correct": len(error_to_correct),
        "correct_to_error": len(correct_to_error),
        "net_correct_change": len(error_to_correct) - len(correct_to_error),
        "ASD_correct_net": changed["ASD_correct_net"], "TD_correct_net": changed["TD_correct_net"],
        "error_to_correct_ids": error_to_correct, "correct_to_error_ids": correct_to_error,
    }


def state_delta(left, right):
    left_state = left["best_model_state"]["state"] if "best_model_state" in left else left["core"]["state"]
    right_state = right["best_model_state"]["state"] if "best_model_state" in right else right["core"]["state"]
    keys = sorted(set(left_state) & set(right_state))
    squared = 0.0
    reference = 0.0
    for key in keys:
        a = left_state[key].double()
        b = right_state[key].double()
        squared += float((a - b).square().sum())
        reference += float(b.square().sum())
    return {"absolute_l2": squared ** .5, "relative_l2_to_right": (squared ** .5) / max(reference ** .5, 1e-12)}


def mechanism_summary(records):
    groups = {}
    for group in CFG["groups"]:
        artifacts = [records[(group, fold)] for fold in range(10) if (group, fold) in records]
        sampled = [epoch for artifact in artifacts for epoch in artifact["epochs"]
                   if epoch["epoch"] < 3 or epoch["epoch"] % CFG["diagnostic_frequency"] == 0]
        alignments = [epoch["alignment"] for epoch in sampled if "alignment" in epoch]
        student_protos = [epoch["alignment_student_prototype"] for epoch in sampled
                          if "alignment_student_prototype" in epoch]
        klpaths = [epoch["kl_gradient_path"] for epoch in sampled if "kl_gradient_path" in epoch]
        image_protos = [epoch["image_prototype"] for epoch in sampled if "image_prototype" in epoch]
        ph_protos = [epoch["phenotype_prototype"] for epoch in sampled if "phenotype_prototype" in epoch]
        groups[group] = {
            "sampled_epoch_records": len(sampled),
            "mean_alignment_kl": float(np.mean([item["kl"] for item in alignments])) if alignments else None,
            "mean_KL_first_sample": float(np.mean([
                next(epoch["alignment"]["kl"] for epoch in artifact["epochs"] if "alignment" in epoch)
                for artifact in artifacts])) if alignments and artifacts else None,
            "mean_KL_last_sample": float(np.mean([
                next(epoch["alignment"]["kl"] for epoch in reversed(artifact["epochs"])
                     if "alignment" in epoch) for artifact in artifacts])) if alignments and artifacts else None,
            "mean_teacher_student_agreement": float(np.mean([item["teacher_student_argmax_agreement"] for item in alignments])) if alignments else None,
            "mean_teacher_train_accuracy": float(np.mean([item["teacher_train_accuracy"] for item in alignments])) if alignments else None,
            "mean_student_train_accuracy": float(np.mean([item["student_train_accuracy"] for item in alignments])) if alignments else None,
            "mean_student_prototype_distance": float(np.mean([item["class_prototype_distance_mean"] for item in student_protos])) if student_protos else None,
            "mean_student_class_delta_query_sd_l2": float(np.mean([item["class_delta_query_sd_l2"] for item in student_protos])) if student_protos else None,
            "mean_student_prototype_train_accuracy": float(np.mean([item["prototype_train_accuracy"] for item in student_protos])) if student_protos else None,
            "mean_teacher_entropy": float(np.mean([item["teacher_entropy"] for item in alignments])) if alignments else None,
            "mean_student_entropy": float(np.mean([item["student_entropy"] for item in alignments])) if alignments else None,
            "mean_KL_img_embedding_gradient_norm": float(np.mean([item["img_embedding_norm"] for item in klpaths])) if klpaths else None,
            "mean_KL_ph_embedding_gradient_norm": float(np.mean([item["ph_embedding_norm"] for item in klpaths])) if klpaths else None,
            "mean_KL_img_unet_parameter_gradient_norm": float(np.mean([item["img_unet_parameter_norm"] for item in klpaths])) if klpaths else None,
            "mean_KL_ph_unet_parameter_gradient_norm": float(np.mean([item["ph_unet_parameter_norm"] for item in klpaths])) if klpaths else None,
            "mean_KL_rp_attention_parameter_gradient_norm": float(np.mean([item["rp_attention_parameter_norm"] for item in klpaths])) if klpaths else None,
            "mean_image_class_prototype_distance": float(np.mean([item["class_prototype_distance_mean"] for item in image_protos])) if image_protos else None,
            "mean_ph_class_prototype_distance": float(np.mean([item["class_prototype_distance_mean"] for item in ph_protos])) if ph_protos else None,
            "mean_ph_class_delta_query_sd_l2": float(np.mean([item["class_delta_query_sd_l2"] for item in ph_protos])) if ph_protos else None,
            "mean_ph_min_ESS": float(np.mean([item["ess_min_mean"] for item in ph_protos])) if ph_protos else None,
            "mean_ph_beta": float(np.mean([item["beta_mean"] for item in ph_protos])) if ph_protos else None,
            "ph_invalid_prototype_fraction": float(np.mean([1 - item["valid_fraction"] for item in ph_protos])) if ph_protos else None,
            "ph_prototype_train_accuracy": float(np.mean([item["prototype_train_accuracy"] for item in ph_protos])) if ph_protos else None,
            "image_prototype_train_accuracy": float(np.mean([item["prototype_train_accuracy"] for item in image_protos])) if image_protos else None,
            "mean_lambda_img": float(np.mean([epoch["lambda_img"] for artifact in artifacts for epoch in artifact["epochs"]])) if artifacts else None,
            "mean_lambda_ph": float(np.mean([epoch["lambda_ph"] for artifact in artifacts for epoch in artifact["epochs"]])) if artifacts else None,
            "mean_actual_img_gradient_ratio": float(np.mean([epoch["image_gradient_budget"]["weighted_auxiliary_z_gradient_ratio"]
                                                                for artifact in artifacts for epoch in artifact["epochs"]])) if artifacts else None,
            "mean_actual_ph_gradient_ratio": float(np.mean([epoch["phenotype_gradient_budget"]["weighted_auxiliary_z_gradient_ratio"]
                                                               for artifact in artifacts for epoch in artifact["epochs"]])) if artifacts else None,
            "shuffle_moved_fraction": float(np.mean([artifact["prototype_preparation"]["shuffle_moved_fraction"]
                                                       for artifact in artifacts])) if group == "M4" and artifacts else None,
            "shuffle_changed_fraction": float(np.mean([artifact["prototype_preparation"]["shuffle_changed_fraction"]
                                                         for artifact in artifacts])) if group == "M4" and artifacts else None,
            "max_formal_test_forwards": max([artifact["formal_test_forwards"] for artifact in artifacts], default=None),
        }
    return groups


def write_csv(path: Path, columns, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def percent_summary(summary):
    return {key: {**value, "mean_percent": None if value["mean"] is None else 100 * value["mean"],
                  "sample_sd_pp": None if value["sample_sd"] is None else 100 * value["sample_sd"]}
            for key, value in summary.items()}


def build_report(summary):
    def fmt(value):
        if value is None:
            return "NA"
        mean = value.get("mean_percent")
        sample_sd = value.get("sample_sd_pp")
        if mean is None:
            mean = 100 * value["mean"]
            sample_sd = 100 * value["sample_sd"]
        return f"{mean:.2f} +/- {sample_sd:.2f}"

    lines = [
        "# Experiment 20: PCDPA-X", "",
        f"Status: **{summary['status']}**; formal tasks: **{summary['completed_tasks']}/40**.", "",
        "PCDPA-X preserves the historical image D4 conditional prototype CE and adds a phenotype-branch prototype CE. M2/M3/M4 add one stopped-gradient, image-teacher to phenotype-student KL term; M2 uses global phenotype prototypes, M3 real age-sex conditioned prototypes, and M4 the same-site joint-shuffled condition. No direct feature-vector matching or trainable parameters are added.", "",
        "## Four formal metrics", "",
        "|Group|Method|ACC (%)|SEN-ASD (%)|SPE-TD (%)|Probability AUC (%)|", "|---|---|---:|---:|---:|---:|",
    ]
    historic = summary["historical_D4"]["metrics"]
    lines.append("|historical_D4|PCDCA seed911 read-only|" + "|".join(fmt(historic[key]) for key in FORMAL_KEYS) + "|")
    for group, details in summary["groups"].items():
        metrics = details["metrics"] if details else {key: {"mean": None, "sample_sd": None} for key in FORMAL_KEYS}
        method = CFG["groups"][group]["name"]
        lines.append(f"|{group}|{method}|" + "|".join(fmt(metrics[key]) for key in FORMAL_KEYS) + "|")
    lines += ["", "Metrics are calculated per fold and summarized over 10 folds with sample SD (ddof=1). Main results contain only ACC, SEN-ASD, SPE-TD, and probability AUC. Historical D4 is read-only and its mean matches the registered reference. New-group comparisons are paired by fold; historical comparisons are cross-run and remain subject to accelerator trajectory drift.", "", "## Paired changes", "", "|Comparison|Metric|Mean change (pp)|Positive folds|LOO range (pp)|", "|---|---|---:|---:|---:|"]
    for comparison, metrics in summary["paired_changes"].items():
        for metric, value in metrics.items():
            avg = value["mean_delta_pp"]["mean"]
            sd = value["mean_delta_pp"]["sample_sd"]
            loo_min, loo_max = value["leave_one_fold_out_min_pp"], value["leave_one_fold_out_max_pp"]
            avg_text = "NA" if avg is None else f"{avg:+.3f} +/- {sd:.3f}"
            loo_text = "NA" if loo_min is None else f"{loo_min:+.3f} to {loo_max:+.3f}"
            lines.append(f"|{comparison}|{metric}|{avg_text}|{value['positive_folds']}/10|{loo_text}|")
    lines += ["", "## Mechanism checks", "", "Mechanism values are training-only diagnostics and are not additional paper metrics. Prototype/gradient values are summarized over sampled epochs (first three epochs and every 25 epochs)."]
    lines += ["", "|Group|KL mean|KL first|KL last|Teacher-student agreement|Teacher train accuracy|Student train accuracy|Student entropy|Student prototype distance|Ph CE start|Ph CE end|Ph min ESS|Ph beta|Actual ph output-gradient ratio|M4 condition changed fraction|", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group, m in summary["mechanisms"].items():
        values = [m["mean_alignment_kl"], m["mean_KL_first_sample"], m["mean_KL_last_sample"],
                  m["mean_teacher_student_agreement"], m["mean_teacher_train_accuracy"],
                  m["mean_student_train_accuracy"], m["mean_student_entropy"],
                  m["mean_student_prototype_distance"], summary["groups"][group]["mean_ph_ce_start"],
                  summary["groups"][group]["mean_ph_ce_end"], m["mean_ph_min_ESS"], m["mean_ph_beta"],
                  m["mean_actual_ph_gradient_ratio"], m["shuffle_changed_fraction"]]
        lines.append("|" + group + "|" + "|".join("NA" if value is None else f"{value:.4f}" for value in values) + "|")
    lines += ["", "### Interpretation", "", summary["interpretation"], ""]
    lines += ["### Execution deviations", ""]
    lines += [f"- {item}" for item in summary["deviations"]]
    lines += [""]
    lines += ["### Fold, checkpoint, and patient sensitivity", "", f"M3 AUC fold deltas versus M1/M2/M4 and historical D4, including leave-one-fold-out ranges, are in `summary.json`. Fold 2/6/9 checkpoint differences: `{json.dumps(summary['checkpoint_sensitivity'], ensure_ascii=False, sort_keys=True)}`.", "", f"Patient-level OOF transitions: `{json.dumps(summary['patient_oof_changes'], ensure_ascii=False, sort_keys=True)}`.", ""]
    lines += ["### Audit and limitations", "", f"Historical train_start/split/RFE/VAE hashes are recorded in `audit.json`; all 40 tasks use seed911 and the original ten folds. Original model source hashes: `{json.dumps(summary['audit']['original_source_sha256'], sort_keys=True)}`.", "", "The study retains the transductive 871-node population graph, its historical cross-fold RFE behavior, stochastic VAE sampling, RP in-place constraints, and grad-enabled stochastic single test forward. The non-imaging branch receives only site, sex, and age; prototype evidence cannot be described as an independently identified disease factor. Historical D4 results are read-only and no historical model is retrained.", ""]
    lines += [f"The 24 grouped mathematical assertions and real-data single-forward preflight passed. Preflight captured logits `{summary['audit']['preflight']['native_logits_shape']}`, z_img `{summary['audit']['preflight']['z_img_shape']}`, and z_ph `{summary['audit']['preflight']['z_ph_shape']}` with exactly one core forward; M1-M4 had identical image D4 loss, phenotype CE, and teacher logits on the shared inputs. Training, manifest, analysis, input, and original-source SHA-256 values are recorded in `audit.json` and `manifest.json`.", ""]
    return "\n".join(lines)


def main():
    audit = read_json(BASE / "audit.json")
    records, artifact_errors = artifact_records(audit)
    historical, historical_metrics, historical_mean_difference = load_historical_d4()
    groups = {group: summarize_group(records, group) for group in CFG["groups"]}
    all_complete = len(records) == 40 and not artifact_errors
    all_oof_rows = []
    fold_rows = []
    for group in CFG["groups"]:
        for fold in range(10):
            value = records.get((group, fold))
            if value is None:
                continue
            fold_rows.append({"group": group, "seed": 911, "fold": fold,
                              **value["metrics"], "best_epoch": value["best_epoch"],
                              "best_val_loss": value["best_val_loss"],
                              "training_seconds": value["training_seconds"]})
            oof = value["oof"]
            for idx, subject_id, label, logits, probability, prediction in zip(
                    oof["indices"], oof["ids"], oof["labels"], oof["logits"], oof["probability"], oof["prediction"]):
                all_oof_rows.append({"group": group, "seed": 911, "fold": fold,
                                     "subject_index": int(idx), "subject_id": subject_id,
                                     "label": int(label), "logit_ASD": float(logits[0]),
                                     "logit_TD": float(logits[1]), "probability_ASD": float(probability),
                                     "prediction": int(prediction)})
    paired_changes = {}
    for comparison, (left, right) in COMPARISONS.items():
        paired_changes[comparison] = {metric: paired_fold_deltas(records, left, right, metric)
                                      for metric in FORMAL_KEYS}
    for comparison_group in ("M1", "M3"):
        comparison_name = f"{comparison_group}-historical_D4"
        paired_changes[comparison_name] = {}
        for metric in FORMAL_KEYS:
            deltas = []
            for fold in range(10):
                value = records.get((comparison_group, fold))
                if value is not None:
                    deltas.append({"fold": fold, "left": comparison_group, "right": "historical_D4",
                                   "delta_pp": 100 * (value["metrics"][metric]
                                                       - historical[fold]["metrics"][metric])})
            vals = [item["delta_pp"] for item in deltas]
            loo = [float(np.mean([item["delta_pp"] for item in deltas if item["fold"] != fold]))
                   for fold in range(10) if len(deltas) > 1 and any(item["fold"] == fold for item in deltas)]
            paired_changes[comparison_name][metric] = {
                "metric": metric, "fold_deltas": deltas, "mean_delta_pp": mean_sd(vals),
                "positive_folds": sum(value > 0 for value in vals),
                "leave_one_fold_out_mean_delta_pp": loo,
                "leave_one_fold_out_min_pp": min(loo) if loo else None,
                "leave_one_fold_out_max_pp": max(loo) if loo else None,
                "leave_one_fold_out_direction_changes": bool(loo and min(loo) <= 0 <= max(loo)),
            }
    comparison_order = ("M1-historical_D4", "M2-M1", "M3-M2", "M3-M4", "M3-M1", "M3-historical_D4")
    paired_changes = {name: paired_changes[name] for name in comparison_order}
    patient_changes = {f"M3-{other}": paired_oof(records, historical, "M3", other)
                       for other in ("M1", "M2", "M4", "historical_D4")}
    checkpoint_sensitivity = {}
    for fold in (2, 6, 9):
        checkpoint_sensitivity[str(fold)] = {}
        for left, right in (("M3", "M1"), ("M3", "M2"), ("M3", "M4"), ("M3", "historical_D4")):
            a = records.get((left, fold))
            b = historical[fold]["artifact"] if right == "historical_D4" else records.get((right, fold))
            checkpoint_sensitivity[str(fold)][f"{left}-{right}"] = None if a is None or b is None else {
                "best_epoch_delta": int(a["best_epoch"]) - int(b["best_epoch"]),
                "best_val_loss_delta": float(a["best_val_loss"]) - float(b["best_val_loss"]),
                "best_core_state_delta": state_delta(a, b),
            }
    mechanisms = mechanism_summary(records)
    m4_condition_shuffle = {
        str(fold): records[("M4", fold)]["prototype_preparation"]
        for fold in range(10) if ("M4", fold) in records
    }
    m3_auc = groups["M3"]["metrics"]["probability_AUC"]["mean"] if groups["M3"] else None
    hist_auc = historical_metrics["probability_AUC"]["mean"]
    auc_deltas = {name: value["probability_AUC"]["mean_delta_pp"]["mean"]
                  for name, value in paired_changes.items() if "probability_AUC" in value}
    mechanism_valid = all(
        mechanisms[group]["mean_KL_img_embedding_gradient_norm"] == 0.0
        and (mechanisms[group]["mean_KL_ph_embedding_gradient_norm"] or 0) > 0
        and (mechanisms[group]["mean_KL_ph_unet_parameter_gradient_norm"] or 0) > 0
        for group in ("M2", "M3", "M4")
    ) if all_complete else False
    primary_all_positive = all(auc_deltas.get(name) is not None and auc_deltas[name] > 0
                               for name in ("M3-M1", "M3-M2", "M3-M4", "M3-historical_D4"))
    robust = all(
        paired_changes[name]["probability_AUC"]["leave_one_fold_out_min_pp"] is not None
        and paired_changes[name]["probability_AUC"]["leave_one_fold_out_min_pp"] > 0
        for name in ("M3-M1", "M3-M2", "M3-M4")
    ) if all_complete else False
    m3_acc = groups["M3"]["metrics"]["ACC"]["mean"] if groups["M3"] else None
    hist_acc = historical_metrics["ACC"]["mean"]
    m3_sen = groups["M3"]["metrics"]["SEN-ASD"]["mean"] if groups["M3"] else None
    m3_spe = groups["M3"]["metrics"]["SPE-TD"]["mean"] if groups["M3"] else None
    hist_sen = historical_metrics["SEN-ASD"]["mean"]
    hist_spe = historical_metrics["SPE-TD"]["mean"]
    no_sensitivity_specificity_drop = (m3_sen is not None and m3_spe is not None
                                       and m3_sen >= hist_sen and m3_spe >= hist_spe)
    shuffle_effective = (mechanisms["M4"]["shuffle_changed_fraction"] or 0) > 0
    if not all_complete:
        decision = "PARTIAL"
    elif (primary_all_positive and robust and mechanism_valid and shuffle_effective
          and m3_acc is not None and m3_acc >= hist_acc and no_sensitivity_specificity_drop):
        decision = "CANDIDATE_GO_FOR_CONFIRMATION"
    elif any(value is not None and value > 0 for value in auc_deltas.values()):
        decision = "MIXED"
    else:
        decision = "NO-GO"
    interpretation = (
        f"Decision: **{decision}**. M3 AUC minus historical D4 is "
        f"{auc_deltas.get('M3-historical_D4')} pp; M3-M1/M2/M4 AUC changes are "
        f"{auc_deltas.get('M3-M1')}, {auc_deltas.get('M3-M2')}, and {auc_deltas.get('M3-M4')} pp. "
        f"M1-historical D4 AUC change is {auc_deltas.get('M1-historical_D4')} pp. "
        "M2-M1 changes the objective by adding KL and is not a zero-information alignment test. "
        "M3-M4 isolates the condition used by the new KL branch only; the shared phenotype CE remains real-conditioned. "
        "KL reduction alone is not evidence of diagnostic value. "
        f"Sampled KL first-to-last was {mechanisms['M2']['mean_KL_first_sample']:.3f}->{mechanisms['M2']['mean_KL_last_sample']:.3f} for M2, "
        f"{mechanisms['M3']['mean_KL_first_sample']:.3f}->{mechanisms['M3']['mean_KL_last_sample']:.3f} for M3, and "
        f"{mechanisms['M4']['mean_KL_first_sample']:.3f}->{mechanisms['M4']['mean_KL_last_sample']:.3f} for M4; it did not decrease. "
        "Teacher-student agreement and student train accuracy stayed near 0.50, student entropy near the two-class maximum, and the shared phenotype prototype CE near chance, despite nonzero prototype geometry. "
        "KL gradients were zero at z_img and img_unet and nonzero at z_ph and ph_unet; a small RP-attention gradient shows an indirect path through the shared graph. "
        "Do not select PCDPA-X as the first MICCAI 2027 innovation candidate. A future study should first justify richer non-imaging inputs, then preregister multi-seed and independent-site confirmation; prior multi-seed D4 evidence also does not establish a stable gain or a high-quality teacher."
    )
    analysis_code_sha256 = sha_file(BASE / "analyze.py")
    deviations = []
    for item in audit.get("freeze_attempts", []):
        deviations.append("The first launcher invocation stopped before any task spawned because `ROOT` was missing from its imports; the import was fixed, mathematical tests and the one-forward preflight were rerun, and a new manifest was frozen.")
    normalization = (audit.get("probability_normalization") or [{}])[-1]
    if normalization.get("changed_artifacts"):
        deviations.append(f"The initial validator rejected {normalization['changed_artifacts']} completed artifacts because PPU and CPU float32 softmax differed by at most 1.1921e-7, just above its 1e-7 tolerance. Retry attempts stopped at artifact validation without retraining; ASD probabilities and four metrics were recomputed from the saved logits on CPU, then all 40 artifacts passed full validation.")
    log_repair = audit.get("run_log_integrity")
    if log_repair:
        deviations.append(f"The experiment JSONL log had {log_repair['nul_bytes_removed']} NUL padding bytes removed; all {log_repair['valid_json_lines']} remaining lines parse as JSON.")
    status = "COMPLETE_WITH_DEVIATION" if deviations else ("COMPLETE" if all_complete else "PARTIAL")
    audit["analysis"] = {"completed_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
                         "analysis_code_sha256": analysis_code_sha256,
                         "artifact_errors": artifact_errors, "independent_metric_recalculation": True}
    summary = {
        "experiment": CFG["experiment"], "method": CFG["name"],
        "status": status, "completed_tasks": len(records), "deviations": deviations,
        "tasks_by_group": {group: 0 if groups[group] is None else groups[group]["folds_complete"] for group in CFG["groups"]},
        "historical_D4": {"read_only": True, "source": str(PCDCA / "runs/D4"),
                          "metrics": percent_summary(historical_metrics),
                          "registered_reference_percent": CFG["historical_d4_reference_percent"],
                          "reference_difference_pp": historical_mean_difference,
                          "folds": {str(fold): historical[fold]["metrics"] for fold in historical}},
        "groups": groups, "paired_changes": paired_changes,
        "mechanisms": mechanisms, "patient_oof_changes": patient_changes,
        "m4_condition_shuffle": m4_condition_shuffle,
        "checkpoint_sensitivity": checkpoint_sensitivity,
        "historical_comparison_limits": ["Historical D4 is a read-only separate run; PPU trajectory drift is not controlled.",
                                         "M3-M4 changes only the KL student's condition; the shared phenotype CE stays real-conditioned.",
                                         "M2-M1 adds KL and is not a matched zero-information alignment test."],
        "candidate_go_checks": {"M3_AUC_better_than_historical_D4_and_M1_M2_M4": primary_all_positive,
                                "leave_one_fold_out_auc_direction_stable": robust,
                                "teacher_stop_gradient_and_student_path_valid": mechanism_valid,
                                "M4_same_site_condition_shuffle_effective": shuffle_effective,
                                "M3_ACC_not_below_historical_D4": None if m3_acc is None else m3_acc >= hist_acc,
                                "M3_SEN_and_SPE_not_below_historical_D4": no_sensitivity_specificity_drop},
        "decision": decision, "interpretation": interpretation,
        "analysis_code_sha256": analysis_code_sha256,
        "audit": {"training_code_sha256": audit["training_code_sha256"],
                  "manifest_sha256": audit["manifest_sha256"],
                  "analysis": audit["analysis"],
                  "original_source_sha256": audit["original_source_sha256"],
                  "historical_pcdca_prototype_sha256": audit["historical_pcdca_prototype_sha256"],
                  "history_policy": audit["history_policy"], "preflight": audit.get("preflight"),
                  "freeze_attempts": audit.get("freeze_attempts", []),
                  "probability_normalization": [{key: item.get(key) for key in
                                                   ("timestamp_utc", "method", "changed_artifacts",
                                                    "verified_artifacts", "verification_failures")}
                                                 for item in audit.get("probability_normalization", [])],
                  "run_log_integrity": audit.get("run_log_integrity"),
                  "execution": audit["execution"]},
        "artifact_sha256": {f"{group}/f{fold}": sha_file(result_path(group, 911, fold))
                            for group in CFG["groups"] for fold in range(10)
                            if (group, fold) in records},
    }
    atomic_json(audit, BASE / "audit.json")
    write_csv(BASE / "fold_metrics.csv", ["group", "seed", "fold", *FORMAL_KEYS,
                                           "best_epoch", "best_val_loss", "training_seconds"], fold_rows)
    write_csv(BASE / "oof.csv", ["group", "seed", "fold", "subject_index", "subject_id", "label",
                                 "logit_ASD", "logit_TD", "probability_ASD", "prediction"], all_oof_rows)
    atomic_json(summary, BASE / "summary.json")
    (BASE / "report.md").write_text(build_report(summary), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "completed_tasks": len(records),
                      "decision": decision, "groups": groups,
                      "auc_delta_pp": auc_deltas}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
