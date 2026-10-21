"""Independent PCDPA-X2 validation, paired analysis, and report generation."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))

from run import (CFG, PCDCA, atomic_json, formal_metrics, load_torch, read_json,
                 result_path, sha_file, valid_result)


PREVIOUS = ROOT / "experiments/PCDPAX_20261021"
GROUPS = ("N1", "N2", "N3", "N4")
METRICS = ("ACC", "SEN-ASD", "SPE-TD", "probability_AUC")
COMPARE = {
    "N2-N1": ("N2", "N1"),
    "N3-N1": ("N3", "N1"), "N3-N2": ("N3", "N2"), "N3-N4": ("N3", "N4"),
    "N3-historical_M3": ("N3", "historical_M3"),
    "N3-historical_D4": ("N3", "historical_D4"),
}


def _unwrap(value, path: Path):
    if "payload" in value and "content_sha256" in value:
        from run import payload_hash
        if payload_hash(value["payload"]) != value["content_sha256"]:
            raise AssertionError(f"Historical artifact checksum mismatch: {path}")
        return value["payload"]
    return value


def _load_group(directory: Path, group: str, historical=False) -> dict:
    records = {}
    for fold in range(10):
        path = directory / "runs" / group / f"s911_f{fold}.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        value = _unwrap(load_torch(path), path)
        if value.get("state") != "complete" or (value.get("group"), value.get("seed"), value.get("fold")) != (group, 911, fold):
            raise AssertionError(f"Incomplete or mismatched read-only artifact: {path}")
        oof = value["oof"]
        labels = np.asarray(oof["labels"], dtype=np.int64)
        probability = np.asarray(oof["probability"], dtype=np.float64)
        prediction = np.asarray(oof["prediction"], dtype=np.int64)
        recalculated = formal_metrics(labels, probability, prediction)
        stored_metrics = value["metrics"]
        aliases = {"ACC": "ACC", "SEN-ASD": "SEN", "SPE-TD": "SPE",
                   "probability_AUC": "AUC"}
        if max(abs(recalculated[key] - stored_metrics.get(key, stored_metrics.get(aliases[key])))
               for key in METRICS) > 1e-12:
            raise AssertionError(f"Stored metrics do not recompute: {path}")
        records[fold] = {
            "artifact": value, "metrics": recalculated, "oof": oof,
            "artifact_sha256": sha_file(path), "historical": historical,
        }
    return records


def load_historical() -> dict:
    return {
        "historical_D4": _load_group(PCDCA, "D4", historical=True),
        "historical_M3": _load_group(PREVIOUS, "M3", historical=True),
    }


def load_new(audit: dict) -> tuple[dict, list]:
    records, failures = {}, []
    for group in GROUPS:
        for fold in range(10):
            try:
                value = valid_result(group, 911, fold, audit, required=False)
            except Exception as error:
                failures.append({"group": group, "fold": fold, "error": repr(error)})
                continue
            if value is not None:
                records[(group, fold)] = {
                    "artifact": value, "metrics": value["metrics"], "oof": value["oof"],
                    "artifact_sha256": sha_file(result_path(group, 911, fold)),
                    "historical": False,
                }
    return records, failures


def mean_sd(values, scale=100.0) -> dict:
    values = np.asarray(values, dtype=np.float64) * scale
    return {"mean": float(values.mean()) if len(values) else None,
            "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else (0.0 if len(values) else None),
            "n": int(len(values))}


def group_metrics(records: dict, group: str) -> dict:
    folds = records[group] if group in records else {
        fold: records[(group, fold)] for fold in range(10) if (group, fold) in records}
    if not folds:
        return {"folds_complete": 0, "metrics": {key: mean_sd([]) for key in METRICS}}
    return {
        "folds_complete": len(folds),
        "metrics": {key: mean_sd([folds[fold]["metrics"][key] for fold in sorted(folds)])
                    for key in METRICS},
        "mean_best_epoch": float(np.mean([folds[fold]["artifact"].get("best_epoch", 0)
                                           for fold in sorted(folds)])),
        "mean_best_validation_loss": float(np.mean([folds[fold]["artifact"].get("best_val_loss", 0)
                                                     for fold in sorted(folds)])),
        "mean_training_seconds": float(np.mean([folds[fold]["artifact"].get("training_seconds", 0)
                                                 for fold in sorted(folds)])),
    }


def _folds_for(records: dict, group: str) -> dict:
    if group in records and isinstance(records[group], dict):
        return records[group]
    return {fold: records[(group, fold)] for fold in range(10) if (group, fold) in records}


def paired_delta(records: dict, left: str, right: str, metric: str) -> dict:
    a, b = _folds_for(records, left), _folds_for(records, right)
    deltas = {fold: 100 * (a[fold]["metrics"][metric] - b[fold]["metrics"][metric])
              for fold in sorted(set(a) & set(b))}
    loo = {str(omitted): float(np.mean([delta for fold, delta in deltas.items() if fold != omitted]))
           for omitted in deltas if len(deltas) > 1}
    return {
        "metric": metric,
        "fold_delta_pp": {str(fold): float(value) for fold, value in deltas.items()},
        "mean_delta_pp": mean_sd(list(deltas.values()), scale=1.0),
        "positive_folds": int(sum(value > 0 for value in deltas.values())),
        "negative_folds": int(sum(value < 0 for value in deltas.values())),
        "leave_one_fold_out_mean_delta_pp": loo,
        "leave_one_fold_out_min_pp": min(loo.values()) if loo else None,
        "leave_one_fold_out_max_pp": max(loo.values()) if loo else None,
        "leave_one_fold_out_direction_changes": bool(loo and min(loo.values()) <= 0 <= max(loo.values())),
        "fold_2_6_9_pp": {str(fold): float(deltas[fold]) for fold in (2, 6, 9) if fold in deltas},
    }


def loo_auc(records: dict, group: str) -> dict:
    folds = _folds_for(records, group)
    values = {str(fold): float(np.mean([folds[other]["metrics"]["probability_AUC"]
                                       for other in folds if other != fold]) * 100)
              for fold in folds if len(folds) > 1}
    return {"fold_excluded_mean_auc_percent": values,
            "min_percent": min(values.values()) if values else None,
            "max_percent": max(values.values()) if values else None}


def collect_oof(records: dict, group: str) -> dict:
    items = {}
    folds = _folds_for(records, group)
    for fold, record in folds.items():
        oof = record["oof"]
        for index, subject_id, label, probability, prediction, logits in zip(
                oof["indices"], oof["ids"], oof["labels"], oof["probability"],
                oof["prediction"], oof["logits"]):
            if subject_id in items:
                raise AssertionError(f"Subject {subject_id} appears more than once in {group} OOF")
            items[subject_id] = {"index": int(index), "fold": int(fold), "label": int(label),
                                 "probability": float(probability), "prediction": int(prediction),
                                 "logits": [float(value) for value in logits]}
    return items


def paired_oof(records: dict, left: str, right: str) -> dict:
    left_rows, right_rows = collect_oof(records, left), collect_oof(records, right)
    common = sorted(set(left_rows) & set(right_rows))
    error_to_correct, correct_to_error = [], []
    net = {"ASD": 0, "TD": 0}
    for subject_id in common:
        a, b = left_rows[subject_id], right_rows[subject_id]
        if a["label"] != b["label"]:
            raise AssertionError(f"OOF labels differ for {subject_id}")
        if b["prediction"] != b["label"] and a["prediction"] == a["label"]:
            error_to_correct.append(subject_id)
            net["ASD" if a["label"] == 0 else "TD"] += 1
        elif b["prediction"] == b["label"] and a["prediction"] != a["label"]:
            correct_to_error.append(subject_id)
            net["ASD" if a["label"] == 0 else "TD"] -= 1
    return {"n": len(common), "coverage_complete": len(common) == 871,
            "N3_error_to_correct": len(error_to_correct),
            "N3_correct_to_error": len(correct_to_error),
            "net_correct_change": len(error_to_correct) - len(correct_to_error),
            "ASD_net_correct_change": net["ASD"], "TD_net_correct_change": net["TD"],
            "error_to_correct_ids": error_to_correct, "correct_to_error_ids": correct_to_error}


def mechanism_summary(records: dict) -> dict:
    result = {}
    for group in GROUPS:
        artifacts = [records[(group, fold)]["artifact"] for fold in range(10)
                     if (group, fold) in records]
        sampled = [epoch for artifact in artifacts for epoch in artifact["epochs"]
                   if "alignment" in epoch]
        alignments = [epoch["alignment"] for epoch in sampled]
        paths = [epoch["kl_gradient_path"] for epoch in sampled if "kl_gradient_path" in epoch]
        image_budgets = [epoch["image_gradient_budget"] for artifact in artifacts
                         for epoch in artifact["epochs"]]
        ph_budgets = [epoch["phenotype_gradient_budget"] for artifact in artifacts
                      for epoch in artifact["epochs"]]
        first_kl, last_kl, ph_start, ph_end, img_start, img_end = [], [], [], [], [], []
        for artifact in artifacts:
            epochs = artifact["epochs"]
            aligned = [epoch for epoch in epochs if "alignment" in epoch]
            if aligned:
                first_kl.append(aligned[0]["alignment"]["kl"])
                last_kl.append(aligned[-1]["alignment"]["kl"])
            if epochs:
                ph_start.append(epochs[0]["phenotype_real_ce"])
                ph_end.append(epochs[-1]["phenotype_real_ce"])
                img_start.append(epochs[0]["image_d4_ce"])
                img_end.append(epochs[-1]["image_d4_ce"])
        all_sites = {}
        for item in alignments:
            for site, values in item.get("by_site", {}).items():
                record = all_sites.setdefault(site, {key: [] for key in (
                    "n", "teacher_asd_probability_mean", "teacher_asd_probability_sd",
                    "teacher_entropy_mean", "teacher_train_accuracy")})
                for key in record:
                    record[key].append(values[key])
        result[group] = {
            "sampled_epochs": len(sampled),
            "mean_KL": float(np.mean([item["kl"] for item in alignments])) if alignments else None,
            "mean_KL_first_sample": float(np.mean(first_kl)) if first_kl else None,
            "mean_KL_last_sample": float(np.mean(last_kl)) if last_kl else None,
            "mean_teacher_entropy": float(np.mean([item["teacher_entropy"] for item in alignments])) if alignments else None,
            "mean_teacher_asd_probability": float(np.mean([item["teacher_asd_probability_mean"] for item in alignments])) if alignments else None,
            "mean_teacher_asd_probability_sd_across_subjects": float(np.mean([item["teacher_asd_probability_sd"] for item in alignments])) if alignments else None,
            "mean_teacher_predicted_asd_fraction": float(np.mean([
                item["teacher_predicted_class_fraction"]["0"] for item in alignments])) if alignments else None,
            "mean_teacher_train_accuracy": float(np.mean([item["teacher_train_accuracy"] for item in alignments])) if alignments else None,
            "mean_student_entropy": float(np.mean([item["student_entropy"] for item in alignments])) if alignments else None,
            "mean_student_train_accuracy": float(np.mean([item["student_train_accuracy"] for item in alignments])) if alignments else None,
            "mean_teacher_student_argmax_agreement": float(np.mean([
                item["teacher_student_argmax_agreement"] for item in alignments])) if alignments else None,
            "mean_ph_CE_start": float(np.mean(ph_start)) if ph_start else None,
            "mean_ph_CE_end": float(np.mean(ph_end)) if ph_end else None,
            "mean_image_D4_CE_start": float(np.mean(img_start)) if img_start else None,
            "mean_image_D4_CE_end": float(np.mean(img_end)) if img_end else None,
            "mean_KL_to_ph_embedding_gradient_norm": float(np.mean([
                item["ph_embedding_norm"] for item in paths])) if paths else None,
            "mean_KL_to_img_embedding_gradient_norm": float(np.mean([
                item["img_embedding_norm"] for item in paths])) if paths else None,
            "mean_KL_to_ph_unet_parameter_gradient_norm": float(np.mean([
                item["ph_unet_parameter_norm"] for item in paths])) if paths else None,
            "mean_KL_to_img_unet_parameter_gradient_norm": float(np.mean([
                item["img_unet_parameter_norm"] for item in paths])) if paths else None,
            "mean_KL_to_RP_attention_parameter_gradient_norm": float(np.mean([
                item["rp_attention_parameter_norm"] for item in paths])) if paths else None,
            "mean_ph_CE_embedding_gradient_norm": float(np.mean([
                item["ph_ce_embedding_norm"] for item in paths])) if paths else None,
            "mean_KL_embedding_gradient_norm": float(np.mean([
                item["kl_embedding_norm"] for item in paths])) if paths else None,
            "mean_ph_CE_KL_gradient_cosine": float(np.mean([
                item["ph_ce_kl_gradient_cosine"] for item in paths])) if paths else None,
            "mean_lambda_img": float(np.mean([epoch["lambda_img"] for artifact in artifacts
                                                for epoch in artifact["epochs"]])) if artifacts else None,
            "mean_lambda_ph": float(np.mean([epoch["lambda_ph"] for artifact in artifacts
                                               for epoch in artifact["epochs"]])) if artifacts else None,
            "mean_actual_img_gradient_ratio": float(np.mean([
                item["weighted_auxiliary_z_gradient_ratio"] for item in image_budgets])) if image_budgets else None,
            "mean_actual_ph_gradient_ratio": float(np.mean([
                item["weighted_auxiliary_z_gradient_ratio"] for item in ph_budgets])) if ph_budgets else None,
            "mean_site_teacher_diagnostics": {
                site: {key: float(np.average(values)) for key, values in fields.items()}
                for site, fields in all_sites.items()},
            "best_epoch_by_fold": {str(fold): records[(group, fold)]["artifact"]["best_epoch"]
                                   for fold in range(10) if (group, fold) in records},
            "best_validation_loss_by_fold": {str(fold): records[(group, fold)]["artifact"]["best_val_loss"]
                                             for fold in range(10) if (group, fold) in records},
        }
    return result


def checkpoint_comparison(records: dict) -> dict:
    result = {}
    for name in ("N3-N1", "N3-N2", "N3-N4"):
        left, right = COMPARE[name]
        values = {}
        for fold in range(10):
            if (left, fold) not in records or (right, fold) not in records:
                continue
            a = records[(left, fold)]["artifact"]["best_model_state"]["state"]
            b = records[(right, fold)]["artifact"]["best_model_state"]["state"]
            squared = reference = 0.0
            for key in sorted(set(a) & set(b)):
                delta = a[key].double() - b[key].double()
                squared += float(delta.square().sum())
                reference += float(b[key].double().square().sum())
            values[str(fold)] = {"absolute_l2": squared ** 0.5,
                                 "relative_l2_to_right": (squared ** 0.5) / max(reference ** 0.5, 1e-12)}
        result[name] = {
            "folds": values,
            "mean_absolute_l2": float(np.mean([item["absolute_l2"] for item in values.values()])) if values else None,
            "mean_relative_l2": float(np.mean([item["relative_l2_to_right"] for item in values.values()])) if values else None,
            "fold_2_6_9": {fold: values[fold] for fold in ("2", "6", "9") if fold in values},
        }
    return result


def coverage_audit(records: dict) -> dict:
    result = {"tasks_complete": len(records), "expected_tasks": 40,
              "groups": {}, "same_fold_start_consistency": {}, "test_contract": {}}
    for group in GROUPS:
        subjects = collect_oof(records, group) if len([key for key in records if key[0] == group]) == 10 else {}
        result["groups"][group] = {"folds_complete": len([key for key in records if key[0] == group]),
                                   "oof_subjects": len(subjects),
                                   "indices_once_each": len(subjects) == 871}
    for fold in range(10):
        row = [records[(group, fold)]["artifact"] for group in GROUPS if (group, fold) in records]
        result["same_fold_start_consistency"][str(fold)] = {
            "initial_core_equal": len(row) == 4 and len({item.get("initial_core_sha256") for item in row}) == 1,
            "start_rng_equal": len(row) == 4 and len({json.dumps(item.get("start_rng_fingerprint"), sort_keys=True) for item in row}) == 1,
            "input_digest_equal": len(row) == 4 and len({item.get("input_digest") for item in row}) == 1,
        }
    all_artifacts = [record["artifact"] for record in records.values()]
    result["test_contract"] = {
        "formal_test_forward_count_one_each": bool(all_artifacts) and all(item.get("formal_test_forwards") == 1 for item in all_artifacts),
        "test_grad_enabled_each": bool(all_artifacts) and all(item.get("formal_test_grad_enabled") is True for item in all_artifacts),
        "no_test_label_selection": bool(all_artifacts) and all(item.get("uses_test_labels_for_training_or_selection") is False for item in all_artifacts),
        "all_new_trainable_parameter_counts_zero": bool(all_artifacts) and all(item.get("new_trainable_parameters") == 0 for item in all_artifacts),
        "cpu_softmax_max_abs_diff_max": max(
            [item.get("probability_cpu_recalc_max_abs_diff", 0.0) for item in all_artifacts], default=None),
        "cpu_softmax_tolerance_2e6_pass": bool(all_artifacts) and all(
            item.get("probability_cpu_recalc_max_abs_diff", float("inf")) <= 2e-6 for item in all_artifacts),
        "oof_coverage_871_each_group": all(value["indices_once_each"] for value in result["groups"].values()),
    }
    result["same_fold_start_consistency_all_groups"] = all(
        all(value.values()) for value in result["same_fold_start_consistency"].values())
    return result


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_report(summary: dict) -> str:
    stats = summary["formal_metrics"]
    lines = [
        "# 实验21：PCDPA-X2",
        "",
        "## 研究问题与方法",
        "",
        "保持实验20 M3 的双分支原型 CE、学生 q_ph、0.25 KL 系数、两条分支各自 5% 梯度预算和原生训练/验证/测试协议。本实验唯一改变 KL 教师目标：N1 为同 D4 私有折其他训练者的类别先验，N2 为其影像 D4 概率均值，N3 为年龄与性别核加权的影像概率，N4 为使用历史同站点联合置换年龄/性别的负对照。教师仅在 outer-train 内构造，参考不含自身且只取同私有折；空参考行在四组共用有效掩码下跳过。",
        "",
        "年龄/性别只调整影像教师概率的聚合权重。影像教师、真实非影像 D4 CE、学生 logits、gamma、原型和梯度预算在四组定义中相同。教师概率 detach；KL 方向为 KL(T || softmax(q_ph))。训练中只捕获原生两条 GTUNet 输出，不增加参数或模型 forward。",
        "",
        "## 正式结果",
        "",
        "单位为百分比；每项为 10 折均值 ± sample SD（ddof=1）。",
        "",
        "|组别|ACC|SEN-ASD|SPE-TD|probability AUC|",
        "|---|---:|---:|---:|---:|",
    ]
    order = ("historical_D4", "historical_M3", *GROUPS)
    for group in order:
        metric = stats[group]["metrics"]
        cells = ["n/a" if metric[key]["mean"] is None else
                 f'{metric[key]["mean"]:.2f} ± {metric[key]["sample_sd"]:.2f}' for key in METRICS]
        lines.append(f"|{group}|" + "|".join(cells) + "|")
    lines += ["", "## 配对差异", "", "单位为百分点；按左侧组减右侧组。", "",
              "|比较|ΔACC|ΔSEN-ASD|ΔSPE-TD|ΔAUC|AUC 正向折数|AUC LOO 范围|",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name in COMPARE:
        metric = summary["paired_differences"][name]
        values = [metric[key]["mean"] for key in METRICS]
        auc = metric["probability_AUC"]
        loo_min, loo_max = auc["leave_one_fold_out_min_pp"], auc["leave_one_fold_out_max_pp"]
        loo_text = "n/a" if loo_min is None else f"{loo_min:.3f}…{loo_max:.3f}"
        lines.append(f"|{name}|" + "|".join("n/a" if value is None else f"{value:+.3f}" for value in values)
                     + f'|{auc["positive_folds"]}/10|{loo_text}|')
    lines += ["", "## 教师与学生诊断", "",
              "训练内诊断仅用于机制解释，不是第五项正式性能指标。历史人口图仍为转导式，原有 RFE 跨折信息泄漏、单 seed 和重复探索等限制均保留。"]
    mechanisms = summary["mechanisms"]
    lines += ["", "|组别|KL 首次→末次|教师 ASD 概率均值|跨患者 SD|教师熵|预测 ASD 比例|教师训练准确率|学生训练准确率|学生熵|ph CE 首→末|KL→ph 梯度范数|CE/KL 梯度余弦|",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group in GROUPS:
        item = mechanisms[group]
        def fmt(a, b=None):
            if a is None:
                return "n/a"
            return f"{a:.4f}" if b is None else f"{a:.4f} → {b:.4f}"
        lines.append(f'|{group}|{fmt(item["mean_KL_first_sample"], item["mean_KL_last_sample"])}'
                     f'|{fmt(item["mean_teacher_asd_probability"])}'
                     f'|{fmt(item["mean_teacher_asd_probability_sd_across_subjects"])}'
                     f'|{fmt(item["mean_teacher_entropy"])}'
                     f'|{fmt(item["mean_teacher_predicted_asd_fraction"])}'
                     f'|{fmt(item["mean_teacher_train_accuracy"])}'
                     f'|{fmt(item["mean_student_train_accuracy"])}'
                     f'|{fmt(item["mean_student_entropy"])}'
                     f'|{fmt(item["mean_ph_CE_start"], item["mean_ph_CE_end"])}'
                     f'|{fmt(item["mean_KL_to_ph_embedding_gradient_norm"])}'
                     f'|{fmt(item["mean_ph_CE_KL_gradient_cosine"])}|')
    lines += ["", "CE 与 KL 共用同一非影像学生 logits；表中的 KL/CE 梯度范数与余弦用于判断 KL 是额外跨模态方向，还是主要与硬标签 CE 重叠。教师分类准确率是 outer-train 内部机制诊断，不是正式性能指标。", ""]
    sites = mechanisms["N3"]["mean_site_teacher_diagnostics"]
    lines += ["N3 站点内教师诊断（站点内均值，不代表未见站点泛化）：", "",
              "|站点|有效 query 数|教师 ASD 概率均值|跨患者 SD|教师熵|训练准确率|",
              "|---|---:|---:|---:|---:|---:|"]
    for site, item in sorted(sites.items()):
        lines.append(f'|{site}|{item["n"]:.1f}|{item["teacher_asd_probability_mean"]:.4f}'
                     f'|{item["teacher_asd_probability_sd"]:.4f}|{item["teacher_entropy_mean"]:.4f}'
                     f'|{item["teacher_train_accuracy"]:.4f}|')
    preflight = summary["preflight_teacher_differences"]
    lines += ["", "fold0 共享前向教师差异：",
              "|比较|最大绝对差|平均绝对差|非零元素比例|", "|---|---:|---:|---:|"]
    for name, item in preflight.items():
        lines.append(f'|{name}|{item["max_abs"]:.6f}|{item["mean_abs"]:.6f}|{item["nonzero_fraction"]:.3f}|')
    lines += ["", "## 敏感性与患者级 OOF", "",
              "每组均计算删除任一 fold 后的十次 AUC 均值范围。Fold2、Fold6、Fold9 差异单独列于 summary.json；完整十折配对和 checkpoint 权重差异也均保留。患者级变化是辅助解释，不取代十折正式 ACC。",
              "", "|比较|N3 错→对|N3 对→错|净正确变化|ASD 净变化|TD 净变化|",
              "|---|---:|---:|---:|---:|---:|"]
    for name, item in summary["patient_oof_changes"].items():
        lines.append(f'|{name}|{item["N3_error_to_correct"]}|{item["N3_correct_to_error"]}'
                     f'|{item["net_correct_change"]:+d}|{item["ASD_net_correct_change"]:+d}'
                     f'|{item["TD_net_correct_change"]:+d}|')
    lines += ["", "各组 AUC leave-one-fold-out 均值范围（百分比）：", "",
              "|组别|最小|最大|", "|---|---:|---:|"]
    for group, item in summary["leave_one_fold_out_auc"].items():
        low, high = item["min_percent"], item["max_percent"]
        low_text = "n/a" if low is None else f"{low:.3f}"
        high_text = "n/a" if high is None else f"{high:.3f}"
        lines.append(f"|{group}|{low_text}|{high_text}|")
    lines += ["", "关键折次 AUC 差异与 best epoch：", "",
              "|折|N3−N1 AUC pp|N3−N2 AUC pp|N3−N4 AUC pp|best epoch N1/N2/N3/N4|",
              "|---:|---:|---:|---:|---:|"]
    for fold in (2, 6, 9):
        auc_values = [summary["paired_differences"][name]["probability_AUC"]["fold_2_6_9_pp"].get(str(fold))
                      for name in ("N3-N1", "N3-N2", "N3-N4")]
        auc_cells = ["n/a" if value is None else f"{value:+.3f}" for value in auc_values]
        epochs = [summary["mechanisms"][group]["best_epoch_by_fold"].get(str(fold))
                  for group in ("N1", "N2", "N3", "N4")]
        epoch_cell = "/".join("n/a" if value is None else str(value) for value in epochs)
        lines.append(f"|{fold}|" + "|".join(auc_cells) + f"|{epoch_cell}|")
    lines += ["", "关键折次最佳 checkpoint 相对权重 L2：", "",
              "|比较|Fold2|Fold6|Fold9|", "|---|---:|---:|---:|"]
    for name in ("N3-N1", "N3-N2", "N3-N4"):
        values = summary["checkpoint_comparison"][name]["fold_2_6_9"]
        cells = []
        for fold in ("2", "6", "9"):
            item = values.get(fold)
            cells.append("n/a" if item is None else f'{item["relative_l2_to_right"]:.6f}')
        lines.append(f"|{name}|" + "|".join(cells) + "|")
    lines += ["", "## 完整性与结论", "",
              f'正式任务 {summary["coverage_audit"]["tasks_complete"]}/40。GPU0 {summary["execution"]["gpu0"]}/20，GPU1 {summary["execution"]["gpu1"]}/20。',
              f'科研判断：**{summary["decision"]["status"]}**。',
              summary["decision"]["rationale"],
              "", "本轮结果是同一历史 seed911 十折上的探索性发现。即使满足候选条件，也只能标为 CANDIDATE_GO_FOR_CONFIRMATION，不代表已确认 MICCAI 2027 创新点。后续独立多 seed、训练折内 RFE、独立站点和方法预注册均不属于本轮。",
              "", "`audit.json`、`manifest.json`、`summary.json`、`fold_metrics.csv` 与 `oof.csv` 保存逐任务输入/源码 fingerprint、概率复核、机制诊断和配对结果。未执行 Git 提交或推送。"]
    return "\n".join(lines) + "\n"


def main() -> int:
    audit = read_json(BASE / "audit.json")
    if not audit.get("formal_authorized"):
        raise AssertionError("Formal training is not frozen and authorized")
    from run import guard
    guard(require_manifest=True)
    historical = load_historical()
    records, invalid = load_new(audit)
    historical.update({})
    combined = {**historical, **{group: _folds_for(records, group) for group in GROUPS}}
    metrics = {group: group_metrics(combined, group)
               for group in ("historical_D4", "historical_M3", *GROUPS)}
    differences = {name: {metric: paired_delta(combined, *pair, metric)
                          for metric in METRICS} for name, pair in COMPARE.items()}
    all_new_records = records
    coverage = coverage_audit(all_new_records)
    coverage["analysis_complete"] = (coverage["tasks_complete"] == 40
                                     and coverage["test_contract"]["oof_coverage_871_each_group"])
    coverage["same_fold_starts_pass"] = coverage["same_fold_start_consistency_all_groups"]
    coverage["test_contract_all_pass"] = all(coverage["test_contract"][key] for key in (
        "formal_test_forward_count_one_each", "test_grad_enabled_each", "no_test_label_selection",
        "all_new_trainable_parameter_counts_zero", "cpu_softmax_tolerance_2e6_pass"))

    fold_rows, oof_rows = [], []
    for group in ("historical_D4", "historical_M3", *GROUPS):
        for fold, item in sorted(_folds_for(combined, group).items()):
            fold_rows.append({"group": group, "seed": 911, "fold": fold,
                              **{key: item["metrics"][key] for key in METRICS}})
    for group in GROUPS:
        values = collect_oof(combined, group)
        for subject_id, value in sorted(values.items(), key=lambda pair: pair[1]["index"]):
            oof_rows.append({"group": group, "seed": 911, "fold": value["fold"],
                             "subject_index": value["index"], "subject_id": subject_id,
                             "label": value["label"], "logit_ASD": value["logits"][0],
                             "logit_TD": value["logits"][1],
                             "probability_ASD": value["probability"],
                             "prediction": value["prediction"],
                             "correct": int(value["prediction"] == value["label"])})
    write_csv(BASE / "fold_metrics.csv", ["group", "seed", "fold", *METRICS], fold_rows)
    write_csv(BASE / "oof.csv", ["group", "seed", "fold", "subject_index", "subject_id",
                                 "label", "logit_ASD", "logit_TD", "probability_ASD", "prediction", "correct"], oof_rows)

    patient_changes = {name: paired_oof(combined, "N3", right)
                       for name, (_, right) in COMPARE.items()
                       if name in ("N3-N1", "N3-N2", "N3-N4", "N3-historical_M3")}
    mechanisms = mechanism_summary(records)
    checkpoint = checkpoint_comparison(records)
    preflight_diffs = audit["preflight"]["teacher_target_differences"]
    loo = {group: loo_auc(combined, group)
           for group in ("historical_D4", "historical_M3", *GROUPS)}
    execution = audit["execution"]

    n3_auc = metrics["N3"]["metrics"]["probability_AUC"]["mean"]
    complete = coverage["analysis_complete"] and coverage["test_contract_all_pass"] \
        and coverage["same_fold_starts_pass"]
    all_auc_greater = n3_auc is not None and complete and all(
        metrics[group]["metrics"]["probability_AUC"]["mean"] is not None
        and n3_auc > metrics[group]["metrics"]["probability_AUC"]["mean"]
        for group in ("historical_D4", "historical_M3", "N1", "N2", "N4"))
    robust_auc = complete and all(differences[name]["probability_AUC"]["leave_one_fold_out_min_pp"] is not None
                     and differences[name]["probability_AUC"]["leave_one_fold_out_min_pp"] > 0
                     for name in ("N3-N1", "N3-N2", "N3-N4",
                                  "N3-historical_M3", "N3-historical_D4"))
    conditions_nontrivial = (preflight_diffs["N2-N3"]["mean_abs"] > 1e-6
                             and preflight_diffs["N3-N4"]["mean_abs"] > 1e-6)
    teacher_has_diagnostic_signal = (mechanisms["N3"]["mean_teacher_train_accuracy"] or 0) > 0.5
    student_gradient_present = (mechanisms["N3"]["mean_KL_to_ph_embedding_gradient_norm"] or 0) > 0
    comparison_means = [differences["N3-historical_M3"][key]["mean"]["mean"]
                        for key in ("ACC", "SEN-ASD", "SPE-TD")]
    no_point_metric_regression_vs_m3 = all(value is not None and value >= 0
                                           for value in comparison_means)
    if (all_auc_greater and robust_auc and conditions_nontrivial
            and teacher_has_diagnostic_signal and student_gradient_present
            and no_point_metric_regression_vs_m3):
        decision_status = "CANDIDATE_GO_FOR_CONFIRMATION"
        rationale = ("N3 exceeds all registered AUC references, remains positive after every leave-one-fold-out check, "
                     "has measurable condition-specific teacher changes and nonzero student KL gradient, and its three "
                     "threshold metrics are not below historical M3. This remains an exploratory candidate only.")
    elif (not complete or (n3_auc is not None
          and metrics["historical_D4"]["metrics"]["probability_AUC"]["mean"] is not None
          and n3_auc > metrics["historical_D4"]["metrics"]["probability_AUC"]["mean"])
          or conditions_nontrivial or student_gradient_present):
        decision_status = "MIXED"
        rationale = ("Some prespecified method or performance signals are present, but the complete candidate criteria "
                     "are not all satisfied. Interpret the paired controls and fold sensitivity before proposing confirmation.")
    else:
        decision_status = "NO_GO"
        rationale = "N3 did not show both a useful teacher difference and a sufficient AUC signal under the frozen contrasts."
    decision = {
        "status": decision_status,
        "rationale": rationale,
        "checks": {
            "N3_AUC_higher_than_historical_D4_M3_N1_N2_N4": bool(all_auc_greater),
            "all_N3_AUC_leave_one_fold_out_directions_positive": bool(robust_auc),
            "real_vs_global_and_shuffled_teacher_nontrivial": bool(conditions_nontrivial),
            "N3_teacher_train_accuracy_above_chance": bool(teacher_has_diagnostic_signal),
            "N3_KL_has_student_gradient": bool(student_gradient_present),
            "N3_ACC_SEN_SPE_not_below_historical_M3": bool(no_point_metric_regression_vs_m3),
        },
        "classification_limit": "A positive decision is only CANDIDATE_GO_FOR_CONFIRMATION, not a confirmed innovation.",
    }
    summary = {
        "experiment": CFG["experiment"], "name": CFG["name"], "created_utc": audit["created_utc"],
        "status": "COMPLETE" if complete else "PARTIAL",
        "formal_metrics": metrics, "paired_differences": differences,
        "leave_one_fold_out_auc": loo, "patient_oof_changes": patient_changes,
        "mechanisms": mechanisms,
        "preflight_teacher_differences": preflight_diffs,
        "preflight": {"passed": audit["preflight"]["passed"],
                      "seed": audit["preflight"]["seed"], "fold": audit["preflight"]["fold"],
                      "native_forwards": audit["preflight"]["forward_count"],
                      "valid_queries": audit["preflight"]["common_valid_query_count"],
                      "shared_losses_students": "bitwise equal across N1-N4"},
        "checkpoint_comparison": checkpoint, "coverage_audit": coverage,
        "execution": execution, "invalid_artifacts": invalid,
        "decision": decision,
        "history_limits": [
            "Supervised RFE information crosses folds under the inherited historical protocol.",
            "The 871-node population graph is transductive and includes heldout nodes.",
            "VAE sampling and native evaluation dropout remain stochastic.",
            "PPU training trajectories are not fully deterministic.",
            "Single seed with ten folds and repeated method exploration on the same dataset.",
            "Non-imaging branch contains site, sex, and age only.",
            "Same-private-fold aggregation controls direct D4 prototype-reference label reuse only; it is not full leave-one-subject-out model training.",
        ],
        "code_sha256": audit["training_code_sha256"],
        "manifest_sha256": audit["manifest_sha256"],
        "source_sha256": audit["original_source_sha256"],
        "historical_inputs_sha256": {fold: item["sha256"] for fold, item in audit["anchors"].items()},
        "no_git_commit_or_push": True,
    }
    atomic_json(summary, BASE / "summary.json")
    (BASE / "report.md").write_text(build_report(summary), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
