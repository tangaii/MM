"""Independent PCDPA-CM artifact validation, paired analysis, and reporting."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
sys.path.insert(0, str(BASE))

import run


GROUPS = ("P1", "P2", "P3", "P4")
HIST = {
    "historical_D4": (ROOT / "experiments/PCDCA_20261019", "D4", True),
    "historical_M1": (ROOT / "experiments/PCDPAX_20261021", "M1", False),
    "historical_M3": (ROOT / "experiments/PCDPAX_20261021", "M3", False),
    "historical_N3": (ROOT / "experiments/PCDPAX2_20261021", "N3", False),
}
METRICS = ("ACC", "SEN-ASD", "SPE-TD", "probability_AUC")
PAIRS = {
    "P1-historical_M1": ("P1", "historical_M1"),
    "P2-P1": ("P2", "P1"),
    "P3-P2": ("P3", "P2"),
    "P3-P4": ("P3", "P4"),
    "P3-historical_M3": ("P3", "historical_M3"),
    "P3-historical_D4": ("P3", "historical_D4"),
    "P3-historical_N3": ("P3", "historical_N3"),
}


def unwrap(value, path: Path):
    if "payload" in value and "content_sha256" in value:
        if run.payload_hash(value["payload"]) != value["content_sha256"]:
            raise AssertionError(f"Historical artifact checksum mismatch: {path}")
        return value["payload"]
    return value


def normalize_metrics(metrics: dict, old_format=False) -> dict:
    if old_format:
        return {"ACC": metrics["ACC"], "SEN-ASD": metrics["SEN"], "SPE-TD": metrics["SPE"],
                "probability_AUC": metrics["AUC"]}
    return {key: metrics[key] for key in METRICS}


def recompute(labels, probability, prediction) -> dict:
    return run.formal_metrics(labels, probability, prediction)


def load_history(name: str) -> dict:
    directory, artifact_group, wrapped = HIST[name]
    records = {}
    for fold in range(10):
        path = directory / "runs" / artifact_group / f"s911_f{fold}.pt"
        value = unwrap(run.load_torch(path), path)
        if (value.get("state"), value.get("group"), value.get("seed"), value.get("fold")) != (
                "complete", artifact_group, 911, fold):
            raise AssertionError(f"Historical artifact identity mismatch: {path}")
        oof = value["oof"]
        metrics = recompute(oof["labels"], oof["probability"], oof["prediction"])
        saved = normalize_metrics(value["metrics"], old_format=wrapped)
        if max(abs(metrics[key] - saved[key]) for key in METRICS) > 1e-12:
            raise AssertionError(f"Historical metrics do not recompute: {path}")
        records[fold] = {"artifact": value, "oof": oof, "metrics": metrics,
                         "artifact_sha256": run.sha_file(path), "historical": True,
                         "stored_probability": np.asarray(oof["probability"])}
    return records


def load_new(audit: dict) -> tuple[dict, list]:
    records, failures = {}, []
    for group in GROUPS:
        for fold in range(10):
            try:
                value = run.valid_result(group, 911, fold, audit, required=False)
            except Exception as error:
                failures.append({"group": group, "fold": fold, "error": repr(error)})
                continue
            if value is not None:
                records[(group, fold)] = {"artifact": value, "oof": value["oof"],
                    "metrics": value["metrics"],
                    "artifact_sha256": run.sha_file(run.result_path(group, 911, fold)),
                    "historical": False}
    return records, failures


def fold_records(records: dict, group: str) -> dict:
    if group in records and isinstance(records[group], dict):
        return records[group]
    return {fold: records[(group, fold)] for fold in range(10) if (group, fold) in records}


def mean_sd(values, scale=100.0) -> dict:
    values = np.asarray(values, dtype=np.float64) * scale
    return {"mean": float(values.mean()) if len(values) else None,
            "sample_sd": float(values.std(ddof=1)) if len(values) > 1 else (0.0 if len(values) else None),
            "n": int(len(values))}


def summarize_group(records: dict, group: str) -> dict:
    folds = fold_records(records, group)
    result = {"folds_complete": len(folds),
              "metrics": {metric: mean_sd([folds[f]["metrics"][metric] for f in sorted(folds)])
                          for metric in METRICS}}
    if folds:
        result.update({"mean_best_epoch": float(np.mean([folds[f]["artifact"].get("best_epoch", 0)
                                                            for f in folds])),
                       "best_epoch_by_fold": {str(f): int(folds[f]["artifact"].get("best_epoch", 0))
                                               for f in folds},
                       "mean_training_seconds": float(np.mean([folds[f]["artifact"].get("training_seconds", 0)
                                                                  for f in folds]))})
    return result


def paired_delta(records: dict, left: str, right: str, metric: str) -> dict:
    a, b = fold_records(records, left), fold_records(records, right)
    deltas = {fold: 100 * (a[fold]["metrics"][metric] - b[fold]["metrics"][metric])
              for fold in sorted(set(a) & set(b))}
    loo = {str(omitted): float(np.mean([d for fold, d in deltas.items() if fold != omitted]))
           for omitted in deltas if len(deltas) > 1}
    return {"metric": metric, "fold_delta_pp": {str(k): float(v) for k, v in deltas.items()},
            "mean_delta_pp": mean_sd(list(deltas.values()), scale=1.0),
            "positive_folds": int(sum(v > 0 for v in deltas.values())),
            "negative_folds": int(sum(v < 0 for v in deltas.values())),
            "leave_one_fold_out_mean_delta_pp": loo,
            "leave_one_fold_out_min_pp": min(loo.values()) if loo else None,
            "leave_one_fold_out_max_pp": max(loo.values()) if loo else None,
            "leave_one_fold_out_direction_changes": bool(loo and min(loo.values()) <= 0 <= max(loo.values())),
            "fold_2_6_9_pp": {str(f): float(deltas[f]) for f in (2, 6, 9) if f in deltas}}


def loo_auc(records: dict, group: str) -> dict:
    folds = fold_records(records, group)
    values = {str(f): float(np.mean([folds[x]["metrics"]["probability_AUC"]
                                     for x in folds if x != f]) * 100)
              for f in folds if len(folds) > 1}
    return {"fold_excluded_mean_auc_percent": values,
            "min_percent": min(values.values()) if values else None,
            "max_percent": max(values.values()) if values else None}


def collect_oof(records: dict, group: str) -> dict:
    result = {}
    for fold, record in fold_records(records, group).items():
        oof = record["oof"]
        indices, ids, labels = oof["indices"], oof["ids"], oof["labels"]
        probabilities, predictions, logits = oof["probability"], oof["prediction"], oof["logits"]
        cpu = np.exp(np.asarray(logits, dtype=np.float64)[:, 0] -
                     np.logaddexp(np.asarray(logits, dtype=np.float64)[:, 0],
                                  np.asarray(logits, dtype=np.float64)[:, 1]))
        if np.max(np.abs(cpu - np.asarray(probabilities, dtype=np.float64))) > 2e-6:
            raise AssertionError(f"CPU probability recomputation exceeds 2e-6: {group}/{fold}")
        if not np.array_equal(np.asarray(logits).argmax(axis=1), np.asarray(predictions)):
            raise AssertionError(f"OOF predictions differ from logits argmax: {group}/{fold}")
        for i, subject_id, y, p, pred, logit in zip(indices, ids, labels, probabilities, predictions, logits):
            if subject_id in result:
                raise AssertionError(f"Duplicate OOF subject: {group}/{subject_id}")
            result[subject_id] = {"index": int(i), "fold": int(fold), "label": int(y),
                                  "probability": float(p), "prediction": int(pred),
                                  "logits": [float(v) for v in logit]}
    return result


def patient_changes(records: dict, left: str, right: str) -> dict:
    a, b = collect_oof(records, left), collect_oof(records, right)
    common = sorted(set(a) & set(b))
    old_to_new, new_to_old = [], []
    class_net = {"ASD": 0, "TD": 0}
    for subject in common:
        x, y = a[subject], b[subject]
        if x["label"] != y["label"] or x["fold"] != y["fold"]:
            raise AssertionError(f"OOF labels or folds differ for {subject}: {left}/{right}")
        if x["prediction"] != x["label"] and y["prediction"] == y["label"]:
            old_to_new.append(subject)
            class_net["ASD" if x["label"] == 0 else "TD"] += 1
        elif x["prediction"] == x["label"] and y["prediction"] != y["label"]:
            new_to_old.append(subject)
            class_net["ASD" if x["label"] == 0 else "TD"] -= 1
    return {"n_common": len(common), "coverage_complete": len(common) == 871,
            "old_error_to_new_correct": len(old_to_new), "old_correct_to_new_error": len(new_to_old),
            "net_correct_change": len(old_to_new) - len(new_to_old),
            "ASD_net_correct_change": class_net["ASD"], "TD_net_correct_change": class_net["TD"],
            "error_to_correct_ids": old_to_new, "correct_to_error_ids": new_to_old}


def describe(values) -> dict:
    a = np.asarray(values, dtype=np.float64).reshape(-1)
    if not a.size:
        return {"n": 0, "mean": None, "sample_sd": None, "min": None, "max": None}
    return {"n": int(a.size), "mean": float(a.mean()),
            "sample_sd": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "min": float(a.min()), "max": float(a.max())}


def mechanism(records: dict) -> dict:
    result = {}
    for group in GROUPS:
        artifacts = [records[(group, f)]["artifact"] for f in range(10) if (group, f) in records]
        samples = [epoch for artifact in artifacts for epoch in artifact["epochs"]
                   if "pair_mechanism" in epoch]
        fields = ("pair_ess_asd", "pair_ess_td", "pair_min_ess", "pair_beta",
                  "pair_age_group_mean_abs_gap", "pair_sex_female_fraction_abs_gap",
                  "pair_site_total_variation", "d4_independent_age_group_mean_abs_gap",
                  "d4_independent_sex_female_fraction_abs_gap", "d4_independent_site_total_variation",
                  "pair_weight_mean_l1_vs_independent_d4", "pair_weight_mean_l1_vs_p1_triplet",
                  "pair_prototype_mean_l2_vs_d4", "pair_class_prototype_distance",
                  "pair_ce_low_ess", "pair_ce_high_ess", "d4_ce_low_ess", "d4_ce_high_ess")
        stats = {field: describe([sample["pair_mechanism"][field]["mean"] for sample in samples
                                 if field in sample["pair_mechanism"]]) for field in fields}
        stats.update({"best_epoch": describe([x["best_epoch"] for x in artifacts]),
                      "epochs_run": describe([x["epochs_run"] for x in artifacts]),
                      "training_seconds": describe([x["training_seconds"] for x in artifacts]),
                      "best_core_change_from_start": describe([
                          x["best_core_change_from_start"]["relative_l2"] for x in artifacts]),
                      "best_core_component_change_from_start": {
                          key: describe([x["best_core_change_from_start"]["per_component_relative_l2"][key]
                                         for x in artifacts])
                          for key in (artifacts[0]["best_core_change_from_start"]["per_component_relative_l2"]
                                      if artifacts else ())},
                      "sampled_epochs_per_fold": {str(x["fold"]): [e["epoch"] for e in x["epochs"]
                          if "pair_mechanism" in e] for x in artifacts}})
        stats["pair_accuracy_low_ess"] = describe([
            sample["pair_mechanism"]["pair_accuracy_low_ess"] for sample in samples
            if sample["pair_mechanism"]["pair_accuracy_low_ess"] is not None])
        stats["pair_accuracy_high_ess"] = describe([
            sample["pair_mechanism"]["pair_accuracy_high_ess"] for sample in samples
            if sample["pair_mechanism"]["pair_accuracy_high_ess"] is not None])
        c_stats = [sample["pair_mechanism"].get("phenotype_embedding") for sample in samples
                   if sample["pair_mechanism"].get("phenotype_embedding")]
        if c_stats:
            for key in ("C_std", "C_vs_age_sex_kernel_correlation", "C_vs_same_site_correlation",
                        "per_dimension_variance_mean"):
                stats[f"phenotype_{key}"] = describe([x[key] for x in c_stats])
            stats["phenotype_C_distribution"] = {
                field: describe([x["C"][field] for x in c_stats]) for field in ("mean", "sd", "p10", "p50", "p90")}
            stats["phenotype_raw_dimension_variance"] = describe([
                x["raw_per_dimension_variance"]["mean"] for x in c_stats])
        gradients = [e.get("embedding_gradient_diagnostics", {}) for x in artifacts for e in x["epochs"]
                     if "embedding_gradient_diagnostics" in e]
        for key in ("img_native_norm", "img_d4_norm", "img_pair_norm", "img_combined_aux_norm",
                    "img_native_d4_cosine", "img_native_pair_cosine", "img_native_aux_cosine",
                    "ph_native_norm", "ph_ce_norm", "pair_direct_ph_gradient_norm"):
            stats[f"gradient_{key}"] = describe([g[key] for g in gradients if g.get(key) is not None])
        stats["mean_lambda_img"] = describe([e["lambda_img"] for x in artifacts for e in x["epochs"]])
        stats["mean_lambda_ph"] = describe([e["lambda_ph"] for x in artifacts for e in x["epochs"]])
        result[group] = stats
    return result


def core_differences(left: dict, right: dict) -> dict:
    a, b = left["best_model_state"], right["best_model_state"]
    modules = {}
    for key, av in a["state"].items():
        bv = b["state"][key]
        prefix = key.split(".")[0]
        slot = modules.setdefault(prefix, [0.0, 0.0])
        slot[0] += float((av.float() - bv.float()).square().sum())
        slot[1] += float(av.float().square().sum())
    for key in ("rp_weights",):
        av, bv = a[key].float(), b[key].float()
        modules[key] = [float((av - bv).square().sum()), float(av.square().sum())]
    relative = {key: float(np.sqrt(diff / max(norm, 1e-30))) for key, (diff, norm) in modules.items()}
    numerator = sum(x[0] for x in modules.values())
    denominator = sum(x[1] for x in modules.values())
    return {"overall_relative_l2": float(np.sqrt(numerator / max(denominator, 1e-30))),
            "per_component_relative_l2": relative,
            "best_epoch_delta": int(left["best_epoch"] - right["best_epoch"])}


def synthetic_self_test() -> dict:
    # Exercise every statistics and serialization field before the source freeze.
    toy = {g: {f: {"metrics": {m: (0.5 + .01 * f) for m in METRICS},
                   "oof": {"indices": np.array([f]), "ids": [f"s{f}"], "labels": np.array([0]),
                           "probability": np.array([0.7310585786300049]), "prediction": np.array([0]),
                           "logits": np.array([[1., 0.]])}, "artifact": {"best_epoch": f}}
              for f in range(10)} for g in ("A", "B")}
    delta = paired_delta(toy, "A", "B", "ACC")
    summary = {"mean": mean_sd([.01, .02], scale=1), "sd": np.std([.01, .02], ddof=1)}
    patient = patient_changes(toy, "A", "B")
    toy_metrics = {metric: mean_sd([.5] * 10) for metric in METRICS}
    toy_report_summary = {
        "status": "synthetic", "coverage": {"tasks_complete": 40},
        "execution": {"gpu0": 20, "gpu1": 20},
        "decision": {"status": "MIXED", "rationale": "synthetic validation"},
        "formal_metrics": {"P1": {"folds_complete": 10, "metrics": toy_metrics}},
        "paired_differences": {"P3-P4": {metric: paired_delta(toy, "A", "B", metric)
                                            for metric in METRICS}},
        "audit": {"mathematical_tests_passed": True, "preflight_passed": True},
        "mechanisms": {"P1": {"pair_age_group_mean_abs_gap": summary["mean"],
                                  "d4_independent_age_group_mean_abs_gap": summary["mean"]},
                       "P3": {"phenotype_C_std": summary["mean"]}},
        "checkpoint_sensitivity": {}, "patient_oof_changes": {}}
    report = report_text(toy_report_summary)
    json.dumps({"delta": delta, "summary": summary, "patient": patient, "report": report}, allow_nan=False)
    checks = {"metric_keys_exactly_four": tuple(METRICS) == ("ACC", "SEN-ASD", "SPE-TD", "probability_AUC"),
              "mean_and_sample_sd": summary["mean"]["mean"] == .015 and np.isclose(summary["mean"]["sample_sd"], .007071067811865475),
              "paired_fields_and_fold_delta": delta["mean_delta_pp"]["mean"] == 0.0 and len(delta["fold_delta_pp"]) == 10,
              "loo_and_fold_2_6_9": len(delta["leave_one_fold_out_mean_delta_pp"]) == 10 and set(delta["fold_2_6_9_pp"]) == {"2", "6", "9"},
              "patient_oof_pairing": patient["n_common"] == 10 and not patient["coverage_complete"],
              "report_format_and_metric_columns": "# PCDPA-CM 实验报告" in report and all(
                  metric in report for metric in METRICS),
              "json_serialization": True}
    checks = {key: bool(value) for key, value in checks.items()}
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise AssertionError(f"Analyzer synthetic self-test failed: {failed}")
    return {"passed": True, "checks": checks}


def csv_outputs(rows: dict):
    with (BASE / "fold_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["group", "fold", *METRICS, "best_epoch", "training_seconds"])
        writer.writeheader()
        for group in [*HIST, *GROUPS]:
            for fold, record in sorted(fold_records(rows, group).items()):
                writer.writerow({"group": group, "fold": fold,
                                 **record["metrics"], "best_epoch": record["artifact"].get("best_epoch"),
                                 "training_seconds": record["artifact"].get("training_seconds")})
    with (BASE / "oof.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["group", "subject_id", "index", "fold", "label", "probability_ASD", "prediction", "logit0", "logit1"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for group in [*HIST, *GROUPS]:
            for subject, row in collect_oof(rows, group).items():
                writer.writerow({"group": group, "subject_id": subject, "index": row["index"],
                                 "fold": row["fold"], "label": row["label"],
                                 "probability_ASD": row["probability"], "prediction": row["prediction"],
                                 "logit0": row["logits"][0], "logit1": row["logits"][1]})


def report_text(summary: dict) -> str:
    def pp(value):
        return "n/a" if value is None else f"{value:+.3f}"

    lines = ["# PCDPA-CM 实验报告", "",
             f"状态：{summary['status']}；正式任务 {summary['coverage']['tasks_complete']}/40，"
             f"GPU0 {summary['execution']['gpu0']}/20，GPU1 {summary['execution']['gpu1']}/20。",
             f"科研判断：{summary['decision']['status']}。{summary['decision']['rationale']}", "",
             "## 四项正式指标（十折均值 ± 样本 SD，百分比）", "",
             "| 组别 | ACC | SEN-ASD | SPE-TD | 概率 AUC |", "|---|---:|---:|---:|---:|"]
    for group in [*HIST, *GROUPS]:
        item = summary["formal_metrics"].get(group)
        if not item or not item["folds_complete"]:
            lines.append(f"| {group} | 未完成 | 未完成 | 未完成 | 未完成 |")
            continue
        cells = []
        for metric in METRICS:
            v = item["metrics"][metric]
            cells.append(f"{v['mean']:.3f} ± {v['sample_sd']:.3f}")
        lines.append(f"| {group} | " + " | ".join(cells) + " |")
    lines += ["", "## 配对差值（百分点）", "", "| 比较 | 指标 | 均值 ± 样本 SD | 正向折数 | LOO方向变化 |", "|---|---|---:|---:|---|"]
    for pair, metrics in summary["paired_differences"].items():
        for metric in METRICS:
            v = metrics[metric]
            m = v["mean_delta_pp"]
            lines.append(f"| {pair} | {metric} | {pp(m['mean'])} ± {pp(m['sample_sd'])} | "
                         f"{v['positive_folds']}/{len(v['fold_delta_pp'])} | {v['leave_one_fold_out_direction_changes']} |")
    lines += ["", "## 机制与审计", "",
              f"三元边际 oracle、D4 等价和 heldout 不变性审计：{summary['audit']['mathematical_tests_passed']}；"
              f"真实 fold0 单次 forward：{summary['audit']['preflight_passed']}。",
              f"P2/P3 的 C 标准差汇总：{json.dumps(summary['mechanisms'].get('P3', {}).get('phenotype_C_std', {}), ensure_ascii=False)}。",
              f"P1 成对年龄差和独立 D4 年龄差：{json.dumps({k: summary['mechanisms'].get('P1', {}).get(k) for k in ('pair_age_group_mean_abs_gap', 'd4_independent_age_group_mean_abs_gap')}, ensure_ascii=False)}。",
              f"P3/P4 最佳 checkpoint 相对差：{json.dumps(summary['checkpoint_sensitivity'].get('P3-P4', {}), ensure_ascii=False)}。",
              f"最佳 checkpoint 相对 train_start 的参数变化：{json.dumps({g: summary['mechanisms'].get(g, {}).get('best_core_change_from_start', {}) for g in GROUPS}, ensure_ascii=False)}。",
              f"高低 ESS 的成对/D4 CE 对照：{json.dumps({g: {k: summary['mechanisms'].get(g, {}).get(k) for k in ('pair_ce_low_ess', 'pair_ce_high_ess', 'd4_ce_low_ess', 'd4_ce_high_ess', 'pair_accuracy_low_ess', 'pair_accuracy_high_ess')} for g in GROUPS}, ensure_ascii=False)}。",
              f"患者级 OOF 比较：{json.dumps(summary['patient_oof_changes'], ensure_ascii=False)}。",
              f"历史 source SHA 与 anchor 输入 SHA 已记录在 summary.json；历史 D4/M1/M3/N3 均只读。",
              "保留限制：监督式 RFE 跨折信息泄漏、871 节点 transductive 图、VAE 随机采样和原模型 eval 随机行为均未改变。",
              "非影像 GTUNet 与影像分支共享原生融合人口图，因此 z_ph 不代表独立纯人口统计因子。", ""]
    return "\n".join(lines)


def scientific_decision(summary: dict) -> dict:
    if summary["coverage"]["tasks_complete"] < 40:
        return {"status": "PARTIAL", "criteria": {}, "rationale": "正式 40 任务未全部完成，不能作完整科研判断。"}
    paired = summary["paired_differences"]
    p3p4_auc = paired["P3-P4"]["probability_AUC"]["mean_delta_pp"]["mean"]
    p3p4_acc = paired["P3-P4"]["ACC"]["mean_delta_pp"]["mean"]
    p3p4_sen = paired["P3-P4"]["SEN-ASD"]["mean_delta_pp"]["mean"]
    p3p4_spe = paired["P3-P4"]["SPE-TD"]["mean_delta_pp"]["mean"]
    p3_balance = abs(summary["formal_metrics"]["P3"]["metrics"]["SEN-ASD"]["mean"] -
                     summary["formal_metrics"]["P3"]["metrics"]["SPE-TD"]["mean"])
    p4_balance = abs(summary["formal_metrics"]["P4"]["metrics"]["SEN-ASD"]["mean"] -
                     summary["formal_metrics"]["P4"]["metrics"]["SPE-TD"]["mean"])
    p3p4_balanced = p3_balance < p4_balance
    p1m1 = paired["P1-historical_M1"]["probability_AUC"]["mean_delta_pp"]["mean"]
    p2p1 = paired["P2-P1"]["probability_AUC"]["mean_delta_pp"]["mean"]
    p3p2 = paired["P3-P2"]["probability_AUC"]["mean_delta_pp"]["mean"]
    p3m3 = paired["P3-historical_M3"]["probability_AUC"]["mean_delta_pp"]["mean"]
    p3d4 = paired["P3-historical_D4"]["probability_AUC"]["mean_delta_pp"]["mean"]
    mech = summary["mechanisms"]
    c_std = mech.get("P3", {}).get("phenotype_C_std", {}).get("mean")
    weight_delta = mech.get("P3", {}).get("pair_weight_mean_l1_vs_p1_triplet", {}).get("mean")
    pair_age = mech.get("P1", {}).get("pair_age_group_mean_abs_gap", {}).get("mean")
    d4_age = mech.get("P1", {}).get("d4_independent_age_group_mean_abs_gap", {}).get("mean")
    ess = mech.get("P3", {}).get("pair_min_ess", {}).get("mean")
    loo_stable = not paired["P3-P4"]["probability_AUC"]["leave_one_fold_out_direction_changes"]
    criteria = {
        "P3_AUC_better_than_P4": p3p4_auc > 0,
        "P3_ACC_or_SEN_SPE_balance_better_than_P4": p3p4_acc > 0 or p3p4_balanced,
        "P2_incremental_relation_value": p2p1 > 0 and (weight_delta or 0) > 1e-6,
        "P3_removing_phCE_helpful": p3p2 > 0,
        "P3_better_than_M3": p3m3 > 0,
        "P3_better_than_D4": p3d4 > 0,
        "pairing_improves_demographic_comparability": pair_age is not None and d4_age is not None and pair_age < d4_age,
        "weights_nontrivially_differ_from_P1": (weight_delta or 0) > 1e-6,
        "C_not_collapsed": c_std is not None and c_std > 1e-5,
        "effective_reference_ESS_reasonable": ess is not None and ess > 8,
        "not_single_fold_dominated": loo_stable,
        "train_only_reference_and_code_audit_pass": summary["audit"]["reference_label_bounds_pass"] and summary["audit"]["all_protocol_checks_pass"],
        "P1_pairing_improves_on_M1": p1m1 > 0}
    if all(criteria.values()):
        status, rationale = "CANDIDATE_GO_FOR_CONFIRMATION", "所有预注册候选条件均满足；仍需独立种子/队列确认，不能宣称达到 MICCAI 2027 创新要求。"
    elif p3p4_auc <= 0 and p3m3 <= 0 and p3d4 <= 0:
        status, rationale = "NO-GO", "P3 未优于错配负对照及历史 M3/D4，当前证据不支持候选方法。"
    else:
        status, rationale = "MIXED", "性能或机制条件有正有负；不能把提升归因于正确跨模态患者对应。"
    return {"status": status, "criteria": criteria, "rationale": rationale}


def append_handoff(summary: dict, report: str) -> dict:
    path = ROOT / "handoff.md"
    marker = "PCDPACM_20261021_END"
    with (BASE / "analyze.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = path.read_bytes()
        if marker.encode() in before:
            return {"appended": False, "idempotent_skip": True,
                    "marker_count": before.count(marker.encode()), "after_sha256": run.sha_file(path)}
        audit = run.read_json(BASE / "audit.json")
        expected = audit["handoff_before"]
        if len(before) != expected["bytes"] or run.sha_file(path) != expected["sha256"]:
            raise AssertionError("handoff changed after audit; preserving it and saving pending text")
        text = ("\n\n<!-- PCDPACM_20261021_START -->\n\n" + report + "\n\n"
                + f"训练任务 {summary['coverage']['tasks_complete']}/40；GPU0 {summary['execution']['gpu0']}/20；"
                + f"GPU1 {summary['execution']['gpu1']}/20。科研判断：{summary['decision']['status']}。\n\n"
                + "数学定义：P1=D4+真实年龄性别三元成对影像原型+真实 ph D4 CE；P2=P1 并以 detach 的真实 z_ph 相似度 C 修正 query-ASD、query-TD、ASD-TD 三条关系；P3=P2 去掉 ph CE；P4=P3 仅将 C 中的 z_ph 按训练集内同站点 label-blind derangement 错配。所有 pair prototype 使用 train-only、非自身且跨私有折参考、精确矩阵边际、ESS 收缩与 D4 回退。\n\n"
                + f"历史 D4/M1/M3/N3 对照为只读；四指标、配对差、患者级 OOF、匹配质量、C 分布、ESS、梯度和 checkpoint 诊断见 [report.md](experiments/PCDPACM_20261021/report.md) 与 summary.json。\n\n"
                + f"训练秒数累计：{summary['execution']['worker_training_seconds']}；队列墙钟秒数：{summary['execution']['queue_wall_seconds']}；失败/重试：{summary['execution']['failures']}/{summary['execution']['retries']}。\n\n"
                + f"训练代码 SHA：{summary['audit']['training_code_sha256']}；原始源码和十折输入 SHA 保存在 audit.json/manifest.json。保留监督 RFE 跨折信息泄漏及 871 节点 transductive 图等历史限制。\n\n"
                + "下一步：若结果支持候选，须在未参与历史探索的独立随机种子和未见站点协议上验证；本轮不追加任务。\n\n"
                + marker + "\n")
        pending = BASE / "handoff_pending.md"
        run.atomic_bytes(text.encode(), pending)
        after_data = before + text.encode()
        run.atomic_bytes(after_data, path)
        after = path.read_bytes()
        prefix_sha = run.sha_bytes(after[:len(before)])
        if after[:len(before)] != before or prefix_sha != expected["sha256"]:
            raise AssertionError("Append-only handoff prefix validation failed")
        count = after.count(marker.encode())
        if count != 1:
            raise AssertionError(f"Expected one end marker, found {count}")
        pending.unlink(missing_ok=True)
        return {"appended": True, "before_bytes": len(before), "before_sha256": expected["sha256"],
                "after_prefix_sha256": prefix_sha, "after_bytes": len(after),
                "after_sha256": run.sha_file(path), "end_marker_count": count}


def run_analysis(final: bool, stop_reason=None) -> dict:
    audit = run.guard(require_manifest=bool(read_audit().get("formal_authorized")))
    rows = {name: load_history(name) for name in HIST}
    new, failures = load_new(audit)
    for (group, fold), value in new.items():
        rows.setdefault(group, {})[fold] = value
    formal = {group: summarize_group(rows, group) for group in [*HIST, *GROUPS]}
    pairs = {name: {metric: paired_delta(rows, *comparison, metric) for metric in METRICS}
             for name, comparison in PAIRS.items()}
    patient = {name: patient_changes(rows, *comparison)
               for name, comparison in {"P3-P1": ("P3", "P1"), "P3-P2": ("P3", "P2"),
                                        "P3-P4": ("P3", "P4"), "P3-historical_M3": ("P3", "historical_M3")}.items()
               if all(len(collect_oof(rows, group)) == 871 for group in comparison)}
    complete_by_group = {group: len(fold_records(rows, group)) for group in GROUPS}
    total = sum(complete_by_group.values())
    new_counts = {"gpu0": sum(records["artifact"]["physical_gpu"] == 0 for records in new.values()),
                  "gpu1": sum(records["artifact"]["physical_gpu"] == 1 for records in new.values())}
    queue = run.read_json(BASE / "queue.json") if (BASE / "queue.json").exists() else {}
    execution = audit.get("execution", {})
    queue_wall = queue.get("elapsed_seconds", execution.get("queue_wall_seconds"))
    worker_train = float(sum(record["artifact"].get("training_seconds", 0) for record in new.values()))
    checkpoint = {}
    if len(fold_records(rows, "P3")) and len(fold_records(rows, "P4")):
        common_checkpoint_folds = sorted(set(fold_records(rows, "P3")) & set(fold_records(rows, "P4")))
        by_fold = {str(f): core_differences(rows["P3"][f]["artifact"], rows["P4"][f]["artifact"])
                   for f in common_checkpoint_folds}
        checkpoint["P3-P4"] = {"by_fold": by_fold,
                               "mean_overall_relative_l2": float(np.mean([v["overall_relative_l2"]
                                                                           for v in by_fold.values()])),
                               "mean_best_epoch_delta": float(np.mean([v["best_epoch_delta"]
                                                                        for v in by_fold.values()]))}
    mechanisms = mechanism(rows)
    reference_bounds = True
    for group in GROUPS:
        for fold, record in fold_records(rows, group).items():
            prepared = record["artifact"]["prototype_preparation"]
            split = run.load_torch(run.anchor(911, fold) / "splits.pt")
            train_indices = set(map(int, split["train"]))
            recorded_train = set(map(int, prepared["train_indices"]))
            shuffle_sources = set(map(int, prepared["shuffle_sources"]))
            if recorded_train != train_indices or not shuffle_sources.issubset(train_indices):
                reference_bounds = False
            if prepared.get("uses_heldout_labels_or_covariates") is not False:
                reference_bounds = False
    all_protocol = bool((audit.get("mathematical_tests") or {}).get("passed")
                        and (audit.get("preflight") or {}).get("passed")
                        and audit.get("formal_authorized"))
    summary = {
        "experiment": run.CFG["experiment"], "status": "complete" if total == 40 else "partial",
        "stop_reason": stop_reason, "coverage": {"tasks_complete": total, "group_counts": complete_by_group,
                                                     "expected_tasks": 40, "new_oof_rows": 871 * sum(v == 10 for v in complete_by_group.values())},
        "formal_metrics": formal, "paired_differences": pairs,
        "patient_oof_changes": patient, "leave_one_fold_out_auc": {g: loo_auc(rows, g) for g in [*HIST, *GROUPS]},
        "mechanisms": mechanisms, "checkpoint_sensitivity": checkpoint,
        "decision": {}, "audit": {"training_code_sha256": audit["training_code_sha256"],
            "training_code_files_sha256": audit["training_code_files_sha256"],
            "runtime_environment": audit.get("runtime_environment", {}),
            "original_source_sha256": audit["original_source_sha256"],
            "historical_pcdca_prototype_sha256": audit["historical_pcdca_prototype_sha256"],
            "anchors": audit["anchors"], "historical_zero_retraining": audit["history_policy"],
            "mathematical_tests_passed": bool((audit.get("mathematical_tests") or {}).get("passed")),
            "preflight_passed": bool((audit.get("preflight") or {}).get("passed")),
            "reference_label_bounds_pass": reference_bounds, "all_protocol_checks_pass": all_protocol,
            "manifest_sha256": audit.get("manifest_sha256"), "input_context_sha256": audit["context_sha256"],
            "raw_feature_tensor_sha256": audit["raw_feature_tensor_sha256"]},
        "artifact_sha256": {f"{g}/911/{f}": rows[g][f]["artifact_sha256"]
                            for g in GROUPS for f in fold_records(rows, g)},
        "execution": {"gpu0": new_counts["gpu0"], "gpu1": new_counts["gpu1"],
            "worker_training_seconds": worker_train, "queue_wall_seconds": queue_wall,
            "failures": queue.get("failures", []), "retries": len(queue.get("retries", [])),
            "peak_workers": queue.get("peak_workers"), "resource_samples": queue.get("resource_samples", []),
            "formal_neural_training": "PPU only", "RFE_fits": 0, "VAE_pretrains": 0,
            "new_trainable_parameters": 0},
        "analysis_failures": failures, "statistics": "ten-fold mean and sample SD (ddof=1); fold deltas are paired on fixed seed/fold; historical comparisons are read-only and not strict causal estimates",
        "created_utc": run.utc(), "handoff": {"appended": False}}
    summary["decision"] = scientific_decision(summary)
    csv_outputs(rows)
    report = report_text(summary)
    run.atomic_bytes(report.encode(), BASE / "report.md")
    run.atomic_json(summary, BASE / "summary.json")
    if final:
        try:
            summary["handoff"] = append_handoff(summary, report)
        except Exception as error:
            summary["handoff"] = {"appended": False, "error": repr(error),
                                  "pending_append_content": report}
            run.atomic_json(summary, BASE / "summary.json")
            raise
        summary["final_verified_utc"] = run.utc()
        run.atomic_json(summary, BASE / "summary.json")
    return summary


def read_audit():
    return run.read_json(BASE / "audit.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--stop-reason")
    args = parser.parse_args()
    if args.self_test:
        result = synthetic_self_test()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    result = run_analysis(args.final, args.stop_reason)
    print(json.dumps({"status": result["status"], "coverage": result["coverage"],
                      "decision": result["decision"], "handoff": result["handoff"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
