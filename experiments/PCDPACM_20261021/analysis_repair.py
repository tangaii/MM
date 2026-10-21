"""Post-training read-only repair for PCDPA-CM analysis aggregation/reporting."""
from __future__ import annotations

import argparse
import fcntl
import json
from pathlib import Path
import sys

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import analyze
import run

BASE_REPORT_TEXT = analyze.report_text


PAIR_DISTRIBUTIONS = (
    "pair_ess_asd", "pair_ess_td", "pair_min_ess", "pair_beta",
    "pair_age_group_mean_abs_gap", "pair_sex_female_fraction_abs_gap",
    "pair_site_total_variation", "d4_independent_age_group_mean_abs_gap",
    "d4_independent_sex_female_fraction_abs_gap", "d4_independent_site_total_variation",
    "pair_weight_mean_l1_vs_independent_d4", "pair_weight_mean_l1_vs_p1_triplet",
    "pair_prototype_mean_l2_vs_d4", "pair_class_prototype_distance",
    "pair_ce_low_ess", "pair_ce_high_ess", "d4_ce_low_ess", "d4_ce_high_ess",
)
PAIR_SCALARS = (
    "pair_beta_zero_fraction", "pair_fallback_fraction", "pair_supported_queries",
    "d4_valid_queries", "ess_median_split",
)
GRADIENT_FIELDS = (
    "img_native_norm", "img_d4_norm", "img_pair_norm", "img_combined_aux_norm",
    "img_native_d4_cosine", "img_native_pair_cosine", "img_native_aux_cosine",
    "ph_native_norm", "ph_ce_norm", "pair_direct_ph_gradient_norm",
)
PARAMETER_GRADIENT_FIELDS = ("img_unet", "ph_unet", "rp_attention", "mm_attention", "classifier")
BUDGET_FIELDS = (
    "native_gradient_norm", "auxiliary_gradient_norm", "lambda",
    "weighted_gradient_ratio", "gradient_cosine",
)


def mechanism_fixed(records: dict) -> dict:
    """Read the persisted group -> fold structure used by run_analysis."""
    result = {}
    for group in analyze.GROUPS:
        folds = analyze.fold_records(records, group)
        artifacts = [folds[f]["artifact"] for f in sorted(folds)]
        samples = [epoch for artifact in artifacts for epoch in artifact.get("epochs", [])
                   if "pair_mechanism" in epoch]
        stats = {
            field: analyze.describe([
                sample["pair_mechanism"][field]["mean"] for sample in samples
                if field in sample["pair_mechanism"]
                and sample["pair_mechanism"][field].get("mean") is not None
            ]) for field in PAIR_DISTRIBUTIONS
        }
        stats.update({field: analyze.describe([
            sample["pair_mechanism"][field] for sample in samples
            if sample["pair_mechanism"].get(field) is not None
        ]) for field in PAIR_SCALARS})

        stats.update({
            "best_epoch": analyze.describe([x["best_epoch"] for x in artifacts]),
            "epochs_run": analyze.describe([x["epochs_run"] for x in artifacts]),
            "training_seconds": analyze.describe([x["training_seconds"] for x in artifacts]),
            "best_core_change_from_start": analyze.describe([
                x["best_core_change_from_start"]["relative_l2"] for x in artifacts
            ]),
            "best_core_component_change_from_start": {
                key: analyze.describe([
                    x["best_core_change_from_start"]["per_component_relative_l2"][key]
                    for x in artifacts
                ]) for key in (artifacts[0]["best_core_change_from_start"]["per_component_relative_l2"]
                               if artifacts else ())
            },
            "sampled_epochs_per_fold": {
                str(x["fold"]): [e["epoch"] for e in x.get("epochs", [])
                                 if "pair_mechanism" in e]
                for x in artifacts
            },
        })
        for key in ("pair_accuracy_low_ess", "pair_accuracy_high_ess"):
            stats[key] = analyze.describe([
                sample["pair_mechanism"][key] for sample in samples
                if sample["pair_mechanism"].get(key) is not None
            ])

        embeddings = [sample["pair_mechanism"].get("phenotype_embedding") for sample in samples
                      if sample["pair_mechanism"].get("phenotype_embedding")]
        scalar_embedding_fields = (
            "C_std", "C_vs_age_sex_kernel_correlation", "C_vs_same_site_correlation",
            "per_dimension_variance_mean",
        )
        for field in scalar_embedding_fields:
            stats[f"phenotype_{field}"] = analyze.describe([x[field] for x in embeddings])
        for field in ("C", "cosine", "raw_per_dimension_variance", "raw_per_dimension_std", "row_norm"):
            stats[f"phenotype_{field}"] = analyze.describe([x[field]["mean"] for x in embeddings])
        stats["phenotype_C_distribution"] = {
            field: analyze.describe([x["C"][field] for x in embeddings])
            for field in ("mean", "sd", "p10", "p50", "p90", "min", "max")
        }

        sampled_epochs = [epoch for artifact in artifacts for epoch in artifact.get("epochs", [])]
        gradients = [epoch.get("embedding_gradient_diagnostics", {}) for epoch in sampled_epochs]
        for key in GRADIENT_FIELDS:
            stats[f"gradient_{key}"] = analyze.describe([
                item[key] for item in gradients if item.get(key) is not None
            ])
        for key in PARAMETER_GRADIENT_FIELDS:
            stats[f"parameter_gradient_{key}"] = analyze.describe([
                epoch["total_parameter_gradient_norms"][key] for epoch in sampled_epochs
                if epoch.get("total_parameter_gradient_norms", {}).get(key) is not None
            ])
        for budget_name in ("image_gradient_budget", "phenotype_gradient_budget"):
            for field in BUDGET_FIELDS:
                stats[f"{budget_name}_{field}"] = analyze.describe([
                    epoch[budget_name][field] for epoch in sampled_epochs
                    if epoch.get(budget_name, {}).get(field) is not None
                ])
        for name, field in (("mean_pair_ce", "pair_ce"), ("mean_d4_ce", "image_d4_ce"),
                            ("mean_phenotype_real_ce", "phenotype_real_ce"),
                            ("mean_lambda_img", "lambda_img"), ("mean_lambda_ph", "lambda_ph")):
            stats[name] = analyze.describe([
                epoch[field] for epoch in sampled_epochs if epoch.get(field) is not None
            ])
        result[group] = stats
    return result


def _mean_sd(value: dict) -> str:
    if not value or value.get("mean") is None:
        return "n/a"
    sd = value.get("sample_sd")
    return f"{value['mean']:.4f} +/- {sd:.4f}" if sd is not None else f"{value['mean']:.4f}"


def _number(value, digits=3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _field(mechanisms: dict, group: str, name: str) -> dict:
    return mechanisms.get(group, {}).get(name, {})


def report_text(summary: dict) -> str:
    base = BASE_REPORT_TEXT(summary)
    mechanisms = summary.get("mechanisms", {})
    lines = [base, "## 逐折 AUC 配对差", "",
             "以下为左组减右组的十折概率 AUC 差（百分点）；历史对照只读且受训练随机轨迹影响。", ""]
    for pair, metrics in summary.get("paired_differences", {}).items():
        item = metrics.get("probability_AUC", {})
        folds = item.get("fold_delta_pp", {})
        values = ", ".join(f"f{fold}={float(folds[str(fold)]):+.3f}" for fold in range(10)
                            if str(fold) in folds)
        loo_min = item.get("leave_one_fold_out_min_pp")
        loo_max = item.get("leave_one_fold_out_max_pp")
        selected = item.get("fold_2_6_9_pp", {})
        selected_text = ", ".join(f"f{fold}={float(selected[str(fold)]):+.3f}"
                                   for fold in (2, 6, 9) if str(fold) in selected)
        lines.append(f"- **{pair}**：{values}；LOO范围 {_number(loo_min)} 至 {_number(loo_max)} pp；"
                     f"折2/6/9：{selected_text or 'n/a'}。")

    lines += ["", "## 匹配质量与参考 ESS", "",
              "数值为各训练 epoch 诊断样本的均值 +/- 样本 SD；年龄差单位为预处理年龄单位，站点差为总变差。", "",
              "| 组别 | 成对年龄差 | 独立D4年龄差 | 成对性别差 | 独立D4性别差 | 成对站点TV | D4站点TV | 最小ESS | beta=0比例 | 回退比例 | 权重L1 vs D4 | 权重L1 vs P1三元 | 原型L2 vs D4 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    pairing_keys = (
        "pair_age_group_mean_abs_gap", "d4_independent_age_group_mean_abs_gap",
        "pair_sex_female_fraction_abs_gap", "d4_independent_sex_female_fraction_abs_gap",
        "pair_site_total_variation", "d4_independent_site_total_variation", "pair_min_ess",
        "pair_beta_zero_fraction", "pair_fallback_fraction", "pair_weight_mean_l1_vs_independent_d4",
        "pair_weight_mean_l1_vs_p1_triplet", "pair_prototype_mean_l2_vs_d4",
    )
    for group in analyze.GROUPS:
        fields = " | ".join(_mean_sd(_field(mechanisms, group, key)) for key in pairing_keys)
        lines.append(f"| {group} | {fields} |")

    lines += ["", "## 非影像表示与错配", "",
              "C分布及嵌入诊断按P2/P3/P4报告；P4只对训练集内的非影像嵌入做同站点、label-blind错配。", "",
              "| 组别 | C标准差 | C均值 | C p10/p50/p90 | cosine均值 | 原始维方差均值 | C与年龄性别核相关 | C与同站点相关 | 权重L1 vs P1三元 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for group in ("P2", "P3", "P4"):
        c_dist = mechanisms.get(group, {}).get("phenotype_C_distribution", {})
        quantiles = "/".join(_number(c_dist.get(q, {}).get("mean"), 4) for q in ("p10", "p50", "p90"))
        fields = (
            _mean_sd(_field(mechanisms, group, "phenotype_C_std")),
            _mean_sd(c_dist.get("mean", {})), quantiles,
            _mean_sd(_field(mechanisms, group, "phenotype_cosine")),
            _mean_sd(_field(mechanisms, group, "phenotype_raw_per_dimension_variance")),
            _mean_sd(_field(mechanisms, group, "phenotype_C_vs_age_sex_kernel_correlation")),
            _mean_sd(_field(mechanisms, group, "phenotype_C_vs_same_site_correlation")),
            _mean_sd(_field(mechanisms, group, "pair_weight_mean_l1_vs_p1_triplet")),
        )
        lines.append(f"| {group} | " + " | ".join(fields) + " |")

    perturbation = summary.get("preflight_perturbation", {})
    lines += ["", "P3/P4真实fold0单forward预检中的参考权重扰动：", ""]
    for contrast in ("P1-P2", "P3-P4"):
        item = perturbation.get(contrast, {})
        lines.append(
            f"- {contrast} w0矩阵：mean_abs={_number(item.get('mean_abs'), 7)}，"
            f"max_abs={_number(item.get('max_abs'), 6)}，"
            f"非零差异项比例={_number(item.get('nonzero_fraction'), 4)}。"
        )
    lines.append(
        "预检确认P4使用同站点label-blind错配，并使w0参考权重矩阵出现上述变化；"
        "逐元素C差异矩阵未写入artifact，因此本报告不声称已量化每个C条目的变化。"
    )

    lines += ["", "## CE、梯度与训练更新", "",
              "梯度范数和余弦来自预先登记的诊断 epoch；直接pair对z_ph梯度应为0，ph原生支路仍由原生损失训练。", "",
              "| 组别 | pair CE | D4 CE | ph CE | lambda img | lambda ph | img native norm | img D4 norm | img pair norm | native-pair cosine | native-aux cosine | ph native norm | ph CE norm | pair直接ph梯度 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    gradient_keys = (
        "mean_pair_ce", "mean_d4_ce", "mean_phenotype_real_ce", "mean_lambda_img", "mean_lambda_ph",
        "gradient_img_native_norm", "gradient_img_d4_norm", "gradient_img_pair_norm",
        "gradient_img_native_pair_cosine", "gradient_img_native_aux_cosine",
        "gradient_ph_native_norm", "gradient_ph_ce_norm", "gradient_pair_direct_ph_gradient_norm",
    )
    for group in analyze.GROUPS:
        fields = " | ".join(_mean_sd(_field(mechanisms, group, key)) for key in gradient_keys)
        lines.append(f"| {group} | {fields} |")
    lines += ["", "按ESS中位数分层的成对/D4 CE与pair准确率：", ""]
    for group in analyze.GROUPS:
        labels = ("pair_ce_low_ess", "pair_ce_high_ess", "d4_ce_low_ess", "d4_ce_high_ess",
                  "pair_accuracy_low_ess", "pair_accuracy_high_ess")
        values = "; ".join(f"{key}={_mean_sd(_field(mechanisms, group, key))}" for key in labels)
        lines.append(f"- {group}: {values}。")

    repair = summary.get("analysis_repair", {})
    lines += ["", "## Checkpoint敏感性与分析修复", "",
              "P3/P4逐折最佳checkpoint变化：", "",
              "| 折 | overall relative L2 | best epoch差(P3-P4) |", "|---:|---:|---:|"]
    checkpoints = summary.get("checkpoint_sensitivity", {}).get("P3-P4", {}).get("by_fold", {})
    for fold in sorted(checkpoints, key=int):
        item = checkpoints[fold]
        lines.append(f"| {fold} | {_number(item.get('overall_relative_l2'), 6)} | {item.get('best_epoch_delta', 'n/a')} |")
    lines += ["", f"P3/P4平均checkpoint相对L2：{_number(summary.get('checkpoint_sensitivity', {}).get('P3-P4', {}).get('mean_overall_relative_l2'), 6)}；"
              f"平均best epoch差：{_number(summary.get('checkpoint_sensitivity', {}).get('P3-P4', {}).get('mean_best_epoch_delta'), 3)}。",
              "最佳checkpoint相对train_start的变化：", ""]
    for group in analyze.GROUPS:
        lines.append(f"- {group}: {_mean_sd(_field(mechanisms, group, 'best_core_change_from_start'))}。")
    lines += ["", f"原始analyze.py SHA256：{repair.get('original_analyze_sha256', 'n/a')}；"
              f"只读分析修复包装器 SHA256：{repair.get('wrapper_sha256', 'n/a')}。",
              "原问题是机制汇总读取了不存在的元组键；修复只重算已保存的正式artifact，训练源码、模型权重和历史目录均未改写。", ""]
    log_repair = repair.get("run_log_repair")
    if log_repair:
        lines += [
            f"run.log清理：原始SHA256 {log_repair['original_sha256']}，移除填充NUL {log_repair['nul_bytes_removed']}字节；"
            f"现为{log_repair['valid_json_rows']}条合法JSONL记录，SHA256 {log_repair['cleaned_sha256']}。",
            "逐折日志、queue.json和训练artifact未改动。", "",
        ]
    return "\n".join(lines)


def _synthetic_records() -> dict:
    records = {}
    pair = {field: {"mean": 1.0} for field in PAIR_DISTRIBUTIONS}
    pair.update({field: 0.25 for field in PAIR_SCALARS})
    pair.update({"pair_accuracy_low_ess": 0.5, "pair_accuracy_high_ess": 0.75,
                 "phenotype_embedding": {
                     "C_std": 0.2, "C_vs_age_sex_kernel_correlation": 0.3,
                     "C_vs_same_site_correlation": 0.1, "per_dimension_variance_mean": 0.4,
                     "C": {key: 0.5 for key in ("mean", "sd", "p10", "p50", "p90", "min", "max")},
                     "cosine": {"mean": 0.0}, "raw_per_dimension_variance": {"mean": 0.01},
                     "raw_per_dimension_std": {"mean": 0.1}, "row_norm": {"mean": 1.0},
                 }})
    grad = {key: 0.2 for key in GRADIENT_FIELDS}
    budget = {key: 0.1 for key in BUDGET_FIELDS}
    for group in analyze.GROUPS:
        records[group] = {}
        for fold in range(10):
            epoch = {"epoch": fold, "pair_mechanism": pair,
                     "embedding_gradient_diagnostics": grad,
                     "image_gradient_budget": budget, "phenotype_gradient_budget": budget,
                     "total_parameter_gradient_norms": {key: 0.1 for key in PARAMETER_GRADIENT_FIELDS},
                     "pair_ce": 0.5, "image_d4_ce": 0.6, "phenotype_real_ce": 0.7,
                     "lambda_img": 0.01, "lambda_ph": 0.02}
            artifact = {"fold": fold, "epochs": [epoch], "best_epoch": fold,
                        "epochs_run": fold + 1, "training_seconds": 10.0,
                        "best_core_change_from_start": {
                            "relative_l2": 0.1,
                            "per_component_relative_l2": {"img": 0.1},
                        }}
            records[group][fold] = {"artifact": artifact}
    return records


def self_test() -> dict:
    result = mechanism_fixed(_synthetic_records())
    checks = {
        "nested_group_fold_artifacts_read": result["P3"]["phenotype_C_std"]["n"] == 10,
        "pair_ess_aggregated": result["P1"]["pair_min_ess"]["mean"] == 1.0,
        "embedding_distribution_aggregated": result["P4"]["phenotype_C_distribution"]["p50"]["mean"] == 0.5,
        "gradient_and_loss_aggregated": result["P2"]["gradient_img_pair_norm"]["mean"] == 0.2
        and result["P2"]["mean_pair_ce"]["mean"] == 0.5,
        "sampled_fold_counts": len(result["P3"]["sampled_epochs_per_fold"]) == 10,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    failed = [key for key, value in checks.items() if not value]
    if failed:
        raise AssertionError(f"analysis repair self-test failed: {failed}")
    original_test = analyze.synthetic_self_test()
    return {"passed": True, "checks": checks, "base_analyzer_self_test": original_test}


def append_correction(summary: dict, report: str) -> dict:
    path = run.ROOT / "handoff.md"
    marker = "<!-- PCDPACM_20261021_ANALYSIS_CORRECTION -->"
    with (BASE / "analyze.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = path.read_bytes()
        old_end_marker = b"PCDPACM_20261021_END"
        if before.count(old_end_marker) != 1:
            raise AssertionError("Expected exactly one original PCDPACM completion marker")
        if marker.encode() in before:
            return {"appended": False, "idempotent_skip": True,
                    "marker_count": before.count(old_end_marker), "after_sha256": run.sha_file(path)}
        text = ("\n\n" + marker + "\n\n"
                + "### Experiment 22 分析器修复补充\n\n"
                + "初次正式分析发现机制汇总按元组键读取，而正式分析记录实际按组名再按fold嵌套；这使首版报告的机制字段为空，"
                + "但四项指标、paired fold差、OOF与checkpoint结果仍由有效artifact计算。现通过只读包装器重算summary和report，"
                + "没有重新训练、修改checkpoint、改动冻结训练源码或历史实验。以下是完整修正版报告：\n\n"
                + report + "\n\n"
                + f"修复包装器 SHA256：{summary['analysis_repair']['wrapper_sha256']}；"
                + f"冻结 analyze.py SHA256：{summary['analysis_repair']['original_analyze_sha256']}。\n\n"
                + f"本次追加前字节数：{len(before)}；追加前SHA256：{run.sha_bytes(before)}；"
                + "追加后前缀SHA256见summary.json；原始完成标记计数保持1。\n")
        after_data = before + text.encode("utf-8")
        run.atomic_bytes(after_data, path)
        after = path.read_bytes()
        if after[:len(before)] != before:
            raise AssertionError("Append-only analysis correction prefix check failed")
        count = after.count(old_end_marker)
        if count != 1 or after.count(marker.encode()) != 1:
            raise AssertionError("Completion/correction marker count mismatch")
        return {"appended": True, "before_bytes": len(before),
                "before_sha256": run.sha_bytes(before),
                "after_prefix_sha256": run.sha_bytes(after[:len(before)]),
                "after_bytes": len(after), "after_sha256": run.sha_file(path),
                "end_marker_count": count, "correction_marker_count": after.count(marker.encode())}


def append_perturbation_note(summary: dict) -> dict:
    path = run.ROOT / "handoff.md"
    marker = "<!-- PCDPACM_20261021_PERTURBATION_NOTE -->"
    original_marker = b"PCDPACM_20261021_END"
    perturbation = summary.get("preflight_perturbation", {})
    with (BASE / "analyze.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = path.read_bytes()
        if before.count(original_marker) != 1:
            raise AssertionError("Expected exactly one original PCDPACM completion marker")
        if marker.encode() in before:
            return {"appended": False, "idempotent_skip": True,
                    "end_marker_count": before.count(original_marker),
                    "after_sha256": run.sha_file(path)}
        lines = ["", "", marker, "", "### Experiment 22 错配强度补充", "",
                 "预检中P1/P2的w0权重矩阵mean_abs差为 "
                 + f"{perturbation.get('P1-P2', {}).get('mean_abs')}，"
                 + f"55.29%项非零差异；P3/P4为 {perturbation.get('P3-P4', {}).get('mean_abs')}，"
                 + f"55.62%项非零差异，max_abs={perturbation.get('P3-P4', {}).get('max_abs')}。",
                 "这证明错配预检改变了参考权重，但逐元素C变化未存盘；不能从P3/P4相近的C边际分布推断C矩阵逐项不变或已充分改变。",
                 "修正版逐折结果和完整机制表见 experiments/PCDPACM_20261021/report.md；本补充只更新报告说明，不改变任何训练artifact。",
                 f"分析修复包装器 SHA256：{summary['analysis_repair']['wrapper_sha256']}。", ""]
        data = "\n".join(lines).encode("utf-8")
        run.atomic_bytes(before + data, path)
        after = path.read_bytes()
        if after[:len(before)] != before:
            raise AssertionError("Append-only perturbation note prefix check failed")
        if after.count(original_marker) != 1 or after.count(marker.encode()) != 1:
            raise AssertionError("Perturbation/completion marker count mismatch")
        return {"appended": True, "before_bytes": len(before),
                "before_sha256": run.sha_bytes(before),
                "after_prefix_sha256": run.sha_bytes(after[:len(before)]),
                "after_bytes": len(after), "after_sha256": run.sha_file(path),
                "end_marker_count": after.count(original_marker),
                "perturbation_marker_count": after.count(marker.encode())}


def append_run_log_note(summary: dict) -> dict:
    path = run.ROOT / "handoff.md"
    marker = "<!-- PCDPACM_20261021_RUN_LOG_CLEANUP -->"
    original_marker = b"PCDPACM_20261021_END"
    info = summary["analysis_repair"]["run_log_repair"]
    with (BASE / "analyze.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        before = path.read_bytes()
        if before.count(original_marker) != 1:
            raise AssertionError("Expected exactly one original PCDPACM completion marker")
        if marker.encode() in before:
            return {"appended": False, "idempotent_skip": True,
                    "end_marker_count": before.count(original_marker),
                    "after_sha256": run.sha_file(path)}
        text = ("\n\n" + marker + "\n\n### Experiment 22 run.log完整性补充\n\n"
                + f"原始汇总日志 SHA256：{info['original_sha256']}；其中两条记录前共含 {info['nul_bytes_removed']} 个 NUL 填充字节。"
                + f"移除填充后保留全部非 NUL 字节，{info['valid_json_rows']} 条 JSONL 记录均可解析，"
                + f"修复日志 SHA256：{info['cleaned_sha256']}。逐折日志、queue.json和40份训练artifact未改动。\n\n")
        run.atomic_bytes(before + text.encode("utf-8"), path)
        after = path.read_bytes()
        if after[:len(before)] != before:
            raise AssertionError("Append-only run.log note prefix check failed")
        if after.count(original_marker) != 1 or after.count(marker.encode()) != 1:
            raise AssertionError("run.log/completion marker count mismatch")
        return {"appended": True, "before_bytes": len(before),
                "before_sha256": run.sha_bytes(before),
                "after_prefix_sha256": run.sha_bytes(after[:len(before)]),
                "after_bytes": len(after), "after_sha256": run.sha_file(path),
                "end_marker_count": after.count(original_marker),
                "log_note_marker_count": after.count(marker.encode())}


def run_corrected_analysis() -> dict:
    audit = run.read_json(BASE / "audit.json")
    previous = run.read_json(BASE / "summary.json")
    original_analyzer_sha = run.sha_file(BASE / "analyze.py")
    expected_analyzer_sha = audit["training_code_files_sha256"]["analyze.py"]
    if original_analyzer_sha != expected_analyzer_sha:
        raise AssertionError("Frozen analyze.py SHA changed; refusing the post-training repair")

    analyze.mechanism = mechanism_fixed
    analyze.report_text = report_text
    summary = analyze.run_analysis(final=False)
    current_artifacts = {
        f"{group}/911/{fold}": run.sha_file(run.result_path(group, 911, fold))
        for group in analyze.GROUPS for fold in range(10)
    }
    if current_artifacts != previous.get("artifact_sha256"):
        raise AssertionError("Training artifact SHA changed during read-only re-analysis")
    queue = run.read_json(BASE / "queue.json")
    summary["execution"]["queue_worker_seconds"] = queue.get("worker_seconds")
    summary["preflight_perturbation"] = audit.get("preflight", {}).get("pair_loss_differences", {})
    summary["handoff"] = previous.get("handoff", {})
    summary["analysis_repair"] = {
        "type": "post-training read-only analyzer wrapper",
        "reason": "mechanism aggregation used tuple keys instead of the persisted group/fold mapping",
        "original_analyze_sha256": original_analyzer_sha,
        "wrapper_sha256": run.sha_file(Path(__file__)),
        "training_code_sha256": audit["training_code_sha256"],
        "training_artifacts_modified": False,
        "all_40_artifact_sha256_revalidated": True,
        "metric_and_oof_calculation_changed": False,
    }
    summary["decision"] = analyze.scientific_decision(summary)
    summary["created_utc"] = run.utc()
    summary["final_verified_utc"] = summary["created_utc"]
    report = report_text(summary)
    run.atomic_bytes(report.encode("utf-8"), BASE / "report.md")
    summary["analysis_repair"]["handoff_addendum"] = append_correction(summary, report)
    summary["analysis_repair"]["perturbation_note"] = append_perturbation_note(summary)
    run.atomic_json(summary, BASE / "summary.json")
    return summary


def record_run_log_repair() -> dict:
    summary = run.read_json(BASE / "summary.json")
    log_path = BASE / "run.log"
    data = log_path.read_bytes()
    if b"\x00" in data:
        raise AssertionError("run.log still contains NUL padding")
    rows, events = 0, {}
    for line in data.splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        rows += 1
        event = value.get("event")
        events[event] = events.get(event, 0) + 1
    original_sha = "e68c0e9c992b37e26000ee3671fcaf6e65ed71a6eb5e8f4070c7b34680af6d70"
    expected_clean_sha = "833cbecced0a3a9f1ae8e63f54cfdcb7022b892cb78293a4aeca68afe14ca886"
    clean_sha = run.sha_file(log_path)
    if clean_sha != expected_clean_sha or rows != 363:
        raise AssertionError("Clean run.log does not match the validated NUL-stripped journal")
    summary["analysis_repair"]["wrapper_sha256"] = run.sha_file(Path(__file__))
    summary["analysis_repair"]["run_log_repair"] = {
        "original_bytes": 70018, "original_sha256": original_sha,
        "nul_bytes_removed": 371, "cleaned_bytes": len(data),
        "cleaned_sha256": clean_sha, "valid_json_rows": rows,
        "event_counts": events, "invalid_rows": 0,
        "fold_logs_and_training_artifacts_modified": False,
    }
    summary["analysis_repair"]["run_log_handoff"] = append_run_log_note(summary)
    run.atomic_bytes(report_text(summary).encode("utf-8"), BASE / "report.md")
    run.atomic_json(summary, BASE / "summary.json")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--record-run-log-repair", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=False, indent=2))
        return 0
    if args.record_run_log_repair:
        summary = record_run_log_repair()
        print(json.dumps(summary["analysis_repair"]["run_log_repair"], ensure_ascii=False, indent=2))
        return 0
    if not args.final:
        parser.error("pass --final to re-analyze the completed frozen artifacts")
    summary = run_corrected_analysis()
    print(json.dumps({"status": summary["status"], "coverage": summary["coverage"],
                      "decision": summary["decision"], "analysis_repair": summary["analysis_repair"]},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
