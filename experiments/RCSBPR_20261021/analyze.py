"""Independent artifact validation, paired analysis, and report generation."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

import run

METRICS = ("ACC", "SEN", "SPE", "AUC")
HISTORICAL = ("historical_D4", "historical_G2", "historical_G3", "historical_J1")
GROUPS = ("K1", "K2", "K3", "K4")
REPORT_GROUPS = HISTORICAL + GROUPS
PAIRS = (("K3", "K1"), ("K3", "K2"), ("K3", "K4"),
         ("K3", "historical_J1"), ("K3", "historical_D4"))


def percent(metrics):
    return {m: float(metrics[m] * 100.0) for m in METRICS}


def summarize(rows):
    return {m: {"mean": float(np.mean([x[m] for x in rows])),
                "sample_sd": float(np.std([x[m] for x in rows], ddof=1))} for m in METRICS}


def write_csv(path, fields, rows):
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def all_results(audit, require_complete):
    ctx = run.load_torch(run.OLD / "artifacts/context.pt")
    by_group = {g: [] for g in GROUPS}
    fold_rows, oof_rows, result_hashes = [], [], {}
    gpu_counts = {"0": 0, "1": 0}
    oof = {g: {"pred": np.full(871, -1, dtype=np.int64), "prob": np.full(871, np.nan),
               "fold": np.full(871, -1, dtype=np.int64), "by_id": {}} for g in GROUPS}
    for group in GROUPS:
        coverage = np.zeros(871, dtype=np.int64)
        for fold in range(10):
            value = run.valid_result(group, 911, fold, audit, required=require_complete)
            if value is None:
                continue
            split = run.load_torch(run.anchor(911, fold) / "splits.pt")
            indices, labels = value["oof"]["indices"], value["oof"]["labels"]
            logits = np.asarray(value["oof"]["logits"])
            probability, prediction = np.asarray(value["oof"]["probability"]), np.asarray(value["oof"]["prediction"])
            assert np.array_equal(indices, split["test"])
            assert np.array_equal(labels, ctx["labels"][indices])
            assert value["oof"]["ids"] == [ctx["ids"][int(i)] for i in indices]
            assert np.array_equal(prediction, logits.argmax(axis=1))
            prob_check = torch.softmax(torch.as_tensor(logits, dtype=torch.float64), dim=1)[:, 0].numpy()
            assert np.allclose(probability, prob_check, atol=1e-7, rtol=1e-6)
            assert np.isfinite(probability).all() and np.all((probability >= 0) & (probability <= 1))
            measured = run.binary_metrics(labels, probability, prediction)
            assert max(abs(measured[m] - value["metrics"][m]) for m in METRICS) < 1e-12
            assert value["physical_gpu"] == fold % 2
            assert value["formal_start_rng_sha256"] == audit["anchors"][str(fold)]["pairing"]["classifier_start_rng"]
            assert value["pairing"] == audit["anchors"][str(fold)]["pairing"]
            assert value["forward_calls"] == {"train": value["epochs_run"], "val": value["epochs_run"], "test": 1}
            assert value["test_forwards"] == 1 and value["formal_test_grad_enabled"]
            assert value["new_trainable_parameters"] == value["rfe_fit_count"] == value["vae_train_count"] == 0
            assert value["best_val_loss"] == min(e["val_loss"] for e in value["epochs"])
            assert value["best_epoch"] == min(range(len(value["epochs"])), key=lambda i: value["epochs"][i]["val_loss"])
            assert value["best_epoch"] < value["epochs_run"] <= run.CFG["epochs"]
            for epoch_no, epoch in enumerate(value["epochs"]):
                assert epoch["epoch"] == epoch_no
                assert np.isfinite(epoch["lambda_eff"]) and 0 <= epoch["lambda_eff"] <= run.CFG["lambda_cap"]
                assert np.isfinite(epoch["weighted_joint_grad_ratio"])
                assert -1e-8 <= epoch["weighted_joint_grad_ratio"] <= run.CFG["auxiliary_gradient_ratio"] + 1e-6
                assert epoch["objective"]["cross_mode"] == run.GROUP_SPECS[group]["cross_mode"]
                for key in ("train_ce", "train_graph", "train_native", "train_auxiliary", "val_ce", "val_graph", "val_loss"):
                    assert np.isfinite(epoch[key]), (group, fold, epoch_no, key)
                if epoch_no >= run.CFG["warmup_last_epoch"]:
                    assert epoch["aux_joint_grad_norm"] > 0
            coverage[indices] += 1
            oof[group]["pred"][indices], oof[group]["prob"][indices], oof[group]["fold"][indices] = prediction, probability, fold
            by_group[group].append(value)
            fold_rows.append({"group": group, "seed": 911, "fold": fold, **percent(measured)})
            for i, subject_id in enumerate(value["oof"]["ids"]):
                assert subject_id not in oof[group]["by_id"]
                oof[group]["by_id"][subject_id] = {
                    "y": int(labels[i]), "pred": int(prediction[i]), "probability": float(probability[i]),
                    "fold": fold,
                }
                oof_rows.append({"group": group, "seed": 911, "fold": fold, "subject_id": subject_id,
                                 "y_true": int(labels[i]), "y_pred": int(prediction[i]),
                                 "probability_asd": float(probability[i])})
            result_hashes[f"{group}/911/{fold}"] = run.sha_file(run.result_path(group, 911, fold))
            gpu_counts[str(value["physical_gpu"])] += 1
        if by_group[group]:
            assert np.all(coverage == 1)
        if require_complete:
            assert len(by_group[group]) == 10
    completed = sum(len(rows) for rows in by_group.values())
    if require_complete:
        assert completed == 40 and gpu_counts == {"0": 20, "1": 20}
        assert audit["execution"]["queue"]["completed_unique_tasks"] == 40
        assert audit["execution"]["queue"]["gpu_completed"] == gpu_counts and len(result_hashes) == 40
    for fold in range(10):
        vals = {g: next(v for v in by_group[g] if v["fold"] == fold) for g in GROUPS if by_group[g]}
        if len(vals) == 4:
            assert len({v["data_input_digest"] for v in vals.values()}) == 1
            assert len({v["formal_start_rng_sha256"] for v in vals.values()}) == 1
            k2, k3, k4 = (vals[g]["objective_static"] for g in ("K2", "K3", "K4"))
            assert k2["cross_real_coefficient_sha256"] == k3["cross_real_coefficient_sha256"]
            assert k3["within_r_coefficient_sha256"] == k4["within_r_coefficient_sha256"]
            assert k3["cross_real_coefficient_sha256"] == k4["cross_real_coefficient_sha256"]
            assert k3["omega_R"] == k4["omega_R"] and k3["omega_st"] == k4["omega_st"]
            assert k4["cross_shuffle_coefficient_sha256"] != k3["cross_real_coefficient_sha256"]
            assert np.isclose(k4["cross_shuffle_coefficient_sum"], 1, atol=1e-6)
    return ctx, by_group, oof, fold_rows, oof_rows, result_hashes, gpu_counts, completed


def history_rows(audit, ctx):
    d4, d4_mean, d4_sd, d4_hashes = run.historical_d4(ctx)
    g23, g23_summary, g23_hashes = run.historical_g2_g3(ctx)
    j1, j1_mean, j1_sd, j1_hashes = run.historical_j1(ctx)
    assert d4_hashes == audit["historical_d4_readonly"]["checkpoint_sha256"]
    assert g23_hashes == audit["historical_g2_g3_readonly"]["checkpoint_sha256"]
    assert j1_hashes == audit["historical_j1_readonly"]["checkpoint_sha256"]
    rows = {"historical_D4": d4, "historical_G2": g23["G2"], "historical_G3": g23["G3"], "historical_J1": j1}
    summary = {
        "historical_D4": {m: {"mean": d4_mean[m], "sample_sd": d4_sd[m]} for m in METRICS},
        "historical_J1": {m: {"mean": j1_mean[m], "sample_sd": j1_sd[m]} for m in METRICS},
    }
    for group in ("G2", "G3"):
        summary[f"historical_{group}"] = {
            m: {"mean": g23_summary[group]["mean_percent"][m],
                "sample_sd": g23_summary[group]["sample_sd_percent"][m]} for m in METRICS}
    return rows, summary


def paired(left, right, by_fold):
    out = {}
    for metric in METRICS:
        diff = np.asarray([(by_fold[left][f]["metrics"][metric] - by_fold[right][f]["metrics"][metric]) * 100
                           for f in range(10)])
        out[metric] = {
            "mean_pp": float(diff.mean()), "sample_sd_pp": float(diff.std(ddof=1)),
            "improved_folds": int(np.sum(diff > 1e-12)), "ties_folds": int(np.sum(np.abs(diff) <= 1e-12)),
            "fold_differences_pp": diff.tolist(),
            "leave_one_fold_out_mean_pp": {str(i): float(np.delete(diff, i).mean()) for i in range(10)},
        }
    return out


def site_mechanism(groups):
    weight_key = {"K1": "omega_R", "K2": "omega_J1", "K3": "omega_R", "K4": "omega_R"}
    by_group, all_deltas = {}, {}
    for group in GROUPS:
        shares, deltas = {}, {}
        for task in groups[group]:
            for site, x in task["objective_static"]["site_counts"].items():
                shares.setdefault(site, []).append(float(x[weight_key[group]]))
                deltas.setdefault(site, []).append(float(x["weight_delta"]))
        by_group[group] = {
            s: {"weight_mean": float(np.mean(v)), "weight_sample_sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
                "weight_delta_R_minus_J1_mean": float(np.mean(deltas[s]))} for s, v in sorted(shares.items())
        }
        if group == "K1":
            all_deltas = {s: float(np.mean(v)) for s, v in deltas.items()}
    ranked = sorted(all_deltas.items(), key=lambda x: x[1])
    pair_weights, exposure = {}, {}
    for task in groups["K3"]:
        for pair, info in task["objective_static"]["cross_site_pairs"].items():
            pair_weights.setdefault(pair, []).append(float(info["omega"]))
            s, t = pair.split("->")
            exposure[s] = exposure.get(s, 0.0) + float(info["omega"]) / 10
            exposure[t] = exposure.get(t, 0.0) + float(info["omega"]) / 10
    top_pairs = sorted(((p, float(np.mean(w))) for p, w in pair_weights.items()), key=lambda x: x[1], reverse=True)
    nyu = {
        "within_K1_mean": float(np.mean([x["objective_static"]["site_counts"].get("NYU", {}).get("omega_R", 0) for x in groups["K1"]])),
        "within_J1_mean": float(np.mean([x["objective_static"]["site_counts"].get("NYU", {}).get("omega_J1", 0) for x in groups["K2"]])),
        "cross_pair_endpoint_exposure": float(exposure.get("NYU", 0)),
    }
    active = [e for g in GROUPS for task in groups[g] for e in task["epochs"] if e["epoch"] >= run.CFG["warmup_last_epoch"]]
    counts = [x["objective_static"]["cross_valid_pair_count"] for x in groups["K3"]]
    sums = [x["objective_static"]["cross_coefficient_sum"] for x in groups["K3"]]
    shuffle_by_fold = {}
    for task in groups["K4"]:
        static = task["objective_static"]
        change = static["shuffle_change"]
        prep = task["ranking_preparation"]
        shuffle_by_fold[str(task["fold"])] = {
            "age_changed_fraction": prep["age_changed_fraction"],
            "sex_changed_fraction": prep["sex_changed_fraction"],
            "joint_phenotype_changed_fraction": prep["joint_phenotype_changed_fraction"],
            **change,
        }
    shuffle_valid = all(x["age_sex_kernel_changed_fraction"] > 0 and x["coefficient_l1_change"] > 1e-12
                        for x in shuffle_by_fold.values())
    return {
        "site_weights_by_group": by_group,
        "reliability_weight_delta_R_minus_J1": {"increased": ranked[-5:][::-1], "decreased": ranked[:5]},
        "cross_valid_site_pairs_by_fold": {str(x["fold"]): x["objective_static"]["cross_valid_pair_count"] for x in groups["K3"]},
        "cross_valid_site_pairs_min": int(min(counts)), "cross_valid_site_pairs_max": int(max(counts)),
        "cross_coefficient_sum_min": float(min(sums)), "cross_coefficient_sum_max": float(max(sums)),
        "top_cross_site_pair_omega": top_pairs[:10], "nyu_dominance": nyu,
        "k4_shuffle_by_fold": shuffle_by_fold, "k4_negative_control_effective": shuffle_valid,
        "k4_cross_kernel_changed_fraction_mean": float(np.mean([x["age_sex_kernel_changed_fraction"] for x in shuffle_by_fold.values()])),
        "k4_cross_coefficient_l1_change_mean": float(np.mean([x["coefficient_l1_change"] for x in shuffle_by_fold.values()])),
        "lambda_mean_after_warmup": float(np.mean([e["lambda_eff"] for e in active])),
        "weighted_joint_gradient_ratio_mean_after_warmup": float(np.mean([e["weighted_joint_grad_ratio"] for e in active])),
        "aux_joint_gradient_min_after_warmup": float(min(e["aux_joint_grad_norm"] for e in active)),
        "all_epoch_losses_and_gradients_finite": True, "all_test_forwards_exactly_one": True,
        "uses_heldout_training_coefficients": False,
    }


def oof_transitions(ctx, oof):
    transitions = {}
    subject_ids = list(ctx["ids"])
    for left, right in (("K3", "K1"), ("K3", "K2"), ("K3", "K4")):
        assert set(oof[left]["by_id"]) == set(subject_ids) == set(oof[right]["by_id"])
        y = np.asarray([oof[left]["by_id"][sid]["y"] for sid in subject_ids], dtype=np.int64)
        assert np.array_equal(y, [oof[right]["by_id"][sid]["y"] for sid in subject_ids])
        lp = np.asarray([oof[left]["by_id"][sid]["pred"] for sid in subject_ids])
        rp = np.asarray([oof[right]["by_id"][sid]["pred"] for sid in subject_ids])
        a, b = lp == y, rp == y
        classes = {}
        for label, name in ((0, "ASD"), (1, "TD")):
            mask = y == label
            classes[name] = {"new_correct": int(np.sum(mask & a & ~b)),
                             "new_wrong": int(np.sum(mask & ~a & b)),
                             "net_correct": int(np.sum(mask & a) - np.sum(mask & b))}
        transitions[f"{left}-{right}"] = {
            "new_correct": int(np.sum(a & ~b)), "new_wrong": int(np.sum(~a & b)),
            "net_correct": int(np.sum(a) - np.sum(b)),
            "both_correct": int(np.sum(a & b)), "both_wrong": int(np.sum(~a & ~b)), "by_class": classes,
        }
    return transitions


def render_report(data):
    lines = [
        "# Experiment 19: RC-SBPR", "",
        f"状态：{data['status']}；科研判断：{data['scientific_judgment']['label']}。", "",
        "## 四项主指标", "",
        "均值 ± 十折 sample SD，百分比；SEN-ASD、SPE-TD，AUC 按 ASD 概率计算。", "",
        "| 组别 | 方法 | ACC | SEN-ASD | SPE-TD | Probability AUC |", "|---|---|---:|---:|---:|---:|",
    ]
    methods = {
        "historical_D4": "历史D4，只读年龄×性别原型 CE",
        "historical_G2": "实验16 G2，只读全局+同站点条件排序",
        "historical_G3": "实验16 G3，只读原型调节条件排序",
        "historical_J1": "实验18 J1，只读站点均衡条件排序",
        "K1": "J1 同站点汇总改用类别/表型可靠性权重",
        "K2": "J1 条件项 75% 同站点 + 25% 真实跨站点",
        "K3": "K1 + 75% 同站点 + 25% 真实跨站点",
        "K4": "K3，仅跨站点年龄/性别按站点内联合打乱",
    }
    for group in REPORT_GROUPS:
        values = data["metrics"][group]
        stats = [f"{values[m]['mean']:.4f} ± {values[m]['sample_sd']:.4f}" for m in METRICS]
        lines.append("| " + group + " | " + methods[group] + " | " + " | ".join(stats) + " |")
    lines += ["", "## 配对折差", "", "差值为 K3 减比较组，百分点；改善折数为正差折数。", "",
              "| 比较 | ΔACC | ΔSEN | ΔSPE | ΔAUC | AUC 改善折数 |", "|---|---:|---:|---:|---:|---:|"]
    for pair, value in data["paired_comparisons"].items():
        vals = [f"{value['metrics'][m]['mean_pp']:+.4f} ± {value['metrics'][m]['sample_sd_pp']:.4f}" for m in METRICS]
        lines.append(f"| {pair} | " + " | ".join(vals) + f" | {value['metrics']['AUC']['improved_folds']}/10 |")
    mech = data["mechanism"]
    lines += ["", "## 站点与跨站点机制", "",
              f"K1 相比 J1 的可靠性权重平均上调站点：{mech['reliability_weight_delta_R_minus_J1']['increased']}；下调站点：{mech['reliability_weight_delta_R_minus_J1']['decreased']}。NYU 的 K1/J1 平均站点权重为 {mech['nyu_dominance']['within_K1_mean']:.4f}/{mech['nyu_dominance']['within_J1_mean']:.4f}。",
              f"每折有效有向站点对 {mech['cross_valid_site_pairs_min']}–{mech['cross_valid_site_pairs_max']}；真实 C_cross 总和 {mech['cross_coefficient_sum_min']:.8f}–{mech['cross_coefficient_sum_max']:.8f}。最高 omega 站点对：{mech['top_cross_site_pair_omega'][:5]}。NYU 站点对端点 omega 曝光为 {mech['nyu_dominance']['cross_pair_endpoint_exposure']:.4f}。",
              f"K4 负对照逐折均有非零核变化及系数变化：{mech['k4_negative_control_effective']}；跨站点核平均变化比例 {mech['k4_cross_kernel_changed_fraction_mean']:.4f}，平均 C_cross L1 变化 {mech['k4_cross_coefficient_l1_change_mean']:.4g}。K2/K3 的真实跨站点系数逐折 SHA 相同；K3/K4 的同站点系数、可靠性权重、固定 omega_st 及模型输入逐折相同。",
              "", "## 真实与打乱表型", "",
              "K4 使用历史 PCDCA condition-shuffle/{site} 私有种子，在 outer-train 的每个站点内联合置换年龄与性别。每折的年龄、性别、联合表型和 C_cross 变化率见 summary.json；K3/K4 唯一目标差异是跨站点条件核。",
              "", "## Fold 敏感性", "", "全部十折结果及预设 Fold 2/6/9 best epoch：", "",
              "| Fold | K1 best | K2 best | K3 best | K4 best | K3−K1 AUC | K3−K2 AUC | K3−K4 AUC |", "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    diffs = data["paired_comparisons"]
    for fold in range(10):
        epoch = {g: next(x for x in data["group_fold_records"][g] if x["fold"] == fold)["best_epoch"] for g in GROUPS}
        auc = [diffs[f"K3-{g}"]["metrics"]["AUC"]["fold_differences_pp"][fold] for g in ("K1", "K2", "K4")]
        label = f"**{fold}**" if fold in (2, 6, 9) else str(fold)
        lines.append(f"| {label} | {epoch['K1']} | {epoch['K2']} | {epoch['K3']} | {epoch['K4']} | " +
                     " | ".join(f"{x:+.4f}" for x in auc) + " |")
    lines += ["", "移除单折后的 AUC 变化："]
    for pair in ("K3-K1", "K3-K2", "K3-K4", "K3-historical_J1", "K3-historical_D4"):
        auc = data["paired_comparisons"][pair]["metrics"]["AUC"]
        flipped = [int(k) for k, v in auc["leave_one_fold_out_mean_pp"].items() if (auc["mean_pp"] > 0) != (v > 0)]
        lines.append(f"- {pair}: 全折 {auc['mean_pp']:+.4f} pp；移除后正负翻转折 {flipped}。")
    lines += ["", "## 患者级 OOF", "", "| 比较 | 新变正确 | 新变错误 | 净增正确 | ASD 净增正确 | TD 净增正确 |", "|---|---:|---:|---:|---:|---:|"]
    for pair, value in data["patient_oof_transitions"].items():
        lines.append(f"| {pair} | {value['new_correct']} | {value['new_wrong']} | {value['net_correct']} | " +
                     f"{value['by_class']['ASD']['net_correct']} | {value['by_class']['TD']['net_correct']} |")
    lines += ["", "## 科研判断与完整性", "", data["scientific_judgment"]["reason"],
              f"正式任务 {data['formal_tasks']}/40；GPU0 {data['gpu_completed']['0']}/20，GPU1 {data['gpu_completed']['1']}/20；失败 {data['failed_tasks']}，重试 {data['retry_count']}。墙钟 {data['wall_seconds']:.1f}s，成功 worker 累计 {data['worker_seconds']:.1f}s。",
              f"34 项 CPU 数学测试全通过；独立循环 oracle 最大差 {data['math_tests']['oracle_max_abs_error']:.3g}。真实预检模型 forward={data['preflight']['model_forward_count']} 次。正式任务测试 forward 每项恰为一次，新增可训练参数为 0。",
              f"训练代码 SHA256 {data['code_sha256']}；manifest {data['manifest_sha256']}；prompt {data['prompt_sha256']}。原始 handoff 前缀 SHA256 {data['handoff_before_sha256']}。",
              "保留限制：跨折监督 RFE、871 节点转导人口图、随机 VAE 与 eval 随机行为、PPU 长轨迹非完全确定性、单 seed 十折。跨运行历史比较包含不同随机训练轨迹。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    audit = run.guard()
    if not audit.get("formal_authorized") or not audit.get("manifest"):
        raise RuntimeError("Analysis requires the passed preflight and frozen manifest")
    ctx, groups, oof, fold_rows, oof_rows, result_hashes, gpu_counts, completed = all_results(audit, args.final)
    if args.final and completed != 40:
        raise AssertionError(f"Only {completed}/40 valid formal tasks")
    histories, hist_metrics = history_rows(audit, ctx)
    metrics = {g: summarize([percent(x["metrics"]) for x in groups[g]]) for g in GROUPS if groups[g]}
    metrics.update(hist_metrics)
    by_fold = {g: {x["fold"]: x for x in values} for g, values in groups.items()}
    by_fold.update({g: {x["fold"]: x for x in values} for g, values in histories.items()})
    comparisons = {f"{left}-{right}": {"metrics": paired(left, right, by_fold), "paired_by_fold": True}
                   for left, right in PAIRS}
    preflight = audit["preflight"]["911/0"]
    mechanism = site_mechanism(groups) if completed == 40 else {}
    patient = oof_transitions(ctx, oof) if completed == 40 else {}
    auc_comp = [comparisons[f"K3-{g}"]["metrics"]["AUC"]["mean_pp"] for g in ("K1", "K2", "K4", "historical_J1")]
    spe_comp = [comparisons[f"K3-{g}"]["metrics"]["SPE"]["mean_pp"] for g in ("K1", "K2", "K4", "historical_J1")]
    candidate = all(x > 0 for x in auc_comp) and all(x >= -0.5 for x in spe_comp) and mechanism.get("k4_negative_control_effective", False)
    judgment = {
        "label": "GO_CANDIDATE" if candidate else "MIXED",
        "k3_auc_positive_vs_k1_k2_k4_j1": all(x > 0 for x in auc_comp),
        "k3_spe_drop_no_worse_than_0_5pp_vs_k1_k2_k4_j1": all(x >= -0.5 for x in spe_comp),
        "confirmation_required": True,
        "reason": ("K3 满足本轮预设比较方向，可作为预注册多 seed 与独立站点确认候选；单 seed 十折不能证明统计或外部泛化价值。"
                   if candidate else
                   "K3 未能同时相对 K1、K2、K4 与历史 J1 获得正 AUC 配对均差并保持 SPE；完整方案不满足预设候选筛选条件。依据实际结果选择后续确认方向，不进行 test-driven 调参。"),
    }
    queue = audit["execution"].get("queue", {})
    events = audit["execution"].get("events", [])
    data = {
        "status": "COMPLETE" if completed == 40 and audit.get("formal_authorized") else "PARTIAL",
        "experiment": run.CFG["experiment"], "formal_tasks": completed, "gpu_completed": gpu_counts,
        "failed_tasks": len(audit["execution"].get("failures", [])),
        "retry_count": len(audit["execution"].get("failures", [])),
        "wall_seconds": float(queue.get("wall_seconds", 0.0)),
        "worker_seconds": float(sum(x.get("wall_seconds", 0.0) for x in events)),
        "metrics": metrics, "paired_comparisons": comparisons,
        "group_fold_records": {g: [{"fold": x["fold"], "best_epoch": x["best_epoch"],
                                    "best_val_loss": x["best_val_loss"], "epochs_run": x["epochs_run"]}
                                   for x in groups[g]] for g in GROUPS},
        "mechanism": mechanism, "patient_oof_transitions": patient,
        "oof_coverage": {g: int(np.isfinite(oof[g]["prob"]).sum()) for g in GROUPS},
        "result_sha256": result_hashes, "math_tests": audit["math_tests"], "preflight": preflight,
        "scientific_judgment": judgment, "manifest": audit["manifest"],
        "manifest_sha256": run.sha_file(run.BASE / "manifest.json"),
        "code_sha256": audit["code_sha256"], "prompt_sha256": audit["prompt_sha256"],
        "handoff_before_sha256": audit["handoff_before_sha256"], "handoff_before_bytes": audit["handoff_before_bytes"],
        "source_sha256": audit["source_sha256"], "protected_file_count": len(audit["protected_sha256"]),
        "protected_sha256": audit["protected_sha256"], "execution": queue,
    }
    if args.final:
        assert data["status"] == "COMPLETE"
        assert all(len(by_fold[g]) == 10 for g in REPORT_GROUPS)
        write_csv(run.BASE / "fold_metrics.csv", ["group", "seed", "fold", *METRICS], fold_rows)
        write_csv(run.BASE / "oof.csv", ["group", "seed", "fold", "subject_id", "y_true", "y_pred", "probability_asd"], oof_rows)
        (run.BASE / "summary.json").write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
        (run.BASE / "report.md").write_text(render_report(data), encoding="utf-8")
        print(json.dumps({"status": data["status"], "formal_tasks": completed,
                          "gpu_completed": gpu_counts, "scientific_judgment": judgment["label"],
                          "report": str(run.BASE / "report.md")}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"formal_tasks": completed, "metrics_available": list(metrics),
                          "gpu_completed": gpu_counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
