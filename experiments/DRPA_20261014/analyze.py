"""OOF verification, descriptive paired repeated-CV statistics, and handoff."""

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess

import fcntl
import numpy as np

from run import (BASE, ROOT, anchor_dir, atomic_json, binary_metrics, completed,
                 config, context, fold_dir, load_torch, sha_file, source_guard,
                 training_code_hash, utc)

KEYS = ("ACC", "BA", "AUC", "SEN", "SPE", "F1")
PRECEDING = {"E1": "E0", "E2": "E1", "E3": "E1", "E4": "E3",
             "E5": "E4", "E6": "E5", "E7": "E6"}
HYPOTHESES = {
    "E0": "保留官方行为，复核历史结果并建立配对初始化锚点。",
    "E1": "控制投影、Predictor、辅助监督及额外参数本身的收益。",
    "E2": "常规受试者级双向 InfoNCE 是否提供额外收益。",
    "E3": "不使用负样本的双向 stop-gradient 一致性是否更适合弱表型。",
    "E4": "影像锚定的非对称一致性是否优于双向一致性。",
    "E5": "按两模态疾病支持度选择受试者，是否优于等权对齐。",
    "E6": "排除本站点的诊断原型可靠性是否进一步改善疾病门控。",
    "E7": "保持本组门控权重分布但打乱受试者对应后，收益是否下降。"
}
CHANGES = {
    "E0": "不增加任何研究模块；新增代码只记录状态和输出。",
    "E1": "增加相同的 16-64-16 投影/Predictor及二分类辅助头，λaux=0.05。",
    "E2": "在 E1 上增加训练受试者双向 InfoNCE，温度0.2，除以 log(Ntrain)。",
    "E3": "在 E1 上增加双向 stop-gradient 余弦损失，不使用负样本。",
    "E4": "相对 E3 仅将双向对齐改为表型 Predictor→影像共享目标。",
    "E5": "相对 E4 仅增加 detached 诊断可靠性权重及阈值0.15。",
    "E6": "相对 E5 仅乘以 detached 跨站点类别边际可靠性。",
    "E7": "相对 E6 仅用私有随机源打乱训练受试者的权重对应关系。"
}


def read_rows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def metrics_from_rows(rows):
    return binary_metrics(np.array([int(r["label_original"]) for r in rows]),
                          np.array([int(r["pred_original"]) for r in rows]),
                          np.array([float(r["prob_asd"]) for r in rows]))


def check_fold_rows(result, rows, ctx):
    indices = [int(r["index"]) for r in rows]
    if len(indices) != len(set(indices)) or len(indices) != result["split_sizes"]["test"]:
        raise ValueError("Incomplete or duplicate fold OOF rows")
    split = load_torch(anchor_dir(result["seed"], result["fold"]) / "splits.pt")
    if set(indices) != set(map(int, split["test"])):
        raise ValueError("OOF rows are not the actual held-out fold subjects")
    for row in rows:
        idx = int(row["index"])
        if row["subject_id"] != ctx["ids"][idx] or row["site"] != ctx["site_names"][idx]:
            raise ValueError("OOF subject/site index mismatch")
        if int(row["label_original"]) != int(ctx["labels"][idx]):
            raise ValueError("OOF diagnostic label changed")
        if int(row["label_asd"]) != int(ctx["labels"][idx] == 0):
            raise ValueError("OOF ASD mapping mismatch")
        if int(row["pred_asd"]) != int(row["pred_original"] == "0"):
            raise ValueError("OOF prediction mapping mismatch")
        if not 0 <= float(row["prob_asd"]) <= 1 or abs(float(row["prob_asd"]) + float(row["prob_hc"]) - 1) > 1e-6:
            raise ValueError("Invalid continuous probabilities")
        if int(row["seed"]) != result["seed"] or int(row["fold"]) != result["fold"] or row["method"] != result["method"]:
            raise ValueError("OOF task identity mismatch")
    recomputed = metrics_from_rows(rows)
    for key in KEYS:
        if abs(recomputed[key] - result["metrics"][key]) > 1e-10:
            raise ValueError(f"Stored vs actual OOF metric differs: {key}")


def assert_pairing(method_result, reference_result):
    keys = ("split_fingerprints", "features_fingerprint", "rfe_support_sha256",
            "initial_core_fingerprint", "train_start_core_fingerprint", "vae_sha256", "classifier_start_rng")
    for key in keys:
        if method_result["pairing"][key] != reference_result["pairing"][key]:
            raise ValueError(f"Unpaired experiment: {key}")
    if reference_result["method"] != "E0" and method_result["method"] != "E0":
        key = "auxiliary_initial_fingerprint"
        if method_result["pairing"][key] != reference_result["pairing"][key]:
            raise ValueError("New-head initializations differ across E1-E7")


def load_results(method, verify=True):
    cfg, ctx = config(), context()
    results, by_seed = [], {}
    for seed in cfg["seeds"]:
        rows_seed = []
        for fold in range(cfg["folds"]):
            if not completed(method, seed, fold, verify=verify):
                raise RuntimeError(f"Incomplete {method}/{seed}/{fold}")
            out = fold_dir(method, seed, fold)
            result = json.loads((out / "metrics.json").read_text())
            rows = read_rows(out / "oof.csv")
            check_fold_rows(result, rows, ctx)
            results.append(result); rows_seed.extend(rows)
        rows_seed.sort(key=lambda row: int(row["index"]))
        if len(rows_seed) != 871 or [int(r["index"]) for r in rows_seed] != list(range(871)):
            raise ValueError("Each seed must cover 871 subjects exactly once")
        if len({r["subject_id"] for r in rows_seed}) != 871:
            raise ValueError("Duplicate OOF subject IDs")
        by_seed[seed] = rows_seed
    return results, by_seed


def mean_diagnostics(results, key):
    records = [r["diagnostics"][key] for r in results if key in r["diagnostics"]]
    return {k: float(np.mean([r[k] for r in records])) for k in records[0]} if records else None


def paired_statistics(differences, seeds):
    return {
        "mean_delta": {k: float(np.mean([d[k] for d in differences])) for k in KEYS},
        "improved_folds": {k: sum(d[k] > 1e-12 for d in differences) for k in ("ACC", "BA", "AUC")},
        "tied_folds": {k: sum(abs(d[k]) <= 1e-12 for d in differences) for k in ("ACC", "BA", "AUC")},
        "improved_seeds": {k: int(sum(np.mean([d[k] for d in differences if d["seed"] == seed]) > 1e-12
                                     for seed in seeds)) for k in ("ACC", "BA", "AUC")},
        "fold0": [d for d in differences if d["fold"] == 0],
        "other_folds_mean_delta": {k: float(np.mean([d[k] for d in differences if d["fold"] != 0])) for k in KEYS},
        "paired_folds": differences}


def summary(method):
    cfg = config()
    results, oof = load_results(method)
    seed_stats = {}
    for seed in cfg["seeds"]:
        folds = [r for r in results if r["seed"] == seed]
        seed_stats[str(seed)] = {
            "fold_mean": {k: float(np.mean([r["metrics"][k] for r in folds])) for k in KEYS},
            "fold_std_ddof0": {k: float(np.std([r["metrics"][k] for r in folds])) for k in KEYS},
            "oof_pooled": metrics_from_rows(oof[seed]),
            "official_fold_mean": {k: float(np.mean([r["official_metrics"][k] for r in folds])) for k in KEYS},
            "folds": [{"fold": r["fold"], **r["metrics"], "best_epoch": r["best_epoch"]} for r in folds]
        }
    means = {k: float(np.mean([s["fold_mean"][k] for s in seed_stats.values()])) for k in KEYS}
    std = {k: float(np.std([s["fold_mean"][k] for s in seed_stats.values()], ddof=1)) for k in KEYS}
    gate_records = [r["last_sampled_gate"] for r in results if r["last_sampled_gate"] is not None]
    gate = None
    if gate_records:
        gate = {"mean_nonzero_coverage": float(np.mean([g["weights"]["nonzero_fraction"] for g in gate_records])),
                "mean_q_pred": float(np.mean([g["q_pred"]["mean"] for g in gate_records])),
                "mean_q_site": float(np.mean([g["q_site"]["mean"] for g in gate_records])),
                "mean_site_fallback_count": float(np.mean([g["site_fallback_count"] for g in gate_records])),
                "classes": {}, "sites": {}}
        for cls in ("0", "1"):
            gate["classes"][cls] = {k: float(np.mean([g["classes"][cls][k] for g in gate_records]))
                                    for k in ("mean", "std", "p10", "p50", "p90", "nonzero_fraction")}
        sites = sorted({key for g in gate_records for key in g["sites"]})
        for site in sites:
            records = [g["sites"][site] for g in gate_records if site in g["sites"]]
            gate["sites"][site] = {
                "mean_q_pred": float(np.mean([r["q_pred"]["mean"] for r in records])),
                "mean_q_site": float(np.mean([r["q_site"]["mean"] for r in records])),
                "mean_nonzero_coverage": float(np.mean([r["weights"]["nonzero_fraction"] for r in records]))}
        if method == "E7":
            gate["mean_shuffle_moved_fraction"] = float(np.mean([g["shuffle_moved_fraction"] for g in gate_records]))
            gate["mean_shuffle_changed_weight_fraction"] = float(np.mean([g["shuffle_changed_weight_fraction"] for g in gate_records]))
            gate["mean_cross_site_singleton_fallback"] = float(np.mean([g["shuffle_cross_site_singletons"] for g in gate_records]))
    auxiliary = None
    if method != "E0":
        auxiliary = {mod: {k: float(np.mean([r["diagnostics"]["auxiliary"]["test"][mod][k] for r in results]))
                           for k in KEYS} for mod in ("img", "ph")}
    scale = []
    all_zero_steps = 0
    for r in results:
        for line in (fold_dir(method, r["seed"], r["fold"]) / "epochs.jsonl").read_text().splitlines():
            ep = json.loads(line)
            all_zero_steps += int(ep["gate_all_zero"])
            if "gradient_norms" in ep:
                scale.append(ep)
    loss_scale = {name: float(np.mean([ep[name] for ep in scale])) for name in
                  ("train_ce", "train_graph", "train_official", "train_aux", "train_align", "train_total")}
    grad_scale = {name: float(np.mean([ep["gradient_norms"][name] for ep in scale]))
                  for name in ("official", "weighted_aux", "weighted_align")}
    pooled_sites = {}
    for seed, rows in oof.items():
        for site in sorted({r["site"] for r in rows}):
            subset = [r for r in rows if r["site"] == site]
            counts = [sum(int(r["label_original"]) == cls for r in subset) for cls in (0, 1)]
            entry = {"n": len(subset), "n_asd": counts[0], "n_hc": counts[1]}
            eligible = len(subset) >= cfg["minimum_site_metric_samples"] and min(counts) >= cfg["minimum_site_metric_per_class"]
            entry["eligible"] = eligible
            if eligible:
                entry.update(metrics_from_rows(subset))
            pooled_sites.setdefault(site, {})[str(seed)] = entry
    retries = sum(json.loads((fold_dir(method, r["seed"], r["fold"]) / "status.json").read_text())["attempt"] - 1 for r in results)
    report = {"method": method, "name": cfg["methods"][method]["name"], "time_utc": utc(),
        "n_tasks": 30, "n_success": 30, "n_incomplete": 0, "failed_attempts_before_success": retries,
        "oof_count": sum(len(rows) for rows in oof.values()), "seeds": seed_stats,
        "mean": means, "std_across_three_seed_means_ddof1": std,
        "std_across_30_seed_folds_ddof0": {k: float(np.std([r["metrics"][k] for r in results])) for k in KEYS},
        "raw_alignment": mean_diagnostics(results, "raw"), "projected_alignment": mean_diagnostics(results, "shared"),
        "gate": gate, "auxiliary_test": auxiliary, "pooled_oof_sites": pooled_sites,
        "loss_scale": loss_scale, "gradient_scale_at_embeddings": grad_scale,
        "all_zero_gate_training_steps": all_zero_steps,
        "classifier_seconds_total": float(sum(r["classifier_seconds"] for r in results)),
        "classifier_seconds_mean": float(np.mean([r["classifier_seconds"] for r in results])),
        "vae_seconds_total": float(sum(r["vae_pretrain_seconds"] for r in results)),
        "peak_allocated_memory_bytes": max(r["peak_allocated_memory_bytes"] for r in results),
        "parameter_counts": results[0]["parameters"], "training_code_sha256": training_code_hash(),
        "statistics_unit": "3 repeated-CV seed means; 30 paired folds are correlated and descriptive, not independent cohorts",
        "metric_positive_class": "ASD; training encoding unchanged", "comparisons": {}}
    by_task = {(r["seed"], r["fold"]): r for r in results}
    for reference in ("E0", "E1", "E4", "E5", "E6", PRECEDING.get(method)):
        if reference is None or reference == method or reference in report["comparisons"]:
            continue
        if not all(completed(reference, seed, fold, verify=False) for seed in cfg["seeds"] for fold in range(10)):
            continue
        differences = []
        for key, result in by_task.items():
            ref = json.loads((fold_dir(reference, *key) / "metrics.json").read_text())
            assert_pairing(result, ref)
            differences.append({"seed": key[0], "fold": key[1],
                                **{k: result["metrics"][k] - ref["metrics"][k] for k in KEYS}})
        report["comparisons"][reference] = paired_statistics(differences, cfg["seeds"])
    out = BASE / "outputs" / method
    atomic_json(report, out / "summary.json")
    with (out / "oof_all.csv.tmp").open("w", newline="") as stream:
        rows = [row for seed in cfg["seeds"] for row in oof[seed]]
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    os.replace(out / "oof_all.csv.tmp", out / "oof_all.csv")
    return report


def format_metrics(values):
    return ", ".join(f"{key} {values[key] * 100:.2f}%" for key in KEYS)


def delta_text(comparison):
    return "、".join(f"Δ{k} {comparison['mean_delta'][k] * 100:+.2f} 个百分点" for k in ("ACC", "BA", "AUC"))


def handoff_group(report):
    method = report["method"]
    lines = [f"## {method} {report['name']}", f"- 假设：{HYPOTHESES[method]}",
             f"- 唯一研究变化：{CHANGES[method]}",
             "- 数据/协议：ABIDE 871人、AAL116；3 seed×10折、val_ratio=0.1；原始跨折RFE覆盖、传导式图和随机验证均保留。训练标签ASD=0/HC=1；下列SEN/SPE/F1以ASD为阳性，官方口径另存。",
             f"- 运行：成功30/30，未完成0，失败后重试次数{report['failed_attempts_before_success']}；OOF {report['oof_count']}条。"]
    for seed, record in report["seeds"].items():
        lines.append(f"- seed {seed} 10折均值：{format_metrics(record['fold_mean'])}。")
    lines.append(f"- 三seed均值：{format_metrics(report['mean'])}；seed均值SD："
                 + ", ".join(f"{k} {report['std_across_three_seed_means_ddof1'][k] * 100:.2f}" for k in KEYS) + " 个百分点。")
    lines.append("- 30个seed-fold指标的描述性SD（非独立样本）："
                 + ", ".join(f"{k} {report['std_across_30_seed_folds_ddof0'][k] * 100:.2f}" for k in KEYS) + " 个百分点；不与seed均值SD混用。")
    if "E0" in report["comparisons"]:
        lines.append(f"- 相对E0：{delta_text(report['comparisons']['E0'])}；ACC/BA/AUC改善折数：{report['comparisons']['E0']['improved_folds']} /30。")
    elif method == "E0":
        check = json.loads((BASE / "artifacts/history_check.json").read_text())
        lines.append(f"- 历史seed911核对：输出精度完全一致={check['exact_at_original_output_precision']}，ΔACC={check['mean_delta_ACC']*100:+.4f}、ΔAUC={check['mean_delta_AUC']*100:+.4f} 个百分点。")
        diagnostic = BASE / "artifacts/baseline_diagnostics/verification.json"
        if diagnostic.exists() and not check["exact_at_original_output_precision"]:
            evidence = json.loads(diagnostic.read_text())
            lines.append(f"- 非精确重放诊断：同一PPU、同初始化/VAE/测试RNG的官方原函数重放ACC跨度{evidence['same_physical_card_acc_span']*100:.2f}个百分点；具体底层算子尚未定位。证据：{diagnostic}。不能称为历史逐折完全一致。")
            acceptance = BASE / "artifacts/history_acceptance.json"
            if acceptance.exists():
                record = json.loads(acceptance.read_text())
                lines.append(f"- 历史对照判定：{record['decision']}；原始严格容差检查passed={check['passed']}，不覆盖原始核对结果。")
    previous = PRECEDING.get(method)
    if previous in report["comparisons"]:
        comp = report["comparisons"][previous]
        lines.append(f"- 相对直接对照{previous}：{delta_text(comp)}；三seed改善数量：{comp['improved_seeds']} /3。")
        direction = "均值指标一致上升，但不能据此宣称临床显著性" if all(comp["mean_delta"][k] > 0 for k in ("ACC", "BA", "AUC")) else "指标未一致改善，当前结果不能直接支持该子模块的优势"
        lines.append(f"- 关键发现：{direction}。")
    elif method == "E0":
        lines.append("- 关键发现：本组建立可配对原始实现基线；不同seed的初始化由顺序训练历史决定。")
    else:
        lines.append(f"- 关键发现：直接对照{previous}尚未完成，暂不判断本组相对该对照的效应；全部组完成后统一重算。")
    alignment = report["projected_alignment"] or report["raw_alignment"]
    lines.append(f"- 对齐诊断：同人cos={alignment['same_subject_cosine']:.4f}，异人cos={alignment['different_subject_cosine']:.4f}，gap={alignment['cosine_gap']:.4f}，I→P R@1={alignment['R_at_1_img_to_ph']*100:.2f}%，共享均值方差={alignment['mean_shared_variance']:.6f}。E0使用原始嵌入，不能与投影空间数值混同。")
    if report["gate"]:
        gate = report["gate"]
        lines.append(f"- 门控：平均非零覆盖率{gate['mean_nonzero_coverage']*100:.2f}%，qpred={gate['mean_q_pred']:.4f}，qsite={gate['mean_q_site']:.4f}，站点回退均值{gate['mean_site_fallback_count']:.2f}人/折；逐类/站点分布见summary.json。")
        if method == "E7":
            lines.append(f"- 打乱有效性：索引移动{gate['mean_shuffle_moved_fraction']*100:.2f}%，实际权重改变{gate['mean_shuffle_changed_weight_fraction']*100:.2f}%。")
    if report["auxiliary_test"]:
        lines.append(f"- 辅助分类：影像BA/AUC={report['auxiliary_test']['img']['BA']*100:.2f}/{report['auxiliary_test']['img']['AUC']*100:.2f}%，表型BA/AUC={report['auxiliary_test']['ph']['BA']*100:.2f}/{report['auxiliary_test']['ph']['AUC']*100:.2f}%。")
    lines.append(f"- 损失/梯度：训练损失均值{report['loss_scale']}；原始嵌入处加权梯度范数{report['gradient_scale_at_embeddings']}。全零门控步数{report['all_zero_gate_training_steps']}。")
    lines.append(f"- 计算：分类训练合计{report['classifier_seconds_total']/60:.2f}分钟，VAE合计{report['vae_seconds_total']/60:.2f}分钟，峰值显存{report['peak_allocated_memory_bytes']/1024**3:.2f}GiB；参数{report['parameter_counts']}。")
    lines.append("- 数值检查口径：原模型/关闭新增损失的模型均在测试专用确定性后端比较；正式训练保留原始后端设置。生产路径自身存在TopK数值不确定性，不能由测试通过推断生产训练逐位可重放。")
    lines.append("- 异常/局限：保留了官方RFE跨折泄漏；同站点传导式评估不能证明未见站点泛化；30折相关，2613条OOF只对应871个重复受试者；未显式分解私有信息；不进行测试调参。")
    lines.append("- 下一步：继续E1–E7混合队列中的未完成任务；全部完成后进行最终OOF、跨站点及配对统计；保持预注册设置。")
    lines.append(f"- 版本/产物：根源码{config()['source_commit'][:12]}，训练代码SHA256 {report['training_code_sha256']}；{BASE / 'outputs' / method}。")
    return "\n".join(lines)


def update_handoff(final_text=None):
    lock = BASE / "artifacts/handoff.lock"
    with lock.open("a") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        target = ROOT / "handoff.md"
        old = target.read_text() if target.exists() else ""
        start, end = "<!-- DRPA_20261014_START -->", "<!-- DRPA_20261014_END -->"
        blocks = [start, "# DRPA_20261014 实验交接", f"更新时间（UTC）：{utc()}"]
        policy_path = BASE / "artifacts/scheduling_policy.json"
        if policy_path.exists():
            policy = json.loads(policy_path.read_text())
            blocks.append("## 双卡调度\n"
                          f"- E1–E7共210个seed-fold任务，每组每卡15个，合计每卡105个；已完成任务保留。每卡最多{policy['workers_per_gpu']}个独立进程，各组混合轮询。\n"
                          "- 同一seed/fold的E1–E7绑定同一物理PPU；复用已验证初始化、RNG、划分、RFE和VAE锚点，不改训练代码或超参数。\n"
                          f"- 每新任务预留{policy['worker_memory_budget_MiB']}MiB，卡上保留{policy['reserve_MiB_per_gpu']}MiB；同时计入其他项目占用和启动中进程的预期增长，不停止其他项目进程。\n"
                          "- 显存查询仅用于派发前检查；监控超时只暂停新派发，不终止已运行任务，不使用过期显存数据启动。首次监控超时造成的中断保留记录，未完成折从原锚点重启。\n"
                          f"- 分配表：{BASE / 'artifacts/variant_plan.json'}；实时状态：{BASE / 'artifacts/queue.json'}。")
        recovery_path = BASE / "artifacts/analysis_serialization_recovery.json"
        if recovery_path.exists():
            recovery = json.loads(recovery_path.read_text())
            blocks.append("## 汇总中断与恢复\n"
                          "- E1完成后的汇总曾因improved_seeds中的NumPy int64无法写入JSON而停止；不是模型数值异常或显存不足。\n"
                          "- 仅将统计计数转换为Python int，增加3个回归测试并在调度前执行；不改变指标数值、训练代码、模型或超参数。\n"
                          f"- 恢复前完整校验并保留{recovery['verified_complete_before_resume']}个已完成任务；受中断的未完成折从原初始化锚点重启。原始停止状态、逐组计数和代码哈希保存在{recovery_path}。")
        for method in config()["methods"]:
            path = BASE / "outputs" / method / "summary.json"
            if path.exists():
                blocks.append(handoff_group(json.loads(path.read_text())))
        final_path = BASE / "outputs/final_report.txt"
        if final_text is None and final_path.exists():
            final_text = final_path.read_text()
        if final_text is not None:
            blocks.append("## 总总结\n" + final_text)
        blocks.append(end)
        block = "\n\n".join(blocks) + "\n"
        if start in old and end in old:
            new = old[:old.index(start)] + block + old[old.index(end) + len(end):].lstrip("\n")
        else:
            new = old.rstrip() + ("\n\n" if old.strip() else "") + block
        tmp = BASE / "artifacts/handoff.tmp"
        tmp.write_text(new)
        os.replace(tmp, target)
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def key_comparison_text(reports, method, reference):
    comp = reports[method]["comparisons"][reference]
    return f"{method}−{reference}：{delta_text(comp)}；ACC/BA/AUC改善seed数{comp['improved_seeds']} /3，改善折数{comp['improved_folds']} /30。"


def cluster_bootstrap(method, reference, rows_by_method):
    cfg = config()
    seeds = cfg["seeds"]
    first = rows_by_method[method][seeds[0]]
    sites = sorted({r["site"] for r in first})
    groups = [np.array([i for i, r in enumerate(first) if r["site"] == site]) for site in sites]
    rng = np.random.default_rng(726104)
    draws = {key: [] for key in ("ACC", "BA", "AUC")}
    def task_arrays(name):
        return {s: (np.array([int(r["label_original"]) for r in rows_by_method[name][s]]),
                    np.array([int(r["pred_original"]) for r in rows_by_method[name][s]]),
                    np.array([float(r["prob_asd"]) for r in rows_by_method[name][s]])) for s in seeds}
    ma, ra = task_arrays(method), task_arrays(reference)
    for _ in range(cfg["bootstrap_replicates"]):
        indices = np.concatenate([groups[g] for g in rng.integers(0, len(groups), size=len(groups))])
        delta = {key: [] for key in draws}
        for seed in seeds:
            m = binary_metrics(*(v[indices] for v in ma[seed]))
            r = binary_metrics(*(v[indices] for v in ra[seed]))
            for key in draws:
                delta[key].append(m[key] - r[key])
        for key in draws:
            draws[key].append(np.mean(delta[key]))
    return {"method": method, "reference": reference, "clusters": len(groups),
            "replicates": cfg["bootstrap_replicates"],
            "conditional_95_percentile_intervals": {key: np.quantile(values, [0.025, 0.975]).tolist() for key, values in draws.items()},
            "unit": "Acquisition-site clusters, same resampling for methods and all three seeds",
            "limitation": "Conditional on fitted transductive CV models. No refitting, no independent training-cohort uncertainty, no claim of unseen-site generalization; no p values."}


def verify_immutable_inputs():
    audit = source_guard()
    folder = Path(config()["data_folder"])
    for relative, digest in audit["data_sha256"].items():
        if sha_file(folder / relative) != digest:
            raise ValueError(f"Original input changed during the experiment: {relative}")
    if sha_file(BASE / "artifacts/context.pt") != audit["context_sha256"]:
        raise ValueError("Audited experiment context changed")
    protected = ["model", "utils", "opt.py", "train_mm_gtunets.py", "requirements.txt",
                 "tests", "experiments/official_source_ppu_20261014"]
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *protected],
                   cwd=ROOT, check=True, capture_output=True)
    record = {"time_utc": utc(), "passed": True,
              "unchanged_data_files": len(audit["data_sha256"]),
              "data_digest": audit["data_digest"],
              "unchanged_source_files": len(audit["source_sha256"]),
              "protected_tracked_files_unchanged_against_HEAD": protected,
              "context_sha256": audit["context_sha256"]}
    atomic_json(record, BASE / "artifacts/final_input_integrity.json")
    return record


def final_analysis():
    cfg = config()
    integrity = verify_immutable_inputs()
    reports = {method: summary(method) for method in cfg["methods"]}
    all_results, all_oof = {}, {}
    for method in cfg["methods"]:
        all_results[method], all_oof[method] = load_results(method)
    if sum(len(rows) for seeds in all_oof.values() for rows in seeds.values()) != 20904:
        raise ValueError("Expected exactly 20904 actual OOF records")
    bootstrap = [cluster_bootstrap(m, r, all_oof) for m, r in (("E6", "E1"), ("E6", "E5"), ("E7", "E6"))]
    table_header = "| Method | ACC | BA | AUC | SEN | SPE | F1 | ΔACC vs E0 | ΔAUC vs E0 |"
    lines = ["DRPA 正式实验报告", "完成240/240任务；每组2613条OOF，总计20904条，无重复、缺失或标签错位。",
             "主表：三seed的10折均值 ± 三seed均值SD（百分点，ddof=1）；ASD为阳性，训练编码未改变。",
             table_header, "|---|---|---|---|---|---|---|---|---|"]
    for method, report in reports.items():
        values = [f"{report['mean'][key]*100:.2f}±{report['std_across_three_seed_means_ddof1'][key]*100:.2f}" for key in KEYS]
        delta_acc = (report["mean"]["ACC"] - reports["E0"]["mean"]["ACC"]) * 100
        delta_auc = (report["mean"]["AUC"] - reports["E0"]["mean"]["AUC"]) * 100
        lines.append(f"| {method} {report['name']} | " + " | ".join(values) + f" | {delta_acc:+.2f} | {delta_auc:+.2f} |")
    lines += ["补表：30个seed-fold的描述性SD（百分点，ddof=0），用于展示折间波动；不是30个独立临床样本，也不是主表seed均值SD。",
              "| Method | ACC SD | BA SD | AUC SD | SEN SD | SPE SD | F1 SD |",
              "|---|---|---|---|---|---|---|"]
    for method, report in reports.items():
        lines.append(f"| {method} | " + " | ".join(
            f"{report['std_across_30_seed_folds_ddof0'][key]*100:.2f}" for key in KEYS) + " |")
    history = json.loads((BASE / "artifacts/history_check.json").read_text())
    lines.append(f"历史核对：seed911逐折ACC/AUC按官方5位小数一致={history['exact_at_original_output_precision']}；均值ΔACC={history['mean_delta_ACC']*100:+.4f}、ΔAUC={history['mean_delta_AUC']*100:+.4f}个百分点。")
    evidence_path = BASE / "artifacts/baseline_diagnostics/verification.json"
    if evidence_path.exists():
        evidence = json.loads(evidence_path.read_text())
        lines.append(f"PPU重复性诊断：同初始化/VAE/测试RNG、无hooks/额外反向的官方原函数三次Fold0 ACC="
                     + "/".join(f"{c['ACC']*100:.2f}%" for c in evidence['controls'])
                     + "；同一物理PPU也有漂移。具体内核尚未定位，不能把配对锚点说成历史逐折精确复现。")
    acceptance_path = BASE / "artifacts/history_acceptance.json"
    if acceptance_path.exists():
        acceptance = json.loads(acceptance_path.read_text())
        lines.append(f"历史对照继续条件：{acceptance['decision']}；原始严格检查passed={history['passed']}，完整逐折差值保留在history_check.json。")
    paired = json.loads((BASE / "artifacts/paired_disabled_checks.json").read_text())
    if not paired["passed"] or paired["n_actual_seed_folds"] != 30:
        raise ValueError("Missing full actual-anchor equivalence evidence")
    lines.append("数值等价性验证：30个实际锚点均通过；原模型和关闭新增损失的模型都在测试专用确定性执行环境中比较，正式训练未开启该附加开关。最大前向/参数梯度/Adam更新误差="
                 + "/".join(f"{max(c[k] for c in paired['checks']):.3g}" for k in
                              ("max_forward_abs_error", "max_gradient_abs_error", "max_parameter_step_abs_error")) + "。")
    lines.append("原始生产路径的单步诊断：相同VAE潜向量、奖励图和融合输入图下，GTUNet/TopK选择可变化；原模型对自身的logits误差约0.005。因而很小的性能差值不能仅凭共享初始化归因为新增机制。证据见anchor_equivalence_diagnosis*.json。")
    lines += ["Q1 普通全局对齐：" + key_comparison_text(reports, "E2", "E1"),
              "Q1 相比原始模型：" + key_comparison_text(reports, "E2", "E0"),
              "Q2 非对称对齐：" + key_comparison_text(reports, "E4", "E3"),
              "Q3 疾病可靠性：" + key_comparison_text(reports, "E5", "E4"),
              "Q4 跨站点可靠性：" + key_comparison_text(reports, "E6", "E5"),
              "Q5 打乱对照：" + key_comparison_text(reports, "E7", "E6")]
    for method in ("E1", "E2", "E6", "E7"):
        a = reports[method]["projected_alignment"]
        lines.append(f"Q6 {method}共享表示：同人/异人cos={a['same_subject_cosine']:.4f}/{a['different_subject_cosine']:.4f}，gap={a['cosine_gap']:.4f}，I→P R@1={a['R_at_1_img_to_ph']*100:.2f}%，均值方差={a['mean_shared_variance']:.6f}。")
    lines.append("Q6 判断限制：低维表型有相似受试者，R@1与cos上升不等于疾病语义对齐；同时检查分类、方差及辅助分类。")
    aux = reports["E6"]["auxiliary_test"]
    lines.append(f"Q7 E6影像/表型辅助BA={aux['img']['BA']*100:.2f}/{aux['ph']['BA']*100:.2f}%，AUC={aux['img']['AUC']*100:.2f}/{aux['ph']['AUC']*100:.2f}%；原融合路径保留，但未显式分解shared/private，不能证明独有诊断信息完全保留。")
    positive = all(reports["E6"]["comparisons"]["E1"]["mean_delta"][k] > 0 for k in ("ACC", "BA", "AUC"))
    contribution = [m for m in ("E2", "E3", "E4", "E5", "E6")
                    if all(reports[m]["comparisons"][PRECEDING[m]]["mean_delta"][k] > 0 for k in ("ACC", "BA", "AUC"))]
    lines.append("Q8 " + ("完整DRPA相对辅助控制的三个均值指标均改善；" if positive else "完整DRPA相对辅助控制未实现三个均值指标一致改善；")
                 + f"直接对照三个均值指标一致改善的模块为{contribution}。当前只能提供原始协议下的机制证据，不能作为MICCAI创新已获验证的充分证据。")
    lines.append("Fold0分析（E6−E0，ACC/BA/AUC，百分点）：")
    comp = reports["E6"]["comparisons"]["E0"]
    for row in comp["fold0"]:
        lines.append(f"seed{row['seed']}：{row['ACC']*100:+.2f}/{row['BA']*100:+.2f}/{row['AUC']*100:+.2f}；其余9折总体均值："
                     + "/".join(f"{comp['other_folds_mean_delta'][k]*100:+.2f}" for k in ("ACC", "BA", "AUC")) + "。")
    e6_gate = reports["E6"]["gate"]; e5_gate = reports["E5"]["gate"]
    sites = e6_gate["sites"]
    lines.append(f"站点门控：E5/E6覆盖率={e5_gate['mean_nonzero_coverage']*100:.2f}/{e6_gate['mean_nonzero_coverage']*100:.2f}%；E6各站点覆盖范围={min(s['mean_nonzero_coverage'] for s in sites.values())*100:.2f}–{max(s['mean_nonzero_coverage'] for s in sites.values())*100:.2f}%；逐站点BA/AUC见summary.json，不能推断未见站点泛化。")
    lines += ["站点OOF对照（E6−E5，先每seed计算，再取三seed均值；仅两类样本量均足够的站点）：",
              "| Site | N ASD/HC | ΔBA | ΔAUC |",
              "|---|---|---|---|"]
    for site, by_seed in reports["E6"]["pooled_oof_sites"].items():
        if not all(record["eligible"] for record in by_seed.values()):
            continue
        first = by_seed[str(cfg["seeds"][0])]
        deltas = {key: np.mean([by_seed[str(seed)][key]
                               - reports["E5"]["pooled_oof_sites"][site][str(seed)][key]
                               for seed in cfg["seeds"]]) for key in ("BA", "AUC")}
        lines.append(f"| {site} | {first['n_asd']}/{first['n_hc']} | {deltas['BA']*100:+.2f} | {deltas['AUC']*100:+.2f} |")
    lines.append(f"E7打乱：索引移动{reports['E7']['gate']['mean_shuffle_moved_fraction']*100:.2f}%，实际权重改变{reports['E7']['gate']['mean_shuffle_changed_weight_fraction']*100:.2f}%。")
    for record in bootstrap:
        interval = record["conditional_95_percentile_intervals"]
        lines.append(f"条件式站点簇bootstrap {record['method']}−{record['reference']} 95%范围："
                     + "，".join(f"{k}[{interval[k][0]*100:+.2f},{interval[k][1]*100:+.2f}]" for k in ("ACC", "BA", "AUC"))
                     + "个百分点；固定已拟合模型、未重新训练，不作为独立临床显著性。")
    lines.append("局限：跨折RFE泄漏、传导式同站点CV、只有3个重复seed、弱表型近似负样本、无明确私有信息分解；PPU非逐位确定性使很小的增益也可能包含数值重复性误差，共用设备上的其他作业影响耗时；不隐瞒负向消融。")
    lines.append(f"输入完整性复核：{integrity['unchanged_data_files']}个原始数据文件及受保护源码哈希未变，根目录已跟踪文件和历史日志无改动。")
    lines.append("下一阶段：先在独立、无跨折筛选泄漏协议及留站点验证中检验可重复的疾病判别增益，再研究共享/私有信息的可识别性；不能以本轮测试结果继续调参。")
    full = "\n".join(lines) + "\n"
    final = {"time_utc": utc(), "tasks_completed": 240, "oof_records": 20904,
             "groups": reports, "conditional_site_cluster_bootstrap": bootstrap,
             "input_integrity": integrity, "disabled_loss_numerical_checks": paired,
             "limitations": ["Inherited feature-selection leakage", "Correlated repeated CV",
                             "No unseen-site validation", "Non-bitwise PPU training repeatability",
                             "Shared accelerator workloads affect timing comparisons"]}
    atomic_json(final, BASE / "outputs/final_summary.json")
    tmp = BASE / "outputs/final_report.tmp"; tmp.write_text(full)
    os.replace(tmp, BASE / "outputs/final_report.txt")
    update_handoff(full)
    print(full, flush=True)
    return final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=list(config()["methods"]))
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    if args.final:
        final_analysis()
    elif args.group:
        report = summary(args.group)
        update_handoff()
        print(f"GROUP COMPLETE {args.group}: {format_metrics(report['mean'])}", flush=True)
    else:
        parser.error("Use --group E0..E7 or --final")


if __name__ == "__main__":
    main()
