"""NO-GO reporting and independent statistics checks. Never runs a model."""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import time
import numpy as np
from common import (BASE, ROOT, D4, CFG, STAGE_A_CODE, sha_file, digest, load_torch,
                    atomic_json, utc)

METRICS = ("ACC", "SEN-ASD", "SPE-TD", "probability_AUC")


def four_metrics(labels, probability, prediction):
    from sklearn.metrics import roc_auc_score
    y, p, pr = np.asarray(labels), np.asarray(probability, float), np.asarray(prediction)
    assert np.isin(y, [0, 1]).all() and np.isin(pr, [0, 1]).all()
    assert len(y) == len(p) == len(pr) and np.isfinite(p).all()
    assert np.all((p >= 0) & (p <= 1)) and (y == 0).any() and (y == 1).any()
    return {"ACC": float(np.mean(pr == y)), "SEN-ASD": float(np.mean(pr[y == 0] == 0)),
            "SPE-TD": float(np.mean(pr[y == 1] == 1)),
            "probability_AUC": float(roc_auc_score(y == 0, p))}


def mean_sample_sd(values, scale=1.0):
    a = np.asarray(values, float) * scale
    assert len(a) and np.isfinite(a).all()
    return {"n": len(a), "mean": float(np.mean(a)), "sample_sd": float(np.std(a, ddof=1)) if len(a) > 1 else None}


def paired_changes(candidate, reference, scale=100.0):
    a, b = np.asarray(candidate, float), np.asarray(reference, float)
    assert a.shape == b.shape and a.ndim == 1 and len(a) > 1
    d = (a - b) * scale
    loo = [float(np.delete(d, i).mean()) for i in range(len(d))]
    return {**mean_sample_sd(d), "fold_delta_pp": d.tolist(), "positive_folds": int(np.sum(d > 0)),
            "leave_one_fold_out_min_pp": min(loo), "leave_one_fold_out_max_pp": max(loo)}


def patient_changes(candidate, reference):
    """Positive is candidate improvement; rows match by subject ID/fold/label."""
    a = {str(r["subject_id"]): r for r in candidate}
    b = {str(r["subject_id"]): r for r in reference}
    assert len(a) == len(candidate) and len(b) == len(reference) and set(a) == set(b)
    corrected, harmed = [], []
    class_net = {0: 0, 1: 0}
    for sid in sorted(a):
        ca, rb = a[sid], b[sid]
        assert (ca["fold"], ca["label"]) == (rb["fold"], rb["label"]), f"Mismatched patient row {sid}"
        c_good, r_good = ca["prediction"] == ca["label"], rb["prediction"] == rb["label"]
        if c_good and not r_good:
            corrected.append(sid); class_net[ca["label"]] += 1
        elif r_good and not c_good:
            harmed.append(sid); class_net[ca["label"]] -= 1
    return {"n_common": len(a), "corrected_reference_errors": len(corrected),
            "introduced_new_errors": len(harmed), "net_correct_change": len(corrected)-len(harmed),
            "ASD_net_correct_change": class_net[0], "TD_net_correct_change": class_net[1],
            "corrected_subject_ids": corrected, "harmed_subject_ids": harmed,
            "direction": "candidate minus reference"}


def statistical_tests():
    checks = {}
    y = np.array([0, 0, 0, 1, 1, 1])
    pr = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([.9, .6, .4, .3, .7, .4])
    out = four_metrics(y, p, pr)
    pairwise_auc = np.mean([(float(a > b) + .5 * float(a == b)) for a in p[y == 0] for b in p[y == 1]])
    checks["ACC_explicit_count"] = out["ACC"] == 4/6
    checks["SEN_ASD_is_class0"] = out["SEN-ASD"] == 2/3
    checks["SPE_TD_is_class1"] = out["SPE-TD"] == 2/3
    checks["probability_AUC_pairwise_rank_oracle_with_ties"] = abs(out["probability_AUC"]-pairwise_auc) < 1e-14
    summary = mean_sample_sd([1., 2., 3.])
    checks["sample_SD_ddof1"] = summary == {"n":3,"mean":2.,"sample_sd":1.}
    diff = paired_changes([.7,.8,.9], [.6,.8,.7])
    checks["paired_delta_pp_correct_direction"] = np.allclose(diff["fold_delta_pp"], [10.,0.,20.])
    checks["paired_positive_fold_count"] = diff["positive_folds"] == 2
    checks["LOO_independent_values"] = np.isclose(diff["leave_one_fold_out_min_pp"], 5) and np.isclose(diff["leave_one_fold_out_max_pp"], 15)
    cand = [{"subject_id":"a","fold":0,"label":0,"prediction":0},
            {"subject_id":"b","fold":1,"label":1,"prediction":1},
            {"subject_id":"c","fold":2,"label":0,"prediction":1}]
    ref = [{"subject_id":"c","fold":2,"label":0,"prediction":0},
           {"subject_id":"b","fold":1,"label":1,"prediction":0},
           {"subject_id":"a","fold":0,"label":0,"prediction":1}]
    trans = patient_changes(cand, ref)
    checks["asymmetric_patient_transition_2_rescued_1_harmed"] = trans["corrected_reference_errors"] == 2 and trans["introduced_new_errors"] == 1
    checks["patient_net_candidate_minus_reference"] = trans["net_correct_change"] == 1
    checks["class_specific_net_counts"] = trans["ASD_net_correct_change"] == 0 and trans["TD_net_correct_change"] == 1
    inverse = patient_changes(ref, cand)
    checks["patient_reversed_comparison_sign"] = inverse["net_correct_change"] == -1
    bad = [dict(r) for r in ref]; bad[0]["fold"] = 99
    try:
        patient_changes(cand, bad)
        checks["patient_fold_mismatch_blocks"] = False
    except AssertionError:
        checks["patient_fold_mismatch_blocks"] = True
    checks["all_statistics_JSON_roundtrip"] = json.loads(json.dumps({"metrics":out,"summary":summary,"delta":diff,"transitions":trans},allow_nan=False))["transitions"] == trans
    checks = {name: bool(value) for name, value in checks.items()}
    assert all(checks.values()), checks
    return {"utc": utc(), "passed": True, "passed_count": len(checks), "checks": checks,
            "new_model_forwards": 0, "formal_test_metrics_computed": 0, "data": "asymmetric synthetic only"}


def historical_metrics_from_saved_fields():
    """Aggregate saved metric fields only; no logits, labels, or probability scoring."""
    result = {}
    for group, directory, method in (
        ("historical_D0", D4, "D0"), ("historical_D4", D4, "D4"),
        ("historical_M1", ROOT / "experiments/PCDPAX_20261021", "M1"),
        ("historical_M3", ROOT / "experiments/PCDPAX_20261021", "M3")):
        folds = []
        for f in range(10):
            path = directory / "runs" / method / f"s911_f{f}.pt"
            saved = load_torch(path)
            m = saved["metrics"]
            translated = {"ACC": m["ACC"], "SEN-ASD": m.get("SEN-ASD", m.get("SEN")),
                          "SPE-TD": m.get("SPE-TD", m.get("SPE")),
                          "probability_AUC": m.get("probability_AUC", m.get("AUC"))}
            assert all(v is not None for v in translated.values())
            folds.append({"fold":f, "metrics":translated, "checkpoint_sha256":sha_file(path)})
        result[group] = {"metrics": {m: mean_sample_sd([r["metrics"][m] for r in folds], 100.) for m in METRICS},
                         "folds":folds, "only_saved_metric_fields_aggregated":True,
                         "not_used_for_stage_a_decision":True}
    return result


def final_audit(stage):
    audit = json.loads((BASE / "audit.json").read_text())
    manifest = json.loads((BASE / "manifest.json").read_text())
    mismatches = [p for p, h in audit["protected_sha256"].items() if sha_file(p) != h]
    frozen_mismatches = [n for n, h in manifest["code_sha256"].items() if sha_file(BASE/n) != h]
    assert not mismatches and not frozen_mismatches
    assert sha_file(BASE/"audit.json") == manifest["input_audit_sha256"]
    folds = stage["folds"]
    positive_count, above_count = 0, 0
    deltas = []
    for row in folds:
        probe = row["probe"]
        recompute = (probe["SSE_site"]-probe["SSE_full"]) / probe["SST"]
        assert recompute == probe["real_delta_R2"]
        assert row["native_forward_count"] == 1 and all(c == 1 for c in row["module_calls"].values())
        assert row["BN_eval_update_count"] == row["test_metric_evaluations"] == row["neural_training_count"] == 0
        assert len(probe["nulls"]) == len({r["permutation_sha256"] for r in probe["nulls"]}) == 32
        values = sorted(float(r["delta_R2"]) for r in probe["nulls"])
        # Independent order-statistic interpolation, rather than calling decide().
        position = .95 * (len(values)-1)
        lo = int(math.floor(position)); hi = int(math.ceil(position))
        p95 = values[lo] + (position-lo) * (values[hi]-values[lo])
        assert abs(p95-probe["null_p95"]) < 1e-14
        deltas.append(recompute)
        positive_count += int(recompute > 0)
        above_count += int(recompute > p95)
    ordered = sorted(deltas)
    independent_median = (ordered[4]+ordered[5])/2
    assert positive_count == stage["decision"]["gates"]["positive_delta"]["observed"] == 2
    assert above_count == stage["decision"]["gates"]["above_null_p95"]["observed"] == 4
    assert independent_median == stage["decision"]["median_delta_R2"]
    handoff = json.loads((BASE/"handoff_status.json").read_text()) if (BASE/"handoff_status.json").exists() else None
    assert not list((BASE/"runs").glob("**/*.pt"))
    formal_refusal = (BASE/"formal_gate_refusal.log").read_text()
    assert "NO_GO_STAGE_A: formal training and train-start preflight are forbidden" in formal_refusal
    result = {"utc":utc(), "status":"NO_GO_STAGE_A", "passed":True,
              "immutable_input_audit_sha256":sha_file(BASE/"audit.json"),
              "protected_files_verified":len(audit["protected_sha256"]), "protected_mismatches":mismatches,
              "frozen_stage_a_mismatches":frozen_mismatches,
              "stage_a_native_eval_forwards":10, "stage_a_null_fits":10*32*3,
              "historical_retrains":{"D0":0,"D4":0,"M1":0,"M3":0,"others":0},
              "formal_tasks_started":0, "formal_tasks_completed":0, "formal_task_limit":40,
              "formal_train_forwards":0,"formal_validation_forwards":0,"formal_test_forwards":0,
              "train_start_real_preflight_forwards":0,"new_checkpoint_count":0,"new_OOF_rows":0,
              "RFE_fits":0,"VAE_pretraining":0,"stage_a_logit_scoring":0,
              "formal_launcher_NO_GO_refusal_verified":True,
              "native_eval_BN_updates":0,"ridge_SVD_global_RNG_side_effects":False,
              "math_stage_a":json.loads((BASE/"math_stage_a.json").read_text())["passed"],
              "analysis_synthetic_tests":json.loads((BASE/"statistics_tests.json").read_text())["passed"],
              "independent_gate_counts_verified":True,
              "handoff":handoff,
              "formal_gradient_acceptance":"NOT_RUN: correctly conditional on Stage A GO; no S3 training protection claim",
              "stage_b_40_item_acceptance":"NOT_RUN for all Stage B items; Stage A boundaries/ridge/null tests passed",
              "git_commit_push_count":0,"all_new_artifacts_location":str(BASE)}
    atomic_json(result, BASE/"audit_final.json")
    return result


def aggregate_stage_a(stage):
    folds=stage["folds"]
    probes=[r["probe"] for r in folds]
    directions=[d for r in folds for d in r["shared_directions"]]
    nullrows=[b for p in probes for b in p["nulls"]]
    min_p=[min(d["protected_norm_squared"] for d in r["shared_directions"]) for r in folds]
    result={"decision":stage["decision"],"folds":[{
        "fold":r["fold"],"real_delta_R2":r["probe"]["real_delta_R2"],
        "null_mean":r["probe"]["null_mean"],"null_p95":r["probe"]["null_p95"],
        "real_minus_null_mean":r["probe"]["real_minus_null_mean"],
        "real_minus_null_p95":r["probe"]["real_minus_null_p95"],
        "minimum_protected_norm_squared":min(d["protected_norm_squared"] for d in r["shared_directions"]),
        "valid_shared_rank_min":min(d["effective_rank"] for d in r["shared_directions"]),
        "actual_joint_changed_fraction_mean":float(np.mean([b["actual_joint_changed_fraction"] for b in r["probe"]["nulls"]]))}
        for r in folds],
        "delta_R2":mean_sample_sd([p["real_delta_R2"] for p in probes]),
        "null_mean":mean_sample_sd([p["null_mean"] for p in probes]),
        "real_minus_null_mean":mean_sample_sd([p["real_minus_null_mean"] for p in probes]),
        "median_real_minus_null_mean":float(np.median([p["real_minus_null_mean"] for p in probes])),
        "minimum_protection_by_fold":min_p,
        "shared_rank_values":[d["effective_rank"] for d in directions],
        "in_sample_direction_variance_fraction":{"mean":float(np.mean([d["shared_direction_variance_fraction"] for d in directions])),
                                                 "min":float(min(d["shared_direction_variance_fraction"] for d in directions)),
                                                 "max":float(max(d["shared_direction_variance_fraction"] for d in directions)),
                                                 "interpretation":"fraction carried by direction, not cross-validated explained variance"},
        "shared_direction_stability_absolute_cosine_raw_axes":[r["shared_direction_stability_absolute_cosine_raw_axes"] for r in folds],
        "shared_disease_absolute_cosine":{"mean":float(np.mean([d["absolute_u_dot_d"] for d in directions])),
                                          "min":float(min(d["absolute_u_dot_d"] for d in directions)),
                                          "max":float(max(d["absolute_u_dot_d"] for d in directions))},
        "shared_disease_acute_angle_degrees":{"mean":float(np.mean([d["acute_angle_degrees"] for d in directions])),
                                              "min":float(min(d["acute_angle_degrees"] for d in directions)),
                                              "max":float(max(d["acute_angle_degrees"] for d in directions))},
        "null_actual_joint_changed_fraction":{"mean":float(np.mean([b["actual_joint_changed_fraction"] for b in nullrows])),
                                               "min":float(min(b["actual_joint_changed_fraction"] for b in nullrows)),
                                               "max":float(max(b["actual_joint_changed_fraction"] for b in nullrows))},
        "null_index_changed_fraction_mean":float(np.mean([b["index_changed_fraction"] for b in nullrows])),
        "null_sex_changed_fraction_mean":float(np.mean([b["sex_changed_fraction"] for b in nullrows])),
        "null_age_change_years_mean":float(np.mean([b["mean_absolute_age_change_years"] for b in nullrows]))}
    return result


def text_report(summary):
    a=summary["stage_a"]
    hist=summary["history_review"]
    bug=hist["experiment23_additional_implementation_failure"]
    lines=["# 实验24：TPSSA", "", "STATUS：**NO_GO_STAGE_A**。正式训练 **0/40**；历史重训 **0**；正式测试 forward **0**。",
           "", "## 阶段A：共享信息可行性", "",
           "只读恢复10份真实历史D4 best core及RP特殊权重；每折在其指定PPU上，使用冻结的私有诊断RNG执行一次原生eval forward。保留VAE采样和GTUNet eval dropout；每次仍有871个转导节点，只提取outer-train的16维影像表示。没有使用heldout诊断拟合或评价探针，也没有计算新模型的测试指标。",
           "", "Site-only和site+age+sex的ridge alpha均为1，使用历史D4完全相同的三个private folds，参考折拟合、query折OOF评价。年龄合法值中位数/标准化及站点类别均限制在outer-train；ridge输入再按各参考集合中心化标准化。历史private-fold分配曾用train标签做平衡，故不声称整个分区过程无标签。",
           "", "`ΔR²=(SSE_site−SSE_full)/SST`；下表为原始比例，0.002对应新增解释方差0.2个百分点。Null为每外折32次、private-fold×site内年龄/性别联合无标签错配。",
           "", "| Fold | 真实ΔR² | Null均值 | Null P95 | 真实−Null均值 | min保护范数² | 实际条件改变率 |",
           "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in a["folds"]:
        lines.append(f"| {r['fold']} | {r['real_delta_R2']:+.6f} | {r['null_mean']:+.6f} | {r['null_p95']:+.6f} | {r['real_minus_null_mean']:+.6f} | {r['minimum_protected_norm_squared']:.6f} | {100*r['actual_joint_changed_fraction_mean']:.2f}% |")
    lines += ["", "## 冻结门槛及停止决定", "", "| 门槛 | 观测 | 通过 |", "|---|---:|:---:|",
              f"| ΔR²>0至少7折 | 2/10 | 否 |", "| 严格超过Null P95至少7折 | 4/10 | 否 |",
              f"| 中位数>0.002 | {a['decision']['median_delta_R2']:+.9f} | 否 |",
              "| 至少7折三个参考集合都有非零有效方向 | 10/10 | 是 |",
              "| 至少7折三个参考方向保护后范数²均≥0.20 | 7/10 | 是 |", "",
              "三项信息充分性条件失败，因此停止本轮。真实条件相对Null均值较好不能替代真实ΔR²>0及预设规模；非零奇异值同样不能证明有可泛化的共享信息。未执行fold0 train-start训练模式预检，未创建S1–S4模型、优化器或checkpoint。正式启动器已实测拒绝NO-GO后的训练请求。",
              "", "## 机制解释", "",
              f"真实ΔR²十折中位数为{a['decision']['median_delta_R2']:+.6f}，即{100*a['decision']['median_delta_R2']:+.4f}个百分点影像方差的预测误差改善，方向为负。当前固定线性探针没有建立控制site后的足量年龄/性别新增信息。这不证明年龄/性别在所有表示、非线性关系或独立数据中永远无效。",
              f"32份Null的实际联合年龄/性别改变率平均为{100*a['null_actual_joint_changed_fraction']['mean']:.2f}%，最小{100*a['null_actual_joint_changed_fraction']['min']:.2f}%；索引改变率平均{100*a['null_index_changed_fraction_mean']:.2f}%，性别改变率{100*a['null_sex_changed_fraction_mean']:.2f}%，年龄绝对变化平均{a['null_age_change_years_mean']:.2f}岁。重复值没有被计作有效条件变化。该结论针对原始条件Null；S4表示错配未执行。",
              f"30个合法参考的共享方向非零，rank值为{sorted(set(a['shared_rank_values']))}；方向承载的站点残差影像方差比例均值为{100*a['in_sample_direction_variance_fraction']['mean']:.2f}%。这只是训练内方向方差，不能写成OOF新增可解释方差。共享/疾病方向绝对余弦范围{a['shared_disease_absolute_cosine']['min']:.3f}–{a['shared_disease_absolute_cosine']['max']:.3f}，夹角范围{a['shared_disease_acute_angle_degrees']['min']:.2f}–{a['shared_disease_acute_angle_degrees']['max']:.2f}度。保护几何门槛通过并不挽救信息门槛。",
              "三个private参考方向的稳定性在summary.json中按公共原始输出轴的绝对余弦报告。没有正式训练，故参数梯度保护、S2/S3/S4梯度比较、空间塌缩随epoch变化及对齐loss是否恒定均未检验；不把未执行项写成验收通过。",
              "", "## 论文与服务器代码核查", "",
              "原论文：[arXiv v3](https://arxiv.org/html/2406.14455v3)；指定仓库：[tangaii/MM](https://github.com/tangaii/MM)。实际服务器代码优先。FC上三角6670维由历史RFE压缩为500维；非影像仅SITE_ID、SEX、AGE_AT_SCAN，经冻结VAE升至500维；两个GTUNet各输出16维，共用由影像与非影像共同形成的融合人口图，之后原注意力融合与16→256→2分类器诊断。ph_unet输出不构成独立于影像的纯人口统计表示。",
              "论文描述分类训练300 epochs、8:1:1和相关距离；服务器历史执行500 epochs、约81:9:10（704/79/88或705/79/87），Torch相似度实现使用欧氏距离。原图edge dropout调用还使用其默认概率。实验24保留服务器协议，没有借机改成论文文字设置。",
              "保留的特殊行为：VAE eval仍采样（mm_gtunets.py:34）；GTUNet F.dropout未传training（gtunet.py:117）；rp_graph.py循环内return（:96，实际行号以源文件为准）；RP权重特殊注册/保存与共享Conv的.data约束（mm_gtunets.py:59、:76）；fusion贡献权重由torch.tensor重建而detach（:133）；原训练跨折覆盖x后的监督RFE污染（train_mm_gtunets.py）；全部871节点转导图；PPU长期轨迹非完全确定。8份指定源码SHA均与开始时一致。",
              "", "## 实验20–23真实失败机制复盘", ""]
    for key,value in hist["failures_review"].items():
        lines.append(f"- **{key}**：{value}")
    lines += ["", "**额外核实的实验23实现错误**：其冻结residual.py将query logits按private fold拼接，却使用未重排的原train-order self.y计算残差CE（191–194行）；raw P1对照也在228行重复此问题。10折中实际有46.52%–52.77%的训练诊断目标错配，平均50.19%。冻结manifest和当前历史源码SHA完全匹配，正式run.py调用该类，因此不是只存在于未采用的草稿代码。",
              f"独立合成嵌入检查：原实现R3 loss={bug['synthetic_oracle']['reported_R3_loss']:.6f}，保持同一logits而按query_indices使用正确标签后为{bug['synthetic_oracle']['same_logits_correct_query_target_loss']:.6f}；原raw P1诊断也共用错误顺序。故报告的0.99918不能单独证明它与**正确实现的历史P1**梯度重复。该错误混淆了历史机制解释，但不能据此唯一归因所有正式性能变化。历史源码、报告、模型均未修改，也未重训。",
              "实验22 patient_changes把left候选与right参考的变化反向计数，原合成自测两组完全相同未能发现。新分析器用2人纠正、1人新增错误的非对称样例检验，candidate−reference应为+1；同时验证ASD/TD净变化和fold/label匹配，未改历史文件。",
              "", "## 预注册S1–S4与正式四指标", "",
              "完整公式见registration.md，以下仅说明被门槛阻止的研究方案。S1为参考标准化后固定ridge映射的全16维MSE；S2用站点残差age/sex与影像交叉协方差首左奇异向量u（k=1）投影MSE；S3只在新增影像对齐分支采用`h.detach()+(h−h.detach())@(I−ddᵀ)`；S4仅替换private-fold×site内非影像对齐query对应。D4 CE独立保留其5%预算/epoch0..9 warmup/cap0.5；新增image/ph各自5%输出梯度预算，采用独立detach分支尺度。没有旧ph疾病原型CE、疾病概率KL、新MLP、Gate、Margin或InfoNCE。",
              "S1–S4均未训练，无正式四指标、逐折AUC、OOF、checkpoint差异或消融结论。以下只汇总只读历史artifact已有逐折四指标，未对历史logits重新评分，也未参与阶段A规则调整。",
              "", "| 只读历史组 | ACC % | SEN-ASD % | SPE-TD % | probability AUC % |", "|---|---:|---:|---:|---:|"]
    for name,row in summary["historical_saved_metrics"].items():
        cells=[f"{row['metrics'][m]['mean']:.4f} ± {row['metrics'][m]['sample_sd']:.4f}" for m in METRICS]
        lines.append("| "+name+" | "+" | ".join(cells)+" |")
    lines += ["", "历史表先每折、再10折mean±sample SD(ddof=1)，没有混入seed912/913。fold_metrics.csv与oof.csv仅保留新组输出字段表头，数据行均为0，不能当作完成40折。",
              "", "## 科研判断与后续边界", "",
              "科研判断：**NO_GO_STAGE_A**，本轮TPSSA不进入MICCAI 2027第一创新点候选确认。最早失败点是信息充分性：保护可用且存在非零方向，仍没有达到年龄/性别控制站点后的OOF新增解释方差要求。没有证据支持强行投入40折对齐训练，也没有证据称S3优于D4/M3或S4。",
              "后续独立注册应先解决实验23行顺序审计与可重复的信息充分性问题；任何新方法、无泄漏RFE、多seed或独立站点验证都需要另行授权，本轮不新增这些任务。仅对seed911反复设计的结果不支持临床泛化或录用标准判断。",
              "", "## 完整性、资源、SHA与handoff", "",
              f"阶段A的69项数值/边界检查和{summary['statistics_tests']['passed_count']}项统计合成检查通过。225份原始源码、输入、历史模型及报告SHA完成终检；冻结manifest和input audit保持不变。正式梯度40项验收与真实train-start预检因NO-GO未执行；本轮不声称S3投影实现已经在真实训练上验收。",
              f"GPU0只执行f0/2/4/6/8的5次历史eval；GPU1只执行f1/3/5/7/9的5次历史eval。Stage A双卡墙钟{summary['execution']['seconds']:.2f}秒；单worker峰值allocated显存约{summary['resources']['max_peak_allocated_GiB']:.3f} GiB。小矩阵无梯度ridge/SVD在CPU，未改变原生RNG。所有新文件位于/tmp，这是用户明确指定路径，不冒称写入/data实验目录。",
              f"冻结阶段A manifest SHA256：`{summary['sha256']['manifest.json']}`；config SHA256：`{summary['sha256']['config.json']}`；stage_a.json SHA256：`{summary['sha256']['stage_a.json']}`。各折历史D4 checkpoint、train_start、splits、RFE、VAE、CSV及源码SHA清单见audit.json与manifest.json。新checkpoint数量为0。",
              "handoff追加实际失败：/data返回errno=5 Input/output error。原文件仍为218804字节，SHA256仍为fd2ae885fe9af470fcb1c78f5503d3435e3a8ee9ae8cba98ccc879f9940dccff，旧前缀完全不变；原文件结束标记为0，未伪报成功。待追加正文位于handoff_pending.md，包含唯一结束标记；存储恢复后应先核对原SHA再追加，不需要重训。具体审计见handoff_status.json。", ""]
    return "\n".join(lines)


def main_report():
    stage=json.loads((BASE/"stage_a.json").read_text())
    assert stage["decision"]["status"] == "NO_GO_STAGE_A" and not stage["decision"]["passed"]
    tests=statistical_tests()
    atomic_json(tests,BASE/"statistics_tests.json")
    analysis_manifest={"utc":utc(),"stage":"post-decision analysis only",
                       "code_sha256":{n:sha_file(BASE/n) for n in ("analyze.py","history_review.py")},
                       "stage_a_manifest_sha256":sha_file(BASE/"manifest.json"),
                       "statistical_tests_sha256":sha_file(BASE/"statistics_tests.json")}
    atomic_json(analysis_manifest,BASE/"analysis_manifest.json")
    history=json.loads((BASE/"history_review.json").read_text())
    final=final_audit(stage)
    execution=json.loads((BASE/"stage_a_execution.json").read_text())
    # Empty new-group artifacts are explicit and cannot be misread as completed results.
    with (BASE/"fold_metrics.csv").open("w") as f:
        csv.writer(f).writerow(["group","seed","fold","ACC","SEN-ASD","SPE-TD","probability_AUC"])
    with (BASE/"oof.csv").open("w") as f:
        csv.writer(f).writerow(["group","seed","subject_id","fold","label","prediction","probability_ASD"])
    names=["manifest.json","config.json","stage_a.json","audit.json","audit_final.json","analysis_manifest.json","registration.md"]
    handoff=json.loads((BASE/"handoff_status.json").read_text())
    summary={"experiment":24,"name":"TPSSA","status":"NO_GO_STAGE_A","utc":utc(),
             "scientific_decision":"NO_GO_STAGE_A","formal_tasks_started":0,"formal_tasks_completed":0,
             "formal_task_limit":40,"formal_group_metrics":{g:None for g in CFG["groups"]},
             "formal_OOF_rows":0,"formal_checkpoint_count":0,
             "stage_a":aggregate_stage_a(stage),"history_review":history,
             "historical_saved_metrics":historical_metrics_from_saved_fields(),
             "statistics_tests":tests,"audit_final":final,"execution":execution,
             "resources":{"max_peak_allocated_GiB":max(f["device"]["peak_allocated_bytes"] for f in stage["folds"])/2**30,
                          "device_names":sorted(set(f["device"]["name"] for f in stage["folds"])),
                          "CPU_neural_training":False,"host_threads_per_worker":1},
             "sha256":{n:sha_file(BASE/n) for n in names},
             "deviations":["Experiment23 history resolved to /tmp/PCRPA_20261022 because requested /data path is absent.",
                           "Conservative all-three-reference aggregation for rank/protectability frozen before any Stage A extraction.",
                           "handoff.md append failed with errno=5 EIO; original bytes/SHA unchanged; body retained in handoff_pending.md."],
             "output_location":{"actual":str(BASE),"per_user_tmp_request":True,"written_data_experiment_directory":False},
             "handoff":handoff}
    atomic_json(summary,BASE/"summary.json")
    report=text_report(summary)
    (BASE/"report.md").write_text(report)
    print(json.dumps({"status":summary["status"],"formal_tasks":0,"stage_a_gates":summary["stage_a"]["decision"]["gates"],
                      "report":str(BASE/"report.md"),"protected_files":final["protected_files_verified"]},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["test","report"])
    args=p.parse_args()
    if args.mode=="test":
        print(json.dumps(statistical_tests(),indent=2,ensure_ascii=False))
    else:
        main_report()
