"""Independent acceptance, descriptive uncertainty and Chinese handoff."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import softmax
from sklearn.metrics import roc_auc_score

from diagnose import (BASE, ROOT, PILOT, OLD, CFG, STAT_NAMES, SEM, DESC, array_sha,
    tree_hash, sha_file, code_hash, atomic_bytes, save_json, load, read_inputs,
    protected_guard, diagnostic_metrics, private_seed, log, utc)


def plain(value):
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def mean_sd(values):
    a = np.asarray(values, dtype=float)
    return {'mean': float(a.mean()), 'fold_sample_sd': float(a.std(ddof=1)),
            'min': float(a.min()), 'max': float(a.max())}


def aggregate_retrieval(frozen):
    output = {}
    for setting in CFG['C']['settings']:
        folds = [frozen['retrieval'][f][setting] for f in range(10)]
        metrics = {key: mean_sd([d['metrics'][key] for d in folds]) for key in folds[0]['metrics']}
        baseline = {key: float(np.mean([d['random_baseline'][key] for d in folds])) for key in folds[0]['random_baseline']}
        true_gap = float(np.mean([d['cosine_advantage'].mean() for d in folds]))
        null = {}
        for key in ('R1', 'R5', 'MRR', 'median_rank', 'mean_rank_percentile', 'cosine_advantage'):
            values = np.array([np.mean([d['nulls'][rep]['metrics'][key] for d in folds]) for rep in range(64)])
            actual = true_gap if key == 'cosine_advantage' else metrics[key]['mean']
            null[key] = {'mean': float(values.mean()), 'descriptive_95_interval': np.quantile(values, [.025, .975]),
                'replicate_values': values, 'true_minus_null_mean': float(actual-values.mean()),
                'upper_tail_permutation_fraction': float((1+np.sum(values >= actual))/65)}
        output[setting] = {'metrics': metrics, 'random_baseline': baseline, 'null': null,
            'mean_matched_cosine': float(np.mean([d['paired_cosine'].mean() for d in folds])),
            'mean_unmatched_cosine': float(np.mean([d['unpaired_mean_cosine'].mean() for d in folds])),
            'mean_cosine_advantage': true_gap,
            'fraction_above_mean_unmatched': float(np.mean([d['matched_above_mean_unmatched_fraction'] for d in folds])),
            'R1_excess_percentage_points': 100*(metrics['R1']['mean']-baseline['R1']),
            'R1_ratio_to_random': metrics['R1']['mean']/baseline['R1'],
            'MRR_above_own_null_median_folds': sum(d['metrics']['MRR'] > np.median([n['metrics']['MRR'] for n in d['nulls']]) for d in folds),
            'candidate_counts': [d['n_candidates'] for d in folds],
            'folds': [{'fold': f, 'n_queries': d['n_queries'], 'n_candidates': d['n_candidates'],
                'metrics': d['metrics'], 'random_baseline': d['random_baseline'],
                'mean_cosine_advantage': float(d['cosine_advantage'].mean()),
                'matched_fraction_above_unmatched': d['matched_above_mean_unmatched_fraction'],
                'advantage_quantiles': d['advantage_quantiles_P05_P25_P50_P75_P95'],
                'null_R1_interval': np.quantile([n['metrics']['R1'] for n in d['nulls']], [.025, .975]),
                'null_MRR_interval': np.quantile([n['metrics']['MRR'] for n in d['nulls']], [.025, .975]),
                'exact_score_ties_at_truth': d['exact_score_ties_at_truth']} for f, d in enumerate(folds)]}
    return output


def independent_acceptance(context, cache, splits, rows, plan, result):
    assert result['complete'] and result['code_sha256'] == code_hash() and plan['code_sha256'] == code_hash()
    assert array_sha(plan['embedding']) == plan['embedding_sha256']
    max_metric_error = 0.
    max_regression_error = 0.
    n_probes = 0
    null_maps = 0
    retrieval_checks = 0
    text_lookup = {t: i for i, t in enumerate(plan['texts'])}
    for f in range(10):
        outer = splits[f]['train']
        outside = np.concatenate([splits[f]['val'], splits[f]['test']])
        for stage in ('A3', 'B'):
            coverage = np.zeros(871, int)
            for part in plan[stage][f]:
                tr, va = part['train'], part['holdout']
                assert not np.intersect1d(tr, outside).size and not np.intersect1d(va, outside).size
                assert not np.intersect1d(tr, va).size
                coverage[va] += 1
                ref = DESC.reference(cache['raw_stats'], tr)
                assert tree_hash(ref) == tree_hash(part['reference'])
                altered = cache['raw_stats'].copy()
                altered[np.concatenate([va, outside])] += 1000
                assert tree_hash(DESC.reference(altered, tr)) == tree_hash(ref)
                members = np.concatenate([tr, va])
                regenerated = DESC.captions(cache['raw_stats'][members], [cache['pairs'][i] for i in members], ref)
                np.testing.assert_array_equal([text_lookup[t] for t in regenerated], np.concatenate([part['text_train'], part['text_holdout']]))
                if stage == 'B':
                    for key in ('train', 'holdout'):
                        perm = part[key + '_permutation']
                        np.testing.assert_array_equal(np.sort(perm), np.arange(len(part[key])))
                        null_maps += 1
            assert np.all(coverage[outer] == 1) and np.all(coverage[outside] == 0)
        a3 = result['probes']['A3'][f]
        np.testing.assert_array_equal(a3['indices'], outer)
        target = cache['raw_stats'][outer]
        for predictions, r_key, m_key in ((a3['prediction'], 'R2', 'MAE'), (a3['baseline_prediction'], 'baseline_R2', 'baseline_MAE')):
            manual_r2 = 1-((target-predictions)**2).sum(0)/((target-target.mean(0))**2).sum(0)
            manual_mae = np.abs(target-predictions).mean(0)
            max_regression_error = max(max_regression_error, float(np.max(np.abs(manual_r2-a3[r_key]))), float(np.max(np.abs(manual_mae-a3[m_key]))))
        b = result['probes']['B'][f]
        np.testing.assert_array_equal(b['indices'], outer)
        y = (context['labels'][outer] == 0).astype(int)
        for g, p in b['probability'].items():
            pred = p >= .5
            manual = {'ACC': float((pred == y).mean()),
                'BA': .5*(float(pred[y == 1].mean())+float((~pred[y == 0]).mean())),
                'AUC': float(roc_auc_score(y, p)),
                'CE': float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean()), 'Brier': float(((y-p)**2).mean())}
            for key, value in manual.items():
                max_metric_error = max(max_metric_error, abs(value-b['metrics'][g][key]))
            n_probes += 1
        for fit in b['fits']:
            assert set(fit['train_indices']).issubset(set(outer))
            assert fit['train_labels_sha256'] == array_sha(context['labels'][fit['train_indices']])
        with torch.random.fork_rng(devices=[]):
            head = SEM.SemanticHead().eval()
        head.load_state_dict(rows['S1', f]['best']['head'], strict=True)
        for p in head.parameters():
            p.requires_grad_(False)
        original_head_sha = tree_hash(head.state_dict())
        with torch.no_grad():
            z = torch.nn.functional.normalize(head.project(rows['S1', f]['test_image_embedding']), dim=1).double().numpy()
            text = torch.nn.functional.normalize(cache['folds'][f]['embedding'], dim=1).double().numpy()
        for setting, d in result['frozen']['retrieval'][f].items():
            query = np.asarray([context['ids'].index(sid) for sid in d['query_ids']])
            candidate = np.asarray([context['ids'].index(sid) for sid in d['candidate_ids']])
            assert np.array_equal(np.sort(candidate), candidate)
            scores = z[query] @ text[candidate].T
            assert array_sha(scores) == d['score_sha256']
            for rep in range(-1, 64):
                if rep == -1:
                    positions = d['truth_positions']
                    expected = d['ranks']
                else:
                    perm = np.random.default_rng(private_seed(911, f, setting, rep, 'GBSA-Diagnostic-retrieval-null')).permutation(len(candidate))
                    assert array_sha(perm) == d['nulls'][rep]['permutation_sha256']
                    inverse = np.argsort(perm)
                    positions = inverse[d['truth_positions']]
                    expected = d['nulls'][rep]['ranks']
                selected = scores[np.arange(len(query)), positions]
                independent = 1+(scores > selected[:, None]).sum(1)+((scores == selected[:, None]) & (np.arange(len(candidate))[None] < positions[:, None])).sum(1)
                np.testing.assert_array_equal(independent, expected)
                retrieval_checks += 1
            true = scores[np.arange(len(query)), d['truth_positions']]
            wrong = np.array([np.delete(row, pos).mean() for row, pos in zip(scores, d['truth_positions'])])
            np.testing.assert_allclose(true-wrong, d['cosine_advantage'], rtol=0, atol=1e-12)
            assert not d['labels_accessed']
        assert tree_hash(head.state_dict()) == original_head_sha
        for g in ('C0', 'S1'):
            r = rows[g, f]
            np.testing.assert_array_equal(softmax(r['logits'], axis=1)[:, 0], r['prob_asd'])
        assert result['frozen']['checkpoint_audit'][f]['test_rng_equal'] == (f not in (6, 8))
    assert max_metric_error < 1e-12 and max_regression_error < 1e-12
    return {'completed_folds': 10, 'historical_checkpoints': 20, 'inner_prediction_groups': n_probes,
        'inner_reference_no_heldout_dependence_checks': 60, 'probe_partition_null_bijections': null_maps,
        'independent_retrieval_rank_checks': retrieval_checks, 'probe_metric_max_abs_error': max_metric_error,
        'regression_metric_max_abs_error': max_regression_error, 'outer_heldout_labels_used_for_probes': False,
        'retrieval_diagnostic_labels_accessed': False, 'GTUNet_forward': 0,
        'historical_text_geometry_reproduced': True, 'frozen_head_state_preserved': True}


def choose_outcome(summary):
    reconstruction = np.array([summary['A3'][name]['R2']['mean'] for name in STAT_NAMES])
    representation = np.sum(reconstruction > 0) >= 4 and all(d['separability']['mean'] > .5 for d in summary['A2']['concepts'].values())
    b = summary['B']['groups']
    probe = (b['B1_text384']['AUC']['mean'] > max(.5, b['B2_permuted_text384']['AUC']['mean']) and
             b['B1_text384']['CE']['mean'] < b['B2_permuted_text384']['CE']['mean'] and summary['B']['B1_AUC_above_B2_folds'] >= 6)
    def above(setting):
        d = summary['C'][setting]
        return all(d['metrics'][k]['mean'] > d['null'][k]['descriptive_95_interval'][1] for k in ('R1', 'MRR'))
    matching = above('train_to_train') and above('test_to_test') and summary['C']['test_to_test']['MRR_above_own_null_median_folds'] >= 7
    weak = not above('test_to_test') and not above('test_to_all871')
    outcome = 'C' if representation and matching else 'B' if representation and probe and weak else 'A' if not representation and weak else 'D'
    return outcome, {'controlled_and_numeric_structure_retained': bool(representation), 'exploratory_text_ASD_probe_signal': bool(probe),
        'consistent_patient_matching': bool(matching), 'weak_test_matching': bool(weak),
        'rule': CFG['decision_rules']['automatic_decision']}


def compact_summary(audit, result, acceptance):
    a1 = result['A1']
    a2 = result['A2']
    a3 = result['probes']['A3']
    b = result['probes']['B']
    summary = {'experiment': CFG['experiment'], 'complete': True, 'folds': 10, 'seed': 911,
        'code_sha256': code_hash(), 'config_sha256': sha_file(BASE / 'config.json'),
        'A1': {'folds': a1, 'aggregate': {k: mean_sd([d[k] for d in a1.values()]) for k in
            ('mean_norm', 'centered_energy', 'covariance_effective_rank', 'historical_singular_value_effective_rank', 'duplicate_text_fraction')},
            'mean_cosine_quantiles': np.mean([d['cosine_quantiles_P05_P25_P50_P75_P95'] for d in a1.values()], axis=0),
            'fraction_cosine_above': {str(t): mean_sd([d['fraction_cosine_above'][str(t)] for d in a1.values()]) for t in CFG['A1']['cosine_thresholds']}},
        'A2': {'folds': a2, 'roi_distance': mean_sd([d['roi_cosine_distance'].mean() for d in a2.values()]),
            'between_concept_direction_cosine': np.mean([d['between_concept_mean_direction_cosine'] for d in a2.values()], axis=0),
            'concepts': {name: {
                'low_high_distance': mean_sd([d['concepts'][name]['low_to_high_cosine_distance'].mean() for d in a2.values()]),
                'small_numeric_distance': mean_sd([d['concepts'][name]['small_numeric_cosine_distance'].mean() for d in a2.values()]),
                'separability': mean_sd([d['concepts'][name]['leave_one_patient_out_balanced_accuracy'] for d in a2.values()]),
                'direction_consistency': mean_sd([d['concepts'][name]['mean_direction_cosine'] for d in a2.values()])} for name in CFG['A2']['concepts']}},
        'A3': {name: {key: mean_sd([d[key][j] for d in a3.values()]) for key in ('R2', 'MAE', 'baseline_R2', 'baseline_MAE')} for j, name in enumerate(STAT_NAMES)},
        'B': {'groups': {g: {k: mean_sd([d['metrics'][g][k] for d in b.values()]) for k in CFG['B']['metrics']} for g in CFG['B']['groups']},
            'folds': {f: {'n': len(d['indices']), 'metrics': d['metrics'], 'null_actual_subject_change_fraction': d['null_actual_subject_change_fraction']} for f, d in b.items()},
            'B1_AUC_above_B2_folds': sum(d['metrics']['B1_text384']['AUC'] > d['metrics']['B2_permuted_text384']['AUC'] for d in b.values()),
            'B1_AUC_above_B0_folds': sum(d['metrics']['B1_text384']['AUC'] > d['metrics']['B0_raw7']['AUC'] for d in b.values()),
            'convergence_warnings': [fit for d in b.values() for fit in d['fits'] if fit['warnings']]},
        'C': aggregate_retrieval(result['frozen']), 'checkpoint_audit': result['frozen']['checkpoint_audit'],
        'late_training': result['frozen']['late_training'], 'historical_11_results': audit['historical_11_results'],
        'historical_11_delta': audit['historical_11_delta'], 'execution': audit['execution'], 'acceptance': acceptance,
        'statistics': 'Mean and sample SD over ten overlapping historical outer-training sets; not independent cohorts or three-seed SD. Null64 is a fixed randomization reference, not 64 datasets.',
        'limitations': CFG['limitations'], 'handoff': {'status': 'not_yet_appended'}}
    summary['outcome'], summary['decision_evidence'] = choose_outcome(summary)
    return summary


def fmt_stat(entry, scale=1, digits=3):
    return f'{entry["mean"]*scale:.{digits}f} +/- {entry["fold_sample_sd"]*scale:.{digits}f}'


def report_text(s, audit):
    geometry = s['A1']
    c = s['C']
    b = s['B']['groups']
    evidence = s['decision_evidence']
    names = {'mean_fc': '平均FC', 'global_efficiency': '全局效率', 'modularity': '模块度'}
    outcome_text = {
        'A': '当前文本监督缺乏足够有效表示。停止当前MiniLM+固定模板+患者级InfoNCE配置，换用能明确保留结构/数值且诊断相关的方法。',
        'B': '文本保留一定脑网络与探索性ASD信息，但当前对齐没有建立稳定、有用的患者级对应。停止直接扩大当前配置；结构保持软对齐仅列为未来候选，必须与直接数值监督比较，本轮不实现。',
        'C': '患者图文配对存在一致优势，但实验11没有稳定疾病收益。优先研究面向诊断的表示学习，而不是继续强化身份匹配。',
        'D': 'Inconclusive：预注册表示、判别与配对证据未共同满足明确归因条件。不能强行写成语义编码或对齐的单一失败，停止自动扩展并保留各项矛盾证据。'}
    lines = ['# 实验11冻结机制诊断：GBSA-Diagnostic', '', '## 1. 问题与范围', '',
        '本轮不是实验12。全部10个seed911历史fold纳入；C0/S1、GTUNet、VAE、语义头及文本编码器训练均为0次，GTUNet前向也是0次。检验文本表示、训练集内部疾病探针和冻结患者级检索三个独立命题，不用语义梯度非零代替配对或诊断有效性。',
        '实验11的S1相对配对C0：ACC +0.112pp、AUC -0.218pp、CE -0.000789；没有达到预注册Go要求。其统计是单seed十折，不与历史三seed均值标准差混用。实验9中原生图效应极弱；实验10没有共同建立年龄/性别超越影像与错配Null的稳定排序/概率增益，这些只是研究背景，不预设本轮阴性。', '',
        '## 2. 原论文、源码及文本信息', '',
        'fMRI-LM包含多类影像描述、量化重建/域对抗/图文学习以及后续LLM和指令阶段；并不等同于本项目的冻结MiniLM辅助头。其描述消融同时移除预训练配对目标和F2T，不能单独验证我们的ASD辅助Loss。[论文3.1/3.2、Fig.2/4及4.3](https://arxiv.org/html/2511.21760v3)。',
        '论文称SigLIP-style，但式(3)写为softmax比值；源码分别实现CLIP、soft-CLIP、SigLIP BCE和soft-SigLIP，不应混称。[官方Loss实现](https://github.com/yuxiangwei0808/fMRI-LM/blob/main/utils_loss.py)。官方量化训练包含独立投影、池化与多种目标选项，本轮未复现这些完整阶段。[训练器](https://github.com/yuxiangwei0808/fMRI-LM/blob/main/train_quantizer_contr.py)。',
        '实验11描述来自同一AAL116 FC：7项全局数值（有符号平均、绝对平均、加权效率、模块度、聚类、节点强度均值/SD），再加最强2正/2负ROI编号及连接值。正相关Top10%图、1/r最短路径和确定性greedy模块度，与官方绝对值阈值、unweighted global_efficiency及Louvain实现有区别。[描述源码](https://github.com/yuxiangwei0808/fMRI-LM/blob/main/nbs_data/get_fmri_discriptor.py)。',
        '无诊断、年龄、性别、站点或症状文字；ROI编号不是已验证解剖知识。16→128→384头对原生影像输出施加对称InfoNCE，tau=.07，Loss除log(Ntrain)，系数热身至.05；分类推理不用文本。**同一份FC转换成文本不会凭空增加疾病信息**，这里只检验信息保留和辅助表示偏置。', '',
        '## 3. 冻结协议与数据边界', '',
        '所有方法、参数、64份Null规则和Outcome判据已在分析前写入config并登记SHA。A1/A2/A3不用ASD标签；A3为训练集内KFold3。B仅用历史外层train，StratifiedKFold3(random_state=911+fold)；每个内部训练子折重新拟合描述参考均值/SD和各自StandardScaler，使用同一固定MiniLM重新编码所需文本。Ridge alpha=1；LR C=1/L2/lbfgs/max_iter=2000/tol=1e-6，无搜索。B2仅在内部train/holdout各自独立、无标签置换文本行，标签和原始数值不动。',
        'C直接恢复已保存best head，投影同一次原生随机前向缓存的871×16影像表示。匹配真值仅由ID确定，不用诊断标签。候选按全局索引排序；分数并列以索引破同分。64份随机身份双射覆盖完整候选，不挑最差Null。测试样本虽未进入语义Loss，但仍处于历史转导人口图，不能称外院归纳式泛化。', '',
        '## 4. A1：文本几何与有效维度', '',
        '|fold|中心化能量|协方差有效秩|历史奇异值有效秩|余弦P05/P25/P50/P75/P95|>0.99 %|重复文本 %|',
        '|---|---|---|---|---|---|---|']
    for f, d in geometry['folds'].items():
        q = '/'.join(f'{x:.6f}' for x in d['cosine_quantiles_P05_P25_P50_P75_P95'])
        lines.append(f'|{f}|{d["centered_energy"]:.7f}|{d["covariance_effective_rank"]:.2f}|{d["historical_singular_value_effective_rank"]:.2f}|{q}|{100*d["fraction_cosine_above"]["0.99"]:.2f}|{100*d["duplicate_text_fraction"]:.2f}|')
    lines += ['', f'平均范数{geometry["aggregate"]["mean_norm"]["mean"]:.9f}；不同患者余弦中位数的折均值{geometry["mean_cosine_quantiles"][2]:.6f}。超过0.95/0.99/0.995的比例分别为'+ '/'.join(f'{geometry["fraction_cosine_above"][str(t)]["mean"]*100:.2f}%' for t in CFG['A1']['cosine_thresholds'])+'。全部无序非同人训练配对均被计入，不依赖少量随机抽样。逐折最小/平均/最大L2距离与各阈值均在summary中。',
        f'历史约148–149的rank是归一化奇异值熵；本轮协方差rank用其平方归一化，均值为{geometry["aggregate"]["covariance_effective_rank"]["mean"]:.2f}。不是历史结果被改动，而是定义不同。复算历史原定义逐折一致。共同模板方向占主导、细小差异仍可存在；高余弦并不意味着数学上完全坍缩。', '',
        '## 5. A2：受控语义/数值变化', '',
        '每折按ID哈希取32名train患者；概念低/高为训练P10/P90，保持其他模板文字与ROI连接不变，数值和等级同步；平均FC受该患者绝对FC均值界限约束。小变化为保留等级的±0.001；ROI控制只循环替换一处编号。它们是语言敏感性控制，不声称改变单个指标后对应一张物理可实现的新脑图。',
        '|概念|低→高余弦距离|±0.001余弦距离|留一患者低/高BA %|方向一致性cos|', '|---|---|---|---|---|']
    for name, d in s['A2']['concepts'].items():
        lines.append(f'|{names[name]}|{d["low_high_distance"]["mean"]:.7g}|{d["small_numeric_distance"]["mean"]:.7g}|{fmt_stat(d["separability"],100)}|{d["direction_consistency"]["mean"]:.4f}|')
    lines += ['', f'ROI编号替换距离折均值{s["A2"]["roi_distance"]["mean"]:.7g}；不同概念平均方向cos矩阵（平均FC/效率/模块度顺序）：`{np.array2string(np.asarray(s["A2"]["between_concept_direction_cosine"]),precision=4)}`。重复同批文本要求逐位相等，编码器权重前后SHA一致，检查结果见执行部分。',
        '低/高可分说明模型对这些文本字段有稳定响应；等级词也随低/高变化，因此不能单独证明理解了神经科学含义。纯小数与ROI编号控制分别量化数值敏感性和非解剖ID敏感性，不用任意距离阈值宣布成功。', '',
        '## 6. A3：七项连续统计的保留', '', '|指标|OOF R² 均值±SD|均值基线R²|OOF MAE|均值基线MAE|', '|---|---|---|---|---|']
    for name, d in s['A3'].items():
        lines.append(f'|{name}|{fmt_stat(d["R2"])}|{d["baseline_R2"]["mean"]:.4f}|{d["MAE"]["mean"]:.7g}|{d["baseline_MAE"]["mean"]:.7g}|')
    lines += ['', '所有回归在原物理单位；mean baseline逐个内部训练子折计算，不用全train均值冒充OOF基线。预测与真值可在runs/diagnostics.bin逐人复算。成功重构仅说明已写入文本的数值可被恢复，不是额外疾病知识，也没有测试直接数值监督能否改善GTUNet。', '',
        '## 7. B：仅外层训练集的疾病探针', '', '|组别|输入|ACC %|BA %|prob-AUC %|CE|Brier|', '|---|---|---|---|---|---|---|']
    descriptions = {'B0_raw7': '7维原数值', 'B1_text384': '384维真实文本', 'B2_permuted_text384': '384维标签无关错配文本'}
    for g, d in b.items():
        lines.append(f'|{g}|{descriptions[g]}|{fmt_stat(d["ACC"],100)}|{fmt_stat(d["BA"],100)}|{fmt_stat(d["AUC"],100)}|{fmt_stat(d["CE"],1,6)}|{fmt_stat(d["Brier"],1,6)}|')
    lines += ['', f'B1 AUC优于B2为{s["B"]["B1_AUC_above_B2_folds"]}/10折、优于B0为{s["B"]["B1_AUC_above_B0_folds"]}/10折。十个外层训练集合重叠，不能把其SD或90次LR拟合解释为独立临床证据；B0/B1维数不同，不是容量配平。B1即便超过B0，也可能利用ROI连接、非线性文本变换或容量，而非增加来自FC之外的信息。',
        f'LR收敛警告条数{len(s["B"]["convergence_warnings"])}。内部三折划分、全部训练标签/参考/特征SHA、每人OOF概率与原始三子折指标均保留，不接触外层val/test标签选参数。', '',
        '|fold|B0 AUC %|B1 AUC %|B2 AUC %|B0 CE|B1 CE|B2 CE|', '|---|---|---|---|---|---|---|']
    for f, d in s['B']['folds'].items():
        m = d['metrics']
        lines.append('|'+str(f)+'|'+'|'.join(f'{m[g]["AUC"]*100:.3f}' for g in descriptions)+'|'+'|'.join(f'{m[g]["CE"]:.6f}' for g in descriptions)+'|')
    lines += ['', '## 8. C：冻结患者图文检索', '',
        '下表为十折指标均值；R@1/R@5/MRR均为百分比，Median Rank是折内中位排名再平均；排名百分位0最好、随机期望50。三个候选范围分开，不混用随机基准。',
        '|设置|候选数|R@1 %|随机R@1 %|超出随机 pp|R@5 %|MRR %|随机MRR %|Median Rank|平均排名百分位|', '|---|---|---|---|---|---|---|---|---|---|']
    for setting, d in c.items():
        m, r = d['metrics'], d['random_baseline']
        lines.append(f'|{setting}|{min(d["candidate_counts"])}–{max(d["candidate_counts"])}|{m["R1"]["mean"]*100:.4f}|{r["R1"]*100:.4f}|{d["R1_excess_percentage_points"]:+.4f}|{m["R5"]["mean"]*100:.4f}|{m["MRR"]["mean"]*100:.4f}|{r["MRR"]*100:.4f}|{m["median_rank"]["mean"]:.2f}|{m["mean_rank_percentile"]["mean"]:.2f}|')
    lines += ['', '|设置|真实cos|平均错误cos|真实−错误cos|真实高于平均错配比例 %|Null R@1 95% %|Null MRR 95% %|', '|---|---|---|---|---|---|---|']
    for setting, d in c.items():
        r1 = '/'.join(f'{x*100:.4f}' for x in d['null']['R1']['descriptive_95_interval'])
        mrr = '/'.join(f'{x*100:.4f}' for x in d['null']['MRR']['descriptive_95_interval'])
        lines.append(f'|{setting}|{d["mean_matched_cosine"]:.7f}|{d["mean_unmatched_cosine"]:.7f}|{d["mean_cosine_advantage"]:+.8f}|{d["fraction_above_mean_unmatched"]*100:.2f}|{r1}|{mrr}|')
    lines += ['', '64份Null均在各自候选内置换身份，不移动图、不重算主干、不挑最差一次。Null分布完整保存。跨折按相同rep号平均形成描述性参照，不把64次或十个重叠fold当独立外部样本；有限随机置换尾部比例也不等于多重校正临床p值。', '',
        '|fold|train R@1 %|test R@1 %|test→871 R@1 %|train MRR %|test MRR %|train gap|test gap|', '|---|---|---|---|---|---|---|---|']
    for f in range(10):
        tr, te, all_ = [c[k]['folds'][f] for k in CFG['C']['settings']]
        lines.append(f'|{f}|{tr["metrics"]["R1"]*100:.4f}|{te["metrics"]["R1"]*100:.4f}|{all_["metrics"]["R1"]*100:.4f}|{tr["metrics"]["MRR"]*100:.4f}|{te["metrics"]["MRR"]*100:.4f}|{tr["mean_cosine_advantage"]:+.8f}|{te["mean_cosine_advantage"]:+.8f}|')
    late_gap = np.mean([d['matched_minus_unmatched_cosine'] for d in s['late_training'].values()])
    lines += ['', f'历史最后采样训练forward的匹配gap折均值为{late_gap:+.8f}；这里是最佳Checkpoint恢复后的原生随机缓存，再经其最佳语义头投影，权重/训练状态/随机性不同，不能要求等于训练中约3e-5的记录。',
        '高cos可能主要来自所有文本共享方向；只有正确相对错误的排名和间隔能判断身份对应。语义梯度非零和归一化Loss接近1均不是配对成功证据。', '',
        '## 9. Fold 6/8及其他八折状态审计', '', '|fold|C0/S1最佳epoch|实际训练epoch|BN计数|测试RNG同一|概率MAD|影像参数相对L2|BN最大绝对差|', '|---|---|---|---|---|---|---|---|']
    for f, d in s['checkpoint_audit'].items():
        lines.append(f'|{f}|{d["best_epochs"][0]}/{d["best_epochs"][1]}|{d["actual_training_epochs"][0]}/{d["actual_training_epochs"][1]}|{d["BN_batch_counts_C0_S1"][0]}/{d["BN_batch_counts_C0_S1"][1]}|{d["test_rng_equal"]}|{d["probability_MAD"]:.6f}|{d["state_difference_by_part"]["img_unet."]["relative_L2"]:.7g}|{d["BN_difference"]["max_abs"]:.7g}|')
    lines += ['', 'Fold6/8不仅训练权重不同，还选中了不同epoch和BN运行统计，实际训练长度及原生测试随机状态不同。VAE和eval Dropout使随机状态成为实际前向因素，不能把其较大概率差全部当成语义配对的直接效果。其他八折最佳epoch和测试RNG相同，但学习后的参数/BN依然可以不同；这也不等于唯一因果归因。完整core、BN、核心Adam、完整optimizer、best/test/end RNG哈希及各组件相等性均在summary/checkpoint_audit。没有重选epoch，没有补推理或改历史指标。', '',
        '## 10. 科学决策与主要瓶颈', '', f'**Outcome {s["outcome"]}**：{outcome_text[s["outcome"]]}',
        f'预注册证据标志：结构/数值保留={evidence["controlled_and_numeric_structure_retained"]}；探索性疾病探针信号={evidence["exploratory_text_ASD_probe_signal"]}；稳定患者匹配={evidence["consistent_patient_matching"]}；测试对应不足={evidence["weak_test_matching"]}。完整操作性判断写在config，未依据结果换阈值或模型。',
        '直接实测事实是A/B/C各项数字；模板共性、数值语义损失或硬负样本身份目标不适配，是需要结合这些数字解释的机制，不是本轮唯一因果证明。即便身份匹配成功，也不必然有ASD增益；即便当前失败，也不能推出所有语义/多模态对齐都失败。', '',
        '## 11. 下一步唯一优先方向', '',
        '优先寻找**直接面向疾病、保留脑网络结构和连续数值的表示学习机制**，并带直接数值监督对照和独立跨站点验证。fMRI-LM可保留为结构化描述的背景启发，不建议照搬本次MiniLM模板身份InfoNCE再扩seed或堆Gate。若本轮为B，软关系目标只作为未来待验证候选；若为D，先解决本轮证据矛盾而不是换一个更复杂头。没有自动实施后续方法。', '',
        '## 12. 执行、SHA与存储验收', '',
        f'分析10/10折；读取20份历史最佳产物但未复制。Ridge {s["execution"]["Ridge_fits"]}次、LR {s["execution"]["Logistic_fits"]}次；训练GTUNet/VAE/语义头/文本模型均0，GTUNet前向0，RFE0。三种检索×10折×64Null=1920次离线随机身份评价。',
        f'受控文本上限{s["execution"]["controlled_sentence_count_upper_bound"]}条；合并真实内部CV描述后唯一文本缓存{s["execution"]["unique_text_cache_rows"]}行。两张PPU在数值预检的最大差约2.83e-6，未通过未改变的2e-6绝对/1e-5相对容差；默认已为最高矩阵精度。没有放宽断言，A2/A3/B全体文本统一改用CPU FP32编码，不混用新旧设备向量。A1/C仍严格读取历史缓存，任何历史结果不改动。该纯执行后端修订发生在探针/检索结果产生之前，完整旧/新指纹和失败记录见audit.technical_revisions。',
        '|逻辑编码分片/后端|编码唯一文本数|编码墙钟秒|显存峰值GiB|实际CPU/PPU一致性|与历史缓存差异|重复同批逐位一致|', '|---|---|---|---|---|---|---|']
    for d in s['execution']['text_encoding']:
        comparison = f'{d["CPU_PPU_max_abs_error"]:.3g}' if d['CPU_PPU_max_abs_error'] is not None else 'PPU预检未通过，统一CPU'
        lines.append(f'|{d["logical_shard"]}/{d["backend"]}|{d["unique_texts_encoded"]}|{d["seconds"]:.2f}|{d["peak_allocated_bytes"]/2**30:.3f}|{comparison}|{d["historical_embedding_max_abs_error"]:.3g}|{d["repeat_exact"]}|')
    lines += ['', f'纯分析入口墙钟{s["execution"]["analysis_seconds"]:.2f}秒；包含审计/报告的总执行跨度见summary.execution.total_elapsed_seconds，不包含此前论文阅读与代码编写。存储写入/fsync/rename/重读/目录fsync及锁排他通过；全部内容内置SHA，完整结果重入跳过，源代码/config SHA `{s["code_sha256"]}`。',
        f'自动复核：{len(audit["protected_files"])}项保护文件SHA保持，历史20产物与模型文件manifest一致；独立指标最大差{s["acceptance"]["probe_metric_max_abs_error"]:.3g}；独立检索排名检查{s["acceptance"]["independent_retrieval_rank_checks"]}项；内部参考对held-out扰动不变60项。原prompt已被用户更新，与实验11旧prompt的差异单独登记，不伪报其旧SHA不变。',
        '结果产物只有两份必要缓存（统一文本、诊断/预测/Null）、代码/config、审计、summary和中文报告。临时编码分片合并验证后清除。不自动Git提交或push。handoff追加状态、前后SHA和历史字节保留数以summary.handoff为准。', '',
        '## 13. 适用边界', '',
        '单seed、重叠训练集合、小样本高维LR、有限64次随机化、转导人口图及历史跨折RFE泄漏均限制解释。B只证明这些FC衍生特征中的可读信息，不测影像之外的条件增量；没有新外部医院数据或训练干预来识别唯一失败原因。所有检验均未用外层测试标签选择方法、词汇、阈值或超参数。', '']
    return '\n'.join(lines)


def handoff_addition(summary):
    start = '<!-- GBSA_Diagnostic_20261019_START -->'
    end = '<!-- GBSA_Diagnostic_20261019_END -->'
    c = summary['C']
    lines = ['\n\n'+start, '', '## 实验11冻结机制诊断：GBSA-Diagnostic', '',
        '本轮不是实验12。十个seed911 fold全部纳入，只读20份C0/S1历史产物；神经网络训练、GTUNet前向、VAE/RFE训练均0。A检验文本数值/受控语义，B只在历史外层train内部三折拟合固定LR，C恢复冻结head并投影缓存影像做身份检索。',
        f'Outcome {summary["outcome"]}。证据标志：{summary["decision_evidence"]}。',
        f'非同人文本余弦中位数折均值{summary["A1"]["mean_cosine_quantiles"][2]:.6f}，>0.99比例{summary["A1"]["fraction_cosine_above"]["0.99"]["mean"]*100:.2f}%；协方差有效秩{summary["A1"]["aggregate"]["covariance_effective_rank"]["mean"]:.3f}。历史约149的rank基于奇异值熵，定义不同，两者均保留且不改历史。', '',
        '|训练内部探针|AUC %|CE|', '|---|---|---|']
    for g, d in summary['B']['groups'].items():
        lines.append(f'|{g}|{fmt_stat(d["AUC"],100)}|{fmt_stat(d["CE"],1,6)}|')
    lines += ['', '7维与384维容量不同，十个训练集合重叠，不能解释为独立疾病增益或外部泛化；文本来自同一FC，没有增加患者新疾病信息。内部参考/标准化逐子折拟合，无外层val/test标签用于探针。', '',
        '|检索|R@1 %|随机 %|R@5 %|MRR %|正确−错误cos|', '|---|---|---|---|---|---|']
    for setting, d in c.items():
        lines.append(f'|{setting}|{d["metrics"]["R1"]["mean"]*100:.5f}|{d["random_baseline"]["R1"]*100:.5f}|{d["metrics"]["R5"]["mean"]*100:.5f}|{d["metrics"]["MRR"]["mean"]*100:.5f}|{d["mean_cosine_advantage"]:+.8f}|')
    lines += ['', '三个候选范围各64份固定身份Null全部保留；不把随机化次数当独立队列，不用语义梯度非零证明正确患者匹配。Fold6/8的153/121、454/263最佳epoch、BN及测试RNG不同，较大概率变化不能唯一归因于语义监督；其他八折同epoch/RNG仍可能有不同权重和BN。没有重选Checkpoint或改历史成绩。',
        '下一步优先：直接面向疾病并保持脑网络结构/连续数值的表示学习，必须有直接数值监督对照和跨站点验证。fMRI-LM可作背景启发，不自动扩大当前MiniLM固定模板患者身份InfoNCE；软关系目标只是未来候选，未实现。',
        f'执行：Ridge30、LR90、检索Null1920；统一CPU冻结文本分片{[(d["logical_shard"],d["unique_texts_encoded"]) for d in summary["execution"]["text_encoding"]]}。两PPU预检差约2.83e-6，未通过原数值容差，因此正式新文本未采用PPU；未改变模型或放宽阈值，旧A1/C缓存不变。分析入口{summary["execution"]["analysis_seconds"]:.2f}秒。历史保护SHA{summary["acceptance"]["protected_files_SHA_unchanged"]}项，独立排名检查1950项。代码SHA `{summary["code_sha256"]}`。',
        f'全部完成并验收，完整数字、受控变化、数值回归、90份LR指纹、患者级内部OOF、64份Null及状态审计见 `{BASE}/report.md`、`summary.json`、`audit.json`、`runs/diagnostics.bin`。保留历史转导/监督RFE泄漏局限，不宣称无泄漏临床泛化，不自动Git提交或push。', '', end, '']
    return '\n'.join(lines)


def append_handoff(addition):
    path = ROOT / 'handoff.md'
    start = b'<!-- GBSA_Diagnostic_20261019_START -->'
    end = b'<!-- GBSA_Diagnostic_20261019_END -->'
    encoded = addition.encode()
    try:
        with path.open('r+b', buffering=0) as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            old = f.read()
            if start in old:
                assert old.count(start) == old.count(end) == 1
                expected_block = encoded[encoded.index(start):encoded.index(end)+len(end)]
                assert old[old.index(start):old.index(end)+len(end)] == expected_block
                return {'status': 'already_appended_verified', 'after_sha256': hashlib.sha256(old).hexdigest()}
            f.seek(0, os.SEEK_END)
            assert f.write(encoded) == len(encoded)
            os.fsync(f.fileno())
            f.seek(0)
            new = f.read()
            assert new == old+encoded and new.count(start) == new.count(end) == 1
            return {'status': 'appended_verified', 'before_sha256': hashlib.sha256(old).hexdigest(),
                'after_sha256': hashlib.sha256(new).hexdigest(), 'prior_bytes_preserved': len(old),
                'addition_sha256': hashlib.sha256(encoded).hexdigest(),
                'method': 'exclusive existing-inode lock, single append, fsync, exact full-file readback'}
    except (OSError, AssertionError) as exc:
        atomic_bytes(BASE / 'handoff_pending.md', encoded)
        return {'status': 'pending', 'error': repr(exc), 'path': str(BASE / 'handoff_pending.md')}


def analyze():
    audit = json.loads((BASE / 'audit.json').read_text())
    protected_guard(audit)
    result = load(BASE / 'runs/diagnostics.bin')
    plan = load(BASE / 'runs/text_cache.bin')
    assert sha_file(BASE / 'runs/diagnostics.bin') == audit['diagnostics_sha256']
    assert sha_file(BASE / 'runs/text_cache.bin') == audit['text_cache_sha256']
    context, cache, splits, rows = read_inputs()
    acceptance = independent_acceptance(context, cache, splits, rows, plan, result)
    protected_guard(audit)
    acceptance['protected_files_SHA_unchanged'] = len(audit['protected_files'])
    audit['execution']['total_elapsed_seconds'] = time.time()-audit['started_epoch']
    summary = compact_summary(audit, result, acceptance)
    report = report_text(summary, audit)
    save_json(BASE / 'summary.json', plain(summary))
    atomic_bytes(BASE / 'report.md', report.encode())
    assert (BASE / 'report.md').read_text() == report
    summary['handoff'] = append_handoff(handoff_addition(summary))
    summary['report_sha256'] = sha_file(BASE / 'report.md')
    save_json(BASE / 'summary.json', plain(summary))
    audit.update({'completed_utc': utc(), 'acceptance': acceptance, 'handoff': summary['handoff'],
        'outcome': summary['outcome'], 'report_sha256': summary['report_sha256'], 'summary_sha256': sha_file(BASE / 'summary.json')})
    save_json(BASE / 'audit.json', audit)
    protected_guard(audit)
    assert sha_file(BASE / 'runs/diagnostics.bin') == audit['diagnostics_sha256']
    assert sha_file(BASE / 'runs/text_cache.bin') == audit['text_cache_sha256']
    log(f'ACCEPTED 10/10 frozen folds; Outcome {summary["outcome"]}; handoff={summary["handoff"]["status"]}')
    for setting, d in summary['C'].items():
        log(f'{setting}: R1={d["metrics"]["R1"]["mean"]:.8f}, random={d["random_baseline"]["R1"]:.8f}, cosine_gap={d["mean_cosine_advantage"]:+.8g}')
    log('Probe AUC: ' + str({g: d['AUC']['mean'] for g, d in summary['B']['groups'].items()}))


if __name__ == '__main__':
    analyze()
