"""Independent metric/OOF acceptance, site-cluster uncertainty and Chinese report."""
from __future__ import annotations

import collections
import concurrent.futures
import datetime as dt
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from data_protocol import (BASE, ROOT, CFG, array_sha, atomic_bytes, file_sha,
    load_artifact, save_json, source_fingerprints, task_definitions)
from probes import METRICS, metrics, per_patient_ce, rescue

NULL_GROUPS = ['P3', 'K2', 'N2a', 'N2b']
PERCENT = ['ACC', 'BA', 'AUC', 'SEN', 'SPE', 'F1']
GROUP_ORDER = ['P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8',
               'K0', 'K1', 'K2', 'N0', 'N1', 'N2a', 'N2b']


def nanmean(values):
    valid = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(valid)) if valid else None


def mean_metrics(rows):
    return {m: nanmean([r[m] for r in rows]) for m in METRICS}


def independent_metrics(y, p):
    target = np.asarray(y) == 0
    pred = np.asarray(p) >= .5
    tp, tn = np.sum(target & pred), np.sum(~target & ~pred)
    fp, fn = np.sum(~target & pred), np.sum(target & ~pred)
    sen = float(tp / (tp + fn)) if tp + fn else None
    spe = float(tn / (tn + fp)) if tn + fp else None
    pc = np.clip(p, np.finfo(np.float64).eps, 1 - np.finfo(np.float64).eps)
    return {'ACC': float((tp + tn) / len(y)), 'BA': (sen + spe) / 2 if sen is not None and spe is not None else None,
            'AUC': float(roc_auc_score(target, p)) if target.any() and (~target).any() else None,
            'SEN': sen, 'SPE': spe, 'F1': float(2 * tp / (2 * tp + fp + fn)) if target.any() else None,
            'CE': float(-np.where(target, np.log(pc), np.log1p(-pc)).mean()),
            'Brier': float(np.mean((p - target) ** 2))}


def group_keys(group):
    return [f'{group}_{i}' for i in range(8)] if group in NULL_GROUPS else [group]


def base_group(group):
    return 'K0' if group.startswith('K') else 'N0' if group.startswith('N') else 'P0'


def assemble(results, protocol, inputs):
    selected = [r for r in results if r['task']['protocol'] == protocol]
    seeds = CFG['seeds'] if protocol == 'R' else [20261019]
    keys = list(selected[0]['output']['probabilities'])
    oof, tables = {}, {}
    for seed in seeds:
        oof[seed] = {key: np.full(871, np.nan) for key in keys}
        cover = np.zeros(871, int)
        for r in selected:
            if r['task']['seed'] != seed:
                continue
            idx = r['task']['split']['test']
            cover[idx] += 1
            for key in keys:
                oof[seed][key][idx] = r['output']['probabilities'][key][idx]
        assert np.all(cover == 1)
        assert all(np.isfinite(v).all() for v in oof[seed].values())
    groups = [g for g in GROUP_ORDER if group_keys(g)[0] in keys]
    for group in groups:
        variants = group_keys(group)
        if protocol == 'R':
            per_seed = {}
            for seed in seeds:
                per_seed[str(seed)] = mean_metrics([r['output']['metrics']['test'][key]
                    for r in selected if r['task']['seed'] == seed for key in variants])
            avg = mean_metrics(list(per_seed.values()))
            sd = {m: float(np.std([v[m] for v in per_seed.values()], ddof=1)) if avg[m] is not None else None for m in METRICS}
            table = {'mean': avg, 'sample_SD': sd, 'per_seed_fold_mean': per_seed,
                     'per_seed_pooled_OOF': {str(seed): mean_metrics([metrics(inputs['y'], oof[seed][key]) for key in variants]) for seed in seeds},
                     'split_count': 30, 'test_subject_seed_predictions': 2613 * len(variants),
                     'variants': {key: {'per_seed': {str(seed): mean_metrics([r['output']['metrics']['test'][key]
                        for r in selected if r['task']['seed'] == seed]) for seed in seeds}} for key in variants}}
        else:
            pooled = mean_metrics([metrics(inputs['y'], oof[seeds[0]][key]) for key in variants])
            site_rows = {r['task']['site']: mean_metrics([r['output']['metrics']['test'][key] for key in variants]) for r in selected}
            table = {'pooled_OOF': pooled, 'valid_site_macro': mean_metrics(list(site_rows.values())), 'per_site': site_rows,
                     'valid_sites_per_metric': {m: sum(v[m] is not None for v in site_rows.values()) for m in METRICS},
                     'site_weighted_CE': float(sum(len(r['task']['split']['test']) * site_rows[r['task']['site']]['CE'] for r in selected) / 871),
                     'split_count': 20, 'test_subject_predictions': 871 * len(variants),
                     'variants': {key: {'pooled_OOF': metrics(inputs['y'], oof[seeds[0]][key]),
                                      'per_site': {r['task']['site']: r['output']['metrics']['test'][key] for r in selected}} for key in variants}}
            assert abs(table['site_weighted_CE'] - pooled['CE']) < 1e-14
        base = base_group(group)
        table['rescue_vs_' + base] = {str(seed): {
            'variants': {key: rescue(inputs['y'], oof[seed][base], oof[seed][key]) for key in variants},
            'mean_counts': {k: nanmean([rescue(inputs['y'], oof[seed][base], oof[seed][key])[k] for key in variants])
                           for k in rescue(inputs['y'], oof[seed][base], oof[seed][variants[0]])}}
            for seed in seeds}
        tables[group] = table
    return tables, oof


def paired_stats(inputs, oof, first, second):
    per_seed = {}
    for seed, probs in oof.items():
        akeys, bkeys = group_keys(first), group_keys(second)
        a = mean_metrics([metrics(inputs['y'], probs[k]) for k in akeys])
        b = mean_metrics([metrics(inputs['y'], probs[k]) for k in bkeys])
        diff = {m: a[m] - b[m] if a[m] is not None and b[m] is not None else None for m in METRICS}
        ce_a = np.mean([per_patient_ce(inputs['y'], probs[k]) for k in akeys], axis=0)
        ce_b = np.mean([per_patient_ce(inputs['y'], probs[k]) for k in bkeys], axis=0)
        site_ce = {s: float((ce_a - ce_b)[inputs['site'] == s].mean()) for s in sorted(set(inputs['site']))}
        per_seed[str(seed)] = {'OOF_delta': diff, 'site_CE_differences': site_ce,
                              'sites_CE_improved': sum(v < 0 for v in site_ce.values()),
                              'rescue': [rescue(inputs['y'], probs[bkeys[0]], probs[k]) for k in akeys]
                              if len(bkeys) == 1 else None,
                              'null_comparisons': [{m: metrics(inputs['y'], probs[akeys[0]])[m] - metrics(inputs['y'], probs[k])[m]
                                                   for m in METRICS} for k in bkeys] if len(bkeys) == 8 else None}
    return {'per_seed': per_seed, 'mean_OOF_delta': mean_metrics([r['OOF_delta'] for r in per_seed.values()])}


def bootstrap_metric_matrix(y, probabilities, site_codes, counts):
    """Vectorized exact weighted AUC, grouping ties; sites resampled as clusters."""
    target = (y == 0).astype(float)
    p = probabilities
    pred = p >= .5
    # Linear patient statistics can be summed to site aggregates once.
    def site_sums(value):
        return np.bincount(site_codes, weights=value, minlength=20)
    n = counts @ site_sums(np.ones(len(y)))
    pos = counts @ site_sums(target)
    neg = n - pos
    tp = counts @ site_sums(target * pred)
    tn = counts @ site_sums((1 - target) * ~pred)
    correct = tp + tn
    ce = counts @ site_sums(per_patient_ce(y, p))
    brier = counts @ site_sums((p - target) ** 2)
    # Aggregate equal probability scores before counting positive-negative pairs.
    order = np.argsort(p, kind='stable')
    _, inv = np.unique(p[order], return_inverse=True)
    ng = int(inv.max()) + 1
    group_pos = np.zeros((20, ng))
    group_neg = np.zeros((20, ng))
    np.add.at(group_pos, (site_codes[order], inv), target[order])
    np.add.at(group_neg, (site_codes[order], inv), 1 - target[order])
    wp, wn = counts @ group_pos, counts @ group_neg
    cum_neg = np.cumsum(wn, axis=1) - wn
    auc = np.sum(wp * (cum_neg + .5 * wn), axis=1) / (pos * neg)
    return {'ACC': correct / n, 'BA': .5 * (tp / pos + tn / neg), 'AUC': auc,
            'CE': ce / n, 'Brier': brier / n}


def bootstrap(inputs, oof, comparisons):
    sites, codes = np.unique(inputs['site'], return_inverse=True)
    rng = np.random.default_rng(CFG['bootstrap']['seed'])
    counts = rng.multinomial(len(sites), np.full(len(sites), 1 / len(sites)), size=CFG['bootstrap']['replicates'])
    needed = set()
    for a, b in comparisons:
        needed.update(group_keys(a))
        needed.update(group_keys(b))
    values = {}
    for seed, probs in oof.items():
        values[seed] = {}
        for key in sorted(needed):
            values[seed][key] = bootstrap_metric_matrix(inputs['y'], probs[key], codes, counts)
            # Independent weighted sklearn AUC checks on fixed representative resamples.
            for j in (0, 17, 1999):
                direct = roc_auc_score(inputs['y'] == 0, probs[key], sample_weight=counts[j, codes])
                assert abs(direct - values[seed][key]['AUC'][j]) < 1e-12
    result = {}
    for a, b in comparisons:
        metrics_out = {}
        for metric in ('ACC', 'BA', 'AUC', 'CE', 'Brier'):
            paired = []
            per_seed = {}
            for seed in oof:
                av = np.mean([values[seed][k][metric] for k in group_keys(a)], axis=0)
                bv = np.mean([values[seed][k][metric] for k in group_keys(b)], axis=0)
                paired.append(av - bv)
                per_seed[str(seed)] = np.percentile(av - bv, [2.5, 97.5]).tolist()
            delta = np.mean(paired, axis=0)
            metrics_out[metric] = {'CI95': np.percentile(delta, [2.5, 97.5]).tolist(),
                'seed_specific_CI95': per_seed, 'direction_positive_fraction': float(np.mean(delta > 0))}
        result[f'{a}-{b}'] = metrics_out
    return {'replicates': len(counts), 'cluster_count': 20, 'counts_sha256': array_sha(counts),
            'interpretation': 'Exploratory site-cluster percentile intervals; R seeds use identical patient/site resamples, not 2613 independent patients',
            'comparisons': result}


def stratify(inputs, oof):
    ages = inputs['age']
    groups = {'site': inputs['site'], 'sex': np.array([str(v) for v in inputs['sex']]),
              'age': np.where(ages < 12, '<12', np.where(ages < 18, '12-17', np.where(ages < 30, '18-29', '30+')))}
    result = {}
    for seed, probs in oof.items():
        seed_out = {}
        for a, b in [('P2', 'P0'), ('K1', 'K0'), ('N1', 'N0')]:
            cells = {}
            for typ, labels in groups.items():
                cells[typ] = {}
                for g in sorted(set(labels)):
                    mask = labels == g
                    r = rescue(inputs['y'][mask], probs[b][mask], probs[a][mask])
                    r['n'] = int(mask.sum())
                    r['base_confidence_mean'] = float(np.maximum(probs[b][mask], 1 - probs[b][mask]).mean())
                    r['base_metrics'] = metrics(inputs['y'][mask], probs[b][mask])
                    r['fusion_metrics'] = metrics(inputs['y'][mask], probs[a][mask])
                    cells[typ][g] = r
            seed_out[f'{a}-{b}'] = cells
        result[str(seed)] = seed_out
    return result


def independent_acceptance(inputs, results, audit):
    from run import validate_complete
    tasks = task_definitions(inputs)
    by_id = {r['task']['id']: r for r in results}
    assert len(by_id) == len(tasks) == 50
    max_error, metric_checks, fits = 0., 0, collections.Counter()
    for task in tasks:
        r = by_id[task['id']]
        validate_complete(r, inputs, task, audit['code_sha256'])
        tr = task['split']['train']
        assert np.array_equal(r['preprocessing']['ph_state']['site_encoder'].categories_[0], np.unique(inputs['site'][tr]))
        assert r['rfe']['fit_shape'] == (len(tr), 6670)
        assert r['rfe']['train_raw_features_sha256'] == array_sha(inputs['x'][tr])
        candidates = r['output']['neighbors']['candidates']
        assert np.all(candidates[tr] != tr[:, None])
        assert np.isin(candidates, tr).all()
        assert r['output']['neighbors']['candidates_sha256'] == array_sha(candidates)
        for part, idx in task['split'].items():
            if not len(idx):
                continue
            for key, p in r['output']['probabilities'].items():
                direct = independent_metrics(inputs['y'][idx], p[idx])
                saved = r['output']['metrics'][part][key]
                for m in METRICS:
                    if direct[m] is None:
                        assert saved[m] is None
                    else:
                        error = abs(direct[m] - saved[m])
                        max_error = max(max_error, error)
                        assert error < 1e-12
                        metric_checks += 1
        fits.update(r['output']['fit_counts'])
    assert fits['RFE'] == 50 and fits['LogisticRegression'] == 740 and fits['SVC'] == 500
    assert fits['GTUNet'] == fits['VAE'] == 0
    def check_protected(pair):
        p, record = pair
        assert file_sha(p) == record['sha256'] and Path(p).stat().st_size == record['bytes'], f'Protected file modified: {p}'
        return True
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        assert all(pool.map(check_protected, audit['protected'].items()))
    assert file_sha(BASE / 'inputs.bin') == audit['input']['local_cache_sha256']
    assert audit['code_sha256'] == source_fingerprints()[1]
    return {'passed': True, 'tasks': 50, 'R': 30, 'S': 20, 'RFE_fits': 50,
            'fit_counts': dict(fits), 'independent_metric_checks': metric_checks, 'max_metric_difference': max_error,
            'protected_files_SHA_unchanged': len(audit['protected']), 'raw_cache_SHA_unchanged': True,
            'all_train_ID_label_raw6670_hashes_checked': True, 'all_nulls_regenerated_exactly': True,
            'all_neighbors_train_only_no_self': True, 'complete_artifact_SHA_verified': True,
            'RFE_once_per_task_shared_by_all_groups': True, 'original_neural_model_training': 0}


def decide(summary):
    r, s = summary['R']['groups'], summary['S']['groups']
    # A conservative evidence rule fixed independently of finding the highest score.
    # This is an interpretation aid, not a significance or hyperparameter selector.
    evidence = {}
    for name, full, image, null in [('linear', 'P2', 'P0', 'P3'), ('nonlinear', 'K1', 'K0', 'K2'), ('neighbor', 'N1', 'N0', 'N2b')]:
        rd = {m: r[full]['mean'][m] - r[image]['mean'][m] for m in METRICS}
        sd = {m: s[full]['pooled_OOF'][m] - s[image]['pooled_OOF'][m] for m in METRICS}
        rn = {m: r[full]['mean'][m] - r[null]['mean'][m] for m in METRICS}
        sn = {m: s[full]['pooled_OOF'][m] - s[null]['pooled_OOF'][m] for m in METRICS}
        evidence[name] = {'R_delta_vs_image': rd, 'S_delta_vs_image': sd, 'R_delta_vs_null': rn, 'S_delta_vs_null': sn,
            'supported_direction_R': rd['AUC'] > 0 and rd['CE'] < 0 and rn['AUC'] > 0 and rn['CE'] < 0,
            'supported_direction_S': sd['AUC'] > 0 and sd['CE'] < 0 and sn['AUC'] > 0 and sn['CE'] < 0}
    linear, nonlinear, neighbor = [evidence[k] for k in ('linear', 'nonlinear', 'neighbor')]
    if not linear['supported_direction_R'] and nonlinear['supported_direction_R'] and nonlinear['supported_direction_S']:
        outcome = 'C'
        why = '线性没有共同支持而固定非线性模型在随机与留站点协议均有方向性增量；是否可靠仍取决于区间和Null完整分布。'
    elif (linear['supported_direction_R'] or nonlinear['supported_direction_R']) and not (linear['supported_direction_S'] or nonlinear['supported_direction_S']):
        outcome = 'B'
        why = '随机划分中出现支持性迹象，但未在站点外同时保留AUC/CE及真实相对Null优势。'
    elif (linear['supported_direction_R'] and linear['supported_direction_S']) or (nonlinear['supported_direction_R'] and nonlinear['supported_direction_S']):
        outcome = 'A'
        why = '至少一条固定分类路径在两个协议均保留方向性增量；仍需结合宽区间、站点异质性判断为探索性而非确证。'
    else:
        outcome = 'D'
        why = '没有分类路径同时建立相对纯影像及错配Null的稳定AUC/CE增量；局部单指标改善不构成可靠互补。'
    return {'outcome': outcome, 'reason': why, 'evidence': evidence,
            'priority_next_direction': '跨站点影像诊断泛化与采集混杂控制；暂缓当前年龄/性别复杂对齐或Gate' if outcome in ('B', 'D') else
                '无泄漏、跨站点的固定非线性疾病条件互补验证，先独立确认而非立刻堆叠对齐网络',
            'caution': '方向性规则不等于统计显著；20站点聚类区间为探索性，不能据此宣称独立多中心外部验证。'}


def fmt(value, digits=2, percent=False):
    if value is None:
        return 'NA'
    return f'{value * (100 if percent else 1):.{digits}f}'


def result_table(groups, protocol):
    rows = ['|组别|ACC %|BA %|prob-AUC %|SEN-ASD %|SPE-HC %|F1-ASD %|CE|Brier|',
            '|---|---|---|---|---|---|---|---|---|']
    for group, result in groups.items():
        if protocol == 'R':
            cells = [f'{fmt(result["mean"][m], 2 if m in PERCENT else 6, m in PERCENT)} +/- {fmt(result["sample_SD"][m], 2 if m in PERCENT else 6, m in PERCENT)}' for m in METRICS]
        else:
            cells = [fmt(result['pooled_OOF'][m], 2 if m in PERCENT else 6, m in PERCENT) for m in METRICS]
        rows.append('|' + '|'.join([group] + cells) + '|')
    return rows


def build_report(summary, audit):
    R, S = summary['R'], summary['S']
    lines = ['# 实验10：CMCV跨模态诊断互补性严格验证', '',
        '## 一、问题与假设', '',
        '在复杂对齐之前，分别检验统计关联、影像之外的诊断增量、真实患者对应及站点外泛化。本轮不因单一最高ACC判成功，不新增Gate、对齐Loss或神经网络。全部固定方法与8份Null完整运行，未使用外层测试选择参数或阈值。', '',
        '## 二、论文与真实源码审计', '',
        'RFE将影像压至500维，VAE将站点/性别/年龄升至500维，首先是维度适配而非显式患者语义对齐。AMRS联合影像相似度与表型奖惩构图；双GTUNet各输出16维，ACMGL融合两模态及算术均值，再经16→256/ReLU/BN→2分类。论文Table 3的无VAE、MLP、AE、VAE之ABIDE ACC依次为67.06、80.86、76.02、82.92%；Table 5表型、影像、联合ACC为52.00、76.49、82.92%，AUC为50.28、85.22、88.21%。完整框架消融不能隔离正确表型的新增诊断信息。[论文](https://arxiv.org/html/2406.14455v3)', '',
        '真实源码人口图为fused_sim*rp_graph，Tensor相似度用欧氏距离；RP在站点循环内部提前return，存在未注册权重、tensor重建截断梯度及.data前向投影。VAE在eval仍采样，GTUNet的functional Dropout在eval仍随机。原训练x=feature_selection(x,...)跨折覆盖，首折6670→500，后续500→500保留此前监督选择，导致历史跨折污染。上述历史文件全部只读，本轮不实例化原网络。', '',
        '实验8局部关系目标发生变化但未确认稳定诊断收益；实验9原生图修改L1约0.007891%、测试概率MAD约4.09e-6，放大及固定邻居也未共同证明新增信息；其实验邻居使用已经接受L3表型对齐的投影。本轮邻居改用独立训练集RFE空间，不使用已对齐投影。原论文也指出转导预测需处理新增测试数据的局限。[官方源码](https://github.com/NZWANG/MM-GTUNets)、[当前仓库](https://github.com/tangaii/MM)。', '',
        '## 三、协议与预处理', '',
        'H：只读背景C0 ACC81.88±1.39、BA81.68±1.27、prob-AUC88.54±0.31%；L3 ACC82.15±1.53、BA81.94±1.46、AUC88.54±0.42%。与新探针的模型、RFE、转导图和随机性均不同，不作绝对方法排名。', '',
        'R：保留历史3seed×10折精确成员及独立验证集；标准化、缺失填补、RFE和分类器全部只拟合当前train。验证仅记录，不搜索参数。S：20站点逐一留出，其他19站点训练，没有站点编码输入；全871人恰好一次站点外OOF。', '',
        '原始FC871×6670与每人AAL116矩阵经Fisher arctanh的上三角逐值一致。ASD=0共403，HC=1共468。每任务RFE(RidgeClassifier(),500,step=100)在原始未标准化训练FC拟合一次，再训练StandardScaler并除sqrt(500)。年龄训练中位数填补、训练均值/SD标准化及缺失指示；性别训练OHE+未知指示；完整表型除训练RMS范数，所有消融及Null保留相同槽位和倍率。站点OHE不作连续编码，未见站点安全全零。', '',
        '8份联合错配均在分区×站点内以私有SHA种子随机排序后非零循环错位；组大小≥2不自配，单例保留，完全不用诊断标签。P3/K2/N2a/N2b共用映射；每份Null的实际变化率、单例、源索引和SHA存于任务产物。Null主成绩是8个独立拟合结果的指标均值，不是先平均概率，不当作8个独立数据集。', '',
        '## 四、各组定义与完整结果', '',
        '|组别|输入/做法|控制与目的|', '|---|---|---|']
    for g in GROUP_ORDER:
        lines.append(f'|{g}|{CFG["groups"][g]}|固定预注册参数、同任务共享RFE；' + ('仅R站点混杂控制' if g in ('P6', 'P7', 'P8') else 'R30与S20完整') + '|')
    lines += ['', 'LR固定C=1/L2/lbfgs/max_iter=2000/tol=1e-6；RBF-SVM固定C=1/gamma=.5/probability=True，同任务私有校准seed一致，libsvm仅训练集内部概率校准。最终均以ASD连续概率及0.5阈值评价。', '',
        '邻居固定Top64训练锚点、训练查询排除自身，cos/.15 softmax；年龄高斯核带宽为训练非自身年龄差中位数、最低2岁，性别同/异/未知权重1/.5/.75；表型核/.20 softmax，与影像权重相乘再归一化投票。N2a仅训练锚点错配，N2b查询/锚点在各自分区内错配；不校准投票分数。', '',
        '### Protocol R：每seed十折均值，再三个seed均值±样本SD', '']
    lines += result_table(R['groups'], 'R')
    lines += ['', '每个正常组30折、2613条subject-seed外层预测；每Null组8×30次评价。重复seed不是三组独立患者。完整每fold/validation/train、每seed、每Null、患者概率和算法指纹见runs与summary。', '',
        '### Protocol S：871人合并LOSO OOF', '']
    lines += result_table(S['groups'], 'S')
    lines += ['', '### Protocol S：有效站点宏平均', '', '|组别|ACC %|BA %|AUC %|CE|Brier|', '|---|---|---|---|---|---|']
    for g, r in S['groups'].items():
        v = r['valid_site_macro']
        lines.append(f'|{g}|{fmt(v["ACC"],percent=True)}|{fmt(v["BA"],percent=True)}|{fmt(v["AUC"],percent=True)}|{fmt(v["CE"],6)}|{fmt(v["Brier"],6)}|')
    lines += ['', '单类别站点AUC/BA为NA，缺失类别召回或ASD-F1也为NA；保留其CE/Brier与OOF。宏平均只计该指标有效站点，不填0、不删除患者。', '',
        '## 五、配对改善、纠错与Null', '']
    for protocol in ('R', 'S'):
        lines += [f'### Protocol {protocol}配对', '', '|比较 前者−后者|ΔACC pp|ΔBA pp|ΔAUC pp|ΔCE|ΔBrier|', '|---|---|---|---|---|---|']
        for pair, stats in summary[protocol]['paired'].items():
            d = stats['mean_OOF_delta']
            lines.append(f'|{pair}|{fmt(d["ACC"],3,True)}|{fmt(d["BA"],3,True)}|{fmt(d["AUC"],3,True)}|{fmt(d["CE"],6)}|{fmt(d["Brier"],6)}|')
        lines += ['', '这里配对差由完整OOF计算，可能与随机协议“先均值十折”的主表差略有不同，两者均保留。CE/Brier负值更好；ACC/BA/AUC正值更好。', '',
                  '|主比较|seed|rescued|harmed|NetRescue|ΔCE|', '|---|---|---|---|---|---|']
        for a, b in [('P2', 'P0'), ('K1', 'K0'), ('N1', 'N0')]:
            for seed, result in summary[protocol]['groups'][a]['rescue_vs_' + b].items():
                r = result['mean_counts']
                lines.append(f'|{a}-{b}|{seed}|{r["rescued"]:.0f}|{r["harmed"]:.0f}|{r["NetRescue"]:.0f}|{r["CE_difference"]:+.6f}|')
        lines += ['', '纠错/误伤是在所有模型冻结预测之后的诊断性统计，未用于训练或学习Gate。按年龄段、性别、站点及原影像置信度的完整统计见summary.strata。', '',
                  '|比较|ΔAUC pp 探索性95%CI|ΔCE 探索性95%CI|', '|---|---|---|']
        for pair, row in summary[protocol]['bootstrap']['comparisons'].items():
            au, ce = row['AUC']['CI95'], row['CE']['CI95']
            lines.append(f'|{pair}|[{au[0]*100:+.3f}, {au[1]*100:+.3f}]|[{ce[0]:+.6f}, {ce[1]:+.6f}]|')
        lines += ['', '2000次20站点聚类Bootstrap；R的三个seed共用同一站点重采样、平均seed统计，不把2613行当独立病例。区间探索性且站点数量有限，不代表独立多中心外部验证或多子组校正后的显著性。', '',
                  '|真实/Null|Null8 ΔAUC范围 pp（真实−各Null）|Null8 ΔCE范围|', '|---|---|---|']
        for a, b in [('P2', 'P3'), ('K1', 'K2'), ('N1', 'N2a'), ('N1', 'N2b')]:
            vs = summary[protocol]['paired'][f'{a}-{b}']['per_seed']
            # Keep each null identity paired across seeds, never choose a favorable null.
            au = [np.mean([v['null_comparisons'][i]['AUC'] for v in vs.values()]) for i in range(8)]
            ce = [np.mean([v['null_comparisons'][i]['CE'] for v in vs.values()]) for i in range(8)]
            lines.append(f'|{a}/{b}|[{min(au)*100:+.3f}, {max(au)*100:+.3f}]|[{min(ce):+.6f}, {max(ce):+.6f}]|')
    lines += ['', '## 六、20站点外泛化与混杂', '',
              '|站点|n|ASD/HC|P0 AUC|P2 AUC|P2−P0 CE|K0 AUC|K1 AUC|K1−K0 CE|N1−N0 CE|',
              '|---|---|---|---|---|---|---|---|---|']
    for site, row in summary['sites'].items():
        v = row['metrics']
        lines.append(f'|{site}|{row["n"]}|{row["ASD"]}/{row["HC"]}|{fmt(v["P0"]["AUC"],percent=True)}|{fmt(v["P2"]["AUC"],percent=True)}|{v["P2"]["CE"]-v["P0"]["CE"]:+.6f}|{fmt(v["K0"]["AUC"],percent=True)}|{fmt(v["K1"]["AUC"],percent=True)}|{v["K1"]["CE"]-v["K0"]["CE"]:+.6f}|{v["N1"]["CE"]-v["N0"]["CE"]:+.6f}|')
    lines += ['', '完整20站点×全部适用组×8指标及8份Null见summary.S.groups。P8站点单模态、P6−P0和P7−P6分别衡量站点混杂及其之外的年龄/性别贡献；站点编码带来的提升不视为疾病生物学证据。', '',
              '## 七、科学判断', '', f'**Outcome {summary["decision"]["outcome"]}**：{summary["decision"]["reason"]}', '']
    for path, ev in summary['decision']['evidence'].items():
        r, s, rn, sn = ev['R_delta_vs_image'], ev['S_delta_vs_image'], ev['R_delta_vs_null'], ev['S_delta_vs_null']
        lines.append(f'- {path}：相对影像R ΔAUC={r["AUC"]*100:+.3f}pp、ΔCE={r["CE"]:+.6f}；S ΔAUC={s["AUC"]*100:+.3f}pp、ΔCE={s["CE"]:+.6f}。相对Null R ΔAUC={rn["AUC"]*100:+.3f}pp、ΔCE={rn["CE"]:+.6f}；S ΔAUC={sn["AUC"]*100:+.3f}pp、ΔCE={sn["CE"]:+.6f}。')
    lines += ['', '统计相关不等于诊断增量；不因P1弱就推断融合无效，也不因单个ACC上升就认可互补。本轮固定参数阴性只能约束当前年龄/性别、样本和三条预注册探针，不能证明信息论意义上完全不存在互补。若邻居AUC/CE方向不同，需区分排序与未经校准的投票概率。', '',
              '## 八、下一步唯一优先方向', '', summary['decision']['priority_next_direction'] + '。理由是需要先解决当前证据中跨站点稳定性及真实/错误对应的区分，不能用更复杂模块掩盖尚未确认的诊断信息。', '',
              '## 九、运行与审计', '',
              f'全部R30/30、S20/20；逻辑队列任务数{summary["execution"]["lane_task_counts"]}。实际正式计算为单线程CPU sklearn/NumPy，PPU仅兼容性/矩阵速度实测，不声称双卡训练50任务。每卡benchmark数据见audit.matrix_benchmark。', '',
              f'RFE={summary["acceptance"]["fit_counts"]["RFE"]}；LR={summary["acceptance"]["fit_counts"]["LogisticRegression"]}；SVC={summary["acceptance"]["fit_counts"]["SVC"]}（含各自libsvm训练内部概率校准）；GTUNet=0、VAE=0、MM-GTUNets=0。', '',
              f'墙钟{summary["execution"]["wall_seconds"]:.2f}秒；累计任务{summary["execution"]["sum_task_seconds"]:.2f}秒；最高任务RSS={summary["execution"]["peak_task_RSS_bytes"]/2**30:.3f}GiB。四个完整技术预检复用于正式结果，重入跳过不重训且文件SHA不变；50个RFE先保存再拟合探针。失败记录{len(audit["failures"])}，详见audit与run.log。', '',
              f'独立指标核验{summary["acceptance"]["independent_metric_checks"]}项，最大差{summary["acceptance"]["max_metric_difference"]:.3g}；受保护历史文件{summary["acceptance"]["protected_files_SHA_unchanged"]}项SHA保持；全部训练ID、标签、raw6670哈希、映射、训练锚点、自身排除和OOF覆盖均自动验证。', '',
              f'源码/配置组合SHA256：`{audit["code_sha256"]}`。根模型、历史结果和数据未修改。输入为通过871原MAT逐值验证的必要本地缓存；任务产物内置内容SHA，单文件原子阶段保存与锁排他。未进行Git提交或push。', '',
              '全部新产物：`/tmp/CMCV_20261019`；本地临时存储并非持久备份，待原存储恢复后可整体迁回。handoff追加状态见summary.handoff；若FUSE无法安全追加，待追加原文保存在本目录，不伪报更新。', '']
    return '\n'.join(lines)


def append_handoff(summary, audit):
    start_marker = '<!-- CMCV_20261019_START -->'
    end_marker = '<!-- CMCV_20261019_END -->'
    lines = ['', '', start_marker, '', '## 实验10：CMCV跨模态诊断互补性严格验证', '',
        '用户因/data I/O故障授权在/tmp完成；历史1–9全部只读。只训练固定LR和RBF-SVM，训练MM-GTUNets/GTUNet/VAE次数均0。', '',
        'H为历史背景不作绝对排名；R保留30份历史train/val/test但独立raw6670训练RFE；S为20站点LOSO。每任务一次RFE共享所有P/K/N。8份Null在分区×站点内部联合年龄/性别错配，N2a仅锚点、N2b查询及锚点错配。', '',
        'P0纯影像、P1纯年龄/性别、P2真实融合、P3错配融合、P4年龄、P5性别、P6影像+站点、P7全输入、P8站点。K0/K1/K2为固定RBF-SVM。N0/N1/N2a/N2b为未对齐RFE空间的固定Top64投票。', '',
        '### 无泄漏随机十折：三seed的折均值±样本SD', '']
    lines += result_table(summary['R']['groups'], 'R')
    lines += ['', '### 20站点LOSO：871人合并OOF', '']
    lines += result_table(summary['S']['groups'], 'S')
    lines += ['', f'结论Outcome {summary["decision"]["outcome"]}：{summary["decision"]["reason"]}', '',
              'R与S的全配对差、各seed/各Null结果、20站点异质性、纠错/误伤及2000次站点聚类Bootstrap区间完整见report与summary。三个seed不是独立队列，Null8不当作8个独立数据集；预测错误只用于事后分析，没有训练Gate。', '',
              '唯一优先方向：' + summary['decision']['priority_next_direction'] + '。不以单个最高ACC宣称成功，本轮结果不等同原神经网络无泄漏临床成绩。', '',
              f'验收R30/30、S20/20；逻辑队列{summary["execution"]["lane_task_counts"]}，实际探针为CPU，不虚报GPU任务；双PPU仅数值/速度实测。RFE50、LR740、SVC500、神经网络训练0。墙钟{summary["execution"]["wall_seconds"]:.2f}秒。历史SHA{summary["acceptance"]["protected_files_SHA_unchanged"]}项及OOF完整性自动验收通过。', '',
              f'产物`/tmp/CMCV_20261019/report.md`、`summary.json`、`audit.json`及`runs`；代码SHA `{audit["code_sha256"]}`。全部50任务及分析完成；未自动Git提交或push。临时存储待原存储恢复后迁回。', '', end_marker, '']
    addition = '\n'.join(lines).encode()
    pending = BASE / 'handoff_pending.md'
    atomic_bytes(pending, addition)
    path = ROOT / 'handoff.md'
    # Lock the existing handoff inode; do not create a root-side lock or temp file.
    try:
        with open(path, 'r+b', buffering=0) as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            old = f.read()
            if start_marker.encode() in old:
                assert end_marker.encode() in old, 'Incomplete prior CMCV handoff marker'
                return {'status': 'already_appended', 'sha256': hashlib.sha256(old).hexdigest()}
            assert hashlib.sha256(old).hexdigest() == audit['handoff_before_sha256'], 'Handoff changed concurrently; pending preserved'
            f.seek(0, os.SEEK_END)
            n = f.write(addition)
            assert n == len(addition)
            os.fsync(f.fileno())
            f.seek(0)
            new = f.read()
            assert new == old + addition
            assert new.count(start_marker.encode()) == new.count(end_marker.encode()) == 1
            result = {'status': 'appended_verified', 'before_sha256': hashlib.sha256(old).hexdigest(),
                      'after_sha256': hashlib.sha256(new).hexdigest(), 'appended_bytes': len(addition),
                      'safe_method': 'locked single append write + fsync + exact full-file verification; prior bytes preserved'}
        pending.unlink()
        return result
    except (OSError, AssertionError) as exc:
        return {'status': 'pending_due_to_IO_or_concurrency', 'error': repr(exc), 'pending_path': str(pending),
                'warning': 'No successful update claimed; original handoff preserved when append could not start'}


def complete():
    from run import log
    started = time.time()
    inputs = load_artifact(BASE / 'inputs.bin')
    audit = json.loads((BASE / 'audit.json').read_text())
    tasks = task_definitions(inputs)
    results = [load_artifact(BASE / 'runs' / (t['id'] + '.bin')) for t in tasks]
    acceptance = independent_acceptance(inputs, results, audit)
    log(f'INDEPENDENT_ACCEPTANCE passed {acceptance}')
    r, ro = assemble(results, 'R', inputs)
    s, so = assemble(results, 'S', inputs)
    comparisons = [('P2', 'P0'), ('K1', 'K0'), ('N1', 'N0'), ('P2', 'P3'), ('K1', 'K2'), ('N1', 'N2a'), ('N1', 'N2b')]
    R_pairs = comparisons + [('P4', 'P0'), ('P5', 'P0'), ('P6', 'P0'), ('P7', 'P6'), ('P7', 'P2'), ('P7', 'P0')]
    S_pairs = comparisons + [('P4', 'P0'), ('P5', 'P0')]
    summary = {'experiment': CFG['experiment'], 'status': 'complete_accepted', 'configuration': CFG,
        'historical_reference_only': {'C0': {'ACC': [81.88, 1.39], 'BA': [81.68, 1.27], 'AUC': [88.54, .31]},
                                      'L3': {'ACC': [82.15, 1.53], 'BA': [81.94, 1.46], 'AUC': [88.54, .42]},
                                      'not_directly_comparable': True},
        'R': {'groups': r, 'paired': {f'{a}-{b}': paired_stats(inputs, ro, a, b) for a, b in R_pairs}},
        'S': {'groups': s, 'paired': {f'{a}-{b}': paired_stats(inputs, so, a, b) for a, b in S_pairs}},
        'strata': {'R': stratify(inputs, ro), 'S': stratify(inputs, so)}, 'acceptance': acceptance}
    log('BOOTSTRAP start 2000 site-cluster replicates, all pre-registered main comparisons')
    summary['R']['bootstrap'] = bootstrap(inputs, ro, comparisons)
    summary['S']['bootstrap'] = bootstrap(inputs, so, comparisons)
    sites = {}
    for site in sorted(set(inputs['site'])):
        mask = inputs['site'] == site
        sites[site] = {'n': int(mask.sum()), 'ASD': int((inputs['y'][mask] == 0).sum()),
                       'HC': int((inputs['y'][mask] == 1).sum()),
                       'metrics': {g: v['per_site'][site] for g, v in s.items()}}
    summary['sites'] = sites
    null_changed = {}
    for proto in ('R', 'S'):
        null_changed[proto] = {}
        for part in ('train', 'val', 'test'):
            records = [rec['parts'][part] for r in results if r['task']['protocol'] == proto for rec in r['null_info'] if rec['parts'][part]['n']]
            if records:
                null_changed[proto][part] = {'changed_fraction_range': [min(v['changed_fraction'] for v in records), max(v['changed_fraction'] for v in records)],
                    'singleton_subjects_range': [min(v['singletons'] for v in records), max(v['singletons'] for v in records)],
                    'n_records': len(records)}
    summary['null_effectiveness'] = null_changed
    summary['decision'] = decide(summary)
    summary['execution'] = {'lane_task_counts': {str(i): sum(r['task']['lane'] == i for r in results) for i in (0, 1)},
        'lane_R_S_counts': {str(i): {p: sum(r['task']['lane'] == i and r['task']['protocol'] == p for r in results) for p in ('R', 'S')} for i in (0, 1)},
        'actual_PPU_formal_tasks': {'0': 0, '1': 0}, 'actual_compute': 'CPU sklearn/NumPy single-thread workers; two PPUs benchmarked only',
        'sum_task_seconds': sum(r['seconds'] for r in results), 'wall_seconds': time.time() - audit['started_epoch'],
        'peak_task_RSS_bytes': max(r['resources']['peak_RSS_bytes'] for r in results),
        'per_task': {r['task']['id']: {'seconds': r['seconds'], 'RFE_seconds': r['rfe']['fit_seconds'],
                    'lane': r['task']['lane'], 'fit_counts': r['output']['fit_counts']} for r in results},
        'analysis_seconds': time.time() - started, 'training_failures': audit['failures'], 'neural_training': 0}
    summary['OOF'] = {'R_each_seed': 871, 'R_per_group': 2613, 'S_per_group': 871,
                      'stored_at': 'Each runs/<task>.bin output.probabilities and task.split.test, subject order in inputs.bin',
                      'patient_coverage_exactly_once_checked': True}
    summary['handoff'] = {'status': 'not_yet_appended'}
    save_json(BASE / 'summary.json', summary)
    atomic_bytes(BASE / 'report.md', build_report(summary, audit).encode())
    # Re-read necessary outputs before the only permitted root-file write.
    assert json.loads((BASE / 'summary.json').read_text())['acceptance']['passed']
    assert (BASE / 'report.md').stat().st_size > 1000
    summary['handoff'] = append_handoff(summary, audit)
    summary['execution']['wall_seconds'] = time.time() - audit['started_epoch']
    save_json(BASE / 'summary.json', summary)
    atomic_bytes(BASE / 'report.md', build_report(summary, audit).encode())
    audit['final_acceptance'] = acceptance
    audit['completed_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    audit['handoff'] = summary['handoff']
    save_json(BASE / 'audit.json', audit)
    log(f'FINAL outcome={summary["decision"]["outcome"]}; handoff={summary["handoff"]}; wall={summary["execution"]["wall_seconds"]:.2f}')
    print('\n'.join(result_table(r, 'R')), flush=True)
    print('\n'.join(result_table(s, 'S')), flush=True)


if __name__ == '__main__':
    complete()
