"""Independent OOF validation, paired fold summaries, and final PCFRA report."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os

import numpy as np
import torch

import run

METRICS = ('ACC', 'SEN', 'SPE', 'AUC')
PAIRS = (('J2', 'J3'), ('J4', 'J3'), ('J4', 'J1'),
         ('J4', 'historical_G2'), ('J4', 'historical_G3'),
         ('J4', 'historical_D4'), ('J1', 'historical_G2'),
         ('J2', 'historical_G3'))


def percent_metrics(metrics):
    return {k: float(metrics[k] * 100) for k in METRICS}


def summarize(rows):
    return {metric: {'mean': float(np.mean([r[metric] for r in rows])),
                     'sample_sd': float(np.std([r[metric] for r in rows], ddof=1))}
            for metric in METRICS}


def write_csv(path, fields, rows):
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with tmp.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def inspect_results(audit, require_complete):
    audit = run.guard(audit)
    ctx = run.load_torch(run.OLD / 'artifacts/context.pt')
    groups = {group: [] for group in run.GROUPS}
    fold_rows, oof_rows = [], []
    result_hashes = {}
    gpu_counts = {'0': 0, '1': 0}
    for group in run.GROUPS:
        coverage = np.zeros(871, dtype=np.int64)
        observed_ids = []
        for fold in range(10):
            value = run.valid_result(group, 911, fold, audit, required=require_complete)
            if value is None:
                continue
            split = run.load_torch(run.anchor(911, fold) / 'splits.pt')
            indices = value['oof']['indices']
            y = value['oof']['labels']
            p = value['oof']['probability']
            pred = value['oof']['prediction']
            logits = np.asarray(value['oof']['logits'])
            ids = value['oof']['ids']
            assert np.array_equal(indices, split['test'])
            assert np.array_equal(y, ctx['labels'][indices])
            assert ids == [ctx['ids'][int(i)] for i in indices]
            assert np.array_equal(pred, logits.argmax(axis=1))
            p_check = torch.softmax(torch.as_tensor(logits, dtype=torch.float64), dim=1)[:, 0].numpy()
            assert np.allclose(p, p_check, atol=1e-7, rtol=1e-6)
            assert np.isfinite(p).all() and np.all((p >= 0.0) & (p <= 1.0))
            measured = run.binary_metrics(y, p, pred)
            assert max(abs(measured[k] - value['metrics'][k]) for k in METRICS) < 1e-12
            assert value['bn_best_batches'] == value['best_epoch'] + 1
            assert value['physical_gpu'] == fold % 2
            assert value['formal_start_rng_sha256'] == audit['anchors'][str(fold)]['pairing']['classifier_start_rng']
            assert value['forward_calls'] == {'train': value['epochs_run'], 'val': value['epochs_run'], 'test': 1}
            assert value['new_trainable_parameters'] == value['rfe_fit_count'] == value['vae_train_count'] == 0
            assert value['formal_test_grad_enabled'] is True and value['test_forwards'] == 1
            assert value['gradient_target'] == 'joint_embed'
            assert value['best_val_loss'] == min(e['val_loss'] for e in value['epochs'])
            assert value['best_epoch'] == min(range(len(value['epochs'])),
                                               key=lambda i: value['epochs'][i]['val_loss'])
            for epoch in value['epochs']:
                assert np.isfinite(epoch['lambda_eff']) and 0.0 <= epoch['lambda_eff'] <= 0.5
                assert epoch['objective']['prototype_used'] == (group != 'J1')
                if epoch['epoch'] >= 9:
                    assert epoch['aux_joint_grad_norm'] > 0
                if group in ('J2', 'J3', 'J4'):
                    assert all(abs(v - 1.0) <= 1e-5
                               for v in epoch['objective']['site_margin_weighted_mean'].values())
            coverage[indices] += 1
            observed_ids.extend(ids)
            groups[group].append(value)
            fold_metrics = percent_metrics(measured)
            fold_rows.append({'group': group, 'seed': 911, 'fold': fold, **fold_metrics})
            for i, subject_id in enumerate(ids):
                oof_rows.append({'group': group, 'seed': 911, 'fold': fold, 'subject_id': subject_id,
                                 'y_true': int(y[i]), 'y_pred': int(pred[i]),
                                 'probability_asd': float(p[i])})
            result_hashes[f'{group}/911/{fold}'] = run.sha_file(run.result_path(group, 911, fold))
            gpu_counts[str(value['physical_gpu'])] += 1
        if groups[group]:
            assert np.all(coverage == 1)
            assert len(observed_ids) == len(set(observed_ids)) == 871
        if require_complete:
            assert len(groups[group]) == 10

    completed = sum(len(rows) for rows in groups.values())
    if require_complete:
        assert completed == 40 and gpu_counts == {'0': 20, '1': 20}
        assert audit['execution']['queue']['completed_unique_tasks'] == 40
        assert audit['execution']['queue']['gpu_completed'] == gpu_counts
        assert len(result_hashes) == 40
        task_names = {(x['group'], x['seed'], x['fold']) for x in audit['manifest']['tasks']}
        assert task_names == {(g, 911, f) for g in run.GROUPS for f in range(10)}
        assert not any(g in {'D0', 'D4', 'G1', 'G2', 'G3', 'G4', 'H1', 'H2', 'H3', 'H4'}
                       for g, _, _ in task_names)

    hist_rows, hist_mean, hist_sd, hist_protected = run.historical_d4(ctx)
    assert hist_protected == audit['historical_d4_readonly']['checkpoint_sha256']
    hist_by_fold = {r['fold']: r for r in hist_rows}
    hist_g23, hist_g23_summary, hist_g23_hashes = run.historical_g2_g3(ctx)
    assert hist_g23_hashes == audit['historical_g2_g3_readonly']['checkpoint_sha256']
    hist_by_group_fold = {g: {r['fold']: r for r in rows} for g, rows in hist_g23.items()}
    group_summary = {k: summarize([percent_metrics(v['metrics']) for v in values])
                     for k, values in groups.items() if values}
    group_summary['historical_D4'] = {
        key: {'mean': hist_mean[key], 'sample_sd': hist_sd[key]} for key in METRICS
    }
    for group in ('G2', 'G3'):
        group_summary[f'historical_{group}'] = {
            key: {'mean': hist_g23_summary[group]['mean_percent'][key],
                  'sample_sd': hist_g23_summary[group]['sample_sd_percent'][key]} for key in METRICS
        }

    pair_summary = {}
    for left, right in PAIRS:
        left_values = groups.get(left, [])
        right_values = groups.get(right, [])
        if left == 'historical_D4':
            left_values = hist_by_fold.values()
        elif left.startswith('historical_'):
            left_values = hist_by_group_fold[left.removeprefix('historical_')].values()
        if right == 'historical_D4':
            right_values = hist_by_fold.values()
        elif right.startswith('historical_'):
            right_values = hist_by_group_fold[right.removeprefix('historical_')].values()
        if len(left_values) != 10 or len(right_values) != 10:
            continue
        left_by_fold = (hist_by_fold if left == 'historical_D4' else
                        hist_by_group_fold[left.removeprefix('historical_')] if left.startswith('historical_')
                        else {v['fold']: v for v in groups[left]})
        right_by_fold = (hist_by_fold if right == 'historical_D4'
                         else hist_by_group_fold[right.removeprefix('historical_')] if right.startswith('historical_')
                         else {v['fold']: v for v in groups[right]})
        values = {}
        leave_one_fold_out = {}
        for metric in METRICS:
            diff = np.asarray([(left_by_fold[f]['metrics'][metric] - right_by_fold[f]['metrics'][metric]) * 100
                               for f in range(10)])
            values[metric] = {'mean_pp': float(diff.mean()), 'sample_sd_pp': float(diff.std(ddof=1)),
                              'improved_folds': int(np.sum(diff > 1e-12)),
                              'ties_folds': int(np.sum(np.abs(diff) <= 1e-12)),
                              'fold_differences_pp': diff.tolist()}
            leave_one_fold_out[metric] = {
                str(drop): float(np.delete(diff, drop).mean()) for drop in range(10)
            }
        pair_summary[f'{left}-{right}'] = {
            'metrics': values, 'leave_one_fold_out_mean_pp': leave_one_fold_out,
            'paired_by_fold': True,
        }

    mechanism = {}
    preflight_grad_ok = bool(audit.get('preflight', {}).get('911/0', {}).get('per_site_loss_gradients')) and all(
        x['requires_grad'] and x['logit_grad_norm'] > 0
        for group_data in audit.get('preflight', {}).get('911/0', {}).get('per_site_loss_gradients', {}).values()
        for x in group_data.values())
    for group in ('J1', 'J2', 'J3', 'J4'):
        values = groups.get(group, [])
        if not values:
            continue
        active = [e for v in values for e in v['epochs'] if e['epoch'] >= 9]
        mechanism[group] = {
            'conditional_pairs_by_fold': {str(v['fold']): v['site_objective_static']['n_conditional_pairs']
                                          for v in values},
            'conditional_asd_anchor_fraction_by_fold': {
                str(v['fold']): v['site_objective_static']['conditional_asd_anchor_fraction'] for v in values},
            'conditional_td_anchor_fraction_by_fold': {
                str(v['fold']): v['site_objective_static']['conditional_td_anchor_fraction'] for v in values},
            'lambda_mean': float(np.mean([e['lambda_eff'] for e in active])),
            'weighted_joint_grad_ratio_mean': float(np.mean([e['weighted_joint_grad_ratio'] for e in active])),
            'aux_joint_grad_norm_min': float(np.min([e['aux_joint_grad_norm'] for e in active])),
            'site_weight_shares_by_fold': {str(v['fold']): v['site_objective_static']['site_weight_shares']
                                           for v in values},
            'original_anchor_weight_shares_by_fold': {
                str(v['fold']): v['site_objective_static']['original_anchor_weight_shares'] for v in values},
            'site_weight_change_vs_anchor_by_fold': {
                str(v['fold']): v['site_objective_static']['site_weight_change_vs_anchor'] for v in values},
            'site_conditional_contribution_shares_by_fold': {
                str(v['fold']): v['epochs'][-1]['objective']['site_conditional_contribution_shares']
                for v in values},
            'site_conditional_contribution_mean_after_warmup_by_fold': {
                str(v['fold']): {
                    site: float(np.mean([e['objective']['site_conditional_contribution_shares'].get(site, 0.0)
                                         for e in v['epochs'] if e['epoch'] >= 9]))
                    for site in v['site_objective_static']['site_counts']
                } for v in values
            },
            'site_counts_by_fold': {str(v['fold']): v['site_objective_static']['site_counts']
                                    for v in values},
            'nyu_weight_share_by_fold': {str(v['fold']): v['site_objective_static']['nyu_condition_weight_share']
                                         for v in values},
            'per_site_loss_has_gradient': preflight_grad_ok and bool(values) and all(
                bool(v['site_objective_static']['effective_sites']) for v in values),
        }
        if group in ('J2', 'J3', 'J4'):
            mechanism[group].update({
                'margin_abs_deviation_mean': float(np.mean([e['objective']['margin_abs_deviation'] for e in active])),
                'margin_std_mean': float(np.mean([e['objective']['margin_std'] for e in active])),
                'temperature_mean': float(np.mean([e['objective']['temperature'] for e in active])),
                'teacher_evidence_sd_mean': float(np.mean([
                    e['objective']['teacher_evidence_std'] for e in active])),
                'margin_min_by_fold': {str(v['fold']): min(e['objective']['margin_min'] for e in v['epochs'])
                                       for v in values},
                'margin_max_by_fold': {str(v['fold']): max(e['objective']['margin_max'] for e in v['epochs'])
                                       for v in values},
                'site_center_mu_by_fold_epoch': {str(v['fold']): [e['objective']['site_center_mu'] for e in v['epochs']]
                                                  for v in values},
                'site_center_mu_abs_mean_after_warmup': float(np.mean([
                    abs(mu) for v in values for e in v['epochs'] if e['epoch'] >= 9
                    for mu in e['objective']['site_center_mu'].values()])),
                'site_center_mu_nonzero_fraction_after_warmup': float(np.mean([
                    abs(mu) > 1e-8 for v in values for e in v['epochs'] if e['epoch'] >= 9
                    for mu in e['objective']['site_center_mu'].values()])),
                'site_margin_weighted_mean_by_fold': {str(v['fold']): v['epochs'][-1]['objective']['site_margin_weighted_mean']
                                                       for v in values},
            })
    checkpoint_by_fold = {
        group: {str(v['fold']): {'best_epoch': v['best_epoch'], 'best_val_loss': v['best_val_loss'],
                                 'epochs_run': v['epochs_run'], 'training_seconds': v['training_seconds'],
                                 'best_core_sha256': v['best_core_sha256']}
                for v in values}
        for group, values in groups.items() if values
    }
    checkpoint_by_fold['historical_D4'] = {
        str(r['fold']): {'best_epoch': r['best_epoch'], 'best_val_loss': r['best_val_loss'],
                         'epochs_run': r['epochs_run'], 'training_seconds': r['training_seconds'],
                         'best_core_sha256': r['best_core_sha256']}
        for r in hist_rows
    }
    for group, values in hist_g23.items():
        checkpoint_by_fold[f'historical_{group}'] = {
            str(r['fold']): {'best_epoch': r['best_epoch'], 'best_val_loss': r['best_val_loss'],
                             'epochs_run': r['epochs_run'], 'training_seconds': r['training_seconds'],
                             'best_core_sha256': r['best_core_sha256']}
            for r in values
        }
    return {'groups': groups, 'fold_rows': fold_rows, 'oof_rows': oof_rows,
            'summary': group_summary, 'pairs': pair_summary, 'mechanism': mechanism,
            'result_hashes': result_hashes, 'gpu_counts': gpu_counts, 'completed_tasks': completed,
            'checkpoint_by_fold': checkpoint_by_fold,
            'historical_rows': hist_rows, 'historical_mean': hist_mean, 'historical_sd': hist_sd,
            'historical_g2_g3_rows': hist_g23, 'historical_g2_g3_summary': hist_g23_summary}


def research_judgment(data):
    if data['completed_tasks'] != 40 or len(data['groups'].get('J4', [])) != 10:
        return 'PARTIAL', {'all_tasks_complete': False}
    pairs = data['pairs']
    j4_d4 = pairs['J4-historical_D4']['metrics']
    robust_auc_vs_g2 = all(v > 0 for v in
                           pairs['J4-historical_G2']['leave_one_fold_out_mean_pp']['AUC'].values())
    robust_auc_vs_g3 = all(v > 0 for v in
                           pairs['J4-historical_G3']['leave_one_fold_out_mean_pp']['AUC'].values())
    checks = {
        'j4_auc_above_historical_g2': pairs['J4-historical_G2']['metrics']['AUC']['mean_pp'] > 0,
        'j4_auc_above_historical_g3': pairs['J4-historical_G3']['metrics']['AUC']['mean_pp'] > 0,
        'j4_auc_above_historical_d4': j4_d4['AUC']['mean_pp'] > 0,
        'j4_acc_not_below_d4': j4_d4['ACC']['mean_pp'] >= 0,
        'j4_spe_not_substantially_below_d4': j4_d4['SPE']['mean_pp'] >= -0.5,
        'j4_sen_not_substantially_below_g3': pairs['J4-historical_G3']['metrics']['SEN']['mean_pp'] >= -0.5,
        'j4_has_gain_vs_j3': any(pairs['J4-J3']['metrics'][m]['mean_pp'] > 0 for m in METRICS),
        'j4_auc_above_j1': pairs['J4-J1']['metrics']['AUC']['mean_pp'] > 0,
        'positive_auc_not_single_fold': (pairs['J4-historical_G2']['metrics']['AUC']['improved_folds'] > 1 and
                                         robust_auc_vs_g2 and robust_auc_vs_g3),
        'site_balancing_active': data['mechanism']['J4']['per_site_loss_has_gradient'],
        'centered_margin_valid': all(abs(v - 1.0) <= 1e-5 for fold in
                                     data['mechanism']['J4']['site_margin_weighted_mean_by_fold'].values()
                                     for v in fold.values()),
        'protocol_passed': True,
    }
    if all(checks.values()):
        return 'GO', checks
    positive_mechanism = checks['site_balancing_active'] and checks['centered_margin_valid'] and checks['protocol_passed']
    if positive_mechanism:
        return 'MIXED', checks
    return 'NO-GO', checks


def render_report(data, decision, checks, audit):
    metrics = data['summary']
    method_names = {
        'historical_D4': '历史年龄×性别原型辅助 CE（只读）',
        'historical_G2': '实验16 G2（只读）',
        'historical_G3': '实验16 G3（只读）',
        'J1': 'G2 + sqrt(n) 站点均衡条件聚合',
        'J2': 'G3 + 0.5 站点内中心化 margin',
        'J3': 'G3 + 0.25 温和中心化 margin',
        'J4': 'J3 + sqrt(n) 站点均衡条件聚合',
    }
    execution = audit.get('execution', {})
    queue = execution.get('queue', {})
    completed_values = [v for values in data['groups'].values() for v in values]
    worker_seconds = sum(v['training_seconds'] for v in completed_values)
    wall_seconds = queue.get('wall_seconds')
    if wall_seconds is None and completed_values:
        starts = [datetime.fromisoformat(v['started_utc']) for v in completed_values]
        ends = [datetime.fromisoformat(v['completed_utc']) for v in completed_values]
        wall_seconds = (max(ends) - min(starts)).total_seconds()
    wall_seconds = float(wall_seconds or 0.0)
    failure_count = len(execution.get('failures', []))
    retry_count = sum(1 for x in execution.get('failures', [])
                      if tuple(x.get('task', [])) in {tuple(e.get('task', [])) for e in execution.get('events', [])})
    status = 'COMPLETE_WITH_DEVIATION' if failure_count else 'COMPLETE'

    lines = [
        '# PCFRA-E18 正式实验报告', '', f'STATUS={status}', '',
        '## 设计与执行', '',
        '正式训练仅含 J1、J2、J3、J4，seed 911、fold 0–9，共 40 项。历史 D0、D4、实验16 G2/G3 未重训；D4/G2/G3 的 OOF 和检查点仅从只读 artifact 独立复算。每项新训练从 DRPA 对应 fold 的 train_start core、相同 split/RFE/VAE 和训练起始 RNG 恢复。',
        '',
        f"任务 {data['completed_tasks']}/40；GPU0 {data['gpu_counts'].get('0', 0)}/20，GPU1 {data['gpu_counts'].get('1', 0)}/20；正式队列墙钟 {wall_seconds / 3600:.2f} 小时，worker 累计 {worker_seconds / 3600:.2f} 小时；失败尝试 {failure_count}，重试 {retry_count}。",
        '',
        'J1 相对 G2 只替换条件项聚合：先分别计算每个有效站点内的双向 anchor 均值，再以 sqrt(n_site) 占比汇总。J2/J3 相对 G3 使用相同的历史 D4 年龄×性别条件原型 stop-gradient 证据，对同站点 W 加权 tanh 证据作中心化和有界归一化；Margin 系数分别为 0.5、0.25。J4 相对 J3 只替换条件项聚合为 J1 的站点均衡方式。',
        '',
        '四组均采用 L_aux = 0.5 L_global + 0.5 L_conditional；辅助梯度在 joint_embed 上按 5% cap-aware 预算缩放，lambda 上限 0.5，epoch 0–9 线性 warmup。best checkpoint 只按 validation CE+原生 Graph 严格下降选择；测试使用训练结束 RNG、eval 模式、grad-enabled 状态，且仅一次原生 forward。',
        '',
        'J1–J4 使用共同起点、split 和训练规则，构成本轮直接消融。跨运行历史比较仍受随机训练轨迹影响；单 seed 十折不能视为多 seed 或独立外部验证。',
        '',
        '## 四指标结果', '',
        '每折先由 OOF logits/probability 独立计算指标，再报告十折 mean ± sample SD（ddof=1），单位为百分比。SEN 正类为 ASD，SPE 正类为 TD，AUC 使用连续 ASD 概率。',
        '',
        '| 组别 | 方法 | ACC (%) | SEN-ASD (%) | SPE-TD (%) | probability AUC (%) |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for group in ('historical_D4', 'historical_G2', 'historical_G3', 'J1', 'J2', 'J3', 'J4'):
        if group not in metrics:
            continue
        row = [f"{metrics[group][m]['mean']:.4f} ± {metrics[group][m]['sample_sd']:.4f}" for m in METRICS]
        lines.append(f"| {group} | {method_names[group]} | " + ' | '.join(row) + ' |')
    lines += ['', '历史 D4、G2、G3 均为 READ_ONLY，均值和 sample SD 由历史 artifact 复算。跨运行比较仅属探索性参考。', '',
              '## 配对差值', '',
              '差值为左组减右组、按相同 fold 配对，单位百分点。改善折数仅为方向描述，不是独立样本显著性检验。',
              '',
              '| 比较 | ΔACC | ΔSEN-ASD | ΔSPE-TD | Δprobability AUC | AUC 改善折数 |',
              '|---|---:|---:|---:|---:|---:|']
    for pair in ('J2-J3', 'J4-J3', 'J4-J1', 'J4-historical_G2',
                 'J4-historical_G3', 'J4-historical_D4', 'J1-historical_G2', 'J2-historical_G3'):
        if pair not in data['pairs']:
            continue
        values = data['pairs'][pair]['metrics']
        diffs = [f"{values[m]['mean_pp']:+.4f} ± {values[m]['sample_sd_pp']:.4f}" for m in METRICS]
        lines.append(f"| {pair} | " + ' | '.join(diffs) + f" | {values['AUC']['improved_folds']}/10 |")
    lines += ['', '### 十折逐折 AUC 差值', '',
              '| 比较 | Fold0 | Fold1 | Fold2 | Fold3 | Fold4 | Fold5 | Fold6 | Fold7 | Fold8 | Fold9 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for pair in ('J2-J3', 'J4-J3', 'J4-J1', 'J4-historical_G2',
                 'J4-historical_G3', 'J4-historical_D4', 'J1-historical_G2', 'J2-historical_G3'):
        if pair in data['pairs']:
            diffs = data['pairs'][pair]['metrics']['AUC']['fold_differences_pp']
            lines.append(f"| {pair} | " + ' | '.join(f'{x:+.3f}' for x in diffs) + ' |')
    lines += ['', '### Leave-one-fold-out AUC', '',
              '每次移除一个 fold 后重新计算其余九折的平均配对差值。summary.json 保存全部八个比较、四项指标和十种删除情形；下表展示完整范围及重点移除 Fold 2/6/9。',
              '', '| 比较 | 全折 ΔAUC | LOO 范围 | 去 Fold2 | 去 Fold6 | 去 Fold9 |',
              '|---|---:|---:|---:|---:|---:|']
    for pair in ('J2-J3', 'J4-J3', 'J4-J1', 'J4-historical_G2',
                 'J4-historical_G3', 'J4-historical_D4', 'J1-historical_G2', 'J2-historical_G3'):
        if pair not in data['pairs']:
            continue
        item = data['pairs'][pair]
        loo = item['leave_one_fold_out_mean_pp']['AUC']
        loo_values = list(loo.values())
        full = item['metrics']['AUC']['mean_pp']
        lines.append(f"| {pair} | {full:+.4f} | {min(loo_values):+.4f} to {max(loo_values):+.4f} | "
                     f"{loo['2']:+.4f} | {loo['6']:+.4f} | {loo['9']:+.4f} |")
    lines += ['', '## Fold 2/6/9 检查点', '',
              'best_epoch、best validation loss 和 best-core SHA 前缀用于展示跨组/历史检查点轨迹差异；这些历史对照来自只读结果，不能解释为同批次随机控制。完整十折检查点记录位于 summary.json。',
              '', '| Fold | Group | best_epoch | epochs_run | best_val_loss | core SHA256 前缀 |',
              '|---:|---|---:|---:|---:|---|']
    for fold in (2, 6, 9):
        for group in ('historical_D4', 'historical_G2', 'historical_G3', 'J1', 'J2', 'J3', 'J4'):
            value = data['checkpoint_by_fold'].get(group, {}).get(str(fold))
            if value is None:
                continue
            lines.append(f"| {fold} | {group} | {value['best_epoch']} | {value['epochs_run']} | "
                         f"{value['best_val_loss']:.6f} | {value['best_core_sha256'][:12]} |")
    lines += ['', '## 机制审计', '',
              '',
              '| 组别 | 有效条件配对数（逐折） | 热身后平均 lambda | 加权 joint_embed 梯度比例 | 原型教师 |',
              '|---|---|---:|---:|---|']
    for group in ('J1', 'J2', 'J3', 'J4'):
        info = data['mechanism'][group]
        pairs = ', '.join(f"F{f}:{n}" for f, n in info['conditional_pairs_by_fold'].items())
        teacher = (f"margin |M−1| 均值 {info['margin_abs_deviation_mean']:.4f}；"
                   f"margin SD {info['margin_std_mean']:.4f}；温度 {info['temperature_mean']:.4f}；"
                   f"站点 |μ| {info['site_center_mu_abs_mean_after_warmup']:.4f}"
                   if group in ('J2', 'J3', 'J4') else '未使用')
        lines.append(f"| {group} | {pairs} | {info['lambda_mean']:.5f} | {info['weighted_joint_grad_ratio_mean']:.5f} | {teacher} |")
    lines += ['', 'J2/J3/J4 的站点 W 加权 margin 均值须为 1，J2 margin 范围为 [0.5, 1.5]，J3/J4 为 [0.75, 1.25]。单次共享输入预检验证 J2/J3 的 D、T、C 相同，J3/J4 Margin 相同。辅助目标只使用 outer-train 标签/协变量， heldout 扰动不改变目标；原型教师为 stop-gradient。',
              '', '| 组别 | 站点 | 旧 anchor 权重 | sqrt(n) 权重 | 权重变化 | 训练期条件损失贡献 |',
              '|---|---|---:|---:|---:|---:|']
    for group in ('J1', 'J4'):
        info = data['mechanism'][group]
        sites = sorted(info['site_counts_by_fold']['0'],
                       key=lambda s: -np.mean([v.get(s, {}).get('n', 0)
                                               for v in info['site_counts_by_fold'].values()]))
        for site in sites[:8]:
            old = np.mean([x.get(site, 0.0) for x in info['original_anchor_weight_shares_by_fold'].values()])
            new = np.mean([x.get(site, 0.0) for x in info['site_weight_shares_by_fold'].values()])
            actual = np.mean([x.get(site, 0.0)
                              for x in info['site_conditional_contribution_mean_after_warmup_by_fold'].values()])
            lines.append(f'| {group} | {site} | {old:.4f} | {new:.4f} | {new - old:+.4f} | {actual:.4f} |')
    lines += ['', 'sqrt(n) 权重与旧 anchor 方式的理论有效贡献在静态上分别记录；训练期贡献还受各站点当前损失影响，表中两者分开呈现。若最大站点的归一化权重下降，只能说明目标权重下降，不代表站点偏差已消除。各折全部站点、NYU 权重和实际贡献见 summary.json。',
              '', '## 研究问题回答', '',
              f"- J1 相对历史 G2：SPE {data['pairs']['J1-historical_G2']['metrics']['SPE']['mean_pp']:+.4f} pp，AUC {data['pairs']['J1-historical_G2']['metrics']['AUC']['mean_pp']:+.4f} pp；站点聚合是否有效由非零站点梯度与权重变化诊断确认。",
              f"- J2 相对历史 G3 的 SEN 差值为 {data['pairs']['J2-historical_G3']['metrics']['SEN']['mean_pp']:+.4f} pp；J2/J3 中心化的逐站点统计和 Margin 范围见机制审计。",
              f"- J3 相对 J2：SPE {-data['pairs']['J2-J3']['metrics']['SPE']['mean_pp']:+.4f} pp，AUC {-data['pairs']['J2-J3']['metrics']['AUC']['mean_pp']:+.4f} pp；SEN {-data['pairs']['J2-J3']['metrics']['SEN']['mean_pp']:+.4f} pp。",
              f"- J4 相对 J3 的 ACC/SEN/SPE/AUC 依次为 " + ', '.join(
                  f"{data['pairs']['J4-J3']['metrics'][m]['mean_pp']:+.4f} pp" for m in METRICS) + '。',
              '- 实现上 J3/J4 共用原型证据与 Margin 构造，仅条件项聚合不同；共享输入预检验证 Margin 张量逐元素一致。',
              '- AUC 是否由个别折主导以逐折差值和 LOO 范围呈现；删除 Fold 2/6/9 的结果见上表。',
              '- 新增可训练参数为 0；每 epoch 一次训练和一次验证 forward，结束时一次测试 forward；heldout 标签/协变量扰动的预检结果写入 audit.json。',
              '',
              '## 预注册判断', '',
              f'**{decision}**', '', '判定检查：']
    labels = {
        'j4_auc_above_historical_g2': 'J4 AUC 高于历史 G2',
        'j4_auc_above_historical_g3': 'J4 AUC 高于历史 G3',
        'j4_auc_above_historical_d4': 'J4 AUC 高于历史 D4',
        'j4_acc_not_below_d4': 'J4 ACC 不低于历史 D4',
        'j4_spe_not_substantially_below_d4': 'J4 SPE 未明显低于历史 D4',
        'j4_sen_not_substantially_below_g3': 'J4 SEN 未明显低于历史 G3',
        'j4_has_gain_vs_j3': 'J4 相对 J3 至少一项主指标提升',
        'j4_auc_above_j1': 'J4 AUC 高于 J1',
        'positive_auc_not_single_fold': '相对 G2/G3 的 AUC 正收益在逐折及全部 LOO 检查中不由单折决定',
        'site_balancing_active': '站点内损失具非零梯度',
        'centered_margin_valid': '站点内 Margin 加权均值为 1',
        'protocol_passed': '训练、checkpoint 和测试协议通过',
    }
    for key, passed in checks.items():
        lines.append(f"- [{'x' if passed else ' '}] {labels[key]}")
    lines += ['', 'GO 仅表示建议进入预注册三 seed 确认实验，不能证明真实表型条件疾病原型的独立有效性，也不能直接满足投稿证据要求。单 seed 十折不是十个独立样本。MIXED 表示机制执行有效但候选性能条件未全部满足；NO-GO 表示当前方案不建议扩展。',
              '',
              '## 执行边界与局限', '',
              '- 正式任务：40/40；GPU0：20/20；GPU1：20/20。没有额外 baseline，也未重训 D0/D4/G1-G4/H1-H4。',
              '- 每组 OOF 恰好覆盖 871 人各一次；指标由保存 logits/probability 独立复算。所有 fold 使用对应相同 train_start core/RNG、split、RFE500 和 VAE SHA。',
              '- RFE 历史跨折监督预处理、871 节点转导式人口图、随机 VAE/eval 行为和 PPU 长轨迹数值波动均保留。seed911 十折不是十个独立队列，也不能证明未见站点泛化。',
              '- MICCAI 2027 判断：当前结果只能决定是否将局部改进列为候选，需三 seed 确认和独立队列/站点验证后才能评估创新有效性。',
              '',
              f"PCFRA-E18 冻结训练代码/config SHA256：{audit['code_sha256']}。",
              f"Prompt SHA256：{audit['prompt_sha256']}。",
              f"源码修改审计分析器 SHA256：{run.sha_file(run.BASE / 'analyze.py')}。",
              f"历史 D4/G2/G3 检查点校验数：{len(audit['historical_d4_readonly']['checkpoint_sha256'])}/10、"
              f"{sum(k.startswith(str(run.ROOT / run.CFG['historical_g2_g3'])) for k in audit['historical_g2_g3_readonly']['checkpoint_sha256'])} 个 G2/G3 artifact。",
              '产物：/data/adodas/MM-GTUNets/experiments/PCFRA_E18_20261021。']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--final', action='store_true')
    args = parser.parse_args()
    audit = run.guard()
    data = inspect_results(audit, require_complete=args.final)
    write_csv(run.BASE / 'fold_metrics.csv',
              ['group', 'seed', 'fold', 'ACC', 'SEN', 'SPE', 'AUC'], data['fold_rows'])
    write_csv(run.BASE / 'oof.csv',
              ['group', 'seed', 'fold', 'subject_id', 'y_true', 'y_pred', 'probability_asd'],
              data['oof_rows'])
    decision, checks = research_judgment(data)
    failures = audit.get('execution', {}).get('failures', [])
    status = ('PARTIAL' if data['completed_tasks'] != 40 else
              'COMPLETE_WITH_DEVIATION' if failures else 'COMPLETE')
    execution_values = [v for values in data['groups'].values() for v in values]
    formal_wall_seconds = audit.get('execution', {}).get('queue', {}).get('wall_seconds')
    if formal_wall_seconds is None and execution_values:
        starts = [datetime.fromisoformat(v['started_utc']) for v in execution_values]
        ends = [datetime.fromisoformat(v['completed_utc']) for v in execution_values]
        formal_wall_seconds = (max(ends) - min(starts)).total_seconds()
    summary = {
        'status': status,
        'experiment': run.CFG['experiment'], 'formal_tasks': data['completed_tasks'],
        'gpu_completed': data['gpu_counts'], 'metrics_percent_mean_sample_sd': data['summary'],
        'paired_fold_differences_pp': data['pairs'], 'mechanism': data['mechanism'],
        'checkpoint_by_fold': data['checkpoint_by_fold'],
        'formal_wall_seconds': formal_wall_seconds,
        'cumulative_worker_seconds': sum(v['training_seconds'] for v in execution_values),
        'failed_attempts': len(failures), 'retry_count': len(failures),
        'pre_manifest_code_revisions': audit.get('code_revisions', []),
        'historical_g2_g3_readonly': audit['historical_g2_g3_readonly'],
        'historical_d4_readonly': audit['historical_d4_readonly'],
        'result_sha256': data['result_hashes'], 'research_decision': decision,
        'decision_checks': checks,
        'preflight': audit['preflight']['911/0'] if audit.get('preflight') else None,
        'protocol_limitations': run.CFG['historical_protocol_limitations'],
        'handoff': {'status': 'pending append after final audit',
                    'original_prefix_sha256': audit['handoff_before_sha256'],
                    'original_prefix_bytes': audit['handoff_before_bytes']},
    }
    run.atomic_json(summary, run.BASE / 'summary.json')
    if args.final:
        if data['completed_tasks'] != 40:
            raise RuntimeError('Refusing STATUS=COMPLETE with incomplete formal tasks')
        report = render_report(data, decision, checks, audit)
        run.atomic_bytes(report.encode(), run.BASE / 'report.md')
    print(json.dumps({'status': summary['status'], 'formal_tasks': summary['formal_tasks'],
                      'gpu_completed': summary['gpu_completed'], 'decision': decision},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
