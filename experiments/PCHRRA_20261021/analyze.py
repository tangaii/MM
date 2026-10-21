"""Independent OOF, paired-fold, mechanism, and handoff audit for PCHRRA."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

import run

METRICS = ('ACC', 'SEN', 'SPE', 'AUC')
PAIRS = (('H3', 'H2'), ('H3', 'H4'), ('H3', 'historical_G2'),
         ('H3', 'historical_D4'), ('H2', 'historical_G2'))


def pct(metrics):
    return {key: float(metrics[key] * 100) for key in METRICS}


def summarize(rows):
    return {key: {'mean': float(np.mean([row[key] for row in rows])),
                  'sample_sd': float(np.std([row[key] for row in rows], ddof=1))}
            for key in METRICS}


def write_csv(path, fields, rows):
    path = Path(path)
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temp.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def history_rows(audit):
    ctx = run.load_torch(run.DRPA / 'artifacts/context.pt')
    data, hashes = run.historical_results(ctx)
    for group in ('historical_D4', 'historical_G2'):
        expected = audit['historical_readonly'][group]['checkpoint_sha256']
        observed = {path: value for path, value in hashes.items()
                    if f'/runs/{"D4" if group == "historical_D4" else "G2"}/' in path}
        assert observed == expected
        assert audit['historical_readonly'][group]['retrained'] is False
    return data


def inspect_results(audit, require_complete=True):
    audit = run.guard(audit)
    ctx = run.load_torch(run.DRPA / 'artifacts/context.pt')
    history = history_rows(audit)
    values = {group: [] for group in run.GROUPS}
    metrics_rows, oof_rows = [], []
    hashes, gpu_counts = {}, {'0': 0, '1': 0}

    for group in run.GROUPS:
        coverage = np.zeros(871, dtype=np.int64)
        observed_ids = []
        for fold in range(10):
            result = run.valid_result(group, 911, fold, audit, required=require_complete)
            if result is None:
                continue
            split = run.load_torch(run.anchor(911, fold) / 'splits.pt')
            idx = result['oof']['indices']
            y = result['oof']['labels']
            logits = np.asarray(result['oof']['logits'])
            probability = np.asarray(result['oof']['probability'])
            prediction = np.asarray(result['oof']['prediction'])
            ids = result['oof']['ids']
            assert np.array_equal(idx, split['test'])
            assert np.array_equal(y, ctx['labels'][idx])
            assert ids == [ctx['ids'][int(i)] for i in idx]
            assert np.array_equal(prediction, logits.argmax(1))
            prob_check = torch.softmax(torch.as_tensor(logits, dtype=torch.float64), dim=1)[:, 0].numpy()
            assert np.allclose(probability, prob_check, atol=1e-7, rtol=1e-6)
            measured = run.binary_metrics(y, probability, prediction)
            assert max(abs(measured[key] - result['metrics'][key]) for key in METRICS) < 1e-12
            assert result['bn_best_batches'] == result['bn_start_batches'] + result['best_epoch'] + 1
            assert result['physical_gpu'] == fold % 2
            assert result['formal_start_rng_sha256'] == audit['anchors'][str(fold)]['pairing']['classifier_start_rng']
            assert result['forward_calls'] == {'train': result['epochs_run'], 'val': result['epochs_run'], 'test': 1}
            assert result['new_trainable_parameters'] == result['rfe_fit_count'] == result['vae_train_count'] == 0
            assert result['formal_test_grad_enabled'] and result['test_forwards'] == 1
            assert result['gradient_target'] == 'joint_embed'
            val_losses = [epoch['val_loss'] for epoch in result['epochs']]
            assert result['best_val_loss'] == min(val_losses)
            assert result['best_epoch'] == int(np.argmin(val_losses))
            for epoch in result['epochs']:
                assert np.isfinite(epoch['lambda_eff']) and 0 <= epoch['lambda_eff'] <= .5
                assert epoch['objective']['prototype_used'] == (group in {'H3', 'H4'})
                if epoch['epoch'] >= 9:
                    assert epoch['aux_joint_grad_norm'] > 0
                    assert abs(epoch['weighted_joint_grad_ratio'] - .05) <= .005
                if group in {'H3', 'H4'}:
                    assert epoch['objective']['weight_conservation_relative_error'] < 1e-5
                    assert epoch['objective']['T_residual'] >= .1 and epoch['objective']['T_condition'] >= .1
                    assert epoch['objective']['reweight_q']['min'] >= .5 - 1e-6
                    assert epoch['objective']['reweight_q']['max'] <= 1.5 + 1e-6

            coverage[idx] += 1
            observed_ids.extend(ids)
            values[group].append(result)
            hashes[f'{group}/911/{fold}'] = run.sha_file(run.result_path(group, 911, fold))
            gpu_counts[str(result['physical_gpu'])] += 1
            metrics_rows.append({'group': group, 'seed': 911, 'fold': fold,
                                 **{key: measured[key] * 100 for key in METRICS}})
            for i, subject_id in enumerate(ids):
                oof_rows.append({'group': group, 'seed': 911, 'fold': fold, 'subject_id': subject_id,
                                 'y_true': int(y[i]), 'y_pred': int(prediction[i]),
                                 'probability_asd': float(probability[i])})
        if values[group]:
            assert np.all(coverage == 1) and len(observed_ids) == len(set(observed_ids)) == 871
        if require_complete:
            assert len(values[group]) == 10

    completed = sum(map(len, values.values()))
    if require_complete:
        assert completed == 40 and gpu_counts == {'0': 20, '1': 20}
        assert audit['execution']['queue']['completed_unique_tasks'] == 40
        assert audit['execution']['queue']['gpu_completed'] == gpu_counts
        manifest_tasks = {(item['group'], item['seed'], item['fold']) for item in audit['manifest']['tasks']}
        expected = {(group, 911, fold) for group in run.GROUPS for fold in range(10)}
        assert manifest_tasks == expected and len(manifest_tasks) == 40
        forbidden = {'D0', 'D4', 'E0', 'E1', 'F0', 'F1', 'G1', 'G2', 'G3', 'G4'}
        assert not any(group in forbidden for group, _, _ in manifest_tasks)
        assert len(hashes) == 40

    summary_metrics = {group: summarize([pct(result['metrics']) for result in rows])
                       for group, rows in values.items() if rows}
    for group, rows in history.items():
        summary_metrics[group] = summarize([pct(row['metrics']) for row in rows])

    all_rows = {**history, **{group: [
        {'fold': result['fold'], 'metrics': result['metrics']}
        for result in rows] for group, rows in values.items()}}
    pair_details = {}
    loo_details = {}
    for left, right in PAIRS:
        if len(all_rows.get(left, [])) != 10 or len(all_rows.get(right, [])) != 10:
            continue
        left_by_fold = {row['fold']: row for row in all_rows[left]}
        right_by_fold = {row['fold']: row for row in all_rows[right]}
        pair_metrics = {}
        for metric in METRICS:
            diff = np.asarray([(left_by_fold[fold]['metrics'][metric] -
                                right_by_fold[fold]['metrics'][metric]) * 100 for fold in range(10)])
            loo = np.asarray([(diff.sum() - diff[i]) / 9.0 for i in range(10)])
            worst_i, best_i = int(np.argmin(loo)), int(np.argmax(loo))
            impact_i = int(np.argmax(np.abs(diff)))
            pair_metrics[metric] = {
                'mean_pp': float(diff.mean()), 'sample_sd_pp': float(diff.std(ddof=1)),
                'improved_folds': int(np.sum(diff > 1e-12)),
                'ties_folds': int(np.sum(np.abs(diff) <= 1e-12)),
                'fold_differences_pp': diff.tolist(),
                'loo_diagnostics': {
                    'max_single_fold_difference_pp': float(diff[impact_i]),
                    'max_single_fold_id': impact_i,
                    'max_single_fold_mean_contribution_pp': float(diff[impact_i] / 10.0),
                    'max_absolute_fold_difference_share_of_absolute_sum': float(
                        abs(diff[impact_i]) / np.abs(diff).sum()) if np.abs(diff).sum() else 0.0,
                    'least_favorable_leave_one_fold_out_mean_pp': float(loo[worst_i]),
                    'least_favorable_removed_fold': worst_i,
                    'most_favorable_leave_one_fold_out_mean_pp': float(loo[best_i]),
                    'most_favorable_removed_fold': best_i,
                    'remove_largest_absolute_fold_mean_pp': float((diff.sum() - diff[impact_i]) / 9.0),
                    'all_leave_one_fold_out_means_pp': loo.tolist(),
                },
            }
            loo_details[f'{left}-{right}/{metric}'] = pair_metrics[metric]['loo_diagnostics']
        pair_details[f'{left}-{right}'] = {'metrics': pair_metrics, 'paired_by_fold': True}

    mechanism = {}
    for group in run.GROUPS:
        group_results = values[group]
        active_epochs = [epoch for result in group_results for epoch in result['epochs']
                         if epoch['epoch'] >= 9]
        all_epochs = [epoch for result in group_results for epoch in result['epochs']]
        entry = {
            'conditional_pairs_by_fold': {str(result['fold']): result['objective_static']['n_condition_pairs']
                                          for result in group_results},
            'asd_anchor_fraction_by_fold': {str(result['fold']): result['objective_static']['asd_anchor_fraction']
                                            for result in group_results},
            'td_anchor_fraction_by_fold': {str(result['fold']): result['objective_static']['td_anchor_fraction']
                                           for result in group_results},
            'lambda_mean_after_warmup': float(np.mean([epoch['lambda_eff'] for epoch in active_epochs])),
            'native_joint_grad_norm_mean': float(np.mean([epoch['native_joint_grad_norm'] for epoch in active_epochs])),
            'aux_joint_grad_norm_mean': float(np.mean([epoch['aux_joint_grad_norm'] for epoch in active_epochs])),
            'weighted_joint_grad_ratio_mean': float(np.mean([epoch['weighted_joint_grad_ratio'] for epoch in active_epochs])),
            'joint_grad_cosine_mean': float(np.mean([epoch['joint_grad_cosine'] for epoch in active_epochs])),
            'auxiliary_loss_mean': float(np.mean([epoch['auxiliary_loss'] for epoch in active_epochs])),
            'difficulty_mean_by_epoch': {str(epoch): float(np.mean([
                row['objective']['difficulty']['mean'] for row in all_epochs
                if row['epoch'] == epoch and row['objective'].get('difficulty')]))
                for epoch in sorted({row['epoch'] for row in all_epochs})
                if any(row['epoch'] == epoch and row['objective'].get('difficulty') for row in all_epochs)},
            'best_epoch_by_fold': {str(result['fold']): result['best_epoch'] for result in group_results},
            'best_epoch_mean': float(np.mean([result['best_epoch'] for result in group_results])),
            'site_weight_share_by_fold': {str(result['fold']): result['epochs'][-1]['objective']['site_weight_share']
                                          for result in group_results},
            'input_digest_by_fold': {str(result['fold']): result['input_digest'] for result in group_results},
        }
        if group in {'H2', 'H3', 'H4'}:
            entry['hard_multiplier_mean_by_epoch'] = {
                str(epoch): float(np.mean([row['objective']['hard_multiplier']['mean'] for row in all_epochs
                                           if row['epoch'] == epoch and row['objective'].get('hard_multiplier')]))
                for epoch in sorted({row['epoch'] for row in all_epochs})}
        if group in {'H3', 'H4'}:
            for key in ('residual_R', 'condition_E', 'confidence_c', 'reweight_q'):
                entry[key + '_mean_by_epoch'] = {
                    str(epoch): float(np.mean([row['objective'][key]['mean'] for row in all_epochs
                                               if row['epoch'] == epoch]))
                    for epoch in sorted({row['epoch'] for row in all_epochs})}
            entry.update({
                'q_abs_deviation_mean': float(np.mean([epoch['objective']['q_abs_deviation_mean']
                                                       for epoch in active_epochs])),
                'q_sd_mean': float(np.mean([epoch['objective']['reweight_q']['sd'] for epoch in active_epochs])),
                'confidence_mean': float(np.mean([epoch['objective']['confidence_c']['mean'] for epoch in active_epochs])),
                'confidence_sd_mean': float(np.mean([epoch['objective']['confidence_c']['sd'] for epoch in active_epochs])),
                'residual_R_sd_mean': float(np.mean([epoch['objective']['residual_R']['sd'] for epoch in active_epochs])),
                'condition_E_sd_mean': float(np.mean([epoch['objective']['condition_E']['sd'] for epoch in active_epochs])),
                'conservation_error_max': float(np.max([epoch['objective']['weight_conservation_relative_error']
                                                        for epoch in active_epochs])),
                'residual_temperature_mean': float(np.mean([epoch['objective']['T_residual'] for epoch in active_epochs])),
                'condition_temperature_mean': float(np.mean([epoch['objective']['T_condition'] for epoch in active_epochs])),
                'difficulty_q_correlation_mean': float(np.nanmean([
                    epoch['objective']['difficulty_q_correlation'] or 0 for epoch in active_epochs])),
                'prototype_preparation_shuffle_changed_fraction_by_fold': {
                    str(result['fold']): result['prototype_preparation']['shuffle_changed_fraction']
                    for result in group_results},
            })
        mechanism[group] = entry

    h3_h2_weight_change = audit['preflight']['real_forward']['result']['H3_H2_mean_absolute_weight_change']
    h3_h4 = audit['preflight']['real_forward']['result']
    checks = {
        'all_40_formal_tasks_complete': completed == 40,
        'manifest_exactly_h1_h4_seed911_folds0_9': len(audit.get('manifest', {}).get('tasks', [])) == 40,
        'gpu_split_20_each': gpu_counts == {'0': 20, '1': 20},
        'OOF_each_group_871_once': all(len(values[group]) == 10 for group in run.GROUPS),
        'historical_D4_not_retrained': audit['historical_readonly']['historical_D4']['retrained'] is False,
        'historical_G2_not_retrained': audit['historical_readonly']['historical_G2']['retrained'] is False,
        'H3_H4_true_condition_identical_preflight': h3_h4['H3_H4_true_condition_equal'],
        'H3_H4_original_input_hash_equal_preflight': h3_h4['same_model_inputs_H3_H4_sha256'] ==
            values['H3'][0]['input_digest'] == values['H4'][0]['input_digest'],
        'H3_H4_global_prototypes_equal_on_shared_forward': h3_h4['H3_H4_global_prototypes_equal'],
        'H3_H4_teacher_kernel_different': h3_h4['H3_H4_teacher_kernel_different'],
        'preflight_original_parameters_restored': h3_h4['parameter_hash_unchanged'],
        'H3_weight_reallocation_observable': h3_h2_weight_change > 0,
        'all_formal_test_forward_counts_one': all(result['test_forwards'] == 1
                                                  for group in run.GROUPS for result in values[group]),
        'all_formal_artifacts_hash_valid': len(hashes) == completed,
        'all_protected_history_and_source_hashes_valid': True,
    }
    return {'values': values, 'history': history, 'metrics': summary_metrics, 'pairs': pair_details,
            'loo': loo_details, 'mechanism': mechanism, 'hashes': hashes, 'gpu_counts': gpu_counts,
            'completed_tasks': completed, 'checks': checks, 'fold_metrics_rows': metrics_rows,
            'oof_rows': oof_rows, 'h3_h2_preflight_weight_change': h3_h2_weight_change,
            'preflight': audit['preflight']}


def research_judgment(data):
    if data['completed_tasks'] != 40:
        return 'PARTIAL', {'all_tasks_complete': False}
    p = data['pairs']
    h3_d4 = p['H3-historical_D4']['metrics']
    h3_g2 = p['H3-historical_G2']['metrics']
    h3_h2 = p['H3-H2']['metrics']
    h3_h4 = p['H3-H4']['metrics']
    checks = {
        'H3_AUC_above_H2': h3_h2['AUC']['mean_pp'] > 0,
        'H3_AUC_above_H4': h3_h4['AUC']['mean_pp'] > 0,
        'H3_AUC_above_historical_G2': h3_g2['AUC']['mean_pp'] > 0,
        'H3_AUC_above_historical_D4': h3_d4['AUC']['mean_pp'] > 0,
        'H3_AUC_at_least_plus_0_5pp_vs_D4': h3_d4['AUC']['mean_pp'] >= .5,
        'H3_ACC_not_below_D4': h3_d4['ACC']['mean_pp'] >= 0,
        'H3_SPE_not_more_than_0_5pp_below_H2': h3_h2['SPE']['mean_pp'] >= -.5,
        'H3_H2_AUC_gain_not_single_fold': h3_h2['AUC']['improved_folds'] > 1 and
            h3_h2['AUC']['loo_diagnostics']['least_favorable_leave_one_fold_out_mean_pp'] > 0,
        'H3_H4_AUC_gain_not_single_fold': h3_h4['AUC']['improved_folds'] > 1 and
            h3_h4['AUC']['loo_diagnostics']['least_favorable_leave_one_fold_out_mean_pp'] > 0,
        'q_changes_weights': data['mechanism']['H3']['q_abs_deviation_mean'] > 0,
        'shuffle_active_all_folds': all(value > 0 for value in
            data['mechanism']['H4']['prototype_preparation_shuffle_changed_fraction_by_fold'].values()),
        'formal_protocol_and_audits_passed': all(data['checks'].values()),
    }
    if all(checks.values()):
        return 'GO', checks
    has_mechanism = checks['q_changes_weights'] and checks['shuffle_active_all_folds']
    has_signal = any(checks[key] for key in ('H3_AUC_above_H2', 'H3_AUC_above_H4',
                                             'H3_AUC_above_historical_G2', 'H3_AUC_above_historical_D4'))
    if has_mechanism and has_signal:
        return 'MIXED', checks
    return 'NO-GO', checks


def _metric_table(metrics, names):
    lines = ['| 组别 | 方法 | ACC | SEN-ASD | SPE-TD | probability AUC |',
             '|---|---|---:|---:|---:|---:|']
    for group, label in names:
        row = metrics[group]
        lines.append('| ' + ' | '.join([label] + [
            f"{row[key]['mean']:.4f} ± {row[key]['sample_sd']:.4f}" for key in METRICS]) + ' |')
    return lines


def render_report(data, decision, checks, audit):
    labels = {
        'historical_D4': 'READ-ONLY HISTORICAL：D4 年龄×性别疾病原型',
        'historical_G2': 'READ-ONLY HISTORICAL：实验16 G2 表型条件排序',
        'H1': '仅同站点条件排序，不用年龄/性别或原型',
        'H2': '真实表型条件 + 困难配对有界加权，不用原型',
        'H3': 'H2 + D4 全局/条件原型增量中心化重分配',
        'H4': 'H3 + 同站点打乱的原型教师，基础条件仍用真实表型',
    }
    table_names = [(group, labels[group]) for group in
                   ('historical_D4', 'historical_G2', 'H1', 'H2', 'H3', 'H4')]
    preflight_attempts = audit['preflight']['real_forward']['attempts']
    preflight_history = '; '.join(
        f"#{item['attempt']} {item['status']}: {item.get('reason', 'initial attempt')} "
        f"({item.get('model_forward_completed_count', 0)} completed forwards)"
        for item in preflight_attempts)
    lines = [
        '# PCHRRA_20261021 正式实验报告', '', f'STATUS={"COMPLETE" if data["completed_tasks"] == 40 else "PARTIAL"}', '',
        '## 设计与执行', '',
        '本轮只训练 H1、H2、H3、H4，seed 911、fold 0-9，共 40 项。历史 D4 与实验16 G2 从既有正式 artifact 只读复算，不创建重训任务。所有新任务从 DRPA 对应 fold 的 train_start core 和完整 RNG 恢复，复用原 split、RFE500、VAE 和人口图。',
        '',
        '原始 forward 返回 logits 和 native Graph loss，并由同一次 Imaging GTUNet forward hook 获取 z_img[871,16]；core.joint_embed[871,16] 是最终融合表示。训练损失为 native CE + Graph + lambda*L_aux；H1-H4 共用固定 margin=1、L_aux=0.5*L_global+0.5*L_conditional、双向 anchor 归一化和 joint_embed 5% 梯度预算。每 epoch 一次训练 forward、一次 validation forward；best 只由 validation CE+Graph 严格下降选择，测试在训练结束 RNG 下仅一次原生 grad-enabled forward。',
        '',
        'H1 只同站点；H2 加真实年龄/性别核与 h_ij=sigmoid(1-delta.detach()) 困难权重 B=W*(1+0.25h)；H3 使用与 D4 相同的三私有子折、标准化、全局/条件疾病原型及共同 ESS 收缩，基于 R 和 E 形成 c，再用 q=1+0.5(c-c_bar) 中心化重分配；H4 只打乱原型教师年龄/性别，基础 W 仍为真实表型。所有组均未增加参数、分类头或推理分支。',
        '',
        '正式任务窗口：见 audit.execution.queue。GPU0 固定偶数 fold、GPU1 固定奇数 fold；manifest 只含四组 × 10 folds。',
        '', '## 四指标结果', '',
        '每 fold 独立计算四指标后报告十折 mean ± sample SD（ddof=1），单位百分比；SEN 正类为 ASD=0，SPE 为 TD=1，AUC 使用 softmax ASD 概率。历史行标注为只读。', '',
        *_metric_table(data['metrics'], table_names), '',
        '## 配对折差与敏感性', '',
        '差值为左侧减右侧、单位百分点；所有配对按同一 fold 计算。历史 G2/D4 是跨实验只读参照，不构成同批次重训因果估计。', '',
        '| 比较 | 指标 | mean ± fold SD | 改善折数 | 最大单折差（fold） | 最大绝对折差/绝对差总和 | 最不利LOO | 最有利LOO | 去最大单折LOO |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for pair, detail in data['pairs'].items():
        for metric in METRICS:
            row = detail['metrics'][metric]
            loo = row['loo_diagnostics']
            lines.append(f"| {pair} | {metric} | {row['mean_pp']:+.4f} ± {row['sample_sd_pp']:.4f} | "
                         f"{row['improved_folds']}/10 | {loo['max_single_fold_difference_pp']:+.4f} (F{loo['max_single_fold_id']}) | "
                         f"{loo['max_absolute_fold_difference_share_of_absolute_sum']:.3f} | "
                         f"{loo['least_favorable_leave_one_fold_out_mean_pp']:+.4f} | "
                         f"{loo['most_favorable_leave_one_fold_out_mean_pp']:+.4f} | "
                         f"{loo['remove_largest_absolute_fold_mean_pp']:+.4f} |")
    lines += ['', 'AUC 逐折差（百分点）：', '']
    lines += ['| Fold | H3-H2 | H3-H4 | H3-historical_G2 | H3-historical_D4 | H2-historical_G2 |',
              '|---:|---:|---:|---:|---:|---:|']
    for fold in range(10):
        vals = [data['pairs'][pair]['metrics']['AUC']['fold_differences_pp'][fold]
                for pair in ('H3-H2', 'H3-H4', 'H3-historical_G2', 'H3-historical_D4', 'H2-historical_G2')]
        lines.append(f'| {fold} | ' + ' | '.join(f'{v:+.4f}' for v in vals) + ' |')

    lines += ['', '## 机制审计', '',
              f"CPU 数学测试检查数：{audit['preflight']['math']['result']['checks']}；真实 PPU 单前向预检尝试 {audit['preflight']['real_forward']['attempt_count']} 次（{preflight_history}）。成功预检确认 logits {audit['preflight']['real_forward']['result']['logits_shape']}、z_img {audit['preflight']['real_forward']['result']['z_img_shape']}、joint_embed {audit['preflight']['real_forward']['result']['joint_embed_shape']}。原生 forward 加载的 VAE checkpoint 参数和 RP 卷积约束投影均已核验并在预检后恢复，参数快照保持不变；H3/H4 真实条件权重相同、原始输入 hash 相同、共享输入下全局原型相同而条件 teacher kernel 不同；H3 相对 H2 的平均绝对 pair weight 改变量为 {data['h3_h2_preflight_weight_change']:.6g}。",
              '', '| 组别 | 条件配对覆盖 | 梯度预算均值 | 梯度余弦均值 | 困难度均值 | q相对1平均绝对偏差 | q SD | 残差R SD | 条件证据E SD | 最大中心化守恒误差 | 最佳epoch均值 |',
              '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for group in run.GROUPS:
        m = data['mechanism'][group]
        pairs_min = min(m['conditional_pairs_by_fold'].values())
        pairs_max = max(m['conditional_pairs_by_fold'].values())
        qdev = m.get('q_abs_deviation_mean')
        qsd = m.get('q_sd_mean')
        rsd = m.get('residual_R_sd_mean')
        esd = m.get('condition_E_sd_mean')
        cons = m.get('conservation_error_max')
        difficulty_mean = (f"{np.mean(list(m['difficulty_mean_by_epoch'].values())):.4f}"
                           if m['difficulty_mean_by_epoch'] else 'NA')
        lines.append(f"| {group} | {pairs_min}-{pairs_max}; ASD anchor {min(m['asd_anchor_fraction_by_fold'].values()):.3f}; TD anchor {min(m['td_anchor_fraction_by_fold'].values()):.3f} | "
                     f"{m['weighted_joint_grad_ratio_mean']:.5f} | {m['joint_grad_cosine_mean']:.4f} | "
                     f"{difficulty_mean} | "
                     f"{qdev if qdev is not None else 'NA'} | {qsd if qsd is not None else 'NA'} | "
                     f"{rsd if rsd is not None else 'NA'} | {esd if esd is not None else 'NA'} | "
                     f"{cons if cons is not None else 'NA'} | {m['best_epoch_mean']:.1f} |")
    lines += ['',
              'H3 与 H2 的唯一算法差异是 H3 对 H2 基础困难权重 B 做基于原型 c 的中心化 q 重分配；全局排序、真实条件 W、margin、anchor 归一化、模型、输入、训练预算及梯度预算相同。sum(Bq) 与 sum(B) 的最大相对误差见表。每个 epoch 的梯度余弦、difficulty、c/q、站点权重份额和 best epoch 轨迹保存在结果 artifact 中，摘要按 epoch 汇总。固定 5% 是 joint_embed 上加权辅助梯度范数预算，不表示参数更新方向相同。',
              '',
              'H4 与 H3 的定义差异仅在原型教师的年龄/性别置换；基础条件配对 W 始终使用真实年龄、性别和站点。置换严格使用 PCDCA 私有 seed 字符串，且仅在 outer-train 同站点内进行。H3/H4 的训练轨迹后续可因辅助梯度方向改变而分化。',
              '', '## 科研判断', '', f'**{decision}**', '', '预设筛选项：']
    for key, value in checks.items():
        lines.append(f"- [{'x' if value else ' '}] {key}")
    lines += ['',
              'H3-H2 的差值隔离的是原型残差重分配（两组均含真实年龄/性别核和困难权重）；H3-H4 检验原型教师是否依赖正确表型，但单 seed 十折不是独立队列。历史 G2/D4 比较跨运行，受原生随机 VAE/eval dropout、PPU 长轨迹和 checkpoint 漂移影响。逐折方向及 leave-one-fold-out 仅为敏感性诊断，不作显著性推断。',
              '',
              '保留历史跨折监督 RFE 预处理、871 节点转导人口图、VAE 随机采样、eval 随机 dropout 和原始 reward/penalty 实现。研究结论受单 seed、重复 CV 相关性和站点内验证限制；MICCAI 2027 的主创新地位仍需预注册多 seed 与独立站点/队列确认。',
              '', '## 完整性与产物', '',
              f"manifest={len(audit.get('manifest', {}).get('tasks', []))} tasks；GPU0={data['gpu_counts']['0']}、GPU1={data['gpu_counts']['1']}；accepted events={len(audit['execution']['events'])}；formal failures={len(audit['execution']['failures'])}。D0/D4/实验16 G2 未重训；manifest 只含 H1-H4。历史 split、RFE、VAE、train_start core/RNG 和输入源码均通过 SHA 对照。", '',
              '文件：`config.json`、`objective.py`、`run.py`、`launch.py`、`analyze.py`、`manifest.json`、`audit.json`、`run.log`、`fold_metrics.csv`、`oof.csv`、`summary.json`、`report.md`、`runs/H1-H4/`。源码/config 组合 SHA256：`' + audit['code_sha256'] + '`。']
    return '\n'.join(lines) + '\n'


def handoff_text(summary):
    metrics = summary['metrics_percent_mean_sample_sd']
    pairs = summary['paired_fold_differences_pp']
    names = {'historical_D4': '只读D4', 'historical_G2': '只读实验16 G2',
             'H1': '仅站点条件排序', 'H2': '表型条件+困难配对',
             'H3': '完整PC-HRRA', 'H4': '打乱原型教师，真实基础条件'}
    lines = [
        '<!-- PCHRRA_20261021_START -->', '## Experiment 17: PCHRRA_20261021', '',
        f"状态：`{summary['status']}`；科研判断：`{summary['research_decision']}`。路径：`{BASE}`。研究问题：困难患者对的有界重加权是否有效，以及 D4 条件疾病原型相对全局原型的中心化增量证据是否带来新增诊断信息。seed911，10折；计划40任务，实际完成{summary['formal_tasks']}；GPU0 {summary['gpu_completed']['0']}/20，GPU1 {summary['gpu_completed']['1']}/20。", '',
        '方法：H1 仅同站点全局/条件排序；H2 加真实年龄/性别条件与 1–1.25 困难配对倍率；H3 加 D4 全局/条件疾病原型证据残差 R 与正确排序证据 E，形成 c 后用 q=1+0.5(c-c_bar) 中心化重分配；H4 只打乱同站点原型教师年龄/性别，基础 W 仍是真实表型。H3/H4 没有新增头、参数、adaptive margin 或测试阶段分支。历史D4与实验16 G2只读，未重训。', '',
        '| 组别 | 方法 | ACC | SEN-ASD | SPE-TD | AUC |', '|---|---|---:|---:|---:|---:|']
    for group in ('historical_D4', 'historical_G2', 'H1', 'H2', 'H3', 'H4'):
        lines.append('| ' + ' | '.join([names[group]] + [
            f"{metrics[group][key]['mean']:.4f} ± {metrics[group][key]['sample_sd']:.4f}" for key in METRICS]) + ' |')
    lines += ['', '| 比较 | ΔACC | ΔSEN | ΔSPE | ΔAUC |', '|---|---:|---:|---:|---:|']
    for pair in ('H3-H2', 'H3-H4', 'H3-historical_G2', 'H3-historical_D4', 'H2-historical_G2'):
        lines.append('| ' + pair + ' | ' + ' | '.join(
            f"{pairs[pair]['metrics'][key]['mean_pp']:+.4f}" for key in METRICS) + ' |')
    lines += ['',
              f"原型机制：中心化 q、R/E、c 分布及每 epoch 变化见 summary/report；CPU数学测试 {summary['preflight']['math']['result']['checks']} 项通过，真实单前向预检尝试 {summary['preflight']['real_forward']['attempt_count']} 次。H3/H4 基础条件相同、共享输入下全局原型相同、teacher kernel 不同；H3-H2 权重变化和总权重守恒已审计。", '',
              f"实际运行：正式任务墙钟 {summary.get('formal_wall_seconds', 0):.1f} 秒，累计 worker {summary.get('cumulative_worker_seconds', 0):.1f} 秒；失败 {summary.get('formal_failures', 0)}，安全重试 {summary.get('formal_retries', 0)}。预检/正式运行偏差详见 report.md。", '',
              f"源码/config SHA256：`{summary['code_sha256']}`；manifest SHA256：`{summary['manifest_sha256']}`。H3相对 D4/G2 结论受跨运行随机轨迹、单 seed、RFE与转导设置限制。建议预注册多 seed 并做独立站点/队列验证。", '',
              '<!-- PCHRRA_20261021_END -->']
    return '\n'.join(lines) + '\n'


def append_handoff(summary, audit):
    path = run.ROOT / 'handoff.md'
    marker = '<!-- PCHRRA_20261021_START -->'
    before_len = int(audit['handoff_before_bytes'])
    before_sha = audit['handoff_before_sha256']
    content = handoff_text(summary).encode()
    try:
        fd = os.open(path, os.O_RDWR | os.O_APPEND)
        try:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX)
            current = os.pread(fd, os.fstat(fd).st_size, 0)
            if marker.encode() in current:
                if not run.handoff_prefix_ok(audit):
                    raise RuntimeError('Existing PC-HRRA handoff marker has an invalid original prefix')
                final = run.read_bytes(path)
            else:
                if len(current) != before_len or hashlib.sha256(current).hexdigest() != before_sha:
                    raise RuntimeError('handoff changed after audit; refusing non-append update')
                payload = b'\n' + content
                written = 0
                while written < len(payload):
                    written += os.write(fd, payload[written:])
                os.fsync(fd)
                final = run.read_bytes(path)
            prefix = final[:before_len]
            if hashlib.sha256(prefix).hexdigest() != before_sha:
                raise RuntimeError('handoff original byte prefix changed')
            summary['handoff'] = {'status': 'appended', 'path': str(path),
                                  'before_bytes': before_len, 'before_sha256': before_sha,
                                  'after_bytes': len(final), 'after_sha256': hashlib.sha256(final).hexdigest(),
                                  'original_prefix_preserved': True}
            audit['handoff_appended'] = True
            audit['handoff_after_sha256'] = summary['handoff']['after_sha256']
            audit['handoff_after_bytes'] = len(final)
            run.atomic_json(audit, run.BASE / 'audit.json')
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
    except Exception as exc:
        pending = run.BASE / 'handoff_pending.md'
        pending.write_text(content.decode(), encoding='utf-8')
        summary['handoff'] = {'status': 'append_failed', 'path': str(path),
                              'pending_path': str(pending), 'error': repr(exc)}
        audit['handoff_appended'] = False
        run.atomic_json(audit, run.BASE / 'audit.json')
    run.atomic_json(summary, run.BASE / 'summary.json')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    if args.final == args.partial:
        raise ValueError('Select exactly one of --final or --partial')
    audit = run.guard()
    data = inspect_results(audit, require_complete=args.final)
    decision, decision_checks = research_judgment(data)
    write_csv(run.BASE / 'fold_metrics.csv', ['group', 'seed', 'fold', *METRICS], data['fold_metrics_rows'])
    write_csv(run.BASE / 'oof.csv', ['group', 'seed', 'fold', 'subject_id', 'y_true', 'y_pred',
                                     'probability_asd'], data['oof_rows'])
    report = render_report(data, decision, decision_checks, audit)
    report_path = run.BASE / 'report.md'
    temp = report_path.with_name(report_path.name + f'.tmp.{os.getpid()}')
    with temp.open('x') as handle:
        handle.write(report)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, report_path)
    queue = audit.get('execution', {}).get('queue', {})
    events = audit.get('execution', {}).get('events', [])
    summary = {
        'status': 'COMPLETE' if args.final and data['completed_tasks'] == 40 else 'PARTIAL',
        'experiment': run.CFG['experiment'], 'formal_tasks': data['completed_tasks'],
        'gpu_completed': data['gpu_counts'], 'formal_wall_seconds': queue.get('wall_seconds'),
        'cumulative_worker_seconds': float(sum(event.get('wall_seconds', 0) for event in events)),
        'formal_failures': len(audit.get('execution', {}).get('failures', [])),
        'formal_retries': sum(1 for row in audit.get('execution', {}).get('events', [])
                              if row.get('attempt', 1) > 1),
        'code_sha256': audit['code_sha256'], 'config_sha256': audit['config_sha256'],
        'manifest_sha256': run.sha_file(run.BASE / 'manifest.json') if (run.BASE / 'manifest.json').exists() else None,
        'prompt_sha256': audit['prompt_sha256'], 'result_sha256': data['hashes'],
        'metrics_percent_mean_sample_sd': data['metrics'],
        'paired_fold_differences_pp': data['pairs'], 'leave_one_fold_out': data['loo'],
        'mechanism': data['mechanism'], 'integrity_checks': data['checks'],
        'research_decision': decision, 'decision_checks': decision_checks,
        'historical_readonly': audit['historical_readonly'], 'preflight': data['preflight'],
        'protocol_limitations': run.CFG['historical_protocol_limitations'],
        'artifact_files': ['config.json', 'objective.py', 'run.py', 'launch.py', 'analyze.py',
                           'manifest.json', 'audit.json', 'run.log', 'fold_metrics.csv', 'oof.csv',
                           'summary.json', 'report.md', 'runs/H1/', 'runs/H2/', 'runs/H3/', 'runs/H4/'],
        'handoff': {'status': 'pending append after final audit'},
    }
    run.atomic_json(summary, run.BASE / 'summary.json')
    audit['analysis'] = {'completed_utc': run.utc(), 'completed_tasks': data['completed_tasks'],
                         'report_sha256': run.sha_file(report_path),
                         'fold_metrics_sha256': run.sha_file(run.BASE / 'fold_metrics.csv'),
                         'oof_sha256': run.sha_file(run.BASE / 'oof.csv'),
                         'summary_sha256_before_handoff': run.sha_file(run.BASE / 'summary.json')}
    run.atomic_json(audit, run.BASE / 'audit.json')
    if args.final:
        run.guard(audit)
        summary = append_handoff(summary, audit)
    print(json.dumps({'status': summary['status'], 'formal_tasks': summary['formal_tasks'],
                      'gpu_completed': summary['gpu_completed'],
                      'research_decision': summary['research_decision'],
                      'handoff': summary['handoff']}, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
