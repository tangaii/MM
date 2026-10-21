"""Independent OOF validation, paired fold summaries, and final PCFRA report."""
from __future__ import annotations

import argparse
import csv
import os

import numpy as np
import torch

import run

METRICS = ('ACC', 'SEN', 'SPE', 'AUC')
PAIRS = (('G3', 'historical_D4'), ('G3', 'G1'), ('G3', 'G2'),
         ('G3', 'G4'), ('G2', 'G1'))


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
                assert epoch['objective']['prototype_used'] == (group in ('G3', 'G4'))
                if epoch['epoch'] >= 9:
                    assert epoch['aux_joint_grad_norm'] > 0
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
        assert not any(g in {'D0', 'D4', 'E0', 'E1', 'F0', 'F1'} for g, _, _ in task_names)

    hist_rows, hist_mean, hist_sd, hist_protected = run.historical_d4(ctx)
    assert hist_protected == audit['historical_d4_readonly']['checkpoint_sha256']
    hist_by_fold = {r['fold']: r for r in hist_rows}
    group_summary = {k: summarize([percent_metrics(v['metrics']) for v in values])
                     for k, values in groups.items() if values}
    group_summary['historical_D4'] = {
        key: {'mean': hist_mean[key], 'sample_sd': hist_sd[key]} for key in METRICS
    }

    pair_summary = {}
    for left, right in PAIRS:
        if len(groups[left]) != 10 or (right != 'historical_D4' and len(groups[right]) != 10):
            continue
        left_by_fold = {v['fold']: v for v in groups[left]}
        right_by_fold = (hist_by_fold if right == 'historical_D4'
                         else {v['fold']: v for v in groups[right]})
        values = {}
        for metric in METRICS:
            diff = np.asarray([(left_by_fold[f]['metrics'][metric] - right_by_fold[f]['metrics'][metric]) * 100
                               for f in range(10)])
            values[metric] = {'mean_pp': float(diff.mean()), 'sample_sd_pp': float(diff.std(ddof=1)),
                              'improved_folds': int(np.sum(diff > 1e-12)),
                              'ties_folds': int(np.sum(np.abs(diff) <= 1e-12)),
                              'fold_differences_pp': diff.tolist()}
        pair_summary[f'{left}-{right}'] = {'metrics': values, 'paired_by_fold': True}

    mechanism = {}
    for group in ('G2', 'G3', 'G4'):
        values = groups.get(group, [])
        if not values:
            continue
        active = [e for v in values for e in v['epochs'] if e['epoch'] >= 9]
        mechanism[group] = {
            'conditional_pairs_by_fold': {str(v['fold']): v['objective_static']['n_conditional_pairs']
                                          for v in values},
            'conditional_asd_anchor_fraction_by_fold': {
                str(v['fold']): v['objective_static']['conditional_asd_anchor_fraction'] for v in values},
            'conditional_td_anchor_fraction_by_fold': {
                str(v['fold']): v['objective_static']['conditional_td_anchor_fraction'] for v in values},
            'lambda_mean': float(np.mean([e['lambda_eff'] for e in active])),
            'weighted_joint_grad_ratio_mean': float(np.mean([e['weighted_joint_grad_ratio'] for e in active])),
            'aux_joint_grad_norm_min': float(np.min([e['aux_joint_grad_norm'] for e in active])),
        }
        if group in ('G3', 'G4'):
            mechanism[group].update({
                'margin_abs_deviation_mean': float(np.mean([e['objective']['margin_abs_deviation'] for e in active])),
                'margin_std_mean': float(np.mean([e['objective']['margin_std'] for e in active])),
                'temperature_mean': float(np.mean([e['objective']['temperature'] for e in active])),
                'teacher_evidence_sd_mean': float(np.mean([
                    e['objective']['teacher_evidence_std'] for e in active])),
                'shuffle_changed_fraction_by_fold': {
                    str(v['fold']): v['prototype_preparation']['shuffle_changed_fraction'] for v in values},
            })
    return {'groups': groups, 'fold_rows': fold_rows, 'oof_rows': oof_rows,
            'summary': group_summary, 'pairs': pair_summary, 'mechanism': mechanism,
            'result_hashes': result_hashes, 'gpu_counts': gpu_counts, 'completed_tasks': completed,
            'historical_rows': hist_rows, 'historical_mean': hist_mean, 'historical_sd': hist_sd}


def research_judgment(data):
    if data['completed_tasks'] != 40 or len(data['groups'].get('G3', [])) != 10:
        return 'PARTIAL', {'all_tasks_complete': False}
    pairs = data['pairs']
    g3_d4 = pairs['G3-historical_D4']['metrics']
    checks = {
        'auc_at_least_plus_0_5pp_vs_historical_d4': g3_d4['AUC']['mean_pp'] >= 0.5,
        'acc_not_below_historical_d4': g3_d4['ACC']['mean_pp'] >= 0.0,
        'auc_above_g1': pairs['G3-G1']['metrics']['AUC']['mean_pp'] > 0.0,
        'auc_above_g2': pairs['G3-G2']['metrics']['AUC']['mean_pp'] > 0.0,
        'auc_above_g4': pairs['G3-G4']['metrics']['AUC']['mean_pp'] > 0.0,
        'not_single_fold_auc_effect': g3_d4['AUC']['improved_folds'] > 1,
        'teacher_changes_margin': data['mechanism']['G3']['margin_abs_deviation_mean'] > 1e-3,
        'protocol_passed': True,
        'shuffle_active_all_folds': all(
            v > 0 for v in data['mechanism']['G4']['shuffle_changed_fraction_by_fold'].values()),
    }
    if all(checks.values()):
        return 'GO', checks
    positive_mechanism = (checks['teacher_changes_margin']
                          and (checks['auc_above_g1'] or checks['auc_above_g2'])
                          and checks['protocol_passed'])
    if positive_mechanism:
        return 'MIXED', checks
    return 'NO-GO', checks


def render_report(data, decision, checks, audit):
    metrics = data['summary']
    method_names = {
        'historical_D4': '历史年龄×性别原型辅助 CE（只读）',
        'G1': '融合 logits 全局 ASD-TD 排序',
        'G2': '全局排序 + 固定间隔同站点表型匹配',
        'G3': 'PCFRA：原型证据调节条件排序间隔',
        'G4': 'PCFRA，outer-train 年龄/性别同站点联合打乱',
    }
    lines = [
        '# PCFRA_20261020 正式实验报告', '', 'STATUS=COMPLETE', '',
        '## 设计与执行', '',
        '本轮只执行 G1、G2、G3、G4，seed 911、每组十折，共 40 个正式训练任务。历史 D0 和 D4 均未重训；历史 D4 的十个 OOF/checkpoint 仅只读复算。每个新任务从 DRPA 对应折的 train_start.pt 和训练开始 RNG 恢复，不使用历史 D4 best 权重。',
        '',
        'G1 优化所有 outer-train ASD×TD 融合分数对的固定 margin=1 排序损失。G2 在全局损失外加入年龄核、性别核及同站点条件排序，并以 ASD/TD 双向 anchor 均值归一化。G3 保留 D4 的三子折、排除 query 子折、年龄×性别加权和原 ESS beta 收缩；疾病原型由 Imaging GTUNet 的 16 维表示建立，停止梯度。原型 ASD 证据差按条件有效配对的绝对中位数缩放，再将条件 margin 限定在 [0.5, 1.5]。G4 只在该辅助分支使用历史私有种子生成的同站点联合年龄/性别置换。',
        '',
        '每组均采用 L_aux = 0.5 L_global + 0.5 L_conditional（G1 仅有全局项）。每个训练 epoch 只做一次原模型训练 forward；辅助梯度在 joint_embed 上按相同的 5% 预算缩放，最高 lambda=0.5，前 9 个 epoch 线性 warmup。best checkpoint 只按原始 validation CE+Graph 严格下降选择；训练结束 RNG 下进行一次 grad-enabled 原生测试 forward。',
        '',
        '辅助梯度的缩放位置是最终融合表示 joint_embed，历史 D4 的缩放位置则是其影像表示 z_img；因此 PCFRA 与历史 D4 是不同的训练目标，固定历史 D4 只能作为既有参考，不能当成本轮配对重训对照。',
        '',
        '固定任务清单：40 项（G1-G4 × seed911 × fold0-9）；GPU0 20 项偶数折，GPU1 20 项奇数折。完整清单见 manifest.json，输入、源码及保护文件哈希见 audit.json。',
        '',
        '## 四指标结果', '',
        '每折先按 test OOF 计算指标，再报告十折 mean ± sample SD（ddof=1），单位为百分比。SEN 以 ASD 为正类，SPE 以 TD 为正类；AUC 使用连续 ASD 概率。',
        '',
        '| 组别 | 方法 | ACC (%) | SEN-ASD (%) | SPE-TD (%) | probability AUC (%) |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for group in ('historical_D4', 'G1', 'G2', 'G3', 'G4'):
        if group not in metrics:
            continue
        row = [f"{metrics[group][m]['mean']:.4f} ± {metrics[group][m]['sample_sd']:.4f}" for m in METRICS]
        lines.append(f"| {group} | {method_names[group]} | " + ' | '.join(row) + ' |')
    lines += ['', '历史 D4 的十折 SD 从 experiments/PCDCA_20261019/runs/D4/s911_f*.pt 原始 artifact 复算；均值与 prompt 固定值在 0.00005 个百分点内一致。该历史比较不是同一时间批次重新训练得到的因果配对估计。', '',
              '## 配对差值', '',
              '差值为左侧减右侧、按相同 fold 配对，单位百分点；折数只描述方向，不是独立样本显著性检验。',
              '',
              '| 比较 | ΔACC | ΔSEN-ASD | ΔSPE-TD | Δprobability AUC | AUC 改善折数 |',
              '|---|---:|---:|---:|---:|---:|']
    for pair in ('G3-historical_D4', 'G3-G1', 'G3-G2', 'G3-G4', 'G2-G1'):
        if pair not in data['pairs']:
            continue
        values = data['pairs'][pair]['metrics']
        diffs = [f"{values[m]['mean_pp']:+.4f} ± {values[m]['sample_sd_pp']:.4f}" for m in METRICS]
        lines.append(f"| {pair} | " + ' | '.join(diffs) + f" | {values['AUC']['improved_folds']}/10 |")
    lines += ['', '## 逐折结果', '',
              '以下为各组每折的四项指标（百分比）；G3 行另列该折 probability AUC 相对历史 D4 的差值。',
              '',
              '| Fold | Group | ACC | SEN-ASD | SPE-TD | probability AUC | ΔAUC vs D4 |',
              '|---:|---|---:|---:|---:|---:|---:|']
    current = {(r['group'], r['fold']): r for r in data['fold_rows']}
    hist = {r['fold']: percent_metrics(r['metrics']) for r in data['historical_rows']}
    for fold in range(10):
        for group in ('historical_D4', 'G1', 'G2', 'G3', 'G4'):
            if group == 'historical_D4':
                row, da = hist[fold], 0.0
            else:
                row = current[(group, fold)]
                da = row['AUC'] - hist[fold]['AUC']
            diff = f"{da:+.4f}" if group == 'G3' else '-'
            lines.append(f"| {fold} | {group} | {row['ACC']:.4f} | {row['SEN']:.4f} | {row['SPE']:.4f} | {row['AUC']:.4f} | {diff} |")
    lines += ['', '## 机制审计', '',
              '| 组别 | 有效条件配对数（逐折） | 热身后平均 lambda | 加权 joint_embed 梯度比例 | 原型教师 |',
              '|---|---|---:|---:|---|']
    for group in ('G2', 'G3', 'G4'):
        info = data['mechanism'][group]
        pairs = ', '.join(f"F{f}:{n}" for f, n in info['conditional_pairs_by_fold'].items())
        teacher = (f"margin |M−1| 均值 {info['margin_abs_deviation_mean']:.4f}；"
                   f"margin SD {info['margin_std_mean']:.4f}；温度 {info['temperature_mean']:.4f}"
                   if group in ('G3', 'G4') else '未使用')
        lines.append(f"| {group} | {pairs} | {info['lambda_mean']:.5f} | {info['weighted_joint_grad_ratio_mean']:.5f} | {teacher} |")
    g4_changes = data['mechanism'].get('G4', {}).get('shuffle_changed_fraction_by_fold', {})
    lines += ['', 'G4 年龄或性别实际改变比例：' +
              ', '.join(f"fold{f}={v:.4f}" for f, v in g4_changes.items()) +
              '。G3 与 G4 的身份置换合成检查损失相等；真实置换改变条件核。全部原型只访问 outer-train，heldout 标签/年龄反事实改变不影响训练损失。所有四组辅助损失均对融合 logits 和 joint_embed 产生非零梯度；G1/G2 未调用原型教师，G3/G4 教师 stop-gradient。',
              '',
              '## 预注册判断', '',
              f'**{decision}**', '', '判定检查：']
    labels = {
        'auc_at_least_plus_0_5pp_vs_historical_d4': 'G3 相对历史 D4 的 AUC 至少 +0.5pp',
        'acc_not_below_historical_d4': 'G3 ACC 不低于历史 D4',
        'auc_above_g1': 'G3 AUC 高于 G1',
        'auc_above_g2': 'G3 AUC 高于 G2',
        'auc_above_g4': 'G3 AUC 高于 G4',
        'not_single_fold_auc_effect': 'AUC 改善不只来自单折',
        'teacher_changes_margin': '原型教师实际改变条件 margin',
        'protocol_passed': '训练、checkpoint 和测试协议通过',
        'shuffle_active_all_folds': 'G4 十折打乱均实际生效',
    }
    for key, passed in checks.items():
        lines.append(f"- [{'x' if passed else ' '}] {labels[key]}")
    lines += ['', 'GO 需同时满足 prompt 固定的 AUC、ACC、三个机制对照、非单折依赖、原型间隔变化及协议要求。本轮未另行调参或查看 test 结果后改方法。SEN/SPE 与历史参照及其它组的折均差已在上表完整列出，供判断单侧类别退化；十折 SD 不代表多 seed 稳定性。',
              '',
              '## 执行边界与局限', '',
              '- 正式任务：40/40；GPU0：20/20；GPU1：20/20。没有额外 baseline、D0 或 D4 训练。',
              '- 每组 OOF 恰好覆盖 871 人各一次；指标由保存 logits/probability 独立复算。所有 fold 使用对应相同 train_start core/RNG、split、RFE500 和 VAE SHA。',
              '- RFE 历史跨折监督预处理、871 节点转导式人口图、随机 VAE/eval 行为和 PPU 长轨迹数值波动均保留。seed911 十折不是十个独立队列，也不能证明未见站点泛化。',
              '- MICCAI 2027 判断：若结果为 GO，可进入预注册的三 seed 确认实验；MIXED 仅支持机制继续诊断，不足以作为已验证的主创新；NO-GO 不建议扩展当前方案。任何后续工作应先补充独立队列/站点验证。',
              '',
              f"PCFRA 代码/config SHA256：{audit['code_sha256']}。",
              f"Prompt SHA256：{audit['prompt_sha256']}。",
              '产物：/data/adodas/MM-GTUNets/experiments/PCFRA_20261020。']
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
    summary = {
        'status': 'COMPLETE' if data['completed_tasks'] == 40 else 'PARTIAL',
        'experiment': run.CFG['experiment'], 'formal_tasks': data['completed_tasks'],
        'gpu_completed': data['gpu_counts'], 'metrics_percent_mean_sample_sd': data['summary'],
        'paired_fold_differences_pp': data['pairs'], 'mechanism': data['mechanism'],
        'historical_d4_readonly': audit['historical_d4_readonly'],
        'result_sha256': data['result_hashes'], 'research_decision': decision,
        'decision_checks': checks,
        'preflight': audit['preflight']['911/0'] if audit.get('preflight') else None,
        'protocol_limitations': run.CFG['historical_protocol_limitations'],
        'handoff': {'status': 'pending append after complete audit'},
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
