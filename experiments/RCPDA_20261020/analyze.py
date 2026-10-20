"""Four-metric OOF verification and final RCPDA report."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import run

METRICS = ('ACC', 'SEN', 'SPE', 'AUC')
PAIRS = (('F1', 'F0'), ('F2', 'F1'), ('F3', 'F1'), ('F4', 'F1'),
         ('F4', 'F2'), ('F4', 'F3'), ('F4', 'F5'), ('F4', 'F0'))
HISTORICAL_D4 = {'ACC': 82.5588, 'SEN': 79.9268, 'SPE': 84.8427, 'AUC': 88.6897}
HISTORICAL_E4 = {'ACC': 82.6737, 'SEN': 80.9207, 'SPE': 84.1952, 'AUC': 88.1900}
START_MARK = '<!-- RCPDA_20261020_START -->'
END_MARK = '<!-- RCPDA_20261020_END -->'


def independent_metrics(y, p, pred):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.int64)
    asd = y == 0
    td = y == 1
    return {
        'ACC': float(np.mean(pred == y)),
        'SEN': float(np.mean(pred[asd] == 0)),
        'SPE': float(np.mean(pred[td] == 1)),
        'AUC': float(roc_auc_score(asd.astype(np.int8), p)),
    }


def summarize(folds):
    return {k: {'mean': float(np.mean([x[k] for x in folds])),
                'sample_sd': float(np.std([x[k] for x in folds], ddof=1))}
            for k in METRICS}


def write_csv(path, fields, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def inspect_results(a, require_complete):
    ctx = run.load_torch(run.OLD / 'artifacts/context.pt')
    assert len(ctx['ids']) == len(set(ctx['ids'])) == 871
    rows = {g: [] for g in run.CFG['groups']}
    artifacts = {}
    resource = {'gpu_tasks': {'0': 0, '1': 0}, 'training_seconds': 0.0, 'peak_reserved_bytes': 0}
    metrics_rows = []
    oof_rows = []
    gate_records = {g: [] for g in run.CFG['groups']}

    for group in run.CFG['groups']:
        coverage = np.zeros(871, dtype=np.int64)
        observed_ids = []
        for fold in range(10):
            value = run.valid_result(group, 911, fold, a, required=require_complete)
            if value is None:
                continue
            split = run.load_torch(run.anchor(911, fold) / 'splits.pt')
            idx = value['oof']['indices']
            y = value['oof']['labels']
            p = value['oof']['probability']
            pred = value['oof']['prediction']
            logits = np.asarray(value['oof']['logits'])
            ids = value['oof']['ids']
            assert np.array_equal(idx, split['test'])
            assert np.array_equal(y, ctx['labels'][idx])
            assert ids == [ctx['ids'][int(i)] for i in idx]
            assert np.array_equal(pred, logits.argmax(axis=1))
            probability_check = torch.softmax(torch.as_tensor(logits, dtype=torch.float64), dim=1)[:, 0].numpy()
            assert np.allclose(p, probability_check, atol=1e-7, rtol=1e-6)
            assert np.isfinite(p).all() and np.all((p >= 0.0) & (p <= 1.0))
            independent = independent_metrics(y, p, pred)
            assert max(abs(independent[k] - value['metrics'][k]) for k in METRICS) < 1e-12
            assert value['bn_best_batches'] == value['best_epoch'] + 1
            assert value['physical_gpu'] == fold % 2
            assert value['formal_start_rng_sha256'] == a['anchors'][f'911/{fold}']['pairing']['classifier_start_rng']
            assert value['forward_calls'] == {'train': value['epochs_run'], 'val': value['epochs_run'], 'test': 1}
            assert value['new_trainable_parameters'] == value['rfe_fit_count'] == value['vae_train_count'] == 0
            assert value['formal_test_grad_enabled'] is True
            coverage[idx] += 1
            observed_ids.extend(ids)
            rows[group].append(value)
            metrics_rows.append({'group': group, 'seed': 911, 'fold': fold, **independent})
            for i, subject in enumerate(ids):
                oof_rows.append({'group': group, 'seed': 911, 'fold': fold, 'subject_id': subject,
                                 'y_true': int(y[i]), 'y_pred': int(pred[i]),
                                 'probability_asd': float(p[i])})
            gate_records[group].extend(value['gate_diagnostics'])
            artifacts[f'{group}/911/{fold}'] = run.sha_file(run.result_path(group, 911, fold))
            gpu = str(value['physical_gpu'])
            resource['gpu_tasks'][gpu] += 1
            resource['training_seconds'] += value['training_seconds']
            resource['peak_reserved_bytes'] = max(resource['peak_reserved_bytes'], value['peak_reserved_bytes'])
        if rows[group]:
            assert np.all(coverage == 1), f'OOF coverage for {group} is not exactly once'
            assert len(observed_ids) == len(set(observed_ids)) == 871
            assert len(rows[group]) == 10 or not require_complete

    total = sum(len(x) for x in rows.values())
    if require_complete:
        assert total == 60 and all(len(rows[g]) == 10 for g in rows)
        assert resource['gpu_tasks'] == {'0': 30, '1': 30}

    fold_summary = {g: summarize([x['metrics'] for x in rows[g]]) for g in rows if rows[g]}
    pairs = {}
    for left, right in PAIRS:
        if len(rows[left]) != 10 or len(rows[right]) != 10:
            continue
        left_by_fold = {x['fold']: x for x in rows[left]}
        right_by_fold = {x['fold']: x for x in rows[right]}
        values = {}
        for key in METRICS:
            differences = [left_by_fold[f]['metrics'][key] - right_by_fold[f]['metrics'][key] for f in range(10)]
            values[key] = {
                'mean': float(np.mean(differences)),
                'sample_sd': float(np.std(differences, ddof=1)),
                'improved_folds': int(sum(x > 1e-12 for x in differences)),
                'ties_folds': int(sum(abs(x) <= 1e-12 for x in differences)),
                'fold_differences': differences,
            }
        pairs[f'{left}-{right}'] = {'metrics': values, 'paired_by_fold': True}

    return {'rows': rows, 'metrics_rows': metrics_rows, 'oof_rows': oof_rows,
            'fold_summary': fold_summary, 'pairs': pairs, 'artifacts': artifacts,
            'gate_records': gate_records, 'resource': resource, 'completed_tasks': total,
            'context': ctx}


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {'n': 0, 'mean': None, 'sample_sd': None, 'nonzero_fraction': None}
    return {'n': int(len(values)), 'mean': float(values.mean()),
            'sample_sd': float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            'nonzero_fraction': float(np.mean(values > 0.0))}


def mechanism_summary(data):
    out = {}
    for group, values in data['rows'].items():
        if not values:
            continue
        epoch_records = data['gate_records'][group]
        late = [x for x in epoch_records if x['epoch'] >= 9]
        spec = run.CFG['groups'][group]
        info = {
            'site_gate_enabled': spec['site_gain_gate'],
            'fusion_gradient_gate_enabled': spec['fusion_gradient_gate'],
            'cross_site_gate': distribution([x['r_site_mean'] for x in late]) if spec['site_gain_gate'] else None,
            'cross_site_nonzero_fraction': (distribution([x['r_site_nonzero_fraction'] for x in late])
                                            if spec['site_gain_gate'] else None),
            'fusion_gradient_gate': distribution([x['r_grad_mean'] for x in late]) if spec['fusion_gradient_gate'] else None,
            'fusion_gradient_nonzero_fraction': (distribution([x['r_grad_nonzero_fraction'] for x in late])
                                                 if spec['fusion_gradient_gate'] else None),
            'combined_gate': distribution([x['r_combined_mean'] for x in late]) if late else None,
            'combined_nonzero_fraction': (distribution([x['r_combined_nonzero_fraction'] for x in late])
                                          if late else None),
            'valid_auxiliary_fraction': distribution([x['valid_fraction'] for x in late]) if late else None,
            'gated_auxiliary_z_gradient_ratio': distribution(
                [x['gated_weighted_z_gradient_ratio'] for v in values for x in v['epochs'] if x['epoch'] >= 9]),
            'reference_auxiliary_z_gradient_ratio': distribution(
                [x['weighted_z_gradient_ratio'] for v in values for x in v['epochs'] if x['epoch'] >= 9]),
            'image_parameter_gradient_ratio': distribution(
                [x['image_parameter_gradients']['ratio'] for v in values for x in v['epochs']
                 if x['epoch'] >= 9 and 'image_parameter_gradients' in x]),
            'best_epoch': distribution([v['best_epoch'] for v in values]),
            'training_seconds': distribution([v['training_seconds'] for v in values]),
        }
        if spec['site_gain_gate']:
            info['gain_ab_positive_fraction'] = distribution([x['gain_ab_positive_fraction'] for x in late])
            info['gain_ba_positive_fraction'] = distribution([x['gain_ba_positive_fraction'] for x in late])
            info['site_support_fraction'] = distribution([x['site_valid_fraction'] for x in late])
        if group == 'F5':
            prep = [v['prototype_preparation'] for v in values]
            info['shuffle'] = {
                'permutation_sha256': {str(v['fold']): v['prototype_preparation']['shuffle_sha256'] for v in values},
                'changed_fraction_by_fold': {str(v['fold']): v['prototype_preparation']['shuffle_changed_fraction'] for v in values},
                'changed_fraction': distribution([p['shuffle_changed_fraction'] for p in prep]),
                'moved_fraction': distribution([p['shuffle_moved_fraction'] for p in prep]),
                'singleton_subjects_by_fold': {str(v['fold']): v['prototype_preparation']['singleton_subjects'] for v in values},
                'condition_changed_all_folds': all(p['shuffle_changed_fraction'] > 0 for p in prep),
            }
            assert info['shuffle']['condition_changed_all_folds']
        out[group] = info
    return out


def research_judgment(formal, pairs, mechanisms):
    if 'F4' not in formal or not all(k in pairs for k in ('F4-F1', 'F4-F0', 'F4-F5')):
        return 'PARTIAL'
    f4, f1, f0, f5 = (formal[g] for g in ('F4', 'F1', 'F0', 'F5'))
    d41, d40, d45 = (pairs[k]['metrics'] for k in ('F4-F1', 'F4-F0', 'F4-F5'))
    active = (mechanisms['F4']['cross_site_gate']['nonzero_fraction'] > 0
              and mechanisms['F4']['fusion_gradient_gate']['nonzero_fraction'] > 0)
    if not active:
        return 'NO-GO'
    if (d41['ACC']['mean'] > 0 and d41['AUC']['mean'] > 0
            and d40['ACC']['mean'] > 0 and d40['AUC']['mean'] > 0
            and d45['ACC']['mean'] > 0 and d45['AUC']['mean'] > 0
            and f4['SPE']['mean'] >= f0['SPE']['mean']
            and d41['ACC']['mean'] >= pairs['F2-F1']['metrics']['ACC']['mean']
            and d41['AUC']['mean'] >= pairs['F2-F1']['metrics']['AUC']['mean']
            and d41['ACC']['mean'] >= pairs['F3-F1']['metrics']['ACC']['mean']
            and d41['AUC']['mean'] >= pairs['F3-F1']['metrics']['AUC']['mean']):
        return 'GO'
    if any((pairs[k]['metrics']['ACC']['mean'] > 0 or pairs[k]['metrics']['AUC']['mean'] > 0)
           for k in ('F2-F1', 'F3-F1', 'F4-F1')):
        return 'MIXED'
    return 'NO-GO'


def report_text(summary):
    lines = [
        '# RCPDA_20261020 正式实验报告', '',
        f"状态：{summary['status']}；正式任务 {summary['completed_tasks']}/60；科研判断：**{summary['judgment']}**。", '',
        '## 研究问题与边界', '',
        '原始 D4 用年龄×性别条件化疾病原型对影像分支表示做训练期辅助监督。本轮检验两个可靠性问题：条件原型是否在互斥站点半区上比全局类别原型有更低的类别均衡风险，以及影像原型梯度是否与最终多模态分类 CE 的影像表示梯度相容。门控只调节原始 D4 的逐样本辅助 CE。',
        '既有历史发现显示，原始 D4 与重复实验受随机轨迹影响；同方向原型不能单独说明条件原型更有判别力。RCPDA 不预设 F4 最优，也不以测试集选择模型、阈值或超参数。',
        '论文方法与本地根源码核对：[MM-GTUNets 论文](https://arxiv.org/html/2406.14455v3)，[指定 GitHub 仓库](https://github.com/tangaii/MM)。本地仓库版本另保留了已记录的 PPU 兼容实现；本轮不改根模型。', '',
        '## 原模型接入与复现', '',
        '调用链为原始 RFE 影像特征及真实非影像输入 → Imaging GTUNet → Non-imaging GTUNet → 原 Multi-modal Attention → 原 Classifier → 原生 CE + Graph Loss。通过 `img_unet` forward hook 取得 `z_img=[871,16]`。辅助目标只从该次影像输出接收梯度；无新增可训练参数，验证和测试不调用辅助模块。',
        '只读复用 DRPA_20261014 的 seed 911 十折 `splits.pt`、`train_start.pt`、selected features、RFE 支持索引、VAE、完整 RNG 与 fingerprints。保持 ASD=0、TD=1，Adam lr=1e-4、weight decay=5e-4、最多500 epoch、patience=100、原训练/验证 CE+Graph、严格下降选模规则和随机前向行为。每 epoch 一次训练 forward、一次验证 forward；测试恢复 best checkpoint 后使用训练结束 RNG，以 grad-enabled 模式执行一次 forward。',
        '历史限制继续保留：RFE 使用原历史预处理、人口图为871节点传导式图，VAE随机采样，eval 路径仍有随机性；同站点十折不证明未见站点泛化。', '',
        '## 固定组别', '',
        '|组别|训练目标|', '|---|---|',
        '|F0|原生 MM-GTUNets，无原型辅助损失|',
        '|F1|原始 D4 年龄×性别疾病原型|',
        '|F2|D4 + 跨站点判别增益门控|',
        '|F3|D4 + 融合分类梯度一致性门控|',
        '|F4|两种门控的完整 RCPDA|',
        '|F5|F4，但辅助分支年龄与性别在 outer-train 同站点联合打乱|', '',
        'F1 保留历史 D4 原型、beta、tau、辅助 CE、梯度和训练接入；预检逐项对照历史 PCDCA D4。F2/F4/F5 的站点门控只在 outer-train 内进行 A→B 与 B→A 两向类别平衡风险比较；两边必须都是互斥完整站点集合。F3/F4/F5 的梯度门控比较最终融合分类 CE 与原始 D4 辅助损失对同一 `z_img` 的梯度。门控和参考量均 stop-gradient。',
        '共同 lambda 按未门控原始 D4 梯度计算。门控辅助损失使用 `sum(r_i * CE_i) / sum(valid_i)` 固定分母；不会按可靠性权重和重新放大少数样本。F5 仅改变辅助原型 kernel，真实人口输入、图、标签、切分与 RNG 锚点不变。', '',
        '## 四指标结果', '',
        '每个指标先按十折计算均值和样本标准差（ddof=1）。表格单位为百分比；该单 seed 的折间 SD 不是独立队列置信区间。', '',
        '|组别|ACC (%)|SEN-ASD (%)|SPE-TD (%)|probability AUC (%)|', '|---|---:|---:|---:|---:|',
    ]
    for group in run.CFG['groups']:
        x = summary['formal'].get(group)
        if not x:
            lines.append(f"|{group}|未完成|未完成|未完成|未完成|")
            continue
        vals = [f"{100*x[k]['mean']:.2f} ± {100*x[k]['sample_sd']:.2f}" for k in METRICS]
        lines.append('|' + group + '|' + '|'.join(vals) + '|')
    lines += ['', '## 配对折差', '',
              '差值为同一折左组减右组，单位百分点；“改善折数”仅描述折向，不是独立样本显著性检验。', '',
              '|比较|ΔACC|ΔSEN-ASD|ΔSPE-TD|Δprob-AUC|', '|---|---:|---:|---:|---:|']
    for name, value in summary['paired_differences'].items():
        m = value['metrics']
        diffs = [f"{100*m[k]['mean']:+.3f} ({m[k]['improved_folds']}/10折+)" for k in METRICS]
        lines.append('|' + name + '|' + '|'.join(diffs) + '|')
    lines += ['', '## 历史参照与门控审计', '',
              '提示中给定的历史单次 D4 seed911 仅用于描述随机轨迹漂移，不替代本轮新鲜 F1：ACC 82.5588%、SEN 79.9268%、SPE 84.8427%、AUC 88.6897%。SCPDA_E4_REPEAT911_20261020 是精确 SCPDA 路径缺失时读取的已有归档；其历史 E4 三次重复参照为 ACC 82.6737%、SEN 80.9207%、SPE 84.1952%、AUC 88.1900%，不得称作本轮同批配对结果。', '',
              '|组别|站点门控均值/非零比例|梯度门控均值/非零比例|组合门控均值/非零比例|门控后 z 梯度缩放比例|影像参数梯度缩放比例|',
              '|---|---:|---:|---:|---:|---:|']
    for group, value in summary['mechanism_audit'].items():
        site = value['cross_site_gate']
        grad = value['fusion_gradient_gate']
        comb = value['combined_gate']
        def fmt(dist, key='mean'):
            return '未启用' if dist is None else f"{dist[key]:.4f}"
        site_text = '未启用' if site is None else f"{fmt(site)}/{fmt(value['cross_site_nonzero_fraction'])}"
        grad_text = '未启用' if grad is None else f"{fmt(grad)}/{fmt(value['fusion_gradient_nonzero_fraction'])}"
        combined_text = '无' if comb is None else f"{fmt(comb)}/{fmt(value['combined_nonzero_fraction'])}"
        lines.append(f"|{group}|{site_text}|{grad_text}|{combined_text}|"
                     f"{fmt(value['gated_auxiliary_z_gradient_ratio'])}|{fmt(value['image_parameter_gradient_ratio'])}|")
    if 'F5' in summary['mechanism_audit']:
        shuffle = summary['mechanism_audit']['F5']['shuffle']
        lines += ['', f"F5 实际改变表型条件比例：均值 {100*shuffle['changed_fraction']['mean']:.2f}%；每折排列 SHA256、改变比例及单例受试者见 `summary.json`。所有十折均发生实际改变：{shuffle['condition_changed_all_folds']}。F4/F5 的非打乱配置字段相同。"]
    lines += ['', '## 科学解释', '']
    if 'F4-F1' in summary['paired_differences']:
        p41 = summary['paired_differences']['F4-F1']['metrics']
        p40 = summary['paired_differences']['F4-F0']['metrics']
        p45 = summary['paired_differences']['F4-F5']['metrics']
        lines += [f"F4 相对 F1 的 ACC/AUC 配对均差分别为 {100*p41['ACC']['mean']:+.3f}/{100*p41['AUC']['mean']:+.3f} 个百分点；相对 F0 为 {100*p40['ACC']['mean']:+.3f}/{100*p40['AUC']['mean']:+.3f}；相对 F5 为 {100*p45['ACC']['mean']:+.3f}/{100*p45['AUC']['mean']:+.3f}。",
                  '同时查看 SEN-ASD 与 SPE-TD 的变化，避免把单一阈值附近的 ACC 改善解释为概率排序改善。若 ACC 上升但 AUC 下降，不能据此宣称创新成功；若 F4 与 F5 接近，不主张正确表型具有独立贡献；若 F2 或 F3 优于 F4，不声称组合协同。']
    else:
        lines.append('正式组别尚未全部完成，现有差值不用于结论。')
    lines += ['', '## 判断、执行与局限', '',
              f"研究判断：**{summary['judgment']}**。", '',
              f"完成 {summary['completed_tasks']}/60 个任务；GPU0 {summary['execution']['gpu_tasks']['0']}/30，GPU1 {summary['execution']['gpu_tasks']['1']}/30。累计 worker 训练 {summary['execution']['training_seconds']:.1f} 秒；单 worker 峰值保留显存 {summary['execution']['peak_reserved_gib']:.2f} GiB。未训练 VAE、未拟合 RFE、新增可训练参数为0。", '',
              '结论只属于 seed911 的单 seed 候选结果。原生 PPU 长轨迹、VAE/eval 随机性及 checkpoint 轨迹漂移会影响微小差异；十个折相互关联。保留历史 RFE 预处理和转导式人口图等限制，不据此宣称无泄漏、已验证跨站点泛化或达到临床部署要求。', '',
              f"代码/配置指纹 `{summary['code_sha256']}`；历史输入保护审计 `{summary['protected_verification']['unchanged']}`（{summary['protected_verification']['files']} 文件）；完整 OOF 覆盖每组871人各一次。", '',
              f"产物目录：`{run.BASE}`。"]
    return '\n'.join(lines) + '\n'


def append_handoff(report, audit):
    path = run.ROOT / 'handoff.md'
    text = ('\n\n' + START_MARK + '\n\n## Experiment: RCPDA_20261020\n\n' + report + '\n' + END_MARK + '\n').encode()
    with path.open('r+b') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        before = f.read()
        if START_MARK.encode() in before:
            assert END_MARK.encode() in before
            return {'appended': True, 'idempotent_skip': True, 'after_sha256': run.sha_file(path)}
        assert len(before) == audit['handoff_before_bytes']
        assert hashlib.sha256(before).hexdigest() == audit['handoff_before_sha256']
        f.seek(0, os.SEEK_END)
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
        f.seek(0)
        after = f.read()
        assert after[:len(before)] == before and after[len(before):] == text
        assert hashlib.sha256(after[:len(before)]).hexdigest() == audit['handoff_before_sha256']
        assert after.count(START_MARK.encode()) == 1 and after.count(END_MARK.encode()) == 1
        return {'appended': True, 'before_sha256': audit['handoff_before_sha256'],
                'before_bytes_preserved': len(before), 'after_sha256': hashlib.sha256(after).hexdigest(),
                'marker': START_MARK, 'time': run.utc()}


def main(final=False, partial=False):
    audit = run.guard()
    if not audit.get('formal_authorized'):
        raise RuntimeError('Audit and preflight have not authorized this run')
    manifest = run.read_json(run.BASE / 'manifest.json')
    assert manifest['code_sha256'] == audit['code_sha256']
    data = inspect_results(audit, require_complete=final)
    formal = data['fold_summary']
    pairs = data['pairs']
    mechanisms = mechanism_summary(data)
    decision = research_judgment(formal, pairs, mechanisms)
    if decision == 'PARTIAL' and not partial:
        raise RuntimeError('Cannot finalize a report before all 60 tasks are verified')
    protected = run.verify_protected(audit) if hasattr(run, 'verify_protected') else None
    if protected is None:
        mismatches = [p for p, h in audit['protected_sha256'].items() if run.sha_file(p) != h]
        if mismatches:
            raise AssertionError(f'Protected historical inputs changed: {mismatches}')
        protected = {'files': len(audit['protected_sha256']), 'unchanged': True, 'time': run.utc()}
    tasks = {g: len(data['rows'][g]) for g in run.CFG['groups']}
    summary = {
        'experiment': run.CFG['experiment'], 'status': 'complete' if data['completed_tasks'] == 60 else 'partial',
        'completed_tasks': data['completed_tasks'], 'task_total': 60, 'group_counts': tasks,
        'gpu_task_counts': data['resource']['gpu_tasks'], 'formal': formal, 'paired_differences': pairs,
        'mechanism_audit': mechanisms, 'judgment': decision,
        'historical_references': {'D4_seed911': HISTORICAL_D4, 'E4_three_repeat_mean': HISTORICAL_E4,
                                  'use': 'descriptive only; fresh F1 is the same-batch causal comparator'},
        'artifact_sha256': data['artifacts'], 'code_sha256': audit['code_sha256'],
        'config_sha256': run.sha_file(run.BASE / 'config.json'),
        'input_context_sha256': audit['context_sha256'], 'input_protection': protected,
        'execution': {'gpu_tasks': data['resource']['gpu_tasks'],
                      'training_seconds': data['resource']['training_seconds'],
                      'peak_reserved_gib': data['resource']['peak_reserved_bytes'] / 1024**3,
                      'formal_neural_training': 'PPU only', 'RFE_fits': 0, 'VAE_pretrains': 0,
                      'new_trainable_parameters': 0, 'failures': audit.get('execution', {}).get('failures', [])},
        'OOF': {g: {'subjects': 871, 'each_once': len(data['rows'][g]) == 10} for g in data['rows'] if data['rows'][g]},
        'handoff': {'appended': False}, 'created_utc': run.utc(),
    }
    report = report_text(summary)
    write_csv(run.BASE / 'fold_metrics.csv', ['group', 'seed', 'fold', 'ACC', 'SEN', 'SPE', 'AUC'], data['metrics_rows'])
    write_csv(run.BASE / 'oof.csv', ['group', 'seed', 'fold', 'subject_id', 'y_true', 'y_pred', 'probability_asd'], data['oof_rows'])
    run.atomic_bytes(report.encode(), run.BASE / 'report.md')
    run.atomic_json(summary, run.BASE / 'summary.json')
    if final:
        if data['completed_tasks'] != 60 or any(tasks[g] != 10 for g in tasks):
            raise RuntimeError('Final handoff requires 60/60 validated task results')
        try:
            summary['handoff'] = append_handoff(report, audit)
        except Exception as exc:
            run.atomic_bytes(report.encode(), run.BASE / 'handoff_pending.md')
            summary['handoff'] = {'appended': False, 'error': repr(exc), 'pending_file': 'handoff_pending.md'}
            run.atomic_json(summary, run.BASE / 'summary.json')
            raise
        summary['final_verified_utc'] = run.utc()
        run.atomic_json(summary, run.BASE / 'summary.json')
    run.log('analysis_complete', status=summary['status'], tasks=data['completed_tasks'],
            judgment=decision, handoff=summary['handoff']['appended'])
    print(report, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    main(final=args.final, partial=args.partial)
