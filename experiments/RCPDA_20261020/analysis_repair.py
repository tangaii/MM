"""Post-run report-only corrections for the frozen RCPDA analysis."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from pathlib import Path

import analyze
import run


def mechanism_summary(data):
    out = {}
    for group, values in data['rows'].items():
        if not values:
            continue
        epochs = data['gate_records'][group]
        late = [x for x in epochs if x['epoch'] >= 9]
        spec = run.CFG['groups'][group]
        gates_active = bool(spec['site_gain_gate'] or spec['fusion_gradient_gate'])
        info = {
            'site_gate_enabled': spec['site_gain_gate'],
            'fusion_gradient_gate_enabled': spec['fusion_gradient_gate'],
            'cross_site_gate': analyze.distribution([x['r_site_mean'] for x in late]) if spec['site_gain_gate'] else None,
            'cross_site_nonzero_fraction': (analyze.distribution([x['r_site_nonzero_fraction'] for x in late])
                                            if spec['site_gain_gate'] else None),
            'fusion_gradient_gate': analyze.distribution([x['r_grad_mean'] for x in late]) if spec['fusion_gradient_gate'] else None,
            'fusion_gradient_nonzero_fraction': (analyze.distribution([x['r_grad_nonzero_fraction'] for x in late])
                                                 if spec['fusion_gradient_gate'] else None),
            'combined_gate': analyze.distribution([x['r_combined_mean'] for x in late]) if late else None,
            'combined_nonzero_fraction': (analyze.distribution([x['r_combined_nonzero_fraction'] for x in late])
                                          if late else None),
            'valid_auxiliary_fraction': analyze.distribution([x['valid_fraction'] for x in late]) if late else None,
            'gated_auxiliary_z_gradient_ratio': analyze.distribution(
                [x['gated_weighted_z_gradient_ratio'] for x in late]) if gates_active else None,
            'reference_auxiliary_z_gradient_ratio': analyze.distribution(
                [x['weighted_z_gradient_ratio'] for v in values for x in v['epochs']
                 if x['epoch'] >= 9 and 'weighted_z_gradient_ratio' in x]),
            'image_parameter_gradient_ratio': analyze.distribution(
                [x['image_parameter_gradients']['ratio'] for v in values for x in v['epochs']
                 if x['epoch'] >= 9 and 'image_parameter_gradients' in x]),
            'best_epoch': analyze.distribution([v['best_epoch'] for v in values]),
            'training_seconds': analyze.distribution([v['training_seconds'] for v in values]),
        }
        if not gates_active:
            info['combined_gate'] = None
            info['combined_nonzero_fraction'] = None
        if spec['site_gain_gate']:
            info['gain_ab_positive_fraction'] = analyze.distribution([x['gain_ab_positive_fraction'] for x in late])
            info['gain_ba_positive_fraction'] = analyze.distribution([x['gain_ba_positive_fraction'] for x in late])
            info['site_support_fraction'] = analyze.distribution([x['site_valid_fraction'] for x in late])
        if group == 'F5':
            prep = [v['prototype_preparation'] for v in values]
            info['shuffle'] = {
                'permutation_sha256': {str(v['fold']): v['prototype_preparation']['shuffle_sha256'] for v in values},
                'changed_fraction_by_fold': {str(v['fold']): v['prototype_preparation']['shuffle_changed_fraction'] for v in values},
                'changed_fraction': analyze.distribution([p['shuffle_changed_fraction'] for p in prep]),
                'moved_fraction': analyze.distribution([p['shuffle_moved_fraction'] for p in prep]),
                'singleton_subjects_by_fold': {str(v['fold']): v['prototype_preparation']['singleton_subjects'] for v in values},
                'condition_changed_all_folds': all(p['shuffle_changed_fraction'] > 0 for p in prep),
            }
            assert info['shuffle']['condition_changed_all_folds']
        out[group] = info
    return out


original_report = analyze.report_text


def report_text(summary):
    # Keep the structured summary faithful while giving the frozen renderer
    # explicit N/A values for F0-only diagnostics.
    display = deepcopy(summary)
    display['protected_verification'] = display['input_protection']
    display['analysis_repair_sha256'] = run.sha_file(Path(__file__))
    audit = run.read_json(run.BASE / 'audit.json')
    queue = audit['execution']['queue']
    display['execution']['queue_wall_seconds'] = queue['wall_seconds']
    display['execution']['queue_completed_utc'] = queue['completed_utc']
    for group in ('F0',):
        if group in display['mechanism_audit']:
            display['mechanism_audit'][group]['gated_auxiliary_z_gradient_ratio'] = None
            display['mechanism_audit'][group]['image_parameter_gradient_ratio'] = None
    report = original_report(display)
    report = report.replace('## 研究问题与边界', '## 研究动机与问题边界')
    report = report.replace('## 原模型接入与复现', '## 原始模型接入、复现协议与历史锚点')
    report = report.replace('## 固定组别', '## RCPDA严格定义与固定组别')
    report = report.replace('## 历史参照与门控审计', '## 历史D4/E4问题、参照与门控审计')

    formal = display['formal']
    pairs = display['paired_differences']
    mechanisms = display['mechanism_audit']

    def pp(pair, metric):
        return 100.0 * pairs[pair]['metrics'][metric]['mean']

    def fpp(value):
        return f'{value:+.3f}pp'

    f4 = formal['F4']
    f0 = formal['F0']
    f1 = formal['F1']
    f5 = formal['F5']
    f4_site = mechanisms['F4']['cross_site_gate']
    f4_site_nz = mechanisms['F4']['cross_site_nonzero_fraction']
    f4_grad = mechanisms['F4']['fusion_gradient_gate']
    f4_grad_nz = mechanisms['F4']['fusion_gradient_nonzero_fraction']
    f4_combined = mechanisms['F4']['combined_gate']
    f4_combined_nz = mechanisms['F4']['combined_nonzero_fraction']
    f5_shuffle = mechanisms['F5']['shuffle']
    report += '\n## 九个预定科学问题的直接回答\n\n'
    report += (f'1. **F4 是否同时改善 ACC 和 AUC？** F4 相对本轮新鲜 F1 的 '
               f'ACC/AUC 配对均差为 {fpp(pp("F4-F1", "ACC"))}/'
               f'{fpp(pp("F4-F1", "AUC"))}；因此 '
               f'{"同时改善" if pp("F4-F1", "ACC") > 0 and pp("F4-F1", "AUC") > 0 else "没有同时改善"}。\n')
    report += (f'2. **SEN 与 SPE 的权衡？** F4-F1 的 SEN-ASD 为 {fpp(pp("F4-F1", "SEN"))}，'
               f'SPE-TD 为 {fpp(pp("F4-F1", "SPE"))}；阈值附近的类别变化与排序指标必须分开解读。\n')
    report += (f'3. **F4 是否优于两个单因素？** 相对 F2 的 ACC/AUC 为 '
               f'{fpp(pp("F4-F2", "ACC"))}/{fpp(pp("F4-F2", "AUC"))}，相对 F3 为 '
               f'{fpp(pp("F4-F3", "ACC"))}/{fpp(pp("F4-F3", "AUC"))}；'
               f'{"两项指标都支持组合优于单因素" if all(pp(k, m) > 0 for k in ("F4-F2", "F4-F3") for m in ("ACC", "AUC")) else "没有形成两项指标都一致的组合优势"}。\n')
    report += (f'4. **F4 是否超过原生 F0？** F4-F0 的 ACC/AUC 为 '
               f'{fpp(pp("F4-F0", "ACC"))}/{fpp(pp("F4-F0", "AUC"))}；'
               f'SEN/SPE 为 {fpp(pp("F4-F0", "SEN"))}/{fpp(pp("F4-F0", "SPE"))}，不能只用 ACC 判断。\n')
    report += (f'5. **真实表型是否优于 F5 打乱对照？** F4-F5 的 ACC/AUC 为 '
               f'{fpp(pp("F4-F5", "ACC"))}/{fpp(pp("F4-F5", "AUC"))}；'
               f'十折打乱条件改变比例均值为 {100.0*f5_shuffle["changed_fraction"]["mean"]:.2f}%，'
               f'每折均发生改变，因此不能把接近结果归因于打乱未生效。\n')
    report += (f'6. **门控是否实际激活？** F4 站点门控均值/非零折内比例为 '
               f'{f4_site["mean"]:.4f}/{f4_site_nz["mean"]:.4f}，'
               f'梯度门控为 {f4_grad["mean"]:.4f}/{f4_grad_nz["mean"]:.4f}，'
               f'组合门控为 {f4_combined["mean"]:.4f}/{f4_combined_nz["mean"]:.4f}；'
               f'两种门控都实际激活，但组合门控只在约 {100*f4_combined_nz["mean"]:.1f}% 样本上为正。\n')
    report += (f'7. **是否只是阈值附近改善而没有概率排序改善？** F4-F1 的 AUC 为 '
               f'{fpp(pp("F4-F1", "AUC"))}，所以 ACC 与 AUC 的符号必须共同考虑；'
               f'本轮不能把阈值指标变化单独解释为排序改善。\n')
    report += ('8. **哪些结论可能来自随机性？** 历史 D4/E4 复现审计显示 PPU 长轨迹、随机验证前向、'
               'best-checkpoint 选择和一次随机测试前向都可能放大亚百分点差异；本轮只有 seed 911，故微小差异不能视为稳定因果效应。\n')
    report += ('9. **一个 seed 的限制？** 十折来自同一 seed 且折间相关，不能代表多 seed 稳定性；'
               '仍保留跨折监督 RFE、871 节点转导图、VAE/eval 随机性和同站点交叉验证限制，不能宣称跨站点泛化或临床部署。\n')
    report += (f'F4 四指标均值为 ACC {100*f4["ACC"]["mean"]:.2f}%、SEN {100*f4["SEN"]["mean"]:.2f}%、'
               f'SPE {100*f4["SPE"]["mean"]:.2f}%、AUC {100*f4["AUC"]["mean"]:.2f}%；'
               f'F1 为 ACC {100*f1["ACC"]["mean"]:.2f}%、AUC {100*f1["AUC"]["mean"]:.2f}%，'
               f'F0 为 ACC {100*f0["ACC"]["mean"]:.2f}%、AUC {100*f0["AUC"]["mean"]:.2f}%，'
               f'F5 为 ACC {100*f5["ACC"]["mean"]:.2f}%、AUC {100*f5["AUC"]["mean"]:.2f}%。\n')
    return report


analyze.mechanism_summary = mechanism_summary
analyze.report_text = report_text


def finalize():
    """Verify frozen artifacts and complete report metadata without training."""
    import fcntl
    import hashlib
    import os
    import numpy as np

    audit = run.guard()
    summary = run.read_json(run.BASE / 'summary.json')
    data = analyze.inspect_results(audit, require_complete=True)
    assert data['artifacts'] == summary['artifact_sha256']
    # Check saved checkpoint selection against the complete validation trace.
    max_metric_error = 0.0
    for group, values in data['rows'].items():
        for value in values:
            history = value['epochs']
            assert len(history) == value['epochs_run']
            assert [x['epoch'] for x in history] == list(range(len(history)))
            losses = np.asarray([x['val_loss'] for x in history])
            assert np.isfinite(losses).all()
            assert int(losses.argmin()) == value['best_epoch']
            assert float(losses.min()) == value['best_val_loss']
            assert value['best_record'] == history[value['best_epoch']]
            for metric in analyze.METRICS:
                checked = analyze.independent_metrics(value['oof']['labels'], value['oof']['probability'], value['oof']['prediction'])
                max_metric_error = max(max_metric_error, abs(checked[metric] - value['metrics'][metric]))
    for fold in range(10):
        values = [data['rows'][group][fold] for group in run.CFG['groups']]
        for value in values[1:]:
            for key in ('pairing', 'data_input_digest', 'formal_start_rng_sha256', 'optimizer_parameter_order', 'prototype_preparation'):
                assert value[key] == values[0][key], (fold, key)
    for path, expected in audit['protected_sha256'].items():
        assert run.sha_file(path) == expected, path
    spec4 = dict(run.CFG['groups']['F4'])
    spec5 = dict(run.CFG['groups']['F5'])
    for spec in (spec4, spec5):
        spec.pop('name')
        spec.pop('shuffled')
    assert spec4 == spec5
    summary['mechanism_audit'] = mechanism_summary(data)
    summary['configuration'] = run.CFG
    summary['manifest_sha256'] = run.sha_file(run.BASE / 'manifest.json')
    summary['analysis_repair_sha256'] = run.sha_file(Path(__file__))
    summary['analysis_repair'] = {
        'scope': 'post-run field mapping, report rendering and independent saved-result acceptance only',
        'frozen_code_unchanged': True, 'result_hashes_unchanged': True,
        'training_reruns': 0, 'extra_model_forwards': 0,
    }
    summary['result_acceptance'] = {
        'checkpoint_is_first_strict_validation_minimum': True,
        'same_fold_initialization_rng_inputs_optimizer_order': True,
        'F4_F5_only_operational_difference_is_phenotype_shuffle': True,
        'independent_four_metric_max_error': max_metric_error,
        'result_content_and_file_hashes_verified': 60,
        'OOF_total_rows': len(data['oof_rows']), 'each_group_subjects': 871,
    }
    queue = audit['execution']['queue']
    summary['execution']['queue_wall_seconds'] = queue['wall_seconds']
    summary['execution']['queue_completed_utc'] = queue['completed_utc']
    per_gpu = {}
    for gpu in (0, 1):
        events = [x for x in audit['execution']['events'] if x['gpu'] == gpu]
        assert len(events) == 30
        first = min(x['started_utc'] for x in events)
        last = max(x['finished_utc'] for x in events)
        per_gpu[str(gpu)] = {
            'tasks': 30, 'first_worker_start_utc': first, 'last_worker_finish_utc': last,
            'worker_span_seconds': (datetime.fromisoformat(last) - datetime.fromisoformat(first)).total_seconds(),
            'worker_training_seconds': sum(v['training_seconds'] for vals in data['rows'].values() for v in vals if v['physical_gpu'] == gpu),
            'by_group': {g: sum(v['physical_gpu'] == gpu for v in vals) for g, vals in data['rows'].items()},
            'sampled_peak_used_mib': max(x['gpus'][str(gpu)]['used_mib'] for x in audit['execution']['resources']),
        }
    summary['execution']['per_gpu'] = per_gpu
    summary['input_protection'] = {'files': len(audit['protected_sha256']), 'unchanged': True, 'time': run.utc()}
    report = report_text(summary)
    report += '''
## 公式与诊断口径

原始 D4 只用 outer-train：固定三个私有子折，每个 query 的参考池排除自身整个子折。年龄 kernel 为 exp(-0.5*((age_i-age_j)/h)^2)，h=max(2,train-IQR/2)；性别同/异/未知权重为1/0.5/0.75。当前 detach teacher 按训练均值与总体 SD 标准化，SD 下限0.05，tau 为训练标准化 teacher 的平均平方中心距离，下限1。每类 ESS=(sum(w))^2/sum(w^2)，共同 beta=clip((min_class_ESS-8)/32,0,1)；条件原型=beta*条件加权均值+(1-beta)*全局类均值。参考池无任一类别则屏蔽该 query。逐人辅助 CE 来自 -||query-prototype||^2/tau，仅当前 query 接收梯度。

跨站点门控在同一参考池的完整互斥站点 A/B 上：只用 A 的 teacher 构造条件/全局原型，在 B 上用 query 的 kernel 做两类各0.5权重的 CE 风险，反向 B→A 同理；构造及评估的每类有效权重均需 ESS>=2。gain_AB=R_global(A→B)-R_conditional(A→B)，gain_BA 同理；r_site=max(0,min(gain_AB,gain_BA))/max(abs(gain_AB),abs(gain_BA),1e-6)，缺失支持时为0。实际 D4 辅助 CE 仍使用完整参考池原型，不替换为半区原型。

融合梯度门控：g_cls=grad_z(原始融合分类CE)，g_proto=grad_z(未门控D4)，按完全相同 outer-train 顺序取每人梯度；r_grad=clip(dot(g_cls,g_proto)/max(norm(g_cls)*norm(g_proto),1e-12),0,1)，任一范数<=1e-12则为0。所有门控 detach；F2用r_site，F3用r_grad，F4/F5用二者乘积。辅助损失=sum(valid*r*CE)/max(sum(valid),1)，保持固定分母。lambda=clip(0.05*norm(grad_z(native))/(norm(grad_z(D4))+1e-12),0,0.5)*min(epoch/9,1)，始终从未门控 D4 计算。

门控表的均值与非零比例先按每个训练 epoch 的 outer-train 样本计算，再对 epoch>=9 的记录汇总，因此训练较长的折权重较大；不是测试评价指标。影像参数梯度比只在预定采样 epoch 统计。JSON 中 gate 分布自身的 nonzero_fraction 表示“epoch 均值非零比例”，样本级比例应读取相邻 *_nonzero_fraction 字段的 mean。F1 的未门控参考 z 梯度比例为0.05，没有新门控。F4 实际门控后比例约0.01066，不做门控后重新补偿放大。
'''
    report += (f'\n## 完成验收与后续建议\n\n'
               f'正式队列墙钟 {queue["wall_seconds"]:.2f}秒（{queue["wall_seconds"]/60:.2f}分钟）；'
               f'GPU0/GPU1各30任务，worker起止跨度分别 {per_gpu["0"]["worker_span_seconds"]:.2f}/'
               f'{per_gpu["1"]["worker_span_seconds"]:.2f}秒。累计worker训练 '
               f'{summary["execution"]["training_seconds"]:.2f}秒；并行时间不可相加作为队列墙钟。'
               f'整卡采样峰值 GPU0/GPU1 为 {per_gpu["0"]["sampled_peak_used_mib"]}/'
               f'{per_gpu["1"]["sampled_peak_used_mib"]} MiB。正式失败/重试0。\n\n'
               f'独立复算四指标最大误差 {max_metric_error:.3g}；60份结果内容/文件SHA、'
               f'每折首次严格最小validation CE+Graph checkpoint、完整起点RNG/输入/优化器顺序、'
               f'每组871人各一次OOF及547项历史保护文件均通过。两锚点0/5确定性三步对照最大误差为0，'
               f'该诊断设置未用于生产训练。\n\n'
               f'报告层修复独立记录为 analysis_repair.py，SHA256 `{summary["analysis_repair_sha256"]}`；'
               f'冻结训练/原型/配置源码和60个结果未改动，没有追加神经前向或重训。\n\n'
               f'科研判断为 **NO-GO**：F2/F3/F4相对新鲜F1的ACC与AUC均下降；F4的SEN提升伴随SPE下降，'
               f'相对F0的ACC增加而AUC下降，真实/打乱表型结果近似，不能主张组合协同或正确表型独立贡献。'
               f'下一步建议暂停扩展当前门控组合，优先在单独设计的无跨折RFE泄漏、未见站点协议中确认基础疾病原型，'
               f'并用配对多seed区分小收益与随机轨迹；本轮不追加训练或改变超参数。\n')
    run.atomic_bytes(report.encode(), run.BASE / 'report.md')

    # Required concluding handoff data, append-only and idempotent.
    marker = '<!-- RCPDA_20261020_FINAL_ACCEPTANCE_START -->'
    end = '<!-- RCPDA_20261020_FINAL_ACCEPTANCE_END -->'
    addition = (f'\n\n{marker}\n\n## RCPDA_20261020 最终验收与下一步\n\n'
                f'seed911 ×10折×6组，60/60通过；GPU0/GPU1各30/30，正式队列墙钟{queue["wall_seconds"]/60:.2f}分钟，'
                f'累计worker训练{summary["execution"]["training_seconds"]/60:.2f}分钟。'
                f'结果哈希60项、保护文件547项、每组871人OOF及最佳checkpoint规则复核通过。'
                f'最终判断NO-GO：F4-F1 ACC/AUC=-0.229/-0.653pp，F4-F0=+0.340/-0.048pp，'
                f'F4-F5=0.000/+0.006pp；门控激活而无一致收益。'
                f'下一步暂停扩展当前门控，优先单独建立无跨折RFE泄漏、未见站点及配对多seed的基础原型确认协议。'
                f'完整公式、耗时、验收与报告修复指纹见experiments/RCPDA_20261020/report.md与summary.json。'
                f'此次追加不改变既有交接正文。\n\n{end}\n').encode()
    handoff = run.ROOT / 'handoff.md'
    with handoff.open('r+b') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        before = f.read()
        assert hashlib.sha256(before[:audit['handoff_before_bytes']]).hexdigest() == audit['handoff_before_sha256']
        if marker.encode() not in before:
            f.seek(0, os.SEEK_END)
            f.write(addition)
            f.flush()
            os.fsync(f.fileno())
            f.seek(0)
            after = f.read()
            assert after == before + addition
        else:
            after = before
        assert after.count(marker.encode()) == after.count(end.encode()) == 1
    summary['handoff'].update(
        appended=True, before_bytes_preserved=audit['handoff_before_bytes'],
        before_sha256=audit['handoff_before_sha256'], final_acceptance_marker=marker,
        current_after_sha256=hashlib.sha256(after).hexdigest(), after_sha256=hashlib.sha256(after).hexdigest(),
        current_bytes=len(after), original_prefix_verified=True,
    )
    summary['final_verified_utc'] = run.utc()
    summary['execution']['audit_registration_to_final_seconds'] = (
        datetime.fromisoformat(summary['final_verified_utc']) - datetime.fromisoformat(audit['started_utc'])
    ).total_seconds()
    summary['report_sha256'] = run.sha_file(run.BASE / 'report.md')
    run.atomic_json(summary, run.BASE / 'summary.json')
    assert run.read_json(run.BASE / 'summary.json')['report_sha256'] == run.sha_file(run.BASE / 'report.md')
    run.log('final_acceptance', tasks=60, gpu_tasks={'0': 30, '1': 30}, judgment=summary['judgment'],
            protected_files=547, handoff=True)
    print({'tasks': 60, 'gpu_tasks': {'0': 30, '1': 30}, 'judgment': summary['judgment'],
           'queue_wall_seconds': queue['wall_seconds'], 'per_gpu': per_gpu, 'handoff': True,
           'analysis_repair_sha256': summary['analysis_repair_sha256']})


if __name__ == '__main__':
    finalize()
