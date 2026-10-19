"""Post-training acceptance without changing the registered training snapshot."""
from __future__ import annotations

import argparse
import collections
import datetime
import fcntl
import importlib.metadata
from pathlib import Path

import numpy as np

import run
import analyze

ORIGINAL_INDEPENDENT = analyze.independent
ORIGINAL_REPORT = analyze.make_report


def independent64(y, probability, prediction):
    # Match the stored metric's numerical domain, not a relaxed tolerance.
    return ORIGINAL_INDEPENDENT(np.asarray(y), np.asarray(probability, np.float64), np.asarray(prediction))


def collect(a):
    rows = {g: [run.valid_result(g, s, f, a, True) for s in run.CFG['seeds'] for f in range(10)]
            for g in run.CFG['groups']}
    ctx = run.load_torch(run.OLD/'artifacts/context.pt')
    ids = np.loadtxt(run.ROOT/run.CFG['data_folder']/'id.txt', dtype=str).reshape(-1).tolist()
    assert ids == list(ctx['ids']) and len(set(ids)) == 871
    run.metadata(ctx)
    for s in run.CFG['seeds']:
        for f in range(10):
            paired = [next(v for v in rows[g] if (v['seed'], v['fold']) == (s, f)) for g in rows]
            assert len({v['data_input_digest'] for v in paired}) == 1
            assert len({run.digest(v['prototype_preparation']) for v in paired}) == 1
            assert len({run.digest(v['formal_start_rng_sha256']) for v in paired}) == 1
            assert len({tuple(v['optimizer_parameter_order']) for v in paired}) == 1
    raw_error, error, frozen_error, ties = 0., 0., 0., 0
    for g, values in rows.items():
        for s in run.CFG['seeds']:
            coverage = np.zeros(871, int)
            for v in (v for v in values if v['seed'] == s):
                o = v['oof']; coverage[o['indices']] += 1
                assert np.array_equal(o['labels'], ctx['labels'][o['indices']])
                assert np.array_equal(o['prediction'], o['logits'].argmax(1))
                raw = ORIGINAL_INDEPENDENT(o['labels'], o['probability'], o['prediction'])
                corrected = independent64(o['labels'], o['probability'], o['prediction'])
                raw_error = max(raw_error, max(abs(raw[k]-v['metrics'][k]) for k in analyze.KEYS))
                error = max(error, max(abs(corrected[k]-v['metrics'][k]) for k in analyze.KEYS))
                ties += int(np.sum(o['prediction'] != np.where(o['probability'] >= .5, 0, 1)))
                if g in run.CFG['frozen_groups']:
                    assert len(v['frozen']['rows']) == 5
                    assert v['frozen']['auxiliary_not_used_in_inference']
                    for base, draw in zip(run.CFG['frozen_draw_bases'], v['frozen']['rows']):
                        assert draw['base'] == base
                        assert draw['restored_core_sha256'] == v['best_core_sha256']
                        assert not draw['native_inplace_drift'], draw['native_inplace_drift']
                        expected = independent64(o['labels'], draw['probability'], draw['prediction'])
                        frozen_error = max(frozen_error, max(abs(expected[k]-draw['metrics'][k]) for k in analyze.KEYS))
                        assert np.array_equal(draw['prediction'], np.where(draw['probability'] >= .5, 0, 1))
                    probabilities = np.stack([r['probability'] for r in v['frozen']['rows']])
                    assert np.array_equal(probabilities.mean(0), v['frozen']['probability_mean'])
                    assert np.array_equal(probabilities.std(0), v['frozen']['probability_sd'])
                    mean = probabilities.mean(0)
                    expected = independent64(o['labels'], mean, np.where(mean >= .5, 0, 1))
                    frozen_error = max(frozen_error, max(abs(expected[k]-v['frozen']['mean_probability_metrics'][k]) for k in analyze.KEYS))
                    prediction = np.stack([r['prediction'] for r in v['frozen']['rows']])
                    assert float(np.mean(prediction.min(0) != prediction.max(0))) == v['frozen']['label_flip_fraction']
            assert np.all(coverage == 1)
    assert error < 1e-12 and frozen_error < 1e-12 and ties == 0
    d = {'created_utc': run.utc(), 'paired_input_RNG_optimizer_reference_checks': True,
         'id_txt_context_csv_order_verified': True, 'complete_OOF_all_groups': True,
         'raw_float32_independent_max_error': raw_error, 'corrected_float64_independent_max_error': error,
         'frozen_independent_max_error': frozen_error,
         'unchanged_tolerance': 1e-12, 'threshold_argmax_disagreements': ties, 'groups': {},
         'boundary_scope': 'Heldout perturbation invariance is for the new objective at fixed native z. '
                           'Native transductive graph/BN dependence on heldout features is preserved.',
         'sampling': 'prototype and parameter-gradient diagnostics first 3 then every 25 epochs; '
                     'late summaries epoch>=9; z gradients every auxiliary epoch; no interpolation'}
    for g, values in rows.items():
        epochs = [e for v in values for e in v['epochs'] if e['epoch'] >= 9]
        ps = [e['prototype'] for e in epochs if 'prototype' in e]
        item = {'shuffle_applied': run.CFG['groups'][g]['shuffled'],
                'best_epoch': run.describe([v['best_epoch'] for v in values]),
                'total_training_epochs': sum(v['epochs_run'] for v in values),
                'reached_epoch_cap': sum(v['epochs_run'] == run.CFG['epochs'] for v in values),
                'native_state_change': {p: run.describe([v['best_core_change'][p]['relative_l2'] for v in values])
                                        for p in ('img_unet.', 'ph_unet.', 'mm_attention.', 'clf.')},
                'positive_test_image_margin_fraction': float(np.mean(np.concatenate([v['posthoc_image_margin'] for v in values]) > 0))}
        for name in ('train_ce', 'train_graph', 'val_ce', 'val_graph'):
            item['best_'+name] = run.describe([v['best_record'][name] for v in values])
        for name in ('lambda_eff', 'z_gradient_cosine', 'weighted_z_gradient_ratio'):
            item[name] = run.describe([e[name] for e in epochs if name in e])
        item['lambda_capped_fraction'] = (float(np.mean([e['lambda_eff'] >= run.CFG['lambda_cap']-1e-9 for e in epochs]))
                                          if g != 'D0' else None)
        for name in ('invalid_prototype_fraction', 'beta_zero_fraction'):
            item[name] = run.describe([p[name] for p in ps])
        for kind, left, right in (('age', '0', '3'), ('sex', '1', '2')):
            norms = []
            for p in ps:
                vec = p[f'class_delta_by_{kind}']
                if left in vec and right in vec:
                    norms.append(float(np.linalg.norm(np.array(vec[left])-vec[right])))
            item[f'class_delta_{kind}_contrast_l2'] = run.describe(norms)
        item['conditional_shift_to_class_delta_ratio'] = run.describe([
            p['conditional_global_l2']['mean']/max(p['class_delta_l2']['mean'], 1e-12) for p in ps])
        item['r_within_class_cv'] = {str(k): run.describe([p['r_by_class'][str(k)]['cv'] for p in ps]) for k in (0, 1)}
        item['r_class_mean_ASD_minus_TD'] = run.describe([p['r_by_class']['0']['mean']-p['r_by_class']['1']['mean'] for p in ps])
        if g in run.CFG['frozen_groups']:
            item['patient_probability_sd_across_five_draws'] = run.describe(np.concatenate([v['frozen']['probability_sd'] for v in values]))
        d['groups'][g] = item
    plans = [v['prototype_preparation'] for v in rows['D0']]
    d['permutation'] = {k: run.describe([p[k] for p in plans]) for k in
                        ('shuffle_moved_fraction', 'shuffle_changed_fraction', 'mean_abs_age_change', 'sex_changed_fraction')}
    d['permutation']['singleton_subject_occurrences'] = sum(len(p['singleton_subjects']) for p in plans)
    epoch_delta = []
    for v in rows['D0']:
        old = run.read_json(run.OLD/f"outputs/E0/seed_{v['seed']}/fold_{v['fold']}/metrics.json")
        epoch_delta.append(v['best_epoch']-old['best_epoch'])
    d['baseline_best_epoch_replay'] = {'unchanged_folds': sum(x == 0 for x in epoch_delta),
                                     'delta': run.describe(epoch_delta)}
    d['field_notes'] = {
        'class_delta_query_sd': 'D1 also varies because its three reference subfolds differ, not phenotype conditioning.',
        'r_fallback_fraction': 'Support diagnostic for every group; fallback actually applied only in D6/D7.',
        'effective_condition_changes': 'Common available permutation-plan coverage, applied only in D5/D7.',
        'native_state_change': 'Floating module state including buffers, not an isolated auxiliary parameter update.',
        'raw_margin': 'Scale-sensitive post-hoc embedding distance; not clinical separation or native classifier performance.'}
    d['environment_versions'] = {}
    for name in ('torch', 'torch-geometric', 'torch-scatter', 'torch-sparse', 'numpy', 'scipy', 'pandas', 'scikit-learn'):
        try:
            d['environment_versions'][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            d['environment_versions'][name] = None
    return d


def discussion(s, a, d):
    m = s['mechanisms']; f = s['formal']
    delta = lambda x, y, k: 100*(f[x][k]['mean']-f[y][k]['mean'])
    ds = lambda g, k: d['groups'][g][k]['mean']
    gate = a['baseline_gate']
    replay = d['baseline_best_epoch_replay']
    lines = ['', '## 新鲜基线与双卡时间审计', '',
             f"D0的ACC/AUC为{100*gate['fresh']['ACC']:.4f}/{100*gate['fresh']['AUC']:.4f}%，历史C0为{100*gate['historical']['ACC']:.4f}/{100*gate['historical']['AUC']:.4f}%；差值{gate['delta_pp']['ACC']:+.4f}/{gate['delta_pp']['AUC']:+.4f}pp，均在预注册0.5pp范围内。没有高分重跑或替换历史C0。",
             f"D0与历史C0有{replay['unchanged_folds']}/30个best epoch相同，平均差{replay['delta']['mean']:+.3f}epoch。五锚点严格短轨迹等价，但原生PPU长轨迹非逐位一致；此次偏差未显示训练器结构性不等价。", '',
             '|物理PPU|正式fold数|每组fold数|正式首派发至末完成min|冻结fold数|冻结首派发至末完成min|',
             '|---|---|---|---|---|---|']
    for gpu, e in s['execution']['physical_gpu_timing'].items():
        lines.append(f"|{gpu}|{e['train']['tasks']}|15|{e['train']['span_seconds']/60:.2f}|{e['frozen']['tasks']}|{e['frozen']['span_seconds']/60:.2f}|")
    lines += ['', '逐卡墙钟包含两段正式队列之间的基线审计间隔，不是worker累计时间；两卡并行运行，不能将两张卡墙钟相加作为总耗时。每组每卡15折；冻结每组每卡15折、每折5次。', '',
             '## 八项机制问题的直接回答', '',
             f"1. 条件原型确实改变目标。D4相对全局原型的平均L2为{m['D4']['conditional_global_l2_sampled_means']['mean']:.4f}（辅助标准化坐标）；不是用条件名义运行全局原型。",
             '2. 年龄/性别下的类别差异确有变化，见下表。同一步同一模型内比较向量后再汇总标量，未跨模型直接平均任意方向的16维向量。D1也有三个参考子折导致的差异，不能把所有query变化都归因于表型。',
             f"3. 变化不是正确对应特有。D4/D5的类别差异query SD为{m['D4']['class_delta_query_sd_l2']['mean']:.4f}/{m['D5']['class_delta_query_sd_l2']['mean']:.4f}，非常接近；D4-D5的AUC为{delta('D4','D5','AUC'):+.4f}pp，三seed方向并不完全一致。不能仅用原型异质性证明条件互补。",
             f"4. 没有普遍退化为全局原型。D4最小类ESS平均{m['D4']['ess_min_sampled_means']['mean']:.2f}，beta平均{m['D4']['beta_sampled_means']['mean']:.4f}，beta=0比例{100*ds('D4','beta_zero_fraction'):.3f}%；无效类别原型比例{100*ds('D4','invalid_prototype_fraction'):.3f}%。稀疏支持仍按固定规则收缩，未用heldout补足。",
             f"5. D6权重不是常数：均值{m['D6']['r_sampled_means']['mean']:.4f}、query CV均值{m['D6']['r_CV']['mean']:.4f}；但主要接近上界，存在分布差异不等于可靠性有效。D6-D4的ACC/AUC为{delta('D6','D4','ACC'):+.4f}/{delta('D6','D4','AUC'):+.4f}pp，没有增强收益。类内CV见下表，年龄/性别/站点分布见summary与单fold产物。",
             f"6. 新目标真实作用于编码器。热身后实测输出梯度比约5%；D4的影像参数梯度比为{100*m['D4']['image_parameter_gradient_ratio']['mean']:.2f}%，D6为{100*m['D6']['image_parameter_gradient_ratio']['mean']:.2f}%。这是梯度比，不是Adam更新比；输出梯度与参数梯度不能等同。",
             f"7. 相对纯类别原型D1，D4的ACC/AUC为{delta('D4','D1','ACC'):+.4f}/{delta('D4','D1','AUC'):+.4f}pp；相对D0的AUC仅{delta('D4','D0','AUC'):+.4f}pp、仅1/3 seed改善，CE变化{f['D4']['CE']['mean']-f['D0']['CE']['mean']:+.6f}。不能据此声称稳定超越全局类别监督。",
             f"8. 影像训练类中心距离的测试AUC（事后、非主分类器）D0/D1/D4/D6为{100*m['D0']['posthoc_test_image_prototype_AUC']['mean']:.3f}/{100*m['D1']['posthoc_test_image_prototype_AUC']['mean']:.3f}/{100*m['D4']['posthoc_test_image_prototype_AUC']['mean']:.3f}/{100*m['D6']['posthoc_test_image_prototype_AUC']['mean']:.3f}%。原始距离间隔也会随尺度改变，不能单靠间隔增大证明新增诊断信息。", '',
             '|组别|年龄两端类别差异L2|性别类别差异L2|r类内CV ASD/TD|输出native-aux梯度cos均值|',
             '|---|---|---|---|---|']
    for g in ('D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7'):
        p = d['groups'][g]
        lines.append(f"|{g}|{ds(g,'class_delta_age_contrast_l2'):.6f}|{ds(g,'class_delta_sex_contrast_l2'):.6f}|"+
                     f"{p['r_within_class_cv']['0']['mean']:.6f}/{p['r_within_class_cv']['1']['mean']:.6f}|{ds(g,'z_gradient_cosine'):.6f}|")
    lines += ['', '年龄两端为训练query<12岁与>=30岁，性别为CSV的1/2；均为描述性训练诊断，不是亚组显著性检验。非可靠性组r=1；其支持不足统计仅诊断、未启用r回退。D1的beta=0是全局组定义，不是支持失败。', '',
              '### 五次随机推理的波动', '',
              '|组别|五个draw平均ACC的范围%|五个draw平均AUC的范围%|患者概率SD均值|',
              '|---|---|---|---|']
    for g in run.CFG['frozen_groups']:
        draws = s['frozen'][g]['draws'].values()
        acc = [100*v['ACC']['mean'] for v in draws]
        auc = [100*v['AUC']['mean'] for v in draws]
        lines.append(f"|{g}|{min(acc):.3f} - {max(acc):.3f}|{min(auc):.3f} - {max(auc):.3f}|{ds(g,'patient_probability_sd_across_five_draws'):.6f}|")
    lines += ['', '每个draw先按原规则形成三seed结果；本表只是固定推理随机性的观察范围，不是置信区间，也不与训练seed的样本SD混用。五个draw完整指标见summary.frozen，不能用最好draw替代主成绩。', '',
              '### 研究结论与限制', '',
              '本轮判定为No-Go：D4有轻微正确条件迹象（相对错配的ACC三seed同向、AUC两seed同向），但未达到相对新鲜基线的预注册AUC门槛，且CE变差；D6没有恢复收益。不能说所有表型条件机制均被否定，也不能用最高ACC宣称创新成立。',
              '当前证据不支持直接把PC-DCA扩展为MICCAI 2027第一个创新点。唯一优先下一步是建立无跨折RFE泄漏、含未见站点的疾病判别监督确认协议；暂不叠加Gate、LLM、InfoNCE或提高lambda补救。',
              '持出数据扰动检查只证明：固定一次原生前向后，新增目标不拟合heldout标签、表型或参考Embedding。原始转导人口图及全871节点BN仍可能使heldout特征影响train表示，绝不声称整个模型对heldout输入改变不变。',
              '模块状态L2含浮点Buffer；训练/验证CE来自不同模式和随机前向。它们均是描述性诊断，不是隔离后的因果参数更新量或纯泛化差距。', '',
              '### 数值验收修正与交接审计', '',
              f"原独立复核直接对float32概率裁剪，最大偏差{d['raw_float32_independent_max_error']:.3g}；正式指标使用float64。终检统一到相同数值域后，最大差{d['corrected_float64_independent_max_error']:.3g}，阈值仍为1e-12。未更换公式、修改预测、训练指标或Checkpoint；一次初步验收断言如实保留为已解决的分析精度问题。",
              '为保持正式训练快照SHA完全不变，独立终检入口finalize.py记录自身SHA并调用原分析器；原分析器与训练源码没有改动。正式恢复仍使用原launch，最终统计与交接使用finalize。',
              '训练前另有两项输入格式核对修正：历史RFE support实际为500个列索引、原始FC缓存为NumPy数组；均在正式代码冻结前解决，不是更换数据或重做RFE。',
              f"协调器重启去重检查：{a['resume_verification']['completed_method_artifacts_checked']}个完成产物SHA前后相同，重训0；未人为杀死健康fold测试电源故障或epoch续训。", 
              f"真实错配覆盖{100*d['permutation']['shuffle_changed_fraction']['mean']:.3f}%，平均年龄改变{d['permutation']['mean_abs_age_change']['mean']:.3f}岁；单例累计{d['permutation']['singleton_subject_occurrences']}。只D5/D7实际使用错配。",
              f"完整formal OOF共{8*3*871}条，每组每seed871人恰好一次；固定评估180份fold×5次=900次。所有恢复后的核心状态SHA与best一致，900次前向后核心state无漂移；固定推理指标独立复算最大差{d['frozen_independent_max_error']:.3g}。",
              f"训练队列墙钟合计{s['execution']['training_queue_wall_seconds']/60:.2f}分钟；从审计登记至统计生成{s['execution']['audit_to_report_seconds']/60:.2f}分钟（不含登记前阅读/编写）。GPU0/1整卡采样峰值MiB为{s['execution']['physical_gpu_peak_used_mib']}；驱动每卡总显存98304MiB，不将其误写成98GiB。",
              f"终检源码SHA `{s['postprocessing_sha256']}`；环境版本 `{d['environment_versions']}`。", '']
    return '\n'.join(lines)


def prepare():
    a = run.guard()
    done = [v for v in a['execution']['events'] if v['action'] == 'frozen']
    assert len(done) == 180 and len({tuple(v['task']) for v in done}) == 180
    d = collect(a)
    a['postprocessing'] = {'sha256': run.sha_file(Path(__file__)), 'registered_utc': run.utc(),
                           'reason': 'independent CE float64 numerical-domain correction; no change to training snapshot/results',
                           'raw_max_error': d['raw_float32_independent_max_error'],
                           'corrected_max_error': d['corrected_float64_independent_max_error'], 'tolerance': 1e-12}
    run.atomic_json(a, run.BASE/'audit.json')
    analyze.independent = independent64
    def report(s, audit):
        s['diagnostic_supplement'] = d
        s['postprocessing_sha256'] = run.sha_file(Path(__file__))
        s['resume_verification'] = audit['resume_verification']
        e = s['execution']
        e['training_queue_wall_seconds'] = sum(v['wall_seconds'] for v in e['queue_walls'] if v['action'] == 'train' and not v['skipped'])
        e['audit_to_report_seconds'] = (datetime.datetime.fromisoformat(s['created_utc'])-datetime.datetime.fromisoformat(audit['started_utc'])).total_seconds()
        e['physical_gpu_peak_used_mib'] = {str(g): max(r['gpus'][str(g)]['used_mib'] for r in e['resource_samples'] if 'gpus' in r) for g in (0, 1)}
        e['physical_gpu_timing'] = {}
        for gpu in (0, 1):
            timing = {}
            for mode in ('train', 'frozen'):
                events = [v for v in audit['execution']['events'] if v['gpu'] == gpu and v['action'] == mode]
                start = min(datetime.datetime.fromisoformat(v['started']) for v in events)
                end = max(datetime.datetime.fromisoformat(v['finished']) for v in events)
                groups = dict(collections.Counter(v['task'][0] for v in events))
                assert all(n == 15 for n in groups.values())
                assert len(events) == (120 if mode == 'train' else 90)
                timing[mode] = {'tasks': len(events), 'by_group': groups, 'first_dispatch_utc': start.isoformat(),
                                'last_finish_utc': end.isoformat(), 'span_seconds': (end-start).total_seconds()}
            e['physical_gpu_timing'][str(gpu)] = timing
        e['formal_epochs'] = sum(v['total_training_epochs'] for v in d['groups'].values())
        e['frozen_fold_tasks'] = 180; e['frozen_forward_passes'] = 900
        text = ORIGINAL_REPORT(s, audit)
        text = text.replace('r回退比例', '两半支持不足比例')
        text = text.replace('患者标签变化比例', '折内标签变化比例均值')
        text = text.replace('独立无监督RFE污染、含未见站点的疾病判别监督确认', '消除跨折RFE泄漏并在未见站点确认疾病判别监督')
        return text+discussion(s, audit, d)
    analyze.make_report = report
    analyze.main(final=False)
    s = run.read_json(run.BASE/'summary.json')
    assert s['status'] == 'complete' and s['completed_tasks'] == 240
    assert all(len(s['frozen'][g]['draws']) == 5 for g in run.CFG['frozen_groups'])
    s['report_sha256'] = run.sha_file(run.BASE/'report.md')
    s['delivery_stage'] = 'verified_research_awaiting_handoff'
    run.atomic_json(s, run.BASE/'summary.json')


def append():
    a = run.guard(); s = run.read_json(run.BASE/'summary.json')
    assert s['status'] == 'complete' and s['completed_tasks'] == 240
    assert s['postprocessing_sha256'] == run.sha_file(Path(__file__)) == a['postprocessing']['sha256']
    assert s['report_sha256'] == run.sha_file(run.BASE/'report.md')
    for task, expected in s['artifact_sha256'].items():
        g, seed, fold = task.split('/')
        assert run.sha_file(run.result_path(g, int(seed), int(fold))) == expected
    assert len(s['artifact_sha256']) == 240
    analyze.verify_protected(a)
    report = run.read_bytes(run.BASE/'report.md').decode()
    try:
        s['handoff'] = analyze.append_handoff(s, report)
    except Exception as error:
        s['handoff'] = {'appended': False, 'error': repr(error), 'pending_append_content': report}
        run.atomic_json(s, run.BASE/'summary.json')
        raise
    s['delivery_stage'] = 'complete'
    s['final_verified_utc'] = run.utc()
    run.atomic_json(s, run.BASE/'summary.json')
    run.log('final_acceptance', tasks=240, frozen=180, decision=s['decision'], handoff=s['handoff'])
    print(s['handoff'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['prepare', 'append'])
    args = parser.parse_args()
    with (run.BASE/'launch.py').open('r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare() if args.action == 'prepare' else append()
