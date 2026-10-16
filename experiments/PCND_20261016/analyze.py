"""Independent OOF checks, paired summaries and fixed-logit PCND diagnostics."""
from datetime import datetime
import json
import os
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
import run as r

NAMES = {'P0': 'Historical C0', 'P1': 'PCA容量对照', 'P2': '无条件TD偏差', 'P3': '年龄条件偏差',
         'P4': '年龄/性别偏差', 'P5': '可靠PCND', 'P6': '分区内同站点错配PCND'}


def atomic_text(value, path):
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    try:
        with tmp.open('w') as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def independent_metrics(labels, pred, probability):
    y, prediction, score = np.asarray(labels), np.asarray(pred), np.asarray(probability, dtype=float)
    positive, negative = y == 0, y == 1
    tp, fn = int(np.sum(positive & (prediction == 0))), int(np.sum(positive & (prediction == 1)))
    tn, fp = int(np.sum(negative & (prediction == 1))), int(np.sum(negative & (prediction == 0)))
    sen = tp / (tp + fn) if tp + fn else None
    spe = tn / (tn + fp) if tn + fp else None
    auc = None
    if positive.any() and negative.any():
        ranks = rankdata(score, method='average')
        n1, n0 = int(positive.sum()), int(negative.sum())
        auc = float((ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return {'ACC': float((prediction == y).mean()),
        'BA': (sen + spe) / 2 if sen is not None and spe is not None else None,
        'AUC': auc, 'SEN': sen, 'SPE': spe, 'F1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def packed(rows):
    seeds = {}
    for seed in r.config()['seeds']:
        chosen = [v for v in rows if v['seed'] == seed]
        if len(chosen) != 10:
            raise AssertionError('Each seed must have ten folds')
        seeds[str(seed)] = {k: float(np.mean([v['metrics'][k] for v in chosen])) for k in r.KEYS}
    return {'seed_means': seeds, 'mean': {k: float(np.mean([m[k] for m in seeds.values()])) for k in r.KEYS},
        'seed_sd_ddof1': {k: float(np.std([m[k] for m in seeds.values()], ddof=1)) for k in r.KEYS},
        'fold_sd_ddof0_descriptive_only': {k: float(np.std([v['metrics'][k] for v in rows])) for k in r.KEYS}}


def get_fold(group, seed, fold, state):
    if group == 'P0':
        b = state['audit']['baseline'][f'{seed}/{fold}']
        return {**b, 'oof': {k: b[k] for k in ('indices', 'labels', 'pred', 'prob_asd', 'ids', 'sites', 'metrics')},
                'frozen': state['audit']['baseline_frozen'][f'{seed}/{fold}']}
    if not r.verify_result(group, seed, fold, state, complete=group in r.config()['repeat_groups']):
        raise AssertionError('Incomplete fold')
    return r.load_torch(r.result_path(group, seed, fold), mmap=True)


def summarize_group(group, state):
    rows, probabilities, draw_rows, avg_rows, instabilities = [], [], [[] for _ in range(5)], [], []
    timings, delta, gate, epochs, gpu = [], [], [], [], {'0': 0, '1': 0}
    features = []
    for seed in r.config()['seeds']:
        covered = []
        for fold in range(10):
            s = get_fold(group, seed, fold, state)
            row = {'seed': seed, 'fold': fold, 'metrics': s['oof']['metrics'],
                   'best_epoch': s['best_epoch'], 'epochs_run': s['epochs_run']}
            rows.append(row)
            covered.extend(s['oof']['indices'])
            if group != 'P0':
                timings.append(s['classifier_seconds'])
                gpu[str(s['physical_gpu'])] += 1
                delta.append(s['test_delta_abs_mean'])
                gate.append(s['gate_distribution']['test']['mean'])
                best = s['epochs'][s['best_epoch']]
                epochs.append({**row, 'train_ce': best['train_ce'], 'train_graph': best['train_graph'],
                    'val_ce': best['val_ce'], 'val_graph': best['val_graph'], 'val_loss': best['val_loss'],
                    'train_val_gap': best['val_loss'] - best['train_total']})
                features.append({'seed': seed, 'fold': fold, 'train_td': s['normative_train_td_count'],
                    'pca_explained_sum': sum(s['pca_variance_ratio']), 'clip_fraction': s['clip_fraction']})
            frozen = s.get('frozen')
            if frozen:
                avg_rows.append({'seed': seed, 'fold': fold, 'metrics': frozen['mean_probability_metrics']})
                instabilities.append(frozen['test_prediction_instability_fraction'])
                for draw in range(5):
                    draw_rows[draw].append({'seed': seed, 'fold': fold, 'metrics': frozen['draw_metrics'][draw]})
        if sorted(covered) != list(range(871)):
            raise AssertionError('OOF does not cover each subject exactly once')
    result = {'name': NAMES[group], 'formal': packed(rows), 'formal_folds': rows, 'completed_folds': 30,
        'best_epoch_mean': float(np.mean([v['best_epoch'] for v in rows])),
        'epochs_run_mean': float(np.mean([v['epochs_run'] for v in rows])),
        'physical_gpu_training_totals': gpu if group != 'P0' else 'historical_readonly',
        'classifier_seconds': {'sum': sum(timings), 'per_fold': r.distribution(timings)},
        'formal_test_delta_abs_mean': r.distribution(delta),
        'formal_test_gate_mean': r.distribution(gate), 'best_training_records': epochs,
        'feature_diagnostics': features}
    if avg_rows:
        if len(avg_rows) != 30:
            raise AssertionError('Incomplete auxiliary group')
        result.update(frozen_probability_mean=packed(avg_rows), frozen_folds=avg_rows,
            frozen_draws=[packed(v) for v in draw_rows],
            stochastic_metric_distribution={k: r.distribution([packed(v)['mean'][k] for v in draw_rows]) for k in r.KEYS},
            test_label_instability=r.distribution(instabilities))
    return result


def paired(a, b):
    old = {(v['seed'], v['fold']): v for v in b['formal_folds']}
    rows = [{'seed': v['seed'], 'fold': v['fold'],
             'metrics': {k: v['metrics'][k] - old[v['seed'], v['fold']]['metrics'][k] for k in r.KEYS}}
            for v in a['formal_folds']]
    aggregate = packed(rows)
    return {'aggregate': aggregate, 'paired_folds': rows,
        'positive_seeds': {k: sum(v[k] > 1e-12 for v in aggregate['seed_means'].values()) for k in r.KEYS},
        'tied_seeds': {k: sum(abs(v[k]) <= 1e-12 for v in aggregate['seed_means'].values()) for k in r.KEYS},
        'positive_folds': {k: sum(v['metrics'][k] > 1e-12 for v in rows) for k in r.KEYS},
        'tied_folds': {k: sum(abs(v['metrics'][k]) <= 1e-12 for v in rows) for k in r.KEYS},
        'independence_warning': 'Same 871 people across seeds; fold/seed wins are descriptive, not independent external cohorts'}


def ce(probability, labels):
    p = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    return -np.log(np.where(np.asarray(labels) == 0, p, 1 - p))


def partitions(meta):
    parts = {'all': np.arange(871), 'diagnosis_ASD': np.flatnonzero(meta['labels'] == 0),
             'diagnosis_TD': np.flatnonzero(meta['labels'] == 1)}
    for value in ('1', '2'):
        parts['sex_' + value] = np.flatnonzero(meta['sex'] == value)
    for lo, hi in zip(r.config()['age_bins'][:-1], r.config()['age_bins'][1:]):
        parts[f'age_{lo}_{hi}'] = np.flatnonzero((meta['age'] >= lo) & (meta['age'] < hi))
    for site in np.unique(meta['site']):
        parts['site_' + str(site)] = np.flatnonzero(meta['site'] == site)
    return parts


def conditional_audit(group, state, meta):
    fold_rows, seed_rows, by_seed, effective = [], {}, {}, {}
    ranking_changes, delta, flipped, benefits = [], [], [], []
    for seed in r.config()['seeds']:
        true_all, shuffle_all, off_all = [np.full((5, 871), np.nan) for _ in range(3)]
        effective_all = np.zeros(871, dtype=bool)
        current = []
        for fold in range(10):
            saved = get_fold(group, seed, fold, state)
            frozen = saved['frozen']
            idx, y = np.asarray(saved['oof']['indices']), np.asarray(saved['oof']['labels'])
            p = np.asarray(frozen['counterfactual_prob_asd']['true'])
            q = np.asarray(frozen['counterfactual_prob_asd']['shuffled'])
            off = np.asarray(frozen['counterfactual_prob_asd']['off'])
            true_all[:, idx], shuffle_all[:, idx], off_all[:, idx] = p, q, off
            cache = r.load_torch(r.feature_path(seed, fold))
            changed = np.asarray(cache['diagnostics']['shuffle']['changed_mask'], dtype=bool)[idx]
            effective_all[idx] = changed
            diff_ce = ce(p, y) - ce(q, y)
            changes = []
            for d in range(5):
                a, b = np.sign(p[d, :, None] - p[d, None, :]), np.sign(q[d, :, None] - q[d, None, :])
                upper = np.triu(np.ones(a.shape, dtype=bool), 1)
                changes.append(float((a[upper] != b[upper]).mean()))
            ranking_changes.extend(changes)
            row = {'seed': seed, 'fold': fold, 'CE_true_minus_shuffle': float(diff_ce.mean()),
                'actually_changed_age_or_sex_count': int(changed.sum()), 'test_n': len(idx),
                'actually_changed_fraction': float(changed.mean()),
                'probability_MAD': float(np.abs(p - q).mean()),
                'benefited_patient_fraction_5draw_mean_CE': float((diff_ce.mean(0) < -1e-12).mean()),
                'label_flip_fraction': float(((p >= .5) != (q >= .5)).mean()),
                'pairwise_ranking_changed_fraction': float(np.mean(changes)), 'draw_metric_differences': []}
            for d in range(5):
                m, n = r.metrics(y, np.where(p[d] >= .5, 0, 1), p[d]), r.metrics(y, np.where(q[d] >= .5, 0, 1), q[d])
                row['draw_metric_differences'].append({k: m[k] - n[k] for k in r.KEYS})
            row['metrics'] = {k: float(np.mean([v[k] for v in row['draw_metric_differences']])) for k in r.KEYS}
            current.append(row); fold_rows.append(row)
            delta.append(float(np.abs(frozen['counterfactual_delta']['true']).mean()))
        if not all(np.isfinite(v).all() for v in (true_all, shuffle_all, off_all)):
            raise AssertionError('Incomplete conditional OOF')
        seed_rows[str(seed)] = {'CE_true_minus_shuffle': float(np.mean([v['CE_true_minus_shuffle'] for v in current])),
            'probability_MAD': float(np.mean([v['probability_MAD'] for v in current])),
            **{k: float(np.mean([v['metrics'][k] for v in current])) for k in r.KEYS}}
        by_seed[str(seed)] = (true_all, shuffle_all, off_all)
        idx = np.flatnonzero(effective_all)
        labels = meta['labels'][idx]
        p, q = true_all[:, idx], shuffle_all[:, idx]
        differences = []
        for draw in range(5):
            a = independent_metrics(labels, np.where(p[draw] >= .5, 0, 1), p[draw])
            b = independent_metrics(labels, np.where(q[draw] >= .5, 0, 1), q[draw])
            differences.append({k: a[k] - b[k] if a[k] is not None and b[k] is not None else None for k in r.KEYS})
        effective[str(seed)] = {'n': len(idx), 'fraction': float(effective_all.mean()), 'indices': idx.tolist(),
            'CE_true_minus_shuffle': float((ce(p, labels) - ce(q, labels)).mean()),
            'probability_MAD': float(np.abs(p - q).mean()),
            'CE_benefited_patient_fraction': float(((ce(p, labels) - ce(q, labels)).mean(0) < -1e-12).mean()),
            'five_draw_metric_differences': differences,
            'mean_draw_metric_difference': {k: float(np.mean([v[k] for v in differences]))
                if all(v[k] is not None for v in differences) else None for k in r.KEYS},
            'metric_unit': 'Pooled OOF on actually changed subjects within each seed; auxiliary, not formal ten-fold mean'}
    strata = {}
    for name, idx in partitions(meta).items():
        if not len(idx):
            continue
        labels = meta['labels'][idx]
        per_seed = {}
        small_auc = name.startswith('site_') and (len(idx) < r.config()['minimum_site_auc_samples'] or
            min(int((labels == 0).sum()), int((labels == 1).sum())) < r.config()['minimum_site_auc_per_class'])
        for seed in r.config()['seeds']:
            p, q, off = [v[:, idx] for v in by_seed[str(seed)]]
            m, n = r.metrics(labels, np.where(p.mean(0) >= .5, 0, 1), p.mean(0)), r.metrics(labels, np.where(q.mean(0) >= .5, 0, 1), q.mean(0))
            if small_auc:
                m['AUC'] = n['AUC'] = None
            diff = ce(p, labels) - ce(q, labels)
            per_seed[str(seed)] = {'CE_true_minus_shuffle': float(diff.mean()), 'probability_MAD': float(np.abs(p - q).mean()),
                'CE_benefited_fraction': float((diff.mean(0) < -1e-12).mean()),
                'true_mean_probability_metrics': m, 'shuffle_mean_probability_metrics': n,
                'label_flip_fraction_mean_probability': float(((p.mean(0) >= .5) != (q.mean(0) >= .5)).mean()),
                'mean_probability_MAD_true_vs_off': float(np.abs(p - off).mean())}
        strata[name] = {'subjects_per_seed': len(idx), 'ASD': int((labels == 0).sum()), 'TD': int((labels == 1).sum()),
            'small_site_AUC_suppressed': small_auc, 'seeds': per_seed,
            'mean_CE_true_minus_shuffle': float(np.mean([v['CE_true_minus_shuffle'] for v in per_seed.values()])),
            'mean_probability_MAD': float(np.mean([v['probability_MAD'] for v in per_seed.values()]))}
    return {'direction': 'CE(true)-CE(shuffle); negative favors true; probabilities share one native core realization',
        'folds': fold_rows, 'seeds': seed_rows,
        'CE_true_minus_shuffle_mean': float(np.mean([v['CE_true_minus_shuffle'] for v in seed_rows.values()])),
        'CE_true_minus_shuffle_seed_sd': float(np.std([v['CE_true_minus_shuffle'] for v in seed_rows.values()], ddof=1)),
        'true_CE_better_seeds': sum(v['CE_true_minus_shuffle'] < -1e-12 for v in seed_rows.values()),
        'probability_MAD': float(np.mean([v['probability_MAD'] for v in seed_rows.values()])),
        'metric_differences': packed(fold_rows), 'strata': strata, 'effective_shuffle_subset': effective,
        'pairwise_ranking_changed_fraction': r.distribution(ranking_changes),
        'mean_true_delta_abs': float(np.mean(delta)), 'nonindependence_warning': 'Five draws and repeated seeds share patients'}


def reliability_audit(state, meta):
    rows = {}
    def correlation(a, b, ranked=False):
        a, b = np.asarray(a, float), np.asarray(b, float)
        if ranked:
            a, b = rankdata(a), rankdata(b)
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-12 and b.std() > 1e-12 else None
    for seed in r.config()['seeds']:
        gate = np.full(871, np.nan)
        probabilities = {k: np.full((5, 871), np.nan) for k in ('true', 'shuffled', 'off', 'unit_gate')}
        for fold in range(10):
            saved = get_fold('P5', seed, fold, state)
            idx = np.asarray(saved['oof']['indices'])
            cache = r.load_torch(r.feature_path(seed, fold))
            gate[idx] = cache['features']['P5']['gate'][idx]
            for k in probabilities:
                probabilities[k][:, idx] = saved['frozen']['counterfactual_prob_asd'][k]
        if not np.isfinite(gate).all() or not all(np.isfinite(p).all() for p in probabilities.values()):
            raise AssertionError('Incomplete reliability OOF')
        loss = {k: ce(v, meta['labels']).mean(0) for k, v in probabilities.items()}
        benefits = {'CE_off_minus_true': loss['off'] - loss['true'],
                    'CE_shuffle_minus_true': loss['shuffled'] - loss['true'],
                    'CE_unit_gate_minus_gated': loss['unit_gate'] - loss['true']}
        correlations = {k: {'Pearson': correlation(gate, value), 'Spearman': correlation(gate, value, True),
            'mean_benefit': float(value.mean()), 'benefited_fraction': float((value > 1e-12).mean())}
            for k, value in benefits.items()}
        bins = {}
        for lo, hi in ((.25, .5), (.5, .75), (.75, 1.000001)):
            idx = np.flatnonzero((gate >= lo) & (gate < hi))
            bins[f'{lo}-{min(hi, 1.0)}'] = {'n': len(idx),
                'ASD_fraction': float((meta['labels'][idx] == 0).mean()) if len(idx) else None,
                **{k: float(v[idx].mean()) if len(idx) else None for k, v in benefits.items()}}
        rows[str(seed)] = {'gate': r.distribution(gate), 'correlations': correlations, 'predefined_gate_bins': bins,
            'gate_vs_ASD_label_Pearson': correlation(gate, (meta['labels'] == 0).astype(float)),
            'same_checkpoint_gate_minus_unit_gate_CE': float((loss['true'] - loss['unit_gate']).mean()),
            'unit_gate_vs_gated_probability_MAD': float(np.abs(probabilities['true'] - probabilities['unit_gate']).mean())}
    return {'seeds': rows,
        'definition': 'Same P5 checkpoint/core realization: g versus unit gate; positive CE benefit favors gated/true condition',
        'caution': 'Descriptive within-OOF association, not calibrated reliability or causal proof; patients recur across seeds. P5-P4 is a separate jointly trained comparison.'}


def execution_audit(state):
    result = {'physical_gpu_training_totals': {'0': 0, '1': 0}, 'groups': {},
              'phases': state['scheduling']['phases'], 'peak_workers': state['scheduling']['peak_workers'],
              'resource_extrema': state['scheduling']['resource_extrema']}
    starts, ends = [], []
    per_gpu = {'0': {'seconds': 0.0, 'first': [], 'last': []}, '1': {'seconds': 0.0, 'first': [], 'last': []}}
    for group in r.GROUPS:
        counts = {'0': 0, '1': 0}
        for seed in r.config()['seeds']:
            for fold in range(10):
                s = r.load_torch(r.result_path(group, seed, fold), mmap=True)
                gpu = str(s['physical_gpu'])
                counts[gpu] += 1
                result['physical_gpu_training_totals'][gpu] += 1
                starts.append(s['started_utc']); ends.append(s['completed_utc'])
                per_gpu[gpu]['first'].append(s['started_utc']); per_gpu[gpu]['last'].append(s['completed_utc'])
                per_gpu[gpu]['seconds'] += s['classifier_seconds']
        if counts != {'0': 15, '1': 15}:
            raise AssertionError('Every group must contribute 15 folds per GPU')
        result['groups'][group] = counts
    if result['physical_gpu_training_totals'] != {'0': 90, '1': 90}:
        raise AssertionError('Training did not finish 90/90')
    result['training_wall_span_seconds'] = (datetime.fromisoformat(max(ends)) - datetime.fromisoformat(min(starts))).total_seconds()
    result['per_gpu'] = {g: {'training_tasks': result['physical_gpu_training_totals'][g],
        'sum_classifier_seconds': v['seconds'], 'wall_span_seconds': (datetime.fromisoformat(max(v['last'])) - datetime.fromisoformat(min(v['first']))).total_seconds()}
        for g, v in per_gpu.items()}
    result['total_attempts_train'] = sum(v for k, v in state['attempts'].items() if k.startswith('train/'))
    result['failed_training_attempts'] = len([v for v in state['errors'] if v['task'].startswith('train/')])
    result['sampling_warning'] = 'Resource extrema are sampled, not continuous peaks; device usage may include other processes'
    return result


def verify_completed_bundle(state, rehash=True):
    r.guard()
    formal_count, frozen_count = 0, 0
    hashes, cache_hashes = {}, {}
    initial = {}
    ctx = r.load_torch(r.OLD / 'artifacts/context.pt')
    meta = r.metadata(ctx)
    for seed in r.config()['seeds']:
        for fold in range(10):
            if not r.verify_cache(seed, fold, state):
                raise AssertionError('Missing cache')
            cache = r.load_torch(r.feature_path(seed, fold))
            previous = state['p6_revision']['previous_caches'][f'{seed}/{fold}']
            formal_hashes = {g: {k: r.array_hash(cache['features'][g][k]) for k in ('features', 'gate')}
                             for g in r.GROUPS[:-1]}
            if formal_hashes != previous['P1_P5_formal_sha256']:
                raise AssertionError('Independent P1-P5 preservation check failed')
            split = r.load_torch(r.anchor_dir(seed, fold) / 'splits.pt')
            perm, record = r.make_shuffle(meta['site'], meta['age'], meta['sex'].astype(int), seed, fold, split)
            if record != cache['diagnostics']['shuffle']:
                raise AssertionError('Saved permutation/SHA/changed fractions are not reproducible')
            if not np.array_equal(np.sort(perm), np.arange(871)):
                raise AssertionError('Permutation not bijective')
            covered_inner = []
            for proof in cache['boundary_checks']['crossfit']:
                reference, query = np.asarray(proof['reference_td']), np.asarray(proof['heldout_train'])
                if np.intersect1d(reference, query).size or not np.isin(reference, split['train']).all():
                    raise AssertionError('TD self-fit or nontraining reference detected')
                if not np.all(np.asarray(ctx['labels'])[reference] == 1):
                    raise AssertionError('Normative reference includes ASD')
                covered_inner.extend(query.tolist())
            if sorted(covered_inner) != sorted(np.asarray(split['train']).tolist()):
                raise AssertionError('Crossfit training coverage is not one-to-one')
            weights = cache['boundary_checks']['P6_reference_weights']
            if [r.array_hash(w) for w in weights] != cache['boundary_checks']['reference_weight_sha256'][-6:]:
                raise AssertionError('P6 saved fitting weights checksum failed')
            pca = cache['pca_state']
            if r.state_hash({'components': pca['components'], 'mean': pca['mean'],
                             'variance': pca['explained_variance'], 'singular': pca['singular']}) != cache['boundary_checks']['pca_sha256']:
                raise AssertionError('Saved PCA basis differs from accepted training-only basis')
            cache_hashes[f'features/s{seed}_f{fold}.pt'] = r.sha_file(r.feature_path(seed, fold))
    for group in ('P0', *r.GROUPS):
        for seed in r.config()['seeds']:
            covered = []
            for fold in range(10):
                s = get_fold(group, seed, fold, state)
                oof = s['oof']
                calculated = independent_metrics(oof['labels'], oof['pred'], oof['prob_asd'])
                if any(abs(calculated[k] - oof['metrics'][k]) > 1e-12 for k in r.KEYS):
                    raise AssertionError('Independent rank-AUC/confusion check failed')
                covered.extend(oof['indices'])
                if group != 'P0':
                    formal_count += 1
                    if not np.array_equal(np.asarray(oof['logits']).argmax(1), oof['pred']):
                        raise AssertionError('Formal predictions do not match saved logits')
                    key = f'{seed}/{fold}'
                    initial.setdefault(key, set()).add(s['initial_branch_sha256'])
                    if s['parameters']['branch'] != 1089:
                        raise AssertionError('Capacity mismatch')
                    hashes[f'runs/{group}/s{seed}_f{fold}.pt'] = r.sha_file(r.result_path(group, seed, fold))
                    if s['data_input_digest'] != state['data_input_digests'][key]:
                        raise AssertionError('Native x/ph/graph inputs differ from preflight')
                if s.get('frozen'):
                    f = s['frozen']
                    seeds = [r.private_seed(seed, fold, f"PCCI-repeat-{v}") % (2**32) for v in r.config()['repeat_seed_bases']]
                    if f['draw_seeds'] != seeds:
                        raise AssertionError('Auxiliary seeds differ from historical PCCI policy')
                    for d in range(5):
                        m = independent_metrics(oof['labels'], f['test_pred'][d], f['test_prob_asd'][d])
                        if any(abs(m[k] - f['draw_metrics'][d][k]) > 1e-12 for k in r.KEYS):
                            raise AssertionError('Independent frozen metric check failed')
                        if group != 'P0':
                            frozen_count += 1
                    p = np.asarray(f['test_prob_asd']).mean(0)
                    m = independent_metrics(oof['labels'], np.where(p >= .5, 0, 1), p)
                    if any(abs(m[k] - f['mean_probability_metrics'][k]) > 2e-8 for k in r.KEYS):
                        raise AssertionError('Independent five-draw probability-mean metric check failed')
                    if group in ('P4', 'P5'):
                        for values in f['counterfactual_prob_asd'].values():
                            p = np.asarray(values)
                            if p.shape != (5, len(oof['labels'])) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
                                raise AssertionError('Invalid counterfactual probabilities')
            if sorted(covered) != list(range(871)):
                raise AssertionError('OOF coverage broken')
    if formal_count != 180 or frozen_count != 450 or len(initial) != 30 or any(len(v) != 1 for v in initial.values()):
        raise AssertionError('Fit/draw count or initial pairing incorrect')
    if state.get('formal_signatures_after_training'):
        for group in r.GROUPS:
            for seed in r.config()['seeds']:
                for fold in range(10):
                    name = f'{group}/{seed}/{fold}'
                    if r.load_torch(r.result_path(group, seed, fold), mmap=True)['formal_signature'] != state['formal_signatures_after_training'][name]:
                        raise AssertionError('Auxiliary evaluation changed formal score or checkpoint')
    protected = 0
    if rehash:
        for name, expected in state['audit']['manifest'].items():
            if r.sha_file(name) != expected['sha256']:
                raise AssertionError(f'Protected historical input changed: {name}')
            protected += 1
    for name, expected in state['formal_source_sha256'].items():
        if r.sha_file(r.BASE / name) != expected:
            raise AssertionError('Formal registered source changed')
    saved_hashes = state.get('final_checkpoints_sha256')
    if saved_hashes is not None and saved_hashes != hashes:
        raise AssertionError('Completed checkpoint bytes changed')
    saved_caches = state.get('final_features_sha256')
    if saved_caches is not None and saved_caches != cache_hashes:
        raise AssertionError('Completed feature bytes changed')
    return {'passed': True, 'time_utc': r.utc(), 'new_formal_fits': formal_count,
        'historical_formal_fits': 30, 'new_frozen_draws': frozen_count, 'historical_frozen_draws_reused': 150,
        'feature_caches': 30, 'protected_historical_files_rehashed': protected,
        'OOF_coverage_exact_once_per_seed_group': True, 'OOF_metric_direction_ASD0': True,
        'independent_metric_method': 'Manual confusion counts and Mann-Whitney rank AUC with average ties; no sklearn metrics',
        'capacity_1089_and_same_initial_branch_all_groups': True, 'formal_results_preserved': True,
        'P6_partition_site_permutations_and_SHA_reproduced': 30,
        'P1_P5_formal_arrays_match_prerevision_manifest': True,
        'source_and_data_unchanged': True, 'checkpoints_sha256': hashes, 'features_sha256': cache_hashes}


def num(value, scale=100):
    return f'{value * scale:.2f}'


def research_answers(state):
    groups, pairs = state['groups'], state['comparisons']
    def numbers(a, b):
        return {k: pairs[f'{a}-{b}']['aggregate']['mean'][k] * 100 for k in ('ACC', 'BA', 'AUC')}
    condition = state['conditional']['P5']
    p5p0 = numbers('P5', 'P0')
    p5p1, p5p4, p5p6, p4p3 = [numbers(a, b) for a, b in
                               (('P5', 'P1'), ('P5', 'P4'), ('P5', 'P6'), ('P4', 'P3'))]
    consistent = all(v > 0 for v in p5p0.values()) and all(v > 0 for v in p5p1.values()) and all(v > 0 for v in p5p6.values())
    seed_consistent = all(pairs[f'P5-{g}']['positive_seeds'][k] == 3
                          for g in ('P0', 'P1', 'P6') for k in ('ACC', 'BA', 'AUC'))
    semantic = condition['true_CE_better_seeds'] == 3 and all(v['AUC'] > 0 for v in condition['seeds'].values())
    return {'formal_delta_pp_P5_minus_P0': p5p0, 'P5_minus_P1': p5p1, 'P5_minus_P4': p5p4,
        'P5_minus_P6': p5p6, 'P4_minus_P3': p4p3,
        'all_primary_three_metrics_positive_vs_P0_P1_P6': consistent,
        'all_three_seeds_improve_ACC_BA_AUC_vs_P0_P1_P6': seed_consistent,
        'P5_minus_P2': numbers('P5', 'P2'), 'P5_minus_P3': numbers('P5', 'P3'),
        'true_condition_CE_favors_true_all_three_seeds': condition['true_CE_better_seeds'] == 3,
        'true_condition_CE_and_AUC_favor_true_all_three_seeds': semantic,
        'CE_true_minus_shuffle': condition['CE_true_minus_shuffle_mean'],
        'condition_probability_MAD': condition['probability_MAD'],
        'mechanism_claim': ('同协议主指标和条件配对存在一致正向线索，但尚非独立验证或创新性证明。'
                           if consistent and seed_consistent and semantic else
                           '没有同时满足同协议主指标、等容量对照、错配对照和三seed条件配对的完整证据，不能宣布PCND成功。'),
        'facts': [
            'P1-P6均为1089参数分支、同fold初始化/RFE/VAE/图/随机起点。',
            'PCND只在native logits后加[delta,-delta]，不直接改变原16维joint或Classifier/BN输入。',
            '条件反事实固定同一Checkpoint及一次core realization，不重拟合TD参考或BN。',
            'P6仅分区内部同站点联合错配，不使用heldout协变量拟合训练参考；与P5相同容量和超参数。',
            '三步确定性parity验证训练器接入，不保证原生PPU完整随机优化逐位重现历史C0。'],
        'unresolved': [
            '历史RFE跨折标签泄漏限制所有绝对性能和临床泛化解释。',
            '仅871人内部转导CV、三个重复seed；没有独立队列或外站验证。',
            '新branch改变CE和主干联合优化，增益或退化不能唯一归因于规范参考的医学机制。',
            '近似标准化偏差未进行概率规范模型或临床Z分数校准。',
            '年龄性别条件偏差本身不是原创性证明；本轮不追加未预注册变体。'],
        'next_direction': ('下一步仅预注册严格无泄漏、站点外的受保护logit证据验证，不在本轮测试集继续调参。'
                           if consistent and seed_consistent and semantic else
                           '先建立严格无泄漏且包含站点外测试的独立评估协议，验证简单影像/TD偏差证据；暂不叠加复杂表型Gate。')}


def write_report(state):
    groups, pairs = state['groups'], state['comparisons']
    lines = ['# Experiment 5：PCND 正式机制检验', '', f"生成时间 UTC：{r.utc()}", '',
        '## 研究问题与协议', '',
        '检验训练TD参考的年龄/性别条件偏差能否提供原始MM-GTUNets及同容量影像证据之外的患者级增量。全部180任务预注册，未依据测试表现删组、改系数或挑seed。主表仅原生单次随机测试，不用五次均值替换。',
        'ASD=0、HC/TD=1；SEN针对ASD、SPE针对HC；AUC使用softmax第0列。先每seed平均十折，再报告三个seed均值与样本SD，单位%。30折胜负只作配对描述。',
        '历史官方行为完整保留：RFE跨折覆盖的已知监督泄漏、RP循环早返/未注册权重/forward原位投影、VAE反复加载及采样、eval持续dropout、871人转导图。不能当作无泄漏临床泛化或外站验证。', '',
        '## 正式主表', '', '| Group | 设计 | ACC | BA | AUC | SEN | SPE | F1 | Best Epoch | ΔACC vs P0 pp | 完成 |',
        '|---|---|---|---|---|---|---|---|---|---|---|']
    for group in ('P0', *r.GROUPS):
        g, base = groups[group], groups['P0']['formal']['mean']
        f = g['formal']
        values = [f"{num(f['mean'][k])} +/- {num(f['seed_sd_ddof1'][k])}" for k in r.KEYS]
        lines.append('| ' + ' | '.join([group, NAMES[group], *values, f"{g['best_epoch_mean']:.2f}",
                                        num(f['mean']['ACC'] - base['ACC']), '30/30' if group != 'P0' else '30只读']) + ' |')
    lines.extend(['', '## 每Seed结果', '', '| Group | Seed | ACC | BA | AUC | SEN | SPE | F1 |',
                  '|---|---|---|---|---|---|---|---|'])
    for group in ('P0', *r.GROUPS):
        for seed, values in groups[group]['formal']['seed_means'].items():
            lines.append('| ' + ' | '.join([group, seed, *[num(values[k]) for k in r.KEYS]]) + ' |')
    lines.extend(['', '## 配对主比较', '',
        '| 对比 | ΔACC pp | ΔBA pp | ΔAUC pp | 正向seed ACC/BA/AUC | 正向fold ACC/BA/AUC | 平局fold ACC/BA/AUC |',
        '|---|---|---|---|---|---|---|'])
    for key, pair in pairs.items():
        mean = pair['aggregate']['mean']
        fields = ['/'.join(str(pair[name][k]) for k in ('ACC', 'BA', 'AUC'))
                  for name in ('positive_seeds', 'positive_folds', 'tied_folds')]
        lines.append('| ' + ' | '.join([key, *[num(mean[k]) for k in ('ACC', 'BA', 'AUC')], *fields]) + ' |')
    lines.extend(['', '三个seed共享同一队列，不能将30折作为30个独立临床样本进行伪独立显著性推断；仅报告效应、seed一致性与描述性胜负。', '',
        '## 五次固定RNG辅助评估', '',
        '| Group | 概率均值ACC | BA | AUC | 单draw总体ACC范围 | 单draw总体AUC范围 | 标签变化比例 |',
        '|---|---|---|---|---|---|---|'])
    for group in ('P0', *r.config()['repeat_groups']):
        g = groups[group]
        mean, draws = g['frozen_probability_mean']['mean'], g['frozen_draws']
        lines.append('| ' + ' | '.join([group, *[num(mean[k]) for k in ('ACC', 'BA', 'AUC')],
            f"{num(min(d['mean']['ACC'] for d in draws))} / {num(max(d['mean']['ACC'] for d in draws))}",
            f"{num(min(d['mean']['AUC'] for d in draws))} / {num(max(d['mean']['AUC'] for d in draws))}",
            num(g['test_label_instability']['mean'])]) + ' |')
    lines.extend(['', '仅P4/P5/P6进行了预注册的新增450次冻结随机前向；P0直接复用已有150次。所有repeat种子沿用历史PCCI的SHA256前15hex规则。没有为P1-P3额外生成无关重复推理。', '',
        '## 真实条件与错配条件', '',
        '| Group | CE(true)-CE(shuffle) | 三seed CE差 | true CE更优seed | 概率MAD | ΔACC pp | ΔBA pp | ΔAUC pp |',
        '|---|---|---|---|---|---|---|---|'])
    for group in ('P4', 'P5'):
        c = state['conditional'][group]
        means = c['metric_differences']['mean']
        lines.append('| ' + ' | '.join([group, f"{c['CE_true_minus_shuffle_mean']:.8f}",
            ' / '.join(f"{v['CE_true_minus_shuffle']:.8f}" for v in c['seeds'].values()),
            str(c['true_CE_better_seeds']) + '/3', f"{c['probability_MAD']:.8f}",
            *[num(means[k]) for k in ('ACC', 'BA', 'AUC')]]) + ' |')
    lines.extend(['', 'CE差为负才表示真实条件更好。P4/P5的错配使用同一真实TD参考权重，只改变query年龄/性别与密度gate；真实、错配、关闭均共享一次core forward。P6是错配参考训练的负对照，不混称P5真实条件。关闭分支只得到该方法已联合训练核心，不等同独立C0。', '',
        '分ASD/TD、年龄、性别、站点的逐seed配对CE、概率MAD、受益患者比例、预测翻转、AUC/BA/ACC及排序变化详见summary.conditional；站点n<20或任一类<5不报告AUC，单诊断亚组AUC未定义。', '',
        '| 分层(P5) | 每seed人数 | 平均CE真实-错配 | 概率MAD |',
        '|---|---|---|---|'])
    for name, values in state['conditional']['P5']['strata'].items():
        if not name.startswith('site_'):
            lines.append(f"| {name} | {values['subjects_per_seed']} | {values['mean_CE_true_minus_shuffle']:.8f} | {values['mean_probability_MAD']:.8f} |")
    lines.extend(['', '## 有效错配与可靠性权重', '',
        '| Seed | 实际改变条件人数 | 占比 | 有效子集CE真实-错配 | 有效子集概率MAD | 有效子集ΔAUC pp |',
        '|---|---|---|---|---|---|'])
    for seed, v in state['conditional']['P5']['effective_shuffle_subset'].items():
        lines.append(f"| {seed} | {v['n']} | {num(v['fraction'])} | {v['CE_true_minus_shuffle']:.8f} | {v['probability_MAD']:.8f} | {num(v['mean_draw_metric_difference']['AUC'])} |")
    lines.extend(['', '此子集按实际年龄或性别是否改变定义，不按标签或结果筛选；其AUC为每seed有效子集池化OOF五draw均值，不替代正式十折均值。',
        '', '| Seed | gate与CE关闭-真实 Spearman | gate与CE错配-真实 Spearman | gate与CE单位gate-真实 Spearman | CE真实-单位gate |',
        '|---|---|---|---|---|'])
    for seed, v in state['reliability']['seeds'].items():
        fields = [v['correlations'][k]['Spearman'] for k in ('CE_off_minus_true', 'CE_shuffle_minus_true', 'CE_unit_gate_minus_gated')]
        lines.append('| ' + ' | '.join([seed, *['未定义' if x is None else f'{x:.6f}' for x in fields],
            f"{v['same_checkpoint_gate_minus_unit_gate_CE']:.8f}"]) + ' |')
    lines.extend(['', '上述是同一P5 checkpoint的描述性关联与单位gate反事实；不能由gate与获益相关就宣称可靠性校准成立。P5相较P4的独立训练差异另见正式配对表；全部固定gate分箱和标签关联保存在summary.reliability。',
        '', '## 预处理边界与预注册修订', ''])
    boundary = state['preprocessing_audit']
    lines.extend([
        f"30/30缓存：PCA仅train影像拟合；规范均值仅train TD拟合；5个内部折给ASD/TD相同离折参考，每名训练被试被预测一次且不出现在自身TD参考；P4/P5偏差逐值相同。",
        '真实组P1-P5的PCA/参考权重对heldout年龄替换不变；所有组对heldout诊断标签替换不变；所有组PCA/参考权重对heldout影像替换不变。每fold保存完整crossfit索引证明与参数SHA。',
        'P6定义在正式训练前由全队列同站点错配，修订为Train/Val/Test分区内部同站点错配。用户已正式授权；旧30折边界冲突、旧缓存及源码SHA均保存在summary.p6_revision及各缓存revision中，没有把旧缓存伪装成最终版本。',
        f"修订后P6边界冲突{boundary['P6_boundary_conflict_folds']}/30；heldout条件进入训练TD参考{boundary['P6_transferred_to_TD']['max']:.0f}人。逐折验证每分区双射、站点边界、非单例无自映射；保存完整排列及SHA、实际条件改变比例、单例。",
        'P6的训练参考权重及训练features/gate对heldout年龄、性别、影像和标签扰动不变；不要求heldout推断对heldout条件扰动不变。原有P1-P5正式features/gate逐位保留，P4/P5真实偏差逐值复核；仅重建P6与必要的P4/P5辅助错配查询。',
        f"定义处理：{state['preflight'].get('P6_boundary_policy')}；授权/解释记录：{state['preflight'].get('P6_policy_resolution', '见summary.preflight')}。",
        '这不是测试诊断标签泄漏，也不消除历史RFE监督泄漏。原始表型VAE/RP/图张量未被PCND错配改动，data_input_digest在全部同fold方法间一致。',
        '', '## 数值、梯度与训练器', '',
        '五个预固定fold各比较三个完整确定性优化步骤：原生CPU锚点输入迁移路径与新PPU输入路径、强制零PCND边界下logits、CE、Graph、validation、核心梯度、Adam状态、BN、TopK及完整RNG。诊断确定性不进入正式训练。',
        f"最大数值误差：{max(v['deterministic_max_error'] for v in state['checks'].values()):.8g}；最大历史test grad/no_grad差：{max(v['historical_grad_mode_max_error'] for v in state['checks'].values()):.8g}。",
        '实际PCND两步检查首步末层有梯度、第二步隐藏层及输入偏差有梯度；核心影像、非影像及Classifier梯度均存在，未注册RP参数不进入optimizer；每次core只调用一次Classifier，BN每训练epoch只更新一次，反事实不更新BN。',
        '分支初始化使用CPU默认Generator私有seed并恢复CPU RNG，不采用会连带重置PPU RNG的torch.manual_seed。branch权重与给定初始化相同，所有组同fold逐值相同；容量1089；|delta|<=0.25；P5 gate=1与P4输出严格一致。',
        '原EarlyStopping(save=False)相等验证损失重置计数，最佳checkpoint仍严格<选取；此处继承实际历史行为而不机械采用prompt简化计数器。正式test保留历史grad-enabled单forward，预检确认与no_grad前向数值等价。',
        '没有接入任何PCND Hook，不重演实验4辅助反事实base累积问题。此三步验证不能替代对完整500epoch原生PPU随机轨迹的逐位一致性宣称。', '',
        '## 来源审计', '',
        '原论文[方法及Table5](https://arxiv.org/html/2406.14455v3)说明原模型已利用年龄、性别、站点，非影像VAE和人口图并非本轮新增；Table5的ABIDE影像/非影像/联合ACC分别76.49/52.00/82.92，AUC85.22/50.28/88.21。PCND检验的是已有多模态模型之外的条件增量，不是首次使用表型。',
        '根模型及utils、train、opt逐段审计；实验1训练/早停/锚点、实验3PCCI训练/调度/反事实、实验4独立运行器/冻结修复及报告相关段落已核对。GitHub实验3固定提交e48c177和实验4固定提交d945726源码字节一致；实验4实际只读路径为/tmp/PCCI_Attribution_20261016。',
        f"训练源指纹：`{state['training_code_sha256']}`。只读历史文件复核数：{state['final_verification']['protected_historical_files_rehashed']}。所有原始输入、30缓存、180最佳模型SHA和配对证据在summary及fold产物中，无额外复制历史checkpoint。", '',
        '## 完成情况与资源', ''])
    execution = state['execution']
    lines.extend([f"正式180/180；每组30/30；GPU0完成90、GPU1完成90，每组每卡15；峰值worker {execution['peak_workers']}。",
        f"正式任务墙钟跨度{execution['training_wall_span_seconds'] / 60:.2f}分钟；阶段耗时{json.dumps(execution['phases'], ensure_ascii=False)}。",
        f"训练尝试{execution['total_attempts_train']}次，失败尝试{execution['failed_training_attempts']}次；完整错误记录{len(state['errors'])}条。",
        '显存按16GiB/worker及8GiB余量，结合PPU利用率、16核CPU预算和cgroup真实RAM限流；资源不足只延迟启动，不杀健康任务。每worker单线程及单物理卡隔离，本进程device=cuda:0，未进行CPU神经训练。',
        f"资源采样：{json.dumps(execution['resource_extrema'], ensure_ascii=False)}。",
        '', '## 研究结论', '',
        state['research_answers']['mechanism_claim'],
        '已证明的是缓存/路径/边界与本轮同协议的配对结果；规范参考的医学机制、外部可迁移性及创新性仍未被独立控制实验确定。PCND虽不直接改变BN输入，仍通过CE改变主干联合优化，不能仅凭负结果或正结果唯一归因。',
        state['research_answers']['next_direction'],
        '', '不自行push GitHub。全部验收后才统一追加根handoff一次，绝不改写历史总结。', ''])
    atomic_text('\n'.join(lines), r.BASE / 'report.md')


def handoff_section(state):
    lines = ['## 2026-10-16 — Experiment 5 PCND_20261016', '',
        '研究假设：训练TD交叉拟合的年龄/性别规范偏差，是否提供原始MM-GTUNets和同容量影像辅助证据之外的诊断增量。',
        'P1原始PCA容量对照；P2无条件TD偏差；P3年龄；P4年龄/性别；P5同P4偏差+参考密度g；P6在Train/Val/Test分区内部同站点联合错配年龄/性别，独立拟合错配TD参考。P0只读复用历史C0。六组均1089参数、显式±delta logit校正、上限0.25，不使用Hook，不改Classifier/BN输入。',
        '', '| Group | ACC | BA | AUC | SEN | SPE | F1 |', '|---|---|---|---|---|---|---|']
    for group in ('P0', *r.GROUPS):
        f = state['groups'][group]['formal']
        lines.append('| ' + ' | '.join([group, *[f"{num(f['mean'][k])} +/- {num(f['seed_sd_ddof1'][k])}" for k in r.KEYS]]) + ' |')
    ex = state['execution']
    lines.extend(['', '单位%，先十折均值再三个seed样本SD；单次随机正式test。180/180训练、每组30折，30份缓存；GPU0/1各90任务、各组各15，P4/P5/P6共450次新增冻结推理。',
        f"正式训练墙钟跨度{ex['training_wall_span_seconds'] / 60:.2f}分钟；失败训练尝试{ex['failed_training_attempts']}；资源与各阶段耗时见summary.execution。",
        '已验证train-only PCA及训练TD的ASD/TD共同5折crossfit；heldout标签/影像不改变参考，P4/P5偏差逐值相同，五固定fold三完整优化步骤的logits/梯度/Adam/BN/TopK/RNG parity通过。分支初始化不污染PPU RNG；反事实共享一次core logits且BN不变；独立rank-AUC/混淆矩阵复核全部OOF。',
        f"P6正式预注册修订：{state['preflight'].get('P6_policy_resolution')}。30折均已验证训练参考权重不依赖heldout年龄/性别/标签；每分区双射与站点边界通过；不声称heldout推断不变。旧冲突与缓存SHA保留，不使用heldout诊断标签。",
        '保留历史RFE监督跨折泄漏、RP循环早返/未注册权重/.data投影、VAE重复加载及重采样、eval dropout和全队列转导图；所有绝对成绩不得解读为无泄漏外站泛化。',
        f"代码指纹：{state['training_code_sha256']}。原始源码、历史数据和Checkpoint共{state['final_verification']['protected_historical_files_rehashed']}个文件复核不变；完整各文件SHA见summary。",
        '五draw结果仅辅助，不替代正式成绩；P6的as-trained错配结果不混称真实条件。',
        f"P4真实-错配CE：{state['conditional']['P4']['CE_true_minus_shuffle_mean']:.8f}；P5：{state['conditional']['P5']['CE_true_minus_shuffle_mean']:.8f}（负值支持真实条件）。",
        f"P5真实条件CE改善{state['conditional']['P5']['true_CE_better_seeds']}/3seed；概率MAD={state['conditional']['P5']['probability_MAD']:.8f}；分层、受益比例、排序、翻转及重复推理指标见report/summary。",
        state['research_answers']['mechanism_claim'],
        '尚未确认：仅三重复seed的可靠性、临床校准、外部站点泛化及真正新增表型语义，不因偶然高ACC提前认定创新成立。',
        state['research_answers']['next_direction'],
        '目录：experiments/PCND_20261016；主要结果summary.json、report.md；独立代码run.py/launch.py/analyze.py/normative.py/pcnd.py；每fold一个.pt，不复制历史模型；未自行push。', ''])
    return '\n'.join(lines)


def finalize(state):
    if not state.get('all_formal_training_complete'):
        raise AssertionError('All 180 formal tasks must finish before analysis')
    state['phase'] = 'analysis'
    state['groups'] = {group: summarize_group(group, state) for group in ('P0', *r.GROUPS)}
    pairs = list(dict.fromkeys([(g, 'P0') for g in r.GROUPS] +
        [('P5', g) for g in ('P1', 'P2', 'P3', 'P4', 'P6')] +
        [(f'P{i}', f'P{i-1}') for i in range(2, 7)]))
    state['comparisons'] = {f'{a}-{b}': paired(state['groups'][a], state['groups'][b]) for a, b in pairs}
    ctx = r.load_torch(r.OLD / 'artifacts/context.pt')
    meta = r.metadata(ctx)
    state['conditional'] = {g: conditional_audit(g, state, meta) for g in ('P4', 'P5')}
    state['reliability'] = reliability_audit(state, meta)
    caches = [r.load_torch(r.feature_path(seed, fold)) for seed in r.config()['seeds'] for fold in range(10)]
    state['preprocessing_audit'] = {'caches': 30, 'P6_boundary_conflict_folds': sum(v['boundary_checks']['P6_boundary_conflict_observed'] for v in caches),
        'P6_transferred_to_TD': r.distribution([v['boundary_checks']['P6_heldout_covariate_transfers_to_train_TD'] for v in caches]),
        'train_TD_count': r.distribution([v['diagnostics']['train_td'] for v in caches]),
        'PCA_variance_sum': r.distribution([v['diagnostics']['pca_explained_sum'] for v in caches]),
        'per_fold_proofs_saved': True,
        'P6_definition_revision': 'Before formal training: cohort within-site -> separate train/val/test within-site joint age-sex shuffle',
        'P1_P5_formal_arrays_bitwise_preserved': all(v['revision']['P1_P5_formal_bitwise_preserved'] for v in caches),
        'P6_shuffle_partitions': {part: {'actual_changed_fraction': r.distribution([
            v['diagnostics']['shuffle']['partitions'][part]['changed_fraction'] for v in caches]),
            'singleton_site_groups': sum(s['singleton'] for v in caches
                for s in v['diagnostics']['shuffle']['partitions'][part]['sites'].values())}
            for part in r.SUBSETS}}
    state['execution'] = execution_audit(state)
    state['research_answers'] = research_answers(state)
    verified = verify_completed_bundle(state)
    state['final_checkpoints_sha256'] = verified.pop('checkpoints_sha256')
    state['final_features_sha256'] = verified.pop('features_sha256')
    state['final_verification'] = verified
    state['research_completed_utc'] = r.utc()
    state['phase'] = 'research_verified_handoff_pending'
    state['handoff'] = {'status': 'pending_final_once_append', 'before_sha256': state['audit']['handoff_before_sha256']}
    write_report(state)
    state['report_sha256'] = r.sha_file(r.BASE / 'report.md')
    r.atomic_json(state)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--handoff-text', action='store_true')
    args = parser.parse_args()
    state = r.guard()
    if args.handoff_text:
        if not state.get('final_verification', {}).get('passed'):
            raise RuntimeError('Do not update handoff before full acceptance')
        print(handoff_section(state))
    else:
        finalize(state)

