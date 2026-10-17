"""Read-only attribution of historical PCND references and frozen evidence branches."""
from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import ast
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import queue
import subprocess
import time
import traceback

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
import torch

ROOT = Path('/data/adodas/MM-GTUNets')
E5 = ROOT / 'experiments/PCND_20261016'
OLD = ROOT / 'experiments/DRPA_20261014'
OUT = ROOT / 'experiments/PCND_DualStage_20261016'
SEEDS = (911, 912, 913)
FOLDS = tuple(range(10))
CFG = json.loads((OUT / 'config.json').read_text())
norm = pcnd = None
VERIFIED = {}
KEYS = ('ACC', 'BA', 'AUC', 'SEN', 'SPE', 'F1')
torch.set_num_threads(1)


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_hash(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    a = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode())
        h.update(array_hash(state[key]).encode())
    return h.hexdigest()


def safe_load(path: Path):
    path = Path(path)
    assert path.is_file(), f'Missing verified artifact: {path}'
    assert str(path) in VERIFIED, f'Artifact not SHA-verified before load: {path}'
    return torch.load(path, map_location='cpu', weights_only=False, mmap=True)


def import_exact(name: str, file: Path):
    assert str(file) in VERIFIED
    spec = importlib.util.spec_from_file_location(name, str(file))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def import_readonly_modules():
    global norm, pcnd
    norm = import_exact('pcnd_e5_normative_readonly', E5 / 'normative.py')
    pcnd = import_exact('pcnd_e5_branch_readonly', E5 / 'pcnd.py')


def atomic_text(value, path):
    path = Path(path)
    assert path.parent == OUT
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    try:
        with tmp.open('w') as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_json(value, path):
    atomic_text(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n', path)


def log(event, **fields):
    line = json.dumps({'utc': utc(), 'event': event, **fields}, ensure_ascii=False, allow_nan=False)
    with (OUT / 'run.log').open('a') as stream:
        stream.write(line + '\n')
        stream.flush()
    print(line, flush=True)


def ce(p, y):
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
    y = np.asarray(y)
    return -np.log(np.where(y == 0, p, 1 - p))


def describe(a):
    a = np.asarray(a, dtype=np.float64).ravel()
    assert a.size > 0 and np.isfinite(a).all()
    return {'n': int(a.size), 'mean': float(a.mean()),
            'sd': float(a.std(ddof=1)) if a.size > 1 else None,
            'median': float(np.median(a)), 'p05': float(np.quantile(a, .05)),
            'p95': float(np.quantile(a, .95)), 'min': float(a.min()), 'max': float(a.max())}


def metrics(p, y, small_group=False):
    p, y = np.asarray(p, float), np.asarray(y, int)
    assert len(p) == len(y) and len(y)
    a, b = y == 0, y == 1
    pred = p >= .5
    tp, tn, fp, fn = int((pred & a).sum()), int((~pred & b).sum()), int((pred & b).sum()), int((~pred & a).sum())
    sen = tp / (tp + fn) if a.any() else None
    spe = tn / (tn + fp) if b.any() else None
    auc = None
    eligible = a.any() and b.any()
    if small_group:
        eligible = eligible and len(y) >= 20 and a.sum() >= 5 and b.sum() >= 5
    if eligible:
        ranks = rankdata(p, method='average')
        n = int(a.sum())
        auc = float((ranks[a].sum() - n * (n + 1) / 2) / (n * int(b.sum())))
    return {'ACC': float((pred == a).mean()), 'BA': (sen + spe) / 2 if sen is not None and spe is not None else None,
            'AUC': auc, 'SEN': sen, 'SPE': spe, 'F1': 2 * tp / max(2 * tp + fp + fn, 1)}


def metadata(ctx):
    ids = [str(v) for v in ctx['ids']]
    assert len(ids) == len(set(ids)) == 871
    csv_path = ROOT / 'dataset/ABIDE_aal/Phenotypic_V1_0b_preprocessed1.csv'
    assert str(csv_path) in VERIFIED
    with csv_path.open() as stream:
        rows = list(csv.DictReader(stream))
    lookup = {}
    for row in rows:
        key = str(row['SUB_ID'])
        assert key not in lookup, f'Duplicate CSV SUB_ID: {key}'
        lookup[key] = row
    chosen = [lookup[v] for v in ids]
    meta = {'ids': np.array(ids), 'age': np.array([float(v['AGE_AT_SCAN']) for v in chosen]),
            'sex': np.array([int(v['SEX']) for v in chosen]), 'site': np.array([v['SITE_ID'] for v in chosen]),
            'labels': np.array([int(v['DX_GROUP']) - 1 for v in chosen])}
    assert np.array_equal(meta['labels'], ctx['labels'])
    assert np.array_equal(meta['site'], ctx['site_names'])
    assert np.array_equal(np.bincount(meta['labels']), [403, 468])
    assert set(meta['sex']) <= {1, 2} and np.isfinite(meta['age']).all()
    return meta


def fixed_oof_crossfit(x, y, meta, split, seed, fold, group, cache):
    """Reconstruct the registered analytic Ridge on the existing PCA coordinates."""
    tr = np.asarray(split['train'], np.int64)
    td = tr[np.asarray(y)[tr] == 1]
    pc = cache['pca_state']
    components = np.asarray(pc['components'], np.float64)
    mean = np.asarray(pc['mean'], np.float64)
    sd = np.asarray(pc['pc_std'], np.float64)
    assert components.shape == (32, x.shape[1])
    z = ((np.asarray(x, np.float64) - mean) @ components.T) / sd
    age, sex = np.asarray(meta['age'], float), np.asarray(meta['sex'], int)
    a0, asc, sm = float(age[td].mean()), max(float(age[td].std(ddof=1)), 2.0), float(np.mean(sex[td] == 2))
    kind = {'P2': 'global', 'P3': 'age', 'P4': 'age_sex'}[group]
    d = norm.design(age, sex, kind, a0, asc, sm)
    pred = np.empty_like(z)
    inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=norm.stable_seed(seed, fold, 'PCND-inner'))
    weights, proofs = [], []
    for aa, bb in inner.split(tr, np.asarray(y)[tr]):
        fit = tr[aa]
        fit_td, query = fit[np.asarray(y)[fit] == 1], tr[bb]
        assert not np.intersect1d(fit_td, query).size
        w = norm.ridge_fit(d[fit_td], z[fit_td], lamb=10.0)
        pred[query] = d[query] @ w
        weights.append(w)
        proofs.append({'reference_td': fit_td, 'query': query})
    w_full = norm.ridge_fit(d[td], z[td], lamb=10.0)
    outside = np.r_[split['val'], split['test']]
    pred[outside] = d[outside] @ w_full
    scale = np.maximum((z[td] - pred[td]).std(axis=0, ddof=1), .25)
    deviation = np.clip((z - pred) / scale, -4, 4).astype('float32')
    saved = cache['features'][group]['features']
    err = float(np.max(np.abs(deviation - saved)))
    assert np.allclose(deviation, saved, rtol=2e-5, atol=2e-5), (group, seed, fold, err)
    return {'z': z, 'mu_reference': pred, 'scale': scale, 'deviation': deviation, 'train_td': td,
            'reconstruction_max_abs': err, 'w_full': w_full, 'weights': weights,
            'a0': a0, 'asc': asc, 'sm': sm, 'proofs': proofs}


def restore_same_fold_data(seed, fold):
    cache = safe_load(E5 / f'features/s{seed}_f{fold}.pt')
    x = safe_load(OLD / f'artifacts/seed_{seed}/selected_features.pt')
    split = safe_load(OLD / f'artifacts/seed_{seed}/fold_{fold}/splits.pt')
    return cache, np.asarray(x, np.float64), split


def cosine(a, b):
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    valid = den > 1e-12
    result = np.zeros(len(a), float)
    result[valid] = np.sum(a[valid] * b[valid], axis=1) / den[valid]
    return result, int((~valid).sum())


def changes(a, b, zstd):
    delta = np.asarray(b, float) - np.asarray(a, float)
    cos, undefined = cosine(np.asarray(a, float), np.asarray(b, float))
    raw = {'L2': np.linalg.norm(delta, axis=1), 'mean_square': np.mean(delta * delta, axis=1),
           'cosine': cos, 'relative_z_std_RMS': np.sqrt(np.mean((delta / zstd) ** 2, axis=1))}
    return {**{k: describe(v) for k, v in raw.items()}, 'zero_norm_cosine_undefined_n': undefined}, raw


def stage_a(seed, fold, x, y, meta, split, cache):
    obj = {g: fixed_oof_crossfit(x, y, meta, split, seed, fold, g, cache) for g in ('P2', 'P3', 'P4')}
    t, tr = np.asarray(split['test'], int), np.asarray(split['train'], int)
    td, z = obj['P2']['train_td'], obj['P2']['z']
    assert np.allclose(z, obj['P3']['z']) and np.allclose(z, obj['P4']['z'])
    zstd = z[tr].std(axis=0, ddof=1)
    assert np.allclose(zstd, 1, atol=2e-10)
    C = np.asarray(cache['pca_state']['components'], float)
    orth_err = float(np.abs(C @ C.T - np.eye(32)).max())
    assert orth_err < 2e-10
    v = x[tr][y[tr] == 0].mean(0) - x[tr][y[tr] == 1].mean(0)
    v32 = C.T @ (C @ v)
    ratio = float(v32 @ v32 / max(float(v @ v), 1e-12))
    pca_raw_err = float(np.abs(np.clip(z, -4, 4).astype('float32') - cache['features']['P1']['features']).max())
    assert pca_raw_err < 2e-5
    report = {'seed': seed, 'fold': fold, 'n_test': len(t), 'n_train_td': len(td),
              'pca_ratio_sum': float(np.sum(cache['boundary_checks']['pca_variance_ratio'])),
              'training_mean_difference_retained_ratio': ratio,
              'pca_orthogonality_max_abs': orth_err, 'P1_reconstruction_max_abs': pca_raw_err,
              'ridge_reconstruction_error': {g: obj[g]['reconstruction_max_abs'] for g in obj}}
    subject = {}
    sse = {g: np.sum((z[td] - obj[g]['mu_reference'][td]) ** 2, axis=0) for g in obj}
    tss = np.sum((z[td] - z[td].mean(0)) ** 2, axis=0)
    report['TD_OOF_SSE_per_dim'] = {g: a.tolist() for g, a in sse.items()}
    report['TD_TSS_per_dim'] = tss.tolist()
    for a, b in (('P2', 'P3'), ('P3', 'P4'), ('P2', 'P4')):
        key = f'{a}_to_{b}'
        om, nm = obj[a]['mu_reference'][t], obj[b]['mu_reference'][t]
        od, nd = obj[a]['deviation'][t], obj[b]['deviation'][t]
        mstats, mraw = changes(om, nm, zstd)
        dstats, draw = changes(od, nd, zstd)
        sa, sb = obj[a]['scale'], obj[b]['scale']
        mean_part = (om - nm) / sb
        scale_part = (z[t] - om) * (1 / sb - 1 / sa)
        unclipped_change = (z[t] - nm) / sb - (z[t] - om) / sa
        identity_error = float(np.abs(unclipped_change - mean_part - scale_part).max())
        assert identity_error < 1e-12
        clipped_exact = np.clip((z[t] - nm) / sb, -4, 4) - np.clip((z[t] - om) / sa, -4, 4)
        clip_part = clipped_exact - unclipped_change
        report[key] = {'mu_change': mstats, 'deviation_change': dstats,
                      'mu_change_L2': mstats['L2'], 'deviation_change_L2': dstats['L2'],
                      'ref_scale_change_L2': describe(sb - sa), 'scale_difference_norm_L2': float(np.linalg.norm(sb-sa)),
                      'source_scale': describe(sa), 'target_scale': describe(sb),
                      'unclipped_decomposition_max_abs': identity_error,
                      'mean_part_L2': describe(np.linalg.norm(mean_part, axis=1)),
                      'scale_part_L2': describe(np.linalg.norm(scale_part, axis=1)),
                      'clipping_remainder_L2': describe(np.linalg.norm(clip_part, axis=1))}
        subject[key] = {**{f'mu_{k}': v for k, v in mraw.items()}, **{f'deviation_{k}': v for k, v in draw.items()}}
        gain = 1 - sse[b] / np.maximum(sse[a], 1e-12)
        r2gain = (sse[a] - sse[b]) / np.maximum(tss, 1e-12)
        report[f'{b}_td_crossfit_vs_{a}'] = {
            'pooled_sse_gain': float(1 - sse[b].sum() / max(float(sse[a].sum()), 1e-12)),
            'pooled_sse_difference': float((sse[b] - sse[a]).sum()),
            'per_dim_gain': gain.tolist(), 'per_dim_R2_gain': r2gain.tolist(),
            'mean_dim_gain': float(gain.mean()), 'median_dim_gain': float(np.median(gain)),
            'positive_dimensions': int((gain > 0).sum()), 'positive_dim_proportion': float((gain > 0).mean()),
            'mean_R2_gain': float(r2gain.mean()), 'median_R2_gain': float(np.median(r2gain)),
            'source_R2_per_dim': (1-sse[a]/np.maximum(tss, 1e-12)).tolist(),
            'target_R2_per_dim': (1-sse[b]/np.maximum(tss, 1e-12)).tolist()}
    p, shuffle = norm.make_shuffle(meta['site'], meta['age'], meta['sex'], seed, fold, split)
    assert shuffle == cache['diagnostics']['shuffle']
    assert shuffle['scope'] == 'partition_within_site'
    assert np.array_equal(np.sort(p), np.arange(871))
    for part in ('train', 'val', 'test'):
        idx = np.asarray(split[part], int)
        assert np.array_equal(np.sort(p[idx]), np.sort(idx))
        assert np.all(meta['site'][idx] == meta['site'][p[idx]])
    real = obj['P4']
    design_alt = norm.design(meta['age'][p], meta['sex'][p], 'age_sex', real['a0'], real['asc'], real['sm'])
    mu_alt = design_alt[t] @ real['w_full']
    alt = np.clip((z[t] - mu_alt) / real['scale'], -4, 4).astype('float32')
    sf = np.asarray(cache['features']['P4']['shuffled_features'])[t]
    err = float(np.abs(alt - sf).max())
    assert np.allclose(alt, sf, atol=2e-5, rtol=2e-5), ('P4 shuffled query', seed, fold, err)
    assert np.array_equal(cache['features']['P4']['features'], cache['features']['P5']['features'])
    assert np.array_equal(cache['features']['P4']['shuffled_features'], cache['features']['P5']['shuffled_features'])
    for part, query_ids in [('test', t)]:
        true_gate = norm.density_gate(meta['age'][query_ids], meta['sex'][query_ids], meta['age'][td], meta['sex'][td])
        alt_gate = norm.density_gate(meta['age'][p[query_ids]], meta['sex'][p[query_ids]], meta['age'][td], meta['sex'][td])
        assert np.array_equal(true_gate, cache['features']['P5']['gate'][query_ids])
        assert np.array_equal(alt_gate, cache['features']['P5']['shuffled_gate'][query_ids])
    mc, mr = changes(real['mu_reference'][t], mu_alt, zstd)
    dc, dr = changes(real['deviation'][t], sf, zstd)
    report['P4_true_vs_shuffled'] = {'mu_change': mc, 'deviation_change': dc,
        'deviation_reconstruction_max_abs': err, 'effective_changed_n': int(shuffle['partitions']['test']['changed_count']),
        'effective_changed_fraction': float(shuffle['partitions']['test']['changed_fraction']),
        'P4_full_reference_weight_max_abs': float(np.abs(real['w_full']-cache['boundary_checks']['true_reference_weights'][-1]).max()),
        'query_only_reference_weights_fixed': True}
    subject['P4_true_vs_shuffled'] = {**{f'mu_{k}': v for k, v in mr.items()}, **{f'deviation_{k}': v for k, v in dr.items()}}
    report['P4_true_vs_shuffled_deviation_L2'] = dc['L2']
    return report, subject


def ce_attribution(p0, p5true, p5off, p5shuffle, p5unit, y):
    p0, pt, po, ps, pu = [np.asarray(x, np.float64) for x in (p0, p5true, p5off, p5shuffle, p5unit)]
    assert p0.shape == pt.shape == po.shape == ps.shape == pu.shape
    assert p0.shape == (5, len(y))
    assert all(np.isfinite(a).all() and np.all((a >= 0) & (a <= 1)) for a in (p0, pt, po, ps, pu))
    C0, CT, CO, CS, CU = [ce(a, y) for a in (p0, pt, po, ps, pu)]
    associated, direct, total = CO-C0, CT-CO, CT-C0
    assert np.allclose(total, associated+direct, atol=1e-12, rtol=1e-12)
    is_correct_off = ((po >= .5) == (np.asarray(y)[None, :] == 0))
    is_correct_true = ((pt >= .5) == (np.asarray(y)[None, :] == 0))
    rescue, harm = (~is_correct_off) & is_correct_true, is_correct_off & (~is_correct_true)
    assert np.all(is_correct_true.sum(1)-is_correct_off.sum(1) == rescue.sum(1)-harm.sum(1))
    return {'CE_delta_total': total, 'CE_delta_core_training_associated': associated,
            'CE_delta_direct_branch': direct, 'CE_delta_true_minus_shuffled': CT-CS,
            'CE_delta_true_minus_unit_gate': CT-CU,
            'rescue_count': int(rescue.sum()), 'harm_count': int(harm.sum()),
            'flips': int((rescue | harm).sum()), 'rescue': rescue, 'harm': harm,
            'correct_condition_prob_MAD': float(np.abs(pt-ps).mean()),
            'direct_prob_MAD': float(np.abs(pt-po).mean()),
            'delta_off_native_draw_std_per_subject': po.std(0, ddof=1),
            'identity_max_abs': float(np.abs(total-associated-direct).max())}


SOURCE_NOTES = {
    'model/mm_gtunets.py': 'VAE reparameterizes at eval and reloads saved weights; joint16, classifier Linear16-256/ReLU/BN/Linear2 once. Attention tensor reconstruction detaches scores; RP and 1x1 compatibility forward retain .data projection.',
    'model/gtunet.py': 'Two modality-specific GTUNets restore all nodes; F.dropout without training argument remains stochastic at eval.',
    'model/rp_graph.py': 'return remains inside phenotype loop; categorical motivation edges retain historical behavior.',
    'train_mm_gtunets.py': 'Historical x=feature_selection(x,...) overwrites features across folds; classifier optimization uses CE plus graph loss.',
    'opt.py': 'Output dimension16, hidden128, scores SITE_ID/SEX/AGE_AT_SCAN; original defaults retained in historical experiment.',
    'experiments/PCND_20261016/normative.py': 'Train PCA32; trainTD Ridge10 on intercept/age/age2/additive sex; fivefold crossfit; per-dimension TD OOF residual scale floor0.25; clip[-4,4]; tenth-neighbor density gate; joint partition-site permutation v2.',
    'experiments/PCND_20261016/pcnd.py': 'Frozen 1089-parameter Linear32-32/GELU/Linear32-1; zero final initialization; delta=.25*tanh(net(d))*g; adds [delta,-delta] to logits.',
    'experiments/PCND_20261016/run.py': 'Historical train_one jointly optimized core+branch; strict best validation CE+graph checkpoint; saved full epochs. frozen_one stored same-core counterfactuals and five seeds; branch has no sampling.',
    'experiments/PCND_20261016/analyze.py': 'Audited get_fold(P0), fold-then-seed aggregation, CE sign, mean-draw probability metrics and input checks. No import or call of this historical runner.',
    'experiments/PCND_20261016/config.json': 'Registered P6 revision is partition_within_site; repeats only P4/P5/P6; all hyperparameters historical and read-only.',
    'experiments/PCND_20261016/report.md': 'Prior formal and auxiliary results used only as historical records and numerical cross-checks; new Stage A/B statistics recomputed independently.'
}


def input_audit():
    s = json.loads((E5 / 'summary.json').read_text())
    assert s['final_verification']['passed'] and s['phase'] == 'complete'
    paths = {}
    def add(path, expected=None, role='historical input'):
        p = Path(path)
        actual = sha256(p)
        assert expected is None or actual == expected, f'SHA256 mismatch: {p}; expected {expected}; got {actual}'
        paths[str(p)] = {'sha256': actual, 'bytes': p.stat().st_size, 'role': role,
                         'matched_historical_registered_hash': expected is not None}
    add(E5 / 'summary.json', role='read-only historical summary and embedded P0 OOF/five-draw bank')
    for rel, expected in s['audit']['source_sha256'].items():
        add(ROOT / rel, expected, 'protected original source; checksum only for helper files')
    for rel, expected in s['formal_source_sha256'].items():
        add(E5 / rel, expected, 'registered experiment5 source')
    combined = hashlib.sha256(''.join(paths[str(E5 / n)]['sha256'] for n in
                                      ('run.py', 'normative.py', 'pcnd.py', 'config.json')).encode()).hexdigest()
    assert combined == s['training_code_sha256'], 'Experiment5 combined training fingerprint mismatch'
    add(E5 / 'report.md', s['report_sha256'], 'historical report')
    add(ROOT / 'prompt', role='current analysis authorization')
    historical = [OLD / 'artifacts/context.pt', ROOT / 'dataset/ABIDE_aal/Phenotypic_V1_0b_preprocessed1.csv']
    for seed in SEEDS:
        historical.append(OLD / f'artifacts/seed_{seed}/selected_features.pt')
        for fold in FOLDS:
            historical.append(OLD / f'artifacts/seed_{seed}/fold_{fold}/splits.pt')
            rel = f'features/s{seed}_f{fold}.pt'
            add(E5 / rel, s['final_features_sha256'][rel], 'experiment5 feature cache')
            for group in ('P4', 'P5'):
                rel = f'runs/{group}/s{seed}_f{fold}.pt'
                add(E5 / rel, s['final_checkpoints_sha256'][rel], 'frozen checkpoint; CPU mmap only for core')
    for p in historical:
        expected = s['audit']['manifest'].get(str(p))
        assert expected is not None, f'Historical registered input hash unavailable: {p}'
        add(p, expected['sha256'], 'historical data or split anchor actually read this round')
    for rel, note in SOURCE_NOTES.items():
        text = (ROOT / rel).read_text()
        assert text, rel
        assert str(ROOT / rel) in paths
    local_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    known_remote = subprocess.check_output(['git', 'rev-parse', 'origin/main'], cwd=ROOT, text=True).strip()
    return s, paths, {'passed': True, 'initial_input_files': len(paths),
        'feature_files': 30, 'checkpoint_files': 60, 'historical_data_split_files': len(historical),
        'protected_root_source_files': len(s['audit']['source_sha256']),
        'experiment5_registered_source_files': len(s['formal_source_sha256']),
        'local_git_HEAD': local_commit, 'local_origin_main_tracking_commit': known_remote,
        'local_HEAD_differs_from_known_uploaded_version': local_commit != known_remote,
        'experiment5_training_combined_sha256': combined, 'source_read_findings': SOURCE_NOTES,
        'paper': {'url': CFG['source_paper'], 'sections_read': CFG['paper_audited_sections'],
                  'accessed_utc': utc(),
                  'finding': 'MRRL aligns FC/RFE and VAE phenotype latents, AMRS and multimodal similarity build population graph; dual GTUNets and attention feed MLP. Table5 ABIDE ACC imaging/nonimaging/both76.49/52.00/82.92; AUC85.22/50.28/88.21. Existing phenotype contribution does not establish extra normative patient-level complementarity.'},
        'phenobrain': {'url': 'https://proceedings.mlr.press/v306/meng26f.html',
                      'finding': 'Primary ICML2026 abstract describes phenotype-conditioned multi-hop routing and phenotype-guided attention; simple conditioning is not a first-use claim. No head-to-head experiment or full novelty proof performed.'},
        'historical_1596_files_not_rescanned': True,
        'RNG_pairing_limit': 'P0/P4/P5 saved five seed identifiers match SHA policy. P0 has no saved full RNG states in this bank; equal seeds do not prove identical backend consumption or identical stochastic realizations across trained models.'}


def validate_cache(cache, split, meta, s, seed, fold):
    assert cache['state'] == 'complete' and cache['seed'] == seed and cache['fold'] == fold
    tr, va, te = [np.asarray(split[k], int) for k in ('train', 'val', 'test')]
    assert len(np.unique(np.r_[tr, va, te])) == sum(map(len, (tr, va, te))) == 871
    assert np.array_equal(np.sort(np.r_[tr, va, te]), np.arange(871))
    calculated = digest({g: {k: array_hash(v) for k, v in sorted(f.items())} for g, f in sorted(cache['features'].items())})
    assert calculated == cache['features_sha256'], ('numeric cache digest', seed, fold)
    for g in ('P1', 'P2', 'P3', 'P4', 'P5', 'P6'):
        f = cache['features'][g]
        assert f['features'].shape == (871, 32) and f['gate'].shape == (871,)
        assert np.isfinite(f['features']).all() and np.isfinite(f['gate']).all()
        assert np.abs(f['features']).max() <= 4 and f['gate'].min() >= .25 and f['gate'].max() <= 1
    for g, expected in cache['revision']['P1_P5_formal_sha256_before'].items():
        assert expected == {k: array_hash(cache['features'][g][k]) for k in ('features', 'gate')}
    assert np.array_equal(cache['features']['P4']['features'], cache['features']['P5']['features'])
    assert np.array_equal(cache['boundary_checks']['pca_fit_indices'], tr)
    pc = cache['pca_state']
    assert state_hash({'components': pc['components'], 'mean': pc['mean'], 'variance': pc['explained_variance'],
                       'singular': pc['singular']}) == cache['boundary_checks']['pca_sha256']
    assert not cache['boundary_checks']['P6_boundary_conflict_observed']
    assert cache['boundary_checks']['P6_heldout_covariate_transfers_to_train_TD'] == 0
    covered = []
    for proof in cache['boundary_checks']['crossfit']:
        reference, query_ids = np.asarray(proof['reference_td']), np.asarray(proof['heldout_train'])
        assert np.isin(reference, tr).all() and np.all(meta['labels'][reference] == 1)
        assert not np.intersect1d(reference, query_ids).size
        covered.extend(query_ids.tolist())
    assert sorted(covered) == sorted(tr.tolist())
    inputs = dict(s['audit']['anchors'][f'{seed}/{fold}']['input_sha256'])
    for p in (OLD / 'artifacts/context.pt', ROOT / 'dataset/ABIDE_aal/Phenotypic_V1_0b_preprocessed1.csv'):
        inputs[str(p.relative_to(ROOT))] = VERIFIED[str(p)]['sha256']
    assert cache['fingerprint'] == {'input_sha256': inputs,
        'normative_sha256': VERIFIED[str(E5 / 'normative.py')]['sha256'],
        'config_sha256': VERIFIED[str(E5 / 'config.json')]['sha256']}


def validate_saved(saved, cache, split, meta, s, seed, fold, group):
    assert saved['state'] == 'complete' and saved['group'] == group
    assert saved['seed'] == seed and saved['fold'] == fold
    assert saved['training_code_sha256'] == s['training_code_sha256']
    assert saved['config_digest'] == digest(s['config'])
    assert saved['input_sha256'] == s['audit']['anchors'][f'{seed}/{fold}']['input_sha256']
    assert saved['feature_sha256'] == s['final_features_sha256'][f'features/s{seed}_f{fold}.pt']
    assert state_hash(saved['branch']) == saved['best_branch_sha256']
    assert state_hash({**saved['core']['state'], 'unregistered_rp_weights': saved['core']['rp_weights']}) == saved['best_core_sha256']
    idx = np.asarray(split['test'], int)
    oof = saved['oof']
    assert np.array_equal(oof['indices'], idx) and np.array_equal(oof['labels'], meta['labels'][idx])
    assert np.array_equal(oof['ids'], meta['ids'][idx]) and np.array_equal(oof['sites'], meta['site'][idx])
    f = saved['frozen']
    assert np.asarray(f['test_prob_asd']).shape == (5, len(idx))
    assert np.array_equal(f['test_prob_asd'], f['counterfactual_prob_asd']['true'])
    expected_seeds = [int(hashlib.sha256(f'{seed}/{fold}/PCCI-repeat-{v}'.encode()).hexdigest()[:15], 16) % (2**32)
                      for v in s['config']['repeat_seed_bases']]
    baseline = s['audit']['baseline'][f'{seed}/{fold}']
    assert np.array_equal(baseline['indices'], idx)
    assert np.array_equal(baseline['labels'], oof['labels'])
    assert np.array_equal(baseline['ids'], oof['ids']) and np.array_equal(baseline['sites'], oof['sites'])
    assert f['draw_seeds'] == expected_seeds == s['audit']['baseline_frozen'][f'{seed}/{fold}']['draw_seeds']
    assert f['same_native_logits_for_all_counterfactuals'] and f['BN_unchanged']
    assert len(saved['epochs']) == saved['epochs_run']
    assert saved['epochs'][saved['best_epoch']]['val_loss'] == saved['best_validation_loss']
    for variant, p in f['counterfactual_prob_asd'].items():
        p = np.asarray(p)
        assert p.shape == (5, len(idx)) and np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
        delta = np.asarray(f['counterfactual_delta'][variant])
        assert delta.shape == p.shape and np.isfinite(delta).all()
        assert np.array_equal(delta, np.repeat(delta[:1], 5, axis=0)), ('branch sampling unexpectedly differs', seed, fold, group)
    return {'oof_labels_ids_site_order': True, 'five_seeds_match_P0': True,
            'core_CPU_state_checksum': True, 'branch_state_checksum': True,
            'recorded_BN_unchanged': True, 'branch_delta_identical_across_draws': True}


@torch.no_grad()
def branch_decomposition(saved, cache, group, device):
    assert group in ('P4', 'P5')
    model = pcnd.PCNDEvidence(dim=32, hidden=32, logit_cap=.25).to(device)
    model.load_state_dict(saved['branch'], strict=True)
    model.eval()
    model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters()) == 1089
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in model.modules())
    state_before = state_hash(model.state_dict())
    assert state_before == saved['best_branch_sha256']
    f = cache['features'][group]
    x = torch.as_tensor(f['features'], dtype=torch.float32, device=device)
    xs = torch.as_tensor(f['shuffled_features'], dtype=torch.float32, device=device)
    g = torch.as_tensor(f['gate'], dtype=torch.float32, device=device)
    gs = torch.as_tensor(f['shuffled_gate'], dtype=torch.float32, device=device)
    logits = torch.zeros((len(x), 2), dtype=torch.float32, device=device)
    _, dt = model(logits, x, g)
    _, dfixed = model(logits, xs, g)
    _, ds = model(logits, xs, gs)
    dt, dfixed, ds = [v.cpu().numpy() for v in (dt, dfixed, ds)]
    feature_part, gate_part = dt-dfixed, dfixed-ds
    assert np.allclose(dt-ds, feature_part+gate_part, atol=1e-6, rtol=1e-6)
    if group == 'P4':
        assert np.max(np.abs(gate_part)) < 1e-7
    assert state_hash(model.state_dict()) == state_before
    return {'true_delta': dt, 'shuffled_delta': ds, 'feature_part': feature_part, 'gate_part': gate_part,
            'last_weight_l2': float(model.net[-1].weight.norm().item()),
            'last_bias': float(model.net[-1].bias.item()),
            'first_weight_l2': float(model.net[0].weight.norm().item()),
            'weight_unchanged': True, '_model': model}


def logodds_check(p, off, delta):
    p, off, delta = np.asarray(p, float), np.asarray(off, float), np.asarray(delta, float)
    eps = CFG['logodds_numeric_probability_margin']
    keep = (p > eps) & (p < 1-eps) & (off > eps) & (off < 1-eps)
    assert keep.any(), 'No non-extreme probabilities for numeric logodds validation'
    actual = np.log(p[keep])-np.log1p(-p[keep])-np.log(off[keep])+np.log1p(-off[keep])
    err = np.abs(actual - 2 * delta[keep])
    assert err.max() <= CFG['logodds_numeric_atol'], ('logodds mismatch', float(err.max()))
    return {'max_abs': float(err.max()), 'mean_abs': float(err.mean()), 'kept': int(keep.sum()),
            'excluded_numeric_only': int((~keep).sum()), 'margin': eps, 'atol': CFG['logodds_numeric_atol']}


def epoch_summary(saved):
    rows = saved['epochs']
    gradients = [r['gradient_norms'] for r in rows if 'gradient_norms' in r]
    selected = sorted(set([0, saved['best_epoch'], len(rows)-1]))
    fields = ('epoch', 'train_ce', 'train_graph', 'val_ce', 'val_graph', 'val_loss', 'train_delta_abs', 'val_delta_abs')
    return {'best_epoch': saved['best_epoch'], 'epochs_run': len(rows),
            'epoch_log_resolution': 'every epoch for loss/delta; gradients first3 and epoch%25==0',
            'train_delta_abs': describe([r['train_delta_abs'] for r in rows]),
            'val_delta_abs': describe([r['val_delta_abs'] for r in rows]),
            'gradient_record_count': len(gradients),
            'gradient_norms': {k: describe([g[k] for g in gradients]) for k in gradients[0]} if gradients else 'UNAVAILABLE',
            'selected_epoch_records': [{k: rows[i].get(k) for k in fields} for i in selected]}


@torch.no_grad()
def branch_analysis(saved, cache, split, group, device):
    d = branch_decomposition(saved, cache, group, device)
    model = d.pop('_model')
    t, tr = np.asarray(split['test'], int), np.asarray(split['train'], int)
    f = cache['features'][group]
    true, shuffle = d['true_delta'][t], d['shuffled_delta'][t]
    frozen = saved['frozen']
    checks = {}
    for name, value in [('true', true), ('shuffled', shuffle)]:
        old = np.asarray(frozen['counterfactual_delta'][name])
        err = float(np.abs(old-value[None, :]).max())
        assert np.allclose(old, value[None, :], atol=3e-6, rtol=3e-5), (group, 'saved delta', err)
        checks[name] = err
    x = torch.as_tensor(f['features'], dtype=torch.float32, device=device)
    gate = torch.as_tensor(f['gate'], dtype=torch.float32, device=device)
    mean_feature = np.asarray(f['features'])[tr].mean(0)
    mean_x = torch.as_tensor(np.repeat(mean_feature[None, :], 871, axis=0), dtype=torch.float32, device=device)
    _, fixed = model(torch.zeros((871, 2), device=device), mean_x, gate)
    fixed = fixed.cpu().numpy()
    hidden = model.net[1](model.net[0](x)).cpu().numpy()
    signed = model.net(x).squeeze(-1).cpu().numpy()
    primary = frozen['counterfactual_prob_asd']
    ident = logodds_check(primary['true'], primary['off'], frozen['counterfactual_delta']['true'])
    out = {'last_weight_l2': d['last_weight_l2'], 'last_bias': d['last_bias'], 'first_weight_l2': d['first_weight_l2'],
           'frozen_parameter_count': 1089, 'state_unchanged': d['weight_unchanged'],
           'saved_delta_max_abs': checks, 'logodds_identity': ident,
           'delta_difference_identity_max_abs': float(np.abs(d['true_delta']-d['shuffled_delta']-d['feature_part']-d['gate_part']).max()),
           'delta_test_true_minus_shuffled': describe(true-shuffle),
           'delta_test_abs_true_minus_shuffled': describe(np.abs(true-shuffle)),
           'feature_part_test': describe(d['feature_part'][t]), 'feature_part_abs_test': describe(np.abs(d['feature_part'][t])),
           'gate_part_test': describe(d['gate_part'][t]), 'gate_part_abs_test': describe(np.abs(d['gate_part'][t])),
           'feature_gate_cross_second_moment_test': float(np.mean(d['feature_part'][t]*d['gate_part'][t])),
           'train_mean_feature_replacement_is_input_sensitivity_only': True}
    for name, ids in [('train', tr), ('test', t)]:
        values = d['true_delta'][ids].astype(float)
        replacement = fixed[ids].astype(float)
        out[name] = {'delta': describe(values), 'delta_abs': describe(np.abs(values)),
                     'between_subject_delta_variance': float(values.var(ddof=1)),
                     'mean_feature_same_gate_delta': describe(replacement),
                     'mean_feature_replacement_abs_change': describe(np.abs(values-replacement)),
                     'signed_MLP_output': describe(signed[ids]),
                     'hidden_activation_coordinate_sd': describe(hidden[ids].std(0, ddof=1)),
                     'deviation_coordinate_sd': describe(np.asarray(f['features'])[ids].std(0, ddof=1)),
                     'saturated_abs_tanh_gt_point99_fraction': float(np.mean(np.abs(np.tanh(signed[ids])) > .99))}
    assert state_hash(model.state_dict()) == saved['best_branch_sha256']
    raw = {'delta_true': true, 'delta_shuffle': shuffle, 'delta_feature_part': d['feature_part'][t],
           'delta_gate_part': d['gate_part'][t], 'delta_mean_feature': fixed[t]}
    del model
    return out, raw


def synthetic_tests():
    tests = []
    def passed(name):
        tests.append({'name': name, 'passed': True})
    assert ce(.8, 0) < ce(.2, 0) and ce(.2, 1) < ce(.8, 1)
    passed('CE sign and ASD label0')
    assert np.array_equal(np.where(np.array([.8, .2, .5]) >= .5, 0, 1), [0, 1, 0])
    passed('ASD probability direction and threshold tie')
    base = np.array([-.7, .2, 1.1])
    delta = np.array([.02, -.03, .05])
    a, b = expit(base), expit(base+2*delta)
    assert np.allclose(np.log(b/(1-b))-np.log(a/(1-a)), 2*delta, atol=1e-14)
    passed('corrected logodds equals native plus2delta')
    f, fs = np.array([.02, -.01]), np.array([.01, .04])
    dt, dfixed, ds = f*np.ones(2), fs*np.ones(2), fs*np.ones(2)
    assert np.array_equal(dfixed-ds, [0, 0])
    passed('P4 unit gate has zero gate attribution')
    x = np.arange(64, dtype=np.float32).reshape(2, 32)
    p4, p5 = {'features': x}, {'features': x.copy()}
    assert np.array_equal(p4['features'], p5['features'])
    p5['features'][0, 0] += 1
    assert not np.array_equal(p4['features'], p5['features'])
    passed('P4 P5 equality detects altered input')
    assert np.intersect1d([0, 1, 2], [3, 4]).size == 0
    assert np.intersect1d([0, 1, 3], [3, 4]).size == 1
    passed('crossfit query cannot fit itself and overlap is detected')
    blocks = np.array_split(np.arange(871), 10)
    assert np.array_equal(np.sort(np.concatenate(blocks)), np.arange(871))
    invalid = np.r_[np.arange(870), 0]
    assert len(np.unique(invalid)) != 871
    passed('871 subject OOF coverage rejects duplicate missing subject')
    y = np.array([0, 1, 0, 1])
    po = np.tile([.4, .4, .7, .7], (5, 1))
    pt = np.tile([.6, .6, .4, .4], (5, 1))
    p0 = np.tile([.55, .3, .6, .6], (5, 1))
    attribution = ce_attribution(p0, pt, po, po, pt, y)
    assert attribution['identity_max_abs'] < 1e-14
    passed('per-element CE telescoping identity')
    assert attribution['rescue_count'] == attribution['harm_count'] == 10
    assert attribution['flips'] == 20
    passed('manual rescue2 harm2 each draw and net accuracy identity')
    repeated = np.repeat(np.tanh(np.array([[.1, -.2, .3]])), 5, axis=0)
    assert np.array_equal(repeated, np.repeat(repeated[:1], 5, axis=0))
    passed('deterministic branch delta invariant across native stochastic draws')
    dt, fixed, ds = np.array([.03, -.02]), np.array([.02, -.04]), np.array([.019, -.039])
    assert np.allclose(dt-ds, (dt-fixed)+(fixed-ds), atol=1e-15)
    passed('feature and gate delta decomposition')
    assert metrics(np.array([.8, .2, .5, .5]), np.array([0, 1, 0, 1]))['AUC'] == .875
    passed('independent rank AUC with ASD0 and tied probabilities')
    return {'passed': True, 'count': len(tests), 'tests': tests, 'network_training_operations': 0}


def mean_metrics(draw_rows):
    return {k: float(np.mean([row[k] for row in draw_rows])) if all(row[k] is not None for row in draw_rows) else None
            for k in KEYS}


def stage_b(seed, fold, saved_by_group, cache, split, meta, s, device):
    idx = np.asarray(split['test'], int)
    y = meta['labels'][idx]
    p0 = np.asarray(s['audit']['baseline_frozen'][f'{seed}/{fold}']['test_prob_asd'], float)
    report = {'seed': seed, 'fold': fold, 'n_test': len(idx), 'draws': 5, 'draw_subject_occurrences': 5*len(idx)}
    raw = {}
    checks = {}
    for group, saved in saved_by_group.items():
        checks[group] = validate_saved(saved, cache, split, meta, s, seed, fold, group)
        frozen = saved['frozen']
        p = {k: np.asarray(v, float) for k, v in frozen['counterfactual_prob_asd'].items()}
        branch, delta_raw = branch_analysis(saved, cache, split, group, device)
        variant_metrics = {k: [metrics(v[d], y) for d in range(5)] for k, v in p.items()}
        variant_mean_metrics = {k: mean_metrics(v) for k, v in variant_metrics.items()}
        probability_mean_metrics = {}
        for k, v in frozen['counterfactual_prob_asd'].items():
            probability_mean_metrics[k] = metrics(np.asarray(v).mean(0), y)
        effects = {'CE_delta_direct_branch': ce(p['true'], y)-ce(p['off'], y),
                   'CE_delta_true_minus_shuffled': ce(p['true'], y)-ce(p['shuffled'], y)}
        off_correct = ((p['off'] >= .5) == (y[None, :] == 0))
        true_correct = ((p['true'] >= .5) == (y[None, :] == 0))
        rescue, harm = ~off_correct & true_correct, off_correct & ~true_correct
        assert np.all(true_correct.sum(1)-off_correct.sum(1) == rescue.sum(1)-harm.sum(1))
        extra = {}
        if group == 'P5':
            attribution = ce_attribution(p0, p['true'], p['off'], p['shuffled'], p['unit_gate'], y)
            effects = {k: v for k, v in attribution.items() if k.startswith('CE_delta_')}
            extra['CE_identity_max_abs'] = attribution['identity_max_abs']
            p['P0'] = p0
            variant_mean_metrics['P0'] = mean_metrics([metrics(p0[d], y) for d in range(5)])
            probability_mean_metrics['P0'] = metrics(p0.mean(0), y)
        out = {'effects': {k: describe(v) for k, v in effects.items()},
               'correct_condition_prob_MAD': float(np.abs(p['true']-p['shuffled']).mean()),
               'legacy_dtype_correct_condition_prob_MAD': float(np.abs(
                   np.asarray(frozen['counterfactual_prob_asd']['true']) -
                   np.asarray(frozen['counterfactual_prob_asd']['shuffled'])).mean()),
               'direct_prob_MAD': float(np.abs(p['true']-p['off']).mean()),
               'native_probability_draw_SD': describe(p['off'].std(0, ddof=1)),
               'rescue_count': int(rescue.sum()), 'harm_count': int(harm.sum()),
               'net_rescue_count': int(rescue.sum()-harm.sum()),
               'rescue_fraction_per_draw_subject': float(rescue.mean()),
               'harm_fraction_per_draw_subject': float(harm.mean()),
               'unique_subjects_ever_rescued': int(np.any(rescue, axis=0).sum()),
               'unique_subjects_ever_harmed': int(np.any(harm, axis=0).sum()),
               'rescue_by_draw': rescue.sum(1).tolist(), 'harm_by_draw': harm.sum(1).tolist(),
               'auxiliary_metrics_mean_of_five_draws': variant_mean_metrics,
               'auxiliary_metrics_of_mean_probability': probability_mean_metrics,
               'branch': branch, 'historical_epoch_statistics': epoch_summary(saved), **extra}
        report[group] = out
        raw[group] = {'probabilities': p, 'effects': effects, 'rescue': rescue, 'harm': harm,
                      'gate': np.asarray(cache['features'][group]['gate'])[idx],
                      'changed': np.asarray(cache['diagnostics']['shuffle']['changed_mask'])[idx], **delta_raw}
    report['checks'] = checks
    return report, raw


def forbid_training():
    counters = {'optimizer_creations': 0, 'backward_calls': 0}
    def no_optimizer(*args, **kwargs):
        counters['optimizer_creations'] += 1
        raise RuntimeError('Experiment6 forbids any optimizer construction')
    def no_backward(*args, **kwargs):
        counters['backward_calls'] += 1
        raise RuntimeError('Experiment6 forbids any backward call')
    torch.optim.Optimizer.__init__ = no_optimizer
    torch.autograd.backward = no_backward
    torch.set_grad_enabled(False)
    return counters


def worker(gpu, manifest, result_queue):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    global VERIFIED
    VERIFIED = manifest
    counters = forbid_training()
    try:
        import_readonly_modules()
        s = json.loads((E5 / 'summary.json').read_text())
        ctx = safe_load(OLD / 'artifacts/context.pt')
        meta = metadata(ctx)
        fallback = None
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            device, name = torch.device('cuda:0'), torch.cuda.get_device_name(0)
        else:
            device, name = torch.device('cpu'), 'CPU frozen branch inference only'
            fallback = 'PPU CUDA backend unavailable; CPU 1089-parameter read-only forward permitted by prompt'
        count = 0
        started = time.perf_counter()
        for seed in SEEDS:
            for fold in FOLDS:
                if fold % 2 != gpu:
                    continue
                cache, x, split = restore_same_fold_data(seed, fold)
                validate_cache(cache, split, meta, s, seed, fold)
                a, a_subject = stage_a(seed, fold, x, meta['labels'], meta, split, cache)
                saved = {g: safe_load(E5 / f'runs/{g}/s{seed}_f{fold}.pt') for g in ('P4', 'P5')}
                b, b_subject = stage_b(seed, fold, saved, cache, split, meta, s, device)
                assert 'model.mm_gtunets' not in sys.modules
                count += 1
                result_queue.put({'kind': 'fold', 'gpu': gpu, 'seed': seed, 'fold': fold,
                    'indices': np.asarray(split['test'], int), 'stage_a': a, 'stage_b': b,
                    'a_subject': a_subject, 'b_subject': b_subject})
                del saved, cache, x
        assert counters == {'optimizer_creations': 0, 'backward_calls': 0}
        result_queue.put({'kind': 'worker_done', 'gpu': gpu, 'folds': count, 'branch_checkpoints': 2*count,
            'device': name, 'local_device': str(device), 'fallback': fallback, 'zero_training_counters': counters,
            'core_module_never_imported': True, 'elapsed_seconds': time.perf_counter()-started})
    except BaseException:
        result_queue.put({'kind': 'error', 'gpu': gpu, 'traceback': traceback.format_exc()})
        raise


def scalar_summary(rows, getter):
    per_seed = {}
    for seed in SEEDS:
        selected = [v for v in rows if v['seed'] == seed]
        assert len(selected) == 10
        values = [float(getter(v)) for v in selected]
        assert np.isfinite(values).all()
        per_seed[str(seed)] = float(np.mean(values))
    vals = list(per_seed.values())
    return {'mean': float(np.mean(vals)), 'seed_sample_sd': float(np.std(vals, ddof=1)),
            'per_seed': per_seed, 'folds_per_seed': 10, 'unit': 'fold mean then seed mean; three repeated seeds'}


def stratification(meta, changed=None):
    masks = {'all': np.ones(871, bool), 'ASD': meta['labels'] == 0, 'TD': meta['labels'] == 1,
             'male_SEX1': meta['sex'] == 1, 'female_SEX2': meta['sex'] == 2,
             'age_0_12': (meta['age'] >= 0) & (meta['age'] < 12),
             'age_12_18': (meta['age'] >= 12) & (meta['age'] < 18),
             'age_18_30': (meta['age'] >= 18) & (meta['age'] < 30), 'age_30_plus': meta['age'] >= 30}
    for site in np.unique(meta['site']):
        masks['site_' + str(site)] = meta['site'] == site
    if changed is not None:
        masks['effective_changed'] = changed.astype(bool)
        masks['unchanged'] = ~changed.astype(bool)
    return masks


def combine_subject_rows(rows, key, group=None):
    combined = {}
    counts = np.zeros(871, int)
    for row in rows:
        idx = row['indices']
        counts[idx] += 1
        data = row[key][group] if group else row[key]
        for name, value in data.items():
            if isinstance(value, dict):
                for child, arr in value.items():
                    a = np.asarray(arr)
                    target = combined.setdefault(name, {}).setdefault(child, np.empty(a.shape[:-1]+(871,), dtype=a.dtype))
                    target[..., idx] = a
            else:
                a = np.asarray(value)
                target = combined.setdefault(name, np.empty(a.shape[:-1]+(871,), dtype=a.dtype))
                target[..., idx] = a
    assert np.array_equal(counts, np.ones(871, int))
    return combined


def summarize_stratum(raw, idx, labels):
    n = int(idx.sum())
    if not n:
        return {'n_subjects': 0, 'analysis': None}
    y = labels[idx]
    effects = {k: describe(v[:, idx].mean(0)) for k, v in raw['effects'].items()}
    p = {k: v[:, idx] for k, v in raw['probabilities'].items()}
    rescue, harm = raw['rescue'][:, idx], raw['harm'][:, idx]
    draw_metrics = {k: [metrics(v[d], y, small_group=True) for d in range(5)] for k, v in p.items()}
    return {'n_subjects': n, 'ASD_n': int((y == 0).sum()), 'TD_n': int((y == 1).sum()),
        'unit': 'unique subjects within one seed; CE summarized after averaging five draws per subject',
        'effects': effects, 'effect_means_by_draw': {k: v[:, idx].mean(1).tolist() for k, v in raw['effects'].items()},
        'correct_condition_prob_MAD': float(np.abs(p['true']-p['shuffled']).mean()),
        'direct_prob_MAD': float(np.abs(p['true']-p['off']).mean()),
        'native_draw_SD_per_subject': describe(p['off'].std(0, ddof=1)),
        'rescue_count': int(rescue.sum()), 'harm_count': int(harm.sum()),
        'net_rescue_count': int(rescue.sum()-harm.sum()), 'draw_subject_occurrences': 5*n,
        'unique_ever_rescued': int(np.any(rescue, 0).sum()), 'unique_ever_harmed': int(np.any(harm, 0).sum()),
        'rescue_fraction_per_draw_subject': float(rescue.mean()), 'harm_fraction_per_draw_subject': float(harm.mean()),
        'mean_draw_metrics': {k: mean_metrics(v) for k, v in draw_metrics.items()},
        'draw_metrics': draw_metrics,
        'branch_delta': {k: describe(raw[k][idx]) for k in ('delta_true', 'delta_shuffle', 'delta_feature_part', 'delta_gate_part', 'delta_mean_feature')},
        'AUC_policy': 'null when n<20 or either diagnosis count<5; no clinical inference from small groups'}


def build_aggregates(results, meta, s):
    results = sorted(results, key=lambda r: (r['seed'], r['fold']))
    arows, brows = [r['stage_a'] for r in results], [r['stage_b'] for r in results]
    stage_a_out = {'folds': arows,
        'pca_explained_sum': scalar_summary(arows, lambda r: r['pca_ratio_sum']),
        'training_mean_difference_retained_ratio': scalar_summary(arows, lambda r: r['training_mean_difference_retained_ratio']),
        'TD_crossfit': {}, 'condition_changes': {}}
    for a, b in (('P2', 'P3'), ('P3', 'P4'), ('P2', 'P4')):
        key, ref = f'{a}_to_{b}', f'{b}_td_crossfit_vs_{a}'
        stage_a_out['TD_crossfit'][key] = {m: scalar_summary(arows, lambda r, m=m, ref=ref: r[ref][m])
            for m in ('pooled_sse_gain', 'pooled_sse_difference', 'mean_dim_gain', 'median_dim_gain', 'positive_dim_proportion', 'mean_R2_gain')}
        dim_seed = {str(seed): np.mean([r[ref]['per_dim_gain'] for r in arows if r['seed'] == seed], axis=0).tolist() for seed in SEEDS}
        stage_a_out['TD_crossfit'][key]['per_dim_gain_mean_by_seed'] = dim_seed
        stage_a_out['condition_changes'][key] = {
            'mu_L2': scalar_summary(arows, lambda r, k=key: r[k]['mu_change']['L2']['mean']),
            'deviation_L2': scalar_summary(arows, lambda r, k=key: r[k]['deviation_change']['L2']['mean']),
            'mu_relative_z_std_RMS': scalar_summary(arows, lambda r, k=key: r[k]['mu_change']['relative_z_std_RMS']['mean']),
            'deviation_relative_z_std_RMS': scalar_summary(arows, lambda r, k=key: r[k]['deviation_change']['relative_z_std_RMS']['mean']),
            'scale_difference_L2': scalar_summary(arows, lambda r, k=key: r[k]['scale_difference_norm_L2']),
            'mean_change_part_L2': scalar_summary(arows, lambda r, k=key: r[k]['mean_part_L2']['mean']),
            'scale_change_part_L2': scalar_summary(arows, lambda r, k=key: r[k]['scale_part_L2']['mean']),
            'clipping_remainder_L2': scalar_summary(arows, lambda r, k=key: r[k]['clipping_remainder_L2']['mean'])}
    stage_a_out['P4_query_only_shuffle'] = {
        'mu_L2': scalar_summary(arows, lambda r: r['P4_true_vs_shuffled']['mu_change']['L2']['mean']),
        'deviation_L2': scalar_summary(arows, lambda r: r['P4_true_vs_shuffled']['deviation_change']['L2']['mean']),
        'effective_fraction': scalar_summary(arows, lambda r: r['P4_true_vs_shuffled']['effective_changed_fraction'])}
    stage_b_out = {'folds': brows, 'groups': {}}
    for group in ('P4', 'P5'):
        names = list(brows[0][group]['effects'])
        q = {'effects': {k: scalar_summary(brows, lambda r, k=k, g=group: r[g]['effects'][k]['mean']) for k in names}}
        for k in ('correct_condition_prob_MAD', 'direct_prob_MAD'):
            q[k] = scalar_summary(brows, lambda r, k=k, g=group: r[g][k])
        q['native_probability_draw_SD'] = scalar_summary(brows, lambda r, g=group: r[g]['native_probability_draw_SD']['mean'])
        for k in ('first_weight_l2', 'last_weight_l2', 'last_bias'):
            q[k] = scalar_summary(brows, lambda r, k=k, g=group: r[g]['branch'][k])
        for k in ('feature_part_abs_test', 'gate_part_abs_test', 'delta_test_abs_true_minus_shuffled'):
            q[k] = scalar_summary(brows, lambda r, k=k, g=group: r[g]['branch'][k]['mean'])
        for k, getter in {
            'test_delta_abs': lambda b: b['test']['delta_abs']['mean'],
            'test_delta_mean': lambda b: b['test']['delta']['mean'],
            'test_delta_subject_SD': lambda b: b['test']['delta']['sd'],
            'train_delta_abs': lambda b: b['train']['delta_abs']['mean'],
            'mean_feature_replacement_MAD': lambda b: b['test']['mean_feature_replacement_abs_change']['mean'],
            'hidden_activation_coordinate_SD': lambda b: b['test']['hidden_activation_coordinate_sd']['mean'],
        }.items():
            q[k] = scalar_summary(brows, lambda r, g=group, f=getter: f(r[g]['branch']))
        q['best_epoch'] = scalar_summary(brows, lambda r, g=group: r[g]['historical_epoch_statistics']['best_epoch'])
        q['epochs_run'] = scalar_summary(brows, lambda r, g=group: r[g]['historical_epoch_statistics']['epochs_run'])
        q['historical_gradient_norms'] = {name: scalar_summary(brows,
            lambda r, name=name, g=group: r[g]['historical_epoch_statistics']['gradient_norms'][name]['mean'])
            for name in ('imaging', 'phenotype', 'classifier', 'pcnd')}
        q['rescue_harm_by_seed'] = {str(seed): {
            'rescue_occurrences': sum(r[group]['rescue_count'] for r in brows if r['seed'] == seed),
            'harm_occurrences': sum(r[group]['harm_count'] for r in brows if r['seed'] == seed),
            'net_occurrences': sum(r[group]['net_rescue_count'] for r in brows if r['seed'] == seed),
            'draw_subject_occurrences': 5*871} for seed in SEEDS}
        q['rescue_fraction'] = scalar_summary(brows, lambda r, g=group: r[g]['rescue_fraction_per_draw_subject'])
        q['harm_fraction'] = scalar_summary(brows, lambda r, g=group: r[g]['harm_fraction_per_draw_subject'])
        stage_b_out['groups'][group] = q
    by_seed, oof_internal = {}, {}
    for seed in SEEDS:
        rows = [r for r in results if r['seed'] == seed]
        a = combine_subject_rows(rows, 'a_subject')
        b = {g: combine_subject_rows(rows, 'b_subject', g) for g in ('P4', 'P5')}
        oof_internal[str(seed)] = b
        seed_out = {'n_unique_OOF_subjects': 871, 'folds': 10, 'draws_per_checkpoint': 5, 'groups': {}}
        masks = stratification(meta, b['P5']['changed'])
        seed_out['stage_a_strata'] = {name: {'n_subjects': int(idx.sum()),
            'ASD_n': int((meta['labels'][idx] == 0).sum()), 'TD_n': int((meta['labels'][idx] == 1).sum()),
            'changes': {cond: {key: describe(values[idx]) for key, values in vectors.items()} for cond, vectors in a.items()} if idx.any() else None}
            for name, idx in masks.items()}
        for group in ('P4', 'P5'):
            seed_out['groups'][group] = {name: summarize_stratum(b[group], idx, meta['labels']) for name, idx in masks.items()}
        by_seed[str(seed)] = seed_out
    p5 = stage_b_out['groups']['P5']
    legacy_mad = scalar_summary(brows, lambda r: r['P5']['legacy_dtype_correct_condition_prob_MAD'])
    assert abs(legacy_mad['mean'] - s['conditional']['P5']['probability_MAD']) < 1e-15
    p5['historical_MAD_dtype_crosscheck'] = {
        'passed': True, 'legacy_float32': legacy_mad,
        'current_float64': p5['correct_condition_prob_MAD'],
        'float64_minus_historical_float32': p5['correct_condition_prob_MAD']['mean'] - s['conditional']['P5']['probability_MAD'],
        'policy': 'Independent legacy-dtype recomputation matches historical value; primary analysis remains float64. No branch comparison threshold changed.'}
    assert abs(p5['effects']['CE_delta_true_minus_shuffled']['mean'] - s['conditional']['P5']['CE_true_minus_shuffle_mean']) < 1e-12
    reproduced = {k: scalar_summary(brows, lambda r, k=k: r['P5']['auxiliary_metrics_of_mean_probability']['true'][k]) for k in KEYS}
    assert all(abs(reproduced[k]['mean'] - s['groups']['P5']['frozen_probability_mean']['mean'][k]) < 2e-8 for k in KEYS)
    p5['reproduced_historical_five_draw_probability_mean_metrics'] = reproduced
    return stage_a_out, stage_b_out, by_seed, oof_internal


def own_source_audit():
    tree = ast.parse((OUT / 'run.py').read_text())
    forbidden = []
    banned = {'backward', 'step', 'fit', 'fit_transform', 'train', 'save', 'train_one', 'frozen_one'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fun = node.func
            if isinstance(fun, ast.Attribute) and fun.attr in banned:
                forbidden.append({'line': node.lineno, 'call': fun.attr})
    assert not forbidden, forbidden
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom)) and
                   any('model.mm_gtunets' in a.name or 'PCND_20261016.run' in a.name for a in n.names)
                   for n in ast.walk(tree))
    return {'passed': True, 'forbidden_update_fit_save_calls': forbidden,
            'full_core_not_imported_or_constructed': True,
            'historical_runners_not_imported': True,
            'only_analytic_same_formula_ridge_reconstructed': True}


def finite_json(value, path='summary'):
    if isinstance(value, dict):
        return {str(k): finite_json(v, path + '/' + str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v, path + f'/{i}') for i, v in enumerate(value)]
    if isinstance(value, np.ndarray):
        return finite_json(value.tolist(), path)
    if isinstance(value, np.generic):
        return finite_json(value.item(), path)
    if isinstance(value, float):
        assert np.isfinite(value), f'Nonfinite derived output: {path}'
    return value


def verify_new_aggregation(out, results, old):
    assert len(results) == 30
    for seed in SEEDS:
        assert sorted(r['fold'] for r in results if r['seed'] == seed) == list(FOLDS)
        idx = np.concatenate([r['indices'] for r in results if r['seed'] == seed])
        assert np.array_equal(np.sort(idx), np.arange(871))
    max_error = max(r['stage_b']['P5']['CE_identity_max_abs'] for r in results)
    assert max_error < 1e-12
    reconstruction = max(v for r in results for v in r['stage_a']['ridge_reconstruction_error'].values())
    delta_error = max(v for r in results for g in ('P4', 'P5') for v in r['stage_b'][g]['branch']['saved_delta_max_abs'].values())
    assert reconstruction <= 2e-5 and delta_error <= 3e-6
    assert all(r['stage_b']['P4']['branch']['gate_part_abs_test']['max'] == 0 for r in results)
    p5 = out['stage_b']['groups']['P5']
    means = p5['effects']
    assert abs(means['CE_delta_total']['mean'] - means['CE_delta_core_training_associated']['mean'] - means['CE_delta_direct_branch']['mean']) < 1e-12
    return {'passed': True, 'folds': 30, 'frozen_branch_checkpoints': 60,
            'cached_main_core_draws_used': 300, 'historical_P0_cached_draws_used': 150,
            'new_full_core_forwards': 0, 'new_neural_training_fits': 0,
            'branch_delta_vs_historical_max_abs': delta_error,
            'reference_reconstruction_max_abs': reconstruction,
            'CE_elementwise_identity_max_abs': max_error,
            'P4_gate_contribution_identically_zero': True,
            'P4_P5_true_features_bitwise_equal': True,
            'P6_partition_site_permutations_verified': 30,
            'OOF_each_seed_exactly_871_once': True,
            'historical_P5_CE_MAD_mean_probability_metrics_reproduced': True,
            'historical_formal_scores_not_recomputed': True,
            'no_threshold_chosen_from_test_results': True}


def verdict_from_evidence(out):
    a = out['stage_a']
    b = out['stage_b']['groups']['P5']
    age = a['TD_crossfit']['P2_to_P3']['pooled_sse_gain']
    sex = a['TD_crossfit']['P3_to_P4']['pooled_sse_gain']
    correct = b['effects']['CE_delta_true_minus_shuffled']
    direct = b['effects']['CE_delta_direct_branch']
    facts = (
        f"Age TD-OOF SSE gain {age['mean']:.8f} ({sum(v>0 for v in age['per_seed'].values())}/3 seeds positive); "
        f"additional sex gain {sex['mean']:.8f} ({sum(v>0 for v in sex['per_seed'].values())}/3 positive). "
        f"Fixed-reference query changes deviation L2 by {a['P4_query_only_shuffle']['deviation_L2']['mean']:.8f}; "
        f"frozen branch feature contribution to delta change MAD is {b['feature_part_abs_test']['mean']:.8f}. "
        f"Direct branch CE gain occurs in {sum(v<0 for v in direct['per_seed'].values())}/3 seeds; "
        f"correct conditions improve CE in {sum(v<0 for v in correct['per_seed'].values())}/3 seeds."
    )
    return {
        'interpretation_policy': 'No binary strong/weak thresholds or method choices fitted to test results. Interpret continuous reference/branch/CE measurements jointly.',
        'age_reference_SSE_gain': age,
        'additional_sex_reference_SSE_gain': sex,
        'query_deviation_change_L2': a['P4_query_only_shuffle']['deviation_L2'],
        'direct_branch_CE': b['effects']['CE_delta_direct_branch'],
        'correct_condition_CE': b['effects']['CE_delta_true_minus_shuffled'],
        'native_draw_SD': b['native_probability_draw_SD'],
        'direct_probability_MAD': b['direct_prob_MAD'],
        'observed_chain': facts,
        'intercept_diagnostic': {
            'last_bias': b['last_bias'], 'last_weight_l2': b['last_weight_l2'],
            'test_between_subject_delta_SD': b['test_delta_subject_SD'],
            'train_mean_input_replacement_delta_MAD': b['mean_feature_replacement_MAD'],
            'interpretation': 'Use these continuous measurements together; nonzero input-replacement response precludes an exactly constant branch, but does not establish useful conditional information.'},
        'claim_supported': 'No established patient-level phenotype-conditioned diagnostic complementarity in this version. Reference fit, input change, MLP response and CE benefit must be distinguished.',
        'remaining_alternatives': [
            'Predictable age-related variation need not be incremental ASD information beyond the multimodal core.',
            'Small delta may reflect redundant or noisy diagnostic evidence, regularization, low-dimensional representation or joint optimization; frozen attribution does not identify a unique training cause.',
            'P0 and P5 have different trained core checkpoints and stochastic paths; their core CE difference is associated with training, not a pure causal optimization effect.',
            'Per-dimension scale changes and clipping contribute to deviation changes; a change in d does not measure only reference mean change.',
            'Low reference prediction gains can coexist with measurable query sensitivity; neither alone proves useful normative disease information.'
        ],
        'MICCAI2027': 'Current evidence does not establish a successful new method. Basic phenotype conditioning overlaps existing work including PhenoBrain; no first-use novelty claim.',
        'one_next_experiment_only': 'Independently preregister a leakage-free, site-held-out conditional-increment test of simple TD deviation evidence added to a fixed existing core; verify unique diagnostic information before designing a new fusion module.',
        'new_experiment_executed': False}


def format_summary(value, scale=1, digits=6):
    return f"{scale*value['mean']:.{digits}f} +/- {scale*value['seed_sample_sd']:.{digits}f}"


def write_report(out):
    a, b = out['stage_a'], out['stage_b']['groups']
    audit = out['audits']
    lines = [
        '# Experiment 6：PCND 双阶段只读归因', '',
        f"研究验收时间 UTC：{out['research_verified_at']}。本轮新增神经网络训练0，optimizer创建0，backward/step 0，完整GTUNet前向0。", '',
        '## 来源与统计单位', '',
        f"本地git HEAD：`{audit['initial']['local_git_HEAD']}`；本地已知origin/main：`{audit['initial']['local_origin_main_tracking_commit']}`。本地HEAD落后于已上传版本已如实记录；实际读取文件以逐文件SHA为准，不由commit名推定。",
        f"本轮实际核验输入{len(out['file_hashes'])}个：30缓存、P4/P5共60个Checkpoint、原始源文件及本轮引用的数据/划分。每次pt首次加载前已对照E5登记SHA，使用CPU mmap/weights_only=False；没有重扫全部历史1596文件。完整相对路径、字节数和前后SHA见summary.file_hashes。",
        'E5注册源组合SHA：`' + audit['initial']['experiment5_training_combined_sha256'] + '`。不导入历史run/analyze，不创建完整核心；PCND只有1089参数，冻结eval且禁用梯度。原始核心权重只在CPU mmap中校验。',
        '871人，ASD=0/TD=1，403/468；CSV按唯一SUB_ID对齐历史subject order。每seed的十个外test折恰好覆盖871人一次。五draw是同受试者的随机推理，不能作为五个独立患者。',
        '主要效应单位：先每fold平均五draw和test被试，再平均十fold，最后报告三seed均值与样本SD；OOF池化分层单列，不混入主统计。正CE差表示前者更差；不给30折或15个seed-draw伪独立p值。',
        'P0/P4/P5保存的五个draw seed均逐折匹配SHA规则。P0嵌入bank没有完整RNG快照，不主张跨训练模型使用了完全相同的随机数消费轨迹。P5同Checkpoint的四种条件明确共享同一次native logits。',
        '原论文[3.2-3.5与Table5](https://arxiv.org/html/2406.14455v3)已有非影像VAE、AMRS人口图、两条GTUNet及注意力融合后MLP。Table5 ABIDE影像/非影像/联合ACC为76.49/52.00/82.92，AUC为85.22/50.28/88.21；这不能证明额外规范偏差的患者级互补性。',
        '[PhenoBrain官方论文页](https://proceedings.mlr.press/v306/meng26f.html)已有表型条件长程路由和表型引导注意力；本轮不声称简单条件化首创，也未作该模型头对头实验。', '',
        '## 历史实验5正式成绩（只读复用）', '',
        '下表原样引用E5已验收的三seed统计，单位%。未重跑或重新计算正式单次测试。', '',
        '| Group | ACC | BA | prob-AUC | SEN | SPE | F1 |', '|---|---|---|---|---|---|---|']
    for g, v in out['historical_E5_formal'].items():
        lines.append('| ' + g + ' | ' + ' | '.join(f"{100*v['mean'][k]:.2f} +/- {100*v['seed_sd_ddof1'][k]:.2f}" for k in KEYS) + ' |')
    lines += ['', '## Stage A：参考是否形成', '',
        '| 测量 | 三seed均值 +/- 样本SD | 含义 |', '|---|---|---|',
        f"| PCA32累计解释方差 | {format_summary(a['pca_explained_sum'],100,3)}% | 仅沿用train拟合的现存PCA |",
        f"| 训练集ASD-TD均值差方向保留比例 | {format_summary(a['training_mean_difference_retained_ratio'],100,3)}% | 训练索引计算，非病灶能量或临床可迁移判别力 |"]
    for key in ('P2_to_P3', 'P3_to_P4', 'P2_to_P4'):
        v = a['TD_crossfit'][key]
        lines.append(f"| {key} TD离折SSE相对下降 | {format_summary(v['pooled_sse_gain'],100,4)}% | 正数为参考影像预测改善 |")
    lines += ['', '每维`1-SSE目标/SSE来源`是相对SSE gain；另保存常规R2增量`(SSE来源-SSE目标)/TSS_TD`，避免把两个定义混称。以下均为训练TD的5折离折预测，不是ASD分类性能。', '',
              '| 参考对照 | 平均维SSE gain % | 中位维SSE gain % | 正gain维比例 % | 平均常规R2增量 |', '|---|---|---|---|---|']
    for key, v in a['TD_crossfit'].items():
        lines.append(f"| {key} | {format_summary(v['mean_dim_gain'],100,4)} | {format_summary(v['median_dim_gain'],100,4)} | {format_summary(v['positive_dim_proportion'],100,2)} | {format_summary(v['mean_R2_gain'])} |")
    lines += ['', '每折32维SSE/TSS、每维gain/R2与正向维数均保存在stage_a.folds。该训练参考统计的外fold之间重叠，不是30个独立样本。', '',
        '| test同一受试者对照 | 参考mu变化L2 | 偏差d变化L2 | mu变化相对z标准差RMS | 尺度向量变化L2 |', '|---|---|---|---|---|']
    for key, v in a['condition_changes'].items():
        lines.append(f"| {key} | {format_summary(v['mu_L2'])} | {format_summary(v['deviation_L2'])} | {format_summary(v['mu_relative_z_std_RMS'])} | {format_summary(v['scale_difference_L2'])} |")
    q = a['P4_query_only_shuffle']
    lines += [f"| P4固定参考真实/错配query | {format_summary(q['mu_L2'])} | {format_summary(q['deviation_L2'])} | 详见逐fold和OOF分层 | 同一P4尺度固定 |", '',
        'P2-P3-P4的尺度分别由各自训练TD离折残差标准差产生，不可把d变化全解释为mu变化。按目标尺度先分解均值项，再分解原均值下的尺度项，裁剪差额单列；逐元素恒等式已验证。每人均方、cos、L2、相对z标准差RMS及各项分布均落在逐fold摘要和OOF分层。零范数cos记录未定义计数，不作有效相似度解释。', '',
        '| 对照 | 均值项L2 | 尺度项L2 | 裁剪差额L2 |', '|---|---|---|---|']
    for key, v in a['condition_changes'].items():
        lines.append(f"| {key} | {format_summary(v['mean_change_part_L2'])} | {format_summary(v['scale_change_part_L2'])} | {format_summary(v['clipping_remainder_L2'])} |")
    lines += ['', f"P4错配查询实际年龄或性别改变比例：{format_summary(q['effective_fraction'],100,3)}%。使用同一真实TD参考系数和尺度，年龄/性别整体在test分区同站点错位；单例保持自身。真实与错配偏差均与现存缓存核对。",
        '30折PCA/P1投影、P2/P3/P4重建、P4/P5逐位同偏差、分区/站点双射和离折无自身拟合均通过。P6只读验证现存排列，不拟合新P6参考、不生成新缓存。', '',
        '## Stage B：诊断增益是否传递', '',
        '| Group | 首层权重L2 | 末层权重L2 | 末层bias | test delta均值 | test delta SD | test平均绝对delta |', '|---|---|---|---|---|---|']
    for group, v in b.items():
        lines.append('| ' + group + ' | ' + ' | '.join(format_summary(v[k]) for k in
            ('first_weight_l2', 'last_weight_l2', 'last_bias', 'test_delta_mean', 'test_delta_subject_SD', 'test_delta_abs')) + ' |')
    lines += ['', 'delta的SD是同fold被试之间的变化，再按fold/seed汇总，不是跨随机draw的噪声。每折train/test分位数、绝对值、方差、隐藏激活和输入坐标SD均保存在JSON。', '',
        '| Group | train平均绝对delta | 隐藏激活平均坐标SD | 用训练偏差均值替换输入后的delta MAD | Best Epoch | 实际epoch数 |', '|---|---|---|---|---|---|']
    for group, v in b.items():
        lines.append('| ' + group + ' | ' + ' | '.join(format_summary(v[k]) for k in
            ('train_delta_abs', 'hidden_activation_coordinate_SD', 'mean_feature_replacement_MAD', 'best_epoch', 'epochs_run')) + ' |')
    lines += ['', '均值输入替换保留每人的gate，仅衡量冻结MLP对输入变化的响应，不能等同医学反事实。是否近似截距须结合末层bias、被试间delta方差和替换MAD，不能仅凭delta小就认定没有斜率。', '',
        '| Group | D=CE(true)-CE(off) | C=CE(true)-CE(shuffled) | G=CE(true)-CE(unit gate) |', '|---|---|---|---|']
    for group, v in b.items():
        eff = v['effects']
        gate = format_summary(eff['CE_delta_true_minus_unit_gate']) if 'CE_delta_true_minus_unit_gate' in eff else 'UNAVAILABLE（P4未保存该条件）'
        lines.append(f"| {group} | {format_summary(eff['CE_delta_direct_branch'],1,9)} | {format_summary(eff['CE_delta_true_minus_shuffled'],1,9)} | {gate} |")
    p5 = b['P5']
    lines += ['', '负值表示前者CE更好，三个效应都用同Checkpoint、同draw、同subject的原生主干输出。', '',
        '| P5相对P0的可加CE分解 | 均值 +/- seed样本SD | 三seed均值 |', '|---|---|---|']
    for key in ('CE_delta_total', 'CE_delta_core_training_associated', 'CE_delta_direct_branch'):
        v = p5['effects'][key]
        lines.append(f"| {key} | {format_summary(v,1,9)} | " + ' / '.join(f'{v["per_seed"][str(seed)]:.9f}' for seed in SEEDS) + ' |')
    lines += ['', '逐元素验证 total=主干训练相关差+同模型直接分支效应。第一项包括不同模型权重及随机路径的差，不能称纯优化因果效应。ACC、BA、AUC、F1没有进行可加分解。', '',
        '| Group/seed | 救回次数 | 破坏次数 | 净次数 | draw-subject总次数 |', '|---|---|---|---|---|']
    for group, v in b.items():
        for seed, counts in v['rescue_harm_by_seed'].items():
            lines.append(f"| {group}/{seed} | {counts['rescue_occurrences']} | {counts['harm_occurrences']} | {counts['net_occurrences']} | 4355 |")
    lines += ['', '一次救回定义为同draw off错→true对，破坏反向；每seed4355次来自871人×5draw，不是4355独立患者。逐draw救回-破坏恰等于正确预测数差；去重的曾救回/曾破坏人数见by_seed各组all。', '',
        '| Group | delta真实-错配绝对均值 | 偏差特征项绝对均值 | gate项绝对均值 |', '|---|---|---|---|']
    for group, v in b.items():
        lines.append(f"| {group} | {format_summary(v['delta_test_abs_true_minus_shuffled'],1,9)} | {format_summary(v['feature_part_abs_test'],1,9)} | {format_summary(v['gate_part_abs_test'],1,9)} |")
    lines += ['', '有符号项逐元素满足dt-ds=(dt-dfixed)+(dfixed-ds)；绝对值均值不具可加性，不以该表绝对值相加或比例冒充因果贡献。P4 gate项严格为0。两卡只运行冻结小分支，对照历史delta的最大误差见验收表；每个Checkpoint五draw的delta自身逐值相同。', '',
        '| Group | 原生off概率五draw SD | true-off概率MAD | true-shuffled概率MAD |', '|---|---|---|---|']
    for group, v in b.items():
        lines.append(f"| {group} | {format_summary(v['native_probability_draw_SD'],1,9)} | {format_summary(v['direct_prob_MAD'],1,9)} | {format_summary(v['correct_condition_prob_MAD'],1,9)} |")
    lines += ['', '此表仅比较量级，不把随机重复当作独立样本。log-odds增量2delta使用预注册概率范围(0.001,0.999)验证，排除数及最大误差逐fold记录；该筛选仅用于数值校验，所有CE和临床分层仍使用全部被试。', '',
        '历史P5五draw概率均值ACC/AUC复算为 ' + f"{100*p5['reproduced_historical_five_draw_probability_mean_metrics']['ACC']['mean']:.4f}% / {100*p5['reproduced_historical_five_draw_probability_mean_metrics']['AUC']['mean']:.4f}%" + '，真实错配概率MAD和CE也复现历史记录；这些辅助指标不替代正式主表。', '',
        '历史epochs中loss/delta是每epoch记录，梯度仅最初3步及每25epoch采样；本轮未重新反传，没有插值成完整梯度轨迹。', '',
        '## OOF分层', '',
        '下表按每seed唯一871人池化，各被试先平均五draw。列出原始人数、CE及输入变化；这些池化值与主表等权fold均值有轻微差异，已标记。所有站点及男女、年龄、诊断亚组原始统计详见summary.by_seed，不把跨seed2613行称独立患者。', '',
        '| 分层 | 每seed人数 | P4真实/错配偏差L2 | P5 D | P5 C | P5救回/破坏（三seed分别） |', '|---|---|---|---|---|---|']
    for name in ('all', 'male_SEX1', 'female_SEX2', 'age_0_12', 'age_12_18', 'age_18_30', 'age_30_plus', 'effective_changed'):
        all_seeds = [out['by_seed'][str(seed)] for seed in SEEDS]
        ns = [v['groups']['P5'][name]['n_subjects'] for v in all_seeds]
        mu = [v['stage_a_strata'][name]['changes']['P4_true_vs_shuffled']['deviation_L2']['mean'] for v in all_seeds]
        d = [v['groups']['P5'][name]['effects']['CE_delta_direct_branch']['mean'] for v in all_seeds]
        c = [v['groups']['P5'][name]['effects']['CE_delta_true_minus_shuffled']['mean'] for v in all_seeds]
        events = [f"{v['groups']['P5'][name]['rescue_count']}/{v['groups']['P5'][name]['harm_count']}" for v in all_seeds]
        lines.append(f"| {name} | {'/'.join(map(str,ns))} | {np.mean(mu):.6f} | {np.mean(d):.9f} | {np.mean(c):.9f} | {'; '.join(events)} |")
    lines += ['', '站点n<20或任一诊断类<5时AUC记null；单诊断组同样未定义AUC。小组只作描述，不作医学发现或测试后择组。', '',
        '## 决策与限制', '',
        '下列解释结构预先固定，不用测试成绩设定强弱阈值：', '',
        '| 参考与诊断证据 | 可支持的判断 |', '|---|---|',
        '| A弱、B弱 | 当前参考版本未形成明显且可用的条件信息，先改研究假设/参考表征 |',
        '| A形成差异、B响应弱 | 检查证据分支斜率、近似截距或与原core重复；新模型另行预注册 |',
        '| A形成差异、B响应但CE不改善 | 条件变化未构成互补，噪声/站点混杂等仍可能 |',
        '| A形成差异、B的CE收益且真实优于错配 | 仅为探索线索，仍需外站无泄漏确认 |', '',
        '本轮按连续数值解释：年龄参考TD离折SSE增益 ' + format_summary(a['TD_crossfit']['P2_to_P3']['pooled_sse_gain'],100,4) + '%，额外性别增益 ' + format_summary(a['TD_crossfit']['P3_to_P4']['pooled_sse_gain'],100,4) + '%；P4查询错配偏差L2 ' + format_summary(q['deviation_L2']) + '；P5特征项delta MAD ' + format_summary(p5['feature_part_abs_test'],1,9) + '；直接分支D ' + format_summary(p5['effects']['CE_delta_direct_branch'],1,9) + '，正确条件C ' + format_summary(p5['effects']['CE_delta_true_minus_shuffled'],1,9) + '。这些量需联合阅读，不能只由概率MAD决定是哪一阶段失败。',
        '实际链条：' + out['verdict']['observed_chain'],
        '当前未建立正确年龄/性别提供可靠患者级诊断互补的完整证据。参考能预测发育变异，并不保证有原始多模态核心之外的ASD增量。末层权重、输入替换与delta方差可说明冻结分支响应，无法由此唯一判断训练时为何学成这种响应。',
        '尚不能排除低维PCA丢失特定方向、参考形式过简、历史RFE筛选污染、站点/年龄分布、证据与原core冗余、正则化和联合优化等解释。P4/P5/P6独立训练之间的差异不等同同Checkpoint反事实。',
        '原始RFE跨折监督信息污染、全队列转导图及VAE/eval dropout历史行为完整保留；新增参考train-only不表示整个系统无泄漏。训练集均值差方向的保留比例不能叫真实病灶能量。',
        'MICCAI2027：本轮不构成有效新方法证明，也不能证明所有规范建模无效。需要先验证条件偏差含有原core之外的诊断信息；单靠本轮冻结事后归因，仍不能唯一决定应该换规范参考还是换融合器。',
        '唯一下一步建议：独立预注册无RFE跨折泄漏、站点外验证的简单TD偏差条件增量检验，在固定原core上确认非冗余诊断信息后再考虑设计融合模块。本轮未执行该实验。', '',
        '## 执行验收', '',
        '```json', json.dumps(out['audits']['final'], ensure_ascii=False, indent=2), '```', '',
        '双GPU任务记录：`' + json.dumps(out['execution'], ensure_ascii=False) + '`。两个单线程worker各处理15fold、30个冻结分支Checkpoint；不运行神经训练或整图前向。',
        f"合成确定性检查{out['synthetic_tests']['count']}项全部通过，逐项名称见summary.synthetic_tests。所有输入在分析前后流式SHA一致，JSON无NaN/Infinity；关键量可追溯到fold行、seed与样本数。",
        '产物仅config.json/run.py/summary.json/report.md/run.log；不复制历史模型、缓存、不产生PNG或新fold文件。根handoff只在全部报告和结果验收后一次追加，最终状态见summary.handoff。未push GitHub。', '']
    if out['audits'].get('implementation_revision'):
        lines += ['汇总实现修订：第一次运行30折及60分支重建已通过，最终历史MAD核对因float32/float64口径相差1.7487e-12触发过严断言。已独立按原float32复算与旧值逐位对齐，主分析仍用float64，原分支数值容差未放宽；失败及源码指纹保留在run.log与summary.audits.implementation_revision。未修改任何历史文件，第一次未追加handoff。', '']
    atomic_text('\n'.join(lines), OUT / 'report.md')


def terminal_summary(out):
    a, b = out['stage_a'], out['stage_b']['groups']['P5']
    print('\n实验6已完成：30/30折参考重建、60/60冻结分支核验；物理PPU0/1各15fold。新增神经训练0、完整主干前向0、optimizer/backward/step 0。')
    print('Stage A PCA32解释方差：' + format_summary(a['pca_explained_sum'],100,3) + '%；训练集均值差方向保留：' + format_summary(a['training_mean_difference_retained_ratio'],100,3) + '%。')
    print('Stage A 年龄/额外性别TD离折SSE gain：' + format_summary(a['TD_crossfit']['P2_to_P3']['pooled_sse_gain'],100,4) + '% / ' + format_summary(a['TD_crossfit']['P3_to_P4']['pooled_sse_gain'],100,4) + '%。')
    print('Stage A 同P4参考真实/错配偏差L2：' + format_summary(a['P4_query_only_shuffle']['deviation_L2']))
    for name in ('CE_delta_true_minus_shuffled', 'CE_delta_direct_branch', 'CE_delta_core_training_associated', 'CE_delta_total', 'CE_delta_true_minus_unit_gate'):
        print(name + ': ' + format_summary(b['effects'][name],1,9))
    print('Stage B delta变化特征项/gate项绝对均值：' + format_summary(b['feature_part_abs_test'],1,9) + ' / ' + format_summary(b['gate_part_abs_test'],1,9))
    print('归因：当前参考与冻结分支的连续响应需联合判断；尚未建立正确条件的可靠诊断互补，不能从本轮唯一识别训练失败原因。')
    print('目录：' + str(OUT) + '；handoff已在全部验收后一次追加；GitHub push 0。')


def rehash_inputs(manifest):
    for path, record in manifest.items():
        actual = sha256(Path(path))
        assert actual == record['sha256'], f'Read-only input changed during analysis: {path}'
    return len(manifest)


def handoff_text(out):
    a, b = out['stage_a'], out['stage_b']['groups']['P5']
    lines = ['', '<!-- PCND_DUALSTAGE_20261016_BEGIN -->', CFG['handoff_heading'], '',
        f"完成UTC：{out['research_verified_at']}。实验6只读分析实验5现存产物，新增神经训练0，optimizer创建/backward/step 0，VAE/RFE/PCA训练0，完整GTUNet前向0。",
        '独立新目录：experiments/PCND_DualStage_20261016，仅config.json、run.py、summary.json、report.md、run.log。两张物理PPU各15fold、各30个冻结1089参数P4/P5分支；CPU重算同一个注册解析Ridge公式用于一致性核验，没有创建新诊断模型。',
        f"核验30份缓存、60个Checkpoint、全部本轮实际引用输入{len(out['file_hashes'])}项SHA，未重扫全部1596历史文件。参考/分支重建、P4/P5输入逐值一致、P6分区/站点边界、871人OOF一次覆盖和{out['synthetic_tests']['count']}项合成检查全部通过；最大误差见summary.audits.final。",
        '原论文3.2-3.5及Table5、根源代码与实验5normative/pcnd/run/analyze逐项审计。原MM-GTUNets已使用年龄/性别/site，不能声称首次使用表型。已有PhenoBrain等条件化思想，不主张简单条件化首创。',
        'Stage A：PCA32累计解释方差 ' + format_summary(a['pca_explained_sum'],100,3) + '%；训练集ASD-TD均值差方向保留 ' + format_summary(a['training_mean_difference_retained_ratio'],100,3) + '%；P2→P3年龄TD离折SSE gain ' + format_summary(a['TD_crossfit']['P2_to_P3']['pooled_sse_gain'],100,4) + '%；P3→P4额外性别gain ' + format_summary(a['TD_crossfit']['P3_to_P4']['pooled_sse_gain'],100,4) + '%。',
        'Stage A同P4固定真实TD参考，真实/错配query的mu变化L2 ' + format_summary(a['P4_query_only_shuffle']['mu_L2']) + '，偏差变化L2 ' + format_summary(a['P4_query_only_shuffle']['deviation_L2']) + '。不同参考组的scale变化与裁剪已单独分解，不能全算mu变化。',
        'Stage B：P5 D=CEtrue-CEoff ' + format_summary(b['effects']['CE_delta_direct_branch'],1,9) + '；C=CEtrue-CEshuffle ' + format_summary(b['effects']['CE_delta_true_minus_shuffled'],1,9) + '；G=CEtrue-CEunit ' + format_summary(b['effects']['CE_delta_true_minus_unit_gate'],1,9) + '。负值才表示前者更好。',
        'P5相对P0 CE总差 ' + format_summary(b['effects']['CE_delta_total'],1,9) + ' = 主干训练相关差 ' + format_summary(b['effects']['CE_delta_core_training_associated'],1,9) + ' + 同Checkpoint直接分支差 ' + format_summary(b['effects']['CE_delta_direct_branch'],1,9) + '，逐元素1e-12验证；第一项非纯优化因果效应，不对ACC/AUC/F1作加法分解。',
        'delta真实/错配特征项绝对均值 ' + format_summary(b['feature_part_abs_test'],1,9) + '、gate项 ' + format_summary(b['gate_part_abs_test'],1,9) + '；有符号分解严格相加，绝对量不可相加。P4 gate项0。P5原生off五draw概率SD ' + format_summary(b['native_probability_draw_SD'],1,9) + '，直接概率校正MAD ' + format_summary(b['direct_prob_MAD'],1,9) + '。',
        'P5末层权重L2 ' + format_summary(b['last_weight_l2']) + '，末层bias ' + format_summary(b['last_bias']) + '，test delta被试间SD ' + format_summary(b['test_delta_subject_SD']) + '，训练均值偏差替换后的delta MAD ' + format_summary(b['mean_feature_replacement_MAD']) + '。冻结响应不能唯一确定联合训练中为何学成该斜率。',
        '救回/破坏每seed次数：' + json.dumps(b['rescue_harm_by_seed'], ensure_ascii=False) + '。每seed为871人×5draw，不是4355独立病例；各年龄/性别/站点OOF结果及小站点AUC抑制均在JSON。',
        '结论：尚未建立当前版本PCND可靠的患者级表型条件互补。参考连续影像预测、query扰动、MLP响应与CE收益是不同环节；本轮可定位数值衰减但不唯一证明规范参考或联合优化是唯一失败原因。P0/P4/P5为不同训练模型，只有P5内部四条件有同Checkpoint直接干预定义。',
        '历史跨折RFE监督信息污染、转导人口图与原VAE/eval dropout保留；train-only规范参考不表示整体无泄漏，训练均值差方向不是医学病灶能量。跨seed为重复871人，不作伪独立显著性宣称。',
        '唯一下一步：另行预注册无RFE泄漏、站点外验证的简单TD偏差条件增量检验，在固定core上先查非冗余诊断信号，再决定规范参考或融合设计。当前不将PCND作为已成立MICCAI2027方法，不执行新模型实验。',
        '完整证据：experiments/PCND_DualStage_20261016/report.md、summary.json；代码与输入SHA在JSON。全部研究/报告验收后一次追加本节，历史前缀逐字保留；本轮未push。',
        '<!-- PCND_DUALSTAGE_20261016_END -->', '']
    return '\n'.join(lines)


def append_handoff(out):
    path = ROOT / 'handoff.md'
    before = out['handoff']['before']
    section = out['handoff']['section'].encode()
    with path.open('r+b') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        current = stream.read()
        assert hashlib.sha256(current[:before['bytes']]).hexdigest() == before['sha256'], 'Historical handoff prefix changed'
        suffix = current[before['bytes']:]
        if suffix == section:
            pass
        else:
            assert section.startswith(suffix), 'Unrelated handoff append detected; do not overwrite'
            stream.seek(0, 2)
            stream.write(section[len(suffix):])
            stream.flush()
            os.fsync(stream.fileno())
        stream.seek(0)
        final = stream.read()
        assert final == current[:before['bytes']] + section
        assert final.count(b'<!-- PCND_DUALSTAGE_20261016_BEGIN -->') == 1
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    return {'status': 'complete', 'before': before, 'after_sha256': hashlib.sha256(final).hexdigest(),
            'after_bytes': len(final), 'section_sha256': hashlib.sha256(section).hexdigest(),
            'section_bytes': len(section), 'append_count': 1, 'historical_prefix_unchanged': True,
            'updated_after_all_research_checks': True, 'time_utc': utc()}


def verify_resume(out):
    assert out['status'] == 'COMPLETE'
    checked = rehash_inputs(out['file_hashes'])
    for name, expected in out['output_sha256'].items():
        assert sha256(OUT / name) == expected, f'Completed output fingerprint changed: {name}'
    h = (ROOT / 'handoff.md').read_bytes()
    b = out['handoff']['before']
    assert hashlib.sha256(h[:b['bytes']]).hexdigest() == b['sha256']
    assert hashlib.sha256(h[b['bytes']:out['handoff']['after_bytes']]).hexdigest() == out['handoff']['section_sha256']
    assert h.count(b'<!-- PCND_DUALSTAGE_20261016_BEGIN -->') == 1
    print(json.dumps({'resume_readonly_verified': True, 'input_files': checked,
                      'outputs': len(out['output_sha256']), 'neural_forwards': 0, 'writes': 0}, ensure_ascii=False))


def finish_delivery(out):
    out['handoff'] = append_handoff(out)
    out['status'] = 'COMPLETE'
    out['completed_at'] = utc()
    log('complete', folds=30, frozen_branch_checkpoints=60, new_training=0, new_core_forwards=0,
        gpu_folds=out['execution']['fold_counts'], handoff_appended_once=True)
    out['output_sha256'] = {name: sha256(OUT / name) for name in ('run.py', 'config.json', 'report.md', 'run.log')}
    atomic_json(finite_json(out), OUT / 'summary.json')
    terminal_summary(out)


def main():
    global VERIFIED
    with (OUT / 'run.py').open('r') as owner:
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        summary_path = OUT / 'summary.json'
        previous = None
        if summary_path.exists():
            previous = json.loads(summary_path.read_text())
            if previous.get('status') == 'COMPLETE':
                verify_resume(previous)
                return
            if previous.get('status') == 'RESEARCH_VERIFIED_HANDOFF_PENDING':
                rehash_inputs(previous['file_hashes'])
                assert sha256(OUT / 'report.md') == previous['report_sha256']
                assert sha256(OUT / 'run.py') == previous['meta']['analysis_code_sha256']
                finish_delivery(previous)
                return
        started = time.perf_counter()
        created = utc()
        handoff_before = (ROOT / 'handoff.md').read_bytes()
        assert b'<!-- PCND_DUALSTAGE_20261016_BEGIN -->' not in handoff_before
        own = own_source_audit()
        old, manifest, audit = input_audit()
        VERIFIED = manifest
        import_readonly_modules()
        synthetic = synthetic_tests()
        ctx = safe_load(OLD / 'artifacts/context.pt')
        meta = metadata(ctx)
        log('audit_passed', inputs=len(manifest), features=30, checkpoints=60,
            synthetic_checks=synthetic['count'], neural_training=0)
        initial = {'status': 'RUNNING', 'created_at': created, 'analysis_code_sha256': sha256(OUT / 'run.py'),
                   'input_files': len(manifest), 'config_sha256': sha256(OUT / 'config.json')}
        atomic_json(initial, summary_path)
        context = mp.get_context('spawn')
        q = context.Queue()
        children = [context.Process(target=worker, args=(gpu, manifest, q), name=f'pcnd-readonly-ppu{gpu}') for gpu in (0, 1)]
        for child in children:
            child.start()
        results, workers = [], []
        try:
            while len(workers) < 2:
                try:
                    event = q.get(timeout=1)
                except queue.Empty:
                    dead = [p for p in children if p.exitcode not in (None, 0)]
                    if dead:
                        raise RuntimeError(f'Read-only worker failed: {[(p.name,p.exitcode) for p in dead]}')
                    continue
                if event['kind'] == 'error':
                    raise RuntimeError(event['traceback'])
                if event['kind'] == 'worker_done':
                    workers.append(event)
                    log('worker_done', **{k: v for k, v in event.items() if k != 'kind'})
                else:
                    results.append(event)
                    if len(results) % 5 == 0:
                        log('attribution_progress', completed_folds=len(results), total=30)
        finally:
            for child in children:
                child.join(timeout=5)
            if len(workers) < 2:
                for child in children:
                    if child.is_alive():
                        child.terminate()
                        child.join()
            q.close()
            q.join_thread()
        assert all(p.exitcode == 0 for p in children)
        assert len(results) == 30 and all(w['folds'] == 15 for w in workers)
        log('aggregation_started', folds=30, workers=2)
        a, b, by_seed, _raw = build_aggregates(results, meta, old)
        execution = {'workers': workers, 'fold_counts': {str(w['gpu']): w['folds'] for w in workers},
                     'cpu_threads_per_worker': 1, 'full_core_forwards': 0, 'training_fits': 0,
                     'branch_checkpoints': 60, 'elapsed_seconds_before_final_input_rehash': time.perf_counter()-started}
        limits = [
            'Historical supervised RFE contamination and transductive population graph remain; train-only normative references do not make the full pipeline leakage-free.',
            'Three seeds and five draws repeat the same871 people, not independent clinical cohorts.',
            'Reference TD OOF feature prediction, diagnostic complementarity, and medical causality are separate questions.',
            'P0 and P5 are independently trained cores; matched inference seed identifiers do not prove matching full random-state consumption.',
            'P4/P5/P6 are separately jointly trained; only within-checkpoint counterfactuals identify direct frozen branch interventions.',
            'PCA mean-difference retention is a descriptive training-set direction and may reflect site/age distributions or RFE contamination.',
            'No diagnostic threshold, reference model or hyperparameter selected on test data; this remains posthoc research attribution.'
        ]
        out = {'status': 'RESEARCH_IN_PROGRESS', 'meta': {'experiment': CFG['experiment'], 'created_at': created,
               'root': str(ROOT), 'config': CFG, 'analysis_code_sha256': sha256(OUT / 'run.py'),
               'original_protocol_preserved': True, 'ASD_label': 0, 'subject_counts': [403, 468],
               'primary_effect_unit': CFG['primary_effect_unit'], 'folds': 30},
            'audits': {'initial': audit, 'own_source': own}, 'stage_a': a, 'stage_b': b,
            'by_seed': by_seed, 'pairwise_decomposition': {
                'definition': 'CE(P5true)-CE(P0) = CE(P5off)-CE(P0) + CE(P5true)-CE(P5off)',
                'folds': [{'seed': r['seed'], 'fold': r['fold'],
                           'means': {k: v['mean'] for k, v in r['P5']['effects'].items()},
                           'identity_max_abs': r['P5']['CE_identity_max_abs']} for r in b['folds']],
                'aggregate': b['groups']['P5']['effects'], 'ACC_AUC_F1_not_additively_decomposed': True},
            'synthetic_tests': synthetic, 'limitations': limits, 'file_hashes': manifest,
            'historical_E5_formal': {g: old['groups'][g]['formal'] for g in ('P0', 'P1', 'P2', 'P3', 'P4', 'P5', 'P6')},
            'execution': execution}
        if previous is not None:
            out['audits']['implementation_revision'] = {
                'previous_incomplete_attempt': previous,
                'reason': 'Historical MAD was float32; independent float64 mean differed by1.7486643101409516e-12. Added exact legacy-dtype crosscheck; kept primary float64 and all reference/branch tolerances unchanged.',
                'historical_inputs_modified': False,
                'first_attempt_handoff_written': False}
        out['verdict'] = verdict_from_evidence(out)
        final = verify_new_aggregation(out, results, old)
        final['input_files_rehashed_unchanged'] = rehash_inputs(manifest)
        final['historical_checkpoint_and_cache_writes'] = 0
        final['synthetic_checks_passed'] = synthetic['count']
        assert own_source_audit()['passed']
        assert initial['analysis_code_sha256'] == sha256(OUT / 'run.py')
        assert initial['config_sha256'] == sha256(OUT / 'config.json')
        out['audits']['final'] = final
        out['research_verified_at'] = utc()
        out['execution']['total_research_seconds'] = time.perf_counter()-started
        out = finite_json(out)
        write_report(out)
        assert (OUT / 'report.md').stat().st_size > 5000
        out['report_sha256'] = sha256(OUT / 'report.md')
        out['status'] = 'RESEARCH_VERIFIED_HANDOFF_PENDING'
        out['handoff'] = {'status': 'pending', 'before': {'bytes': len(handoff_before),
                          'sha256': hashlib.sha256(handoff_before).hexdigest()}, 'section': handoff_text(out)}
        atomic_json(out, summary_path)
        finish_delivery(out)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        finished = OUT / 'summary.json'
        if finished.exists() and json.loads(finished.read_text()).get('status') == 'COMPLETE':
            print(traceback.format_exc(), file=sys.stderr)
        else:
            log('FAILED', traceback=traceback.format_exc())
        raise
