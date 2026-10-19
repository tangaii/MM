"""Train-only transforms, immutable historical splits and joint label-free nulls."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import io
import json
import os
import pickle
import struct
import tempfile
import time
from pathlib import Path

import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = Path(__file__).resolve().parent
ROOT = Path('/data/adodas/MM-GTUNets')
CFG = json.loads((BASE / 'config.json').read_text())
MAGIC = b'CMCV10\x00\x01'


def array_sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()


def stable_seed(*parts):
    return int(hashlib.sha256(('/'.join(map(str, ('CMCV',) + parts))).encode()).hexdigest()[:8], 16)


def atomic_bytes(path, value):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(value)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        dfd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
        assert path.read_bytes() == value
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_artifact(path, obj):
    value = pickle.dumps(obj, protocol=5)
    atomic_bytes(path, MAGIC + hashlib.sha256(value).digest() + value)


def load_artifact(path):
    data = Path(path).read_bytes()
    assert data[:len(MAGIC)] == MAGIC, f'Bad magic: {path}'
    digest, payload = data[len(MAGIC):len(MAGIC) + 32], data[len(MAGIC) + 32:]
    assert hashlib.sha256(payload).digest() == digest, f'SHA mismatch: {path}'
    return pickle.loads(payload)


def clean_json(obj):
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return clean_json(obj.tolist())
    if isinstance(obj, np.generic):
        return clean_json(obj.item())
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def save_json(path, obj):
    atomic_bytes(path, (json.dumps(clean_json(obj), indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode())


@contextlib.contextmanager
def task_lock(path):
    # Lock the persistent artifact inode's companion: never unlink a live lock file.
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
    finally:
        os.close(fd)


def source_fingerprints():
    names = ['config.json', 'data_protocol.py', 'probes.py', 'run.py', 'analyze.py']
    values = {n: file_sha(BASE / n) for n in names}
    return values, hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def validate_split(split, n=871):
    arrays = [np.asarray(split[p], np.int64) for p in ('train', 'val', 'test')]
    union = np.concatenate(arrays)
    assert np.array_equal(np.sort(union), np.arange(n))
    assert len(union) == len(set(union.tolist()))
    assert len(arrays[0]) > 64 and len(arrays[2]) > 0
    return dict(zip(('train', 'val', 'test'), arrays))


def task_definitions(inputs):
    tasks = []
    for seed in CFG['seeds']:
        for fold in range(10):
            tasks.append({'id': f'R_s{seed}_f{fold}', 'protocol': 'R', 'seed': seed, 'fold': fold,
                          'lane': (CFG['seeds'].index(seed) * 10 + fold) % 2,
                          'split': validate_split(inputs['splits'][f'{seed}/{fold}'])})
    for i, site in enumerate(sorted(set(inputs['site']))):
        tasks.append({'id': f'S_{site}', 'protocol': 'S', 'seed': 20261019, 'site': site,
                      'lane': i % 2, 'split': validate_split({'train': np.flatnonzero(inputs['site'] != site),
                          'val': np.empty(0, np.int64), 'test': np.flatnonzero(inputs['site'] == site)})})
    assert len(tasks) == 50
    return tasks


def train_image_imputer(x_train):
    finite = np.isfinite(x_train)
    if finite.all():
        median = np.median(x_train, axis=0)
    else:
        with np.errstate(all='ignore'):
            median = np.nanmedian(np.where(finite, x_train, np.nan), axis=0)
        median = np.where(np.isfinite(median), median, 0.)
    return median


def impute_image(x, median):
    return np.where(np.isfinite(x), x, median)


def fit_rfe(x_train, y_train, train_ids):
    """No full-cohort feature array or held-out labels enter this fitting boundary."""
    assert x_train.ndim == 2 and x_train.shape[1] == 6670
    assert len(x_train) == len(y_train) == len(train_ids)
    assert set(y_train.tolist()) == {0, 1}
    start = time.monotonic()
    median = train_image_imputer(x_train)
    clean = impute_image(x_train, median)
    selector = RFE(RidgeClassifier(), n_features_to_select=500, step=100)
    selector.fit(clean, y_train)
    selected = np.flatnonzero(selector.support_)
    assert len(selected) == 500 and selected.min() >= 0 and selected.max() < 6670
    return {'selected': selected, 'impute_median': median, 'ranking': selector.ranking_,
            'train_ids_sha256': array_sha(train_ids), 'train_labels_sha256': array_sha(y_train),
            'train_raw_features_sha256': array_sha(x_train), 'selected_sha256': array_sha(selected),
            'fit_seconds': time.monotonic() - start, 'fits': 1,
            'estimator': selector.estimator_.get_params(), 'fit_shape': clean.shape}


def canonical_age(values):
    values = np.asarray(values, np.float64)
    return np.where(np.isfinite(values) & (values > 0), values, np.nan)


def canonical_sex(values):
    values = np.asarray(values)
    return np.where(values == 1, 'M', np.where(values == 2, 'F', 'UNK'))


def fit_phenotype(age_train, sex_train, site_train):
    age_train = canonical_age(age_train)
    valid = age_train[np.isfinite(age_train)]
    median = float(np.median(valid)) if len(valid) else 0.
    a = np.where(np.isfinite(age_train), age_train, median)
    mean, sd = float(a.mean()), float(a.std())
    sd = sd if sd > 1e-12 else 1.
    sex_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float64)
    site_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float64)
    sex_encoder.fit(canonical_sex(sex_train).reshape(-1, 1))
    site_encoder.fit(np.asarray(site_train).reshape(-1, 1))
    state = {'age_median': median, 'age_mean': mean, 'age_sd': sd,
             'sex_encoder': sex_encoder, 'site_encoder': site_encoder, 'r_ph': 1.,
             'sex_categories': sex_encoder.categories_[0].tolist(), 'site_categories': site_encoder.categories_[0].tolist()}
    ph, _ = transform_phenotype(age_train, sex_train, site_train, state)
    state['r_ph'] = max(float(np.sqrt(np.sum(ph ** 2, axis=1).mean())), 1e-12)
    diff = np.abs(a[:, None] - a[None, :])
    state['age_bandwidth'] = max(float(np.median(diff[~np.eye(len(a), dtype=bool)])), 2.)
    state['train_covariates_sha256'] = {
        'age': array_sha(age_train), 'sex': array_sha(sex_train), 'site': array_sha(site_train)}
    return state


def transform_phenotype(age, sex, site, state):
    age = canonical_age(age)
    missing = ~np.isfinite(age)
    a = np.where(missing, state['age_median'], age)
    cat = canonical_sex(sex)
    ph = np.column_stack(((a - state['age_mean']) / state['age_sd'], missing.astype(float),
                          state['sex_encoder'].transform(cat.reshape(-1, 1)), (cat == 'UNK').astype(float)))
    ph /= state['r_ph']
    sh = state['site_encoder'].transform(np.asarray(site).reshape(-1, 1))
    assert np.isfinite(ph).all() and np.isfinite(sh).all()
    return ph, sh


def fit_preprocessing(inputs, task, rfe):
    tr = task['split']['train']
    selected = rfe['selected']
    selected_raw = impute_image(inputs['x'][tr][:, selected], rfe['impute_median'][selected])
    scaler = StandardScaler().fit(selected_raw)
    ph_state = fit_phenotype(inputs['age'][tr], inputs['sex'][tr], inputs['site'][tr])
    return {'scaler': scaler, 'ph_state': ph_state, 'fit_indices_sha256': array_sha(tr),
            'fit_ids_sha256': array_sha(inputs['ids'][tr]), 'scaler_train_sha256': array_sha(selected_raw)}


def apply_preprocessing(inputs, rfe, prep):
    selected = rfe['selected']
    raw = impute_image(inputs['x'][:, selected], rfe['impute_median'][selected])
    zimg = prep['scaler'].transform(raw) / np.sqrt(500.)
    ph, site = transform_phenotype(inputs['age'], inputs['sex'], inputs['site'], prep['ph_state'])
    assert zimg.shape == (871, 500) and np.isfinite(zimg).all()
    return zimg, ph, site


def make_nulls(inputs, task):
    site, age, sex = inputs['site'], canonical_age(inputs['age']), canonical_sex(inputs['sex'])
    nulls, records = [], []
    n = len(site)
    for b, base_seed in enumerate(CFG['null']['base_seeds']):
        mapping, info = np.arange(n, dtype=np.int64), {}
        for part, members in task['split'].items():
            single = 0
            for name in sorted(set(site[members])):
                idx = members[site[members] == name]
                rng = np.random.default_rng(stable_seed(base_seed, task['id'], part, name))
                order = rng.permutation(idx)
                if len(idx) > 1:
                    mapping[order] = np.roll(order, int(rng.integers(1, len(idx))))
                    assert np.all(mapping[idx] != idx)
                else:
                    single += len(idx)
                assert np.array_equal(np.sort(mapping[idx]), np.sort(idx))
            assert np.array_equal(np.sort(mapping[members]), np.sort(members))
            assert np.array_equal(site[members], site[mapping[members]])
            amoved = ~((age[members] == age[mapping[members]]) |
                       (np.isnan(age[members]) & np.isnan(age[mapping[members]])))
            smoved = sex[members] != sex[mapping[members]]
            info[part] = {'n': len(members), 'source_sha256': array_sha(mapping[members]),
                          'singletons': single, 'moved_fraction': float(np.mean(mapping[members] != members)) if len(members) else None,
                          'changed_fraction': float(np.mean(amoved | smoved)) if len(members) else None,
                          'age_changed_fraction': float(np.mean(amoved)) if len(members) else None,
                          'sex_changed_fraction': float(np.mean(smoved)) if len(members) else None,
                          'age_mean_abs_change': float(np.nanmean(np.abs(age[members] - age[mapping[members]]))) if len(members) else None}
        assert np.array_equal(np.sort(mapping), np.arange(n))
        nulls.append(mapping)
        records.append({'null': b, 'base_seed': base_seed, 'mapping_sha256': array_sha(mapping), 'parts': info})
    return np.stack(nulls), records


def cgroup_resources():
    root = Path('/sys/fs/cgroup')
    if (root / 'memory.max').exists():
        maximum = (root / 'memory.max').read_text().strip()
        current = int((root / 'memory.current').read_text())
        stat = dict(line.split() for line in (root / 'memory.stat').read_text().splitlines())
        cpu_max = (root / 'cpu.max').read_text().strip()
        reclaimable = int(stat.get('inactive_file', 0))
        version = 2
    else:
        maximum = (root / 'memory/memory.limit_in_bytes').read_text().strip()
        current = int((root / 'memory/memory.usage_in_bytes').read_text())
        stat = dict(line.split() for line in (root / 'memory/memory.stat').read_text().splitlines())
        quota = (root / 'cpu/cpu.cfs_quota_us').read_text().strip()
        period = (root / 'cpu/cpu.cfs_period_us').read_text().strip()
        cpu_max = ('max' if int(quota) < 0 else quota) + ' ' + period
        reclaimable = int(stat.get('total_inactive_file', stat.get('inactive_file', 0)))
        version = 1
    return {'version': version, 'memory_max': maximum, 'memory_current': current,
            'inactive_file': reclaimable, 'cpu_max': cpu_max,
            'available_reclaimable': int(maximum) - current + reclaimable if maximum != 'max' else 2 ** 60}


def available_ram():
    return cgroup_resources()['available_reclaimable']
