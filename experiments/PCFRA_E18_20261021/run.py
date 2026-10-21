"""Audited PCFRA preflight and paired PPU training."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
OLD = ROOT / 'experiments/DRPA_20261014'
HIST = ROOT / 'experiments/PCDCA_20261019'
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from opt import OptInit
from model.mm_gtunets import MM_GTUNets
from model.rp_graph import create_reward_penalty_graph
from utils.tools import EarlyStopping
from ranking import PCFRA_E18_Objective, array_hash, prepare

CFG = json.loads((BASE / 'config.json').read_text())
GROUPS = ('J1', 'J2', 'J3', 'J4')
SOURCE = ('opt.py', 'train_mm_gtunets.py', 'model/mm_gtunets.py', 'model/gtunet.py',
          'model/rp_graph.py', 'utils/mydataloader.py', 'utils/tools.py', 'utils/metrics.py')
CODE = ('config.json', 'ranking.py', 'run.py', 'launch.py', 'analyze.py')
SUBSETS = ('train', 'val', 'test')


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_bytes(path):
    path = Path(path)
    for attempt in range(6):
        try:
            return path.read_bytes()
        except FileNotFoundError:
            if attempt == 5:
                raise
            time.sleep(0.3 * (attempt + 1))


def sha_file(path):
    return hashlib.sha256(read_bytes(path)).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                                     separators=(',', ':')).encode()).hexdigest()


def payload_hash(value):
    h = hashlib.sha256()

    def add(v):
        if torch.is_tensor(v):
            h.update(b'tensor')
            add(v.detach().cpu().numpy())
        elif isinstance(v, np.ndarray):
            h.update(b'array')
            h.update(array_hash(v).encode())
        elif isinstance(v, np.generic):
            add(v.item())
        elif isinstance(v, dict):
            h.update(b'dict')
            for k in sorted(v, key=repr):
                add(k)
                add(v[k])
        elif isinstance(v, (list, tuple)):
            h.update(str(type(v)).encode())
            h.update(str(len(v)).encode())
            for x in v:
                add(x)
        elif v is None or isinstance(v, (str, int, float, bool)):
            h.update(str(type(v)).encode())
            h.update(repr(v).encode())
        else:
            raise TypeError(f'Unhashable artifact value {type(v)}')

    add(value)
    return h.hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode())
        v = state[k]
        h.update(array_hash(v.detach().cpu().numpy() if torch.is_tensor(v) else v).encode())
    return h.hexdigest()


def export_core(core):
    return {'state': {k: v.detach().cpu().clone() for k, v in core.state_dict().items()},
            'rp_weights': core.rp_attention.weights.detach().cpu().clone()}


def core_hash(saved):
    return state_hash({**saved['state'], 'unregistered_rp_weights': saved['rp_weights']})


def import_core(core, saved):
    core.load_state_dict(saved['state'], strict=True)
    with torch.no_grad():
        core.rp_attention.weights.copy_(saved['rp_weights'].to(core.device))


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'cpu': torch.get_rng_state().clone(),
            'cuda': torch.cuda.get_rng_state(0).clone(),
            'backend': {'cudnn_deterministic': torch.backends.cudnn.deterministic,
                        'cudnn_benchmark': torch.backends.cudnn.benchmark,
                        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32}}


def restore_rng(s):
    random.setstate(s['python'])
    np.random.set_state(s['numpy'])
    torch.set_rng_state(s['cpu'])
    torch.cuda.set_rng_state(s['cuda'], 0)
    b = s['backend']
    torch.backends.cudnn.deterministic = b['cudnn_deterministic']
    torch.backends.cudnn.benchmark = b['cudnn_benchmark']
    torch.backends.cuda.matmul.allow_tf32 = b['matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32 = b['cudnn_allow_tf32']


def rng_hash(s):
    return {'python': hashlib.sha256(repr(s['python']).encode()).hexdigest(),
            'numpy': hashlib.sha256(repr(s['numpy'][0]).encode() + s['numpy'][1].tobytes()
                                     + repr(s['numpy'][2:]).encode()).hexdigest(),
            'cpu': array_hash(s['cpu'].numpy()), 'cuda': array_hash(s['cuda'].numpy())}


def atomic_bytes(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with tmp.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    if read_bytes(path) != data:
        raise IOError(f'Atomic readback mismatch: {path}')


def atomic_json(value, path):
    atomic_bytes((json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode(), path)


def atomic_torch(value, path):
    b = io.BytesIO()
    torch.save({'payload': value, 'content_sha256': payload_hash(value)}, b)
    atomic_bytes(b.getvalue(), path)


def load_torch(path, artifact=False):
    value = torch.load(io.BytesIO(read_bytes(path)), map_location='cpu', weights_only=False)
    if artifact:
        if payload_hash(value['payload']) != value['content_sha256']:
            raise IOError(f'Artifact content hash mismatch: {path}')
        return value['payload']
    return value


def log(event, **values):
    row = {'utc': utc(), 'event': event, **values}
    line = json.dumps(row, ensure_ascii=False, allow_nan=False)
    if os.environ.get('PCFRA_LOG_STDOUT') == '1':
        print(line, flush=True)
        return
    with (BASE / 'run.log').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(line + '\n')
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


@contextmanager
def task_lock(slot):
    with (BASE / 'run.py').open('r+') as f:
        fcntl.lockf(f, fcntl.LOCK_EX | fcntl.LOCK_NB, 1, 1000 + slot)
        try:
            yield
        finally:
            fcntl.lockf(f, fcntl.LOCK_UN, 1, 1000 + slot)


def code_hash():
    frozen = ('config.json', 'ranking.py', 'run.py', 'launch.py')
    return digest({name: sha_file(BASE / name) for name in frozen})


def anchor(seed, fold):
    return OLD / 'artifacts' / f'seed_{seed}' / f'fold_{fold}'


def historical_anchor(seed, fold):
    return OLD / 'artifacts' / f'seed_{seed}' / f'fold_{fold}'


def result_path(group, seed, fold):
    return BASE / 'runs' / group / f's{seed}_f{fold}.pt'


def metadata(ctx):
    with (ROOT / CFG['data_folder'] / 'Phenotypic_V1_0b_preprocessed1.csv').open() as f:
        rows = {r['SUB_ID']: r for r in csv.DictReader(f)}
    ordered = [rows[i] for i in ctx['ids']]
    y = np.asarray([int(r['DX_GROUP']) - 1 for r in ordered])
    assert np.array_equal(y, ctx['labels']) and np.array_equal(np.bincount(y), [403, 468])

    def number(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return float('nan')

    age = np.asarray([number(r['AGE_AT_SCAN']) for r in ordered])
    sex = np.asarray([number(r['SEX']) for r in ordered])
    site = np.asarray([r['SITE_ID'] for r in ordered])
    assert np.array_equal(site, ctx['site_names']) and len(set(site)) == 20
    assert np.allclose(age, ctx['ph_data'][:, 2], rtol=0, atol=2e-6)
    assert np.array_equal(sex - 1, ctx['ph_data'][:, 1])
    assert [(int(np.sum((y == k) & (sex == 2))), int(np.sum((y == k) & (sex == 1))))
            for k in (0, 1)] == [(54, 349), (90, 378)]
    return {'age': age, 'sex': sex, 'site': site}


def binary_metrics(y, p, pred):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(p, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.int64)
    truth = (y == 0).astype(np.int8)
    return {'ACC': float(accuracy_score(y, pred)),
            'SEN': float(recall_score(y, pred, pos_label=0)),
            'SPE': float(recall_score(y, pred, pos_label=1)),
            'AUC': float(roc_auc_score(truth, p))}


def historical_d4(ctx):
    rows = []
    protected = {}
    for fold in range(10):
        path = HIST / 'runs/D4' / f's911_f{fold}.pt'
        value = load_torch(path, artifact=True)
        assert (value['group'], value['seed'], value['fold']) == ('D4', 911, fold)
        split = load_torch(anchor(911, fold) / 'splits.pt')
        idx, y, logits = value['oof']['indices'], value['oof']['labels'], np.asarray(value['oof']['logits'])
        assert np.array_equal(idx, split['test'])
        assert np.array_equal(y, ctx['labels'][idx])
        assert value['oof']['ids'] == [ctx['ids'][int(i)] for i in idx]
        p = torch.softmax(torch.as_tensor(logits, dtype=torch.float64), dim=1)[:, 0].numpy()
        pred = logits.argmax(axis=1)
        metrics = binary_metrics(y, p, pred)
        assert max(abs(metrics[k] - value['metrics'][k]) for k in metrics) < 1e-12
        rows.append({'fold': fold, 'indices': idx, 'labels': y, 'probability': p, 'prediction': pred,
                     'metrics': metrics, 'best_epoch': value['best_epoch'],
                     'best_val_loss': value['best_val_loss'], 'epochs_run': value['epochs_run'],
                     'training_seconds': value['training_seconds'],
                     'best_core_sha256': value['best_core_sha256']})
        protected[str(path)] = sha_file(path)
    means = {k: float(np.mean([r['metrics'][k] for r in rows]) * 100) for k in ('ACC', 'SEN', 'SPE', 'AUC')}
    for key, target in CFG['historical_reference_percent']['historical_D4'].items():
        if abs(means[key] - target) > 0.00005:
            raise RuntimeError(f'Historical D4 {key} mean {means[key]:.8f} differs from prompt value {target}')
    sds = {k: float(np.std([r['metrics'][k] for r in rows], ddof=1) * 100) for k in means}
    return rows, means, sds, protected


def historical_g2_g3(ctx):
    base = ROOT / CFG['historical_g2_g3']
    rows, protected = {}, {}
    for group in ('G2', 'G3'):
        folds = []
        for fold in range(10):
            path = base / 'runs' / group / f's911_f{fold}.pt'
            value = load_torch(path, artifact=True)
            assert (value['group'], value['seed'], value['fold']) == (group, 911, fold)
            split = load_torch(anchor(911, fold) / 'splits.pt')
            indices = value['oof']['indices']
            labels = value['oof']['labels']
            logits = np.asarray(value['oof']['logits'])
            assert np.array_equal(indices, split['test'])
            assert np.array_equal(labels, ctx['labels'][indices])
            assert value['oof']['ids'] == [ctx['ids'][int(i)] for i in indices]
            probability = torch.softmax(torch.as_tensor(logits, dtype=torch.float64), dim=1)[:, 0].numpy()
            prediction = logits.argmax(axis=1)
            measured = binary_metrics(labels, probability, prediction)
            assert max(abs(measured[k] - value['metrics'][k]) for k in measured) < 1e-12
            folds.append({'fold': fold, 'indices': indices, 'labels': labels,
                          'probability': probability, 'prediction': prediction, 'metrics': measured,
                          'best_epoch': value['best_epoch'], 'best_val_loss': value['best_val_loss'],
                          'epochs_run': value['epochs_run'], 'training_seconds': value['training_seconds'],
                          'best_core_sha256': value['best_core_sha256']})
            protected[str(path)] = sha_file(path)
        rows[group] = folds
    summary = {}
    for group, folds in rows.items():
        summary[group] = {
            'mean_percent': {m: float(np.mean([r['metrics'][m] for r in folds]) * 100)
                             for m in ('ACC', 'SEN', 'SPE', 'AUC')},
            'sample_sd_percent': {m: float(np.std([r['metrics'][m] for r in folds], ddof=1) * 100)
                                  for m in ('ACC', 'SEN', 'SPE', 'AUC')},
            'retrained': False,
        }
        for metric, target in CFG['historical_reference_percent'][f'historical_{group}'].items():
            if abs(summary[group]['mean_percent'][metric] - target) > 0.00005:
                raise RuntimeError(f'Historical {group} {metric} mean differs from fixed reference')
    return rows, summary, protected


def audit():
    path = BASE / 'audit.json'
    if path.exists():
        saved = json.loads(read_bytes(path))
        current_code = code_hash()
        code_changed = saved['code_sha256'] != current_code
        if code_changed:
            has_results = (BASE / 'runs').exists() and any((BASE / 'runs').rglob('*.pt'))
            if saved.get('formal_authorized') or (BASE / 'manifest.json').exists() or has_results:
                raise RuntimeError('Refusing to revise code after manifest freeze or formal task output')
            for protected_path, expected_hash in saved['protected_sha256'].items():
                assert sha_file(protected_path) == expected_hash, f'Protected input changed: {protected_path}'
            for source_path, expected_hash in saved['source_sha256'].items():
                assert sha_file(ROOT / source_path) == expected_hash, f'Protected source changed: {source_path}'
            previous = saved['code_sha256']
            saved['code_sha256'] = current_code
            saved.setdefault('code_revisions', []).append({
                'previous_code_sha256': previous, 'updated_code_sha256': current_code,
                'reason': 'complete protected-input recheck after pre-manifest implementation correction',
                'utc': utc(),
            })
            saved['preflight'] = {}
            atomic_json(saved, path)
        assert saved['code_sha256'] == current_code, 'Registered PCFRA code or config changed'
        guard(saved, deep=not code_changed)
        return saved

    expected = {'groups': list(GROUPS), 'seeds': [911], 'folds': 10, 'tasks': 40}
    assert list(CFG['groups']) == list(GROUPS) and CFG['seeds'] == [911] and CFG['folds'] == 10
    assert not ({'D0', 'D4', 'G1', 'G2', 'G3', 'G4', 'H1', 'H2', 'H3', 'H4'} & set(CFG['groups']))
    ctx = load_torch(OLD / 'artifacts/context.pt')
    meta = metadata(ctx)
    assert len(ctx['ids']) == len(set(ctx['ids'])) == 871
    assert np.array_equal(np.genfromtxt(ROOT / 'data/ABIDE_aal/id.txt', dtype=str).astype(str),
                          np.asarray(ctx['ids']).astype(str))

    protected_paths = set()
    protected_paths.update(ROOT / name for name in SOURCE)
    protected_paths.update(ROOT.glob('model/*.py'))
    protected_paths.update(ROOT.glob('utils/*.py'))
    protected_paths.update([ROOT / 'prompt',
                            ROOT / CFG['data_folder'] / 'id.txt',
                            ROOT / CFG['data_folder'] / 'Phenotypic_V1_0b_preprocessed1.csv',
                            ROOT / 'data/ABIDE_aal/id.txt',
                            ROOT / 'experiments/DRPA_20261014/artifacts/context.pt'])
    protected_paths.update(p for p in (ROOT / CFG['data_folder'] / 'save_tensor').glob('*') if p.is_file())
    for folder in ('PCDCA_20261019', 'PCFRA_20261020', 'PCHRRA_20261021'):
        base = ROOT / 'experiments' / folder
        protected_paths.update(p for p in base.rglob('*') if p.is_file() and p.suffix in ('.py', '.json', '.md'))
    for p in (OLD / 'artifacts').rglob('*'):
        if p.is_file():
            protected_paths.add(p)
    protected_paths.update(p for p in (HIST / 'runs/D4').glob('s911_f*.pt') if p.is_file())
    e16 = ROOT / CFG['historical_g2_g3']
    protected_paths.update(p for p in (e16 / 'runs/G2').glob('s911_f*.pt') if p.is_file())
    protected_paths.update(p for p in (e16 / 'runs/G3').glob('s911_f*.pt') if p.is_file())
    protected = {str(p): sha_file(p) for p in sorted(protected_paths, key=str) if p.is_file()}

    raw = torch.as_tensor(load_torch(ROOT / CFG['data_folder'] / 'save_tensor/up_triangles.pt'))
    assert tuple(raw.shape) == (871, 6670) and bool(torch.isfinite(raw).all())
    historical_pcfra_audit = json.loads(read_bytes(
        ROOT / CFG['historical_g2_g3'] / 'audit.json'))
    anchors = {}
    historical_test_counts = np.zeros(871, dtype=np.int64)
    for fold in range(10):
        a = anchor(911, fold)
        split = load_torch(a / 'splits.pt')
        start = load_torch(a / 'train_start.pt')
        fp = json.loads(read_bytes(a / 'fingerprints.json'))
        union = np.concatenate([split[k] for k in SUBSETS])
        assert len(union) == 871 and np.array_equal(np.sort(union), np.arange(871))
        historical_test_counts[split['test']] += 1
        for k in SUBSETS:
            assert array_hash(split[k]) == fp['split_fingerprints'][k]
        selected = load_torch(a.parent / 'selected_features.pt')
        support = np.load(io.BytesIO(read_bytes(a.parent / 'rfe_support.npy')))
        assert selected.shape == (871, 500) and torch.isfinite(selected).all()
        assert support.shape == (500,) and np.issubdtype(support.dtype, np.integer)
        assert len(np.unique(support)) == 500 and support.min() >= 0 and support.max() < 6670
        assert torch.equal(raw[:, support].to(selected.dtype), selected)
        assert array_hash(selected.numpy()) == fp['features_fingerprint']
        assert core_hash(start['core']) == fp['train_start_core_fingerprint']
        assert rng_hash(start['rng']) == fp['classifier_start_rng']
        vae = a.parent / f'vae/fold{fold}_pretrain.pth'
        assert sha_file(vae) == start['vae_sha256'] == fp['vae_sha256']
        prepared = prepare(ctx['labels'], meta, split['train'], 911, fold, CFG)
        assert np.unique(prepared['folds']).tolist() == [0, 1, 2]
        assert prepared['info']['prototype_fold_sha256'] == historical_pcfra_audit[
            'anchors'][str(fold)]['prototype']['prototype_fold_sha256']
        input_paths = [a / 'splits.pt', a / 'train_start.pt', a / 'fingerprints.json',
                       a.parent / 'selected_features.pt', a.parent / 'rfe_support.npy', vae]
        anchors[str(fold)] = {
            'input_sha256': {str(p): sha_file(p) for p in input_paths},
            'pairing': fp,
            'prototype': prepared['info'],
            'physical_gpu': fold % 2,
        }
    assert np.all(historical_test_counts == 1)
    d4_rows, d4_mean, d4_sd, d4_protected = historical_d4(ctx)
    protected.update(d4_protected)
    historical_rows, historical_summary, historical_hashes = historical_g2_g3(ctx)
    protected.update(historical_hashes)
    out = {
        'experiment': CFG['experiment'], 'started_utc': utc(), 'code_sha256': code_hash(),
        'prompt_sha256': sha_file(ROOT / 'prompt'),
        'source_sha256': {n: sha_file(ROOT / n) for n in SOURCE},
        'protected_sha256': protected,
        'handoff_before_sha256': sha_file(ROOT / 'handoff.md'),
        'handoff_before_bytes': len(read_bytes(ROOT / 'handoff.md')),
        'context_sha256': sha_file(OLD / 'artifacts/context.pt'),
        'anchors': anchors,
        'data': {'n': 871, 'class_counts': [403, 468], 'sites': 20,
                 'image_shape': [871, 500], 'sex_csv': '1=male, 2=female verified against context and published counts',
                 'age_range': [float(meta['age'].min()), float(meta['age'].max())]},
        'historical_d4_readonly': {
            'source': str(HIST / 'runs/D4'),
            'fold_metrics': [{'fold': r['fold'], 'metrics_percent': {k: v * 100 for k, v in r['metrics'].items()}}
                             for r in d4_rows],
            'mean_percent': d4_mean, 'sample_sd_percent': d4_sd,
            'checkpoint_sha256': d4_protected,
            'retrained': False,
        },
        'historical_g2_g3_readonly': {
            'source': str(ROOT / CFG['historical_g2_g3'] / 'runs'),
            'summary': historical_summary,
            'checkpoint_sha256': historical_hashes,
        },
        'environment': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__},
        'preflight': {}, 'formal_authorized': False, 'execution': {'events': [], 'failures': [], 'resources': []},
        'task_boundary': expected,
        'literature': {'MM_GTUNets': 'https://arxiv.org/html/2406.14455v3'},
    }
    atomic_json(out, path)
    log('audit_passed', anchors=len(anchors), historical_d4_folds=len(d4_rows),
        protected_files=len(protected), code_sha256=out['code_sha256'])
    return out


def guard(saved=None, deep=True):
    saved = saved or json.loads(read_bytes(BASE / 'audit.json'))
    assert saved['code_sha256'] == code_hash(), 'Frozen E18 training code/config fingerprint mismatch'
    if deep:
        for path, expected in saved['protected_sha256'].items():
            if sha_file(path) != expected:
                raise RuntimeError(f'Protected historical input changed: {path}')
    for path, expected in saved['source_sha256'].items():
        assert sha_file(ROOT / path) == expected
    handoff = read_bytes(ROOT / 'handoff.md')
    prefix_bytes = saved['handoff_before_bytes']
    if len(handoff) < prefix_bytes or hashlib.sha256(handoff[:prefix_bytes]).hexdigest() != \
            saved['handoff_before_sha256']:
        raise RuntimeError('The original handoff.md prefix changed')
    return saved


class Loaded:
    def __init__(self, seed, fold, audit_info):
        if not torch.cuda.is_available():
            raise RuntimeError('No PPU; CPU neural training is forbidden')
        torch.cuda.set_device(0)
        torch.set_num_threads(CFG['host_threads'])
        device_name = torch.cuda.get_device_name(0)
        if 'PPU' not in device_name.upper():
            raise RuntimeError(f'Expected PPU, found {device_name}')
        self.seed, self.fold = seed, fold
        self.ctx = load_torch(OLD / 'artifacts/context.pt')
        self.meta = metadata(self.ctx)
        self.sp = load_torch(anchor(seed, fold) / 'splits.pt')
        self.start = load_torch(anchor(seed, fold) / 'train_start.pt')
        self.anchor = audit_info['anchors'][str(fold)]
        for p, h in self.anchor['input_sha256'].items():
            assert sha_file(p) == h, f'Input changed: {p}'
        argv = sys.argv
        try:
            sys.argv = ['native_options']
            opt = OptInit(model='MM_GTUNets', dataset='ABIDE', atlas='aal').args
        finally:
            sys.argv = argv
        opt.seed = seed
        opt.device = torch.device('cuda:0')
        opt.data_folder = str(ROOT / CFG['data_folder'])
        opt.ckpt_path = str(anchor(seed, fold).parent / 'vae')
        opt.epoch = CFG['epochs']
        opt.early_stop = CFG['early_stop']
        opt.lr = CFG['lr']
        opt.wd = CFG['weight_decay']
        opt.log_save = False
        opt.model_save = False
        self.opt = opt
        self.core = MM_GTUNets(opt, fold).to(opt.device)
        self.core.vae_save_path = str(anchor(seed, fold).parent / f'vae/fold{fold}_pretrain.pth')
        import_core(self.core, self.start['core'])
        assert core_hash(export_core(self.core)) == self.anchor['pairing']['train_start_core_fingerprint']
        self.x_cpu = load_torch(anchor(seed, fold).parent / 'selected_features.pt').float()
        self.ph_cpu = torch.from_numpy(self.ctx['ph_data']).float()
        affinity = create_reward_penalty_graph(self.ctx['ph_dict'], self.ctx['labels'],
                                                *[self.sp[k] for k in SUBSETS], opt)
        self.affinity_cpu = torch.from_numpy(affinity).float()
        self.x, self.ph, self.affinity = [v.to(opt.device) for v in
                                          (self.x_cpu, self.ph_cpu, self.affinity_cpu)]
        self.y = torch.as_tensor(self.ctx['labels'], dtype=torch.long, device=opt.device)
        self.prepared = prepare(self.ctx['labels'], self.meta, self.sp['train'], seed, fold, CFG)
        assert digest(self.prepared['info']) == digest(self.anchor['prototype'])
        self.z = None
        self.handle = self.core.img_unet.register_forward_hook(self.capture)
        self.input_digest = payload_hash({'x': self.x_cpu, 'ph': self.ph_cpu, 'affinity': self.affinity_cpu,
                                          'labels': self.ctx['labels'], 'ids': self.ctx['ids']})

    def capture(self, module, args, output):
        if tuple(output.shape) != (871, 16):
            raise AssertionError(f'Imaging representation shape {tuple(output.shape)}')
        self.z = output

    def optimizer(self):
        params = list(self.core.parameters())
        assert not any(p is self.core.rp_attention.weights for p in params)
        return torch.optim.Adam(params, lr=CFG['lr'], weight_decay=CFG['weight_decay'])

    def reset(self):
        import_core(self.core, self.start['core'])
        for p in self.core.parameters():
            p.grad = None
        restore_rng(self.start['rng'])
        torch.use_deterministic_algorithms(False)

    def forward(self):
        return self.core(self.x, self.ph, self.affinity)

    def close(self):
        self.handle.remove()
        self.__dict__.clear()
        torch.cuda.empty_cache()


def grad_norm(params):
    terms = [p.grad.detach().square().sum() for p in params if p.grad is not None]
    return float(torch.stack(terms).sum().sqrt()) if terms else 0.0


def gradient_budget(native, auxiliary, joint, epoch):
    g_native = torch.autograd.grad(native, joint, retain_graph=True)[0]
    g_aux = torch.autograd.grad(auxiliary, joint, retain_graph=True)[0]
    n, a = g_native.detach().norm(), g_aux.detach().norm()
    if not bool(torch.isfinite(n)) or not bool(torch.isfinite(a)):
        raise FloatingPointError('Nonfinite fusion gradient')
    if float(n) <= 1e-12:
        raise FloatingPointError('Native joint_embed gradient is zero')
    if epoch >= CFG['warmup_last_epoch'] and float(a) <= 1e-12:
        raise FloatingPointError('PCFRA auxiliary joint_embed gradient is zero after warmup')
    with torch.no_grad():
        lam = (CFG['auxiliary_gradient_ratio'] * n / a.clamp_min(CFG['epsilon'])).clamp(
            0.0, CFG['lambda_cap'])
        warmup = min(max(epoch, 0) / CFG['warmup_last_epoch'], 1.0)
        lam = (lam * warmup).detach()
    ratio = float(lam * a / n.clamp_min(CFG['epsilon']))
    return lam, {'native_joint_grad_norm': float(n), 'aux_joint_grad_norm': float(a),
                 'lambda_eff': float(lam), 'warmup': float(warmup),
                 'weighted_joint_grad_ratio': ratio}


def valid_result(group, seed, fold, audit_info, required=False):
    path = result_path(group, seed, fold)
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    value = load_torch(path, artifact=True)
    assert value['state'] == 'complete' and (value['group'], value['seed'], value['fold']) == (group, seed, fold)
    assert value['code_sha256'] == audit_info['code_sha256']
    assert value['input_sha256'] == audit_info['anchors'][str(fold)]['input_sha256']
    assert value['site_objective_static']['group'] == group
    assert value['objective_static'] == value['site_objective_static']
    assert value['best_core_sha256'] == core_hash(value['core'])
    assert value['new_trainable_parameters'] == value['rfe_fit_count'] == value['vae_train_count'] == 0
    assert value['formal_test_grad_enabled'] and 'PPU' in value['device'].upper()
    assert value['forward_calls'] == {'train': value['epochs_run'], 'val': value['epochs_run'], 'test': 1}
    assert value['prototype_preparation']['uses_heldout_labels_or_covariates'] is False
    split = load_torch(anchor(seed, fold) / 'splits.pt')
    assert np.array_equal(value['oof']['indices'], split['test'])
    assert np.array_equal(value['oof']['prediction'], np.asarray(value['oof']['logits']).argmax(axis=1))
    p = np.asarray(value['oof']['probability'])
    assert np.isfinite(p).all() and np.all((p >= 0.0) & (p <= 1.0))
    expected = torch.softmax(torch.as_tensor(value['oof']['logits'], dtype=torch.float64), dim=1)[:, 0].numpy()
    assert np.allclose(p, expected, atol=1e-7, rtol=1e-6)
    metrics = binary_metrics(value['oof']['labels'], p, value['oof']['prediction'])
    assert max(abs(metrics[k] - value['metrics'][k]) for k in metrics) < 1e-12
    assert value['best_epoch'] < value['epochs_run'] <= CFG['epochs']
    assert value['best_val_loss'] == min(epoch['val_loss'] for epoch in value['epochs'])
    assert value['best_epoch'] == min(range(value['epochs_run']), key=lambda i: value['epochs'][i]['val_loss'])
    for epoch_no, epoch in enumerate(value['epochs']):
        assert epoch['epoch'] == epoch_no
        assert np.isfinite(epoch['lambda_eff']) and 0.0 <= epoch['lambda_eff'] <= CFG['lambda_cap']
        assert np.isfinite(epoch['weighted_joint_grad_ratio'])
        assert -1e-8 <= epoch['weighted_joint_grad_ratio'] <= CFG['auxiliary_gradient_ratio'] + 1e-6
        assert epoch['objective']['prototype_used'] == (group != 'J1')
        if epoch_no >= CFG['warmup_last_epoch'] and group != 'J1':
            assert epoch['aux_joint_grad_norm'] > 0
    return value


def train_one(group, seed, fold, audit_info):
    if valid_result(group, seed, fold, audit_info) is not None:
        log('skip_valid', group=group, seed=seed, fold=fold)
        return
    if not audit_info.get('formal_authorized'):
        raise RuntimeError('Formal training requires completed preflight and frozen task manifest')
    l = Loaded(seed, fold, audit_info)
    call_handle = None
    try:
        objective = PCFRA_E18_Objective(l.prepared, group, l.opt.device)
        assert np.array_equal(l.prepared['train'], l.sp['train'])
        assert np.array_equal(objective.train.detach().cpu().numpy(), l.sp['train'])
        optimizer = l.optimizer()
        stopper = EarlyStopping(patience=CFG['early_stop'], save=False)
        l.reset()
        assert rng_hash(capture_rng()) == l.anchor['pairing']['classifier_start_rng']
        assert not torch.are_deterministic_algorithms_enabled()
        initial_order = [n for n, p in l.core.named_parameters()]
        trainable_names = {n for n, p in l.core.named_parameters() if p.requires_grad}
        new_trainable = trainable_names - set(l.start['core']['state'])
        assert not new_trainable, f'Unexpected trainable parameters: {sorted(new_trainable)}'
        bn_start = int(l.core.clf[2].num_batches_tracked)
        call_counts = {'train': 0, 'val': 0, 'test': 0}
        phase = {'value': 'train'}

        def count_forward(module, args, out):
            call_counts[phase['value']] += 1

        call_handle = l.core.register_forward_hook(count_forward)
        history, best, best_loss = [], None, 1e10
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started, started_utc = time.perf_counter(), utc()
        for epoch in range(CFG['epochs']):
            phase['value'] = 'train'
            l.core.train()
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                logits, graph = l.forward()
            if tuple(logits.shape) != (871, 2):
                raise AssertionError(f'Native logits shape {tuple(logits.shape)}')
            ce = F.cross_entropy(logits[l.sp['train']], l.y[l.sp['train']])
            native = ce + graph
            auxiliary = objective(logits, l.z)
            weight, scale = gradient_budget(native, auxiliary, l.core.joint_embed, epoch)
            total = native + weight * auxiliary
            if not bool(torch.isfinite(total)):
                raise FloatingPointError('Nonfinite total training objective')
            record = {'epoch': epoch, 'train_ce': float(ce.detach()), 'train_graph': float(graph.detach()),
                      'train_native': float(native.detach()), 'train_auxiliary': float(auxiliary.detach()),
                      **scale, 'objective': dict(objective.last_info)}
            total.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in l.core.parameters()):
                raise FloatingPointError('Nonfinite model gradient')
            optimizer.step()
            assert int(l.core.clf[2].num_batches_tracked) == bn_start + epoch + 1

            phase['value'] = 'val'
            l.core.eval()
            with torch.set_grad_enabled(False):
                val, val_graph = l.forward()
            val_ce = F.cross_entropy(val[l.sp['val']], l.y[l.sp['val']])
            val_loss = val_ce + val_graph
            if not bool(torch.isfinite(val_loss)):
                raise FloatingPointError('Nonfinite native validation loss')
            assert int(l.core.clf[2].num_batches_tracked) == bn_start + epoch + 1
            record.update(val_ce=float(val_ce), val_graph=float(val_graph), val_loss=float(val_loss),
                          train_total=float(total.detach()))
            history.append(record)
            if float(val_loss) < best_loss:
                best_loss = float(val_loss)
                best = {'core': export_core(l.core), 'epoch': epoch, 'val_loss': best_loss,
                        'record': record.copy()}
            stopper(val_loss, l.core)
            if epoch % 100 == 0:
                log('progress', group=group, seed=seed, fold=fold, epoch=epoch,
                    best_epoch=best['epoch'], val_loss=float(val_loss))
            if stopper.early_stop:
                break
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        assert best is not None
        end_rng = capture_rng()
        import_core(l.core, best['core'])
        l.core.eval()
        before = capture_rng()
        assert payload_hash(before) == payload_hash(end_rng), 'Loading best core changed RNG'
        phase['value'] = 'test'
        with torch.set_grad_enabled(True):
            test, _ = l.forward()
        call_handle.remove()
        call_handle = None
        assert call_counts == {'train': len(history), 'val': len(history), 'test': 1}
        after = capture_rng()
        indices = l.sp['test']
        logits_np = test.detach().cpu().numpy()[indices]
        probability = torch.softmax(test.detach(), dim=1).cpu().numpy()[indices, 0]
        oof = {'indices': indices, 'ids': [l.ctx['ids'][int(i)] for i in indices],
               'labels': l.ctx['labels'][indices], 'logits': logits_np,
               'probability': probability, 'prediction': logits_np.argmax(axis=1)}
        result = {
            'state': 'complete', 'group': group, 'seed': seed, 'fold': fold,
            'started_utc': started_utc, 'completed_utc': utc(),
            'code_sha256': audit_info['code_sha256'],
            'input_sha256': l.anchor['input_sha256'], 'data_input_digest': l.input_digest,
            'pairing': l.anchor['pairing'], 'prototype_preparation': l.prepared['info'],
            'objective_static': objective.static_info, 'core': best['core'],
            'best_core_sha256': core_hash(best['core']), 'best_epoch': best['epoch'],
            'best_val_loss': best_loss, 'best_record': best['record'], 'epochs': history,
            'epochs_run': len(history), 'training_seconds': seconds, 'oof': oof,
            'metrics': binary_metrics(oof['labels'], probability, oof['prediction']),
            'rng_before_test': before, 'rng_after_test': after,
            'formal_start_rng_sha256': rng_hash(l.start['rng']),
            'formal_test_grad_enabled': True,
            'physical_gpu': int(os.environ.get('CUDA_VISIBLE_DEVICES', '0')),
            'device': torch.cuda.get_device_name(0),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'optimizer_parameter_order': initial_order, 'new_trainable_parameters': len(new_trainable),
                'rfe_fit_count': 0, 'vae_train_count': 0, 'forward_calls': call_counts,
            'bn_best_batches': int(best['core']['state']['clf.2.num_batches_tracked']),
            'gradient_target': 'joint_embed',
                'site_objective_static': objective.static_info,
            'test_forwards': 1,
        }
        atomic_torch(result, result_path(group, seed, fold))
        valid_result(group, seed, fold, audit_info, required=True)
        log('complete', group=group, seed=seed, fold=fold, epochs=len(history),
            best_epoch=best['epoch'], seconds=seconds, physical_gpu=result['physical_gpu'])
    finally:
        if call_handle is not None:
            call_handle.remove()
        l.close()


def historical_prototype_module():
    path = HIST / 'prototype.py'
    spec = importlib.util.spec_from_file_location('pcdca_20261019_prototype', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def equal_prepared(left, right):
    for key in ('train', 'y', 'age', 'sex', 'site', 'folds', 'halves', 'permutation'):
        a, b = left[key], right[key]
        if not np.array_equal(a, b):
            raise AssertionError(f'Historical prepared field mismatch: {key}')
    assert digest(left['info']) == digest(right['info'])


def synthetic_checks(device):
    y = np.asarray([0, 1] * 5 + [0] * 7 + [1] * 2)
    sites = np.asarray(['A'] * 10 + ['B'] * 9 + ['C'] * 2)
    meta = {'age': np.asarray([np.nan, -1, 8, 9, 29, 16, 18, 23, 20, 21] + [19] * 11, float),
            'sex': np.asarray([-1, 1, 2, 1, 2, -1, 1, 2, 1, 2] + [1] * 11, float),
            'site': sites}
    train = np.arange(len(y))
    prepared = prepare(y, meta, train, 911, 0, CFG)
    logits = torch.linspace(-0.7, 0.8, len(y) * 2, device=device).reshape(len(y), 2).requires_grad_()
    z = torch.sin(torch.arange(len(y) * 16, device=device, dtype=torch.float32).reshape(len(y), 16) / 7)
    z.requires_grad_()
    outcomes, objectives, grads = {}, {}, {}
    for group in GROUPS:
        objective = PCFRA_E18_Objective(prepared, group, device)
        loss = objective.forward(logits, z)
        grad = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        assert bool(torch.isfinite(loss)) and bool(torch.isfinite(grad).all()) and float(grad.norm()) > 0
        assert objective.static_info['n_all_pairs'] == int((y == 0).sum() * (y == 1).sum())
        assert objective.static_info['effective_sites'] == ['A', 'B']
        if group != 'J1':
            assert all(abs(v - 1.0) < 1e-5
                       for v in objective.last_info['site_margin_weighted_mean'].values())
        outcomes[group] = float(loss.detach())
        objectives[group] = objective
        grads[group] = grad

    j2, j3, j4 = (objectives[g].last_tensors for g in ('J2', 'J3', 'J4'))
    assert torch.equal(j2['evidence'], j3['evidence']) and torch.equal(j2['D'], j3['D'])
    assert torch.equal(j2['temperature'], j3['temperature']) and torch.equal(j2['C'], j3['C'])
    assert torch.allclose(j2['margin'] - 1.0, 2.0 * (j3['margin'] - 1.0), atol=5e-7, rtol=1e-6)
    assert torch.equal(j3['margin'], j4['margin'])
    assert float(j2['margin'].min()) >= 0.5 and float(j2['margin'].max()) <= 1.5
    assert float(j3['margin'].min()) >= 0.75 and float(j3['margin'].max()) <= 1.25
    assert objectives['J1'].prototype_calls == 0
    assert all(objectives[g].prototype_calls == 1 for g in ('J2', 'J3', 'J4'))
    assert all(not objectives[g].last_tensors['evidence'].requires_grad for g in ('J2', 'J3', 'J4'))
    assert not np.isclose(outcomes['J3'], outcomes['J4'], atol=1e-5, rtol=1e-5)
    assert objectives['J1'].site_weights['A'] < 1.0

    # A site-wide scale on W cancels under anchor normalization; explicit site losses do not.
    obj = objectives['J1']
    row_sites = obj.site[obj.asd.detach().cpu().numpy()]
    pair = torch.as_tensor((row_sites != 'A')[:, None], dtype=torch.float32, device=device)
    pair = pair.expand_as(obj.w)
    original = obj.original_conditional_loss(pair)
    original_w = obj.w
    obj.w = (original_w * torch.where(obj.site_equal, 2.0, 0.0)).detach()
    scaled = obj.original_conditional_loss(pair)
    obj.w = original_w
    site, site_losses = obj.site_balanced_conditional_loss(pair)
    assert torch.allclose(original, scaled, atol=1e-7, rtol=1e-7)
    assert len(site_losses) == 2 and not torch.allclose(original, site, atol=1e-5, rtol=1e-5)
    gradient_site_losses = objectives['J1'].last_site_losses
    assert all(v.requires_grad for v in gradient_site_losses.values())
    assert abs(obj.site_weights['A'] - np.sqrt(10) / (np.sqrt(10) + np.sqrt(9))) < 1e-12

    singleton_y = np.asarray([0, 1, 0, 1, 0, 1])
    singleton_meta = {'age': np.full(6, 18.0), 'sex': np.full(6, 1.0),
                      'site': np.asarray(['A', 'A', 'B', 'B', 'C', 'D'])}
    singleton_prepared = prepare(singleton_y, singleton_meta, np.arange(6), 911, 0, CFG)
    singleton_obj = PCFRA_E18_Objective(singleton_prepared, 'J1', device)
    singleton_logits = torch.randn((6, 2), device=device, requires_grad=True)
    singleton_loss = singleton_obj.forward(singleton_logits, torch.randn((6, 16), device=device))
    assert singleton_obj.static_info['effective_sites'] == ['A', 'B'] and bool(torch.isfinite(singleton_loss))

    params_before = sum(p.numel() for p in torch.nn.Linear(16, 2).parameters())
    assert params_before == 34
    return {'label_encoding_and_global_direction': True, 'missing_age_and_unknown_sex': True,
            'site_mask_and_sqrt_weights': True, 'one_class_sites_skipped_but_global_pairs_kept': True,
            'j1_no_prototype': True, 'prototype_teacher_stop_gradient': True,
            'j2_j3_same_D_T_mu_C': True, 'j3_j4_margin_identical': True,
            'margin_bounds_and_site_means': True, 'site_aggregation_not_cancelled_by_anchor_norm': True,
            'all_objectives_have_logit_gradients': True,
            'site_losses_have_gradients': all(v.requires_grad for v in gradient_site_losses.values()),
            'synthetic_site_weights': obj.site_weights, 'synthetic_losses': outcomes,
            'no_new_parameters_in_objective': True}


def preflight_one(seed, fold, audit_info):
    l = Loaded(seed, fold, audit_info)
    try:
        old_module = historical_prototype_module()
        hist_cfg = json.loads(read_bytes(HIST / 'config.json'))
        historical_prepared = old_module.prepare(l.ctx['labels'], l.meta, l.sp['train'], seed, fold, hist_cfg)
        equal_prepared(historical_prepared, l.prepared)
        outside = np.setdiff1d(np.arange(871), l.sp['train'])
        altered_y = l.ctx['labels'].copy()
        altered_meta = {k: v.copy() for k, v in l.meta.items()}
        altered_y[outside] = 1 - altered_y[outside]
        altered_meta['age'][outside] = 99
        altered_meta['sex'][outside] = -1
        altered_meta['site'][outside] = 'HELDOUT_CHANGED'
        altered_prepared = prepare(altered_y, altered_meta, l.sp['train'], seed, fold, CFG)
        equal_prepared(l.prepared, altered_prepared)

        l.reset()
        l.core.train()
        l.core.zero_grad(set_to_none=True)
        model_calls = {'count': 0}
        handle = l.core.register_forward_hook(
            lambda module, args, output: model_calls.__setitem__('count', model_calls['count'] + 1))
        try:
            logits, graph = l.forward()
        finally:
            handle.remove()
        assert model_calls['count'] == 1
        assert tuple(logits.shape) == (871, 2) and tuple(l.z.shape) == (871, 16)
        native = F.cross_entropy(logits[l.sp['train']], l.y[l.sp['train']]) + graph
        group_results = {}
        objectives = {}
        for group in GROUPS:
            objective = PCFRA_E18_Objective(l.prepared, group, l.opt.device)
            auxiliary = objective(logits, l.z)
            weight, scale = gradient_budget(native, auxiliary, l.core.joint_embed, 9)
            g_logits = torch.autograd.grad(auxiliary, logits, retain_graph=True)[0]
            g_joint = torch.autograd.grad(auxiliary, l.core.joint_embed, retain_graph=True)[0]
            assert bool(torch.isfinite(auxiliary)) and bool(torch.isfinite(g_logits).all())
            assert float(g_logits.norm()) > 0 and float(g_joint.norm()) > 1e-12
            assert float(weight) > 0 and not weight.requires_grad
            if group == 'J1':
                assert objective.prototype_calls == 0
            else:
                assert objective.prototype_calls == 1
            group_results[group] = {'loss': float(auxiliary.detach()), **scale,
                                    'objective': dict(objective.last_info)}
            objectives[group] = objective

        site_loss_gradients = {}
        for group in ('J1', 'J4'):
            site_loss_gradients[group] = {}
            for site_name, site_loss in objectives[group].last_site_losses.items():
                grad_site = torch.autograd.grad(site_loss, logits, retain_graph=True)[0]
                grad_norm = float(grad_site.norm())
                assert site_loss.requires_grad and np.isfinite(grad_norm) and grad_norm > 0
                site_loss_gradients[group][site_name] = {
                    'requires_grad': bool(site_loss.requires_grad), 'logit_grad_norm': grad_norm,
                }

        # Every group receives the same captured logits/z from the one native forward.
        j2, j3, j4 = [objectives[g].last_tensors for g in ('J2', 'J3', 'J4')]
        assert torch.equal(j2['D'], j3['D']) and torch.equal(j2['temperature'], j3['temperature'])
        assert j2['mu_by_site'] == j3['mu_by_site'] and torch.equal(j2['C'], j3['C'])
        assert torch.equal(j3['margin'], j4['margin'])
        assert objectives['J1'].prototype_calls == 0
        real_logits = logits.detach().clone()
        real_z = l.z.detach().clone()
        real_logits[outside] = 10000
        real_z[outside] = -10000
        heldout = []
        # Auxiliary objectives use only outer-train rows and metadata.
        for group in GROUPS:
            first_logits = logits.detach().clone().requires_grad_(True)
            second_logits = real_logits.detach().clone().requires_grad_(True)
            first = PCFRA_E18_Objective(l.prepared, group, l.opt.device)(first_logits, l.z.detach())
            second = PCFRA_E18_Objective(altered_prepared, group, l.opt.device)(second_logits, real_z)
            assert torch.allclose(first, second, atol=1e-7, rtol=1e-7)
            heldout.append(float((first - second).abs()))

        hist_obj = old_module.PrototypeObjective(historical_prepared, 'D4', hist_cfg, l.opt.device)
        _, hist_detail = hist_obj(l.z)
        pcfra_obj = PCFRA_E18_Objective(l.prepared, 'J3', l.opt.device)
        evidence = pcfra_obj.original_d4_prototype_evidence(l.z)
        hist_evidence = hist_detail['logits'][:, 0] - hist_detail['logits'][:, 1]
        assert torch.allclose(evidence, hist_evidence.detach(), atol=1e-6, rtol=1e-6)
        assert not evidence.requires_grad and not hist_detail['mu'].requires_grad

        # Preprocessing is train-only; native model inputs remain unchanged.
        assert array_hash(l.ctx['ph_data']) == array_hash(l.ctx['ph_data'].copy())
        edge = synthetic_checks(l.opt.device)
        out = {
            'passed': True, 'seed': seed, 'fold': fold, 'historical_prepare_exact': True,
            'historical_d4_prototype_evidence_max_abs_error':
                float((evidence - hist_evidence.detach()).abs().max()),
            'model_forward_count': model_calls['count'], 'logits_shape': list(logits.shape), 'image_embedding_shape': list(l.z.shape),
            'gradient_checks': group_results, 'heldout_perturbation_invariant': True,
            'per_site_loss_gradients': site_loss_gradients,
            'j2_j3_D_T_C_equal': True, 'j3_j4_margin_equal': True,
            'site_balancing_effective': objectives['J1'].site_weights != {},
            'original_ph_input_hash': array_hash(l.ctx['ph_data']),
            'synthetic': edge, 'physical_gpu': int(os.environ.get('CUDA_VISIBLE_DEVICES', '0')),
        }
        log('preflight_passed', seed=seed, fold=fold)
        return out
    finally:
        l.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('audit', 'preflight', 'train'))
    parser.add_argument('--group', choices=GROUPS)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--fold', type=int)
    args = parser.parse_args()
    if args.action == 'audit':
        audit()
        return
    a = guard(deep=args.action != 'train')
    if args.action == 'preflight':
        if args.seed != 911 or args.fold != 0:
            raise ValueError('Only the single authorized preflight anchor 911/fold0 is allowed')
        out = preflight_one(args.seed, args.fold, a)
        with task_lock(990):
            a = json.loads(read_bytes(BASE / 'audit.json'))
            a['preflight'][f'{args.seed}/{args.fold}'] = out
            atomic_json(a, BASE / 'audit.json')
        return
    if args.group not in GROUPS or args.seed != 911 or args.fold not in range(10):
        raise ValueError('Formal task outside the fixed J1-J4 seed911 folds0-9 manifest')
    slot = GROUPS.index(args.group) * 10 + args.fold
    with task_lock(slot):
        train_one(args.group, args.seed, args.fold, a)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        try:
            log('worker_error', message=str(exc), group=locals().get('args', None).group
                if 'args' in locals() else None, traceback=__import__('traceback').format_exc())
        finally:
            raise
