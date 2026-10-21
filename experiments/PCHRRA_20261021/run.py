"""Audited PCHRRA train, preflight, and historical-reference utilities."""
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
DRPA = ROOT / 'experiments/DRPA_20261014'
D4 = ROOT / 'experiments/PCDCA_20261019'
G2H = ROOT / 'experiments/PCFRA_20261020'
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from opt import OptInit
from model.mm_gtunets import MM_GTUNets
from model.rp_graph import create_reward_penalty_graph
from utils.tools import EarlyStopping
from objective import PCHRRAObjective, array_hash, prepare, _describe

CFG = json.loads((BASE / 'config.json').read_text())
GROUPS = ('H1', 'H2', 'H3', 'H4')
SOURCE = ('opt.py', 'train_mm_gtunets.py', 'model/mm_gtunets.py', 'model/gtunet.py',
          'model/rp_graph.py', 'utils/mydataloader.py', 'utils/tools.py', 'utils/metrics.py')
CODE = ('config.json', 'objective.py', 'run.py', 'launch.py', 'analyze.py')
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
            time.sleep(0.25 * (attempt + 1))


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
    for key in sorted(state):
        value = state[key]
        h.update(key.encode())
        h.update(array_hash(value.detach().cpu().numpy() if torch.is_tensor(value) else value).encode())
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
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'cpu': torch.get_rng_state().clone(), 'cuda': torch.cuda.get_rng_state(0).clone(),
            'backend': {'cudnn_deterministic': torch.backends.cudnn.deterministic,
                        'cudnn_benchmark': torch.backends.cudnn.benchmark,
                        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32}}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    torch.cuda.set_rng_state(state['cuda'], 0)
    backend = state['backend']
    torch.backends.cudnn.deterministic = backend['cudnn_deterministic']
    torch.backends.cudnn.benchmark = backend['cudnn_benchmark']
    torch.backends.cuda.matmul.allow_tf32 = backend['matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32 = backend['cudnn_allow_tf32']


def rng_hash(state):
    return {'python': hashlib.sha256(repr(state['python']).encode()).hexdigest(),
            'numpy': hashlib.sha256(repr(state['numpy'][0]).encode() + state['numpy'][1].tobytes()
                                    + repr(state['numpy'][2:]).encode()).hexdigest(),
            'cpu': array_hash(state['cpu'].numpy()), 'cuda': array_hash(state['cuda'].numpy())}


def atomic_bytes(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with temp.open('xb') as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)
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
    out = io.BytesIO()
    torch.save({'payload': value, 'content_sha256': payload_hash(value)}, out)
    atomic_bytes(out.getvalue(), path)


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
    if os.environ.get('PCHRRA_LOG_STDOUT') == '1':
        print(line, flush=True)
        return
    with (BASE / 'run.log').open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.write(line + '\n')
        handle.flush()
        fcntl.flock(handle, fcntl.LOCK_UN)


@contextmanager
def task_lock(slot):
    with (BASE / 'run.py').open('r+') as handle:
        fcntl.lockf(handle, fcntl.LOCK_EX | fcntl.LOCK_NB, 1, 1000 + slot)
        try:
            yield
        finally:
            fcntl.lockf(handle, fcntl.LOCK_UN, 1, 1000 + slot)


def code_hash():
    return digest({name: sha_file(BASE / name) for name in CODE})


def anchor(seed, fold):
    return DRPA / 'artifacts' / f'seed_{seed}' / f'fold_{fold}'


def result_path(group, seed, fold):
    return BASE / 'runs' / group / f's{seed}_f{fold}.pt'


def metadata(ctx):
    with (ROOT / CFG['data_folder'] / 'Phenotypic_V1_0b_preprocessed1.csv').open() as handle:
        rows = {row['SUB_ID']: row for row in csv.DictReader(handle)}
    ordered = [rows[subject_id] for subject_id in ctx['ids']]
    labels = np.asarray([int(row['DX_GROUP']) - 1 for row in ordered])
    assert np.array_equal(labels, ctx['labels']) and np.array_equal(np.bincount(labels), [403, 468])

    def number(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return float('nan')

    age = np.asarray([number(row['AGE_AT_SCAN']) for row in ordered])
    sex = np.asarray([number(row['SEX']) for row in ordered])
    site = np.asarray([row['SITE_ID'] for row in ordered])
    assert np.array_equal(site, ctx['site_names']) and len(set(site)) == 20
    assert np.allclose(age, ctx['ph_data'][:, 2], rtol=0, atol=2e-6)
    assert np.array_equal(sex - 1, ctx['ph_data'][:, 1])
    assert [(int(np.sum((labels == k) & (sex == 2))), int(np.sum((labels == k) & (sex == 1))))
            for k in (0, 1)] == [(54, 349), (90, 378)]
    return {'age': age, 'sex': sex, 'site': site}


def binary_metrics(y, probability, prediction):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    pred = np.asarray(prediction, dtype=np.int64)
    truth = (y == 0).astype(np.int8)
    return {'ACC': float(accuracy_score(y, pred)),
            'SEN': float(recall_score(y, pred, pos_label=0)),
            'SPE': float(recall_score(y, pred, pos_label=1)),
            'AUC': float(roc_auc_score(truth, p))}


def _reference_rows(group, ctx, splits):
    root = D4 / 'runs' / group
    rows, hashes = [], {}
    for fold in range(10):
        path = root / f's911_f{fold}.pt'
        value = load_torch(path, artifact=True)
        assert (value['group'], value['seed'], value['fold']) == (group, 911, fold)
        split = load_torch(anchor(911, fold) / 'splits.pt')
        idx, y = value['oof']['indices'], value['oof']['labels']
        logits = np.asarray(value['oof']['logits'])
        assert np.array_equal(idx, split['test']) and np.array_equal(y, ctx['labels'][idx])
        assert value['oof']['ids'] == [ctx['ids'][int(i)] for i in idx]
        probability = torch.softmax(torch.as_tensor(logits, dtype=torch.float64), dim=1)[:, 0].numpy()
        prediction = logits.argmax(1)
        measured = binary_metrics(y, probability, prediction)
        assert max(abs(measured[k] - value['metrics'][k]) for k in measured) < 1e-12
        rows.append({'fold': fold, 'indices': idx, 'labels': y, 'logits': logits,
                     'probability': probability, 'prediction': prediction, 'metrics': measured})
        hashes[str(path)] = sha_file(path)
    return rows, hashes


def historical_results(ctx):
    rows = {}
    hashes = {}
    for name, folder, group in (('historical_D4', D4 / 'runs/D4', 'D4'),
                                ('historical_G2', G2H / 'runs/G2', 'G2')):
        current = []
        for fold in range(10):
            path = folder / f's911_f{fold}.pt'
            value = load_torch(path, artifact=True)
            assert (value['group'], value['seed'], value['fold']) == (group, 911, fold)
            split = load_torch(anchor(911, fold) / 'splits.pt')
            idx, y = value['oof']['indices'], value['oof']['labels']
            logits = np.asarray(value['oof']['logits'])
            assert np.array_equal(idx, split['test']) and np.array_equal(y, ctx['labels'][idx])
            assert value['oof']['ids'] == [ctx['ids'][int(i)] for i in idx]
            probability = torch.softmax(torch.as_tensor(logits, dtype=torch.float64), dim=1)[:, 0].numpy()
            prediction = logits.argmax(1)
            measured = binary_metrics(y, probability, prediction)
            assert max(abs(measured[k] - value['metrics'][k]) for k in measured) < 1e-12
            current.append({'fold': fold, 'indices': idx, 'labels': y, 'logits': logits,
                            'probability': probability, 'prediction': prediction, 'metrics': measured})
            hashes[str(path)] = sha_file(path)
        rows[name] = current
    return rows, hashes


def historical_prototype_module():
    path = D4 / 'prototype.py'
    spec = importlib.util.spec_from_file_location('pchrra_historical_prototype', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def handoff_prefix_ok(saved):
    data = read_bytes(ROOT / 'handoff.md')
    n = int(saved['handoff_before_bytes'])
    return len(data) >= n and hashlib.sha256(data[:n]).hexdigest() == saved['handoff_before_sha256']


def guard(saved=None, check_protected=True):
    saved = saved or json.loads(read_bytes(BASE / 'audit.json'))
    assert saved['code_sha256'] == code_hash(), 'PCHRRA code/config fingerprint mismatch'
    if check_protected:
        for path, expected in saved['protected_sha256'].items():
            if path == str(ROOT / 'handoff.md') and saved.get('handoff_appended'):
                continue
            if sha_file(path) != expected:
                raise RuntimeError(f'Protected historical input changed: {path}')
    for path, expected in saved['source_sha256'].items():
        assert sha_file(ROOT / path) == expected
    if saved.get('handoff_appended'):
        assert handoff_prefix_ok(saved), 'Original handoff prefix changed after append'
    else:
        assert sha_file(ROOT / 'handoff.md') == saved['handoff_before_sha256'], 'handoff changed before final append'
    return saved


def audit():
    audit_path = BASE / 'audit.json'
    if audit_path.exists():
        saved = json.loads(read_bytes(audit_path))
        current = code_hash()
        if saved['code_sha256'] != current:
            if saved.get('formal_authorized') or (BASE / 'manifest.json').exists():
                raise RuntimeError('Refusing source/config changes after manifest freeze')
            for path, expected in saved['protected_sha256'].items():
                assert sha_file(path) == expected, f'Protected file changed: {path}'
            for path, expected in saved['source_sha256'].items():
                assert sha_file(ROOT / path) == expected, f'Protected source changed: {path}'
            previous = saved['code_sha256']
            saved['code_sha256'] = current
            saved.setdefault('code_revisions', []).append({'previous': previous, 'updated': current,
                                                            'reason': 'pre-manifest source correction', 'utc': utc()})
            prior_preflight = saved.get('preflight', {})
            prior_real = prior_preflight.get('real_forward', {})
            saved['preflight'] = {
                'math': {},
                'real_forward': {
                    'attempt_count': prior_real.get('attempt_count', 0),
                    'attempts': prior_real.get('attempts', []),
                },
            }
            atomic_json(saved, audit_path)
        guard(saved, check_protected=False)
        return saved

    assert list(CFG['groups']) == list(GROUPS) and CFG['seeds'] == [911] and CFG['folds'] == 10
    assert len([(g, s, f) for g in GROUPS for s in CFG['seeds'] for f in range(CFG['folds'])]) == 40
    assert not ({'D0', 'D4', 'E0', 'E1', 'F0', 'F1', 'G1', 'G2', 'G3', 'G4'} & set(CFG['groups']))
    ctx = load_torch(DRPA / 'artifacts/context.pt')
    meta = metadata(ctx)
    assert len(ctx['ids']) == len(set(ctx['ids'])) == 871
    assert np.array_equal(np.genfromtxt(ROOT / 'data/ABIDE_aal/id.txt', dtype=str).astype(str),
                          np.asarray(ctx['ids']).astype(str))

    protected_paths = set(ROOT / name for name in SOURCE)
    protected_paths.update(ROOT.glob('model/*.py'))
    protected_paths.update(ROOT.glob('utils/*.py'))
    protected_paths.update([ROOT / 'prompt', ROOT / 'handoff.md',
                            ROOT / CFG['data_folder'] / 'id.txt',
                            ROOT / CFG['data_folder'] / 'Phenotypic_V1_0b_preprocessed1.csv',
                            ROOT / 'data/ABIDE_aal/id.txt', DRPA / 'artifacts/context.pt'])
    protected_paths.update(p for p in (ROOT / CFG['data_folder'] / 'save_tensor').glob('*') if p.is_file())
    for folder in (D4, G2H):
        protected_paths.update(p for p in folder.rglob('*') if p.is_file() and
                               (p.suffix in ('.py', '.json', '.md') or '/runs/' in p.as_posix()))
    for path in (DRPA / 'artifacts').rglob('*'):
        if path.is_file():
            protected_paths.add(path)
    protected = {str(p): sha_file(p) for p in sorted(protected_paths, key=str) if p.is_file()}

    raw = torch.as_tensor(load_torch(ROOT / CFG['data_folder'] / 'save_tensor/up_triangles.pt'))
    assert tuple(raw.shape) == (871, 6670) and bool(torch.isfinite(raw).all())
    anchors = {}
    historical_test_count = np.zeros(871, dtype=np.int64)
    pcdca_cfg = json.loads(read_bytes(D4 / 'config.json'))
    hist_module = historical_prototype_module()
    for fold in range(10):
        a = anchor(911, fold)
        split = load_torch(a / 'splits.pt')
        start = load_torch(a / 'train_start.pt')
        fp = json.loads(read_bytes(a / 'fingerprints.json'))
        all_indices = np.concatenate([split[k] for k in SUBSETS])
        assert len(all_indices) == 871 and np.array_equal(np.sort(all_indices), np.arange(871))
        historical_test_count[split['test']] += 1
        for subset in SUBSETS:
            assert array_hash(split[subset]) == fp['split_fingerprints'][subset]
        selected = load_torch(a.parent / 'selected_features.pt')
        support = np.load(io.BytesIO(read_bytes(a.parent / 'rfe_support.npy')))
        assert tuple(selected.shape) == (871, 500) and bool(torch.isfinite(selected).all())
        assert support.shape == (500,) and len(np.unique(support)) == 500
        assert torch.equal(raw[:, support].to(selected.dtype), selected)
        assert array_hash(selected.numpy()) == fp['features_fingerprint']
        assert core_hash(start['core']) == fp['train_start_core_fingerprint']
        assert rng_hash(start['rng']) == fp['classifier_start_rng']
        vae = a.parent / f'vae/fold{fold}_pretrain.pth'
        assert sha_file(vae) == start['vae_sha256'] == fp['vae_sha256']
        ours = prepare(ctx['labels'], meta, split['train'], 911, fold, CFG)
        theirs = hist_module.prepare(ctx['labels'], meta, split['train'], 911, fold, pcdca_cfg)
        assert digest(ours['info']) == digest(theirs['info']), f'Historical D4 prepare mismatch fold {fold}'
        for key in ('folds', 'halves', 'permutation', 'y', 'age', 'sex', 'site'):
            assert np.array_equal(ours[key], theirs[key]), f'Historical D4 {key} mismatch fold {fold}'
        input_paths = [a / 'splits.pt', a / 'train_start.pt', a / 'fingerprints.json',
                       a.parent / 'selected_features.pt', a.parent / 'rfe_support.npy', vae]
        anchors[str(fold)] = {'input_sha256': {str(p): sha_file(p) for p in input_paths},
                              'pairing': fp, 'prototype': ours['info'], 'physical_gpu': fold % 2}
    assert np.all(historical_test_count == 1)
    historical, historical_hashes = historical_results(ctx)
    protected.update(historical_hashes)

    def summarize(rows):
        return {metric: {'mean': float(np.mean([r['metrics'][metric] for r in rows]) * 100),
                         'sample_sd': float(np.std([r['metrics'][metric] for r in rows], ddof=1) * 100)}
                for metric in ('ACC', 'SEN', 'SPE', 'AUC')}

    history_stats = {name: summarize(values) for name, values in historical.items()}
    expected_d4 = {'ACC': 82.5588, 'SEN': 79.9268, 'SPE': 84.8427, 'AUC': 88.6897}
    for key, expected in expected_d4.items():
        assert abs(history_stats['historical_D4'][key]['mean'] - expected) < 0.00005

    out = {
        'experiment': CFG['experiment'], 'started_utc': utc(), 'code_sha256': code_hash(),
        'config_sha256': sha_file(BASE / 'config.json'), 'prompt_sha256': sha_file(ROOT / 'prompt'),
        'source_sha256': {name: sha_file(ROOT / name) for name in SOURCE},
        'protected_sha256': protected,
        'handoff_before_sha256': sha_file(ROOT / 'handoff.md'),
        'handoff_before_bytes': len(read_bytes(ROOT / 'handoff.md')),
        'context_sha256': sha_file(DRPA / 'artifacts/context.pt'), 'anchors': anchors,
        'data': {'n': 871, 'class_counts': [403, 468], 'sites': 20, 'image_shape': [871, 500]},
        'historical_readonly': {
            name: {'source': str(D4 / 'runs/D4') if name == 'historical_D4' else str(G2H / 'runs/G2'),
                   'fold_metrics': [{'fold': r['fold'], 'metrics_percent':
                                     {k: r['metrics'][k] * 100 for k in ('ACC', 'SEN', 'SPE', 'AUC')}}
                                    for r in values],
                   'mean_sample_sd_percent': history_stats[name],
                   'checkpoint_sha256': {p: h for p, h in historical_hashes.items()
                                         if f'/runs/{"D4" if name == "historical_D4" else "G2"}/' in p},
                   'retrained': False}
            for name, values in historical.items()},
        'environment': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__},
        'preflight': {'math': {'passed': False}, 'real_forward': {}},
        'formal_authorized': False, 'execution': {'events': [], 'failures': [], 'resources': []},
        'task_boundary': {'groups': list(GROUPS), 'seeds': [911], 'folds': 10, 'tasks': 40},
        'literature': {'MM_GTUNets': 'https://arxiv.org/html/2406.14455v3',
                       'repository': 'https://github.com/tangaii/MM'},
    }
    atomic_json(out, audit_path)
    log('audit_passed', anchors=10, historical_d4_folds=10, historical_g2_folds=10,
        protected_files=len(protected), code_sha256=out['code_sha256'])
    return out


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
        self.ctx = load_torch(DRPA / 'artifacts/context.pt')
        self.meta = metadata(self.ctx)
        self.sp = load_torch(anchor(seed, fold) / 'splits.pt')
        self.start = load_torch(anchor(seed, fold) / 'train_start.pt')
        self.anchor_info = audit_info['anchors'][str(fold)]
        for path, expected in self.anchor_info['input_sha256'].items():
            assert sha_file(path) == expected, f'Anchor changed: {path}'
        old_argv = sys.argv
        try:
            sys.argv = ['native_options']
            opt = OptInit(model='MM_GTUNets', dataset='ABIDE', atlas='aal').args
        finally:
            sys.argv = old_argv
        opt.seed = seed
        opt.device = torch.device('cuda:0')
        opt.data_folder = str(ROOT / CFG['data_folder'])
        opt.ckpt_path = str(anchor(seed, fold).parent / 'vae')
        opt.epoch, opt.early_stop = CFG['epochs'], CFG['early_stop']
        opt.lr, opt.wd = CFG['lr'], CFG['weight_decay']
        opt.log_save = False
        opt.model_save = False
        self.opt = opt
        self.core = MM_GTUNets(opt, fold).to(opt.device)
        self.core.vae_save_path = str(anchor(seed, fold).parent / f'vae/fold{fold}_pretrain.pth')
        import_core(self.core, self.start['core'])
        assert core_hash(export_core(self.core)) == self.anchor_info['pairing']['train_start_core_fingerprint']
        self.x_cpu = load_torch(anchor(seed, fold).parent / 'selected_features.pt').float()
        self.ph_cpu = torch.from_numpy(self.ctx['ph_data']).float()
        affinity = create_reward_penalty_graph(self.ctx['ph_dict'], self.ctx['labels'],
                                                *[self.sp[key] for key in SUBSETS], opt)
        self.affinity_cpu = torch.from_numpy(affinity).float()
        self.x, self.ph, self.affinity = [v.to(opt.device) for v in
                                          (self.x_cpu, self.ph_cpu, self.affinity_cpu)]
        self.y = torch.as_tensor(self.ctx['labels'], dtype=torch.long, device=opt.device)
        self.prepared = prepare(self.ctx['labels'], self.meta, self.sp['train'], seed, fold, CFG)
        assert digest(self.prepared['info']) == digest(self.anchor_info['prototype'])
        self.z_img = None
        self.img_hook = self.core.img_unet.register_forward_hook(self._capture_img)
        self.input_sha256 = payload_hash({'x': self.x_cpu, 'ph': self.ph_cpu,
                                          'affinity': self.affinity_cpu,
                                          'labels': self.ctx['labels'], 'ids': self.ctx['ids']})

    def _capture_img(self, module, args, output):
        if tuple(output.shape) != (871, 16):
            raise AssertionError(f'Imaging embedding shape {tuple(output.shape)}')
        self.z_img = output

    def optimizer(self):
        params = list(self.core.parameters())
        assert not any(param is self.core.rp_attention.weights for param in params)
        return torch.optim.Adam(params, lr=CFG['lr'], weight_decay=CFG['weight_decay'])

    def reset(self):
        import_core(self.core, self.start['core'])
        for parameter in self.core.parameters():
            parameter.grad = None
        restore_rng(self.start['rng'])
        torch.use_deterministic_algorithms(False)

    def forward(self):
        return self.core(self.x, self.ph, self.affinity)

    def close(self):
        self.img_hook.remove()
        self.__dict__.clear()
        torch.cuda.empty_cache()


def gradient_budget(native, auxiliary, joint, epoch):
    g_native = torch.autograd.grad(native, joint, retain_graph=True)[0]
    g_aux = torch.autograd.grad(auxiliary, joint, retain_graph=True)[0]
    n, a = g_native.detach().norm(), g_aux.detach().norm()
    if not bool(torch.isfinite(n)) or not bool(torch.isfinite(a)):
        raise FloatingPointError('Nonfinite fusion gradient')
    if float(n) <= 1e-12 or (epoch >= CFG['warmup_last_epoch'] and float(a) <= 1e-12):
        raise FloatingPointError('Zero native or post-warmup auxiliary joint_embed gradient')
    cosine = (g_native.detach() * g_aux.detach()).sum() / (n * a).clamp_min(CFG['epsilon'])
    with torch.no_grad():
        lam = (CFG['auxiliary_gradient_ratio'] * n / a.clamp_min(CFG['epsilon'])).clamp(
            0.0, CFG['lambda_cap'])
        warmup = min(max(epoch, 0) / CFG['warmup_last_epoch'], 1.0)
        lam = (lam * warmup).detach()
    ratio = float(lam * a / n.clamp_min(CFG['epsilon']))
    return lam, {'native_joint_grad_norm': float(n), 'aux_joint_grad_norm': float(a),
                 'lambda_eff': float(lam), 'warmup': float(warmup),
                 'weighted_joint_grad_ratio': ratio, 'joint_grad_cosine': float(cosine)}


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
    assert value['best_core_sha256'] == core_hash(value['best_core'])
    assert value['new_trainable_parameters'] == value['rfe_fit_count'] == value['vae_train_count'] == 0
    assert value['formal_test_grad_enabled'] and 'PPU' in value['device'].upper()
    assert value['forward_calls'] == {'train': value['epochs_run'], 'val': value['epochs_run'], 'test': 1}
    split = load_torch(anchor(seed, fold) / 'splits.pt')
    assert np.array_equal(value['oof']['indices'], split['test'])
    assert np.array_equal(value['oof']['prediction'], np.asarray(value['oof']['logits']).argmax(1))
    probability = np.asarray(value['oof']['probability'])
    assert np.isfinite(probability).all() and np.all((probability >= 0.0) & (probability <= 1.0))
    expected = torch.softmax(torch.as_tensor(value['oof']['logits'], dtype=torch.float64), dim=1)[:, 0].numpy()
    assert np.allclose(probability, expected, atol=1e-7, rtol=1e-6)
    measured = binary_metrics(value['oof']['labels'], probability, value['oof']['prediction'])
    assert max(abs(measured[k] - value['metrics'][k]) for k in measured) < 1e-12
    assert value['best_epoch'] < value['epochs_run'] <= CFG['epochs']
    return value


def train_one(group, seed, fold, audit_info):
    if valid_result(group, seed, fold, audit_info) is not None:
        log('skip_valid', group=group, seed=seed, fold=fold)
        return
    if not audit_info.get('formal_authorized'):
        raise RuntimeError('Formal training requires CPU tests, real preflight, and frozen manifest')
    loaded = Loaded(seed, fold, audit_info)
    forward_hook = None
    try:
        objective = PCHRRAObjective(loaded.prepared, group, loaded.opt.device, CFG)
        assert np.array_equal(loaded.prepared['train'], loaded.sp['train'])
        optimizer = loaded.optimizer()
        stopper = EarlyStopping(patience=CFG['early_stop'], save=False)
        loaded.reset()
        assert rng_hash(capture_rng()) == loaded.anchor_info['pairing']['classifier_start_rng']
        assert not torch.are_deterministic_algorithms_enabled()
        state_names = set(loaded.start['core']['state'])
        trainable_names = {name for name, param in loaded.core.named_parameters() if param.requires_grad}
        new_trainable = trainable_names - state_names
        assert not new_trainable
        batches_start = int(loaded.core.clf[2].num_batches_tracked)
        call_counts = {'train': 0, 'val': 0, 'test': 0}
        phase = {'value': 'train'}

        def count_forward(module, args, output):
            call_counts[phase['value']] += 1

        forward_hook = loaded.core.register_forward_hook(count_forward)
        epochs, best_core, best_loss = [], None, float('inf')
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started, started_utc = time.perf_counter(), utc()
        for epoch in range(CFG['epochs']):
            loaded.core.train()
            optimizer.zero_grad()
            phase['value'] = 'train'
            logits, graph_loss = loaded.forward()
            if tuple(logits.shape) != (871, 2) or tuple(loaded.z_img.shape) != (871, 16):
                raise AssertionError('Native model output shape changed')
            native_ce = F.cross_entropy(logits[loaded.sp['train']], loaded.y[loaded.sp['train']])
            native = native_ce + graph_loss
            auxiliary = objective.forward(logits, loaded.z_img)
            lam, gradient = gradient_budget(native, auxiliary, loaded.core.joint_embed, epoch)
            total = native + lam * auxiliary
            total.backward()
            optimizer.step()

            loaded.core.eval()
            phase['value'] = 'val'
            with torch.no_grad():
                val_logits, val_graph = loaded.forward()
                val_loss = F.cross_entropy(val_logits[loaded.sp['val']], loaded.y[loaded.sp['val']]) + val_graph
            val_value = float(val_loss)
            epoch_row = {'epoch': epoch, 'train_native_loss': float(native.detach()),
                         'train_native_ce': float(native_ce.detach()), 'train_graph_loss': float(graph_loss.detach()),
                         'auxiliary_loss': float(auxiliary.detach()), 'total_loss': float(total.detach()),
                         'val_loss': val_value, **gradient, 'objective': objective.last_info}
            epochs.append(epoch_row)
            if val_value < best_loss:
                best_loss = val_value
                best_core = export_core(loaded.core)
                best_epoch = epoch
            stopper(val_loss, loaded.core)
            if epoch % 25 == 0 or stopper.early_stop:
                log('progress', group=group, seed=seed, fold=fold, epoch=epoch,
                    best_epoch=best_epoch, val_loss=val_value, lambda_eff=gradient['lambda_eff'],
                    q_deviation=objective.last_info.get('q_abs_deviation_mean'))
            if stopper.early_stop:
                break

        end_rng = capture_rng()
        end_rng_fingerprint = rng_hash(end_rng)
        import_core(loaded.core, best_core)
        restore_rng(end_rng)
        loaded.core.eval()
        phase['value'] = 'test'
        test_start_rng = rng_hash(capture_rng())
        with torch.enable_grad():
            test_logits, test_graph = loaded.forward()
        test_end_rng = rng_hash(capture_rng())
        idx = np.asarray(loaded.sp['test'], dtype=np.int64)
        logits_test = test_logits.index_select(0, torch.as_tensor(idx, device=loaded.opt.device)).detach().cpu().numpy()
        y_test = loaded.ctx['labels'][idx]
        probability = torch.softmax(torch.as_tensor(logits_test, dtype=torch.float64), dim=1)[:, 0].numpy()
        prediction = logits_test.argmax(1)
        metrics = binary_metrics(y_test, probability, prediction)
        torch.cuda.synchronize()
        finished_utc = utc()
        result = {
            'state': 'complete', 'group': group, 'seed': seed, 'fold': fold,
            'physical_gpu': int(os.environ.get('CUDA_VISIBLE_DEVICES', fold % 2)),
            'device': torch.cuda.get_device_name(0), 'code_sha256': audit_info['code_sha256'],
            'config_sha256': audit_info['config_sha256'], 'input_sha256': loaded.anchor_info['input_sha256'],
            'input_digest': loaded.input_sha256, 'pairing': loaded.anchor_info['pairing'],
            'prototype_preparation': loaded.prepared['info'], 'objective_static': objective.static_info,
            'objective_group': group, 'epochs': epochs, 'epochs_run': len(epochs),
            'best_epoch': int(best_epoch), 'best_val_loss': float(best_loss),
            'best_core': best_core, 'best_core_sha256': core_hash(best_core),
            'formal_start_rng_sha256': loaded.anchor_info['pairing']['classifier_start_rng'],
            'training_end_rng_sha256': end_rng_fingerprint,
            'test_start_rng_sha256': test_start_rng, 'test_end_rng_sha256': test_end_rng,
            'native_test_graph_loss': float(test_graph.detach()),
            'forward_calls': call_counts, 'test_forwards': call_counts['test'],
            'formal_test_grad_enabled': torch.is_grad_enabled(),
            'new_trainable_parameters': len(new_trainable), 'rfe_fit_count': 0, 'vae_train_count': 0,
            'bn_best_batches': int(best_core['state']['clf.2.num_batches_tracked']),
            'bn_start_batches': batches_start,
            'oof': {'indices': idx, 'ids': [loaded.ctx['ids'][int(i)] for i in idx],
                    'labels': y_test, 'logits': logits_test, 'probability': probability,
                    'prediction': prediction},
            'metrics': metrics,
            'training_started_utc': started_utc, 'finished_utc': finished_utc,
            'wall_seconds': time.perf_counter() - started,
            'peak_gpu_memory_mib': torch.cuda.max_memory_allocated() / (1024 ** 2),
            'gradient_target': 'joint_embed', 'forward_calls_semantics': 'one native training forward and one validation forward per epoch; one test forward',
        }
        assert call_counts == {'train': len(epochs), 'val': len(epochs), 'test': 1}
        assert result['bn_best_batches'] == batches_start + best_epoch + 1
        path = result_path(group, seed, fold)
        atomic_torch(result, path)
        assert valid_result(group, seed, fold, audit_info, required=True) is not None
        log('complete', group=group, seed=seed, fold=fold, epochs=len(epochs), best_epoch=best_epoch,
            seconds=result['wall_seconds'], physical_gpu=result['physical_gpu'], metrics=metrics,
            peak_gpu_memory_mib=result['peak_gpu_memory_mib'])
    finally:
        if forward_hook is not None:
            forward_hook.remove()
        loaded.close()


def _manual_prepared():
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1], dtype=np.int64)
    meta = {'age': np.asarray([10, 12, 16, 20, 11, 13, 17, 22, 14, 18, 24, 26], dtype=float),
            'sex': np.asarray([1, 1, 2, 1, 1, 2, 2, 1, 1, 2, 0, 1], dtype=float),
            'site': np.asarray(['A', 'A', 'A', 'A', 'A', 'A', 'A', 'A', 'B', 'B', 'B', 'B'])}
    return prepare(labels, meta, np.arange(len(labels)), 911, 0, CFG), meta


def math_test():
    prepared, _ = _manual_prepared()
    device = torch.device('cpu')
    n = len(prepared['train'])
    logits = torch.zeros((n, 2), dtype=torch.float64, requires_grad=True)
    h1 = PCHRRAObjective(prepared, 'H1', device, CFG)
    h2 = PCHRRAObjective(prepared, 'H2', device, CFG)
    h3 = PCHRRAObjective(prepared, 'H3', device, CFG)
    h4 = PCHRRAObjective(prepared, 'H4', device, CFG)
    good_score = torch.where(h1.y == 0, 2.0, -2.0).to(torch.float64)
    bad_score = -good_score
    good_delta = good_score[h1.asd, None] - good_score[None, h1.td]
    bad_delta = bad_score[h1.asd, None] - bad_score[None, h1.td]
    assert float(h1.global_ranking_loss(good_delta)) < float(h1.global_ranking_loss(bad_delta))
    site, true = h2.build_condition()
    site_asd = prepared['site'][h2.asd.cpu().numpy()]
    site_td = prepared['site'][h2.td.cpu().numpy()]
    assert torch.all(site[torch.as_tensor(site_asd[:, None] != site_td[None, :])] == 0)
    assert torch.all((site == 0) | (site == 1))
    assert torch.all((true >= 0) & (true <= 1))
    assert float(prepared['bandwidth']) >= 2.0
    # Site-only conditioning is independent of age and sex inputs.
    altered_meta = {'age': np.full(n, 40.0), 'sex': np.where(prepared['sex'] == 1, 2, 1),
                    'site': prepared['site'].copy()}
    altered = prepare(prepared['y'], altered_meta, np.arange(n), 911, 0, CFG)
    h1_altered = PCHRRAObjective(altered, 'H1', device, CFG)
    assert torch.equal(h1.site_condition, h1_altered.site_condition)

    score = torch.tensor([1.2, 0.8, 0.5, 1.1, -0.2, -0.5, -0.1, -0.7, 0.2, -0.3, 0.3, -0.4],
                         dtype=torch.float64)
    delta = score[h2.asd, None] - score[None, h2.td]
    base, difficulty = h2.hard_pair_weights(delta)
    multiplier = base / h2.true_condition.clamp_min(1e-30)
    supported = h2.true_condition > 0
    assert bool((multiplier[supported] >= 1.0).all() and (multiplier[supported] <= 1.25 + 1e-6).all())
    assert not base.requires_grad and not difficulty.requires_grad

    # Hand-check residual and condition evidence directions through the exact pair indexing.
    e_cond = torch.tensor([2., 1., 2.2, 1.8, 0., -0.2, -0.5, -0.3, 0.4, -0.2, 0.1, -0.4])
    e_global = torch.tensor([1., 1., 1.2, 1.8, 0., -0.2, -0.5, -0.3, 0.4, -0.2, 0.1, -0.4])
    R, E = h3.residual_evidence(e_cond, e_global)
    assert torch.isclose(R[0, 0], torch.tensor(1.0))
    assert torch.isclose(E[0, 0], torch.tensor(2.0))
    wrong_cond = e_cond.clone()
    wrong_cond[h3.asd[0]] = -2.0
    _, wrong_E = h3.residual_evidence(wrong_cond, e_global)
    B, _ = h3.hard_pair_weights(delta)
    _, c_pos, q_pos, info = h3.residual_reweight(B, R, E)
    _, c_wrong, _, _ = h3.residual_reweight(B, R, wrong_E, beta_residual=0.0)
    assert c_pos[0, 0] > c_wrong[0, 0]
    assert bool((q_pos >= 0.5 - 1e-6).all() and (q_pos <= 1.5 + 1e-6).all())
    assert abs(float((B * q_pos).sum() - B.sum())) < 1e-6
    _, _, q_const, _ = h3.residual_reweight(B, torch.zeros_like(B), torch.zeros_like(B))
    assert torch.allclose(q_const, torch.ones_like(q_const), atol=1e-7, rtol=0)
    _, _, q_beta0, _ = h3.residual_reweight(B, R, E, beta_residual=0.0)
    assert torch.allclose(q_beta0, torch.ones_like(q_beta0), atol=1e-7, rtol=0)

    # Alpha=0 exactly recovers the historical G2 condition weights and loss formula.
    B0, _ = h2.hard_pair_weights(delta, alpha=0.0)
    pair_loss = F.softplus(1.0 - delta)
    assert torch.equal(B0, h2.true_condition)
    old_g2 = 0.5 * h2.global_ranking_loss(delta) + 0.5 * h2.conditional_ranking_loss(pair_loss, h2.true_condition)
    h2_alpha0 = 0.5 * h2.global_ranking_loss(delta) + 0.5 * h2.conditional_ranking_loss(pair_loss, B0)
    assert torch.allclose(old_g2, h2_alpha0, atol=1e-12, rtol=1e-12)

    # Identity shuffling makes H3 and H4 identical and prototypes stop-gradient.
    identity = dict(prepared)
    identity['permutation'] = np.arange(n)
    id3 = PCHRRAObjective(identity, 'H3', device, CFG)
    id4 = PCHRRAObjective(identity, 'H4', device, CFG)
    z = torch.randn(n, 16, requires_grad=True)
    test_logits = torch.randn(n, 2, requires_grad=True)
    assert torch.allclose(id3.forward(test_logits, z), id4.forward(test_logits, z), atol=1e-7, rtol=1e-7)
    ec, eg = id3.prototype_evidence(z)
    assert not ec.requires_grad and not eg.requires_grad
    assert len(list(id3.__dict__.values())) > 0
    assert not hasattr(id3, 'parameters')
    assert np.isfinite(info['T_residual'])
    for group in GROUPS:
        obj = PCHRRAObjective(prepared, group, device, CFG)
        loss = obj.forward(test_logits, z)
        assert torch.isfinite(loss)
        assert torch.autograd.grad(loss, test_logits, retain_graph=True)[0].norm() > 0

    train_labels = prepared['y']
    labels_all = np.concatenate([train_labels, np.asarray([0, 1, 0, 1])])
    meta_all = {key: np.concatenate([prepared[key], extra]) for key, extra in {
        'age': np.asarray([31., 42., 53., 64.]), 'sex': np.asarray([1., 2., 1., 2.]),
        'site': np.asarray(['C', 'C', 'D', 'D'])}.items()}
    before = prepare(labels_all, meta_all, np.arange(n), 911, 0, CFG)
    labels_all[n:] = 1 - labels_all[n:]
    meta_all['age'][n:] = [90., 91., 92., 93.]
    meta_all['sex'][n:] = [-1., -1., -1., -1.]
    meta_all['site'][n:] = ['HELDOUT', 'HELDOUT', 'OTHER', 'OTHER']
    after = prepare(labels_all, meta_all, np.arange(n), 911, 0, CFG)
    assert digest(before['info']) == digest(after['info'])
    return {'passed': True, 'checks': 20, 'groups': list(GROUPS),
            'site_pairs': int((site > 0).sum()), 'condition_pairs': int((true > 0).sum()),
            'q_bounds': [float(q_pos.min()), float(q_pos.max())],
            'weight_conservation_relative_error': info['weight_conservation_relative_error'],
            'H3_H4_identity_equal': True, 'prototype_stop_gradient': True,
            'heldout_boundary': True}


def _stats_hash(tensor):
    return array_hash(tensor.detach().cpu().numpy())


def preflight_one(seed, fold, audit_info):
    loaded = Loaded(seed, fold, audit_info)
    try:
        hist_module = historical_prototype_module()
        hist_cfg = json.loads(read_bytes(D4 / 'config.json'))
        hist_prepared = hist_module.prepare(loaded.ctx['labels'], loaded.meta, loaded.sp['train'],
                                            seed, fold, hist_cfg)
        assert digest(hist_prepared['info']) == digest(loaded.prepared['info'])
        outside = np.setdiff1d(np.arange(871), loaded.sp['train'])
        parameter_values_before = {name: value.detach().clone()
                                   for name, value in loaded.core.named_parameters()}
        parameter_values_before['@unregistered_rp_weights'] = loaded.core.rp_attention.weights.detach().clone()
        param_before = state_hash(parameter_values_before)
        loaded.reset()
        loaded.core.train()
        loaded.core.zero_grad(set_to_none=True)
        counts = {'model': 0}
        hook = loaded.core.register_forward_hook(lambda module, args, out: counts.__setitem__('model', counts['model'] + 1))
        try:
            log('preflight_model_forward_started', seed=seed, fold=fold)
            logits, graph = loaded.forward()
            log('preflight_model_forward_completed', seed=seed, fold=fold, count=counts['model'])
        finally:
            hook.remove()
        assert counts['model'] == 1
        assert tuple(logits.shape) == (871, 2) and tuple(loaded.z_img.shape) == (871, 16)
        assert tuple(loaded.core.joint_embed.shape) == (871, 16)
        parameter_values_after_forward = {name: value.detach().clone()
                                          for name, value in loaded.core.named_parameters()}
        parameter_values_after_forward['@unregistered_rp_weights'] = loaded.core.rp_attention.weights.detach().clone()
        native_forward_changes = sorted(name for name in parameter_values_before
                                        if not torch.equal(parameter_values_before[name],
                                                           parameter_values_after_forward[name]))
        native_forward_vae_loads = [name for name in native_forward_changes if name.startswith('vae.')]
        native_forward_projections = [name for name in native_forward_changes
                                      if name == 'rp_attention.conv.weight']
        unexpected_native_changes = sorted(set(native_forward_changes) -
                                           set(native_forward_vae_loads) -
                                           set(native_forward_projections))
        assert not unexpected_native_changes, unexpected_native_changes
        if native_forward_vae_loads:
            expected_vae = load_torch(loaded.core.vae_save_path)
            observed_vae = loaded.core.vae.state_dict()
            assert expected_vae.keys() == observed_vae.keys()
            assert all(torch.equal(observed_vae[name].detach().cpu(), expected_vae[name])
                       for name in expected_vae)
        parameters_by_name = dict(loaded.core.named_parameters())
        with torch.no_grad():
            for name in native_forward_changes:
                if name == '@unregistered_rp_weights':
                    loaded.core.rp_attention.weights.copy_(parameter_values_before[name])
                else:
                    parameters_by_name[name].copy_(parameter_values_before[name])
        assert state_hash({name: value.detach() for name, value in loaded.core.named_parameters()} |
                          {'@unregistered_rp_weights': loaded.core.rp_attention.weights.detach()}) == param_before
        native = F.cross_entropy(logits[loaded.sp['train']], loaded.y[loaded.sp['train']]) + graph
        results, objectives = {}, {}
        for group in GROUPS:
            objective = PCHRRAObjective(loaded.prepared, group, loaded.opt.device, CFG)
            auxiliary = objective.forward(logits, loaded.z_img)
            lam, scale = gradient_budget(native, auxiliary, loaded.core.joint_embed, 9)
            g_logits = torch.autograd.grad(auxiliary, logits, retain_graph=True)[0]
            g_joint = torch.autograd.grad(auxiliary, loaded.core.joint_embed, retain_graph=True)[0]
            assert bool(torch.isfinite(auxiliary)) and float(g_logits.norm()) > 0 and float(g_joint.norm()) > 1e-12
            assert float(lam) > 0 and not lam.requires_grad
            assert objective.prototype_calls == (1 if group in ('H3', 'H4') else 0)
            results[group] = {'auxiliary_loss': float(auxiliary.detach()), **scale,
                              'objective': objective.last_info}
            objectives[group] = objective

        h3, h4 = objectives['H3'], objectives['H4']
        assert torch.equal(h3.true_condition, h4.true_condition)
        assert torch.equal(h3.site_condition, h4.site_condition)
        assert torch.equal(h3.global_weights, h4.global_weights)
        assert not torch.equal(h3.teacher_kernel, h4.teacher_kernel)
        assert h3.last_weights is not None and objectives['H2'].last_weights is not None
        h3_h2_delta = float((h3.last_weights - objectives['H2'].last_weights).abs().mean())
        assert h3_h2_delta > 0
        assert torch.equal(h3._prototype_cache['global_mu'], h4._prototype_cache['global_mu'])
        true_input_hash = loaded.input_sha256

        # Heldout values may change without changing any auxiliary loss or teacher evidence.
        changed_logits = logits.detach().clone()
        changed_z = loaded.z_img.detach().clone()
        changed_logits[outside] = 10000.0
        changed_z[outside] = -10000.0
        heldout_diff = {}
        for group in GROUPS:
            original_logits = logits.detach().clone().requires_grad_(True)
            altered_logits = changed_logits.detach().clone().requires_grad_(True)
            original = PCHRRAObjective(loaded.prepared, group, loaded.opt.device, CFG).forward(original_logits, loaded.z_img.detach())
            altered = PCHRRAObjective(loaded.prepared, group, loaded.opt.device, CFG).forward(altered_logits, changed_z)
            assert torch.allclose(original, altered, atol=1e-7, rtol=1e-7)
            heldout_diff[group] = float((original - altered).abs())

        # Independently verify H3 evidence against the historical D4 teacher on the same forward.
        hist_obj = hist_module.PrototypeObjective(hist_prepared, 'D4', hist_cfg, loaded.opt.device)
        _, hist_detail = hist_obj(loaded.z_img)
        hist_evidence = hist_detail['logits'][:, 0] - hist_detail['logits'][:, 1]
        h3_cond, h3_global = h3.prototype_evidence(loaded.z_img)
        assert torch.allclose(h3_cond, hist_evidence.detach(), atol=1e-6, rtol=1e-6)
        assert not h3_cond.requires_grad and not h3_global.requires_grad
        param_after = state_hash({name: value.detach()
                                  for name, value in loaded.core.named_parameters()} |
                                 {'@unregistered_rp_weights': loaded.core.rp_attention.weights.detach()})
        assert param_before == param_after
        assert np.array_equal(loaded.ctx['labels'], loaded.ctx['labels'].copy())
        changed_fraction = loaded.prepared['info']['shuffle_changed_fraction']
        assert changed_fraction > 0.2
        return {
            'passed': True, 'seed': seed, 'fold': fold, 'model_forward_count': 1,
            'logits_shape': list(logits.shape), 'z_img_shape': list(loaded.z_img.shape),
            'joint_embed_shape': list(loaded.core.joint_embed.shape), 'gradient_checks': results,
            'H3_H2_mean_absolute_weight_change': h3_h2_delta,
            'H3_H4_true_condition_equal': True, 'H3_H4_site_condition_equal': True,
            'H3_H4_global_prototypes_equal': True, 'H3_H4_teacher_kernel_different': True,
            'condition_shuffle_changed_fraction': changed_fraction,
            'historical_D4_conditional_evidence_max_abs_error': float((h3_cond - hist_evidence.detach()).abs().max()),
            'heldout_perturbation_loss_diff': heldout_diff,
            'parameter_hash_unchanged': param_before == param_after,
            'native_forward_projected_parameters_restored': native_forward_projections,
            'native_forward_loaded_vae_parameters_restored': native_forward_vae_loads,
            'same_model_inputs_H3_H4_sha256': true_input_hash,
            'physical_gpu': int(os.environ.get('CUDA_VISIBLE_DEVICES', '0')),
        }
    finally:
        loaded.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('audit', 'math-test', 'preflight', 'train'))
    parser.add_argument('--group', choices=GROUPS)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--fold', type=int)
    args = parser.parse_args()
    if args.action == 'math-test':
        result = math_test()
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return
    if args.action == 'audit':
        audit()
        return
    saved = guard()
    if args.action == 'preflight':
        if args.seed != 911 or args.fold != 0:
            raise ValueError('Only the single authorized 911/fold0 preflight is allowed')
        result = preflight_one(args.seed, args.fold, saved)
        with task_lock(990):
            saved = json.loads(read_bytes(BASE / 'audit.json'))
            saved['preflight']['real_forward']['result'] = result
            atomic_json(saved, BASE / 'audit.json')
        log('preflight_passed', seed=args.seed, fold=args.fold)
        return
    if args.group not in GROUPS or args.seed != 911 or args.fold not in range(10):
        raise ValueError('Formal task outside H1-H4, seed911, folds0-9')
    slot = GROUPS.index(args.group) * 10 + args.fold
    with task_lock(slot):
        train_one(args.group, args.seed, args.fold, saved)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        try:
            log('worker_error', message=str(exc), traceback=__import__('traceback').format_exc(),
                group=locals().get('args', None).group if 'args' in locals() else None)
        finally:
            raise
