"""Independent PCND runner; historical models and data are read-only."""
import argparse
import ast
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
import urllib.request
import warnings

sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'
BASE = Path(__file__).resolve().parent
ROOT = Path(json.loads((BASE / 'config.json').read_text())['root'])
OLD = ROOT / 'experiments/DRPA_20261014'
HIST = ROOT / 'experiments/PCCI_20261015'
ATT = Path(json.loads((BASE / 'config.json').read_text())['attribution_readonly_path'])
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE))
import numpy as np
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from torch.nn import functional as F
from opt import OptInit
from model.mm_gtunets import MM_GTUNets
from model.rp_graph import create_reward_penalty_graph
from utils.tools import EarlyStopping
import normative
from normative import prepare_fold, stable_seed, make_shuffle, design
from pcnd import PCNDEvidence

warnings.filterwarnings('ignore', message='To copy construct from a tensor')
warnings.filterwarnings('ignore', message='Converting a tensor with requires_grad=True')
warnings.filterwarnings('ignore', message='Please use the new API settings to control TF32 behavior')
warnings.filterwarnings('ignore', message='Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected')
torch.set_num_threads(1)
KEYS = ('ACC', 'BA', 'AUC', 'SEN', 'SPE', 'F1')
SUBSETS = ('train', 'val', 'test')
GROUPS = ('P1', 'P2', 'P3', 'P4', 'P5', 'P6')

def utc():
    return datetime.now(timezone.utc).isoformat()


def config():
    return json.loads((BASE / 'config.json').read_text())


def emit(kind, **values):
    print(json.dumps({'kind': kind, 'time_utc': utc(), **values}, ensure_ascii=False, allow_nan=False), flush=True)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def array_hash(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(value.shape).encode())
    h.update(value.tobytes())
    return h.hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode())
        h.update(array_hash(state[key]).encode())
    return h.hexdigest()


def atomic_json(value, path=None):
    target = Path(path or BASE / 'summary.json')
    tmp = target.with_name(target.name + f'.tmp.{os.getpid()}')
    try:
        with tmp.open('w') as stream:
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_torch(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    try:
        torch.save(value, tmp)
        with tmp.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def load_torch(path, mmap=False):
    return torch.load(path, map_location='cpu', weights_only=False, mmap=mmap)


def export_core(core):
    return {'state': {k: v.detach().cpu().clone() for k, v in core.state_dict().items()},
            'rp_weights': core.rp_attention.weights.detach().cpu().clone()}


def import_core(core, saved):
    core.load_state_dict(saved['state'], strict=True)
    with torch.no_grad():
        core.rp_attention.weights.copy_(saved['rp_weights'].to(core.device))


def core_hash(saved):
    return state_hash({**saved['state'], 'unregistered_rp_weights': saved['rp_weights']})


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'cpu': torch.get_rng_state().clone(), 'cuda': torch.cuda.get_rng_state(0).clone(),
            'backend': {'cudnn_deterministic': torch.backends.cudnn.deterministic,
                        'cudnn_benchmark': torch.backends.cudnn.benchmark,
                        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
                        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32}}


def restore_rng(saved):
    random.setstate(saved['python'])
    np.random.set_state(saved['numpy'])
    torch.set_rng_state(saved['cpu'])
    torch.cuda.set_rng_state(saved['cuda'], 0)
    b = saved['backend']
    torch.backends.cudnn.deterministic = b['cudnn_deterministic']
    torch.backends.cudnn.benchmark = b['cudnn_benchmark']
    torch.backends.cuda.matmul.allow_tf32 = b['matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32 = b['cudnn_allow_tf32']


def rng_hash(saved):
    return {'python': hashlib.sha256(repr(saved['python']).encode()).hexdigest(),
            'numpy': hashlib.sha256(repr(saved['numpy'][0]).encode() + saved['numpy'][1].tobytes()
                                     + repr(saved['numpy'][2:]).encode()).hexdigest(),
            'cpu': array_hash(saved['cpu']), 'cuda': array_hash(saved['cuda'])}


def require_ppu():
    if not torch.cuda.is_available():
        raise RuntimeError('CPU training forbidden')
    torch.cuda.set_device(0)
    if 'PPU' not in torch.cuda.get_device_name(0).upper():
        raise RuntimeError('Audited PPU backend required')
    return torch.device('cuda:0')


def options(seed):
    previous = sys.argv
    try:
        sys.argv = ['official_options']
        settings = OptInit(model='MM_GTUNets', dataset='ABIDE', atlas='aal')
    finally:
        sys.argv = previous
    args, cfg = settings.args, config()
    args.seed, args.device = seed, require_ppu()
    args.data_folder = cfg['data_folder']
    args.ckpt_path = str(OLD / f'artifacts/seed_{seed}/vae')
    args.epoch, args.early_stop = cfg['epochs'], cfg['early_stop']
    args.lr, args.vae_lr, args.wd = cfg['lr'], cfg['vae_lr'], cfg['weight_decay']
    return args


def anchor_dir(seed, fold):
    return OLD / f'artifacts/seed_{seed}/fold_{fold}'


def result_path(group, seed, fold):
    return BASE / f'runs/{group}/s{seed}_f{fold}.pt'


def metrics(labels, pred, probability):
    y, p = np.asarray(labels) == 0, np.asarray(pred) == 0
    tp, fn = int(np.sum(y & p)), int(np.sum(y & ~p))
    tn, fp = int(np.sum(~y & ~p)), int(np.sum(~y & p))
    sen = tp / (tp + fn) if tp + fn else None
    spe = tn / (tn + fp) if tn + fp else None
    return {'ACC': float(np.mean(y == p)), 'BA': (sen + spe) / 2 if sen is not None and spe is not None else None,
            'AUC': float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else None,
            'SEN': sen, 'SPE': spe, 'F1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0}


def distribution(values):
    if torch.is_tensor(values):
        values = values.detach().cpu().numpy()
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return {'n': 0}
    return {'n': int(values.size), 'mean': float(values.mean()), 'std': float(values.std()),
            'min': float(values.min()), 'max': float(values.max()),
            'p10': float(np.quantile(values, .1)), 'p50': float(np.quantile(values, .5)),
            'p90': float(np.quantile(values, .9))}


def metadata(ctx):
    with (Path(config()['data_folder']) / 'Phenotypic_V1_0b_preprocessed1.csv').open() as f:
        lookup = {row['SUB_ID']: row for row in csv.DictReader(f)}
    rows = [lookup[str(subject)] for subject in ctx['ids']]
    return {'age': np.array([float(row['AGE_AT_SCAN']) for row in rows]),
            'sex': np.array([row['SEX'] for row in rows]), 'site': np.array([row['SITE_ID'] for row in rows]),
            'labels': np.array([int(row['DX_GROUP']) - 1 for row in rows])}


def verify_oof(oof, ctx, split):
    idx = np.asarray(split['test'])
    if not np.array_equal(oof['indices'], idx) or not np.array_equal(oof['labels'], np.asarray(ctx['labels'])[idx]):
        raise AssertionError('OOF subject/label ordering mismatch')
    if oof['ids'] != [ctx['ids'][i] for i in idx] or oof['sites'] != [ctx['site_names'][i] for i in idx]:
        raise AssertionError('OOF ID/site mismatch')
    probability = np.asarray(oof['prob_asd'])
    if probability.shape != idx.shape or not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise AssertionError('Invalid probabilities')
    calculated = metrics(oof['labels'], oof['pred'], probability)
    if any(abs(calculated[k] - oof['metrics'][k]) > 1e-12 for k in KEYS):
        raise AssertionError('OOF metric recomputation failed')


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()} if module is not None else None


def grad_norm(parameters):
    values = [p.grad.detach().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def oof_record(loaded, logits):
    score = logits.detach().cpu().numpy()
    test = loaded.split['test']
    pred, p = score.argmax(1)[test], softmax(score, axis=1)[test, 0]
    return {'indices': test.tolist(), 'labels': np.asarray(loaded.ctx['labels'])[test].tolist(),
            'pred': pred.tolist(), 'prob_asd': p.tolist(), 'logits': score[test].tolist(),
            'ids': [loaded.ctx['ids'][i] for i in test], 'sites': [loaded.ctx['site_names'][i] for i in test],
            'metrics': metrics(np.asarray(loaded.ctx['labels'])[test], pred, p)}


def compare(a, b, label, atol=3e-6, rtol=3e-5):
    if a is None or b is None:
        if a is not None or b is not None:
            raise AssertionError(f'Connectivity mismatch {label}')
        return 0.0
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(f'{label}: max error {float((a-b).abs().max())}')
    return float((a-b).abs().max())


def parameter_changes(current, initial):
    answer = {}
    for prefix in ('img_unet.', 'ph_unet.', 'mm_attention.', 'clf.', 'rp_attention.'):
        pairs = [(current[k].float(), initial[k].float()) for k in current if k.startswith(prefix) and torch.is_floating_point(current[k])]
        distance = sum(float((a-b).double().square().sum()) for a,b in pairs) ** .5
        norm = sum(float(b.double().square().sum()) for _,b in pairs) ** .5
        answer[prefix.rstrip('.')] = {'l2_change': distance, 'relative_l2_change': distance / max(norm, 1e-30)}
    return answer


def private_seed(seed, fold, name):
    return int(hashlib.sha256(f'{seed}/{fold}/{name}'.encode()).hexdigest()[:15], 16)


def draw_rng(seed, fold, draw):
    value = private_seed(seed, fold, f"PCCI-repeat-{config()['repeat_seed_bases'][draw]}") % (2**32)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    return value


def code_hash():
    return hashlib.sha256(''.join(sha_file(BASE / n) for n in
        ('run.py', 'normative.py', 'pcnd.py', 'config.json')).encode()).hexdigest()


def guard():
    state = json.loads((BASE / 'summary.json').read_text())
    if state['training_code_sha256'] != code_hash():
        raise RuntimeError('Registered training source changed; never mix fingerprints')
    for name, saved in state['audit']['source_sha256'].items():
        if sha_file(ROOT / name) != saved:
            raise RuntimeError(f'Protected source changed: {name}')
    return state


def verify_authorized_algorithm():
    import re
    original = (ROOT / 'prompt').read_text()
    supplied = re.search(r'```python\n("""Self-contained.*?)\n```', original, re.S).group(1)
    supplied = supplied.replace('    norm_models={}\n', '').replace('        ref_set={}\n', '').replace('            ref_set[tuple(held.tolist())]=w\n', '')
    expected = ast.parse(supplied)
    actual = ast.parse((BASE / 'normative.py').read_text())
    functions = {n.name: n for n in expected.body if isinstance(n, ast.FunctionDef)}
    for i, node in enumerate(actual.body):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name == 'make_shuffle':
            actual.body[i] = functions['make_shuffle']
        elif node.name == 'prepare_fold':
            node.args.args = node.args.args[:-2]
            node.args.defaults = node.args.defaults[:-2]
            body = []
            for statement in node.body:
                if isinstance(statement, ast.If) and ast.unparse(statement.test) == 'pca is None':
                    body.extend(statement.body)
                elif isinstance(statement, ast.If) and ast.unparse(statement.test) == 'groups_to_compute is not None':
                    continue
                else:
                    body.append(statement)
            node.body = body
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'make_shuffle':
                    call.args = call.args[:-1]
    if ast.dump(expected) != ast.dump(actual):
        raise AssertionError('Unapproved change to supplied normative mathematics')
    model = re.search(r'```python\n(import torch\nfrom torch import nn\n.*?)\n```', original, re.S).group(1)
    if ast.dump(ast.parse(model)) != ast.dump(ast.parse((BASE / 'pcnd.py').read_text())):
        raise AssertionError('PCND architecture differs from preregistered implementation')
    if config()['p6_shuffle_scope'] != 'partition_within_site':
        raise AssertionError('Only the user-authorized partition-site revision is accepted')
    return {'supplied_math_and_PCND_architecture_unchanged': True,
            'authorized_changes': ['partition-site joint shuffle', 'reuse fitted PCA', 'compute affected groups only']}


def audit():
    previous = json.loads((HIST / 'summary.json').read_text())
    attribution = json.loads((ATT / 'summary.json').read_text())
    manifest = {}
    for name, saved in attribution['audit']['manifest'].items():
        path = ROOT / name
        value = sha_file(path)
        if value != saved['sha256']:
            raise AssertionError(f'Historical manifest changed: {name}')
        manifest[str(path)] = {'sha256': value, 'bytes': path.stat().st_size}
    for name, expected in attribution['generated_artifacts_sha256'].items():
        path = ATT / name
        if sha_file(path) != expected:
            raise AssertionError(f'Attribution bundle changed: {name}')
        manifest[str(path)] = {'sha256': expected, 'bytes': path.stat().st_size}
    # Summary itself is excluded from the historical self-referential manifest.
    for folder in (OLD, HIST, ATT):
        for path in folder.iterdir():
            if path.is_file():
                manifest[str(path)] = {'sha256': sha_file(path), 'bytes': path.stat().st_size}
    for path in (ROOT / 'model').rglob('*.py'):
        manifest[str(path)] = {'sha256': sha_file(path), 'bytes': path.stat().st_size}
    for path in (ROOT / 'utils').rglob('*.py'):
        manifest[str(path)] = {'sha256': sha_file(path), 'bytes': path.stat().st_size}
    ctx = load_torch(OLD / 'artifacts/context.pt')
    meta = metadata(ctx)
    if len(set(ctx['ids'])) != 871 or not np.array_equal(meta['labels'], ctx['labels']):
        raise AssertionError('ID/label order changed')
    if list(meta['site']) != ctx['site_names'] or np.bincount(ctx['labels']).tolist() != [403, 468]:
        raise AssertionError('Original phenotype/label mapping changed')
    ph = np.asarray(ctx['ph_data'])
    if ph.shape != (871, 3) or not np.isfinite(ph).all():
        raise AssertionError('Original phenotype tensor changed')
    baseline = previous['audit']['baseline']
    raw = load_torch(Path(config()['data_folder']) / 'save_tensor/up_triangles.pt')
    for seed in config()['seeds']:
        selected = load_torch(OLD / f'artifacts/seed_{seed}/selected_features.pt')
        mask = np.load(OLD / f'artifacts/seed_{seed}/rfe_support.npy')
        if tuple(selected.shape) != (871, 500) or not np.array_equal(np.asarray(raw, dtype=np.float32)[:, mask], selected.numpy()):
            raise AssertionError('Saved RFE feature mismatch')
        outer = StratifiedKFold(10, shuffle=True, random_state=seed)
        covered = []
        for fold, (tv, test) in enumerate(outer.split(np.zeros((871, 1)), ctx['labels'])):
            train, val = train_test_split(tv, test_size=.1, random_state=seed, stratify=ctx['labels'][tv])
            split = load_torch(anchor_dir(seed, fold) / 'splits.pt')
            if not all(np.array_equal(split[k], v) for k, v in zip(SUBSETS, (train, val, test))):
                raise AssertionError('Historical split mismatch')
            if len(np.unique(np.r_[train, val, test])) != 871:
                raise AssertionError('Splits overlap')
            covered.extend(test.tolist())
            start = load_torch(anchor_dir(seed, fold) / 'train_start.pt')
            pairing = previous['audit']['anchors'][f'{seed}/{fold}']['pairing']
            if core_hash(start['core']) != pairing['train_start_core_fingerprint'] or rng_hash(start['rng']) != pairing['classifier_start_rng']:
                raise AssertionError('Core/RNG anchor mismatch')
            b = baseline[f'{seed}/{fold}']
            verify_oof(b, ctx, split)
            if sha_file(ROOT / b['best_reference']) != b['best_sha256']:
                raise AssertionError('Historical C0 best checkpoint changed')
            if np.asarray(previous['baseline_frozen'][f'{seed}/{fold}']['test_prob_asd']).shape != (5, len(test)):
                raise AssertionError('Historical C0 frozen draws incomplete')
        if sorted(covered) != list(range(871)):
            raise AssertionError('Historical OOF coverage mismatch')
    verify_authorized_algorithm()
    remote = {}
    for relative in ('config.json', 'run.py', 'pcci.py', 'analyze.py'):
        url = f"https://raw.githubusercontent.com/tangaii/MM/{config()['github_reference']}/experiments/PCCI_20261015/{relative}"
        data = urllib.request.urlopen(url, timeout=45).read()
        if data != (HIST / relative).read_bytes():
            raise AssertionError('Historical PCCI differs from published GitHub source')
        remote[relative] = hashlib.sha256(data).hexdigest()
    for relative in ('run.py', 'controls.py', 'analyze.py', 'config.json', 'report.md'):
        url = f"https://raw.githubusercontent.com/tangaii/MM/{config()['attribution_reference']}/experiments/PCCI_Attribution_20261016/{relative}"
        data = urllib.request.urlopen(url, timeout=45).read()
        if data != (ATT / relative).read_bytes():
            raise AssertionError('Historical attribution differs from published GitHub source')
        remote['attribution/' + relative] = hashlib.sha256(data).hexdigest()
    old = json.loads((OLD / 'artifacts/audit.json').read_text())
    return {'created_utc': utc(), 'manifest': manifest, 'source_sha256': old['source_sha256'],
            'anchors': previous['audit']['anchors'], 'baseline': baseline,
            'baseline_frozen': previous['baseline_frozen'], 'data_digest': ctx['data_digest'],
            'handoff_before_sha256': sha_file(ROOT / 'handoff.md'),
            'prompt_sha256': sha_file(ROOT / 'prompt'), 'github_bytes_verified': remote,
            'historical_fits_verified_by_checksum': 330, 'label_counts': [403, 468],
            'environment': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                            'host_threads': torch.get_num_threads(), 'cpu_budget': 16},
            'audit_notes': [
                'Historical RFE cross-fold overwrite leakage deliberately retained',
                'RP early return, unregistered weights and in-forward .data projections retained',
                'Native VAE reload/resampling and eval GTUNet dropout retained',
                'No historical runner imported; PCND is explicit logit correction without hooks',
                'Official EarlyStopping equal-loss reset retained; save=False, no external write',
                'Core hidden=128 from Opt defaults; historical PCCI hidden=64 is adapter-only',
                'Formal test grad-enabled as actual historical runner; preflight checks no-grad value parity',
                'P6 revised before formal training to partition-site shuffle by explicit user authorization',
                'Experiment4 path is read-only /tmp, not automatically migrated'],
            'paper_audit': {'url': 'https://arxiv.org/html/2406.14455v3',
                'sections': '3.1-3.5, 4.1-4.2, Table5',
                'interpretation': 'Original model already uses age/sex/site in VAE and population graph; PCND tests additional conditional evidence',
                'table5_ABIDE_ACC_AUC': {'non_imaging': [52.00, 50.28], 'imaging': [76.49, 85.22], 'joint': [82.92, 88.21]}}}


def feature_path(seed, fold):
    return BASE / f'features/s{seed}_f{fold}.pt'


def feature_inputs(seed, fold, state):
    inputs = dict(state['audit']['anchors'][f'{seed}/{fold}']['input_sha256'])
    for path in (OLD / 'artifacts/context.pt',
                 Path(config()['data_folder']) / 'Phenotypic_V1_0b_preprocessed1.csv'):
        inputs[str(path.relative_to(ROOT))] = sha_file(path)
    return {'input_sha256': inputs, 'normative_sha256': sha_file(BASE / 'normative.py'),
            'config_sha256': sha_file(BASE / 'config.json')}


def feature_digest(features):
    return digest({g: {k: array_hash(v) for k, v in sorted(values.items())} for g, values in sorted(features.items())})


def verify_cache(seed, fold, state):
    path = feature_path(seed, fold)
    if not path.exists():
        return False
    saved = load_torch(path)
    if saved.get('state') != 'complete' or saved['fingerprint'] != feature_inputs(seed, fold, state):
        legacy = state.get('p6_revision', {}).get('previous_caches', {}).get(f'{seed}/{fold}')
        if legacy and sha_file(path) == legacy['file_sha256'] and saved['fingerprint'] == legacy['fingerprint']:
            return False
        raise AssertionError('Feature cache fingerprint mismatch; never silently reuse changed inputs')
    if saved['features_sha256'] != feature_digest(saved['features']):
        raise AssertionError('Feature cache numerical checksum mismatch')
    for group in GROUPS:
        f, g = saved['features'][group]['features'], saved['features'][group]['gate']
        if f.shape != (871, 32) or g.shape != (871,) or not np.isfinite(f).all() or not np.isfinite(g).all():
            raise AssertionError('Feature/gate shape or finiteness failed')
        if np.max(np.abs(f)) > 4 or np.min(g) < .25 or np.max(g) > 1:
            raise AssertionError('Feature/gate bounds failed')
    if not np.array_equal(saved['features']['P4']['features'], saved['features']['P5']['features']):
        raise AssertionError('P4/P5 features must be bitwise identical')
    if config()['p6_shuffle_scope'] == 'partition_within_site':
        boundary = saved['boundary_checks']
        if boundary['P6_boundary_conflict_observed'] or boundary['P6_heldout_covariate_transfers_to_train_TD']:
            raise AssertionError('Revised P6 cache violates training covariate boundaries')
        if saved['diagnostics']['shuffle']['scope'] != 'partition_within_site':
            raise AssertionError('Legacy shuffle cannot masquerade as the final cache')
        for g, hashes in saved['revision']['P1_P5_formal_sha256_before'].items():
            if hashes != {k: array_hash(saved['features'][g][k]) for k in ('features', 'gate')}:
                raise AssertionError('Previously accepted P1-P5 formal arrays changed')
    return True


def prepare_with_trace(x, y, meta, split, seed, fold, groups=None, fitted_pca=None):
    pca_class, fit = normative.PCA, normative.ridge_fit
    calls, pcas = [], []
    class TracedPCA(pca_class):
        def fit(self, matrix, y=None):
            if not np.array_equal(matrix, np.asarray(x, dtype=float)[split['train']]):
                raise AssertionError('PCA attempted to fit nontraining images')
            result = super().fit(matrix, y)
            pcas.append(self)
            return result
    def traced_fit(d, z, lamb=10.0):
        w = fit(d, z, lamb)
        calls.append({'design': d.copy(), 'target': z.copy(), 'weights': w.copy()})
        return w
    normative.PCA, normative.ridge_fit = TracedPCA, traced_fit
    try:
        features, diagnostics = prepare_fold(x, y, meta['age'], meta['sex'].astype(int),
                                             meta['site'], split, seed, fold, dims=config()['pca_dim'],
                                             groups_to_compute=groups, pca=fitted_pca)
    finally:
        normative.PCA, normative.ridge_fit = pca_class, fit
    assert len(pcas) == (1 if fitted_pca is None else 0)
    assert len(calls) == (6 if groups == ('P6',) else 40)
    model = pcas[0] if fitted_pca is None else fitted_pca
    pca_hash = state_hash({'components': model.components_, 'mean': model.mean_,
                          'variance': model.explained_variance_, 'singular': model.singular_values_})
    return features, diagnostics, calls, model, pca_hash


def check_feature_boundaries(x, y, meta, split, seed, fold, features, calls, model, pca_hash):
    tr = np.asarray(split['train'])
    heldout = np.r_[split['val'], split['test']]
    td = tr[y[tr] == 1]
    internal = list(StratifiedKFold(5, shuffle=True, random_state=stable_seed(seed, fold, 'PCND-inner')).split(tr, y[tr]))
    z = model.transform(np.asarray(x, float))
    z /= np.maximum(z[tr].std(axis=0, ddof=1), 1e-4)
    a0, a_scale = float(meta['age'][td].mean()), max(float(meta['age'][td].std(ddof=1)), 2.0)
    sex = meta['sex'].astype(int)
    sm = float((sex[td] == 2).mean())
    perm, moved = make_shuffle(meta['site'], meta['age'], sex, seed, fold, split)
    cursor, proofs = 0, []
    coeff_ranges = {}
    for group, kind in (('P2', 'global'), ('P3', 'age'), ('P4', 'age_sex'), ('P5', 'age_sex'), ('P6', 'age_sex')):
        start = cursor
        age_c, sex_c = (meta['age'][perm], sex[perm]) if group == 'P6' else (meta['age'], sex)
        d = design(age_c, sex_c, kind, a0, a_scale, sm)
        for inner, (a, b) in enumerate(internal):
            reference, query = tr[a][y[tr[a]] == 1], tr[b]
            if np.intersect1d(reference, query).size or not np.all(y[reference] == 1):
                raise AssertionError('Crossfit reference includes query or ASD')
            call = calls[cursor]; cursor += 1
            if not np.array_equal(call['design'], d[reference]) or not np.array_equal(call['target'], z[reference]):
                raise AssertionError('Normative fit used nontraining TD data')
            if group == 'P2':
                proofs.append({'inner': inner, 'reference_td': reference.tolist(), 'heldout_train': query.tolist()})
        call = calls[cursor]; cursor += 1
        if not np.array_equal(call['design'], d[td]) or not np.array_equal(call['target'], z[td]):
            raise AssertionError('Full normative reference is not exactly train TD')
        if group in ('P4', 'P5'):
            for a, b in internal:
                reference = tr[a][y[tr[a]] == 1]
                call = calls[cursor]; cursor += 1
                if not np.array_equal(call['design'], d[reference]) or not np.array_equal(call['target'], z[reference]):
                    raise AssertionError('Counterfactual refitted altered reference')
        coeff_ranges[group] = (start, cursor)
    if sorted(np.concatenate([tr[b] for a, b in internal]).tolist()) != sorted(tr.tolist()):
        raise AssertionError('Internal crossfit did not cover each training person once')
    # These are perturbation unit tests, never alternative training features.
    altered_y = np.asarray(y).copy()
    altered_y[heldout] = 1 - altered_y[heldout]
    f_labels, _, c_labels, _, h_labels = prepare_with_trace(x, altered_y, meta, split, seed, fold)
    if feature_digest(f_labels) != feature_digest(features) or h_labels != pca_hash:
        raise AssertionError('Heldout labels affected PCND preprocessing')
    if any(not np.array_equal(a['weights'], b['weights']) for a, b in zip(calls, c_labels)):
        raise AssertionError('Heldout labels changed normative weights')
    changed_x = np.asarray(x).copy()
    changed_x[heldout] += 7.0
    f_x, _, c_x, _, h_x = prepare_with_trace(changed_x, y, meta, split, seed, fold)
    if h_x != pca_hash or any(not np.array_equal(a['weights'], b['weights']) for a, b in zip(calls, c_x)):
        raise AssertionError('Heldout imaging changed train-only PCA/reference')
    changed_meta = {k: v.copy() for k, v in meta.items()}
    changed_meta['age'][heldout] += 17.0
    f_age, _, c_age, _, h_age = prepare_with_trace(x, y, changed_meta, split, seed, fold)
    if h_age != pca_hash:
        raise AssertionError('Heldout age changed PCA')
    age_weights_changed = {}
    for group, (a, b) in coeff_ranges.items():
        changed = any(not np.array_equal(calls[i]['weights'], c_age[i]['weights']) for i in range(a, b))
        age_weights_changed[group] = changed
        if changed:
            raise AssertionError('Heldout covariates changed real normative weights')
    for group in GROUPS:
        for name in ('features', 'gate'):
            if not np.array_equal(features[group][name][tr], f_age[group][name][tr]):
                raise AssertionError('Heldout age changed real training features/gates')
    if not np.array_equal(meta['site'], meta['site'][perm]) or moved['all_changed'] <= 0:
        raise AssertionError('Shuffle crossed site or was entirely empty')
    transferred = np.isin(perm[td], heldout)
    by_site = {}
    for site in np.unique(meta['site']):
        idx = np.flatnonzero(meta['site'] == site)
        by_site[str(site)] = {'n': len(idx), 'unchanged_singleton': len(idx) == 1,
            'changed_fraction': float(((meta['age'][idx] != meta['age'][perm[idx]]) | (sex[idx] != sex[perm[idx]])).mean()),
            'age_before': distribution(meta['age'][idx]), 'age_after': distribution(meta['age'][perm[idx]]),
            'female_before': int((sex[idx] == 2).sum()), 'female_after': int((sex[perm[idx]] == 2).sum())}
    return {'passed_except_pending_p6_policy': True, 'pca_sha256': pca_hash,
        'reference_weight_sha256': [array_hash(v['weights']) for v in calls],
        'pca_fit_indices': tr.tolist(), 'crossfit': proofs,
        'no_heldout_labels': True, 'heldout_images_cannot_change_PCA_or_references': True,
        'heldout_age_reference_weights_changed': age_weights_changed,
        'P6_heldout_covariate_transfers_to_train_TD': int(transferred.sum()),
        'P6_train_TD_reference_count': len(td), 'P6_boundary_conflict_observed': age_weights_changed['P6'],
        'shuffle': moved, 'shuffle_by_site': by_site,
        'pca_variance_ratio': model.explained_variance_ratio_.tolist(),
        'P4_P5_bitwise_features_equal': np.array_equal(features['P4']['features'], features['P5']['features'])}


def revise_feature_cache(seed, fold, state, x, y, meta, split):
    legacy = state['p6_revision']['previous_caches'][f'{seed}/{fold}']
    path = feature_path(seed, fold)
    if sha_file(path) != legacy['file_sha256']:
        raise AssertionError('Legacy cache changed before authorized migration')
    saved = load_torch(path)
    old_boundary = saved['boundary_checks']
    if saved['features_sha256'] != legacy['features_sha256'] or feature_digest(saved['features']) != legacy['features_sha256']:
        raise AssertionError('Legacy numerical checksum changed')
    before = {g: {k: array_hash(saved['features'][g][k]) for k in ('features', 'gate')} for g in GROUPS[:-1]}
    if before != legacy['P1_P5_formal_sha256']:
        raise AssertionError('Accepted P1-P5 formal arrays changed before migration')
    revised, diagnostics, calls, model, pca_hash = prepare_with_trace(x, y, meta, split, seed, fold, groups=('P6',))
    if pca_hash != old_boundary['pca_sha256']:
        raise AssertionError('Training-only PCA differs from the previously accepted PCA')
    tr = np.asarray(split['train']); outside = np.r_[split['val'], split['test']]
    td = tr[y[tr] == 1]
    age, sex, site = meta['age'], meta['sex'].astype(int), meta['site']
    perm, shuffle = make_shuffle(site, age, sex, seed, fold, split)
    p2, _ = make_shuffle(site, age, sex, seed, fold, split)
    if not np.array_equal(perm, p2):
        raise AssertionError('Shuffle is not reproducible')
    for part in SUBSETS:
        idx = np.asarray(split[part])
        if not np.array_equal(np.sort(perm[idx]), np.sort(idx)) or not np.all(site[idx] == site[perm[idx]]):
            raise AssertionError('P6 permutation crossed partition or site boundaries')
        for s in np.unique(site[idx]):
            members = idx[site[idx] == s]
            if len(members) > 1 and np.any(perm[members] == members):
                raise AssertionError('Non-singleton P6 site group has a self match')
    z = model.transform(np.asarray(x, float))
    pc_std = np.maximum(z[tr].std(axis=0, ddof=1), 1e-4)
    z /= pc_std
    a0, a_scale = float(age[td].mean()), max(float(age[td].std(ddof=1)), 2.0)
    sm = float((sex[td] == 2).mean())
    real_design = design(age, sex, 'age_sex', a0, a_scale, sm)
    shuffled_design = design(age[perm], sex[perm], 'age_sex', a0, a_scale, sm)
    internal = list(StratifiedKFold(5, shuffle=True, random_state=stable_seed(seed, fold, 'PCND-inner')).split(tr, y[tr]))
    real_prediction, alternate_prediction = np.empty_like(z), np.empty_like(z)
    alternate_gate, proofs, real_weights = np.empty(len(y), np.float32), [], []
    for i, (a, b) in enumerate(internal):
        reference, query = tr[a][y[tr[a]] == 1], tr[b]
        if np.intersect1d(reference, query).size:
            raise AssertionError('Cross-fitting reference includes its own query')
        call = calls[i]
        if not np.array_equal(call['design'], shuffled_design[reference]) or not np.array_equal(call['target'], z[reference]):
            raise AssertionError('P6 inner reference is not exactly current-training TD')
        w = normative.ridge_fit(real_design[reference], z[reference])
        if array_hash(w) != old_boundary['reference_weight_sha256'][12 + i]:
            raise AssertionError('Reconstructed true P4/P5 reference differs from accepted weights')
        real_weights.append(w)
        real_prediction[query] = real_design[query] @ w
        alternate_prediction[query] = shuffled_design[query] @ w
        alternate_gate[query] = normative.density_gate(age[perm[query]], sex[perm[query]], age[reference], sex[reference])
        proofs.append({'inner': i, 'reference_td': reference.tolist(), 'heldout_train': query.tolist(),
                       'reference_condition_sources': perm[reference].tolist(), 'self_fit_excluded': True})
    if not np.array_equal(calls[5]['design'], shuffled_design[td]) or not np.array_equal(calls[5]['target'], z[td]):
        raise AssertionError('P6 full reference is not exactly current-training TD')
    w = normative.ridge_fit(real_design[td], z[td])
    if array_hash(w) != old_boundary['reference_weight_sha256'][17]:
        raise AssertionError('Reconstructed full real reference differs from accepted reference')
    real_weights.append(w)
    real_prediction[outside] = real_design[outside] @ w
    alternate_prediction[outside] = shuffled_design[outside] @ w
    alternate_gate[outside] = normative.density_gate(age[perm[outside]], sex[perm[outside]], age[td], sex[td])
    scale = np.maximum((z[td] - real_prediction[td]).std(axis=0, ddof=1), .25)
    real_reconstructed = np.clip((z - real_prediction) / scale, -4, 4).astype('float32')
    if not np.array_equal(real_reconstructed, saved['features']['P4']['features']):
        raise AssertionError('True P4/P5 deviations no longer match bitwise')
    alternate = np.clip((z - alternate_prediction) / scale, -4, 4).astype('float32')
    for group in ('P4', 'P5'):
        saved['features'][group]['shuffled_features'] = alternate.copy()
        saved['features'][group]['shuffled_gate'] = alternate_gate.copy() if group == 'P5' else np.ones(len(y), np.float32)
    saved['features']['P6'] = revised['P6']
    mutation_proofs = {}
    for mutation in ('heldout_labels', 'heldout_age', 'heldout_sex', 'heldout_images'):
        xm, ym = np.asarray(x).copy(), np.asarray(y).copy()
        mm = {k: v.copy() for k, v in meta.items()}
        if mutation == 'heldout_labels':
            ym[outside] = 1 - ym[outside]
        elif mutation == 'heldout_age':
            mm['age'][outside] += 17.0
        elif mutation == 'heldout_sex':
            mm['sex'][outside] = np.where(mm['sex'][outside].astype(int) == 1, '2', '1')
        else:
            xm[outside] += 7.0
        result, _, mcalls, _, mh = prepare_with_trace(xm, ym, mm, split, seed, fold, groups=('P6',), fitted_pca=model)
        if mh != pca_hash or any(not np.array_equal(a['weights'], b['weights']) for a, b in zip(calls, mcalls)):
            raise AssertionError(f'{mutation} changed P6 training reference weights')
        for key in ('features', 'gate'):
            if not np.array_equal(result['P6'][key][tr], revised['P6'][key][tr]):
                raise AssertionError(f'{mutation} changed P6 training features/gates')
        if mutation == 'heldout_labels' and feature_digest(result) != feature_digest(revised):
            raise AssertionError('Heldout diagnosis affected P6 inference preprocessing')
        mutation_proofs[mutation] = {'reference_weights_identical': True, 'training_features_and_gates_identical': True,
            'heldout_inference_invariance_claimed': mutation == 'heldout_labels'}
    after = {g: {k: array_hash(saved['features'][g][k]) for k in ('features', 'gate')} for g in GROUPS[:-1]}
    if before != after:
        raise AssertionError('P6 migration altered P1-P5 formal features or gates')
    saved['revision'] = {'version': 2, 'created_utc': utc(), 'previous_file_sha256': legacy['file_sha256'],
        'previous_fingerprint': legacy['fingerprint'], 'previous_features_sha256': legacy['features_sha256'],
        'previous_boundary_checks': old_boundary, 'P1_P5_formal_sha256_before': before,
        'P1_P5_formal_bitwise_preserved': True, 'P4_P5_true_deviations_bitwise_revalidated': True,
        'P4_P5_auxiliary_query_updated_without_changing_real_reference': True}
    saved['diagnostics'] = diagnostics
    saved['boundary_checks'] = {'passed': True, 'pca_sha256': pca_hash,
        'pca_fit_indices': tr.tolist(), 'pca_variance_ratio': model.explained_variance_ratio_.tolist(),
        'reference_weight_sha256': old_boundary['reference_weight_sha256'][:34] + [array_hash(c['weights']) for c in calls],
        'P6_reference_weights': [c['weights'] for c in calls], 'true_reference_weights': real_weights,
        'crossfit': proofs, 'no_heldout_labels': True, 'heldout_images_cannot_change_PCA_or_references': True,
        'heldout_age_reference_weights_changed': {g: False for g in GROUPS[1:]},
        'P6_heldout_covariate_transfers_to_train_TD': int(np.isin(perm[td], outside).sum()),
        'P6_train_TD_reference_count': len(td), 'P6_boundary_conflict_observed': False,
        'shuffle': shuffle, 'P4_P5_bitwise_features_equal': True, 'P6_mutation_proofs': mutation_proofs,
        'training_parameter_invariance_not_heldout_inference_invariance': True}
    saved['pca_state'] = {'components': model.components_, 'mean': model.mean_, 'pc_std': pc_std,
                           'explained_variance': model.explained_variance_, 'singular': model.singular_values_}
    saved['fingerprint'] = feature_inputs(seed, fold, state)
    saved['features_sha256'] = feature_digest(saved['features'])
    saved['updated_utc'] = utc()
    saved['clip_fraction']['P6'] = float(np.mean(np.abs(revised['P6']['features']) >= config()['score_clip']))
    saved['gate_distribution']['P6'] = {k: distribution(revised['P6']['gate'][split[k]]) for k in SUBSETS}
    atomic_torch(saved, path)
    return saved


def feature_one(seed, fold, state):
    if verify_cache(seed, fold, state):
        return {'skipped': True, 'sha256': sha_file(feature_path(seed, fold))}
    ctx = load_torch(OLD / 'artifacts/context.pt')
    meta = metadata(ctx)
    split = load_torch(anchor_dir(seed, fold) / 'splits.pt')
    x = load_torch(anchor_dir(seed, fold).parent / 'selected_features.pt').numpy()
    x_before = array_hash(x)
    y = np.asarray(ctx['labels'])
    cpu_rng = torch.get_rng_state().clone()
    numpy_rng = np.random.get_state()
    if feature_path(seed, fold).exists() and state.get('p6_revision'):
        saved = revise_feature_cache(seed, fold, state, x, y, meta, split)
        features, boundary = saved['features'], saved['boundary_checks']
    else:
        raise RuntimeError('Expected registered legacy caches; refuse unplanned P1-P5 recomputation')
    if array_hash(x) != x_before or not torch.equal(torch.get_rng_state(), cpu_rng):
        raise AssertionError('Preprocessing changed input or Torch CPU RNG')
    nr = np.random.get_state()
    if nr[0] != numpy_rng[0] or not np.array_equal(nr[1], numpy_rng[1]) or nr[2:] != numpy_rng[2:]:
        raise AssertionError('Preprocessing changed global NumPy RNG')
    verify_cache(seed, fold, state)
    return {'passed': True, 'sha256': sha_file(feature_path(seed, fold)),
        'P6_boundary_conflict_observed': boundary['P6_boundary_conflict_observed'],
        'P6_heldout_covariate_transfers_to_train_TD': boundary['P6_heldout_covariate_transfers_to_train_TD']}


class Loaded:
    def __init__(self, seed, fold, state):
        self.seed, self.fold, self.state = seed, fold, state
        self.cfg, self.opt = config(), options(seed)
        self.ctx = load_torch(OLD / 'artifacts/context.pt')
        self.meta = metadata(self.ctx)
        self.split = load_torch(anchor_dir(seed, fold) / 'splits.pt')
        self.start = load_torch(anchor_dir(seed, fold) / 'train_start.pt')
        anchor = state['audit']['anchors'][f'{seed}/{fold}']
        self.pairing = anchor['pairing']
        for name, expected in anchor['input_sha256'].items():
            if sha_file(ROOT / name) != expected:
                raise AssertionError(f'Anchor changed: {name}')
        verify_cache(seed, fold, state)
        self.cache = load_torch(feature_path(seed, fold))
        self.core = MM_GTUNets(self.opt, fold).to(self.opt.device)
        self.core.vae_save_path = str(anchor_dir(seed, fold).parent / f'vae/fold{fold}_pretrain.pth')
        import_core(self.core, self.start['core'])
        if core_hash(export_core(self.core)) != self.pairing['train_start_core_fingerprint']:
            raise AssertionError('Loaded core differs from historical anchor')
        self.x_cpu = load_torch(anchor_dir(seed, fold).parent / 'selected_features.pt')
        affinity = create_reward_penalty_graph(self.ctx['ph_dict'], self.ctx['labels'],
                                               *[self.split[k] for k in SUBSETS], self.opt)
        self.ph_cpu = torch.from_numpy(self.ctx['ph_data']).float()
        self.affinity_cpu = torch.from_numpy(affinity).float()
        self.x, self.ph, self.affinity = [v.to(self.opt.device) for v in (self.x_cpu, self.ph_cpu, self.affinity_cpu)]
        self.labels = torch.as_tensor(self.ctx['labels'], dtype=torch.long, device=self.opt.device)
        self.train = torch.as_tensor(self.split['train'], dtype=torch.long, device=self.opt.device)
        self.val = torch.as_tensor(self.split['val'], dtype=torch.long, device=self.opt.device)
        self.data_input_digest = digest({'x': array_hash(self.x_cpu), 'ph': array_hash(self.ph_cpu),
            'affinity': array_hash(self.affinity_cpu), 'ids': self.ctx['ids'], 'labels': self.ctx['labels'].tolist()})
        self.branch = None

    def attach(self, group):
        self.group = group
        self.branch = make_branch(self.seed, self.fold, self.opt.device)
        self.deviation, self.reliability = self.features(group)

    def features(self, group, shuffled=False):
        c = self.cache['features'][group]
        name = 'shuffled_' if shuffled else ''
        return (torch.as_tensor(c[name + 'features'], device=self.opt.device, dtype=torch.float32),
                torch.as_tensor(c[name + 'gate'], device=self.opt.device, dtype=torch.float32))

    def optimizer(self, branch=True):
        parameters = list(self.core.parameters()) + (list(self.branch.parameters()) if branch else [])
        if any(p is self.core.rp_attention.weights for p in parameters):
            raise AssertionError('Unregistered RP weight was accidentally added to optimizer')
        return torch.optim.Adam(parameters, lr=self.cfg['lr'], weight_decay=self.cfg['weight_decay'])

    def forward(self, cpu_inputs=False):
        values = (self.x_cpu, self.ph_cpu, self.affinity_cpu) if cpu_inputs else (self.x, self.ph, self.affinity)
        base, graph = self.core(*values)
        if tuple(base.shape) != (871, 2) or tuple(self.core.joint_embed.shape) != (871, 16):
            raise AssertionError('Native model dimensions changed')
        return base, graph

    def close(self):
        self.__dict__.clear()
        torch.cuda.empty_cache()


def make_branch(seed, fold, device):
    # Seed only the CPU generator: torch.manual_seed would also alter PPU RNG.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(stable_seed(seed, fold, 'PCND-adapter-init'))
        module = PCNDEvidence(dim=32, hidden=32, logit_cap=config()['logit_cap'])
    return module.to(device)


@contextmanager
def task_lock(group, seed, fold, extra=0):
    offset = extra + (int(group[1:]) * 30 if group in GROUPS else 0) + config()['seeds'].index(seed) * 10 + fold
    with (BASE / 'run.py').open('r+') as stream:
        fcntl.lockf(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB, 1, offset)
        try:
            yield
        finally:
            fcntl.lockf(stream.fileno(), fcntl.LOCK_UN, 1, offset)


def gradient_stats(loaded):
    return {name: grad_norm(module.parameters()) for name, module in
        (('imaging', loaded.core.img_unet), ('phenotype', loaded.core.ph_unet),
         ('classifier', loaded.core.clf), ('pcnd', loaded.branch))}


def early_trace(loaded, logits, ce, graph):
    gradients = {n: p.grad for n, p in loaded.core.named_parameters() if p.grad is not None}
    return {'logits_sha256': array_hash(logits), 'train_ce': float(ce.detach()), 'train_graph': float(graph.detach()),
        'core_gradients_sha256': state_hash(gradients),
        'bn': {k: array_hash(v) for k, v in loaded.core.clf[2].state_dict().items()},
        'topk_sets': [[array_hash(torch.sort(p).values) for p in encoder.perms]
                      for encoder in (loaded.core.img_unet, loaded.core.ph_unet)],
        'rng_after_train': rng_hash(capture_rng())}


def formal_signature(saved):
    return digest({'oof': saved['oof'], 'best_core_sha256': saved['best_core_sha256'],
        'best_branch_sha256': saved['best_branch_sha256'], 'best_epoch': saved['best_epoch'],
        'epochs': saved['epochs'], 'formal_rng_before': rng_hash(saved['rng_before_test']),
        'formal_rng_after': rng_hash(saved['rng_after_test']), 'training_code_sha256': saved['training_code_sha256']})


def train_one(group, seed, fold, state):
    if verify_result(group, seed, fold, state):
        return {'skipped': True}
    if not state.get('preflight', {}).get('formal_authorized', False):
        raise RuntimeError('Formal training forbidden until every preflight/definition is accepted')
    loaded = Loaded(seed, fold, state)
    try:
        loaded.attach(group)
        core, branch = loaded.core, loaded.branch
        initial_branch = cpu_state(branch)
        optimizer = loaded.optimizer()
        stopper = EarlyStopping(patience=loaded.cfg['early_stop'], verbose=False, save=False)
        best, trace, best_loss = None, [], 1e10
        restore_rng(loaded.start['rng'])
        rng_start = capture_rng()
        if rng_hash(rng_start) != loaded.pairing['classifier_start_rng']:
            raise AssertionError('Training start RNG differs')
        if torch.are_deterministic_algorithms_enabled():
            raise AssertionError('Diagnostic determinism must not leak into formal protocol')
        started_utc = utc()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        bn_start = int(core.clf[2].num_batches_tracked)
        for epoch in range(loaded.cfg['epochs']):
            core.train(); branch.train()
            optimizer.zero_grad()
            base, graph = loaded.forward()
            logits, delta = branch(base, loaded.deviation, loaded.reliability)
            ce = F.cross_entropy(logits[loaded.split['train']], loaded.labels[loaded.split['train']])
            total = ce + graph
            if not bool(torch.isfinite(total)) or float(delta.detach().abs().max()) > .250001:
                raise AssertionError('Training loss/logit correction violates bounds')
            total.backward()
            record = {'epoch': epoch, 'train_ce': float(ce.detach()), 'train_graph': float(graph.detach()),
                'train_total': float(total.detach()), 'train_delta_abs': float(delta[loaded.train].detach().abs().mean()),
                'train_delta_l2': float(delta[loaded.train].detach().norm()),
                'epoch_delta_l2': float(delta.detach().norm())}
            if epoch < 3 or epoch % 25 == 0:
                record['gradient_norms'] = gradient_stats(loaded)
            if epoch < 3:
                record['early_trace'] = early_trace(loaded, logits, ce, graph)
            optimizer.step()
            expected_bn = bn_start + epoch + 1
            if int(core.clf[2].num_batches_tracked) != expected_bn:
                raise AssertionError('BN must update exactly once per training epoch')
            core.eval(); branch.eval()
            with torch.no_grad():
                base_val, graph_val = loaded.forward()
                val_logits, val_delta = branch(base_val, loaded.deviation, loaded.reliability)
            val_ce = F.cross_entropy(val_logits[loaded.split['val']], loaded.labels[loaded.split['val']])
            val_loss = val_ce + graph_val
            if not bool(torch.isfinite(val_loss)) or int(core.clf[2].num_batches_tracked) != expected_bn:
                raise AssertionError('Validation loss/BN changed unexpectedly')
            record.update(val_ce=float(val_ce), val_graph=float(graph_val), val_loss=float(val_loss),
                val_delta_abs=float(val_delta[loaded.val].abs().mean()),
                val_delta_l2=float(val_delta[loaded.val].norm()), bn_batches=expected_bn,
                rng_after_val=rng_hash(capture_rng()))
            trace.append(record)
            if float(val_loss) < best_loss:
                best_loss = float(val_loss)
                best = {'core': export_core(core), 'branch': cpu_state(branch), 'epoch': epoch,
                        'val_loss': best_loss, 'training_record': record.copy()}
            stopper(val_loss, core)
            if epoch % 100 == 0:
                emit('progress', group=group, seed=seed, fold=fold, epoch=epoch, validation_loss=float(val_loss))
            if stopper.early_stop:
                break
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        if best is None:
            raise AssertionError('No finite validation checkpoint')
        import_core(core, best['core'])
        branch.load_state_dict(best['branch'])
        core.eval(); branch.eval()
        before = capture_rng()
        # Preserve actual historical single-test grad mode as well as stochasticity.
        with torch.set_grad_enabled(True):
            native_logits, _ = loaded.forward()
            corrected, delta = branch(native_logits, loaded.deviation, loaded.reliability)
        after = capture_rng()
        oof = oof_record(loaded, corrected)
        result = {'state': 'trained', 'group': group, 'seed': seed, 'fold': fold,
            'started_utc': started_utc, 'completed_utc': utc(),
            'training_code_sha256': code_hash(), 'config_digest': digest(config()),
            'pairing': loaded.pairing, 'input_sha256': state['audit']['anchors'][f'{seed}/{fold}']['input_sha256'],
            'data_input_digest': loaded.data_input_digest, 'feature_sha256': sha_file(feature_path(seed, fold)),
            'core': best['core'], 'branch': best['branch'],
            'best_core_sha256': core_hash(best['core']), 'best_branch_sha256': state_hash(best['branch']),
            'initial_branch_sha256': state_hash(initial_branch), 'best_epoch': best['epoch'],
            'best_validation_loss': best_loss, 'best_training_record': best['training_record'],
            'epochs': trace, 'epochs_run': len(trace), 'classifier_seconds': seconds, 'oof': oof,
            'formal_start_rng': rng_start, 'rng_before_test': before, 'rng_after_test': after,
            'physical_gpu': int(os.environ.get('CUDA_VISIBLE_DEVICES', '0')),
            'device': torch.cuda.get_device_name(0), 'peak_memory_bytes': torch.cuda.max_memory_allocated(),
            'optimizer_core_parameter_order': [n for n, p in core.named_parameters()],
            'parameters': {'core': sum(p.numel() for p in core.parameters()),
                           'branch': sum(p.numel() for p in branch.parameters())},
            'normative_train_td_count': loaded.cache['diagnostics']['train_td'],
            'pca_components': loaded.cache['diagnostics']['pca_components'],
            'pca_variance_ratio': loaded.cache['boundary_checks']['pca_variance_ratio'],
            'clip_fraction': loaded.cache['clip_fraction'][group],
            'gate_distribution': loaded.cache['gate_distribution'][group],
            'epoch_delta_l2': [v['epoch_delta_l2'] for v in trace],
            'test_delta_abs': distribution(delta.detach().cpu().numpy()[loaded.split['test']]),
            'test_delta_abs_mean': float(delta[loaded.split['test']].detach().abs().mean()),
            'test_delta_max_abs': float(delta.detach().abs().max()),
            'formal_test_grad_enabled': True, 'pcnd_hook_count': 0}
        result['test_delta_abs'] = distribution(np.abs(delta.detach().cpu().numpy()[loaded.split['test']]))
        result['formal_signature'] = formal_signature(result)
        atomic_torch(result, result_path(group, seed, fold))
        verify_result(group, seed, fold, state)
        # Do not expose test scores until all 180 training tasks have finished.
        return {'best_epoch': best['epoch'], 'epochs_run': len(trace), 'seconds': seconds}
    finally:
        loaded.close()


def verify_result(group, seed, fold, state, complete=False):
    path = result_path(group, seed, fold)
    if not path.exists():
        return False
    saved = load_torch(path, mmap=True)
    expected = {'group': group, 'seed': seed, 'fold': fold,
        'training_code_sha256': state['training_code_sha256'], 'config_digest': digest(config()),
        'input_sha256': state['audit']['anchors'][f'{seed}/{fold}']['input_sha256']}
    if any(saved.get(k) != v for k, v in expected.items()):
        raise AssertionError(f'Result fingerprint mismatch: {path}')
    if saved['state'] not in ('trained', 'complete'):
        return False
    if core_hash(saved['core']) != saved['best_core_sha256'] or state_hash(saved['branch']) != saved['best_branch_sha256']:
        raise AssertionError('Checkpoint numerical checksum mismatch')
    if saved['formal_signature'] != formal_signature(saved):
        raise AssertionError('Formal result changed after save')
    for tensor in list(saved['core']['state'].values()) + list(saved['branch'].values()) + [saved['core']['rp_weights']]:
        if not bool(torch.isfinite(tensor).all()):
            raise AssertionError('Nonfinite checkpoint')
    ctx = load_torch(OLD / 'artifacts/context.pt')
    split = load_torch(anchor_dir(seed, fold) / 'splits.pt')
    verify_oof(saved['oof'], ctx, split)
    scores = np.asarray(saved['oof']['logits'])
    if scores.shape != (len(split['test']), 2) or not np.isfinite(scores).all():
        raise AssertionError('Invalid formal logits')
    if not np.allclose(softmax(scores, axis=1)[:, 0], saved['oof']['prob_asd'], atol=2e-7, rtol=2e-7):
        raise AssertionError('ASD probability is not column zero of saved logits')
    if len(saved['epochs']) != saved['epochs_run'] or saved['epochs'][saved['best_epoch']]['val_loss'] != saved['best_validation_loss']:
        raise AssertionError('Epoch/checkpoint mismatch')
    if saved['feature_sha256'] != sha_file(feature_path(seed, fold)) or saved['test_delta_max_abs'] > .250001:
        raise AssertionError('Feature cache changed or correction unbounded')
    if saved['physical_gpu'] != fold % 2:
        raise AssertionError('Physical card pairing/90-90 assignment changed')
    if complete and group in config()['repeat_groups']:
        if saved['state'] != 'complete' or 'frozen' not in saved:
            return False
        f = saved['frozen']
        p, pred = np.asarray(f['test_prob_asd']), np.asarray(f['test_pred'])
        if p.shape != (5, len(split['test'])) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise AssertionError('Frozen probability invalid')
        for d in range(5):
            m = metrics(saved['oof']['labels'], pred[d], p[d])
            if any(abs(m[k] - f['draw_metrics'][d][k]) > 1e-12 for k in KEYS):
                raise AssertionError('Frozen metrics invalid')
        if not f['same_native_logits_for_all_counterfactuals'] or not f['BN_unchanged']:
            raise AssertionError('Counterfactual pairing failed')
    return True


def check_one(seed, fold, state):
    loaded = Loaded(seed, fold, state)
    original_determinism = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        core = loaded.core
        before_rng = rng_hash(capture_rng())
        initial = {}
        for group in GROUPS:
            branch = make_branch(seed, fold, loaded.opt.device)
            initial[group] = state_hash(cpu_state(branch))
        if len(set(initial.values())) != 1 or rng_hash(capture_rng()) != before_rng:
            raise AssertionError('Branch initialization varies by group or pollutes core RNG')
        trajectories = {}
        classifier_count = [0]
        real_clf_forward = core.clf.forward
        def counted(*args, **kwargs):
            classifier_count[0] += 1
            return real_clf_forward(*args, **kwargs)
        core.clf.forward = counted
        for mode in ('historical_baseline', 'PCND_forced_zero'):
            import_core(core, loaded.start['core'])
            loaded.attach('P5')
            optimizer = loaded.optimizer(branch=mode != 'historical_baseline')
            restore_rng(loaded.start['rng'])
            steps = []
            for step in range(config()['equivalence_steps']):
                core.train(); loaded.branch.train()
                optimizer.zero_grad()
                count = classifier_count[0]
                base, graph = loaded.forward(cpu_inputs=mode == 'historical_baseline')
                if classifier_count[0] != count + 1:
                    raise AssertionError('Native classifier not called exactly once')
                if mode == 'PCND_forced_zero':
                    corrected, delta = loaded.branch(base, loaded.deviation, loaded.reliability)
                    # Isolate trainer/optimizer paths for three whole steps, not an active adapter.
                    logits = base + (corrected - base) * 0
                    if bool(torch.count_nonzero(delta)):
                        raise AssertionError('Parity bridge ceased being zero')
                else:
                    logits = base
                ce = F.cross_entropy(logits[loaded.split['train']], loaded.labels[loaded.split['train']])
                (ce + graph).backward()
                gradients = {n: p.grad.detach().cpu().clone() if p.grad is not None else None
                             for n, p in core.named_parameters()}
                pools = [p.detach().cpu().clone() for encoder in (core.img_unet, core.ph_unet) for p in encoder.perms]
                optimizer.step()
                adam = {n: {k: v.detach().cpu().clone() for k, v in optimizer.state[p].items() if torch.is_tensor(v)}
                        for n, p in core.named_parameters() if p in optimizer.state}
                core.eval(); loaded.branch.eval()
                count = classifier_count[0]
                with torch.no_grad():
                    val, vg = loaded.forward(cpu_inputs=mode == 'historical_baseline')
                    if mode == 'PCND_forced_zero':
                        cv, _ = loaded.branch(val, loaded.deviation, loaded.reliability)
                        val = val + (cv - val) * 0
                if classifier_count[0] != count + 1:
                    raise AssertionError('Validation classifier not called exactly once')
                vc = F.cross_entropy(val[loaded.split['val']], loaded.labels[loaded.split['val']])
                steps.append({'logits': logits.detach().cpu(), 'ce': ce.detach().cpu(), 'graph': graph.detach().cpu(),
                    'val_logits': val.cpu(), 'val_loss': (vc + vg).cpu(), 'gradients': gradients,
                    'adam': adam, 'core': export_core(core), 'pools': pools, 'rng': rng_hash(capture_rng())})
            trajectories[mode] = steps
        max_error = 0.0
        for a, b in zip(trajectories['historical_baseline'], trajectories['PCND_forced_zero']):
            for key in ('logits', 'ce', 'graph', 'val_logits', 'val_loss'):
                max_error = max(max_error, compare(a[key], b[key], key))
            for name in a['gradients']:
                max_error = max(max_error, compare(a['gradients'][name], b['gradients'][name], name))
            for name in a['adam']:
                for key in a['adam'][name]:
                    max_error = max(max_error, compare(a['adam'][name][key], b['adam'][name][key], 'Adam/' + name))
            if core_hash(a['core']) != core_hash(b['core']) or a['rng'] != b['rng']:
                raise AssertionError('Deterministic core/BN/RNG trajectory differs')
            if any(not torch.equal(torch.sort(x).values, torch.sort(y).values) for x, y in zip(a['pools'], b['pools'])):
                raise AssertionError('TopK selected node sets differ')
        del trajectories
        import_core(core, loaded.start['core'])
        loaded.attach('P5')
        restore_rng(loaded.start['rng'])
        optimizer = loaded.optimizer()
        gradients = []
        for step in range(2):
            core.train(); loaded.branch.train(); optimizer.zero_grad()
            base, graph = loaded.forward()
            native_bn = state_hash(core.clf[2].state_dict())
            logits, delta = loaded.branch(base, loaded.deviation, loaded.reliability)
            if step == 0 and (not torch.equal(logits, base) or bool(torch.count_nonzero(delta))):
                raise AssertionError('Initial active branch is not exactly native')
            ce = F.cross_entropy(logits[loaded.train], loaded.labels[loaded.train])
            (ce + graph).backward()
            norms = gradient_stats(loaded)
            norms['pcnd_first_layer'] = grad_norm(loaded.branch.net[0].parameters())
            norms['pcnd_last_layer'] = grad_norm(loaded.branch.net[-1].parameters())
            if any(norms[k] <= 0 for k in ('imaging', 'phenotype', 'classifier', 'pcnd_last_layer')):
                raise AssertionError('A native or PCND gradient path is disconnected')
            if step == 1 and norms['pcnd_first_layer'] <= 0:
                raise AssertionError('PCND hidden layer did not receive gradients after first update')
            if state_hash(core.clf[2].state_dict()) != native_bn:
                raise AssertionError('Explicit branch updated native BN')
            optimizer.step()
            gradients.append(norms)
        core.eval(); loaded.branch.eval()
        saved_core, saved_rng = export_core(core), capture_rng()
        with torch.no_grad():
            eval_base, _ = loaded.forward()
            bn = state_hash(core.clf[2].state_dict())
            p4f, _ = loaded.features('P4')
            p5f, _ = loaded.features('P5')
            one = torch.ones(871, device=loaded.opt.device)
            four, d4 = loaded.branch(eval_base, p4f, one)
            five, d5 = loaded.branch(eval_base, p5f, one)
            shf, shg = loaded.features('P5', shuffled=True)
            shuffled, _ = loaded.branch(eval_base, shf, shg)
            if not torch.equal(four, five) or not torch.equal(d4, d5) or float(d5.abs().max()) > .250001:
                raise AssertionError('P4/P5 gate=1 equivalence or delta cap failed')
            if state_hash(core.clf[2].state_dict()) != bn:
                raise AssertionError('Counterfactual branch calls changed native BN')
        import_core(core, saved_core); restore_rng(saved_rng)
        with torch.set_grad_enabled(True):
            eval_with_grad, _ = loaded.forward()
        grad_mode_error = compare(eval_base.cpu(), eval_with_grad.detach().cpu(), 'Historical test grad-mode equivalence')
        # Overflow-scale synthetic inputs stress the cap, not alternative method tuning.
        with torch.no_grad():
            loaded.branch.net[-1].weight.fill_(1000)
            extreme, d = loaded.branch(eval_base, loaded.deviation * 1000, loaded.reliability)
            if not bool(torch.isfinite(extreme).all()) or float(d.abs().max()) > .250001:
                raise AssertionError('Saturated branch became unbounded')
        module = make_branch(seed, fold, loaded.opt.device)
        deviation = loaded.deviation.detach().clone().requires_grad_(True)
        synthetic_optimizer = torch.optim.Adam(module.parameters(), lr=config()['lr'])
        for step in range(2):
            synthetic_optimizer.zero_grad()
            values, dd = module(eval_base.detach(), deviation, loaded.reliability)
            loss = F.cross_entropy(values[loaded.train], loaded.labels[loaded.train])
            loss.backward()
            if step == 1 and (deviation.grad is None or float(deviation.grad.norm()) <= 0):
                raise AssertionError('Learned evidence is not differentiable with respect to deviation')
            synthetic_optimizer.step()
        capacity = sum(p.numel() for p in module.parameters())
        if capacity != 1089:
            raise AssertionError('PCND capacity changed')
        direction = metrics([0, 0, 1, 1], [0, 1, 0, 1], [.9, .8, .2, .1])
        if direction['AUC'] != 1.0 or direction['SEN'] != .5 or direction['SPE'] != .5:
            raise AssertionError('ASD probability/SEN/SPE direction is wrong')
        # Verify official equal-loss patience behavior without creating a checkpoint.
        stopper = EarlyStopping(patience=2, save=False)
        for value in (1.0, 1.1, 1.0):
            stopper(torch.tensor(value), core)
        if stopper.counter != 0 or stopper.early_stop:
            raise AssertionError('Official equal-loss early-stop behavior drifted')
        buffer = io.BytesIO()
        serialized = {'core': export_core(core), 'branch': cpu_state(module), 'rng': capture_rng()}
        torch.save(serialized, buffer)
        buffer.seek(0)
        roundtrip = torch.load(buffer, map_location='cpu', weights_only=False)
        if core_hash(roundtrip['core']) != core_hash(serialized['core']) or state_hash(roundtrip['branch']) != state_hash(serialized['branch']) or rng_hash(roundtrip['rng']) != rng_hash(serialized['rng']):
            raise AssertionError('Checkpoint/RNG serialization changed values')
        core.clf.forward = real_clf_forward
        return {'passed': True, 'seed': seed, 'fold': fold, 'steps': 3,
            'deterministic_max_error': max_error, 'exact_core_hashes_Adam_BN_RNG_TopK': True,
            'parity_control': 'PCND delta forced zero throughout three steps; not an assertion about active 500-epoch trajectories',
            'classifier_calls_per_native_forward': 1, 'initial_branch_hashes': initial,
            'initial_core_rng_not_polluted': True, 'actual_PCND_gradient_norms_step0_step1': gradients,
            'new_parameters_each_group': capacity, 'P4_P5_gate1_exact': True,
            'same_native_logits_counterfactual_BN_unchanged': True, 'historical_grad_mode_max_error': grad_mode_error,
            'checkpoint_full_rng_roundtrip_exact': True, 'data_input_digest': loaded.data_input_digest,
            'device': torch.cuda.get_device_name(0), 'physical_gpu': int(os.environ['CUDA_VISIBLE_DEVICES']),
            'rtol': 3e-5, 'atol': 3e-6}
    finally:
        torch.use_deterministic_algorithms(original_determinism)
        loaded.close()


def frozen_one(group, seed, fold, state):
    if group not in config()['repeat_groups']:
        raise AssertionError('Only preregistered P4/P5/P6 receive auxiliary repeats')
    if verify_result(group, seed, fold, state, complete=True):
        return {'skipped': True}
    if not state.get('all_formal_training_complete'):
        raise RuntimeError('Auxiliary inference must follow all 180 formal trainings')
    saved = load_torch(result_path(group, seed, fold))
    signature_before = saved['formal_signature']
    loaded = Loaded(seed, fold, state)
    try:
        loaded.attach(group)
        core, branch = loaded.core, loaded.branch
        core.eval(); branch.load_state_dict(saved['branch']); branch.eval()
        test = np.asarray(loaded.split['test'])
        labels = np.asarray(loaded.ctx['labels'])[test]
        main_name = 'as_trained_shuffled' if group == 'P6' else 'true'
        variants = (main_name, 'off') if group == 'P6' else ('true', 'shuffled', 'off', 'unit_gate') if group == 'P5' else ('true', 'shuffled', 'off')
        probabilities = {name: [] for name in variants}
        deltas = {name: [] for name in variants}
        predictions, draws, draw_seeds, rngs = [], [], [], []
        for draw in range(5):
            import_core(core, saved['core'])
            restore_rng(loaded.start['rng'])
            draw_seeds.append(draw_rng(seed, fold, draw))
            rng_before = capture_rng()
            bn = state_hash(core.clf[2].state_dict())
            with torch.no_grad():
                native, graph = loaded.forward()
                logits, delta = branch(native, loaded.deviation, loaded.reliability)
                p = torch.softmax(logits, dim=1)[:, 0].cpu().numpy()
                pred = logits.argmax(1).cpu().numpy()
                probabilities[main_name].append(p[test])
                deltas[main_name].append(delta.cpu().numpy()[test])
                if group != 'P6':
                    sf, sg = loaded.features(group, shuffled=True)
                    alternative, ad = branch(native, sf, sg)
                    probabilities['shuffled'].append(torch.softmax(alternative, dim=1)[:, 0].cpu().numpy()[test])
                    deltas['shuffled'].append(ad.cpu().numpy()[test])
                    if group == 'P5':
                        unit_logits, unit_delta = branch(native, loaded.deviation, torch.ones_like(loaded.reliability))
                        probabilities['unit_gate'].append(torch.softmax(unit_logits, dim=1)[:, 0].cpu().numpy()[test])
                        deltas['unit_gate'].append(unit_delta.cpu().numpy()[test])
                probabilities['off'].append(torch.softmax(native, dim=1)[:, 0].cpu().numpy()[test])
                deltas['off'].append(np.zeros(len(test), dtype=np.float32))
                predictions.append(pred[test])
                draws.append(metrics(labels, pred[test], p[test]))
            if state_hash(core.clf[2].state_dict()) != bn:
                raise AssertionError('Frozen native or counterfactual calls changed BN')
            rngs.append({'before': rng_before, 'after': capture_rng()})
        probability = np.stack(probabilities[main_name])
        mean = probability.mean(0)
        frozen = {'policy': 'five native stochastic forwards, same historical PCCI-repeat SHA rule; not formal replacement',
            'draw_seeds': draw_seeds, 'draw_full_rng': rngs, 'main_condition': main_name,
            'test_prob_asd': probability, 'test_pred': np.stack(predictions), 'draw_metrics': draws,
            'mean_probability_metrics': metrics(labels, np.where(mean >= .5, 0, 1), mean),
            'counterfactual_prob_asd': {k: np.stack(v) for k, v in probabilities.items()},
            'counterfactual_delta': {k: np.stack(v) for k, v in deltas.items()},
            'test_prediction_instability_fraction': float(np.any(np.stack(predictions) != predictions[0], axis=0).mean()),
            'same_native_logits_for_all_counterfactuals': True, 'native_core_forwards': 5,
            'BN_unchanged': True, 'off_is_current_core_not_historical_C0': True, 'pcnd_hook_count': 0,
            'completed_utc': utc()}
        saved['frozen'], saved['state'] = frozen, 'complete'
        if formal_signature(saved) != signature_before:
            raise AssertionError('Frozen evaluation altered formal results')
        atomic_torch(saved, result_path(group, seed, fold))
        verify_result(group, seed, fold, state, complete=True)
        return {'completed_draws': 5, 'BN_unchanged': True}
    finally:
        loaded.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('feature', 'check', 'train', 'frozen'))
    parser.add_argument('--group', default='P1', choices=GROUPS)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--fold', type=int, required=True)
    args = parser.parse_args()
    state = guard()
    if args.action != 'feature':
        require_ppu()
    with task_lock(args.group, args.seed, args.fold, extra={'feature': 500, 'check': 1000, 'train': 0, 'frozen': 1500}[args.action]):
        emit('begin', action=args.action, group=args.group, seed=args.seed, fold=args.fold)
        if args.action == 'feature':
            result = feature_one(args.seed, args.fold, state)
        elif args.action == 'check':
            result = check_one(args.seed, args.fold, state)
        elif args.action == 'train':
            result = train_one(args.group, args.seed, args.fold, state)
        else:
            result = frozen_one(args.group, args.seed, args.fold, state)
        emit('done', action=args.action, group=args.group, seed=args.seed, fold=args.fold, result=result)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.exit(1)

