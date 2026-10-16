"""Independent anchored runner. No historical training script is imported."""
import argparse
import ast
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
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
from controls import PCCI, Bridge, FrozenCore, encode_conditions, private_seed

warnings.filterwarnings('ignore', message='To copy construct from a tensor')
warnings.filterwarnings('ignore', message='Converting a tensor with requires_grad=True')
warnings.filterwarnings('ignore', message='Please use the new API settings to control TF32 behavior')
warnings.filterwarnings('ignore', message='Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected')
torch.set_num_threads(1)
KEYS = ('ACC', 'BA', 'AUC', 'SEN', 'SPE', 'F1')
SUBSETS = ('train', 'val', 'test')
GROUPS = ('R0', 'R1', 'R2', 'R3')


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


def code_hash():
    return hashlib.sha256(''.join(sha_file(BASE / n) for n in ('run.py', 'controls.py', 'config.json')).encode()).hexdigest()


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


def draw_rng(seed, fold, draw):
    value = private_seed(seed, fold, f"PCCI-repeat-{config()['repeat_seed_bases'][draw]}") % (2**32)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)
    return value


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


def guard():
    state = json.loads((BASE / 'summary.json').read_text())
    if state['training_code_sha256'] != code_hash():
        raise RuntimeError('Registered training source changed')
    for name, saved in state['audit']['source_sha256'].items():
        if sha_file(ROOT / name) != saved:
            raise RuntimeError(f'Protected source changed: {name}')
    return state


def audit():
    previous = json.loads((HIST / 'summary.json').read_text())
    manifest = {}
    for name, saved in previous['audit']['manifest'].items():
        path = ROOT / name
        value = sha_file(path)
        if value != saved['sha256']:
            raise AssertionError(f'Historical manifest changed: {name}')
        manifest[name] = {'sha256': value, 'bytes': path.stat().st_size}
    for path in [*HIST.glob('*'), *HIST.glob('runs/C*/*.pt')]:
        if path.is_file():
            manifest[str(path.relative_to(ROOT))] = {'sha256': sha_file(path), 'bytes': path.stat().st_size}
    old = json.loads((OLD / 'artifacts/audit.json').read_text())
    ctx = load_torch(OLD / 'artifacts/context.pt')
    meta = metadata(ctx)
    if len(set(ctx['ids'])) != 871 or not np.array_equal(meta['labels'], ctx['labels']):
        raise AssertionError('ID/label order changed')
    if list(meta['site']) != ctx['site_names'] or np.bincount(ctx['labels']).tolist() != [403, 468]:
        raise AssertionError('Original phenotype/label mapping changed')
    if ast.dump(next(n for n in ast.parse((HIST / 'pcci.py').read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'PCCI')) != ast.dump(next(n for n in ast.parse((BASE / 'controls.py').read_text()).body if isinstance(n, ast.ClassDef) and n.name == 'PCCI')):
        raise AssertionError('PCCI formula/init AST differs from historical implementation')
    baseline = previous['audit']['baseline']
    for seed in config()['seeds']:
        selected = load_torch(OLD / f'artifacts/seed_{seed}/selected_features.pt')
        mask = np.load(OLD / f'artifacts/seed_{seed}/rfe_support.npy')
        raw = load_torch(Path(config()['data_folder']) / 'save_tensor/up_triangles.pt')
        if not np.array_equal(np.asarray(raw, dtype=np.float32)[:, mask], selected.numpy()):
            raise AssertionError('Saved RFE feature mismatch')
        outer = StratifiedKFold(10, shuffle=True, random_state=seed)
        for fold, (tv, test) in enumerate(outer.split(np.zeros((871, 1)), ctx['labels'])):
            train, val = train_test_split(tv, test_size=.1, random_state=seed, stratify=ctx['labels'][tv])
            split = load_torch(anchor_dir(seed, fold) / 'splits.pt')
            if not all(np.array_equal(split[k], v) for k, v in zip(SUBSETS, (train, val, test))):
                raise AssertionError('Historical split mismatch')
            start = load_torch(anchor_dir(seed, fold) / 'train_start.pt')
            pairing = previous['audit']['anchors'][f'{seed}/{fold}']['pairing']
            if core_hash(start['core']) != pairing['train_start_core_fingerprint'] or rng_hash(start['rng']) != pairing['classifier_start_rng']:
                raise AssertionError('Core/RNG anchor mismatch')
            condition, ref, encoding = encode_conditions(meta, train, seed, fold, config())
            for method in ('C1', 'C2', 'C3', 'C4', 'C5', 'C6'):
                saved = load_torch(HIST / f'runs/{method}/s{seed}_f{fold}.pt', mmap=True)
                if saved['state'] != 'complete' or saved['pairing'] != pairing:
                    raise AssertionError('Incomplete/unpaired historical checkpoint')
                if not np.array_equal(saved['encoding']['permutation'], encoding['permutation']) or not np.array_equal(np.array(saved['encoding']['reference'], dtype=np.float32), ref.numpy()):
                    raise AssertionError('Historical condition encoding mismatch')
                verify_oof(saved['oof'], ctx, split)
                if np.asarray(saved['frozen']['test_prob_asd']).shape != (5, len(test)):
                    raise AssertionError('Historical frozen evaluation incomplete')
            b = baseline[f'{seed}/{fold}']
            verify_oof(b, ctx, split)
    return {'created_utc': utc(), 'manifest': manifest, 'source_sha256': old['source_sha256'],
            'anchors': previous['audit']['anchors'], 'baseline': baseline,
            'data_digest': ctx['data_digest'], 'historical_fits_verified': 210,
            'github': {'reference': config()['github_reference'], 'pcci_files_byte_equal': True,
                       'official_reference': config()['official_reference'],
                       'official_only_change': 'CustomConv2d last 1x1 projection; audited by raw GitHub bytes before creation'},
            'handoff_before_sha256': sha_file(ROOT / 'handoff.md'),
            'environment': {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                            'host_threads': torch.get_num_threads(), 'cpu_budget': 16},
            'audit_notes': ['Native eval VAE/dropout randomness retained', 'RP forward .data updates and unregistered tensor retained',
                            'RFE reused, with known historical cross-fold leakage', 'No historical runner imported',
                            'Historical PCCI epoch curves sampled every 25 epochs; no interpolation']}


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


class Loaded:
    def __init__(self, seed, fold, state):
        self.seed, self.fold, self.state = seed, fold, state
        self.cfg, self.opt = config(), options(seed)
        self.ctx = load_torch(OLD / 'artifacts/context.pt')
        self.meta = metadata(self.ctx)
        self.split = load_torch(anchor_dir(seed, fold) / 'splits.pt')
        self.start = load_torch(anchor_dir(seed, fold) / 'train_start.pt')
        self.pairing = state['audit']['anchors'][f'{seed}/{fold}']['pairing']
        for name, expected in state['audit']['anchors'][f'{seed}/{fold}']['input_sha256'].items():
            if sha_file(ROOT / name) != expected:
                raise AssertionError(f'Anchor file changed: {name}')
        self.core = MM_GTUNets(self.opt, fold).to(self.opt.device)
        self.core.vae_save_path = str(anchor_dir(seed, fold).parent / f'vae/fold{fold}_pretrain.pth')
        import_core(self.core, self.start['core'])
        self.x_cpu = load_torch(anchor_dir(seed, fold).parent / 'selected_features.pt')
        affinity = create_reward_penalty_graph(self.ctx['ph_dict'], self.ctx['labels'],
                                               *[self.split[k] for k in SUBSETS], self.opt)
        self.ph_cpu = torch.from_numpy(self.ctx['ph_data']).float()
        self.affinity_cpu = torch.from_numpy(affinity).float()
        self.x, self.ph, self.affinity = [v.to(self.opt.device) for v in (self.x_cpu, self.ph_cpu, self.affinity_cpu)]
        self.labels = torch.as_tensor(self.ctx['labels'], dtype=torch.long, device=self.opt.device)
        self.train = torch.as_tensor(self.split['train'], dtype=torch.long, device=self.opt.device)
        self.conditions, self.reference, self.encoding = encode_conditions(self.meta, self.split['train'], seed, fold, self.cfg)
        self.conditions, self.reference = self.conditions.to(self.opt.device), self.reference.to(self.opt.device)
        self.permutation = torch.as_tensor(self.encoding['permutation'], device=self.opt.device)
        self.module = self.bridge = self.frozen = None
        self.classifier_calls = 0
        self.call_handle = self.core.clf.register_forward_hook(self._classifier)
        self.group = None

    def _classifier(self, module, inputs, output):
        self.classifier_calls += 1

    def attach(self, group):
        if self.bridge is not None:
            self.bridge.close()
        self.group = group
        spec = self.cfg['groups'][group]
        self.module = None
        if spec['adapter'] is not None:
            self.module = PCCI.isolated(self.cfg, private_seed(self.seed, self.fold, 'PCCI-initialization'), self.opt.device)
        self.bridge = Bridge(self.core, self.module, self.conditions, self.reference, self.permutation,
                             spec['adapter'], spec['inert'], spec['detach'])
        if spec['frozen']:
            checkpoint = load_torch(ROOT / self.state['audit']['baseline'][f'{self.seed}/{self.fold}']['best_reference'])
            import_core(self.core, checkpoint['core'])
            self.frozen = FrozenCore(self.core)

    def forward(self, cpu_inputs=False):
        if self.frozen is not None:
            self.frozen.check()
        calls = self.classifier_calls
        result = self.core(*((self.x_cpu, self.ph_cpu, self.affinity_cpu) if cpu_inputs else (self.x, self.ph, self.affinity)))
        if self.classifier_calls != calls + 1:
            raise AssertionError('Classifier execution count is not one')
        if self.module is None:
            self.bridge.base = self.core.joint_embed
        elif self.cfg['groups'][self.group]['inert'] and bool(torch.count_nonzero(self.bridge.last['delta'])):
            raise AssertionError('R1 residual is nonzero')
        if self.frozen is not None:
            self.frozen.check(restore=True)
        return result

    def optimizer(self):
        params = (list(self.module.parameters()) if self.frozen is not None else
                  list(self.core.parameters()) + (list(self.module.parameters()) if self.module is not None else []))
        return torch.optim.Adam(params, lr=self.cfg['lr'], weight_decay=self.cfg['weight_decay'])

    def mode(self, train):
        if self.frozen is None:
            self.core.train(train)
        else:
            self.core.eval()
        if self.module is not None:
            self.module.train(train)

    def close(self):
        if self.bridge is not None:
            self.bridge.close()
        self.call_handle.remove()
        self.__dict__.clear()
        torch.cuda.empty_cache()


@contextmanager
def task_lock(group, seed, fold, extra=0):
    offset = extra + (int(group[1:]) * 30 if group in GROUPS else 0) + config()['seeds'].index(seed) * 10 + fold
    with (BASE / 'run.py').open('r+') as stream:
        fcntl.lockf(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB, 1, offset)
        try:
            yield
        finally:
            fcntl.lockf(stream.fileno(), fcntl.LOCK_UN, 1, offset)


def cpu_state(module):
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()} if module is not None else None


def grad_norm(parameters):
    values = [p.grad.detach().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0


def gradient_stats(loaded):
    answer = {name: grad_norm(module.parameters()) for name, module in
              (('imaging', loaded.core.img_unet), ('phenotype', loaded.core.ph_unet), ('classifier', loaded.core.clf))}
    answer['pcci'] = grad_norm(loaded.module.parameters()) if loaded.module is not None else 0.0
    return answer


def early_trace(loaded, optimizer, logits, ce, graph):
    gradients = {n: p.grad for n, p in loaded.core.named_parameters() if p.grad is not None}
    return {'logits_sha256': array_hash(logits), 'train_ce': float(ce.detach()), 'train_graph': float(graph.detach()),
            'core_gradients_sha256': state_hash(gradients), 'gradient_norms': gradient_stats(loaded),
            'bn': {k: array_hash(v) for k, v in loaded.core.clf[2].state_dict().items()},
            'topk_sets': [[array_hash(torch.sort(p).values) for p in branch.perms]
                          for branch in (loaded.core.img_unet, loaded.core.ph_unet)],
            'rng_after_train': rng_hash(capture_rng())}


def oof_record(loaded, logits):
    score = logits.detach().cpu().numpy()
    test = loaded.split['test']
    pred, p = score.argmax(1)[test], softmax(score, axis=1)[test, 0]
    return {'indices': test.tolist(), 'labels': np.asarray(loaded.ctx['labels'])[test].tolist(),
            'pred': pred.tolist(), 'prob_asd': p.tolist(), 'logits': score[test].tolist(),
            'ids': [loaded.ctx['ids'][i] for i in test], 'sites': [loaded.ctx['site_names'][i] for i in test],
            'metrics': metrics(np.asarray(loaded.ctx['labels'])[test], pred, p)}


def parameter_changes(current, initial):
    answer = {}
    for prefix in ('img_unet.', 'ph_unet.', 'mm_attention.', 'clf.', 'rp_attention.'):
        pairs = [(current[k].float(), initial[k].float()) for k in current if k.startswith(prefix) and torch.is_floating_point(current[k])]
        distance = sum(float((a-b).double().square().sum()) for a,b in pairs) ** .5
        norm = sum(float(b.double().square().sum()) for _,b in pairs) ** .5
        answer[prefix.rstrip('.')] = {'l2_change': distance, 'relative_l2_change': distance / max(norm, 1e-30)}
    return answer


def train_one(group, seed, fold, state):
    if verify_result(group, seed, fold, state):
        return {'skipped': True}
    loaded = Loaded(seed, fold, state)
    try:
        loaded.attach(group)
        core, module = loaded.core, loaded.module
        initial_core, initial_module = export_core(core), cpu_state(module)
        optimizer = loaded.optimizer()
        core_order = [n for n, p in core.named_parameters()]
        stopper = EarlyStopping(patience=loaded.cfg['early_stop'], verbose=False)
        best, history, best_loss = None, [], 1e10
        restore_rng(loaded.start['rng'])
        if rng_hash(capture_rng()) != loaded.pairing['classifier_start_rng']:
            raise AssertionError('Training start RNG differs')
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        bn_start = int(core.clf[2].num_batches_tracked)
        for epoch in range(loaded.cfg['epochs']):
            loaded.mode(True)
            optimizer.zero_grad()
            logits, graph = loaded.forward()
            ce = F.cross_entropy(logits[loaded.split['train']], loaded.labels[loaded.split['train']])
            total = ce + graph
            if not bool(torch.isfinite(total)):
                raise AssertionError('Nonfinite training loss')
            diagnostic = epoch % loaded.cfg['diagnostic_frequency'] == 0
            if diagnostic and loaded.frozen is None:
                loaded.bridge.imaging.retain_grad()
                loaded.bridge.phenotype.retain_grad()
            total.backward()
            record = {'epoch': epoch, 'train_ce': float(ce.detach()), 'train_graph': float(graph.detach()),
                      'train_total': float(total.detach())}
            if diagnostic or epoch < 3:
                record['gradient_norms'] = gradient_stats(loaded)
                base = loaded.bridge.base.detach()
                delta = loaded.bridge.last['delta'].detach() if loaded.bridge.last is not None else torch.zeros_like(base)
                record['joint_l2'] = float(base.norm(dim=1).mean())
                record['residual_l2'] = float(delta.norm(dim=1).mean())
            if epoch < 3:
                record['early_trace'] = early_trace(loaded, optimizer, logits, ce, graph)
            if group == 'R1' and any(p.grad is not None for p in module.parameters()):
                raise AssertionError('Inert module received gradient')
            if group == 'R3' and any(p.grad is not None for p in core.parameters()):
                raise AssertionError('Frozen core received gradient')
            optimizer.step()
            if loaded.frozen is not None:
                loaded.frozen.check()
            expected_bn = bn_start + (epoch + 1 if loaded.frozen is None else 0)
            if int(core.clf[2].num_batches_tracked) != expected_bn:
                raise AssertionError('BN update count mismatch')
            loaded.mode(False)
            with torch.no_grad():
                val_logits, val_graph = loaded.forward()
            val_ce = F.cross_entropy(val_logits[loaded.split['val']], loaded.labels[loaded.split['val']])
            val_loss = val_ce + val_graph
            if not bool(torch.isfinite(val_loss)):
                raise AssertionError('Nonfinite validation loss')
            record.update(val_ce=float(val_ce), val_graph=float(val_graph), val_loss=float(val_loss),
                          bn_batches=expected_bn, rng_after_val=rng_hash(capture_rng()))
            history.append(record)
            if float(val_loss) < best_loss:
                best_loss = float(val_loss)
                best = {'core': export_core(core), 'pcci': cpu_state(module), 'epoch': epoch,
                        'validation_loss': best_loss, 'training_record': record.copy()}
            stopper(val_loss, core)
            if epoch % 100 == 0:
                emit('progress', group=group, seed=seed, fold=fold, epoch=epoch, validation_loss=float(val_loss))
            if stopper.early_stop:
                break
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        import_core(core, best['core'])
        if module is not None:
            module.load_state_dict(best['pcci'])
        loaded.mode(False)
        rng = capture_rng()
        with torch.set_grad_enabled(True):
            logits, _ = loaded.forward()
        oof = oof_record(loaded, logits)
        result = {'state': 'trained', 'group': group, 'seed': seed, 'fold': fold, 'completed_utc': utc(),
                  'training_code_sha256': code_hash(), 'config_digest': digest(config()), 'pairing': loaded.pairing,
                  'input_sha256': state['audit']['anchors'][f'{seed}/{fold}']['input_sha256'],
                  'encoding': loaded.encoding, 'core': best['core'], 'pcci': best['pcci'],
                  'best_core_sha256': core_hash(best['core']),
                  'best_pcci_sha256': state_hash(best['pcci']) if module is not None else None,
                  'initial_pcci_sha256': state_hash(initial_module) if module is not None else None,
                  'best_epoch': best['epoch'], 'best_validation_loss': best_loss,
                  'best_training_record': best['training_record'], 'epochs_run': epoch + 1,
                  'epochs': history, 'classifier_seconds': seconds, 'oof': oof,
                  'formal_rng_before': rng_hash(rng), 'formal_rng_after': rng_hash(capture_rng()),
                  'formal_full_rng_before': rng, 'physical_gpu': int(os.environ.get('CUDA_VISIBLE_DEVICES', '0')),
                  'device': torch.cuda.get_device_name(0), 'peak_memory_bytes': torch.cuda.max_memory_allocated(),
                  'optimizer_core_parameter_order': core_order,
                  'parameter_changes': parameter_changes(best['core']['state'], initial_core['state']),
                  'parameters': {'core': sum(p.numel() for p in core.parameters()),
                                 'adapter': sum(p.numel() for p in module.parameters()) if module is not None else 0},
                  'frozen_guard': loaded.frozen.statistics() if loaded.frozen else None,
                  'r1_module_unchanged': state_hash(initial_module) == state_hash(best['pcci']) if group == 'R1' else None,
                  'r3_two_stage_diagnostic': group == 'R3'}
        if group == 'R3' and core_hash(best['core']) != core_hash(initial_core):
            raise AssertionError('R3 core did not remain frozen')
        atomic_torch(result, result_path(group, seed, fold))
        verify_result(group, seed, fold, state)
        return {'metrics': oof['metrics'], 'best_epoch': best['epoch'], 'seconds': seconds}
    finally:
        loaded.close()


def verify_result(group, seed, fold, state, complete=False):
    path = result_path(group, seed, fold)
    if not path.exists():
        return False
    saved = load_torch(path, mmap=True)
    expected = {'group': group, 'seed': seed, 'fold': fold, 'training_code_sha256': state['training_code_sha256'],
                'config_digest': digest(config()), 'input_sha256': state['audit']['anchors'][f'{seed}/{fold}']['input_sha256']}
    if any(saved.get(k) != v for k,v in expected.items()):
        raise AssertionError(f'Result fingerprint mismatch: {path}')
    if saved['state'] not in ('trained', 'complete'):
        return False
    if core_hash(saved['core']) != saved['best_core_sha256']:
        raise AssertionError('Core checkpoint checksum mismatch')
    if saved['pcci'] is not None and state_hash(saved['pcci']) != saved['best_pcci_sha256']:
        raise AssertionError('PCCI checkpoint checksum mismatch')
    for tensor in list(saved['core']['state'].values()) + list((saved['pcci'] or {}).values()):
        if not bool(torch.isfinite(tensor).all()):
            raise AssertionError('Nonfinite checkpoint')
    ctx = load_torch(OLD / 'artifacts/context.pt')
    split = load_torch(anchor_dir(seed, fold) / 'splits.pt')
    verify_oof(saved['oof'], ctx, split)
    if len(saved['epochs']) != saved['epochs_run'] or saved['epochs'][saved['best_epoch']]['val_loss'] != saved['best_validation_loss']:
        raise AssertionError('Epoch/checkpoint record mismatch')
    if complete:
        if saved['state'] != 'complete' or 'frozen' not in saved:
            return False
        frozen = saved['frozen']
        p, pred = np.asarray(frozen['test_prob_asd']), np.asarray(frozen['test_pred'])
        if p.shape != (5, len(split['test'])) or not np.isfinite(p).all():
            raise AssertionError('Incomplete frozen draws')
        for d in range(5):
            m = metrics(saved['oof']['labels'], pred[d], p[d])
            if any(abs(m[k] - frozen['draw_metrics'][d][k]) > 1e-12 for k in KEYS):
                raise AssertionError('Frozen metric recomputation mismatch')
    return True


def compare(a, b, label, atol=3e-6, rtol=3e-5):
    if a is None or b is None:
        if a is not None or b is not None:
            raise AssertionError(f'Connectivity mismatch {label}')
        return 0.0
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(f'{label}: max error {float((a-b).abs().max())}')
    return float((a-b).abs().max())


def check_one(seed, fold, state):
    loaded = Loaded(seed, fold, state)
    native = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        core = loaded.core
        trajectories = {}
        for group in ('historical_E0_reference', 'R0', 'R1'):
            if loaded.bridge is not None:
                loaded.bridge.close()
            import_core(core, loaded.start['core'])
            loaded.frozen = None
            loaded.attach('R0' if group == 'historical_E0_reference' else group)
            before_module_rng = rng_hash(capture_rng())
            optimizer = loaded.optimizer()
            restore_rng(loaded.start['rng'])
            steps = []
            for step in range(config()['equivalence_steps']):
                core.train()
                optimizer.zero_grad()
                logits, graph = loaded.forward(cpu_inputs=group == 'historical_E0_reference')
                ce = F.cross_entropy(logits[loaded.split['train']], loaded.labels[loaded.split['train']])
                if group == 'historical_E0_reference' and step % 25 == 0:
                    loaded.bridge.imaging.retain_grad()
                    loaded.bridge.phenotype.retain_grad()
                (ce + graph).backward()
                grads = {n: p.grad.detach().cpu().clone() if p.grad is not None else None for n,p in core.named_parameters()}
                pools = [p.detach().cpu().clone() for b in (core.img_unet,core.ph_unet) for p in b.perms]
                optimizer.step()
                adam = {n: {k:v.detach().cpu().clone() for k,v in optimizer.state[p].items() if torch.is_tensor(v)}
                        for n,p in core.named_parameters() if p in optimizer.state}
                core.eval()
                with torch.no_grad():
                    vo, vg = loaded.forward(cpu_inputs=group == 'historical_E0_reference')
                vc = F.cross_entropy(vo[loaded.split['val']], loaded.labels[loaded.split['val']])
                steps.append({'logits':logits.detach().cpu(), 'ce':ce.detach().cpu(), 'graph':graph.detach().cpu(),
                              'val_logits':vo.cpu(), 'val_loss':(vc+vg).cpu(), 'gradients':grads,
                              'adam':adam,'core':export_core(core),'pools':pools,'rng':rng_hash(capture_rng())})
                if group == 'R1' and any(p.grad is not None for p in loaded.module.parameters()):
                    raise AssertionError('R1 is not inert')
            trajectories[group] = steps
        errors = {}
        for group in ('R0','R1'):
            maximum = 0.0
            for a,b in zip(trajectories['historical_E0_reference'],trajectories[group]):
                for k in ('logits','ce','graph','val_logits','val_loss'):
                    maximum = max(maximum,compare(a[k],b[k],f'{group}/{k}'))
                for name in a['gradients']:
                    maximum = max(maximum,compare(a['gradients'][name],b['gradients'][name],name))
                for name in a['adam']:
                    for k in a['adam'][name]:
                        maximum = max(maximum,compare(a['adam'][name][k],b['adam'][name][k],f'Adam/{name}/{k}'))
                if core_hash(a['core']) != core_hash(b['core']) or a['rng'] != b['rng']:
                    raise AssertionError('Deterministic parameter/RNG trajectory diverged')
                if any(not torch.equal(torch.sort(x).values,torch.sort(y).values) for x,y in zip(a['pools'],b['pools'])):
                    raise AssertionError('TopK selected set differs')
            errors[group] = maximum
        del trajectories
        loaded.bridge.close()
        import_core(core, loaded.start['core'])
        loaded.attach('R2')
        restore_rng(loaded.start['rng'])
        core.train()
        optimizer = loaded.optimizer()
        optimizer.zero_grad()
        logits, graph = loaded.forward()
        ce = F.cross_entropy(logits[loaded.train],loaded.labels[loaded.train])
        (ce+graph).backward()
        optimizer.step()
        optimizer.zero_grad()
        logits, graph = loaded.forward()
        ce = F.cross_entropy(logits[loaded.train],loaded.labels[loaded.train])
        direct, = torch.autograd.grad(loaded.bridge.last['delta'].sum(), loaded.bridge.imaging,allow_unused=True,retain_graph=True)
        if direct is not None and bool(torch.count_nonzero(direct)):
            raise AssertionError('R2 direct gradient survived detach')
        original, = torch.autograd.grad(ce,loaded.bridge.imaging,retain_graph=True)
        if float(original.norm()) <= 0:
            raise AssertionError('Original imaging path was detached')
        (ce+graph).backward()
        norms = gradient_stats(loaded)
        if any(norms[k] <= 0 for k in ('imaging','phenotype','classifier','pcci')):
            raise AssertionError('R2 unintended gradient disconnection')
        loaded.bridge.close()
        loaded.frozen = None
        loaded.attach('R3')
        frozen_start = core_hash(export_core(core))
        restore_rng(loaded.start['rng'])
        draw_rng(seed,fold,0)
        rng = capture_rng()
        loaded.bridge.enabled = False
        with torch.no_grad():
            plain,_=loaded.forward()
        restore_rng(rng)
        loaded.bridge.enabled = True
        with torch.no_grad():
            zero,_=loaded.forward()
        zero_error=compare(plain,zero,'R3 zero residual')
        old=json.loads((HIST/'summary.json').read_text())['baseline_frozen'][f'{seed}/{fold}']
        p=softmax(zero.cpu().numpy(),axis=1)[loaded.split['test'],0]
        old_error=float(np.max(np.abs(p-np.asarray(old['test_prob_asd'][0]))))
        if old_error>2e-5:
            raise AssertionError(f'R3 zero does not replay frozen C0: {old_error}')
        optimizer=loaded.optimizer()
        adapter_before=state_hash(loaded.module.state_dict())
        for _ in range(3):
            loaded.mode(True)
            optimizer.zero_grad()
            logits,graph=loaded.forward()
            (F.cross_entropy(logits[loaded.train],loaded.labels[loaded.train])+graph).backward()
            if grad_norm(loaded.module.parameters())<=0:
                raise AssertionError('Frozen adapter has no gradients')
            optimizer.step()
            loaded.frozen.check()
        if core_hash(export_core(core))!=frozen_start or state_hash(loaded.module.state_dict())==adapter_before:
            raise AssertionError('R3 freeze/update check failed')
        return {'passed':True,'seed':seed,'fold':fold,'steps':config()['equivalence_steps'],
                'deterministic_max_error':errors,'r2_gradient_norms':norms,'r2_direct_gradient_zero':True,
                'r3_zero_max_logit_error':zero_error,'r3_vs_saved_C0_max_prob_error':old_error,
                'r3_guard':loaded.frozen.statistics(),'classifier_calls_per_forward':1,
                'BN_Adam_RNG_TopK_core_gradients_checked':True}
    finally:
        torch.use_deterministic_algorithms(native)
        loaded.close()


def frozen_one(group,seed,fold,state):
    if verify_result(group,seed,fold,state,complete=True):
        return {'skipped':True}
    saved=load_torch(result_path(group,seed,fold))
    loaded=Loaded(seed,fold,state)
    try:
        loaded.attach(group)
        core=loaded.core
        core.eval()
        if loaded.module is not None:
            loaded.module.load_state_dict(saved['pcci'])
            loaded.module.eval()
        test=loaded.split['test']
        probabilities,predictions,draw_metrics,seeds=[],[],[],[]
        counter={k:[] for k in config()['counterfactuals']}
        residuals,base_prob=[],[]
        for draw in range(5):
            import_core(core,saved['core'])
            if loaded.frozen:
                loaded.frozen.check()
            restore_rng(loaded.start['rng'])
            seeds.append(draw_rng(seed,fold,draw))
            with torch.no_grad():
                logits,_=loaded.forward()
                scores=logits.cpu().numpy()
                p,pred=softmax(scores,axis=1)[:,0],scores.argmax(1)
                probabilities.append(p[test]);predictions.append(pred[test])
                draw_metrics.append(metrics(np.asarray(loaded.ctx['labels'])[test],pred[test],p[test]))
                if loaded.module is not None:
                    loaded.bridge.enabled=False
                    try:
                        for intervention in config()['counterfactuals']:
                            value=loaded.module.components(loaded.bridge.imaging,loaded.conditions,loaded.reference,
                                                           config()['groups'][group]['adapter'],intervention,loaded.permutation)
                            delta=value['delta'] if group!='R1' else torch.zeros_like(value['delta'])
                            alt=core.clf(loaded.bridge.base+delta).cpu().numpy()
                            counter[intervention].append(softmax(alt,axis=1)[:,0])
                            if intervention=='true': residuals.append(value['delta'].norm(dim=1).cpu().numpy())
                    finally:
                        loaded.bridge.enabled=True
                    base_prob.append(counter['off'][-1][test])
                else:
                    base_prob.append(p[test])
        probability=np.stack(probabilities);prediction=np.stack(predictions);mean=probability.mean(0)
        frozen={'draw_seeds':seeds,'test_prob_asd':probability,'test_pred':prediction,'draw_metrics':draw_metrics,
                'mean_probability_metrics':metrics(np.asarray(loaded.ctx['labels'])[test],np.where(mean>=.5,0,1),mean),
                'test_prediction_instability_fraction':float(np.any(prediction!=prediction[0],axis=0).mean()),
                'base_test_prob_asd':np.stack(base_prob),
                'counterfactual_prob_asd':{k:np.stack(v) for k,v in counter.items() if v},
                'residual_l2':np.stack(residuals) if residuals else None,
                'completed_utc':utc()}
        saved['frozen'],saved['state']=frozen,'complete'
        if loaded.frozen: saved['frozen_guard_evaluation']=loaded.frozen.statistics()
        atomic_torch(saved,result_path(group,seed,fold))
        verify_result(group,seed,fold,state,complete=True)
        return {'mean_probability_metrics':frozen['mean_probability_metrics']}
    finally:
        loaded.close()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('check','train','frozen','diagnose'))
    parser.add_argument('--group',default='R0')
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--fold',type=int,required=True)
    args=parser.parse_args()
    state=guard()
    require_ppu()
    with task_lock(args.group,args.seed,args.fold,extra=200 if args.action=='check' else 0):
        emit('begin',action=args.action,group=args.group,seed=args.seed,fold=args.fold)
        if args.action=='check':
            result=check_one(args.seed,args.fold,state)
        elif args.action=='train':
            result=train_one(args.group,args.seed,args.fold,state)
        elif args.action=='frozen':
            result=frozen_one(args.group,args.seed,args.fold,state)
        else:
            import analyze
            result=analyze.diagnose_one(args.group,args.seed,args.fold,state)
        emit('done',action=args.action,group=args.group,seed=args.seed,fold=args.fold,result=result)


if __name__=='__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.exit(1)
