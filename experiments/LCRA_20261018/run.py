"""Independent LCRA historical loader, acceptance, development and formal runner."""
from __future__ import annotations
import argparse
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
import resource
import subprocess
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
CFG = json.loads((BASE / 'config.json').read_text())
ROOT = Path(CFG['root'])
OLD = ROOT / CFG['history']
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BASE))
import numpy as np
from scipy.special import softmax
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
import torch
from torch.nn import functional as F
from opt import OptInit
from model.mm_gtunets import MM_GTUNets
from model.rp_graph import create_reward_penalty_graph
from utils.tools import EarlyStopping
from lcra import LCRA, LCRAWrapper, prepare_inputs, phenotype_kernel, permutation, js_divergence, stable_seed
warnings.filterwarnings('ignore', message='To copy construct from a tensor')
warnings.filterwarnings('ignore', message='Converting a tensor with requires_grad=True')
warnings.filterwarnings('ignore', message='Please use the new API settings to control TF32 behavior')
warnings.filterwarnings('ignore', message='Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD detected')
torch.set_num_threads(1)
GROUPS = tuple(CFG['groups'])
PARTS = ('train', 'val', 'test')
METRICS = ('ACC', 'BA', 'AUC', 'SEN', 'SPE', 'F1')
SOURCE_FILES = ('opt.py','train_mm_gtunets.py','model/mm_gtunets.py','model/gtunet.py',
                'model/rp_graph.py','utils/tools.py','utils/mydataloader.py','utils/metrics.py')

def utc():
    return datetime.now(timezone.utc).isoformat()

def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()).hexdigest()

def array_hash(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    value = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(value.dtype).encode()); h.update(str(value.shape).encode()); h.update(value.tobytes())
    return h.hexdigest()

def state_hash(state):
    h = hashlib.sha256()
    for key in sorted(state):
        h.update(key.encode()); h.update(array_hash(state[key]).encode())
    return h.hexdigest()

def core_hash(saved):
    return state_hash({**saved['state'], 'unregistered_rp_weights': saved['rp_weights']})

def cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def export_core(core):
    return {'state': cpu_state(core), 'rp_weights': core.rp_attention.weights.detach().cpu().clone()}

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
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu']); torch.cuda.set_rng_state(state['cuda'], 0)
    b = state['backend']
    torch.backends.cudnn.deterministic = b['cudnn_deterministic']
    torch.backends.cudnn.benchmark = b['cudnn_benchmark']
    torch.backends.cuda.matmul.allow_tf32 = b['matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32 = b['cudnn_allow_tf32']

def rng_hash(state):
    return {'python': hashlib.sha256(repr(state['python']).encode()).hexdigest(),
        'numpy': hashlib.sha256(repr(state['numpy'][0]).encode() + state['numpy'][1].tobytes()
                               + repr(state['numpy'][2:]).encode()).hexdigest(),
        'cpu': array_hash(state['cpu']), 'cuda': array_hash(state['cuda'])}

def atomic_json(value, path):
    atomic_bytes((json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n').encode(), path)

def atomic_bytes(data, path):
    path = Path(path)
    assert path.is_relative_to(BASE), f'Unauthorized output: {path}'
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with tmp.open('wb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)

def atomic_torch(value, path):
    path = Path(path)
    assert path.is_relative_to(BASE)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with tmp.open('wb') as stream:
        torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)

def load_torch(path):
    return torch.load(path, map_location='cpu', weights_only=False, mmap=True)

def emit(kind, **values):
    print(json.dumps({'kind': kind, 'utc': utc(), **values}, allow_nan=False, ensure_ascii=False), flush=True)

def log(event, **values):
    value = {'utc': utc(), 'event': event, **values}
    with (BASE / 'run.log').open('a') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n'); stream.flush()
    return value

def anchor_dir(seed, fold):
    return OLD / 'artifacts' / f'seed_{seed}' / f'fold_{fold}'

def metrics(y, probability, pred=None):
    y, probability = np.asarray(y), np.asarray(probability, np.float64)
    assert np.isfinite(probability).all() and np.all((probability >= 0) & (probability <= 1))
    asd = y == 0
    predicted = probability >= .5 if pred is None else np.asarray(pred) == 0
    tp = int(np.sum(asd & predicted)); fn = int(np.sum(asd & ~predicted))
    tn = int(np.sum(~asd & ~predicted)); fp = int(np.sum(~asd & predicted))
    sen, spe = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    return {'ACC': float(np.mean(asd == predicted)), 'BA': (sen + spe) / 2,
        'AUC': float(roc_auc_score(asd, probability)) if np.unique(y).size == 2 else None,
        'SEN': sen, 'SPE': spe, 'F1': 2 * tp / max(2 * tp + fp + fn, 1)}

def ce_values(p, y):
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
    return -np.log(np.where(np.asarray(y) == 0, p, 1 - p))

def describe(a):
    a = np.asarray(a, dtype=np.float64).ravel()
    assert len(a) and np.isfinite(a).all()
    return {'n': len(a), 'mean': float(a.mean()), 'sd': float(a.std(ddof=0)),
        'p10': float(np.quantile(a, .1)), 'p50': float(np.quantile(a, .5)),
        'p90': float(np.quantile(a, .9)), 'min': float(a.min()), 'max': float(a.max())}

def effective_rank(a):
    a = np.asarray(a, np.float64); a = a - a.mean(0)
    eigen = np.maximum(np.linalg.eigvalsh(a.T @ a / max(len(a)-1, 1)), 0)
    if eigen.sum() == 0:
        return 0.0
    p = eigen[eigen > 0] / eigen.sum()
    return float(np.exp(-np.sum(p * np.log(p))))

def grad_norm(params):
    values = [p.grad.detach().square().sum() for p in params if p.grad is not None]
    return float(torch.stack(values).sum().sqrt()) if values else 0.0

def gradient_cosine(a, b):
    aa = sum(float(v.detach().double().square().sum()) for v in a if v is not None)
    bb = sum(float(v.detach().double().square().sum()) for v in b if v is not None)
    dot = sum(float((x.detach().double() * y.detach().double()).sum())
              for x, y in zip(a, b) if x is not None and y is not None)
    return dot / (aa * bb) ** .5 if aa > 0 and bb > 0 else None

def assert_exact(a, b, label):
    if a is None or b is None:
        assert a is b, label
    elif torch.is_tensor(a):
        assert a.shape == b.shape and torch.equal(a, b), f'{label}: max_abs={float((a-b).abs().max())}'
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b), label
    elif isinstance(a, dict):
        assert set(a) == set(b), label
        for k in a:
            assert_exact(a[k], b[k], label + '/' + str(k))
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b), label
        for i, (x, y) in enumerate(zip(a, b)):
            assert_exact(x, y, label + '/' + str(i))
    else:
        assert a == b, label


def code_hash():
    return digest({name: sha_file(BASE/name) for name in ('config.json','lcra.py','run.py')})


def source_hashes():
    return {name: sha_file(BASE/name) for name in ('config.json','lcra.py','run.py','launch.py','analyze.py','tests.py','README.md')}


def task_path(action, group, seed, fold):
    return BASE/'runs'/action/group/f's{seed}_f{fold}.pt'


@contextmanager
def task_lock(action, group, seed, fold):
    slot = {'check':0,'develop':200,'train':400,'frozen':600}[action] + GROUPS.index(group)*30 + CFG['seeds'].index(seed)*10 + fold
    with (BASE/'run.py').open('r+') as stream:
        fcntl.lockf(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB, 1, slot)
        try:
            yield
        finally:
            fcntl.lockf(stream.fileno(), fcntl.LOCK_UN, 1, slot)


def metadata(ctx):
    with (Path(CFG['data_folder'])/'Phenotypic_V1_0b_preprocessed1.csv').open() as stream:
        rows = {row['SUB_ID']:row for row in csv.DictReader(stream)}
    ordered = [rows[i] for i in ctx['ids']]
    labels = np.asarray([int(r['DX_GROUP'])-1 for r in ordered],np.int64)
    assert len(set(ctx['ids'])) == 871 and np.array_equal(labels,ctx['labels'])
    assert np.array_equal(np.bincount(labels),[403,468])
    def number(v, default):
        try: return float(v)
        except (ValueError,TypeError): return default
    sites = np.asarray([r['SITE_ID'] for r in ordered])
    assert np.array_equal(sites,ctx['site_names'])
    return {'age':np.asarray([number(r['AGE_AT_SCAN'],np.nan) for r in ordered]),
        'sex':np.asarray([number(r['SEX'],-1) for r in ordered]),'site':sites,'ids':ctx['ids']}


def audit():
    path = BASE/'audit.json'
    if path.exists():
        previous = json.loads(path.read_text())
        assert previous['code_sha256'] == code_hash(), 'Frozen code changed; require explicit reacceptance'
        for name, expected in previous['source_sha256'].items():
            assert sha_file(ROOT/name) == expected
        return previous
    started = time.perf_counter()
    e7 = Path(CFG['experiment7'])
    prior = json.loads((e7/'audit.json').read_text())
    manifest = {}
    for name, row in prior['manifest'].items():
        if Path(name) in (ROOT/'handoff.md',ROOT/'prompt'):
            continue
        actual = sha_file(name)
        assert actual == row['sha256'], f'Protected historical file changed: {name}'
        manifest[name] = {'sha256':actual,'bytes':Path(name).stat().st_size}
    for directory in (e7,Path(CFG['experiment4']),ROOT/'experiments/PCND_DualStage_20261016'):
        for source in directory.iterdir():
            if source.suffix in ('.py','.md','.json'):
                manifest[str(source)] = {'sha256':sha_file(source),'bytes':source.stat().st_size}
    assert sha_file(ROOT/'prompt') == CFG['storage_gate']['prompt_sha256']
    ctx = load_torch(OLD/'artifacts/context.pt')
    meta = metadata(ctx)
    anchors = {}
    for seed in CFG['seeds']:
        test_seen = []
        x = load_torch(anchor_dir(seed,0).parent/'selected_features.pt')
        assert x.shape == (871,500)
        for fold in range(CFG['folds']):
            adir = anchor_dir(seed,fold)
            pair = json.loads((adir/'fingerprints.json').read_text())
            split = load_torch(adir/'splits.pt')
            assert np.array_equal(np.sort(np.concatenate([split[p] for p in PARTS])),np.arange(871))
            assert all(array_hash(split[p]) == pair['split_fingerprints'][p] for p in PARTS)
            test_seen.extend(split['test'].tolist())
            start = load_torch(adir/'train_start.pt')
            assert core_hash(start['core']) == pair['train_start_core_fingerprint']
            assert rng_hash(start['rng']) == pair['classifier_start_rng']
            assert array_hash(x) == pair['features_fingerprint']
            assert sha_file(adir.parent/'vae'/f'fold{fold}_pretrain.pth') == pair['vae_sha256']
            inputs = prepare_inputs(x.numpy(),meta,split,seed,fold,CFG)
            # Mutating held-out covariates must not alter train fits, candidates or null kernels.
            held = np.concatenate([split['val'],split['test']])
            changed_meta = {k:v.copy() if isinstance(v,np.ndarray) else v for k,v in meta.items()}
            changed_meta['age'][held] += 1000
            changed_meta['sex'][held] = -1
            changed_x = x.numpy().copy(); changed_x[held] += 100
            changed = prepare_inputs(changed_x,changed_meta,split,seed,fold,CFG)
            tr = split['train']
            for key in ('age_bandwidth','age_train_mean','age_train_sd','age_impute_train_median','train_feature_mean_sha256','train_feature_sd_sha256'):
                assert inputs['info'][key] == changed['info'][key], key
            assert np.array_equal(inputs['candidates'][tr],changed['candidates'][tr])
            for key in ('true','shuffled'):
                assert np.array_equal(inputs['null_kernels'][key],changed['null_kernels'][key])
                assert np.array_equal(inputs['kernels'][key][tr],changed['kernels'][key][tr])
            inherited = prior['anchors'][f'{seed}/{fold}']['input_sha256']
            for name, expected in inherited.items(): assert sha_file(ROOT/name) == expected
            anchors[f'{seed}/{fold}'] = {'pairing':pair,'input_sha256':inherited,
                'inputs':inputs['info'],'inputs_digest':digest(inputs['info']),
                'candidate_indices':inputs['candidates'].tolist(),
                'heldout_covariate_train_fit_invariance':True}
        assert np.array_equal(np.sort(test_seen),np.arange(871))
    official = {}
    for name in SOURCE_FILES:
        raw = urllib.request.urlopen('https://raw.githubusercontent.com/NZWANG/MM-GTUNets/'+CFG['official_commit']+'/'+name,timeout=60).read()
        local = (ROOT/name).read_bytes()
        expected = raw
        if name == 'model/mm_gtunets.py':
            expected = raw.replace(b'return super(CustomConv2d, self).forward(x)',
                b'# Equivalent 1x1 projection avoids the PPU Conv2d backward kernel failure.\n        return (x * self.weight.reshape(3, 1, 1)).sum(dim=-3, keepdim=True)')
        assert local == expected, f'Unexpected native source difference: {name}'
        official[name] = {'official_sha256':hashlib.sha256(raw).hexdigest(),'local_sha256':sha_file(ROOT/name),'same':local==raw}
    web = {}
    for url in (CFG['paper'],'https://api.github.com/repos/tangaii/MM/commits/main','https://api.github.com/repos/NZWANG/MM-GTUNets/commits/main'):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'LCRA-read-only-audit'}),timeout=60) as response:
                raw = response.read()
            web[url] = {'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
            if 'api.github.com' in url: web[url]['commit'] = json.loads(raw)['sha']
        except urllib.error.HTTPError as err:
            if 'api.github.com' not in url: raise
            repo = url.split('/repos/')[1].split('/commits/')[0]
            raw = subprocess.check_output(['git','ls-remote',f'https://github.com/{repo}.git','refs/heads/main'],timeout=60)
            web[url] = {'api_error':str(err),'fallback':'read-only git ls-remote; no fetch or repository writes',
                'commit':raw.decode().split()[0],'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
            log('github_api_fallback',url=url,error=str(err),method='git ls-remote')
    out = {'created_utc':utc(),'code_sha256':code_hash(),'new_source_sha256':source_hashes(),
        'source_sha256':{name:sha_file(ROOT/name) for name in SOURCE_FILES},'manifest':manifest,
        'anchors':anchors,'web_primary_sources':web,'official_source_comparison':official,
        'subject_ids_sha256':digest(ctx['ids']),'labels_sha256':array_hash(ctx['labels']),'label_counts':[403,468],
        'storage_gate':CFG['storage_gate'],'handoff_before':{'sha256':sha_file(ROOT/'handoff.md'),'bytes':(ROOT/'handoff.md').stat().st_size},
        'native_audit':[
            'RFE/VAE match dimensions, not an explicit patient-identity semantic objective.',
            'Both GTUNets output16D; ACMGL uses each output and arithmetic mean; classifier16->256/ReLU/BN256->2.',
            'Native similarity uses Euclidean cdist, unlike correlation wording in the paper.',
            'RP early loop return; shared constraint conv .data projection; RP weights unregistered on PPU; value and contribution tensors detach.',
            'VAE reload and sampling every forward; GTUNet eval dropout; native edge-dropout default probability preserved.',
            'Only insert continuous bounded graph update before once-only dualGTUNet; graph regularizer uses A_new.',
            'Keep transductive871 graph and historical cross-fold RFE leakage; no clinical generalization claim.',
            'E7 direct source available under /tmp; no claim that it is in GitHub.',
            'No new test metrics calculated/read by preflight; historical reports are contextual evidence only.'],
        'reading_paths':[str(ROOT/'handoff.md'),str(ROOT/'prompt')]+list(manifest),
        'elapsed_seconds':time.perf_counter()-started}
    atomic_json(out,path)
    log('audit_complete',protected_files=len(manifest),anchors=30,outer_test_performance_computed=False)
    return out


class Loaded:
    def __init__(self,seed,fold,audit_info):
        self.seed,self.fold = seed,fold
        if not torch.cuda.is_available(): raise RuntimeError('PPU unavailable; CPU training forbidden')
        torch.cuda.set_device(0)
        assert 'PPU' in torch.cuda.get_device_name(0).upper()
        self.device = torch.device('cuda:0')
        saved_argv = sys.argv
        try:
            sys.argv = ['LCRA-native-options']
            self.opt = OptInit(model='MM_GTUNets',dataset='ABIDE',atlas='aal').args
        finally: sys.argv = saved_argv
        opt = self.opt
        opt.device,opt.seed = self.device,seed
        opt.data_folder = CFG['data_folder']
        opt.ckpt_path = str(BASE/'never_train_vae')
        opt.epoch,opt.early_stop,opt.lr,opt.wd = CFG['epochs'],CFG['early_stop'],CFG['lr'],CFG['weight_decay']
        self.anchor = audit_info['anchors'][f'{seed}/{fold}']
        for name,expected in self.anchor['input_sha256'].items(): assert sha_file(ROOT/name) == expected
        self.ctx = load_torch(OLD/'artifacts/context.pt')
        self.meta = metadata(self.ctx)
        self.split = load_torch(anchor_dir(seed,fold)/'splits.pt')
        self.start = load_torch(anchor_dir(seed,fold)/'train_start.pt')
        self.x_cpu = load_torch(anchor_dir(seed,fold).parent/'selected_features.pt')
        self.ph_cpu = torch.from_numpy(self.ctx['ph_data']).float()
        self.affinity_cpu = torch.from_numpy(create_reward_penalty_graph(self.ctx['ph_dict'],self.ctx['labels'],*[self.split[k] for k in PARTS],opt)).float()
        self.core = MM_GTUNets(opt,fold).to(self.device)
        self.core.vae_save_path = str(anchor_dir(seed,fold).parent/'vae'/f'fold{fold}_pretrain.pth')
        import_core(self.core,self.start['core'])
        assert core_hash(export_core(self.core)) == self.anchor['pairing']['train_start_core_fingerprint']
        self.x,self.ph,self.affinity = [a.to(self.device) for a in (self.x_cpu,self.ph_cpu,self.affinity_cpu)]
        self.labels = torch.as_tensor(self.ctx['labels'],dtype=torch.long,device=self.device)
        self.train = torch.as_tensor(self.split['train'],dtype=torch.long,device=self.device)
        inputs = prepare_inputs(self.x_cpu.numpy(),self.meta,self.split,seed,fold,CFG)
        assert digest(inputs['info']) == self.anchor['inputs_digest']
        self.raw_inputs = inputs
        self.inputs = {'candidates':torch.as_tensor(inputs['candidates'],device=self.device),
            'raw_cos':torch.as_tensor(inputs['raw_cos'],device=self.device),
            'kernels':{k:torch.as_tensor(v,device=self.device) for k,v in inputs['kernels'].items()},
            'null_kernels':{k:torch.as_tensor(v,device=self.device) for k,v in inputs['null_kernels'].items()}}
        before = rng_hash(capture_rng())
        self.aligner = LCRA.isolated(CFG,seed,fold,self.device)
        assert before == rng_hash(capture_rng()), 'New initialization consumed native RNG'
        self.initial_aligner = cpu_state(self.aligner)
        self.wrapper = LCRAWrapper(self.core,self.aligner,CFG)
        self.data_digest = digest({'x':array_hash(self.x_cpu),'ph':array_hash(self.ph_cpu),'affinity':array_hash(self.affinity_cpu),
            'ids':self.ctx['ids'],'y':self.ctx['labels'].tolist(),'new_inputs':self.anchor['inputs_digest']})

    def reset(self):
        import_core(self.core,self.start['core'])
        self.aligner.load_state_dict(self.initial_aligner)
        restore_rng(self.start['rng'])

    def forward(self,group,alpha=None,condition=None):
        return self.wrapper(self.x,self.ph,self.affinity,self.inputs,self.train,group,alpha=alpha,condition=condition)

    def optimizer(self,new=True):
        params = list(self.core.parameters()) + (list(self.aligner.parameters()) if new else [])
        assert not any(p is self.core.rp_attention.weights for p in params)
        return torch.optim.Adam(params,lr=CFG['lr'],weight_decay=CFG['weight_decay'])


def graph_check(detail,alpha=None):
    alpha = CFG['alpha'] if alpha is None else alpha
    a,b,e = [detail[key].detach() for key in ('A_base','A_new','E')]
    assert all(torch.isfinite(v).all() for v in (a,b,e))
    assert torch.equal(e,e.T) and torch.equal(b,b.T)
    assert bool((a>=0).all() and (b>=0).all() and (e.abs()<=1).all())
    assert torch.equal(a.diag(),b.diag()) and torch.count_nonzero(e.diag()) == 0
    assert torch.equal(a==0,b==0)
    assert bool((b>=a*(1-alpha)).all() and (b<=a*(1+alpha)).all())
    if alpha == 0: assert torch.equal(a,b)


def diagnostic(loaded,detail):
    graph_check(detail)
    a,b = detail['A_base'].detach(),detail['A_new'].detach()
    change = (b-a).abs()
    tr = loaded.split['train']
    gap = (detail['J_null']-detail['J_true']).detach().cpu().numpy()
    p,q = detail['P'].detach(),detail['Q'].detach()
    site = loaded.meta['site']
    pair = torch.as_tensor(site[:,None]==site[None,:],device=a.device)
    out = {'relative_graph_l1':float(change.sum()/a.abs().sum()),
        'relative_graph_l1_by_partition':{part:float(change[idx].sum()/a[idx].abs().sum()) for part,idx in loaded.split.items()},
        'edge_relative_abs':describe((change[a>0]/a[a>0]).cpu().numpy()),
        'row_degree_relative':describe(((b.sum(1)-a.sum(1))/a.sum(1).clamp_min(1e-12)).cpu().numpy()),
        'within_site_relative_signed_change':float((b-a)[pair].sum()/a[pair].sum()),
        'cross_site_relative_signed_change':float((b-a)[~pair].sum()/a[~pair].sum()),
        'centered_score_max':float(detail['centered'].detach().abs().max()),
        'E_abs_mean':float(detail['E'].detach().abs().mean()),
        'P_entropy':float(-(p*p.clamp_min(1e-8).log()).sum(1).mean()),
        'Q_entropy':float(-(q*q.clamp_min(1e-8).log()).sum(1).mean()),
        'max_entropy':float(np.log(p.shape[-1])), 'image_effective_rank':effective_rank(detail['Z'][loaded.train].detach().cpu().numpy()),
        'true_vs_null_JS_gap':describe(gap),'train_true_js':float(detail['J_true'].detach().mean()),
        'train_null_js':float(detail['J_null'].detach().mean()),'local_loss':float(detail['L_local'].detach()),
        'true_minus_null_local_affinity':describe(detail['raw_affinity_gap'].detach().cpu().numpy()),
        'bn_batches_tracked':int(loaded.core.clf[2].num_batches_tracked),
        'bn_running_mean_l2':float(loaded.core.clf[2].running_mean.norm()),
        'bn_running_var_mean':float(loaded.core.clf[2].running_var.mean())}
    age,sex = loaded.meta['age'][tr],loaded.meta['sex'][tr]
    subgroups = {'site/'+str(s):site[tr]==s for s in sorted(set(site[tr]))}
    subgroups.update({'sex/'+str(s):sex==s for s in sorted(set(sex))})
    for low,high in ((0,12),(12,18),(18,30),(30,200)):
        subgroups[f'age/{low}-{high}'] = (age>=low)&(age<high)
    affinity = detail['raw_affinity_gap'].detach().cpu().numpy()
    out['train_strata'] = {name:{'n':int(mask.sum()),'JS_gap':float(gap[mask].mean()),'affinity_gap':float(affinity[mask].mean())}
        for name,mask in subgroups.items() if mask.any()}
    return out


def gradients(loaded,native,local):
    relation = list(loaded.aligner.parameters())
    projection = list(loaded.aligner.image.parameters())
    def norm(loss,params):
        g = torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        assert all(v is None or torch.isfinite(v).all() for v in g)
        return float(torch.stack([v.detach().square().sum() for v in g if v is not None]).sum().sqrt()) if any(v is not None for v in g) else 0.0
    return {'grad_from_native_to_relation':norm(native,relation),
        'grad_from_native_to_projection':norm(native,projection),
        'grad_from_local_loss_to_projection':norm(CFG['lambda_local']*local,projection),
        'local_gradient_measured_even_if_coefficient_disabled':True}


def trajectory(loaded,wrapper,steps=3):
    loaded.reset()
    optimizer = loaded.optimizer(new=wrapper)
    core,captures,rows = loaded.core,{},[]
    counts = {'img':0,'ph':0,'clf':0}
    handles = []
    for name,module in (('img',core.img_unet),('ph',core.ph_unet),('clf',core.clf)):
        def observe(module,inputs,output,name=name):
            captures[name] = output.detach().cpu().clone(); counts[name] += 1
        handles.append(module.register_forward_hook(observe))
    bn0 = int(core.clf[2].num_batches_tracked)
    try:
        for step in range(steps):
            core.train(); loaded.aligner.train(); optimizer.zero_grad()
            if wrapper:
                logits,graph,_ = loaded.forward('L3',alpha=0)
                graph_check(loaded.wrapper.last,alpha=0)
            else: logits,graph = core(loaded.x_cpu,loaded.ph_cpu,loaded.affinity_cpu)
            a = core.fused_graph.detach().cpu().clone()
            embeddings = {k:v.clone() for k,v in captures.items()}
            ce = F.cross_entropy(logits[loaded.split['train']],loaded.labels[loaded.split['train']])
            (ce+graph).backward()
            grads = {n:p.grad.detach().cpu().clone() if p.grad is not None else None for n,p in core.named_parameters()}
            pools = [p.detach().cpu().clone() for enc in (core.img_unet,core.ph_unet) for p in enc.perms]
            optimizer.step()
            adam = {n:{k:v.detach().cpu().clone() if torch.is_tensor(v) else v for k,v in optimizer.state[p].items()}
                for n,p in core.named_parameters() if p in optimizer.state}
            core.eval(); loaded.aligner.eval()
            with torch.no_grad():
                if wrapper: val,vg,_ = loaded.forward('L3',alpha=0)
                else: val,vg = core(loaded.x_cpu,loaded.ph_cpu,loaded.affinity_cpu)
            vc = F.cross_entropy(val[loaded.split['val']],loaded.labels[loaded.split['val']])
            assert int(core.clf[2].num_batches_tracked) == bn0+step+1
            assert all(v==2*(step+1) for v in counts.values())
            rows.append({'logits':logits.detach().cpu().clone(),'graph':a,'ce':ce.detach().cpu().clone(),
                'graph_loss':graph.detach().cpu().clone(),'embeddings':embeddings,'grads':grads,'adam':adam,'pools':pools,
                'val_logits':val.cpu().clone(),'val_loss':(vc+vg).cpu().clone(),'core':export_core(core),'rng':rng_hash(capture_rng())})
    finally:
        for h in handles: h.remove()
    return rows


def check_one(seed,fold,audit_info):
    from tests import synthetic_checks
    started = time.perf_counter()
    assert os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'
    torch.use_deterministic_algorithms(True)
    loaded = Loaded(seed,fold,audit_info)
    cpu = synthetic_checks(torch.device('cpu'))
    ppu = synthetic_checks(loaded.device)
    original = trajectory(loaded,False)
    candidate = trajectory(loaded,True)
    assert_exact(original,candidate,'three_step_native_wrapper')
    del original,candidate
    loaded.reset(); loaded.core.eval(); loaded.aligner.eval()
    with torch.no_grad():
        logits,graph,local = loaded.forward('L3')
        assert torch.equal(loaded.wrapper.last['A_base'],loaded.wrapper.last['A_new'])
        raw = diagnostic(loaded,loaded.wrapper.last)
    initial = export_core(loaded.core)
    import_core(loaded.core,initial)
    assert core_hash(export_core(loaded.core)) == core_hash(initial)
    # Same checkpoint and RNG must survive serialization and explicit unregistered RP restore.
    stream = io.BytesIO(); torch.save(initial,stream); stream.seek(0)
    saved = torch.load(stream,weights_only=False,map_location='cpu')
    assert_exact(initial,saved,'checkpoint_serialization')
    rng = capture_rng()
    with torch.no_grad(): reference = loaded.forward('L3')[:2]
    import_core(loaded.core,saved); restore_rng(rng)
    with torch.no_grad(): recovered = loaded.forward('L3')[:2]
    assert_exact(reference,recovered,'checkpoint_forward_round_trip')
    result = {'passed':True,'code_sha256':code_hash(),'seed':seed,'fold':fold,
        'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'three_step_max_abs':0.0,
        'cpu_synthetic':cpu,'ppu_synthetic':ppu,'initial_graph_exact':True,
        'private_init_sha256':state_hash(loaded.initial_aligner),'data_digest':loaded.data_digest,
        'input_sha256':loaded.anchor['inputs_digest'],'raw_train_relationships':raw,
        'checkpoint_round_trip':True,'checkpoint_logits_round_trip_max_abs':0.0,
        'outer_test_metrics_computed':False,'seconds':time.perf_counter()-started,
        'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20}
    return result


def train_one(action,group,seed,fold,audit_info):
    if action == 'train':
        state = json.loads((BASE/'summary.json').read_text())
        assert state['preflight']['formal_authorized'], 'Formal training forbidden before registered Go'
    assert not torch.are_deterministic_algorithms_enabled(), 'Diagnostic determinism leaked into native training'
    path = task_path(action,group,seed,fold)
    if path.exists():
        existing = load_torch(path)
        assert existing['code_sha256'] == code_hash() and existing['state'] == 'complete'
        assert existing['group'] == group and existing['seed'] == seed and existing['fold'] == fold
        return {'path':str(path),'sha256':sha_file(path),'skipped':True,**existing['execution']}
    formal = action == 'train'
    loaded = Loaded(seed,fold,audit_info)
    loaded.reset()
    opt = loaded.optimizer()
    assert rng_hash(capture_rng()) == loaded.anchor['pairing']['classifier_start_rng']
    stopper = EarlyStopping(patience=CFG['early_stop'],verbose=False)
    core = loaded.core
    torch.cuda.synchronize()
    started = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    curves,stats = [],{}
    best_loss,best_epoch,best = 1e10,0,None
    bn0 = int(core.clf[2].num_batches_tracked)
    epochs = CFG['epochs'] if formal else CFG['development_epochs']
    count = {'img':0,'ph':0,'clf':0}
    hooks = []
    for name,module in (('img',core.img_unet),('ph',core.ph_unet),('clf',core.clf)):
        def observe(module,inputs,output,name=name): count[name] += 1
        hooks.append(module.register_forward_hook(observe))
    for epoch in range(epochs):
        core.train(); loaded.aligner.train(); opt.zero_grad()
        logits,graph,local = loaded.forward(group)
        ce = F.cross_entropy(logits[loaded.split['train']],loaded.labels[loaded.split['train']])
        native = ce+graph
        total = native + (CFG['lambda_local']*local if CFG['groups'][group]['local'] else 0)
        assert torch.isfinite(total), 'Nonfinite training loss'
        diagnostic_step = epoch in CFG['diagnostic_epochs'] or (formal and epoch%50==0)
        if diagnostic_step:
            gd = gradients(loaded,native,local)
            mechanism = diagnostic(loaded,loaded.wrapper.last)
        total.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in list(core.parameters())+list(loaded.aligner.parameters()))
        if diagnostic_step:
            gd['grad_total_relation'] = grad_norm(loaded.aligner.parameters())
            gd['grad_total_projection'] = grad_norm(loaded.aligner.image.parameters())
            gd['grad_original_imaging_encoder'] = grad_norm(core.img_unet.parameters())
            gd['grad_original_phenotype_encoder'] = grad_norm(core.ph_unet.parameters())
            gd['grad_original_classifier'] = grad_norm(core.clf.parameters())
            if epoch == 0:
                assert gd['grad_from_native_to_relation']>0, 'Zero head cannot learn from native objective'
                assert gd['grad_original_imaging_encoder']>0 and gd['grad_original_phenotype_encoder']>0
        opt.step()
        core.eval(); loaded.aligner.eval()
        with torch.no_grad():
            val,vg,_ = loaded.forward(group)
        vc = F.cross_entropy(val[loaded.split['val']],loaded.labels[loaded.split['val']])
        vl = vc+vg
        assert torch.isfinite(vl)
        assert int(core.clf[2].num_batches_tracked) == bn0+epoch+1
        assert all(v==2*(epoch+1) for v in count.values())
        row = {'epoch':epoch,'train_ce':float(ce.detach()),'train_graph':float(graph.detach()),'train_total':float(total.detach()),
            'train_local':float(local.detach()),'val_ce':float(vc),'val_graph':float(vg),'val_loss':float(vl)}
        if diagnostic_step:
            idx = loaded.split['val']
            prob = val[idx].softmax(-1)[:,0].cpu().numpy()
            row['validation_only_metrics'] = metrics(loaded.ctx['labels'][idx],prob,val[idx].argmax(-1).cpu().numpy())
            stats[str(epoch)] = {'gradients':gd,'mechanism':mechanism,'native_rng':rng_hash(capture_rng()),
                'val_rng_aligner_no_random_ops':True}
        curves.append(row)
        if float(vl)<best_loss:
            best_loss,best_epoch = float(vl),epoch
            best = {'core':export_core(core),'aligner':cpu_state(loaded.aligner),'selection':row.copy()}
        stopper(vl,core)
        if epoch%20==0 or epoch==epochs-1:
            emit('progress',action=action,group=group,seed=seed,fold=fold,epoch=epoch,val_ce=float(vc),val_graph=float(vg),best_epoch=best_epoch)
        if formal and stopper.early_stop: break
    for h in hooks: h.remove()
    torch.cuda.synchronize()
    training_seconds = time.perf_counter()-started
    result = {'state':'complete','action':action,'group':group,'seed':seed,'fold':fold,'code_sha256':code_hash(),
        'data_digest':loaded.data_digest,'input_sha256':loaded.anchor['inputs_digest'],'pairing':loaded.anchor['pairing'],
        'private_init_sha256':state_hash(loaded.initial_aligner),'best_epoch':best_epoch,'epochs_run':epoch+1,
        'best_validation_loss':best_loss,'best':best,'curves':curves,'diagnostics':stats,
        'outer_test_metrics_computed':formal,
        'execution':{'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'seconds':training_seconds,
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'core_parameters':sum(p.numel() for p in core.parameters()),
            'host_peak_rss_mib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            'new_parameters':sum(p.numel() for p in loaded.aligner.parameters()),'cpu_training':False,
            'native_submodule_calls':count,'BN_updates':epoch+1},
        'best_core_sha256':core_hash(best['core']),'best_aligner_sha256':state_hash(best['aligner'])}
    if formal:
        result['rng_before_test'] = capture_rng()
        import_core(core,best['core']); loaded.aligner.load_state_dict(best['aligner'])
        core.eval(); loaded.aligner.eval()
        outputs,_,_ = loaded.forward(group)
        idx = loaded.split['test']
        logits_test = outputs[idx].detach().cpu().numpy()
        prob = softmax(logits_test,axis=1)[:,0]
        y = loaded.ctx['labels'][idx]
        result['oof'] = {'indices':idx,'ids':[loaded.ctx['ids'][i] for i in idx],'labels':y,
            'sites':loaded.meta['site'][idx],'logits':logits_test,'prob_asd':prob,'pred':logits_test.argmax(-1)}
        result['metrics'] = metrics(y,prob,logits_test.argmax(-1))
        result['rng_after_test'] = capture_rng()
    else:
        assert epoch+1==80 and len(stats)==8
    atomic_torch(result,path)
    saved = load_torch(path)
    assert saved['best_core_sha256'] == core_hash(saved['best']['core'])
    assert saved['best_aligner_sha256'] == state_hash(saved['best']['aligner'])
    return {'path':str(path),'sha256':sha_file(path),'best_epoch':best_epoch,'best_validation_loss':best_loss,
        'selected_val_ce':best['selection']['val_ce'],'outer_test_metrics_computed':formal,**result['execution']}


def frozen_one(group,seed,fold,audit_info):
    path = task_path('train',group,seed,fold)
    result = load_torch(path)
    assert result['code_sha256'] == code_hash()
    if 'frozen' in result:
        return {'path':str(path),'sha256':sha_file(path),'skipped':True}
    loaded = Loaded(seed,fold,audit_info)
    records = []
    for base_seed in CFG['frozen_draw_bases']:
        draw = int(hashlib.sha256(f'{seed}/{fold}/PCCI-repeat-{base_seed}'.encode()).hexdigest()[:15],16)%(2**32)
        modes = ('true','off','shuffled') if group!='L1' else ('trained','off')
        outputs = {}
        for mode in modes:
            import_core(loaded.core,result['best']['core']); loaded.aligner.load_state_dict(result['best']['aligner'])
            for p in loaded.core.parameters(): p.requires_grad_(False)
            loaded.core.eval(); loaded.aligner.eval()
            restore_rng(loaded.start['rng'])
            random.seed(draw); np.random.seed(draw); torch.manual_seed(draw); torch.cuda.manual_seed_all(draw)
            state_before = export_core(loaded.core)
            with torch.no_grad():
                logits,_,_ = loaded.forward(group,alpha=0 if mode=='off' else None,
                    condition=mode if mode in ('true','shuffled') else None)
            after = export_core(loaded.core)
            # Native constraint projection may alter the shared RP conv. Restore before the next condition.
            changes = [n for n in state_before['state'] if not torch.equal(state_before['state'][n],after['state'][n])]
            assert all(n.startswith('rp_attention.conv') for n in changes), changes
            assert torch.equal(state_before['rp_weights'],after['rp_weights'])
            idx = loaded.split['test']
            outputs[mode] = logits[idx].softmax(-1)[:,0].cpu().numpy()
            import_core(loaded.core,state_before)
            assert core_hash(export_core(loaded.core)) == core_hash(state_before)
        if group!='L1':
            outputs['trained'] = outputs['shuffled' if CFG['groups'][group]['shuffled'] else 'true']
        records.append({'draw_seed':draw,'prob_asd':outputs})
    result['frozen'] = records
    atomic_torch(result,path)
    return {'path':str(path),'sha256':sha_file(path),'draws':len(records),'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action',choices=['audit','check','develop','train','frozen'])
    parser.add_argument('--group',choices=GROUPS,default='L3')
    parser.add_argument('--seed',type=int,default=911)
    parser.add_argument('--fold',type=int,default=0)
    args = parser.parse_args()
    if args.action=='audit':
        info = audit(); emit('audit_done',anchors=len(info['anchors']),files=len(info['manifest'])); return
    info = json.loads((BASE/'audit.json').read_text())
    assert info['code_sha256']==code_hash()
    with task_lock(args.action,args.group,args.seed,args.fold):
        if args.action=='check': result = check_one(args.seed,args.fold,info)
        elif args.action=='frozen': result = frozen_one(args.group,args.seed,args.fold,info)
        else: result = train_one(args.action,args.group,args.seed,args.fold,info)
    emit('done',action=args.action,group=args.group,seed=args.seed,fold=args.fold,result=result)


if __name__=='__main__':
    try: main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)

