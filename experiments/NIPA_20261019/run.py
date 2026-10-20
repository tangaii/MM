"""Independent native PPU training, exact trajectory audit and frozen factorials."""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager, redirect_stdout
import csv
from datetime import datetime, timezone
import difflib
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

sys.dont_write_bytecode = True
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD','1')
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
OLD = ROOT/'experiments/DRPA_20261014'
PREV = ROOT/'experiments/PCDCA_20261019'
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, recall_score, f1_score, log_loss, brier_score_loss
from opt import OptInit
from model.mm_gtunets import MM_GTUNets
from model.rp_graph import create_reward_penalty_graph
from utils.tools import EarlyStopping
from model_intervention import (SUBSETS, array_hash, describe, make_plan, rp_metadata,
                                raw_rp_diagnostic, intervened_forward)

CFG = json.loads((BASE/'config.json').read_text())
SOURCE = ('opt.py','train_mm_gtunets.py','model/mm_gtunets.py','model/gtunet.py',
          'model/rp_graph.py','utils/mydataloader.py','utils/tools.py','utils/metrics.py')
CODE = ('config.json','model_intervention.py','run.py','launch.py','analyze.py')


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_bytes(path):
    for attempt in range(6):
        try:
            return Path(path).read_bytes()
        except FileNotFoundError:
            if attempt == 5:
                raise
            time.sleep(.3*(attempt+1))


def read_json(path):
    return json.loads(read_bytes(path))


def sha_file(path):
    return hashlib.sha256(read_bytes(path)).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def payload_hash(value):
    h = hashlib.sha256()
    def add(v):
        if torch.is_tensor(v):
            h.update(b'tensor'); add(v.detach().cpu().numpy())
        elif isinstance(v,np.ndarray):
            h.update(b'array'); h.update(array_hash(v).encode())
        elif isinstance(v,np.generic):
            add(v.item())
        elif isinstance(v,dict):
            h.update(b'dict')
            for k in sorted(v,key=repr):
                add(k); add(v[k])
        elif isinstance(v,(list,tuple)):
            h.update(str(type(v)).encode()); h.update(str(len(v)).encode())
            for x in v:
                add(x)
        elif v is None or isinstance(v,(str,int,float,bool)):
            h.update(str(type(v)).encode()); h.update(repr(v).encode())
        else:
            raise TypeError(f'Unhashable artifact {type(v)}')
    add(value)
    return h.hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode())
        x = state[k].detach().cpu().numpy() if torch.is_tensor(state[k]) else state[k]
        h.update(array_hash(x).encode())
    return h.hexdigest()


def export_core(core):
    return {'state': {k:v.detach().cpu().clone() for k,v in core.state_dict().items()},
            'rp_weights': core.rp_attention.weights.detach().cpu().clone()}


def core_hash(saved):
    return state_hash({**saved['state'],'unregistered_rp_weights':saved['rp_weights']})


def import_core(core,saved):
    core.load_state_dict(saved['state'],strict=True)
    with torch.no_grad():
        core.rp_attention.weights.copy_(saved['rp_weights'].to(core.device))


def capture_rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.get_rng_state().clone(),
            'cuda':torch.cuda.get_rng_state(0).clone(),
            'backend':{'cudnn_deterministic':torch.backends.cudnn.deterministic,
                       'cudnn_benchmark':torch.backends.cudnn.benchmark,
                       'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
                       'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32}}


def restore_rng(s):
    random.setstate(s['python']); np.random.set_state(s['numpy'])
    torch.set_rng_state(s['cpu']); torch.cuda.set_rng_state(s['cuda'],0)
    b = s['backend']
    torch.backends.cudnn.deterministic = b['cudnn_deterministic']
    torch.backends.cudnn.benchmark = b['cudnn_benchmark']
    torch.backends.cuda.matmul.allow_tf32 = b['matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32 = b['cudnn_allow_tf32']


def rng_hash(s):
    return {'python':hashlib.sha256(repr(s['python']).encode()).hexdigest(),
            'numpy':hashlib.sha256(repr(s['numpy'][0]).encode()+s['numpy'][1].tobytes()+repr(s['numpy'][2:]).encode()).hexdigest(),
            'cpu':array_hash(s['cpu'].numpy()),'cuda':array_hash(s['cuda'].numpy())}


def atomic_bytes(data,path):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+f'.tmp.{os.getpid()}')
    with tmp.open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
    fd = os.open(path.parent,os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    if read_bytes(path) != data:
        raise IOError(f'Storage readback mismatch: {path}')


def atomic_json(value,path):
    atomic_bytes((json.dumps(value,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode(),path)


def atomic_torch(value,path):
    b = io.BytesIO()
    torch.save({'payload':value,'content_sha256':payload_hash(value)},b)
    atomic_bytes(b.getvalue(),path)


def load_torch(path,artifact=False):
    value = torch.load(io.BytesIO(read_bytes(path)),map_location='cpu',weights_only=False)
    if artifact:
        if payload_hash(value['payload']) != value['content_sha256']:
            raise IOError(f'Artifact content SHA mismatch: {path}')
        return value['payload']
    return value


def log(event,**values):
    line = json.dumps({'utc':utc(),'event':event,**values},ensure_ascii=False,allow_nan=False)
    if os.environ.get('NIPA_LOG_STDOUT') == '1':
        print(line,flush=True)
    else:
        with (BASE/'run.log').open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX); f.write(line+'\n'); f.flush()


@contextmanager
def task_lock(slot):
    with (BASE/'run.py').open('r+') as f:
        fcntl.lockf(f,fcntl.LOCK_EX|fcntl.LOCK_NB,1,1000+slot)
        try:
            yield
        finally:
            fcntl.lockf(f,fcntl.LOCK_UN,1,1000+slot)


def code_hash():
    return digest({n:sha_file(BASE/n) for n in CODE})


def anchor(seed,fold):
    return OLD/'artifacts'/f'seed_{seed}'/f'fold_{fold}'


def result_path(group,seed,fold):
    return BASE/'runs'/group/f's{seed}_f{fold}.pt'


def metadata(ctx):
    with (ROOT/CFG['data_folder']/'Phenotypic_V1_0b_preprocessed1.csv').open() as f:
        byid = {r['SUB_ID']:r for r in csv.DictReader(f)}
    rows = [byid[i] for i in ctx['ids']]
    y = np.asarray([int(r['DX_GROUP'])-1 for r in rows])
    age = np.asarray([float(r['AGE_AT_SCAN']) for r in rows])
    sex = np.asarray([int(r['SEX']) for r in rows])
    site = np.asarray([r['SITE_ID'] for r in rows])
    assert np.array_equal(y,ctx['labels']) and np.array_equal(np.bincount(y),[403,468])
    assert np.array_equal(site,ctx['site_names']) and len(np.unique(site)) == 20
    assert np.allclose(age,ctx['ph_data'][:,2],rtol=0,atol=2e-6)
    assert np.array_equal(sex-1,ctx['ph_data'][:,1])
    assert [(int(np.sum((y==k)&(sex==2))),int(np.sum((y==k)&(sex==1)))) for k in (0,1)] == [(54,349),(90,378)]
    assert np.isfinite(age).all()
    assert np.loadtxt(ROOT/CFG['data_folder']/'id.txt',dtype=str).reshape(-1).tolist() == ctx['ids']
    return {'age':age,'sex':sex,'site':site}


def binary_metrics(y,p,pred=None):
    y,p = np.asarray(y),np.asarray(p,np.float64)
    pred = np.where(p>=.5,0,1) if pred is None else np.asarray(pred)
    truth = (y==0).astype(int)
    return {'ACC':float(accuracy_score(y,pred)),'BA':float(balanced_accuracy_score(y,pred)),
            'AUC':float(roc_auc_score(truth,p)),'SEN':float(recall_score(y,pred,pos_label=0)),
            'SPE':float(recall_score(y,pred,pos_label=1)),'F1':float(f1_score(y,pred,pos_label=0)),
            'CE':float(log_loss(truth,np.clip(p,1e-7,1-1e-7),labels=[0,1])),
            'Brier':float(brier_score_loss(truth,p))}


def storage_check():
    p = BASE/'.storage_probe'; data = hashlib.sha256(os.urandom(100)).digest()*17
    try:
        atomic_bytes(data,p)
        assert sha_file(p) == hashlib.sha256(data).hexdigest()
        with task_lock(999):
            import subprocess
            code = "import fcntl,sys; f=open(sys.argv[1],'r+'); fcntl.lockf(f,fcntl.LOCK_EX|fcntl.LOCK_NB,1,1999)"
            c = subprocess.run([sys.executable,'-B','-c',code,str(BASE/'run.py')],capture_output=True)
            assert c.returncode != 0 and b'BlockingIOError' in c.stderr
    finally:
        p.unlink(missing_ok=True)
    return {'passed':True,'checks':['write','fsync','rename','directory_fsync','readback','SHA','cross_process_lock'],
            'ossfs_limit':'No strong POSIX visibility assumption; bounded ENOENT retry, persistent EIO fatal'}


def official_audit():
    commit = 'bb9351b931003535503694ad2f777b6cee1c151f'
    out = {}
    for name in SOURCE:
        url = f'https://raw.githubusercontent.com/NZWANG/MM-GTUNets/{commit}/{name}'
        raw = urllib.request.urlopen(url,timeout=30).read()
        local = read_bytes(ROOT/name)
        diff = list(difflib.unified_diff(raw.decode().splitlines(),local.decode().splitlines(),fromfile='official',tofile='local'))
        if name != 'model/mm_gtunets.py':
            assert not diff, f'Unexpected official difference: {name}'
        else:
            assert 'return (x * self.weight.reshape(3, 1, 1)).sum(dim=-3, keepdim=True)' in local.decode()
            assert len(diff) < 20, 'Unexpected root model changes'
        out[name] = {'official_sha256':hashlib.sha256(raw).hexdigest(),'local_sha256':sha_file(ROOT/name),'diff':diff}
    return {'commit':commit,'files':out,'only_existing_change':'CustomConv2d final 1x1 PPU projection; native .data constraints retained'}


def audit():
    if (BASE/'audit.json').exists():
        return guard()
    started = utc()
    ctx = load_torch(OLD/'artifacts/context.pt'); meta = metadata(ctx)
    assert len(ctx['ids']) == len(set(ctx['ids'])) == CFG['subjects']
    names = [ROOT/n for n in SOURCE]+list(ROOT.glob('model/*.py'))+list(ROOT.glob('utils/*.py'))+[ROOT/'prompt']
    folders = ('DRPA_20261014','PCND_20261016','PCND_DualStage_20261016','LCRA_20261018',
               'LCRA_Attribution_20261019','GBSA_Pilot_20261019','GBSA_Diagnostic_20261019','PCCI_20261015','PCDCA_20261019')
    for name in folders:
        names += [p for p in (ROOT/'experiments'/name).glob('*') if p.is_file() and p.suffix in ('.py','.json','.md')]
    names += [p for p in (OLD/'artifacts').rglob('*') if p.is_file()]
    for s in CFG['seeds']:
        for f in range(10):
            names += [OLD/f'outputs/E0/seed_{s}/fold_{f}'/n for n in ('best.pt','oof.csv','metrics.json','epochs.jsonl')]
            names += [PREV/f'runs/D0/s{s}_f{f}.pt']
    data = ROOT/CFG['data_folder']
    names += [data/'id.txt',data/'Phenotypic_V1_0b_preprocessed1.csv']
    names += [p for p in (data/'save_tensor').glob('*') if p.is_file()]
    names += [Path('/tmp/CMCV_20261019/report.md')]
    protected = {str(p):sha_file(p) for p in sorted(set(p for p in names if p.is_file()),key=str)}
    raw = torch.as_tensor(load_torch(data/'save_tensor/up_triangles.pt'))
    assert tuple(raw.shape) == (871,6670) and bool(torch.isfinite(raw).all())
    anchors = {}
    for s in CFG['seeds']:
        x = load_torch(anchor(s,0).parent/'selected_features.pt')
        support = np.load(io.BytesIO(read_bytes(anchor(s,0).parent/'rfe_support.npy')))
        assert tuple(x.shape) == (871,500) and support.shape == (500,)
        assert np.issubdtype(support.dtype,np.integer) and len(np.unique(support)) == 500
        assert torch.equal(raw[:,support].to(x.dtype),x)
        count = np.zeros(871,int)
        for f in range(10):
            ap = anchor(s,f); sp = load_torch(ap/'splits.pt'); start = load_torch(ap/'train_start.pt')
            fp = read_json(ap/'fingerprints.json'); count[sp['test']] += 1
            assert np.array_equal(np.sort(np.concatenate([sp[k] for k in SUBSETS])),np.arange(871))
            for k in SUBSETS:
                assert array_hash(sp[k]) == fp['split_fingerprints'][k]
            assert array_hash(x.numpy()) == fp['features_fingerprint']
            assert core_hash(start['core']) == fp['train_start_core_fingerprint']
            assert rng_hash(start['rng']) == fp['classifier_start_rng']
            vae = ap.parent/f'vae/fold{f}_pretrain.pth'
            assert sha_file(vae) == start['vae_sha256'] == fp['vae_sha256']
            plan = make_plan(meta,sp,s,f)
            paths = [ap/'splits.pt',ap/'train_start.pt',ap/'fingerprints.json',ap.parent/'selected_features.pt',ap.parent/'rfe_support.npy',vae]
            historical = read_json(OLD/f'outputs/E0/seed_{s}/fold_{f}/metrics.json')
            previous = load_torch(PREV/f'runs/D0/s{s}_f{f}.pt',artifact=True)
            assert np.array_equal(previous['oof']['indices'],sp['test'])
            assert np.array_equal(previous['oof']['labels'],ctx['labels'][sp['test']])
            assert max(abs(previous['metrics'][k]-binary_metrics(previous['oof']['labels'],previous['oof']['probability'],previous['oof']['prediction'])[k]) for k in previous['metrics']) < 1e-12
            anchors[f'{s}/{f}'] = {'input_sha256':{str(p):sha_file(p) for p in paths},'pairing':fp,
                'permutation_plan':plan['records'],'physical_gpu':f%2,
                'historical_C0_metrics':historical['metrics'],'historical_C0_best_epoch':historical['best_epoch'],
                'previous_D0_metrics':previous['metrics'],'previous_D0_best_epoch':previous['best_epoch']}
        assert np.all(count == 1)
    out = {'experiment':CFG['experiment'],'started_utc':started,'code_sha256':code_hash(),
           'source_sha256':{n:sha_file(ROOT/n) for n in SOURCE},'protected_sha256':protected,
           'context_sha256':sha_file(OLD/'artifacts/context.pt'),'anchors':anchors,
           'handoff_before_sha256':sha_file(ROOT/'handoff.md'),'handoff_before_bytes':len(read_bytes(ROOT/'handoff.md')),
           'data':{'subjects':871,'labels':[403,468],'sites':20,'raw_shape':[871,6670],'model_image_shape':[871,500],
                   'sex':'CSV1=male/2=female; native0/1; verified per-subject and class-sex counts'},
           'storage':storage_check(),'official_source_comparison':official_audit(),
           'environment':{'torch':torch.__version__,'numpy':np.__version__,'python':sys.version},
           'preflight':{},'formal_authorized':False,'baseline_accepted':False,'execution':{},'scientific_stop':None}
    atomic_json(out,BASE/'audit.json')
    log('audit_passed',anchors=30,protected_files=len(protected),code_sha256=out['code_sha256'])
    return out


def guard():
    a = read_json(BASE/'audit.json')
    assert a['code_sha256'] == code_hash(), 'Registered code/config changed'
    for name,h in a['source_sha256'].items():
        assert sha_file(ROOT/name) == h, f'Protected root source changed: {name}'
    return a


class Observer:
    """Read-only hooks; no random operations or parameter/buffer mutation."""
    def __init__(self,core):
        self.core = core; self.enabled = False; self.detailed = False; self.values = {}; self.rng = {}; self.counts = {}
        self.handles = []
        for name,module in (('vae',core.vae),('rp',core.rp_attention),('img',core.img_unet),
                            ('ph',core.ph_unet),('attention',core.mm_attention),('clf',core.clf)):
            self.handles.append(module.register_forward_pre_hook(self.before(name)))
            self.handles.append(module.register_forward_hook(self.after(name)))

    def begin(self,detailed=False):
        self.enabled = True; self.detailed = detailed; self.values = {}; self.rng = {}; self.counts = {}

    def before(self,name):
        def hook(module,args):
            if not self.enabled:
                return
            call = self.counts.get(name,0)
            if self.detailed:
                self.rng[f'{name}/{call}/before'] = rng_hash(capture_rng())
            if name in ('img','ph'):
                self.values[f'{name}_input'] = args[0]
                self.values[f'{name}_edge_index'] = args[1]
                self.values[f'{name}_edge_attr'] = args[2]
            elif name == 'clf':
                self.values['joint'] = args[0]
            elif name == 'vae':
                self.values['vae_input'] = args[0]
            elif name == 'rp':
                self.values['affinity'] = args[0]
            elif name == 'attention':
                self.values[f'attention_{call}_ph'] = args[1]
        return hook

    def after(self,name):
        def hook(module,args,out):
            if not self.enabled:
                return
            call = self.counts.get(name,0); self.counts[name] = call+1
            if name == 'vae':
                self.values['ph_z'] = out[0]
            elif name in ('img','ph','clf'):
                self.values[name] = out
            elif name == 'attention':
                self.values[f'attention_{call}_joint'] = out[0]
                self.values[f'attention_{call}_weights'] = torch.stack((out[1],out[2]))
            if self.detailed:
                self.rng[f'{name}/{call}/after'] = rng_hash(capture_rng())
        return hook

    def finish(self,graph):
        self.values.update(rp_graph=self.core.rp_graph,fused_sim=self.core.fused_sim,
                           fused_graph=self.core.fused_graph,value=self.core.value,graph_loss=graph,
                           graph_attention_weights=torch.stack((self.core.img_weight,self.core.ph_weight)))
        assert all(self.counts.get(k) == 1 for k in ('vae','rp','img','ph','clf'))
        assert self.counts.get('attention') in (1,2)
        self.enabled = False

    def snapshot(self):
        values = {k:v.detach().cpu().numpy().copy() for k,v in self.values.items()}
        return {'values':values,'rng':self.rng.copy(),'calls':self.counts.copy(),
                'topk':{name:[p.detach().cpu().numpy().copy() for p in module.perms]
                        for name,module in (('img',self.core.img_unet),('ph',self.core.ph_unet))},
                'bn':{k:v.detach().cpu().numpy().copy() for k,v in self.core.clf[2].state_dict().items()}}

    def close(self):
        for h in self.handles:
            h.remove()
        self.values.clear()


class Loaded:
    def __init__(self,seed,fold,a):
        if not torch.cuda.is_available():
            raise RuntimeError('No PPU; CPU neural training forbidden')
        torch.cuda.set_device(0); torch.set_num_threads(CFG['host_threads'])
        assert 'PPU' in torch.cuda.get_device_name(0).upper()
        self.seed,self.fold = seed,fold
        self.ctx = load_torch(OLD/'artifacts/context.pt'); self.meta = metadata(self.ctx)
        self.sp = load_torch(anchor(seed,fold)/'splits.pt'); self.start = load_torch(anchor(seed,fold)/'train_start.pt')
        self.anchor = a['anchors'][f'{seed}/{fold}']
        for p,h in self.anchor['input_sha256'].items():
            assert sha_file(p) == h, f'Input changed: {p}'
        self.plan = make_plan(self.meta,self.sp,seed,fold)
        assert digest(self.plan['records']) == digest(self.anchor['permutation_plan'])
        argv = sys.argv
        try:
            sys.argv = ['native_options']; opt = OptInit(model='MM_GTUNets',dataset='ABIDE',atlas='aal').args
        finally:
            sys.argv = argv
        opt.seed=seed; opt.device=torch.device('cuda:0'); opt.data_folder=str(ROOT/CFG['data_folder'])
        opt.ckpt_path=str(anchor(seed,fold).parent/'vae'); opt.epoch=CFG['epochs']; opt.early_stop=CFG['early_stop']
        opt.lr=CFG['lr']; opt.wd=CFG['weight_decay']; opt.log_save=False; opt.model_save=False
        self.opt=opt; self.core=MM_GTUNets(opt,fold).to(opt.device)
        self.core.vae_save_path=str(anchor(seed,fold).parent/f'vae/fold{fold}_pretrain.pth')
        import_core(self.core,self.start['core'])
        assert core_hash(export_core(self.core)) == self.anchor['pairing']['train_start_core_fingerprint']
        self.x_cpu=load_torch(anchor(seed,fold).parent/'selected_features.pt').float()
        self.ph_cpu=torch.from_numpy(self.ctx['ph_data']).float()
        args=[self.sp[k] for k in SUBSETS]
        real=create_reward_penalty_graph(self.ctx['ph_dict'],self.ctx['labels'],*args,opt)
        null=create_reward_penalty_graph(rp_metadata(self.ctx['ph_dict'],self.plan['permutations']['R']),self.ctx['labels'],*args,opt)
        self.rp_diagnostic=raw_rp_diagnostic(real,null,self.sp)
        self.affinity_cpu=torch.from_numpy(real).float(); self.null_affinity_cpu=torch.from_numpy(null).float()
        self.x,self.ph,self.affinity,self.null_affinity=[t.to(opt.device) for t in (self.x_cpu,self.ph_cpu,self.affinity_cpu,self.null_affinity_cpu)]
        self.perms={k:torch.as_tensor(p,device=opt.device) for k,p in self.plan['permutations'].items()}
        self.y=torch.as_tensor(self.ctx['labels'],dtype=torch.long,device=opt.device)
        self.observer=Observer(self.core)
        self.input_digest=payload_hash({'x':self.x_cpu,'ph':self.ph_cpu,'affinity':self.affinity_cpu,
                                        'labels':self.ctx['labels'],'ids':self.ctx['ids']})

    def optimizer(self):
        params=list(self.core.parameters())
        assert not any(p is self.core.rp_attention.weights for p in params)
        return torch.optim.Adam(params,lr=CFG['lr'],weight_decay=CFG['weight_decay'])

    def reset(self,diagnostic=False):
        import_core(self.core,self.start['core'])
        for p in self.core.parameters():
            p.grad=None
        restore_rng(self.start['rng']); torch.use_deterministic_algorithms(diagnostic)

    def forward(self,group='N0',observe=False,detailed=False):
        if observe:
            self.observer.begin(detailed)
        if group == 'N0':
            out=self.core(self.x,self.ph,self.affinity)
        else:
            spec=CFG['groups']['N0' if group=='all_real' else group]
            affinity=self.null_affinity if spec['R']=='null' else self.affinity
            out=intervened_forward(self.core,self.x,self.ph,affinity,self.perms['G'],self.perms['F'],spec)
        if observe:
            self.observer.finish(out[1])
            assert self.observer.counts['attention'] == (2 if group not in ('N0','all_real') and CFG['groups'][group]['F']=='null' else 1)
        return out

    def close(self):
        self.observer.close(); self.__dict__.clear(); torch.cuda.empty_cache()


def grad_norm(params):
    a=[p.grad.detach().square().sum() for p in params if p.grad is not None]
    return float(torch.stack(a).sum().sqrt()) if a else 0.


def module_changes(saved,initial):
    out={}
    for prefix in ('img_unet.','ph_unet.','mm_attention.','clf.','rp_attention.'):
        keys=[k for k,v in saved['state'].items() if k.startswith(prefix) and v.is_floating_point()]
        d=sum(float((saved['state'][k].double()-initial['state'][k].double()).square().sum()) for k in keys)**.5
        n=sum(float(initial['state'][k].double().square().sum()) for k in keys)**.5
        out[prefix]={'l2':d,'relative_l2':d/max(n,1e-12)}
    return out


def sampled_statistics(l):
    v=l.observer.values
    return {'graph':describe(v['fused_graph']), 'similarity':describe(v['fused_sim']), 'RP':describe(v['rp_graph']),
            'constant_RP_background':float(.5*l.core.rp_attention.weights[1:].detach().sum()),
            'embedding_norm':{k:describe(v[k].norm(dim=-1)) for k in ('ph_z','img','ph','joint')},
            'modality_graph_weights':v['graph_attention_weights'].detach().cpu().tolist(),
            'calls':l.observer.counts.copy(), 'edge_count':v['img_edge_index'].shape[1]}


def valid_result(group,seed,fold,a,required=False):
    path=result_path(group,seed,fold)
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    v=load_torch(path,True)
    assert v['state']=='complete' and (v['group'],v['seed'],v['fold'])==(group,seed,fold)
    assert v['code_sha256']==a['code_sha256'] and v['input_sha256']==a['anchors'][f'{seed}/{fold}']['input_sha256']
    assert v['best_core_sha256']==core_hash(v['core']) and v['factors']==CFG['groups'][group]
    sp=load_torch(anchor(seed,fold)/'splits.pt')
    assert np.array_equal(v['oof']['indices'],sp['test'])
    assert digest(v['permutation_plan'])==digest(a['anchors'][f'{seed}/{fold}']['permutation_plan'])
    assert v['physical_gpu']==fold%2
    p=v['oof']['probability']; assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
    m=binary_metrics(v['oof']['labels'],p,v['oof']['prediction'])
    assert max(abs(m[k]-v['metrics'][k]) for k in m)<1e-12
    assert v['formal_sha256']==payload_hash({k:x for k,x in v.items() if k not in ('formal_sha256','frozen')})
    return v


def train_one(group,seed,fold,a):
    if valid_result(group,seed,fold,a) is not None:
        log('skip_valid',group=group,seed=seed,fold=fold); return
    assert a['formal_authorized'] and (group=='N0' or a['baseline_accepted'])
    l=Loaded(seed,fold,a)
    try:
        optimizer=l.optimizer(); stopper=EarlyStopping(patience=CFG['early_stop'],save=False)
        l.reset(); assert rng_hash(capture_rng())==l.anchor['pairing']['classifier_start_rng']
        assert not torch.are_deterministic_algorithms_enabled()
        order=[n for n,p in l.core.named_parameters()]
        bn_start=int(l.core.clf[2].num_batches_tracked)
        best=None; best_loss=1e10; history=[]
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter(); started_utc=utc()
        for epoch in range(CFG['epochs']):
            sample=epoch<3 or epoch%CFG['diagnostic_frequency']==0
            l.core.train(); optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                logits,graph=l.forward(group,observe=sample)
            ce=F.cross_entropy(logits[l.sp['train']],l.y[l.sp['train']]); total=ce+graph
            if not bool(torch.isfinite(total)):
                raise FloatingPointError('Nonfinite train loss')
            record={'epoch':epoch,'train_ce':float(ce.detach()),'train_graph':float(graph.detach()),'train_loss':float(total.detach())}
            if sample:
                record['native_path']=sampled_statistics(l)
            total.backward()
            if sample:
                record['gradient_norms']={n:grad_norm(m.parameters()) for n,m in
                    (('img',l.core.img_unet),('ph',l.core.ph_unet),('attention',l.core.mm_attention),('clf',l.core.clf),('RP',l.core.rp_attention))}
                for n,p in l.core.named_parameters():
                    if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
                        raise FloatingPointError(f'Nonfinite gradient {n}')
            optimizer.step()
            assert int(l.core.clf[2].num_batches_tracked)==bn_start+epoch+1
            l.core.eval()
            with torch.set_grad_enabled(False):
                val,vg=l.forward(group)
            vc=F.cross_entropy(val[l.sp['val']],l.y[l.sp['val']]); vl=vc+vg
            if not bool(torch.isfinite(vl)):
                raise FloatingPointError('Nonfinite validation')
            assert int(l.core.clf[2].num_batches_tracked)==bn_start+epoch+1
            record.update(val_ce=float(vc),val_graph=float(vg),val_loss=float(vl)); history.append(record)
            if float(vl)<best_loss:
                best_loss=float(vl); best={'core':export_core(l.core),'epoch':epoch,'record':record.copy()}
            stopper(vl,l.core)
            if epoch%100==0:
                log('progress',group=group,seed=seed,fold=fold,epoch=epoch,val_loss=float(vl))
            if stopper.early_stop:
                break
            if sample:
                l.observer.values.clear()
        torch.cuda.synchronize(); seconds=time.perf_counter()-started
        assert best is not None
        end_rng=capture_rng(); import_core(l.core,best['core']); l.core.eval()
        assert payload_hash(end_rng)==payload_hash(capture_rng())
        with torch.set_grad_enabled(True):
            test,tg=l.forward(group,observe=True)
        after=capture_rng(); idx=l.sp['test']
        logits=test.detach().cpu().numpy()[idx]; p=test.detach().softmax(1).cpu().numpy()[idx,0]
        oof={'indices':idx,'ids':[l.ctx['ids'][int(i)] for i in idx],'labels':l.ctx['labels'][idx],
             'logits':logits,'probability':p,'prediction':logits.argmax(1)}
        v={'state':'complete','group':group,'factors':CFG['groups'][group],'seed':seed,'fold':fold,
           'started_utc':started_utc,'completed_utc':utc(),'code_sha256':a['code_sha256'],
           'input_sha256':l.anchor['input_sha256'],'data_input_digest':l.input_digest,
           'pairing':l.anchor['pairing'],'permutation_plan':l.plan['records'],'raw_RP_intervention':l.rp_diagnostic,
           'core':best['core'],'best_core_sha256':core_hash(best['core']),'best_epoch':best['epoch'],
           'best_val_loss':best_loss,'best_record':best['record'],'epochs':history,'epochs_run':len(history),
           'training_seconds':seconds,'oof':oof,'metrics':binary_metrics(oof['labels'],p,oof['prediction']),
           'rng_before_test':end_rng,'rng_after_test':after,'formal_start_rng_sha256':rng_hash(l.start['rng']),
           'formal_test_grad_enabled':True,'formal_test_path':sampled_statistics(l),
           'formal_test_graph_loss':float(tg.detach()),'physical_gpu':int(os.environ.get('CUDA_VISIBLE_DEVICES','0')),
           'device':torch.cuda.get_device_name(0),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
           'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'optimizer_parameter_order':order,
           'parameter_count':sum(p.numel() for p in l.core.parameters()),'new_trainable_parameters':0,
           'rfe_fit_count':0,'vae_train_count':0,'bn_best_batches':int(best['core']['state']['clf.2.num_batches_tracked']),
           'best_core_change':module_changes(best['core'],l.start['core'])}
        v['formal_sha256']=payload_hash(v)
        atomic_torch(v,result_path(group,seed,fold)); valid_result(group,seed,fold,a,True)
        log('complete',group=group,seed=seed,fold=fold,epochs=len(history),best_epoch=best['epoch'],seconds=seconds,physical_gpu=v['physical_gpu'])
    finally:
        l.close()


def compare_exact(a,b,label):
    if payload_hash(a) != payload_hash(b):
        raise AssertionError(f'Exact equivalence failed: {label}')


def trace_forward(l,logits,graph):
    snap=l.observer.snapshot()
    return {'phase':'train' if l.core.training else 'val',
            'tensors':{k:array_hash(v) for k,v in snap['values'].items()},
            'topk':{k:[array_hash(v) for v in ps] for k,ps in snap['topk'].items()},
            'bn':payload_hash(snap['bn']),'intermediate_rng':snap['rng'],'calls':snap['calls'],
            'rng_after':payload_hash(capture_rng()),
            'CE':float(F.cross_entropy(logits[l.sp['train' if l.core.training else 'val']],
                                        l.y[l.sp['train' if l.core.training else 'val']]).detach())}


def trajectory(l,mode):
    l.reset(diagnostic=True); optimizer=l.optimizer(); trace=[]
    saved_perms=l.perms
    if mode=='all_real':
        l.perms={k:torch.arange(871,device=l.opt.device) for k in ('G','F','R')}
    original_step=optimizer.step
    def step(*args,**kwargs):
        trace[-1]['gradients']=state_hash({n:p.grad for n,p in l.core.named_parameters() if p.grad is not None})
        out=original_step(*args,**kwargs)
        trace[-1]['updated_core']=core_hash(export_core(l.core))
        trace[-1]['adam']=payload_hash(optimizer.state_dict())
        trace[-1]['rng_after_step']=payload_hash(capture_rng())
        return out
    optimizer.step=step
    handles=[]
    try:
        if mode=='official':
            def before(module,args):
                l.observer.begin(detailed=True)
            def after(module,args,out):
                l.observer.finish(out[1]); trace.append(trace_forward(l,*out))
            handles=[l.core.register_forward_pre_hook(before),l.core.register_forward_hook(after)]
            from utils.metrics import accuracy,auc,metrics
            fn=next(n for n in ast.parse(read_bytes(ROOT/'train_mm_gtunets.py')).body if isinstance(n,ast.FunctionDef) and n.name=='train')
            env={'torch':torch,'np':np,'os':os,'model':l.core,'optimizer':optimizer,'loss_fn':F.cross_entropy,
                 'x':l.x,'ph_features':l.ph,'affinity_graphs':l.affinity,'labels':l.y,'y':l.ctx['labels'],
                 'train_ind':l.sp['train'],'val_ind':l.sp['val'],'fold':l.fold,'opt':l.opt,
                 'early_stopping':EarlyStopping(patience=CFG['early_stop'],save=False),'accuracy':accuracy,'auc':auc,'metrics':metrics}
            for key in ('accs','corrects','aucs','sens','spes','f1'):
                env[key]=np.zeros(10)
            l.opt.epoch=CFG['preflight_steps']
            exec(compile(ast.Module(body=[fn],type_ignores=[]),str(ROOT/'train_mm_gtunets.py'),'exec'),env)
            with redirect_stdout(io.StringIO()):
                env['train']()
        else:
            for epoch in range(CFG['preflight_steps']):
                l.core.train(); optimizer.zero_grad()
                logits,graph=l.forward('N0' if mode=='N0' else 'all_real',observe=True,detailed=True)
                trace.append(trace_forward(l,logits,graph))
                loss=F.cross_entropy(logits[l.sp['train']],l.y[l.sp['train']])+graph
                loss.backward(); optimizer.step()
                l.core.eval()
                with torch.no_grad():
                    logits,graph=l.forward('N0' if mode=='N0' else 'all_real',observe=True,detailed=True)
                trace.append(trace_forward(l,logits,graph))
        assert len(trace)==2*CFG['preflight_steps'] and all(t['calls']['clf']==1 for t in trace)
        return trace
    finally:
        for h in handles:
            h.remove()
        l.perms=saved_perms; l.opt.epoch=CFG['epochs']; l.observer.values.clear()


def factor_isolation(l):
    snapshots={}; checks={}; initial=core_hash(l.start['core'])
    for group,spec in CFG['groups'].items():
        l.reset(diagnostic=True); assert core_hash(export_core(l.core))==initial
        optimizer=l.optimizer(); l.core.train(); optimizer.zero_grad()
        logits,graph=l.forward(group,observe=True,detailed=True)
        snapshots[group]=l.observer.snapshot()
        loss=F.cross_entropy(logits[l.sp['train']],l.y[l.sp['train']])+graph
        loss.backward()
        gradients={name:grad_norm(m.parameters()) for name,m in
                   (('img',l.core.img_unet),('ph',l.core.ph_unet),('attention',l.core.mm_attention),('clf',l.core.clf))}
        assert all(np.isfinite(g) and g>0 for g in gradients.values())
        assert all(bool(torch.isfinite(p.grad).all()) for p in l.core.parameters() if p.grad is not None)
        before_bn=int(l.core.clf[2].num_batches_tracked); assert before_bn==1
        optimizer.step(); l.core.eval()
        with torch.no_grad():
            val,vg=l.forward(group)
        assert int(l.core.clf[2].num_batches_tracked)==before_bn
        checks[group]={'gradients':gradients,'parameters':sum(p.numel() for p in l.core.parameters()),
                       'optimizer_parameter_order':[n for n,p in l.core.named_parameters()],
                       'train_loss':float(loss.detach()),'val_loss':float(F.cross_entropy(val[l.sp['val']],l.y[l.sp['val']])+vg),
                       'one_classifier_and_BN_update':True,'one_VAE_two_distinct_GTUNets':True}
        l.observer.values.clear()
    base=snapshots['N0']['values']
    for group,spec in CFG['groups'].items():
        v=snapshots[group]['values']
        for key in ('vae_input','ph_z','img_input','ph_input'):
            compare_exact(v[key],base[key],f'{group} unchanged {key}')
        compare_exact(v['ph_input'],v['ph_z'],f'{group} real ph_z feeds ph_unet')
        expected=l.null_affinity_cpu.numpy() if spec['R']=='null' else l.affinity_cpu.numpy()
        compare_exact(v['affinity'],expected,f'{group} RP switch')
        if spec['G']=='real':
            compare_exact(v['fused_sim'],base['fused_sim'],f'{group} G-real similarity')
        else:
            assert not np.array_equal(v['fused_sim'],base['fused_sim']), 'G target unchanged'
        if spec['R']=='real':
            compare_exact(v['rp_graph'],base['rp_graph'],f'{group} real RP')
        else:
            assert not np.array_equal(v['rp_graph'],base['rp_graph']), 'R target unchanged'
        expected_ph=v['ph'][l.plan['permutations']['F']] if spec['F']=='null' else v['ph']
        attention_key='attention_1_ph' if spec['F']=='null' else 'attention_0_ph'
        compare_exact(v[attention_key],expected_ph,f'{group} F switch')
        compare_exact(v['attention_0_ph'],v['ph'],f'{group} graph attention real phenotype')
        compare_exact(v['graph_attention_weights'],v['attention_0_weights'],f'{group} native graph weights')
        if spec['F']=='null':
            assert not np.array_equal(v['joint'],v['attention_0_joint']), 'F target unchanged'
    for real,null in (('N0','N2'),('N1','N3'),('N4','N6'),('N5','N7')):
        av,bv=snapshots[real]['values'],snapshots[null]['values']
        for key in ('ph_z','rp_graph','fused_sim','fused_graph','img_edge_index','img_edge_attr',
                    'ph_edge_index','ph_edge_attr','img','ph','attention_0_joint','graph_attention_weights','graph_loss'):
            compare_exact(av[key],bv[key],f'F-only invariant {real}/{null}/{key}')
        assert not np.array_equal(av['joint'],bv['joint'])
        compare_exact(snapshots[real]['topk'],snapshots[null]['topk'],f'F-only TopK {real}/{null}')
    assert len({v['parameters'] for v in checks.values()})==1
    assert len({tuple(v['optimizer_parameter_order']) for v in checks.values()})==1
    changed_labels=l.ctx['labels'].copy()
    heldout=np.concatenate((l.sp['val'],l.sp['test'])); changed_labels[heldout]=1-changed_labels[heldout]
    for perm in (np.arange(871),l.plan['permutations']['R']):
        meta=rp_metadata(l.ctx['ph_dict'],perm)
        old=create_reward_penalty_graph(meta,l.ctx['labels'],*[l.sp[k] for k in SUBSETS],l.opt)
        new=create_reward_penalty_graph(meta,changed_labels,*[l.sp[k] for k in SUBSETS],l.opt)
        compare_exact(old,new,'Heldout labels must not affect native RP')
    rng_before=capture_rng(); plan=make_plan(l.meta,l.sp,l.seed,l.fold)
    compare_exact(capture_rng(),rng_before,'Private permutation must not consume global RNG')
    compare_exact(plan,l.plan,'Fixed permutation reproducibility')
    return {'groups':checks,'F_only_upstream_and_graph_loss_exact':True,'all_combinations_verified':True,
            'heldout_label_invariance_RP':True,'permutations_have_no_label_argument':True,
            'private_permutation_RNG_unchanged':True,'raw_RP_changes':l.rp_diagnostic,
            'target_tensor_changed_all_three_factors':True,
            'scope':'Native transductive dependence on heldout features is deliberately preserved'}


def synthetic_checks():
    # Include a >=3 group: singleton/pair-only fixtures have a unique derangement.
    meta={'site':np.array(['A','A','A','A','B','C','D','D','E','F','F','F']),
          'age':np.arange(12,dtype=float)+7,'sex':np.array([1,2,1,2,1,2,1,1,2,1,2,1])}
    sp={'train':np.arange(6),'val':np.arange(6,9),'test':np.arange(9,12)}
    plan=make_plan(meta,sp,3,0)
    for k in ('G','F'):
        assert plan['permutations'][k][4]==4 and plan['permutations'][k][5]==5
    assert plan['records']['R']['partitions']['test']['site_changed_fraction']==0
    assert binary_metrics([0,0,1,1],[.9,.8,.2,.1])['AUC']==1
    return {'singleton_fixed':True,'one_site_R_unavoidable_zero':True,'bijections':True,'ASD_probability_column':True}


def preflight_one(seed,fold,a):
    l=Loaded(seed,fold,a)
    try:
        oracle=trajectory(l,'official'); replay=trajectory(l,'N0'); wrapped=trajectory(l,'all_real')
        compare_exact(oracle,replay,'Official AST vs N0: three complete optimization steps')
        compare_exact(oracle,wrapped,'Official AST vs identity all-real wrapper: three steps')
        isolation=factor_isolation(l)
        result={'passed':True,'seed':seed,'fold':fold,'official_AST_sha256':sha_file(ROOT/'train_mm_gtunets.py'),
                'strict_trace':oracle,'maximum_numerical_error':0,'three_complete_optimization_steps':True,
                'factor_isolation':isolation,'synthetic':synthetic_checks(),
                'physical_gpu':int(os.environ.get('CUDA_VISIBLE_DEVICES','0')),
                'diagnostic_deterministic_only':True,'does_not_prove_native_500epoch_bitwise_replay':True}
        log('preflight_passed',seed=seed,fold=fold); return result
    finally:
        l.close()


def difference(a,b):
    x,y=np.asarray(a,np.float64),np.asarray(b,np.float64)
    d=abs(x-y)
    return {'mean_absolute':float(d.mean()),'max_absolute':float(d.max()),
            'relative_L1':float(d.sum()/max(abs(y).sum(),1e-12)),
            'changed_fraction':float(np.mean(x!=y))}


def snapshot_difference(current,baseline,splits):
    c,b=current['values'],baseline['values']; out={}
    keys=('ph_z','rp_graph','fused_sim','fused_graph','img','ph','joint',
          'graph_loss','graph_attention_weights','attention_0_joint')
    for key in keys:
        out[key]=difference(c[key],b[key])
    out['actual_edges']={'same_index':np.array_equal(c['img_edge_index'],b['img_edge_index']),
                         'counts':[int(c['img_edge_index'].shape[1]),int(b['img_edge_index'].shape[1])]}
    if out['actual_edges']['same_index']:
        out['actual_edges']['weight_difference']=difference(c['img_edge_attr'],b['img_edge_attr'])
    out['graph_by_partition']={}
    for name,idx in splits.items():
        out['graph_by_partition'][name]=difference(c['fused_graph'][idx],b['fused_graph'][idx])
    out['topk']={}
    for name in ('img','ph'):
        rows=[]; cglobal=np.arange(871); bglobal=np.arange(871)
        for cp,bp in zip(current['topk'][name],baseline['topk'][name]):
            cglobal=cglobal[cp]; bglobal=bglobal[bp]
            rows.append({'local_index_equal':bool(np.array_equal(cp,bp)),
                         'global_order_equal':bool(np.array_equal(cglobal,bglobal)),
                         'global_set_overlap':float(len(set(cglobal)&set(bglobal))/max(len(cglobal),1))})
        out['topk'][name]=rows
    out['intermediate_rng_equal']={k:current['rng'].get(k)==v for k,v in baseline['rng'].items()}
    out['rng_extra_attention_has_no_consumption']=all(current['rng'][k]==current['rng'][k.replace('/before','/after')]
        for k in current['rng'] if k.startswith('attention/') and k.endswith('/before'))
    out['dropout_scope']='Same initial/intermediate RNG audited; changing TopK or edge shapes may change random-number-to-node correspondence'
    idx=splits['test']
    out['per_patient_test']={key:np.linalg.norm(c[key][idx]-b[key][idx],axis=1) for key in ('img','ph','joint')}
    out['per_patient_test']['graph_mean_absolute']=np.mean(abs(c['fused_graph'][idx]-b['fused_graph'][idx]),axis=1)
    return out


def frozen_one(seed,fold,a):
    v=valid_result('N0',seed,fold,a,True)
    if 'frozen' in v:
        log('skip_frozen',seed=seed,fold=fold); return
    original_formal=v['formal_sha256']; l=Loaded(seed,fold,a); started=time.perf_counter()
    try:
        rows=[]
        for base in CFG['frozen_draw_bases']:
            draw=int(hashlib.sha256(f'{seed}/{fold}/PCCI-repeat-{base}'.encode()).hexdigest()[:15],16)%2**32
            restore_rng(l.start['rng'])
            random.seed(draw); np.random.seed(draw); torch.manual_seed(draw); torch.cuda.manual_seed(draw)
            common_rng=capture_rng(); baseline=None; conditions={}
            for group in CFG['frozen_conditions']:
                import_core(l.core,v['core']); l.core.eval()
                for p in l.core.parameters():
                    p.grad=None
                restore_rng(common_rng)
                assert not torch.are_deterministic_algorithms_enabled()
                before=core_hash(export_core(l.core)); assert before==v['best_core_sha256']
                assert payload_hash(capture_rng())==payload_hash(common_rng)
                with torch.set_grad_enabled(True):
                    logits,graph=l.forward(group,observe=True,detailed=True)
                after_rng=rng_hash(capture_rng()); snap=l.observer.snapshot()
                current_state=export_core(l.core)
                drift={k:float((current_state['state'][k]-v['core']['state'][k]).abs().max())
                       for k in current_state['state'] if not torch.equal(current_state['state'][k],v['core']['state'][k])}
                assert all(k.startswith('rp_attention.convs.') or k.startswith('rp_attention.conv.') for k in drift), drift
                assert torch.equal(current_state['rp_weights'],v['core']['rp_weights'])
                assert all(p.grad is None for p in l.core.parameters())
                assert int(l.core.clf[2].num_batches_tracked)==v['bn_best_batches']
                if baseline is None:
                    baseline=snap
                compare_exact(snap['values']['ph_z'],baseline['values']['ph_z'],f'Frozen {group} identical same-draw VAE sample')
                idx=l.sp['test']; out=logits.detach().cpu().numpy()[idx]
                p=logits.detach().softmax(1).cpu().numpy()[idx,0]
                conditions[group]={'probability':p,'prediction':out.argmax(1),'logits':out,
                    'metrics':binary_metrics(v['oof']['labels'],p,out.argmax(1)),
                    'graph_loss':float(graph.detach()),'restored_core_sha256':before,
                    'initial_rng':rng_hash(common_rng),'final_rng':after_rng,'native_inplace_drift':drift,
                    'mechanism':snapshot_difference(snap,baseline,l.sp),'calls':snap['calls'],
                    'intermediate_rng':snap['rng'],
                    'topk_global_inputs':'Native cached local pool indices composed through earlier pools for global set comparison'}
                import_core(l.core,v['core'])
                assert core_hash(export_core(l.core))==before
                l.observer.values.clear(); del logits,graph,current_state,snap
            rows.append({'base':base,'draw_seed':draw,'conditions':conditions})
            del baseline
        v['frozen']={'rows':rows,'checkpoints':1,'full_forward_count':24,'backward_count':0,'optimizer_count':0,
            'physical_gpu':int(os.environ.get('CUDA_VISIBLE_DEVICES','0')),'seconds':time.perf_counter()-started,
            'same_initial_core_BN_RP_full_RNG_per_condition':True,
            'formal_fields_sha256_before':original_formal,'completed_utc':utc()}
        assert payload_hash({k:x for k,x in v.items() if k not in ('formal_sha256','frozen')})==original_formal
        atomic_torch(v,result_path('N0',seed,fold)); valid_result('N0',seed,fold,a,True)
        log('frozen_complete',seed=seed,fold=fold,forwards=24,seconds=v['frozen']['seconds'])
    finally:
        l.close()


def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['audit','preflight','train','frozen'])
    p.add_argument('--group',choices=list(CFG['groups'])); p.add_argument('--seed',type=int); p.add_argument('--fold',type=int)
    args=p.parse_args()
    if args.action=='audit':
        audit(); return
    a=guard()
    if args.action=='preflight':
        result=preflight_one(args.seed,args.fold,a)
        with task_lock(990):
            a=read_json(BASE/'audit.json'); a['preflight'][f'{args.seed}/{args.fold}']=result
            atomic_json(a,BASE/'audit.json')
        return
    group=args.group or 'N0'; slot=int(group[1:])*30+CFG['seeds'].index(args.seed)*10+args.fold
    with task_lock(slot):
        if args.action=='train':
            train_one(group,args.seed,args.fold,a)
        else:
            frozen_one(args.seed,args.fold,a)


if __name__=='__main__':
    try:
        main()
    except Exception as e:
        log('worker_error',type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
        raise
