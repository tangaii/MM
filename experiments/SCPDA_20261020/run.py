"""Independent paired PPU trainer, strict native replay checks and input audit."""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager, redirect_stdout
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

os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
BASE = Path(__file__).resolve().parent
ROOT = Path('/data/adodas/MM-GTUNets')
HIST = ROOT/'experiments/PCDCA_20261019'
OLD = ROOT / 'experiments/DRPA_20261014'
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, recall_score, f1_score, log_loss, brier_score_loss
from opt import OptInit
from model.mm_gtunets import MM_GTUNets
from model.rp_graph import create_reward_penalty_graph
from utils.tools import EarlyStopping
from prototype import PrototypeObjective, prepare, gradient_scale, private_seed, array_hash, describe

CFG = json.loads((BASE / 'config.json').read_text())
SUBSETS = ('train', 'val', 'test')
SOURCE = ('opt.py', 'train_mm_gtunets.py', 'model/mm_gtunets.py', 'model/gtunet.py',
          'model/rp_graph.py', 'utils/mydataloader.py', 'utils/tools.py', 'utils/metrics.py')
CODE = ('config.json', 'prototype.py', 'run.py', 'launch.py', 'analyze.py')


def utc():
    return datetime.now(timezone.utc).isoformat()


def read_bytes(path):
    for attempt in range(6):
        try:
            return Path(path).read_bytes()
        except FileNotFoundError:
            if attempt == 5:
                raise
            time.sleep(.3 * (attempt+1))


def read_json(path):
    return json.loads(read_bytes(path))


def sha_file(path):
    return hashlib.sha256(read_bytes(path)).hexdigest()


def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def payload_hash(value):
    h = hashlib.sha256()
    def add(v):
        if torch.is_tensor(v):
            h.update(b'tensor'); add(v.detach().cpu().numpy())
        elif isinstance(v, np.ndarray):
            h.update(b'array'); h.update(array_hash(v).encode())
        elif isinstance(v, np.generic):
            add(v.item())
        elif isinstance(v, dict):
            h.update(b'dict')
            for k in sorted(v, key=repr):
                add(k); add(v[k])
        elif isinstance(v, (list, tuple)):
            h.update(str(type(v)).encode()); h.update(str(len(v)).encode())
            for x in v:
                add(x)
        elif v is None or isinstance(v, (str, int, float, bool)):
            h.update(str(type(v)).encode()); h.update(repr(v).encode())
        else:
            raise TypeError(f'Unhashable artifact value {type(v)}')
    add(value)
    return h.hexdigest()


def state_hash(state):
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode()); h.update(array_hash(state[k].detach().cpu().numpy() if torch.is_tensor(state[k]) else state[k]).encode())
    return h.hexdigest()


def export_core(core):
    return {'state': {k:v.detach().cpu().clone() for k,v in core.state_dict().items()},
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
    random.setstate(s['python']); np.random.set_state(s['numpy'])
    torch.set_rng_state(s['cpu']); torch.cuda.set_rng_state(s['cuda'], 0)
    b = s['backend']
    torch.backends.cudnn.deterministic = b['cudnn_deterministic']
    torch.backends.cudnn.benchmark = b['cudnn_benchmark']
    torch.backends.cuda.matmul.allow_tf32 = b['matmul_allow_tf32']
    torch.backends.cudnn.allow_tf32 = b['cudnn_allow_tf32']


def rng_hash(s):
    return {'python': hashlib.sha256(repr(s['python']).encode()).hexdigest(),
            'numpy': hashlib.sha256(repr(s['numpy'][0]).encode()+s['numpy'][1].tobytes()+repr(s['numpy'][2:]).encode()).hexdigest(),
            'cpu': array_hash(s['cpu'].numpy()), 'cuda': array_hash(s['cuda'].numpy())}


def atomic_bytes(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp.{os.getpid()}')
    with tmp.open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    if read_bytes(path) != data:
        raise IOError(f'Atomic readback mismatch: {path}')


def atomic_json(v, path):
    atomic_bytes((json.dumps(v, ensure_ascii=False, allow_nan=False, indent=2)+'\n').encode(), path)


def atomic_torch(v, path):
    b = io.BytesIO()
    torch.save({'payload': v, 'content_sha256': payload_hash(v)}, b)
    atomic_bytes(b.getvalue(), path)


def load_torch(path, artifact=False):
    v = torch.load(io.BytesIO(read_bytes(path)), map_location='cpu', weights_only=False)
    if artifact:
        if payload_hash(v['payload']) != v['content_sha256']:
            raise IOError(f'Artifact content hash mismatch: {path}')
        return v['payload']
    return v


def log(event, **values):
    data = {'utc': utc(), 'event': event, **values}
    if os.environ.get('SCPDA_LOG_STDOUT') == '1':
        print(json.dumps(data, ensure_ascii=False, allow_nan=False), flush=True)
        return
    with (BASE/'run.log').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(data, ensure_ascii=False, allow_nan=False)+'\n'); f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


@contextmanager
def task_lock(slot):
    with (BASE/'run.py').open('r+') as f:
        fcntl.lockf(f, fcntl.LOCK_EX | fcntl.LOCK_NB, 1, 1000+slot)
        try:
            yield
        finally:
            fcntl.lockf(f, fcntl.LOCK_UN, 1, 1000+slot)


def code_hash():
    return digest({n:sha_file(BASE/n) for n in CODE})


def anchor(seed, fold):
    return OLD/'artifacts'/f'seed_{seed}'/f'fold_{fold}'


def result_path(group, seed, fold):
    return BASE/'runs'/group/f's{seed}_f{fold}.pt'


def metadata(ctx):
    with (ROOT/CFG['data_folder']/'Phenotypic_V1_0b_preprocessed1.csv').open() as f:
        rows = {r['SUB_ID']:r for r in csv.DictReader(f)}
    ordered = [rows[i] for i in ctx['ids']]
    y = np.asarray([int(r['DX_GROUP'])-1 for r in ordered])
    assert np.array_equal(y, ctx['labels']) and np.array_equal(np.bincount(y), [403,468])
    def number(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return float('nan')
    a = np.asarray([number(r['AGE_AT_SCAN']) for r in ordered])
    s = np.asarray([number(r['SEX']) for r in ordered])
    site = np.asarray([r['SITE_ID'] for r in ordered])
    assert np.array_equal(site, ctx['site_names']) and len(set(site)) == 20
    assert np.allclose(a, ctx['ph_data'][:,2], rtol=0, atol=2e-6)
    assert np.array_equal(s-1, ctx['ph_data'][:,1])
    # Verify sex semantics against the paper's sex counts, not a guessed ordinal direction.
    assert [(int(np.sum((y==k)&(s==2))), int(np.sum((y==k)&(s==1)))) for k in (0,1)] == [(54,349),(90,378)]
    return {'age':a, 'sex':s, 'site':site}


def binary_metrics(y, p, pred=None):
    y, p = np.asarray(y), np.asarray(p, np.float64)
    pred = np.where(p >= .5, 0, 1) if pred is None else np.asarray(pred)
    truth = (y==0).astype(int)
    return {'ACC':float(accuracy_score(y,pred)), 'BA':float(balanced_accuracy_score(y,pred)),
            'AUC':float(roc_auc_score(truth,p)), 'SEN':float(recall_score(y,pred,pos_label=0)),
            'SPE':float(recall_score(y,pred,pos_label=1)), 'F1':float(f1_score(y,pred,pos_label=0)),
            'CE':float(log_loss(truth,np.clip(p,1e-7,1-1e-7),labels=[0,1])),
            'Brier':float(brier_score_loss(truth,p))}


def storage_check():
    p = BASE/'.storage_probe'
    data = hashlib.sha256(os.urandom(100)).digest()*17
    try:
        atomic_bytes(data,p)
        assert sha_file(p) == hashlib.sha256(data).hexdigest()
        with task_lock(999):
            import subprocess
            code = "import fcntl,sys; f=open(sys.argv[1],'r+'); fcntl.lockf(f,fcntl.LOCK_EX|fcntl.LOCK_NB,1,1999)"
            child = subprocess.run([sys.executable,'-B','-c',code,str(BASE/'run.py')],capture_output=True)
            assert child.returncode != 0 and b'BlockingIOError' in child.stderr
    finally:
        p.unlink(missing_ok=True)
    return {'passed':True,'time':utc(),'checks':['write','fsync','rename','directory_fsync','closed_readback','SHA256','cross_process_exclusive_lock'],
            'warning':'ossfs rename visibility is not assumed strongly POSIX; ENOENT bounded retry, EIO fatal'}


def audit():
    if (BASE/'manifest.json').exists():
        saved=read_json(BASE/'manifest.json')
        assert saved['experiment']==CFG['experiment'] and saved['code_sha256']==code_hash(), 'Incompatible manifest/code; do not overwrite'
        return saved
    started=utc()
    ctx=load_torch(OLD/'artifacts/context.pt'); meta=metadata(ctx)
    assert len(ctx['ids'])==len(set(ctx['ids']))==871
    data=ROOT/CFG['data_folder']
    raw=torch.as_tensor(load_torch(data/'save_tensor/up_triangles.pt'))
    x=load_torch(anchor(911,0).parent/'selected_features.pt')
    support=np.load(io.BytesIO(read_bytes(anchor(911,0).parent/'rfe_support.npy')))
    assert tuple(raw.shape)==(871,6670) and tuple(x.shape)==(871,500)
    assert len(np.unique(support))==500 and torch.equal(raw[:,support].to(x.dtype),x)
    names=[ROOT/n for n in SOURCE]+[ROOT/'prompt']
    names+=list(ROOT.glob('model/*.py'))+list(ROOT.glob('utils/*.py'))
    for folder in (HIST,OLD,ROOT/'experiments/official_source_ppu_20261014'):
        names += [p for p in folder.glob('*') if p.is_file() and p.suffix in ('.py','.json','.md')]
    names += list((HIST/'runs').rglob('*.pt'))
    names += [data/'id.txt',data/'Phenotypic_V1_0b_preprocessed1.csv']
    names += [p for p in (data/'save_tensor').glob('*') if p.is_file()]
    names += [OLD/'artifacts/context.pt']
    anchors={}; coverage=np.zeros(871,int)
    for f in range(10):
        a=anchor(911,f);sp=load_torch(a/'splits.pt');start=load_torch(a/'train_start.pt')
        fp=read_json(a/'fingerprints.json')
        assert np.array_equal(np.sort(np.concatenate([sp[k] for k in SUBSETS])),np.arange(871))
        coverage[sp['test']]+=1
        for k in SUBSETS: assert array_hash(sp[k])==fp['split_fingerprints'][k]
        assert array_hash(x.numpy())==fp['features_fingerprint']
        assert core_hash(start['core'])==fp['train_start_core_fingerprint']
        assert rng_hash(start['rng'])==fp['classifier_start_rng']
        vae=a.parent/f'vae/fold{f}_pretrain.pth'
        assert sha_file(vae)==start['vae_sha256']==fp['vae_sha256']
        prepared=prepare(ctx['labels'],meta,sp['train'],911,f,CFG)
        historical=load_torch(HIST/'runs/D4'/f's911_f{f}.pt',artifact=True)
        assert payload_hash(prepared['info'])==payload_hash(historical['prototype_preparation'])
        paths=[a/'splits.pt',a/'train_start.pt',a/'fingerprints.json',a.parent/'selected_features.pt',a.parent/'rfe_support.npy',vae]
        names+=paths
        anchors[f'911/{f}']={'input_sha256':{str(p):sha_file(p) for p in paths},'pairing':fp,'prototype':prepared['info'],'physical_gpu':f%2}
    assert np.all(coverage==1)
    historical=read_json(HIST/'summary.json')
    expected={'D0':[82.1042,79.4573,84.4126,88.1012],'D1':[82.4451,80.6707,83.9778,88.2495],
              'D2':[82.5601,79.6829,85.0555,88.5632],'D3':[81.9854,79.4268,84.1952,88.2499],
              'D4':[82.5588,79.9268,84.8427,88.6897],'D5':[82.2153,79.1829,84.8474,88.5679]}
    historical_911={}
    for g,v in expected.items():
        m=historical['formal'][g]['seed_mean']['911']
        assert np.allclose([100*m[k] for k in ('ACC','SEN','SPE','AUC')],v,atol=.000051,rtol=0)
        historical_911[g]=m
    protected={str(p):sha_file(p) for p in sorted(set(p for p in names if p.is_file()),key=str)}
    handoff=read_bytes(ROOT/'handoff.md')
    out={'experiment':CFG['experiment'],'work_directory':str(BASE),'started_utc':started,
         'code_sha256':code_hash(),'source_sha256':{n:sha_file(ROOT/n) for n in SOURCE},
         'protected_sha256':protected,'context_sha256':sha_file(OLD/'artifacts/context.pt'),
         'handoff_before_sha256':hashlib.sha256(handoff).hexdigest(),'handoff_before_bytes':len(handoff),
         'anchors':anchors,'historical_seed911':historical_911,'data':{'subjects':871,'ASD':403,'TD':468,'sites':20},
         'storage':storage_check(),'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__},
         'preflight':{},'formal_authorized':False,'baseline_accepted':False,'d4_accepted':False,
         'execution':{},'scientific_stop':None,
         'literature':{'paper':'https://arxiv.org/html/2406.14455v3','official':'https://github.com/NZWANG/MM-GTUNets','official_commit':CFG['official_commit']}}
    atomic_json(out,BASE/'manifest.json')
    log('audit_passed',anchors=10,protected_files=len(protected),code_sha256=out['code_sha256'])
    return out


def guard():
    a = read_json(BASE/'manifest.json')
    assert a['code_sha256'] == code_hash(), 'Code/config fingerprint mismatch'
    for n, h in a['source_sha256'].items():
        assert sha_file(ROOT/n) == h, f'Protected source changed: {n}'
    return a


class Loaded:
    def __init__(self, seed, fold, audit_info):
        if not torch.cuda.is_available():
            raise RuntimeError('No PPU; CPU neural training forbidden')
        torch.cuda.set_device(0); torch.set_num_threads(CFG['host_threads'])
        assert 'PPU' in torch.cuda.get_device_name(0).upper()
        self.seed,self.fold = seed,fold
        self.ctx = load_torch(OLD/'artifacts/context.pt')
        self.meta = metadata(self.ctx)
        self.sp = load_torch(anchor(seed,fold)/'splits.pt')
        self.start = load_torch(anchor(seed,fold)/'train_start.pt')
        self.anchor = audit_info['anchors'][f'{seed}/{fold}']
        for p,h in self.anchor['input_sha256'].items():
            assert sha_file(p) == h, f'Input modified: {p}'
        argv = sys.argv
        try:
            sys.argv = ['native_options']
            opt = OptInit(model='MM_GTUNets',dataset='ABIDE',atlas='aal').args
        finally:
            sys.argv = argv
        opt.seed=seed; opt.device=torch.device('cuda:0')
        opt.data_folder=str(ROOT/CFG['data_folder']); opt.ckpt_path=str(anchor(seed,fold).parent/'vae')
        opt.epoch=CFG['epochs']; opt.early_stop=CFG['early_stop']; opt.lr=CFG['lr']; opt.wd=CFG['weight_decay']
        opt.log_save=False; opt.model_save=False
        self.opt=opt
        self.core=MM_GTUNets(opt,fold).to(opt.device)
        self.core.vae_save_path=str(anchor(seed,fold).parent/f'vae/fold{fold}_pretrain.pth')
        import_core(self.core,self.start['core'])
        assert core_hash(export_core(self.core)) == self.anchor['pairing']['train_start_core_fingerprint']
        self.x_cpu=load_torch(anchor(seed,fold).parent/'selected_features.pt').float()
        self.ph_cpu=torch.from_numpy(self.ctx['ph_data']).float()
        affinity=create_reward_penalty_graph(self.ctx['ph_dict'],self.ctx['labels'],*[self.sp[k] for k in SUBSETS],opt)
        self.affinity_cpu=torch.from_numpy(affinity).float()
        self.x,self.ph,self.affinity=[v.to(opt.device) for v in (self.x_cpu,self.ph_cpu,self.affinity_cpu)]
        self.y=torch.as_tensor(self.ctx['labels'],dtype=torch.long,device=opt.device)
        self.train=torch.as_tensor(self.sp['train'],device=opt.device)
        self.val=torch.as_tensor(self.sp['val'],device=opt.device)
        self.prepared=prepare(self.ctx['labels'],self.meta,self.sp['train'],seed,fold,CFG)
        assert digest(self.prepared['info']) == digest(self.anchor['prototype'])
        self.z=None
        self.calls={k:0 for k in ('core','image','phenotype','vae','classifier')}
        self.handle=self.core.img_unet.register_forward_hook(self.capture)
        self.count_handles=[module.register_forward_hook(self.count_hook(name)) for name,module in
            [('core',self.core),('phenotype',self.core.ph_unet),('vae',self.core.vae),('classifier',self.core.clf)]]
        self.input_digest=payload_hash({'x':self.x_cpu,'ph':self.ph_cpu,'affinity':self.affinity_cpu,'labels':self.ctx['labels'],'ids':self.ctx['ids']})

    def count_hook(self,name):
        def count(module,args,out):
            self.calls[name]+=1
        return count

    def capture(self,module,args,output):
        assert tuple(output.shape)==(871,16)
        self.z=output
        self.calls['image']+=1

    def optimizer(self):
        params=list(self.core.parameters())
        assert not any(p is self.core.rp_attention.weights for p in params)
        return torch.optim.Adam(params,lr=CFG['lr'],weight_decay=CFG['weight_decay'])

    def reset(self,diagnostic=False):
        import_core(self.core,self.start['core'])
        self.calls={k:0 for k in self.calls}
        for p in self.core.parameters():
            p.grad=None
        restore_rng(self.start['rng'])
        torch.use_deterministic_algorithms(diagnostic)

    def forward(self):
        return self.core(self.x,self.ph,self.affinity)

    def close(self):
        self.handle.remove()
        for h in self.count_handles: h.remove()
        self.__dict__.clear()
        torch.cuda.empty_cache()


def grad_norm(params):
    terms=[p.grad.detach().square().sum() for p in params if p.grad is not None]
    return float(torch.stack(terms).sum().sqrt()) if terms else 0.


def valid_result(group,seed,fold,audit_info,required=False):
    path=result_path(group,seed,fold)
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return None
    v=load_torch(path,artifact=True)
    assert v['state']=='complete' and (v['group'],v['seed'],v['fold'])==(group,seed,fold)
    assert v['code_sha256']==audit_info['code_sha256'] and v['input_sha256']==audit_info['anchors'][f'{seed}/{fold}']['input_sha256']
    assert v['best_core_sha256']==core_hash(v['core'])
    sp=load_torch(anchor(seed,fold)/'splits.pt')
    assert np.array_equal(v['oof']['indices'],sp['test'])
    p=np.asarray(v['oof']['probability']); assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
    m=binary_metrics(v['oof']['labels'],p,v['oof']['prediction'])
    assert max(abs(m[k]-v['metrics'][k]) for k in m)<1e-12
    return v


def train_one(group,seed,fold,a):
    if valid_result(group,seed,fold,a) is not None:
        log('skip_valid',group=group,seed=seed,fold=fold); return
    assert a['formal_authorized']
    assert group=='E0' or a['baseline_accepted']
    assert group in ('E0','E1') or a['d4_accepted']
    l=Loaded(seed,fold,a)
    try:
        objective=PrototypeObjective(l.prepared,group,CFG,l.opt.device) if group!='E0' else None
        optimizer=l.optimizer(); stopper=EarlyStopping(patience=CFG['early_stop'],save=False)
        l.reset()
        assert rng_hash(capture_rng())==l.anchor['pairing']['classifier_start_rng']
        assert not torch.are_deterministic_algorithms_enabled()
        initial_order=[n for n,p in l.core.named_parameters()]
        bn_start=int(l.core.clf[2].num_batches_tracked)
        history=[]; best=None; best_loss=1e10
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started=time.perf_counter(); started_utc=utc()
        for epoch in range(CFG['epochs']):
            l.core.train(); optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                logits,graph=l.forward()
            ce=F.cross_entropy(logits[l.sp['train']],l.y[l.sp['train']]); native=ce+graph
            total=native
            detail=None; scale_stats=None
            auxiliary=None
            sample=epoch<3 or epoch%CFG['diagnostic_frequency']==0
            record={'epoch':epoch,'train_ce':float(ce.detach()),'train_graph':float(graph.detach())}
            if objective is not None:
                auxiliary,detail=objective(l.z,epoch)
                weight,scale_stats=gradient_scale(native,auxiliary,l.z,epoch,CFG)
                if float(weight)!=0:
                    total=native+weight*auxiliary
                record.update(auxiliary_loss=float(auxiliary.detach()),**scale_stats)
                if sample:
                    record['prototype']=objective.diagnostics(detail)
                    params=[p for p in l.core.img_unet.parameters() if p.requires_grad]
                    ng=torch.autograd.grad(native,params,retain_graph=True,allow_unused=True)
                    ag=torch.autograd.grad(auxiliary,params,retain_graph=True,allow_unused=True)
                    nn=sum(float(g.detach().square().sum()) for g in ng if g is not None)**.5
                    an=sum(float(g.detach().square().sum()) for g in ag if g is not None)**.5
                    record['image_parameter_gradients']={'native':nn,'aux_unweighted':an,'aux_weighted':float(weight)*an,
                                                        'ratio':float(weight)*an/max(nn,1e-12)}
            if not bool(torch.isfinite(total)):
                raise FloatingPointError('Nonfinite train objective')
            total.backward()
            if sample:
                record['parameter_gradient_norms']={n:grad_norm(m.parameters()) for n,m in
                    [('image',l.core.img_unet),('phenotype',l.core.ph_unet),('attention',l.core.mm_attention),('classifier',l.core.clf)]}
                for n,p in l.core.named_parameters():
                    if p.grad is not None and not bool(torch.isfinite(p.grad).all()):
                        raise FloatingPointError(f'Nonfinite gradient {n}')
            optimizer.step()
            assert int(l.core.clf[2].num_batches_tracked)==bn_start+epoch+1
            l.core.eval()
            with torch.set_grad_enabled(False):
                val,vg=l.forward()
            vc=F.cross_entropy(val[l.sp['val']],l.y[l.sp['val']]); vl=vc+vg
            if not bool(torch.isfinite(vl)):
                raise FloatingPointError('Nonfinite validation loss')
            assert int(l.core.clf[2].num_batches_tracked)==bn_start+epoch+1
            record.update(val_ce=float(vc),val_graph=float(vg),val_loss=float(vl),train_total=float(total.detach()))
            history.append(record)
            if float(vl)<best_loss:
                best_loss=float(vl)
                best={'core':export_core(l.core),'epoch':epoch,'val_loss':best_loss,'record':record.copy()}
            stopper(vl,l.core)
            if epoch%100==0:
                log('progress',group=group,seed=seed,fold=fold,epoch=epoch,val_loss=float(vl))
            if stopper.early_stop:
                break
        assert all(v==2*len(history) for v in l.calls.values()), l.calls
        if objective is not None:
            assert objective.ema_updates == (len(history) if CFG['groups'][group]['ema'] else 0)
        torch.cuda.synchronize(); seconds=time.perf_counter()-started
        assert best is not None
        end_rng=capture_rng()
        import_core(l.core,best['core']); l.core.eval()
        before=capture_rng()
        assert payload_hash(before)==payload_hash(end_rng), 'Checkpoint loading consumed random state'
        with torch.set_grad_enabled(True):
            test,_=l.forward()
        after=capture_rng()
        idx=l.sp['test']; logits=test.detach().cpu().numpy()[idx]
        probs=torch.softmax(test.detach(),dim=1).cpu().numpy()[idx,0]
        oof={'indices':idx,'ids':[l.ctx['ids'][int(i)] for i in idx], 'labels':l.ctx['labels'][idx],
             'logits':logits,'probability':probs,'prediction':logits.argmax(1)}
        z=l.z.detach().cpu().numpy()
        # These fixed-checkpoint summaries are post-hoc, never fed back into training.
        tr=l.sp['train']; means=np.stack([z[tr][l.ctx['labels'][tr]==k].mean(0) for k in (0,1)])
        dist=((z[idx,None]-means[None])**2).sum(-1)
        margin=dist[np.arange(len(idx)),1-oof['labels']]-dist[np.arange(len(idx)),oof['labels']]
        result={'state':'complete','group':group,'seed':seed,'fold':fold,'started_utc':started_utc,'completed_utc':utc(),
                'code_sha256':a['code_sha256'],'input_sha256':l.anchor['input_sha256'],
                'data_input_digest':l.input_digest,'pairing':l.anchor['pairing'],'prototype_preparation':l.prepared['info'],
                'core':best['core'],'best_core_sha256':core_hash(best['core']),'best_epoch':best['epoch'],
                'best_val_loss':best_loss,'best_record':best['record'],'epochs':history,'epochs_run':len(history),
                'training_seconds':seconds,'oof':oof,'metrics':binary_metrics(oof['labels'],probs,oof['prediction']),
                'rng_before_test':before,'rng_after_test':after,'formal_start_rng_sha256':rng_hash(l.start['rng']),
                'formal_test_grad_enabled':True,'physical_gpu':int(os.environ.get('CUDA_VISIBLE_DEVICES','0')),
                'device':torch.cuda.get_device_name(0),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes':torch.cuda.max_memory_reserved(), 'optimizer_parameter_order':initial_order,
                'new_trainable_parameters':0,'rfe_fit_count':0,'vae_train_count':0,
                'configuration':CFG,'forward_counts':l.calls.copy(),
                'ema_updates':objective.ema_updates if objective else 0,
                'ema_shape':list(objective.ema.shape) if objective and objective.ema is not None else None,
                'class_calibration':CFG['groups'][group]['class_calibration'],
                'posthoc_image_margin':margin,'posthoc_image_prototype_probability':torch.softmax(torch.from_numpy(-dist),dim=1).numpy()[:,0],
                'bn_best_batches':int(best['core']['state']['clf.2.num_batches_tracked']),
                'best_core_change':parameter_changes(best['core'],l.start['core'])}
        atomic_torch(result,result_path(group,seed,fold))
        valid_result(group,seed,fold,a,required=True)
        log('complete',group=group,seed=seed,fold=fold,epochs=len(history),best_epoch=best['epoch'],seconds=seconds,physical_gpu=result['physical_gpu'])
    finally:
        l.close()


def parameter_changes(saved,initial):
    out={}
    for prefix in ('img_unet.','ph_unet.','mm_attention.','clf.'):
        keys=[k for k in saved['state'] if k.startswith(prefix) and saved['state'][k].is_floating_point()]
        d=sum(float((saved['state'][k].double()-initial['state'][k].double()).square().sum()) for k in keys)**.5
        n=sum(float(initial['state'][k].double().square().sum()) for k in keys)**.5
        out[prefix]={'absolute_l2':d,'relative_l2':d/max(n,1e-12)}
    return out


def compare_exact(a,b,label):
    if payload_hash(a)!=payload_hash(b):
        raise AssertionError(f'Exact native equivalence failed: {label}')


def trajectory(l,mode):
    """Official train() is compiled verbatim from AST, without its main block."""
    l.reset(diagnostic=True)
    optimizer=l.optimizer(); trace=[]; calls=[0]
    def observe(module,args,out):
        logits,graph=out
        assert tuple(l.z.shape)==(871,16)
        trace.append({'phase':'train' if module.training else 'val','logits':array_hash(logits.detach().cpu().numpy()),
                      'graph':float(graph.detach()),'z':array_hash(l.z.detach().cpu().numpy()),
                      'topk':[[array_hash(p.detach().cpu().numpy()) for p in e.perms] for e in (l.core.img_unet,l.core.ph_unet)],
                      'bn':state_hash(l.core.clf[2].state_dict()),'rng':payload_hash(capture_rng()),
                      'CE':float(F.cross_entropy(logits[l.sp['train' if module.training else 'val']],l.y[l.sp['train' if module.training else 'val']]).detach())})
    def clf_hook(module,args,out):
        calls[0]+=1
    h=l.core.register_forward_hook(observe); ch=l.core.clf.register_forward_hook(clf_hook)
    original_step=optimizer.step
    def step(*args,**kwargs):
        trace[-1]['gradients']=state_hash({n:p.grad for n,p in l.core.named_parameters() if p.grad is not None})
        result=original_step(*args,**kwargs)
        trace[-1]['updated_core']=core_hash(export_core(l.core))
        trace[-1]['adam']=payload_hash(optimizer.state_dict())
        return result
    optimizer.step=step
    try:
        if mode=='official':
            from utils.metrics import accuracy,auc,metrics
            fn=next(n for n in ast.parse(read_bytes(ROOT/'train_mm_gtunets.py')).body if isinstance(n,ast.FunctionDef) and n.name=='train')
            env={'torch':torch,'np':np,'os':os,'model':l.core,'optimizer':optimizer,'loss_fn':F.cross_entropy,
                 'x':l.x,'ph_features':l.ph,'affinity_graphs':l.affinity,'labels':l.y,'y':l.ctx['labels'],
                 'train_ind':l.sp['train'],'val_ind':l.sp['val'],'fold':l.fold,
                 'early_stopping':EarlyStopping(patience=100,save=False),'accuracy':accuracy,'auc':auc,'metrics':metrics}
            for name in ('accs','corrects','aucs','sens','spes','f1'):
                env[name]=np.zeros(10)
            env['opt']=l.opt; l.opt.epoch=3
            exec(compile(ast.Module(body=[fn],type_ignores=[]),str(ROOT/'train_mm_gtunets.py'),'exec'),env)
            with redirect_stdout(io.StringIO()):
                env['train']()
            l.opt.epoch=CFG['epochs']
        else:
            objective=PrototypeObjective(l.prepared,'E1',CFG,l.opt.device) if mode=='zero_auxiliary' else None
            for epoch in range(CFG['preflight_steps']):
                l.core.train(); optimizer.zero_grad()
                out,graph=l.forward()
                native=F.cross_entropy(out[l.sp['train']],l.y[l.sp['train']])+graph
                if objective is not None:
                    auxiliary,_=objective(l.z)
                    gradient_scale(native,auxiliary,l.z,epoch,CFG)
                native.backward(); optimizer.step()
                l.core.eval()
                with torch.no_grad():
                    out,graph=l.forward()
                F.cross_entropy(out[l.sp['val']],l.y[l.sp['val']])+graph
        assert calls[0]==6 and len(trace)==6
        return trace
    finally:
        h.remove(); ch.remove()



def historical_oracle():
    env={'__name__':'read_only_D4_preflight_oracle'}
    exec(compile(read_bytes(HIST/'prototype.py'),str(HIST/'prototype.py'),'exec'),env)
    return env,read_json(HIST/'config.json')


def d4_trajectory(l, historical):
    l.reset(diagnostic=True)
    optimizer=l.optimizer()
    env,cfg=historical_oracle()
    prepared=env['prepare'](l.ctx['labels'],l.meta,l.sp['train'],l.seed,l.fold,cfg)
    compare_exact(prepared,l.prepared,'historical private prototype folds and shuffle')
    objective=env['PrototypeObjective'](prepared,'D4',cfg,l.opt.device) if historical else PrototypeObjective(l.prepared,'E1',CFG,l.opt.device)
    scale_fn=env['gradient_scale'] if historical else gradient_scale
    traces=[]
    for epoch in range(CFG['preflight_steps']):
        l.core.train();optimizer.zero_grad()
        out,graph=l.forward()
        ce=F.cross_entropy(out[l.sp['train']],l.y[l.sp['train']]);native=ce+graph
        auxiliary,d=objective(l.z) if historical else objective(l.z,epoch)
        weight,stats=scale_fn(native,auxiliary,l.z,epoch,cfg if historical else CFG)
        ref_keys=('mu','global_mu','teacher','mean','std','tau','ess','beta','r','r_site','support','valid','local_den')
        auxgrad=torch.autograd.grad(auxiliary,l.z,retain_graph=True)[0]
        row={'epoch':epoch,'logits':out.detach().cpu().clone(),'ce':float(ce.detach()),'graph':float(graph.detach()),
             'prototype':{k:d[k].detach().cpu().clone() for k in ref_keys},
             'kernel':objective.kernel.detach().cpu().clone(),'reference':objective.reference.detach().cpu().clone(),
             'auxiliary':float(auxiliary.detach()),'gradient':auxgrad.detach().cpu().clone(),'scale':stats}
        total=native+weight*auxiliary if float(weight)!=0 else native
        total.backward()
        row['parameter_gradients']=state_hash({n:p.grad for n,p in l.core.named_parameters() if p.grad is not None})
        optimizer.step()
        row['core_after_adam']=core_hash(export_core(l.core));row['adam']=payload_hash(optimizer.state_dict())
        l.core.eval()
        with torch.no_grad(): vo,vg=l.forward()
        row['val_logits']=vo.detach().cpu().clone()
        row['val_loss']=float(F.cross_entropy(vo[l.sp['val']],l.y[l.sp['val']])+vg)
        row['bn']=state_hash(l.core.clf[2].state_dict())
        row['topk']=[[array_hash(p.cpu().numpy()) for p in unet.perms] for unet in (l.core.img_unet,l.core.ph_unet)]
        row['rng']=rng_hash(capture_rng())
        traces.append(row)
    assert all(n==6 for n in l.calls.values()),l.calls
    return {'trace_sha256':payload_hash(traces),'step_sha256':[payload_hash(x) for x in traces],
            'auxiliary_loss':[x['auxiliary'] for x in traces],'lambda_eff':[x['scale']['lambda_eff'] for x in traces],
            'calls':l.calls.copy()}


def mechanism_checks(l):
    l.reset(diagnostic=True);l.core.train();l.core.zero_grad()
    logits,graph=l.forward();z=l.z
    native=F.cross_entropy(logits[l.sp['train']],l.y[l.sp['train']])+graph
    baseline_input=l.input_digest; rng_before=rng_hash(capture_rng())
    objectives={};outputs={};records={}
    outside=np.setdiff1d(np.arange(871),l.sp['train'])
    for group in list(CFG['groups'])[1:]:
        obj=PrototypeObjective(l.prepared,group,CFG,l.opt.device);objectives[group]=obj
        loss,d=obj(z,0);outputs[group]=(loss,d)
        g=torch.autograd.grad(loss,z,retain_graph=True)[0]
        assert torch.isfinite(g).all() and float(g.norm())>1e-12
        assert torch.equal(g[outside],torch.zeros_like(g[outside]))
        for key in ('mu','global_mu','local_mu','teacher','raw_teacher','mean','std','tau','ess','beta'):
            assert not d[key].requires_grad and d[key].grad_fn is None,key
        weight,scale=gradient_scale(native,loss,z,9,CFG)
        if obj.spec['ema']:
            assert obj.ema.shape==(len(l.sp['train']),16) and obj.ema_updates==1
            assert torch.equal(obj.ema,z[l.train].detach()) and not obj.ema.requires_grad
        else: assert obj.ema is None and obj.ema_updates==0
        if obj.spec['class_calibration']:
            assert d['beta'].shape==(len(l.train),2)
            expected=d['ess']/(d['ess']+CFG['class_support_constant'])
            expected=torch.where(d['local_den']>CFG['epsilon'],expected,torch.zeros_like(expected))
            assert torch.equal(d['beta'],expected) and torch.all((d['beta']>=0)&(d['beta']<=1))
        records[group]={'loss':float(loss.detach()),'query_gradient':float(g.norm()),**scale,
                        'mechanism':obj.diagnostics(d),'train_only_gradients':True,'teacher_detached':True}
    for a,b in (('E1','E2'),('E3','E4')):
        compare_exact(outputs[a][0],outputs[b][0],'epoch0 EMA identity loss')
        compare_exact(outputs[a][1]['mu'],outputs[b][1]['mu'],'epoch0 EMA identity prototype')
    for key in ('raw_teacher','mean','std','teacher','global_mu','local_mu','ess','local_den'):
        compare_exact(outputs['E1'][1][key],outputs['E3'][1][key],'E1/E3 only beta '+key)
    compare_exact(objectives['E4'].reference,objectives['E5'].reference,'E4/E5 reference and labels')
    compare_exact(objectives['E4'].y,objectives['E5'].y,'E4/E5 true labels')
    assert {k:v for k,v in CFG['groups']['E4'].items() if k not in ('name','shuffled')}=={k:v for k,v in CFG['groups']['E5'].items() if k not in ('name','shuffled')}
    assert not torch.equal(objectives['E4'].kernel,objectives['E5'].kernel)
    assert l.input_digest==baseline_input and rng_hash(capture_rng())==rng_before
    z2=z.detach()+torch.arange(871,device=z.device)[:,None]*.0001+torch.arange(16,device=z.device)[None,:]*.01
    for group in ('E2','E4','E5'):
        obj=objectives[group];previous=obj.ema.clone()
        current=z2[l.train]
        expected=previous.clone().mul_(CFG['ema_decay']).add_(current,alpha=1-CFG['ema_decay'])
        loss,d=obj(z2.requires_grad_(),1)
        assert torch.equal(obj.ema,expected) and obj.ema_updates==2
        assert not torch.equal(obj.ema,current)
        assert torch.equal(d['mean'],expected.mean(0))
        assert torch.equal(d['std'],expected.std(0,unbiased=False).clamp_min(CFG['embedding_std_floor']))
        assert float(torch.autograd.grad(loss,z2,retain_graph=True)[0].norm())>0
        records[group]['epoch1_formula_exact']=True
    changed_labels=l.ctx['labels'].copy();changed_labels[outside]=1-changed_labels[outside]
    changed_meta={k:v.copy() for k,v in l.meta.items()}
    changed_meta['age'][outside]=99;changed_meta['sex'][outside]=-1;changed_meta['site'][outside]='HELDOUT'
    prepared=prepare(changed_labels,changed_meta,l.sp['train'],l.seed,l.fold,CFG)
    compare_exact(prepared,l.prepared,'heldout labels/conditions do not fit prototype')
    changed=z.detach().clone();changed[outside]=10000
    for group in list(CFG['groups'])[1:]:
        a=PrototypeObjective(l.prepared,group,CFG,z.device);b=PrototypeObjective(prepared,group,CFG,z.device)
        la,da=a(z,0);lb,db=b(changed,0)
        compare_exact(la,lb,'heldout embedding boundary')
        compare_exact(da['mu'],db['mu'],'heldout prototype boundary')
    native.backward()
    assert grad_norm(l.core.img_unet.parameters())>0 and grad_norm(l.core.ph_unet.parameters())>0 and grad_norm(l.core.clf.parameters())>0
    assert all(n==1 for n in l.calls.values())
    assert rng_hash(capture_rng())==rng_before
    return {'groups':records,'no_additional_native_forward':True,'no_extra_rng':True,'new_parameters':0,
            'heldout_perturbation_invariant_for_auxiliary_given_fixed_native_z':True}


def synthetic_checks(device):
    meta={'age':np.array([np.nan,-1,8,9,100,16,18,20,23.,12,27,35]),
          'sex':np.array([-1,1,2,1,2,-1,1,2,1,2,1,2]),
          'site':np.array(['A','A','A','B','B','B','C','C','D','D','E','E'])}
    y=np.array([0,1,0,1,0,1,0,1,0,1,0,1])
    p=prepare(y,meta,np.arange(12),911,0,CFG)
    for group in list(CFG['groups'])[1:]:
        obj=PrototypeObjective(p,group,CFG,device)
        z=torch.ones((12,16),device=device,requires_grad=True)
        loss,d=obj(z,0);g=torch.autograd.grad(loss,z)[0]
        assert torch.isfinite(loss) and torch.equal(g,torch.zeros_like(g))
        assert float(d['tau'])==CFG['tau_floor']
    p_one=prepare(np.zeros(12,int),meta,np.arange(12),911,0,CFG)
    for group in ('E1','E3','E4'):
        obj=PrototypeObjective(p_one,group,CFG,device)
        z=torch.ones((12,16),device=device,requires_grad=True)
        loss,d=obj(z,0);g=torch.autograd.grad(loss,z)[0]
        assert float(loss)==0 and not d['valid'].any() and torch.equal(g,torch.zeros_like(g))
    for group in ('E3','E4'):
        obj=PrototypeObjective(p,group,CFG,device);obj.weights.zero_()
        z=torch.arange(192,device=device,dtype=torch.float32).reshape(12,16).requires_grad_()
        loss,d=obj(z,0)
        assert torch.isfinite(loss) and torch.all(d['beta']==0)
        assert torch.equal(d['mu'],d['global_mu'])
    ess=torch.tensor([0.,1.,32.,128.],device=device);beta=ess/(ess+32)
    assert beta[0]==0 and torch.all(beta[1:]>beta[:-1]) and beta[-1]<1
    zero=(z*0).sum();w,stats=gradient_scale(zero,zero,z,9,CFG)
    assert float(w)==0 and stats['weighted_z_gradient_ratio']==0
    return {'missing_age_unknown_sex':True,'empty_class_zero_loss':True,'constant_embedding':True,
            'zero_local_support_global_fallback':True,'beta_monotonic':True,'zero_gradient_scale':True}


def preflight_one(seed,fold,a):
    l=Loaded(seed,fold,a)
    try:
        official=trajectory(l,'official')
        fresh=trajectory(l,'E0');zero=trajectory(l,'zero_auxiliary')
        compare_exact(official,fresh,'official AST/E0 full 3 steps')
        compare_exact(official,zero,'zero auxiliary RNG/Adam/BN')
        old=d4_trajectory(l,True);new=d4_trajectory(l,False)
        compare_exact(old,new,'historical D4/E1 full 3 steps')
        checks=mechanism_checks(l);synthetic=synthetic_checks(l.opt.device)
        out={'passed':True,'seed':seed,'fold':fold,'utc':utc(),'physical_gpu':int(os.environ['CUDA_VISIBLE_DEVICES']),
             'native_3step_trace':official,'D4_3step':old,'mechanisms':checks,'synthetic':synthetic,
             'maximum_native_error':0,'maximum_D4_error':0,'historical_oracle_sha256':sha_file(HIST/'prototype.py')}
        log('preflight_passed',seed=seed,fold=fold)
        return out
    finally: l.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['audit','preflight','train'])
    p.add_argument('--group',choices=list(CFG['groups']))
    p.add_argument('--seed',type=int,default=911);p.add_argument('--fold',type=int)
    args=p.parse_args()
    if args.action=='audit':audit();return
    a=guard()
    assert args.seed==911
    if args.action=='preflight':
        out=preflight_one(args.seed,args.fold,a)
        with task_lock(990):
            a=read_json(BASE/'manifest.json');a['preflight'][f'{args.seed}/{args.fold}']=out
            atomic_json(a,BASE/'manifest.json')
        return
    slot=int(args.group[1:])*10+args.fold
    with task_lock(slot):
        train_one(args.group,args.seed,args.fold,a)


if __name__=='__main__':
    try: main()
    except Exception as e:
        log('worker_error',type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
        raise

