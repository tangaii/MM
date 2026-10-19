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
ROOT = BASE.parents[1]
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
    if os.environ.get('PCDCA_LOG_STDOUT') == '1':
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
    if (BASE/'audit.json').exists():
        saved = read_json(BASE/'audit.json')
        assert saved['code_sha256'] == code_hash(), 'Registered experiment code changed'
        return saved
    ctx = load_torch(OLD/'artifacts/context.pt')
    meta = metadata(ctx)
    assert len(ctx['ids']) == len(set(ctx['ids'])) == 871
    protected = {}
    # Hash source, audited reports/configs, original cached data and all historical anchors/OOF/checkpoints.
    names = list(ROOT.glob('model/*.py'))+list(ROOT.glob('utils/*.py'))+[ROOT/n for n in SOURCE]+[ROOT/'prompt']
    for folder in ('DRPA_20261014','PCND_20261016','PCND_DualStage_20261016','LCRA_20261018',
                   'LCRA_Attribution_20261019','GBSA_Pilot_20261019','GBSA_Diagnostic_20261019','PCCI_20261015'):
        d = ROOT/'experiments'/folder
        names += [p for p in d.glob('*') if p.is_file() and p.suffix in ('.py','.json','.md')]
    names += list((OLD/'artifacts').rglob('*'))
    for s in CFG['seeds']:
        for f in range(10):
            d = OLD/f'outputs/E0/seed_{s}/fold_{f}'
            names += [d/n for n in ('best.pt','oof.csv','metrics.json','epochs.jsonl')]
    data = ROOT/CFG['data_folder']
    names += [data/'id.txt',data/'Phenotypic_V1_0b_preprocessed1.csv']
    names += [p for p in (data/'save_tensor').glob('*') if p.is_file()]
    names += [Path('/tmp/CMCV_20261019/report.md')]
    for p in sorted(set(p for p in names if p.is_file()), key=str):
        protected[str(p)] = sha_file(p)
    anchors = {}
    raw = torch.as_tensor(load_torch(data/'save_tensor/up_triangles.pt'))
    assert tuple(raw.shape) == (871,6670) and bool(torch.isfinite(raw).all())
    for s in CFG['seeds']:
        x = load_torch(anchor(s,0).parent/'selected_features.pt')
        assert tuple(x.shape) == (871,500) and bool(torch.isfinite(x).all())
        support = np.load(io.BytesIO(read_bytes(anchor(s,0).parent/'rfe_support.npy')))
        assert support.shape == (500,) and np.issubdtype(support.dtype,np.integer)
        assert len(np.unique(support)) == 500 and support.min() >= 0 and support.max() < 6670
        assert torch.equal(raw[:,support].to(x.dtype),x), 'Historical RFE indices do not reproduce cached features'
        counts = np.zeros(871,int)
        for f in range(10):
            a = anchor(s,f)
            sp = load_torch(a/'splits.pt'); start = load_torch(a/'train_start.pt')
            fp = read_json(a/'fingerprints.json')
            union = np.concatenate([sp[k] for k in SUBSETS])
            assert len(union) == 871 and np.array_equal(np.sort(union),np.arange(871))
            counts[sp['test']] += 1
            for k in SUBSETS:
                assert array_hash(sp[k]) == fp['split_fingerprints'][k]
            assert array_hash(x.numpy()) == fp['features_fingerprint']
            assert core_hash(start['core']) == fp['train_start_core_fingerprint']
            assert rng_hash(start['rng']) == fp['classifier_start_rng']
            vae = a.parent/f'vae/fold{f}_pretrain.pth'
            assert sha_file(vae) == start['vae_sha256'] == fp['vae_sha256']
            prep = prepare(ctx['labels'],meta,sp['train'],s,f,CFG)
            paths = [a/'splits.pt', a/'train_start.pt',a/'fingerprints.json',a.parent/'selected_features.pt',a.parent/'rfe_support.npy',vae]
            anchors[f'{s}/{f}'] = {'input_sha256':{str(p):sha_file(p) for p in paths}, 'pairing':fp,
                                  'prototype':prep['info'], 'physical_gpu':f%2}
        assert np.all(counts == 1)
    out = {'experiment':CFG['experiment'],'started_utc':utc(),'code_sha256':code_hash(),
           'source_sha256':{n:sha_file(ROOT/n) for n in SOURCE}, 'protected_sha256':protected,
           'handoff_before_sha256':sha_file(ROOT/'handoff.md'), 'handoff_before_bytes':len(read_bytes(ROOT/'handoff.md')),
           'context_sha256':sha_file(OLD/'artifacts/context.pt'), 'anchors':anchors,
           'data':{'n':871,'class_counts':[403,468],'sites':20,'image_shape':[871,500],
                   'sex_csv':'1=male,2=female verified by CSV/context and paper class-sex counts',
                   'age_range':[float(meta['age'].min()),float(meta['age'].max())]},
           'storage':storage_check(), 'environment':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__},
           'preflight':{}, 'formal_authorized':False, 'baseline_accepted':False,
           'execution':{},'scientific_stop':None,
           'literature':{'MM_GTUNets':'https://arxiv.org/html/2406.14455v3',
                         'PhenoBrain':'https://proceedings.mlr.press/v306/meng26f.html',
                         'MGDR':'https://papers.miccai.org/miccai-2024/paper/2494_paper.pdf',
                         'To_Align':'https://arxiv.org/html/2511.12121v1',
                         'WSSADN':'https://papers.miccai.org/miccai-2024/paper/0336_paper.pdf',
                         'Prototypical_Networks':'https://papers.nips.cc/paper/2017/file/cb8da6767461f2812ae4290eac7cbc42-Paper.pdf'}}
    atomic_json(out,BASE/'audit.json')
    log('audit_passed',anchors=30,protected_files=len(protected),code_sha256=out['code_sha256'])
    return out


def guard():
    a = read_json(BASE/'audit.json')
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
        self.handle=self.core.img_unet.register_forward_hook(self.capture)
        self.input_digest=payload_hash({'x':self.x_cpu,'ph':self.ph_cpu,'affinity':self.affinity_cpu,'labels':self.ctx['labels'],'ids':self.ctx['ids']})

    def capture(self,module,args,output):
        assert tuple(output.shape)==(871,16)
        self.z=output

    def optimizer(self):
        params=list(self.core.parameters())
        assert not any(p is self.core.rp_attention.weights for p in params)
        return torch.optim.Adam(params,lr=CFG['lr'],weight_decay=CFG['weight_decay'])

    def reset(self,diagnostic=False):
        import_core(self.core,self.start['core'])
        for p in self.core.parameters():
            p.grad=None
        restore_rng(self.start['rng'])
        torch.use_deterministic_algorithms(diagnostic)

    def forward(self):
        return self.core(self.x,self.ph,self.affinity)

    def close(self):
        self.handle.remove()
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
    assert group=='D0' or a['baseline_accepted']
    l=Loaded(seed,fold,a)
    try:
        objective=PrototypeObjective(l.prepared,group,CFG,l.opt.device) if group!='D0' else None
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
                auxiliary,detail=objective(l.z)
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


def frozen_one(group,seed,fold,a):
    v=valid_result(group,seed,fold,a,True)
    if 'frozen' in v:
        log('skip_frozen',group=group,seed=seed,fold=fold); return
    l=Loaded(seed,fold,a)
    try:
        rows=[]
        for base in CFG['frozen_draw_bases']:
            import_core(l.core,v['core']); l.core.eval()
            restore_rng(l.start['rng'])
            draw=int(hashlib.sha256(f'{seed}/{fold}/PCCI-repeat-{base}'.encode()).hexdigest()[:15],16)%2**32
            random.seed(draw); np.random.seed(draw); torch.manual_seed(draw); torch.cuda.manual_seed(draw)
            before=core_hash(export_core(l.core))
            with torch.set_grad_enabled(True):
                logits,_=l.forward()
            # In-place RP constraints are native; restore from the same checkpoint before every draw.
            after=export_core(l.core)
            drift={k:float((after['state'][k]-v['core']['state'][k]).abs().max()) for k in after['state'] if not torch.equal(after['state'][k],v['core']['state'][k])}
            assert all(k.startswith('rp_attention.convs.') for k in drift), drift
            idx=l.sp['test']; out=logits.detach().cpu().numpy()[idx]
            prob=logits.detach().softmax(1).cpu().numpy()[idx,0]
            rows.append({'base':base,'draw_seed':draw,'probability':prob,'prediction':out.argmax(1),
                         'metrics':binary_metrics(v['oof']['labels'],prob,out.argmax(1)),
                         'native_inplace_drift':drift,'restored_core_sha256':before})
        probability=np.stack([r['probability'] for r in rows])
        prediction=np.stack([r['prediction'] for r in rows])
        v['frozen']={'rows':rows,'probability_mean':probability.mean(0),'probability_sd':probability.std(0),
                     'mean_probability_metrics':binary_metrics(v['oof']['labels'],probability.mean(0)),
                     'label_flip_fraction':float(np.mean(prediction.min(0)!=prediction.max(0))),
                     'auxiliary_not_used_in_inference':True}
        atomic_torch(v,result_path(group,seed,fold))
        log('frozen_complete',group=group,seed=seed,fold=fold)
    finally:
        l.close()


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
            objective=PrototypeObjective(l.prepared,'D4',CFG,l.opt.device) if mode=='zero_auxiliary' else None
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


def preflight_one(seed,fold,a):
    l=Loaded(seed,fold,a)
    try:
        oracle=trajectory(l,'official')
        replay=trajectory(l,'D0')
        zero=trajectory(l,'zero_auxiliary')
        compare_exact(oracle,replay,'official vs D0 3 train/val steps')
        compare_exact(oracle,zero,'official vs zero auxiliary 3 train/val steps')
        l.reset(diagnostic=True)
        l.core.train(); l.core.zero_grad(); logits,graph=l.forward()
        native=F.cross_entropy(logits[l.sp['train']],l.y[l.sp['train']])+graph
        z=l.z
        records={}
        reference_sha={}
        for group in list(CFG['groups'])[1:]:
            obj=PrototypeObjective(l.prepared,group,CFG,l.opt.device)
            loss,detail=obj(z)
            g=torch.autograd.grad(loss,z,retain_graph=True)[0]
            assert bool(torch.isfinite(g).all()) and float(g.norm())>1e-12
            outside=np.setdiff1d(np.arange(871),l.sp['train'])
            assert torch.equal(g[outside],torch.zeros_like(g[outside]))
            assert all(not detail[k].requires_grad for k in ('mu','global_mu','teacher','mean','std','tau','ess','beta','r','r_site'))
            assert not any(p.grad is not None for p in l.core.parameters()), 'autograd.grad accumulated parameter .grad'
            weight,scale=gradient_scale(native,loss,z,9,CFG)
            ps=[p for p in l.core.img_unet.parameters() if p.requires_grad]
            pg=torch.autograd.grad(loss,ps,retain_graph=True,allow_unused=True)
            norm=sum(float(g.detach().square().sum()) for g in pg if g is not None)**.5
            assert norm>1e-12
            records[group]={'loss':float(loss.detach()),'image_parameter_grad':norm,**scale,'prototype':obj.diagnostics(detail)}
            reference_sha[group]=payload_hash({'weights':obj.weights,'global':obj.global_weights,'kernel':obj.kernel})
        native.backward()
        assert grad_norm(l.core.ph_unet.parameters())>0 and grad_norm(l.core.clf.parameters())>0
        # Alter heldout labels, conditions AND embeddings. The auxiliary is unchanged.
        altered_y=l.ctx['labels'].copy(); altered_meta={k:v.copy() for k,v in l.meta.items()}
        altered_y[outside]=1-altered_y[outside]
        altered_meta['age'][outside]=99; altered_meta['sex'][outside]=-1; altered_meta['site'][outside]='HELDOUT_CHANGED'
        prep=prepare(altered_y,altered_meta,l.sp['train'],seed,fold,CFG)
        compare_exact(l.prepared,prep,'heldout metadata and label boundary')
        zx=z.detach().clone(); zx[outside]=10000
        for group in list(CFG['groups'])[1:]:
            obj=PrototypeObjective(prep,group,CFG,l.opt.device)
            old,old_detail=obj(z.detach()); new,new_detail=obj(zx)
            compare_exact(old,new,'heldout embedding boundary')
            compare_exact(old_detail['mu'],new_detail['mu'],'heldout teacher boundary')
        edge_tests=synthetic_checks(l.opt.device)
        out={'passed':True,'seed':seed,'fold':fold,'official_ast_sha256':sha_file(ROOT/'train_mm_gtunets.py'),
             'exact_3step_trace':oracle,'maximum_native_error':0,'classifier_calls_per_trajectory':6,
             'gradient_checks':records,'reference_sha256':reference_sha,'heldout_perturbation_invariant':True,
             'synthetic':edge_tests,'physical_gpu':int(os.environ.get('CUDA_VISIBLE_DEVICES','0'))}
        log('preflight_passed',seed=seed,fold=fold)
        return out
    finally:
        l.close()


def synthetic_checks(device):
    # Includes singletons, unknown sex, missing ages, empty class support and constant embeddings.
    y=np.array([0,0,0,1,1,1,0,1,0])
    meta={'age':np.array([np.nan,-1,8,9,100,16,18,20,23.]),
          'sex':np.array([-1,1,2,1,2,-1,1,2,1]),'site':np.array(['A','A','B','B','C','C','D','E','F'])}
    p=prepare(y,meta,np.arange(9),1,0,CFG)
    obj=PrototypeObjective(p,'D6',CFG,device)
    z=torch.ones((9,16),device=device,requires_grad=True)
    loss,d=obj(z); g=torch.autograd.grad(loss,z)[0]
    assert torch.isfinite(loss) and torch.equal(g,torch.zeros_like(g))
    assert float(d['tau'])==CFG['tau_floor'] and torch.all(d['r']==.25)
    # A true one-class reference cannot invent a teacher from the query/heldout set.
    p=prepare(np.zeros(9,int),meta,np.arange(9),1,0,CFG)
    obj=PrototypeObjective(p,'D4',CFG,device)
    z=torch.ones((9,16),device=device,requires_grad=True)
    loss,d=obj(z); g=torch.autograd.grad(loss,z)[0]
    assert float(loss)==0 and torch.equal(g,torch.zeros_like(g)) and not d['valid'].any()
    gn=(z*0).sum(); aux=(z*0).sum()
    weight,stats=gradient_scale(gn,aux,z,9,CFG)
    assert float(weight)==0 and stats['weighted_z_gradient_ratio']==0
    # Hand-computable two-class distances and finite-difference query gradient.
    point=torch.tensor([[.3,-.2]],device=device,dtype=torch.float64,requires_grad=True)
    means=torch.tensor([[[-1.,0.],[1.,0.]]],device=device,dtype=torch.float64)
    fn=lambda q:F.cross_entropy(-(q[:,None]-means).square().sum(-1),torch.tensor([0],device=device))
    analytic=torch.autograd.grad(fn(point),point)[0]
    numeric=[]
    for j in range(2):
        plus=point.detach().clone(); minus=plus.clone(); plus[0,j]+=1e-5; minus[0,j]-=1e-5
        numeric.append(float((fn(plus)-fn(minus))/2e-5))
    assert np.allclose(analytic.cpu().numpy()[0],numeric,atol=1e-8,rtol=1e-8)
    assert float(fn(torch.tensor([[-1.,0.]],device=device)))<float(fn(torch.tensor([[1.,0.]],device=device)))
    return {'unknown_sex':True,'missing_age':True,'singleton':True,'constant_embedding':True,
            'empty_class_zero':True,'zero_grad_scale':True,'finite_difference_max_error':float(np.max(np.abs(analytic.cpu().numpy()[0]-numeric))),
            'correct_class_direction':True}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['audit','preflight','train','frozen'])
    p.add_argument('--group',choices=list(CFG['groups']))
    p.add_argument('--seed',type=int);p.add_argument('--fold',type=int)
    args=p.parse_args()
    if args.action=='audit':
        audit();return
    a=guard()
    if args.action=='preflight':
        # This process alone uses diagnostic deterministic arithmetic.
        out=preflight_one(args.seed,args.fold,a)
        with task_lock(990):
            a=read_json(BASE/'audit.json');a['preflight'][f'{args.seed}/{args.fold}']=out
            atomic_json(a,BASE/'audit.json')
        return
    slot=int(args.group[1:])*30+CFG['seeds'].index(args.seed)*10+args.fold
    with task_lock(slot):
        if args.action=='train':
            train_one(args.group,args.seed,args.fold,a)
        else:
            frozen_one(args.group,args.seed,args.fold,a)


if __name__=='__main__':
    try:
        main()
    except Exception as e:
        log('worker_error',type=type(e).__name__,message=str(e),traceback=traceback.format_exc())
        raise
