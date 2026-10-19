"""Storage audit, two-PPU queue and read-only checkpoint experiments."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
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
import resource
import subprocess
import sys
import time
import traceback
import warnings
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[1]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(BASE))
import numpy as np
import torch
from torch_geometric.nn import TransformerConv
from opt import OptInit
from model.rp_graph import create_reward_penalty_graph
from frozen_model import FrozenCore,LCRA,FrozenWrapper,prepare_inputs,phenotype_kernel,array_sha,js_divergence,random_signs
torch.set_num_threads(1)
warnings.filterwarnings('ignore',message='To copy construct from a tensor')
warnings.filterwarnings('ignore',message='Please use the new API settings')
CFG=json.loads((BASE/'config.json').read_text())
HIST=ROOT/CFG['history']; OLD=ROOT/CFG['baseline']
HCFG=json.loads((HIST/'config.json').read_text())
PARTS=('train','val','test'); MODES=tuple(CFG['conditions'])


def utc(): return datetime.now(timezone.utc).isoformat()


def read_bytes(path):
    for attempt in range(6):
        try: return Path(path).read_bytes()
        except FileNotFoundError:
            if attempt==5: raise
            time.sleep(.1*(attempt+1))


def read_json(path): return json.loads(read_bytes(path))


def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def state_hash(state):
    h=hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode()); h.update(array_sha(state[k].detach().cpu().numpy()).encode())
    return h.hexdigest()


def core_hash(saved): return state_hash({**saved['state'],'unregistered_rp_weights':saved['rp_weights']})


def verify_manifest(manifest):
    def check(item):
        name,row=item
        assert sha_file(name)==row['sha256'],f'Protected historical file changed: {name}'
        return True
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(pool.map(check,manifest.items()))


def code_hash(): return digest({n:sha_file(BASE/n) for n in ('config.json','frozen_model.py','run.py','analyze.py')})


def atomic_bytes(data,path):
    path=Path(path); assert path.is_relative_to(BASE)
    path.parent.mkdir(exist_ok=True,parents=True)
    tmp=path.with_name(path.name+f'.tmp.{os.getpid()}')
    with tmp.open('wb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
    assert read_bytes(path)==data


def atomic_json(value,path): atomic_bytes((json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n').encode(),path)


def atomic_torch(value,path):
    value['payload_sha256']=payload_hash(value)
    stream=io.BytesIO(); torch.save(value,stream); atomic_bytes(stream.getvalue(),path)


def load_torch(path): return torch.load(path,map_location='cpu',weights_only=False,mmap=True)


def payload_hash(value):
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,dict):
            for key in sorted(v):
                if key=='payload_sha256':continue
                h.update(str(key).encode());visit(v[key])
        elif isinstance(v,(np.ndarray,torch.Tensor)):
            a=v.detach().cpu().numpy() if torch.is_tensor(v) else v
            h.update(array_sha(a).encode())
        elif isinstance(v,(list,tuple)):
            h.update(str(len(v)).encode())
            for x in v:visit(x)
        else:h.update((type(v).__name__+':'+repr(v)).encode())
    visit(value);return h.hexdigest()


def log(event,**values):
    line=json.dumps({'utc':utc(),'event':event,**values},ensure_ascii=False,allow_nan=False)
    with (BASE/'run.log').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX); f.write(line+'\n'); f.flush(); fcntl.flock(f,fcntl.LOCK_UN)
    print(line,flush=True)


@contextmanager
def lock(slot):
    with (BASE/'run.py').open('r+') as f:
        fcntl.lockf(f,fcntl.LOCK_EX|fcntl.LOCK_NB,1,slot)
        try: yield
        finally: fcntl.lockf(f,fcntl.LOCK_UN,1,slot)


def anchor(seed,fold): return OLD/'artifacts'/f'seed_{seed}'/f'fold_{fold}'


def history_path(seed,fold): return HIST/'runs/train/L3'/f's{seed}_f{fold}.pt'


def result_path(seed,fold): return BASE/'runs'/f's{seed}_f{fold}.pt'


def metadata(ctx):
    with (Path(HCFG['data_folder'])/'Phenotypic_V1_0b_preprocessed1.csv').open() as f:
        rows={r['SUB_ID']:r for r in csv.DictReader(f)}
    records=[rows[i] for i in ctx['ids']]
    assert len(set(ctx['ids']))==871
    assert np.array_equal(np.array([int(r['DX_GROUP'])-1 for r in records]),ctx['labels'])
    assert np.array_equal(np.bincount(ctx['labels']),[403,468])
    def number(v,default):
        try:return float(v)
        except (ValueError,TypeError):return default
    out={'ids':ctx['ids'],'age':np.array([number(r['AGE_AT_SCAN'],np.nan) for r in records]),
        'sex':np.array([number(r['SEX'],-1) for r in records]),'site':np.array([r['SITE_ID'] for r in records])}
    assert np.array_equal(out['site'],ctx['site_names'])
    return out


def storage_gate():
    probe=BASE/'storage.probe'; data=b'{"phase":0,"payload":"LCRA9"}\n'
    if probe.exists():
        assert read_json(probe)['payload']=='LCRA9'
        probe.unlink()
    with probe.open('xb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
    assert probe.read_bytes()==data
    f=os.open(BASE,os.O_RDONLY|os.O_DIRECTORY); os.fsync(f); os.close(f)
    # A second process reads while this process replaces the same filename.
    script="""import json,sys,time
from pathlib import Path
p=Path(sys.argv[1]);missing=0
for i in range(10):
 try: assert json.loads(p.read_bytes())['phase']>=0
 except FileNotFoundError: missing+=1
 time.sleep(.005)
print(missing)
"""
    p=subprocess.Popen([sys.executable,'-B','-c',script,str(probe)],stdout=subprocess.PIPE,text=True)
    for i in range(20): atomic_json({'phase':i,'payload':'LCRA9'},probe)
    out=p.communicate(timeout=60)[0]; assert p.returncode==0
    probe.unlink()
    with lock(200):
        child="import fcntl,sys;f=open(sys.argv[1],'r+');fcntl.lockf(f,fcntl.LOCK_EX|fcntl.LOCK_NB,1,200)"
        test=subprocess.run([sys.executable,'-B','-c',child,str(BASE/'run.py')],capture_output=True)
        assert test.returncode!=0 and b'BlockingIOError' in test.stderr
    return {'passed':True,'utc':utc(),'fsync_reopen_sha':True,'replace_visibility_reader_success':True,
        'reader_transient_missing':int(out.strip()),'lock_exclusion':True,'read_retry':'FileNotFoundError only, six bounded attempts; all other IO errors propagate'}


def audit():
    if (BASE/'audit.json').exists():
        a=read_json(BASE/'audit.json')
        if a['code_sha256']!=code_hash():
            assert not list((BASE/'runs').glob('*.pt')),'Code changed after diagnostic data: explicit reacceptance required'
            a.setdefault('pre_execution_code_revisions',[]).append({'utc':utc(),'previous_sha256':a['code_sha256'],
                'new_sha256':code_hash(),'reason':'Pre-execution frozen-adapter/IO fixes, including non-leaf unregistered RP freeze; no diagnostic artifact or neural update exists'})
            a['code_sha256']=code_hash();atomic_json(a,BASE/'audit.json')
        return a
    started=time.time(); storage=storage_gate()
    hist=read_json(HIST/'audit.json'); summ=read_json(HIST/'summary.json')
    manifest=dict(hist['manifest'])
    for name,expected in hist['source_sha256'].items():
        assert sha_file(ROOT/name)==expected
    for p in HIST.iterdir():
        if p.is_file():manifest[str(p)]={'sha256':sha_file(p),'bytes':p.stat().st_size}
    for name,row in summ['artifact_registry'].items():manifest[name]=row
    verify_manifest(manifest)
    ctx=load_torch(OLD/'artifacts/context.pt'); meta=metadata(ctx)
    anchors={}; counts={'0':0,'1':0}
    for seed in CFG['seeds']:
        x=load_torch(anchor(seed,0).parent/'selected_features.pt'); assert x.shape==(871,500)
        seen=[]
        for fold in range(10):
            key=f'{seed}/{fold}'; h=hist['anchors'][key]
            split=load_torch(anchor(seed,fold)/'splits.pt')
            for part in PARTS:assert array_sha(split[part])==h['pairing']['split_fingerprints'][part]
            assert np.array_equal(np.sort(np.concatenate([split[p] for p in PARTS])),np.arange(871))
            seen.extend(split['test'].tolist())
            for name,expected in h['input_sha256'].items():assert sha_file(ROOT/name)==expected
            raw=prepare_inputs(x.numpy(),meta,split,seed,fold,HCFG)
            assert digest(raw['info'])==h['inputs_digest']
            assert raw['candidates'].tolist()==h['candidate_indices']
            checkpoint=load_torch(history_path(seed,fold))
            assert checkpoint['state']=='complete' and checkpoint['group']=='L3'
            assert core_hash(checkpoint['best']['core'])==checkpoint['best_core_sha256']
            assert state_hash(checkpoint['best']['aligner'])==checkpoint['best_aligner_sha256']
            assert np.array_equal(checkpoint['oof']['indices'],split['test'])
            assert [r['draw_seed'] for r in checkpoint['frozen']]==[draw_seed(seed,fold,b) for b in CFG['draw_bases']]
            gpu=str(checkpoint['execution']['physical_gpu']); counts[gpu]+=1
            anchors[key]={'checkpoint_sha256':sha_file(history_path(seed,fold)),'core_sha256':checkpoint['best_core_sha256'],
                'aligner_sha256':checkpoint['best_aligner_sha256'],'historical_gpu':gpu,'inputs_digest':digest(raw['info']),
                'input_info':raw['info'],'input_sha256':h['input_sha256'],'split_sha256':h['pairing']['split_fingerprints']}
        assert np.array_equal(np.sort(seen),np.arange(871))
    assert counts=={'0':15,'1':15}
    a={'experiment':CFG['experiment'],'started_utc':utc(),'started_epoch':started,'code_sha256':code_hash(),
        'storage':storage,'protected':manifest,'anchors':anchors,'labels_sha256':array_sha(ctx['labels']),
        'subject_sha256':digest(ctx['ids']),'handoff_before_sha256':sha_file(ROOT/'handoff.md'),
        'handoff_before_bytes':(ROOT/'handoff.md').stat().st_size,'prompt_sha256':sha_file(ROOT/'prompt'),
        'native_audit':[
            'RFE/VAE dimension adaptation is not patient semantic alignment; native cdist is Euclidean despite paper correlation description.',
            'RP returns inside first SITE loop; shared constraint Conv uses .data; unregistered RP and detached graph/value weights preserved.',
            'Graph correction between fused_sim*rp_graph and dense_to_sparse; TransformerConv adds transformed edge_attr to keys and values.',
            'Two GTUNets, TopK and skip-upsampling, eval functional dropout, sampled VAE, original classifier BN preserved.',
            'E8 local JS directly trains image projection; graph CE/regularizer acts on separate relation MLP through bounded population graph.',
            'No optimizer creation, no backward, no fitting, no CPU neural inference; frozen RP projection is detected and restored per condition.'
        ],'references':[CFG['paper'],'https://github.com/NZWANG/MM-GTUNets','https://github.com/tangaii/MM'],
        'software':{'python':sys.version,'torch':torch.__version__,'numpy':np.__version__},'audit_seconds':time.time()-started}
    atomic_json(a,BASE/'audit.json');log('audit_passed',protected=len(manifest),anchors=30,gpu=counts)
    return a


def forbid_training():
    def denied(*args,**kwargs):raise RuntimeError('Experiment9: training/optimizer/backward/fitting forbidden')
    torch.Tensor.backward=denied;torch.autograd.backward=denied;torch.autograd.grad=denied
    torch.optim.Optimizer.__init__=denied
    from sklearn.feature_selection import RFE
    RFE.fit=denied;RFE.transform=denied;RFE.fit_transform=denied
    import utils.tools as tools
    tools.feature_selection=denied


def restore_rng(state,draw):
    random.setstate(state['python']);np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu']);torch.cuda.set_rng_state(state['cuda'],0)
    b=state['backend']
    torch.backends.cudnn.deterministic=b['cudnn_deterministic'];torch.backends.cudnn.benchmark=b['cudnn_benchmark']
    torch.backends.cuda.matmul.allow_tf32=b['matmul_allow_tf32'];torch.backends.cudnn.allow_tf32=b['cudnn_allow_tf32']
    random.seed(draw);np.random.seed(draw);torch.manual_seed(draw);torch.cuda.manual_seed_all(draw)


def rng_hash():
    return {'python':hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
        'numpy':hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest(),
        'cpu':array_sha(torch.get_rng_state().numpy()),'ppu':array_sha(torch.cuda.get_rng_state(0).cpu().numpy())}


def draw_seed(seed,fold,b):
    return int(hashlib.sha256(f'{seed}/{fold}/PCCI-repeat-{b}'.encode()).hexdigest()[:15],16)%(2**32)


def describe(a):
    a=np.asarray(a,dtype=np.float64).ravel(); assert np.isfinite(a).all()
    if not len(a):return {'n':0,'mean':0.,'p50':0.,'p10':0.,'p90':0.,'min':0.,'max':0.,'sd':0.}
    return {'n':len(a),'mean':float(a.mean()),'sd':float(a.std()),'p10':float(np.quantile(a,.1)),
        'p50':float(np.quantile(a,.5)),'p90':float(np.quantile(a,.9)),'min':float(a.min()),'max':float(a.max())}


class Loaded:
    def __init__(self,seed,fold,a):
        assert torch.cuda.is_available() and 'PPU' in torch.cuda.get_device_name(0).upper()
        torch.cuda.set_device(0);self.device=torch.device('cuda:0')
        self.seed,self.fold=seed,fold
        self.saved=load_torch(history_path(seed,fold));self.ctx=load_torch(OLD/'artifacts/context.pt')
        self.meta=metadata(self.ctx);self.split=load_torch(anchor(seed,fold)/'splits.pt')
        self.start=load_torch(anchor(seed,fold)/'train_start.pt')
        self.x_cpu=load_torch(anchor(seed,fold).parent/'selected_features.pt')
        argv=sys.argv
        try:sys.argv=['frozen-options'];opt=OptInit(model='MM_GTUNets',dataset='ABIDE',atlas='aal').args
        finally:sys.argv=argv
        opt.device=self.device;opt.ckpt_path=str(BASE/'never_created_vae');self.opt=opt
        self.core=FrozenCore(opt,fold).to(self.device)
        # Native Parameter.to(PPU) yields an unregistered non-leaf tensor.
        assert not any(p is self.core.rp_attention.weights for p in self.core.parameters())
        self.core.rp_attention.weights=self.core.rp_attention.weights.detach()
        self.core.vae_save_path=str(anchor(seed,fold).parent/'vae'/f'fold{fold}_pretrain.pth')
        affinity=create_reward_penalty_graph(self.ctx['ph_dict'],self.ctx['labels'],*[self.split[p] for p in PARTS],opt)
        self.x=self.x_cpu.to(self.device);self.ph=torch.as_tensor(self.ctx['ph_data'],dtype=torch.float32,device=self.device)
        self.affinity=torch.as_tensor(affinity,dtype=torch.float32,device=self.device)
        self.raw=prepare_inputs(self.x_cpu.numpy(),self.meta,self.split,seed,fold,HCFG)
        assert digest(self.raw['info'])==a['anchors'][f'{seed}/{fold}']['inputs_digest']
        self.inputs={'candidates':torch.as_tensor(self.raw['candidates'],device=self.device),
            'raw_cos':torch.as_tensor(self.raw['raw_cos'],device=self.device),
            'kernels':{k:torch.as_tensor(v,device=self.device) for k,v in self.raw['kernels'].items()},
            'null_kernels':{k:torch.as_tensor(v,device=self.device) for k,v in self.raw['null_kernels'].items()}}
        self.train=torch.as_tensor(self.split['train'],device=self.device)
        self.aligner=LCRA(HCFG).to(self.device)
        self.wrapper=FrozenWrapper(self.core,self.aligner,self.inputs,self.train,CFG,random_signs(seed,fold,871,self.device))
        self.wrapper.eval();self.wrapper.requires_grad_(False);self.core.rp_attention.weights.requires_grad_(False)
        self.reset(CFG['draw_bases'][0])
        with torch.no_grad():detail=self.aligner(self.x,self.inputs,self.train)
        tr=detail['centered'][self.train]
        s=float(torch.quantile(tr.abs().flatten(),CFG['scale_quantile']))
        noise=max(CFG['scale_floor'],8*torch.finfo(tr.dtype).eps*float(detail['u'][self.train].abs().max()))
        self.wrapper.scale=max(s,CFG['scale_floor']);self.wrapper.degenerate=s<=noise
        self.scale={'s':s,'scale':self.wrapper.scale,'precision_floor':noise,'degenerate':self.wrapper.degenerate,
            'train_centered':describe(tr.cpu().numpy()),'true_train_only':True,'same_for_all_modes':True}
        self.b=stage_b(self,detail)
        self.forward_calls=0

    def reset(self,base):
        core=self.saved['best']['core']
        self.core.load_state_dict(core['state'],strict=True)
        with torch.no_grad():self.core.rp_attention.weights.copy_(core['rp_weights'].to(self.device))
        self.aligner.load_state_dict(self.saved['best']['aligner'],strict=True)
        self.wrapper.eval();self.wrapper.requires_grad_(False);self.core.rp_attention.weights.requires_grad_(False)
        assert self.core.vae_optimizer is None
        assert state_hash(self.core.state_dict())==state_hash(core['state'])
        restore_rng(self.start['rng'],draw_seed(self.seed,self.fold,base))

    def forward(self,base,mode,wrapper=None):
        self.reset(base);before_rng=rng_hash();before={k:v.detach().cpu().clone() for k,v in self.core.state_dict().items()}
        conv=[];handles=[];counts={'img':0,'ph':0,'clf':0,'vae':0};edge_inputs=[]
        for name,module in (('img',self.core.img_unet),('ph',self.core.ph_unet),('clf',self.core.clf),('vae',self.core.vae)):
            def count(m,i,o,n=name):counts[n]+=1
            handles.append(module.register_forward_hook(count))
        for branch,net in (('img',self.core.img_unet),('ph',self.core.ph_unet)):
            for name,module in net.named_modules():
                if isinstance(module,TransformerConv):
                    def observe(m,i,o,n=f'{branch}/{name}'):
                        conv.append((n,o.detach().cpu().clone()))
                    handles.append(module.register_forward_hook(observe))
            def edge(m,i,b=branch):edge_inputs.append((b,i[1].detach().clone(),i[2].detach().clone()))
            handles.append(net.down_convs[0].register_forward_pre_hook(edge))
        try:
            with torch.no_grad():
                if wrapper is None:self.wrapper(self.x,self.ph,self.affinity,mode)
                else:wrapper(self.x,self.ph,self.affinity,self.inputs,self.train,'L3',alpha=0 if mode=='A0' else None,condition='shuffled' if mode=='replay_shuffled' else 'true')
            self.forward_calls+=1
        finally:
            for h in handles:h.remove()
        after=self.core.state_dict();changes=[k for k in before if not torch.equal(before[k],after[k].detach().cpu())]
        assert all(k.startswith('rp_attention.conv') for k in changes),changes
        assert torch.equal(self.saved['best']['core']['rp_weights'],self.core.rp_attention.weights.detach().cpu())
        assert all(v==1 for v in counts.values()),counts
        assert all(not p.requires_grad and p.grad is None for p in self.wrapper.parameters())
        assert all(not m.training for m in self.wrapper.modules())
        if wrapper is not None:
            last=wrapper.last
            d={'base':last['A_base'],'graph':last['A_new'],'E':last['E'],'img':last['img_embed'],
                'ph':last['ph_embed'],'joint':self.core.joint_embed,'logits':last['logits'],'graph_loss':self.core.cal_graph_loss(last['img_embed'],last['ph_embed'])}
        else:d=self.wrapper.last
        for branch,idx,attr in edge_inputs:
            assert torch.equal(idx,d.get('sparse_index',self.wrapper.last['sparse_index']))
            assert torch.equal(attr,d['graph'][idx[0],idx[1]].reshape(-1,1)),branch
        pools={}
        for name,net in (('img',self.core.img_unet),('ph',self.core.ph_unet)):
            nodes=np.arange(871);pools[name]=[]
            for perm in net.perms:
                nodes=nodes[perm.cpu().numpy()];pools[name].append(nodes.copy())
        result={'activations':{k:d[k].detach().cpu().numpy().copy() for k in ('img','ph','joint','logits')},
            'base':d['base'].cpu().numpy().copy(),'graph':d['graph'].cpu().numpy().copy(),
            'E':d['E'].cpu().numpy().copy(),'graph_loss':float(d['graph_loss']),
            'p':d['logits'].softmax(-1)[:,0].cpu().numpy(),'conv':conv,'pools':pools,
            'audit':{'rng_before':before_rng,'rng_after':rng_hash(),'native_calls':counts,
                'allowed_rp_projection_changed_keys':changes,'BN_unchanged':True,'core_weights_unchanged_except_rp':True,
                'core_before_sha256':state_hash(before),'first_transformer_edge_attr_exact':True}}
        # Reset explicitly after every native in-place projection, even on last condition.
        self.core.load_state_dict(self.saved['best']['core']['state'],strict=True)
        assert state_hash(self.core.state_dict())==state_hash(self.saved['best']['core']['state'])
        return result


@torch.no_grad()
def stage_b(loaded,detail):
    raw=loaded.raw;meta=loaded.meta;device=loaded.device
    age=meta['age'].copy();age[~np.isfinite(age)]=raw['info']['age_impute_train_median']
    sex=meta['sex'];candidates=raw['candidates'];width=raw['info']['age_bandwidth']
    # Neighbor voting is a statistical readout: accumulate/normalize in FP64.
    p=(detail['sim'].double()/HCFG['tau_img']).softmax(-1)
    qt=loaded.inputs['kernels']['true'].double().div(HCFG['tau_ph']).softmax(-1)
    qs=loaded.inputs['kernels']['shuffled'].double().div(HCFG['tau_ph']).softmax(-1)
    labels=torch.as_tensor(loaded.ctx['labels'][candidates]==0,dtype=torch.float64,device=device)
    assert np.isin(candidates,loaded.split['train']).all()
    def score(q):
        w=p*q;w=w/w.sum(-1,keepdim=True).clamp_min(1e-8)
        assert torch.allclose(w.sum(-1),torch.ones(len(p),device=device,dtype=torch.float64),atol=1e-12,rtol=0)
        return (w*labels).sum(-1).cpu().numpy()
    null_scores=[];null_js=[]
    for mapping in raw['nulls']:
        k=phenotype_kernel(age,sex,age[mapping[candidates]],sex[mapping[candidates]],width)
        q=torch.as_tensor(k,device=device).double().div(HCFG['tau_ph']).softmax(-1)
        null_scores.append(score(q));null_js.append(js_divergence(p,q).cpu().numpy())
    jt=js_divergence(p,qt).cpu().numpy()
    return {'B0':(p*labels).sum(-1).cpu().numpy(),'B1':score(qt),'B2':score(qs),
        'B3_null':np.stack(null_scores),'relation_gap':np.mean(null_js,axis=0)-jt,
        'weight_sums_finite':True,'uses_train_anchor_labels_only':True}


def graph_stats(a,b,split,site,mode):
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert np.array_equal(a,a.T) and np.array_equal(b,b.T)
    assert (a>=0).all() and (b>=0).all() and np.array_equal(a>0,b>0)
    assert np.array_equal(a.diagonal(),b.diagonal())
    bound=.05 if mode=='A6' else .25
    assert (b>=a*(1-bound)).all() and (b<=a*(1+bound)).all()
    change=np.abs(b-a);positive=a>0;changed=change>0
    rel=change[changed]/a[changed]
    same=site[:,None]==site[None,:]
    degree=(b.sum(1,dtype=np.float64)-a.sum(1,dtype=np.float64))/a.sum(1,dtype=np.float64)
    return {'relative_l1':float(change.sum(dtype=np.float64)/a.sum(dtype=np.float64)),
        'modified_directed_edges':int(changed.sum()),'base_positive_directed_edges':int(positive.sum()),
        'modified_edge_fraction':float(changed.sum()/positive.sum()),
        'modified_base_weight_fraction':float(a[changed].sum(dtype=np.float64)/a.sum(dtype=np.float64)),
        'modified_relative':describe(rel),'modified_absolute':describe(change[changed]),'degree_relative':describe(degree),
        'row_l1_relative':change.sum(1,dtype=np.float64)/a.sum(1,dtype=np.float64),
        'by_partition':{p:float(change[idx].sum(dtype=np.float64)/a[idx].sum(dtype=np.float64)) for p,idx in split.items()},
        'within_site_relative_l1':float(change[same].sum(dtype=np.float64)/a[same].sum(dtype=np.float64)),
        'between_site_relative_l1':float(change[~same].sum(dtype=np.float64)/a[~same].sum(dtype=np.float64)),
        'new_edges':0,'lost_edges':0,'support_unchanged':True,'symmetric':True,'diagonal_unchanged':True,'relative_bound':bound}


def activation_difference(a,b):
    a,b=np.asarray(a,np.float64),np.asarray(b,np.float64)
    diff=np.linalg.norm(b-a,axis=-1);norm=np.linalg.norm(a,axis=-1)
    cos=np.sum(a*b,axis=-1)/np.maximum(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1),1e-30)
    return {'mean_l2':float(diff.mean()),'relative_l2':float(np.linalg.norm(b-a)/max(np.linalg.norm(a),1e-30)),
        'mean_one_minus_cosine':float(np.mean(np.clip(1-cos,0,2))),'max_abs':float(np.abs(b-a).max())}


def paired_stats(off,out,split):
    r={'graph_loss_delta':out['graph_loss']-off['graph_loss'],'rng_after_equal':out['audit']['rng_after']==off['audit']['rng_after'],
        'activations':{k:activation_difference(off['activations'][k],out['activations'][k]) for k in off['activations']},
        'test_activations':{k:activation_difference(off['activations'][k][split['test']],out['activations'][k][split['test']]) for k in off['activations']},
        'pooling':{},'conv':[]}
    for name in off['pools']:
        r['pooling'][name]=[]
        for a,b in zip(off['pools'][name],out['pools'][name]):
            r['pooling'][name].append({'jaccard':float(len(set(a)&set(b))/len(set(a)|set(b))),
                'ordered_equal':bool(np.array_equal(a,b))})
    assert len(off['conv'])==len(out['conv'])
    for (name,a),(other,b) in zip(off['conv'],out['conv']):
        assert name==other
        branch=name.split('/')[0]
        aligned=all(v['ordered_equal'] for v in r['pooling'][branch])
        full=a.shape[0]==871
        r['conv'].append({'layer':name,'nodes':a.shape[0],'row_comparison_valid':full or aligned,
            'difference':activation_difference(a.numpy(),b.numpy()) if full or aligned else None})
    return r


def technical_parity(loaded):
    spec=importlib.util.spec_from_file_location('read_only_e8_lcra',HIST/'lcra.py')
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    original=old.LCRAWrapper(loaded.core,loaded.aligner,HCFG);original.eval()
    previous=torch.are_deterministic_algorithms_enabled();torch.use_deterministic_algorithms(True)
    rows=[]
    try:
        for mode in ('A0','A1','replay_shuffled'):
            candidate=loaded.forward(CFG['draw_bases'][0],mode)
            reference=loaded.forward(CFG['draw_bases'][0],mode,original)
            for k in ('base','graph','p'):
                assert np.array_equal(candidate[k],reference[k]),f'exact frozen E8 parity failed {mode}/{k}'
            for k in candidate['activations']:
                assert np.array_equal(candidate['activations'][k],reference['activations'][k]),f'exact activation parity {mode}/{k}'
            assert candidate['graph_loss']==reference['graph_loss']
            assert candidate['audit']['rng_after']==reference['audit']['rng_after']
            rows.append({'mode':mode,'maximum_absolute_difference':0.0})
    finally:torch.use_deterministic_algorithms(previous)
    return {'passed':True,'deterministic_exact_math_state_rng_parity':rows,'forwards':6}


def valid_result(seed,fold,a,complete=True):
    p=result_path(seed,fold)
    if not p.exists():return False
    r=load_torch(p)
    assert r['payload_sha256']==payload_hash(r),'Saved diagnostic payload SHA256 mismatch'
    assert r['code_sha256']==a['code_sha256'] and r['checkpoint_sha256']==a['anchors'][f'{seed}/{fold}']['checkpoint_sha256']
    return r['state']=='complete' if complete else r['state'] in ('preflight_passed','complete')


def worker(seed,fold,preflight):
    a=read_json(BASE/'audit.json');assert a['code_sha256']==code_hash()
    forbid_training()
    slot=1+CFG['seeds'].index(seed)*10+fold
    with lock(slot):
        if valid_result(seed,fold,a,complete=not preflight):
            log('skip_valid',seed=seed,fold=fold,preflight=preflight);return
        start=time.time();loaded=Loaded(seed,fold,a)
        p=result_path(seed,fold)
        if p.exists():r=load_torch(p)
        else:r={'state':'running','seed':seed,'fold':fold,'code_sha256':code_hash(),
            'checkpoint_sha256':a['anchors'][f'{seed}/{fold}']['checkpoint_sha256'],'split':loaded.split,
            'ids':loaded.ctx['ids'],'labels':loaded.ctx['labels'],'meta':loaded.meta,'scale':loaded.scale,
            'native_official_test_probability':loaded.saved['oof']['prob_asd'],
            'stage_B':loaded.b,'draws':[],'execution':{'gpu':os.environ['CUDA_VISIBLE_DEVICES'],'seconds':0.,'forwards':0,'training':0}}
        if preflight and 'technical_parity' not in r:r['technical_parity']=technical_parity(loaded)
        target=CFG['preflight_draws'] if preflight else 5
        for j in range(len(r['draws']),target):
            base=CFG['draw_bases'][j];outs={};row={'draw_base':base,'derived_seed':draw_seed(seed,fold,base),'modes':{},'historical_replay':{}}
            for mode in MODES+('replay_shuffled',):
                out=loaded.forward(base,mode);outs[mode]=out
                if mode=='A0': off=out
                assert np.array_equal(off['base'],out['base']),f'base graph changed across conditions {mode}'
                assert off['audit']['rng_before']==out['audit']['rng_before']
                if mode=='A5':assert np.array_equal(np.abs(out['E']),np.abs(outs['A3']['E']))
                if mode=='A6':
                    expected=out['base'].copy();mask=~np.eye(871,dtype=bool)&(expected>0);expected[mask]*=np.float32(1.05)
                    assert np.array_equal(expected,out['graph'])
                row['modes'][mode]={'prob_asd':out['p'],'graph':graph_stats(out['base'],out['graph'],loaded.split,loaded.meta['site'],mode),
                    'propagation':paired_stats(off,out,loaded.split),'audit':out['audit']}
                if mode in ('A0','A1','replay_shuffled'):
                    key={'A0':'off','A1':'true','replay_shuffled':'shuffled'}[mode]
                    expected=loaded.saved['frozen'][j]['prob_asd'][key]
                    difference=out['p'][loaded.split['test']]-expected
                    row['historical_replay'][key]={'maximum_absolute_difference':float(np.abs(difference).max()),
                        'mean_absolute_difference':float(np.abs(difference).mean()),'exact':bool(np.array_equal(expected,out['p'][loaded.split['test']])),
                        'label_disagreements':int(np.sum((expected>=.5)!=(out['p'][loaded.split['test']]>=.5)))}
            r['draws'].append(row)
            r['execution'].update(seconds=r['execution']['seconds']+time.time()-start,forwards=r['execution']['forwards']+loaded.forward_calls,
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,host_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            start=time.time();loaded.forward_calls=0
            atomic_torch(r,p)
            log('draw_complete',seed=seed,fold=fold,draw=j,gpu=os.environ['CUDA_VISIBLE_DEVICES'],preflight=preflight,
                replay_max=max(v['maximum_absolute_difference'] for v in row['historical_replay'].values()))
        r['state']='preflight_passed' if preflight else 'complete';r['execution']['completed_utc']=utc()
        atomic_torch(r,p)
        log('anchor_complete',seed=seed,fold=fold,state=r['state'],gpu=os.environ['CUDA_VISIBLE_DEVICES'])


def resources():
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,memory.total,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=20)
    devices={}
    for line in gpu.strip().splitlines():
        v=[float(x.strip()) for x in line.split(',')];devices[str(int(v[0]))]={'used':v[1],'total':v[2],'util':v[3]}
    ram=Path('/sys/fs/cgroup/memory');limit=int((ram/'memory.limit_in_bytes').read_text());usage=int((ram/'memory.usage_in_bytes').read_text())
    return {'utc':utc(),'gpu':devices,'ram_limit_mib':limit/2**20,'ram_usage_mib':usage/2**20,'load1':os.getloadavg()[0]}


def queue(tasks,a,state,preflight):
    pending=[(s,f) for s,f in tasks if not valid_result(s,f,a,complete=not preflight)];active=[];failed=[]
    with (BASE/'run.log').open('a') as logstream:
        while pending or active:
            for item in list(active):
                process,s,f,gpu=item
                ret=process.poll()
                if ret is not None:
                    active.remove(item)
                    if ret:
                        error=f'Worker failed {s}/{f} PPU{gpu}: exit {ret}; see run.log'
                        failed.append(error);state['failures'].append({'utc':utc(),'error':error})
                    else:assert valid_result(s,f,a,complete=not preflight)
            if failed and not active:
                atomic_json(state,BASE/'summary.json')
                raise RuntimeError('; '.join(failed))
            if pending and not failed:
                sample=resources();state['resources'].append(sample)
                for gpu in ('0','1'):
                    cap=1 if preflight else CFG['max_workers_per_card']
                    if sum(x[3]==gpu for x in active)>=cap:continue
                    d=sample['gpu'][gpu]
                    if d['total']-d['used']<CFG['worker_budget_mib']+CFG['ppu_reserve_mib']:continue
                    if sample['ram_limit_mib']-sample['ram_usage_mib']<CFG['host_reserve_mib']+4096:continue
                    task=next(((s,f) for s,f in pending if a['anchors'][f'{s}/{f}']['historical_gpu']==gpu),None)
                    if task is None:continue
                    s,f=task;env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,PYTHONDONTWRITEBYTECODE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
                    cmd=[sys.executable,'-B','-u',str(BASE/'run.py'),'worker','--seed',str(s),'--fold',str(f)]
                    if preflight:cmd.append('--preflight')
                    process=subprocess.Popen(cmd,env=env,cwd=ROOT,stdout=logstream,stderr=subprocess.STDOUT,start_new_session=True)
                    active.append((process,s,f,gpu));pending.remove(task)
                    log('worker_start',seed=s,fold=f,gpu=gpu,pid=process.pid,preflight=preflight)
            state['queue']={'pending':len(pending),'active':len(active),'preflight':preflight,'updated_utc':utc()}
            atomic_json(state,BASE/'summary.json')
            if active:time.sleep(2)
            elif pending:time.sleep(2)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['audit','all','worker','analyze'])
    parser.add_argument('--seed',type=int,default=911);parser.add_argument('--fold',type=int,default=0)
    parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    if args.action=='worker':worker(args.seed,args.fold,args.preflight);return
    with lock(0):
        a=audit()
        if args.action=='audit':return
        state=read_json(BASE/'summary.json') if (BASE/'summary.json').exists() else {'status':'audited','started_utc':a['started_utc'],
            'experiment':CFG['experiment'],'code_sha256':code_hash(),'resources':[],'training_count':0,'failures':[]}
        if not list((BASE/'runs').glob('*.pt')):state['code_sha256']=code_hash()
        if state.get('status')=='complete':
            import analyze
            analyze.verify_terminal(state,a);log('resume_verified',retraining=False,new_forward=False);return
        if args.action=='all':
            state['status']='preflight';queue(CFG['preflight_anchors'],a,state,True)
            # Preflight is technical: no scores are read to choose any gamma.
            state['preflight']={'passed':True,'anchors':5,'draws_each':2,'criterion':'exact independent implementation + graph/state/RNG/partition assertions; no accuracy selection'}
            atomic_json(state,BASE/'summary.json')
            log('preflight_passed',anchors=5)
            state['status']='frozen_analysis'
            queue([(s,f) for f in range(10) for s in CFG['seeds']],a,state,False)
        import analyze
        analyze.finish(state,a)


if __name__=='__main__':
    try:main()
    except Exception:
        log('error',traceback=traceback.format_exc());raise
