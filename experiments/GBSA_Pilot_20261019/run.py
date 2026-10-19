"""Independent historical-behavior GBSA pilot with locked paired PPU jobs."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
import warnings
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
BASE = Path(__file__).resolve().parent
os.environ['HF_HOME'] = str(BASE/'runs/hf_runtime')
CFG = json.loads((BASE/'config.json').read_text())
ROOT = Path(CFG['root'])
OLD = ROOT/CFG['history']
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(BASE))
import numpy as np
import torch
from torch.nn import functional as F
from scipy.special import softmax
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, recall_score, f1_score, log_loss, brier_score_loss
from descriptors import sha_file, array_sha, atomic_bytes, save_json, save, load, history_load, pack, unpack, reference, raw_statistics, captions, download_text_model, encode_texts, text_diagnostics
from semantic_model import ImageHook, SemanticHead, gradient_norm, effective_rank
from model.mm_gtunets import MM_GTUNets
from model.rp_graph import create_reward_penalty_graph
from opt import OptInit
from utils.tools import EarlyStopping
torch.set_num_threads(1)
warnings.filterwarnings('ignore', message='To copy construct from a tensor')
warnings.filterwarnings('ignore', message='Converting a tensor with requires_grad=True')
warnings.filterwarnings('ignore', message='Please use the new API settings')
CODE = ('config.json','descriptors.py','semantic_model.py','run.py','analyze.py')
SOURCE = ('opt.py','train_mm_gtunets.py','model/mm_gtunets.py','model/gtunet.py','model/rp_graph.py','utils/tools.py','utils/mydataloader.py','utils/metrics.py')


def utc():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    line = f'{utc()} {message}'
    print(line,flush=True)
    with (BASE/'run.log').open('a') as f:
        f.write(line+'\n')


def code_hash():
    return hashlib.sha256(json.dumps({n:sha_file(BASE/n) for n in CODE},sort_keys=True).encode()).hexdigest()


def tree_hash(value):
    if torch.is_tensor(value) or isinstance(value,np.ndarray):
        return array_sha(value)
    if isinstance(value,dict):
        value = {str(k):tree_hash(v) for k,v in sorted(value.items(),key=lambda x:str(x[0]))}
    elif isinstance(value,(tuple,list)):
        value = [tree_hash(v) for v in value]
    return hashlib.sha256(repr(value).encode()).hexdigest()


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value,dict):
        return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return type(value)(cpu_copy(v) for v in value)
    return value


def export_core(core):
    return {'state':cpu_copy(core.state_dict()), 'rp_weights':core.rp_attention.weights.detach().cpu().clone()}


def import_core(core,state):
    core.load_state_dict(state['state'],strict=True)
    with torch.no_grad():
        core.rp_attention.weights.copy_(state['rp_weights'].to(core.device))


def capture_rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.get_rng_state().clone(),
        'cuda':torch.cuda.get_rng_state(0).clone(),'backend':{
            'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,
            'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32}}


def restore_rng(rng):
    random.setstate(rng['python']); np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['cpu']); torch.cuda.set_rng_state(rng['cuda'],0)
    b = rng['backend']
    torch.backends.cudnn.deterministic=b['cudnn_deterministic']; torch.backends.cudnn.benchmark=b['cudnn_benchmark']
    torch.backends.cuda.matmul.allow_tf32=b['matmul_allow_tf32']; torch.backends.cudnn.allow_tf32=b['cudnn_allow_tf32']


@contextmanager
def locked(path):
    with Path(path).open('a+b') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(f,fcntl.LOCK_UN)


def accelerator():
    if not torch.cuda.is_available() or 'PPU' not in torch.cuda.get_device_name(0):
        raise RuntimeError('PPU training required; CPU fallback forbidden')
    torch.cuda.set_device(0)
    return torch.device('cuda:0')


def options(fold):
    previous=sys.argv
    try:
        sys.argv=['historical-options']
        opt=OptInit(model='MM_GTUNets',dataset='ABIDE',atlas='aal').args
    finally:
        sys.argv=previous
    opt.device=accelerator(); opt.seed=CFG['seed']
    opt.data_folder=str(ROOT/'dataset/ABIDE_aal')
    opt.ckpt_path=str(OLD/'artifacts/seed_911/vae')
    opt.epoch=CFG['epochs']; opt.early_stop=CFG['early_stop']
    opt.lr=CFG['lr']; opt.wd=CFG['weight_decay']
    return opt


def metrics(y,prob):
    y=np.asarray(y); prob=np.asarray(prob,dtype=np.float64)
    assert np.isfinite(prob).all() and ((prob>=0)&(prob<=1)).all()
    asd=(y==0).astype(int); pred=(prob>=.5).astype(int)
    return {'ACC':float(accuracy_score(asd,pred)),'BA':float(balanced_accuracy_score(asd,pred)),
        'AUC':float(roc_auc_score(asd,prob)),'SEN':float(recall_score(asd,pred)),
        'SPE':float(recall_score(1-asd,1-pred)), 'F1':float(f1_score(asd,pred,zero_division=0)),
        'CE':float(log_loss(asd,prob,labels=[0,1])),'Brier':float(brier_score_loss(asd,prob))}


def input_paths():
    paths=[ROOT/name for name in SOURCE]
    paths.extend([ROOT/'prompt',ROOT/'dataset/ABIDE_aal/id.txt',
        ROOT/'dataset/ABIDE_aal/save_tensor/up_triangles.pt',ROOT/'dataset/ABIDE_aal/save_tensor/labels.pt',
        OLD/'artifacts/context.pt',OLD/'artifacts/seed_911/selected_features.pt',OLD/'artifacts/seed_911/rfe_support.npy',
        OLD/'config.json',OLD/'run.py',OLD/'README.md',
        ROOT/'experiments/LCRA_20261018/README.md',ROOT/'experiments/LCRA_20261018/report.md',
        ROOT/'experiments/LCRA_Attribution_20261019/README.md',ROOT/'experiments/LCRA_Attribution_20261019/report.md',
        Path('/tmp/CMCV_20261019/README.md'),Path('/tmp/CMCV_20261019/report.md')])
    for fold in range(10):
        adir=OLD/f'artifacts/seed_911/fold_{fold}'
        paths.extend(adir/name for name in ('train_start.pt','splits.pt','fingerprints.json'))
        paths.append(OLD/f'artifacts/seed_911/vae/fold{fold}_pretrain.pth')
    return paths


def protected_guard(audit):
    for path,sha in audit['protected_files'].items():
        assert sha_file(path)==sha,'Protected file changed: '+path
    assert code_hash()==audit['code_sha256'],'Frozen code changed'


def audit_inputs():
    if (BASE/'audit.json').exists():
        audit=json.loads((BASE/'audit.json').read_text())
        protected_guard(audit)
        return audit
    assert torch.cuda.is_available() and torch.cuda.device_count()==2
    paths=input_paths()
    ctx=history_load(OLD/'artifacts/context.pt')
    assert len(ctx['ids'])==len(set(ctx['ids']))==871
    y=ctx['labels']
    assert np.bincount(y).tolist()==[403,468]
    data=ROOT/'dataset/ABIDE_aal'
    assert np.loadtxt(data/'id.txt',dtype=str).tolist()==ctx['ids']
    assert np.array_equal(history_load(data/'save_tensor/labels.pt'),y)
    paths.extend(data/f'{sid}_aal_correlation.mat' for sid in ctx['ids'])
    all_test=[]
    for fold in range(10):
        split=history_load(OLD/f'artifacts/seed_911/fold_{fold}/splits.pt')
        tr,va,te=[np.asarray(split[k]) for k in ('train','val','test')]
        assert len(set(np.concatenate([tr,va,te])))==871
        assert sorted(np.concatenate([tr,va,te]).tolist())==list(range(871))
        all_test.extend(te.tolist())
        start=history_load(OLD/f'artifacts/seed_911/fold_{fold}/train_start.pt')
        assert start['vae_sha256']==sha_file(OLD/f'artifacts/seed_911/vae/fold{fold}_pretrain.pth')
    assert sorted(all_test)==list(range(871))
    audit={'started_utc':utc(),'started_epoch':time.time(),'code_sha256':code_hash(),
        'protected_files':{str(p):sha_file(p) for p in paths},
        'handoff_before_sha256':sha_file(ROOT/'handoff.md'),
        'subject_sha256':tree_hash(ctx['ids']),'label_sha256':array_sha(y),
        'storage_preflight':'write/fsync/readback/rename/directory fsync/exclusive lock passed',
        'sources':{'fmri_lm':'https://arxiv.org/html/2511.21760v3',
            'cvpr':'https://openaccess.thecvf.com/content/CVPR2026/html/Wei_fMRI-LM_Towards_a_Universal_Foundation_Model_for_Language-Aligned_fMRI_Understanding_CVPR_2026_paper.html',
            'mm_gtunets':'https://arxiv.org/html/2406.14455v3'},
        'architecture_audit':{'img_output':[871,16],'graph_nodes':'subjects, not anatomical ROIs',
            'fusion':'native arithmetic mean and attention; classifier16->256/ReLU/BN->2',
            'semantic_gradient':'observational native img_unet output hook; no second propagation',
            'preserved':['RFE overwrite history','transductive graph','RP early return','RP .data projection',
                'unregistered RP weights','native tensor reconstruction gradients','stochastic VAE and eval Dropout'],
            'fmri_lm_distinction':'FC+graph caption inspiration only; no tokenizer, quantizer, domain discriminator, LLM tuning or diagnostic descriptors reproduced'}}
    import urllib.request
    for name in ('nbs_data/get_fmri_discriptor.py','train_quantizer_contr.py'):
        with urllib.request.urlopen('https://raw.githubusercontent.com/yuxiangwei0808/fMRI-LM/main/'+name,timeout=60) as response:
            content=response.read()
        audit['sources'][name]={'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)}
    save_json(BASE/'audit.json',audit)
    log(f'Input audit passed: 10 anchors, 871 subjects, {len(paths)} protected files')
    return audit


def prepare(audit):
    path=BASE/'runs/descriptors.bin'
    if path.exists():
        cache=load(path)
        assert cache['code_sha256']==code_hash()
        assert cache['complete']
        return cache
    from scipy.io import loadmat
    from transformers import AutoModel,AutoTokenizer
    ctx=history_load(OLD/'artifacts/context.pt')
    data=ROOT/'dataset/ABIDE_aal'
    raw=np.asarray(history_load(data/'save_tensor/up_triangles.pt'))
    selected=history_load(OLD/'artifacts/seed_911/selected_features.pt')
    selected=selected.detach().cpu().numpy() if torch.is_tensor(selected) else np.asarray(selected)
    support=np.load(OLD/'artifacts/seed_911/rfe_support.npy')
    assert raw.shape==(871,6670) and selected.shape==(871,500)
    assert np.array_equal(raw[:,support].astype(selected.dtype),selected)
    stats=[]; pairs=[]
    for i,sid in enumerate(ctx['ids']):
        fc=loadmat(data/f'{sid}_aal_correlation.mat')['connectivity']
        upper=fc[np.triu_indices(116,1)]
        assert np.array_equal(np.arctanh(upper),raw[i]),'Raw FC/Fisher-Z cache differs'
        s,p=raw_statistics(fc)
        stats.append(s); pairs.append(p)
        if i%100==0: log(f'FC statistics {i+1}/871')
    stats=np.stack(stats)
    refs={}; texts={}; unique={}
    for fold in range(10):
        split=history_load(OLD/f'artifacts/seed_911/fold_{fold}/splits.pt')
        ref=reference(stats,split['train'])
        altered=stats.copy(); altered[split['test']]+=123
        assert tree_hash(reference(altered,split['train']))==tree_hash(ref)
        refs[fold]=ref; texts[fold]=captions(stats,pairs,ref)
        for text in texts[fold]: unique.setdefault(text,len(unique))
    log(f'Caption generation passed, {len(unique)} distinct descriptions; downloading pinned MiniLM')
    folder,manifest=download_text_model()
    tok=AutoTokenizer.from_pretrained(folder,local_files_only=True)
    model=AutoModel.from_pretrained(folder,local_files_only=True,attn_implementation='eager').eval()
    for p in model.parameters(): p.requires_grad_(False)
    before=tree_hash(model.state_dict())
    textlist=list(unique)
    cpu,_=encode_texts(model,tok,textlist[:2],'cpu')
    device=accelerator(); model.to(device)
    ppu,_=encode_texts(model,tok,textlist[:2],device)
    torch.testing.assert_close(cpu,ppu,rtol=1e-5,atol=2e-6)
    embeddings,max_tokens=encode_texts(model,tok,textlist,device)
    assert torch.isfinite(embeddings).all() and not embeddings.requires_grad
    assert before==tree_hash(model.state_dict()) and not any(p.requires_grad for p in model.parameters())
    folds={}
    for fold in range(10):
        split=history_load(OLD/f'artifacts/seed_911/fold_{fold}/splits.pt')
        emb=embeddings[[unique[t] for t in texts[fold]]].clone()
        diag=text_diagnostics(emb,texts[fold],split['train'])
        folds[fold]={'reference':refs[fold],'captions':texts[fold],'embedding':emb,'diagnostics':diag,
            'subject_ids':ctx['ids'],'embedding_sha256':array_sha(emb),'split_sha256':tree_hash(split)}
    cache={'complete':True,'code_sha256':code_hash(),'ids':ctx['ids'],'raw_stats':stats,'pairs':pairs,'folds':folds,
        'text_model':{'name':CFG['text_model'],'revision':CFG['text_revision'],'files_sha256':manifest,
            'state_sha256_before_after':before,'frozen':True,'max_tokens':max_tokens,
            'CPU_PPU_max_error':float((cpu-ppu).abs().max()),'unique_captions':len(unique)},
        'raw_fc_sha256':array_sha(raw),'rfe_selected_sha256':array_sha(selected)}
    save(path,cache)
    audit['descriptors']={k:v for k,v in cache.items() if k in ('text_model','raw_fc_sha256','rfe_selected_sha256')}
    audit['descriptors']['folds']={str(f):d['diagnostics'] for f,d in folds.items()}
    audit['descriptors']['cache_sha256']=sha_file(path)
    save_json(BASE/'audit.json',audit)
    del model
    torch.cuda.empty_cache()
    log('Frozen text cache passed all 10 folds; no collapse, no label or held-out reference dependence')
    return cache


class Loaded:
    def __init__(self,fold,head=False):
        self.fold=fold; self.device=accelerator(); self.opt=options(fold)
        self.ctx=history_load(OLD/'artifacts/context.pt')
        adir=OLD/f'artifacts/seed_911/fold_{fold}'
        self.start=history_load(adir/'train_start.pt')
        self.split=history_load(adir/'splits.pt')
        self.x=history_load(adir.parent/'selected_features.pt')
        if not torch.is_tensor(self.x): self.x=torch.from_numpy(self.x)
        self.x=self.x.float().to(self.device)
        self.ph=torch.from_numpy(self.ctx['ph_data']).float().to(self.device)
        self.y=torch.as_tensor(self.ctx['labels'],dtype=torch.long,device=self.device)
        self.affinity=torch.from_numpy(create_reward_penalty_graph(self.ctx['ph_dict'],self.ctx['labels'],
            self.split['train'],self.split['val'],self.split['test'],self.opt)).float().to(self.device)
        self.core=MM_GTUNets(self.opt,fold).to(self.device)
        import_core(self.core,self.start['core'])
        assert tree_hash(export_core(self.core))==tree_hash(self.start['core'])
        self.head=SemanticHead.isolated(CFG['seed'],fold,self.device) if head else None
        self.head_initial_sha=tree_hash(self.head.state_dict()) if self.head else None
        self.optimizer=torch.optim.Adam(list(self.core.parameters())+(list(self.head.parameters()) if self.head else []),
            lr=CFG['lr'],weight_decay=CFG['weight_decay'])
        assert not any(p is self.core.rp_attention.weights for g in self.optimizer.param_groups for p in g['params'])
        restore_rng(self.start['rng'])
        self.pairing={'core':tree_hash(self.start['core']),'rng':tree_hash(capture_rng()),'split':tree_hash(self.split),
            'x':array_sha(self.x),'ph':array_sha(self.ph),'affinity':array_sha(self.affinity),
            'vae':self.start['vae_sha256'],'original_optimizer_parameter_order':[n for n,p in self.core.named_parameters()]}

    def forward(self):
        return self.core(self.x,self.ph,self.affinity)


def preflight():
    accelerator()
    torch.use_deterministic_algorithms(True)
    traces=[]
    for with_hook in (False,True):
        obj=Loaded(0)
        hook=ImageHook(obj.core) if with_hook else None
        trace=[]
        for step in range(3):
            obj.core.train(); obj.optimizer.zero_grad()
            out,graph=obj.forward(); ce=F.cross_entropy(out[obj.split['train']],obj.y[obj.split['train']])
            loss=ce+graph; loss.backward()
            snapshot={'logits':array_sha(out),'graph':array_sha(graph),
                'grads':tree_hash({n:p.grad for n,p in obj.core.named_parameters()}),
                'topk':tree_hash({'img':obj.core.img_unet.perms,'ph':obj.core.ph_unet.perms})}
            obj.optimizer.step(); obj.core.eval()
            with torch.no_grad(): val,vg=obj.forward()
            snapshot.update({'validation_logits':array_sha(val),'validation_graph':array_sha(vg),
                'state':tree_hash(export_core(obj.core)),'adam':tree_hash(obj.optimizer.state_dict()),'rng':tree_hash(capture_rng())})
            trace.append(snapshot)
        if hook:
            assert hook.calls==6
            hook.close()
        traces.append(trace)
        del obj
        torch.cuda.empty_cache()
    assert traces[0]==traces[1],'Inert hook changed numerical trajectory'
    obj=Loaded(0,head=True); hook=ImageHook(obj.core)
    cache=load(BASE/'runs/descriptors.bin')
    text=cache['folds'][0]['embedding'].to(obj.device)
    obj.core.train(); obj.optimizer.zero_grad()
    initial_bn=int(obj.core.clf[2].num_batches_tracked)
    out,graph=obj.forward()
    native=F.cross_entropy(out[obj.split['train']],obj.y[obj.split['train']])+graph
    sem,detail=obj.head.objective(hook.image,text,obj.split['train'],CFG['temperature'])
    params=list(obj.core.img_unet.parameters())
    norms={'native_to_img':gradient_norm(native,params),'semantic_to_img':gradient_norm(sem,params),
        'semantic_to_head':gradient_norm(sem,list(obj.head.parameters()))}
    assert all(np.isfinite(v) and v>0 for v in norms.values())
    assert torch.autograd.grad(sem,obj.core.ph_unet.down_in_hid_conv.lin_query.weight,allow_unused=True,retain_graph=True)[0] is None
    # Shared subject indices and features are the only contrastive candidates.
    assert text.shape==(871,384) and len(obj.split['train'])==704
    (native+.05*sem).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in obj.head.parameters())
    before=tree_hash(obj.head.state_dict()); obj.optimizer.step()
    assert before!=tree_hash(obj.head.state_dict())
    assert int(obj.core.clf[2].num_batches_tracked)==initial_bn+1
    saved={'core':export_core(obj.core),'head':cpu_copy(obj.head.state_dict()),
        'optimizer':cpu_copy(obj.optimizer.state_dict()),'rng':capture_rng()}
    roundtrip=unpack(pack(saved))
    assert tree_hash(saved)==tree_hash(roundtrip)
    import_core(obj.core,roundtrip['core']); obj.head.load_state_dict(roundtrip['head'])
    obj.optimizer.load_state_dict(roundtrip['optimizer']); restore_rng(roundtrip['rng'])
    obj.core.eval()
    with torch.no_grad(): a,_=obj.forward()
    import_core(obj.core,roundtrip['core']); restore_rng(roundtrip['rng'])
    with torch.no_grad(): b,_=obj.forward()
    assert torch.equal(a,b),'Checkpoint replay not exact under deterministic diagnosis'
    assert int(obj.core.clf[2].num_batches_tracked)==initial_bn+1
    hook.close()
    result={'passed':True,'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],'inert_hook_three_steps_exact':True,
        'logits_graph_gradients_adam_bn_rng_topk_exact':True,'semantic_gradient_norms':norms,
        'checkpoint_core_head_adam_rp_rng_exact':True,'validation_BN_unchanged':True,
        'outer_test_metrics_accessed':False,'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
    save_json(BASE/f'runs/preflight_gpu{os.environ["CUDA_VISIBLE_DEVICES"]}.json',result)
    log(f'PPU preflight passed: {result}')


def result_path(group,fold):
    return BASE/f'runs/{group}_f{fold}.bin'


def completed(group,fold):
    path=result_path(group,fold)
    if not path.exists(): return False
    r=load(path)
    if r.get('status')!='complete': return False
    assert r['code_sha256']==code_hash() and r['group']==group and r['fold']==fold
    assert r['cache_sha256']==sha_file(BASE/'runs/descriptors.bin')
    for name,old_sha in r['input_fingerprints'].items(): assert sha_file(name)==old_sha
    assert all(abs(metrics(r['labels'],r['prob_asd'])[k]-r['metrics'][k])<1e-12 for k in r['metrics'])
    return True


def train(group,fold):
    with locked(BASE/f'runs/.{group}_f{fold}.lock'):
        if completed(group,fold):
            log(f'SKIP complete {group}/f{fold}'); return
        obj=Loaded(fold,head=group=='S1')
        hook=ImageHook(obj.core)
        cache=load(BASE/'runs/descriptors.bin')
        assert cache['code_sha256']==code_hash() and cache['ids']==obj.ctx['ids']
        text=cache['folds'][fold]['embedding'].to(obj.device) if obj.head else None
        stopper=EarlyStopping(patience=CFG['early_stop'],verbose=False)
        best=None; history=[]; best_loss=1e10
        initial_bn=int(obj.core.clf[2].num_batches_tracked)
        start=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        for epoch in range(CFG['epochs']):
            obj.core.train()
            if obj.head: obj.head.train()
            obj.optimizer.zero_grad()
            out,graph=obj.forward()
            ce=F.cross_entropy(out[obj.split['train']],obj.y[obj.split['train']]); native=ce+graph
            semantic=None; detail={}
            coefficient=CFG['lambda_semantic']*min(epoch/(CFG['warmup_epochs']-1),1.) if obj.head else 0.
            total=native
            if obj.head:
                semantic,detail=obj.head.objective(hook.image,text,obj.split['train'],CFG['temperature'])
                total=native+coefficient*semantic
            assert torch.isfinite(total),'Nonfinite training objective'
            diagnostic=epoch in CFG['diagnostic_epochs'] or epoch%CFG['diagnostic_frequency']==0
            record={'epoch':epoch,'train_ce':float(ce.detach()),'train_graph':float(graph.detach()),
                'train_original':float(native.detach()),'semantic_coefficient':coefficient,
                'semantic_normalized':float(semantic.detach()) if semantic is not None else 0.,'total':float(total.detach())}
            if diagnostic:
                params=list(obj.core.img_unet.parameters())
                record['mechanism']={'native_to_img_grad':gradient_norm(native,params),
                    'semantic_to_img_grad':gradient_norm(semantic,params) if semantic is not None else 0.,
                    'semantic_to_head_grad':gradient_norm(semantic,list(obj.head.parameters())) if semantic is not None else 0.,
                    'image_variance':float(hook.image[obj.split['train']].detach().var(0,unbiased=False).mean()),
                    'image_effective_rank':effective_rank(hook.image[obj.split['train']]),
                    **{k:float(v.detach()) for k,v in detail.items()}}
                if obj.head:
                    assert record['mechanism']['semantic_to_img_grad']>0
                    assert record['mechanism']['semantic_to_head_grad']>0
            total.backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in obj.core.parameters())
            if obj.head: assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in obj.head.parameters())
            obj.optimizer.step()
            obj.core.eval()
            if obj.head: obj.head.eval()
            with torch.no_grad(): val,vg=obj.forward()
            vce=F.cross_entropy(val[obj.split['val']],obj.y[obj.split['val']]); vl=vce+vg
            assert torch.isfinite(vl) and int(obj.core.clf[2].num_batches_tracked)==initial_bn+epoch+1
            assert hook.calls==2*(epoch+1),'Repeated image propagation'
            record.update({'val_ce':float(vce),'val_graph':float(vg),'val_original':float(vl)})
            history.append(record)
            if float(vl)<best_loss:
                best_loss=float(vl)
                best={'epoch':epoch,'validation_loss':best_loss,'core':export_core(obj.core),
                    'head':cpu_copy(obj.head.state_dict()) if obj.head else None,
                    'rng':capture_rng(),'optimizer':cpu_copy(obj.optimizer.state_dict())}
                save(result_path(group,fold),{'status':'training','group':group,'fold':fold,'code_sha256':code_hash(),
                    'best':best,'history':history,'pairing':obj.pairing})
            if diagnostic: log(f'{group}/f{fold} e{epoch} val={float(vl):.6f} sem={record["semantic_normalized"]:.6f}')
            stopper(vl,obj.core)
            if stopper.early_stop: break
        torch.cuda.synchronize()
        seconds=time.perf_counter()-start
        end_train_rng=capture_rng()
        import_core(obj.core,best['core'])
        if obj.head: obj.head.load_state_dict(best['head'])
        obj.core.eval()
        assert int(obj.core.clf[2].num_batches_tracked)==initial_bn+best['epoch']+1
        test_rng=capture_rng()
        assert tree_hash(test_rng)==tree_hash(end_train_rng)
        # Historical evaluation has gradients enabled and exactly one stochastic forward.
        logits,_=obj.forward()
        assert hook.calls==2*(epoch+1)+1
        output=logits.detach().cpu().numpy()
        prob=softmax(output,axis=1)[:,0]
        test=obj.split['test']
        anchor=OLD/f'artifacts/seed_911/fold_{fold}'
        paths=[anchor/'train_start.pt',anchor/'splits.pt',anchor.parent/'selected_features.pt',
            OLD/f'artifacts/seed_911/vae/fold{fold}_pretrain.pth']
        result={'status':'complete','group':group,'fold':fold,'seed':CFG['seed'],'completed_utc':utc(),
            'code_sha256':code_hash(),'cache_sha256':sha_file(BASE/'runs/descriptors.bin'),
            'input_fingerprints':{str(p):sha_file(p) for p in paths},'pairing':obj.pairing,
            'head_initial_sha256':obj.head_initial_sha,'best':best,'history':history,'epochs':epoch+1,
            'indices':test,'ids':[obj.ctx['ids'][i] for i in test], 'labels':obj.ctx['labels'][test],
            'logits':output[test],'prob_asd':prob[test],'metrics':metrics(obj.ctx['labels'][test],prob[test]),
            'test_image_embedding':hook.image.detach().cpu(),'test_rng':test_rng,'end_rng':capture_rng(),
            'test_no_text_required':True,'classifier_seconds':seconds,'physical_gpu':os.environ['CUDA_VISIBLE_DEVICES'],
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'BN_batch_count':int(obj.core.clf[2].num_batches_tracked),'single_native_test_forward':True}
        save(result_path(group,fold),result)
        assert completed(group,fold)
        hook.close()
        log(f'COMPLETE {group}/f{fold} seconds={seconds:.2f} best={best["epoch"]} metrics={result["metrics"]}')


def resource_capacity(gpu,active):
    free,_=torch.cuda.mem_get_info(gpu)
    warming=sum(max(0,CFG['worker_memory_budget_GiB']*2**30-p.get('allocated',0)) for p in active)
    # Reserve worst-case growth for existing workers rather than trusting current free memory.
    available=free-warming-CFG['gpu_reserve_GiB']*2**30
    with open('/sys/fs/cgroup/memory/memory.limit_in_bytes') as f: limit=int(f.read())
    with open('/sys/fs/cgroup/memory/memory.usage_in_bytes') as f: used=int(f.read())
    return available>=CFG['worker_memory_budget_GiB']*2**30 and limit-used>8*2**30


def schedule(audit):
    audit.setdefault('execution',{'started_utc':utc(),'started_epoch':time.time(),'attempts':[],'failures':[]})
    save_json(BASE/'audit.json',audit)
    tasks=[(g,f) for f in range(10) for g in CFG['groups'] if not completed(g,f)]
    active=[]; failures=[]; retries={}; cap={0:2,1:2}
    sink=(BASE/'run.log').open('a',buffering=1)
    while tasks or active:
        for item in active[:]:
            rc=item['process'].poll()
            if rc is None: continue
            active.remove(item); key=(item['group'],item['fold'])
            if rc!=0:
                failed={k:v for k,v in item.items() if k!='process'}; failed['returncode']=rc
                audit['execution']['failures'].append(failed)
                cap[item['gpu']]=1
                if retries.get(key,0)<1:
                    retries[key]=retries.get(key,0)+1; tasks.append(key)
                else: failures.append(key)
            else: assert completed(*key)
            save_json(BASE/'audit.json',audit)
        for gpu in (0,1):
            own=[a for a in active if a['gpu']==gpu]
            finished_gpu=sum(completed(g,f) for f in range(10) if f%2==gpu for g in CFG['groups'])
            if finished_gpu>=2 and cap[gpu]>1:
                cap[gpu]=CFG['max_workers_per_gpu']
            candidates=[t for t in tasks if t[1]%2==gpu]
            if not candidates or len(own)>=cap[gpu] or not resource_capacity(gpu,own): continue
            group,fold=candidates[0]; tasks.remove((group,fold))
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONDONTWRITEBYTECODE':'1'}
            cmd=[sys.executable,'-B','-u',str(BASE/'run.py'),'worker','--group',group,'--fold',str(fold)]
            process=subprocess.Popen(cmd,env=env,cwd=BASE,stdout=sink,stderr=subprocess.STDOUT,start_new_session=True)
            item={'process':process,'group':group,'fold':fold,'gpu':gpu,'pid':process.pid,'started':utc(),'allocated':0}
            active.append(item)
            audit['execution']['attempts'].append({k:v for k,v in item.items() if k!='process'})
            save_json(BASE/'audit.json',audit)
            log(f'DISPATCH {group}/f{fold} PPU{gpu} pid={process.pid} active={len(own)+1}/{cap[gpu]}')
        time.sleep(1)
    sink.close()
    if failures: raise RuntimeError(f'Persistent failed tasks: {failures}')
    audit['execution']['completed_utc']=utc(); audit['execution']['completed_epoch']=time.time()
    audit['execution']['wall_seconds']=time.time()-audit['execution']['started_epoch']
    save_json(BASE/'audit.json',audit)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['all','prepare','preflight','worker','analyze'])
    parser.add_argument('--group',choices=CFG['groups']); parser.add_argument('--fold',type=int)
    args=parser.parse_args()
    if args.command=='worker': return train(args.group,args.fold)
    if args.command=='preflight': return preflight()
    if args.command=='analyze':
        from analyze import analyze
        return analyze()
    with locked(BASE/'runs/.coordinator.lock'):
        audit=audit_inputs(); prepare(audit)
        if args.command=='prepare': return
        for gpu in (0,1):
            out=BASE/f'runs/preflight_gpu{gpu}.json'
            if out.exists():
                assert json.loads(out.read_text())['passed']; continue
            env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
            subprocess.run([sys.executable,'-B','-u',str(BASE/'run.py'),'preflight'],env=env,cwd=BASE,check=True)
        audit['preflight']={str(g):json.loads((BASE/f'runs/preflight_gpu{g}.json').read_text()) for g in (0,1)}
        save_json(BASE/'audit.json',audit)
        schedule(audit)
        from analyze import analyze
        analyze()


if __name__=='__main__':
    (BASE/'runs').mkdir(exist_ok=True)
    try:
        main()
    except Exception as exc:
        log('ERROR '+repr(exc)+'\n'+traceback.format_exc())
        raise
