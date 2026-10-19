"""Independent E8 math and frozen native forward. No optimizer or gradient path."""
from __future__ import annotations
import hashlib
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import dense_to_sparse
from model.mm_gtunets import MM_GTUNets
from utils.tools import cal_feature_sim


def stable_seed(seed, fold, purpose):
    return int(hashlib.sha256(f'{seed}/{fold}/LCRA/{purpose}'.encode()).hexdigest()[:15], 16)


def array_sha(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def permutation(site, members, seed, fold, purpose):
    site, members = np.asarray(site), np.asarray(members, dtype=np.int64)
    p = np.arange(len(site), dtype=np.int64)
    singles = 0
    for name in sorted(set(site[members])):
        idx = members[site[members] == name]
        rng = np.random.default_rng(stable_seed(seed, fold, f'{purpose}/{name}'))
        order = rng.permutation(idx)
        if len(idx) > 1:
            p[order] = np.roll(order, int(rng.integers(1, len(idx))))
            assert np.all(p[idx] != idx)
        else:
            singles += len(idx)
        assert np.array_equal(np.sort(p[idx]), np.sort(idx))
    assert np.array_equal(np.sort(p[members]), np.sort(members))
    assert np.array_equal(site[members], site[p[members]])
    outside = np.ones(len(site), dtype=bool); outside[members] = False
    assert np.array_equal(p[outside], np.flatnonzero(outside))
    return p, singles


def phenotype_kernel(age_query, sex_query, age_anchor, sex_anchor, bandwidth):
    gap = age_query[..., None] - age_anchor
    known = ((sex_query[..., None] == 1) | (sex_query[..., None] == 2)) & ((sex_anchor == 1) | (sex_anchor == 2))
    same = sex_query[..., None] == sex_anchor
    weight = np.where(known, np.where(same, 1.0, 0.5), 0.75)
    return (np.exp(-0.5 * (gap / bandwidth) ** 2) * weight).astype(np.float32)


def prepare_inputs(x, meta, split, seed, fold, cfg):
    """Exact E8 preprocessing, copied with original fingerprints verified."""
    x = np.asarray(x, np.float64)
    age, sex, site = [np.asarray(meta[k]).copy() for k in ('age','sex','site')]
    tr = np.asarray(split['train'],np.int64); n = len(x)
    assert len(tr) >= 2 and np.isfinite(x).all()
    mean, sd = x[tr].mean(0), x[tr].std(0,ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    normalized = (x-mean)/sd
    normalized /= np.maximum(np.linalg.norm(normalized,axis=1,keepdims=True),1e-12)
    cos = normalized @ normalized[tr].T
    pos = np.full(n,-1,np.int64); pos[tr] = np.arange(len(tr))
    cos[tr,pos[tr]] = -np.inf
    k = min(cfg['knn'],len(tr)-1)
    local = np.argsort(-cos,axis=1,kind='stable')[:,:k]
    candidates = tr[local]
    raw_cos = np.take_along_axis(cos,local,axis=1).astype(np.float32)
    assert np.all(candidates[tr] != tr[:,None]) and np.isin(candidates,tr).all()
    assert all(len(set(row)) == k for row in candidates)
    missing_age = ~np.isfinite(age)
    median = float(np.nanmedian(age[tr])); assert np.isfinite(median)
    age[missing_age] = median
    diff = np.abs(age[tr,None]-age[tr][None,:])
    bandwidth = max(float(np.median(diff[~np.eye(len(tr),dtype=bool)])),cfg['age_bandwidth_floor'])
    shuffle = np.arange(n,dtype=np.int64); records = {}
    for part in ('train','val','test'):
        idx = np.asarray(split[part],np.int64)
        p,single = permutation(site,idx,seed,fold,f'L4-positive/{part}')
        shuffle[idx] = p[idx]
        records[part] = {'indices':idx.tolist(),'sources':p[idx].tolist(),'sha256':array_sha(p[idx]),
            'n':len(idx),'singleton_subjects':single,'moved_fraction':float(np.mean(p[idx]!=idx)),
            'changed_fraction':float(np.mean((age[idx]!=age[p[idx]])|(sex[idx]!=sex[p[idx]]))),
            'mean_abs_age_change':float(np.abs(age[idx]-age[p[idx]]).mean()),
            'sex_changed_fraction':float(np.mean(sex[idx]!=sex[p[idx]]))}
    nulls, null_info = [], []
    for b in range(cfg['num_null']):
        p,single = permutation(site,tr,seed,fold,f'null-anchor-{b}')
        nulls.append(p)
        null_info.append({'source_train_indices':p[tr].tolist(),'sha256':array_sha(p[tr]),'singleton_subjects':single})
    nulls = np.stack(nulls); kernels, null_kernels = {}, {}
    for name,mapping in (('true',np.arange(n)),('shuffled',shuffle)):
        a,s = age[mapping],sex[mapping]
        kernels[name] = phenotype_kernel(a,s,a[candidates],s[candidates],bandwidth)
        null_kernels[name] = np.stack([phenotype_kernel(a[tr],s[tr],a[p[candidates[tr]]],s[p[candidates[tr]]],bandwidth) for p in nulls])
    info = {'fit_indices':tr.tolist(),'train_feature_mean_sha256':array_sha(mean),'train_feature_sd_sha256':array_sha(sd),
        'age_train_mean':float(age[tr].mean()),'age_train_sd':float(age[tr].std()),'age_impute_train_median':median,
        'age_missing_count':int(missing_age.sum()),'sex_unknown_count':int(np.sum(~np.isin(sex,cfg['sex_codes']))),
        'age_bandwidth':bandwidth,'k':k,'anchor_order_sha256':array_sha(tr),
        'knn_sha256':array_sha(candidates),'raw_cos_sha256':array_sha(raw_cos),'L4':records,'nulls':null_info,
        'shuffle_sha256':array_sha(shuffle),'uses_diagnosis_labels':False,'fit_uses_heldout_covariates':False,
        'kernel_sha256':{key:array_sha(value) for key,value in kernels.items()},
        'null_kernel_sha256':{key:array_sha(value) for key,value in null_kernels.items()}}
    return {'candidates':candidates,'raw_cos':raw_cos,'kernels':kernels,'null_kernels':null_kernels,
        'shuffle':shuffle,'nulls':nulls,'info':info}


def js_divergence(p,q,eps=1e-8):
    m = 0.5*(p+q)
    return 0.5*((p*(p.clamp_min(eps).log()-m.clamp_min(eps).log())).sum(-1)
        +(q*(q.clamp_min(eps).log()-m.clamp_min(eps).log())).sum(-1))


class FrozenCore(MM_GTUNets):
    def init_vae_optimizer(self):
        self.vae_optimizer = None

    def train_vae(self,*args,**kwargs):
        raise RuntimeError('Experiment 9 forbids training')


class LCRA(nn.Module):
    def __init__(self,cfg):
        super().__init__(); self.cfg = cfg
        self.image = nn.Sequential(nn.Linear(500,32),nn.LayerNorm(32),nn.GELU(),nn.Linear(32,16))
        self.relation = nn.Sequential(nn.Linear(5,32),nn.GELU(),nn.Linear(32,1))
        nn.init.zeros_(self.relation[-1].weight); nn.init.zeros_(self.relation[-1].bias)

    def forward(self,x,inputs,train,group='L3',condition=None):
        spec = self.cfg['groups'][group]
        which = condition or ('shuffled' if spec['shuffled'] else 'true')
        candidates,raw = inputs['candidates'],inputs['raw_cos']
        z = F.normalize(self.image(x),p=2,dim=1)
        sim = (z[:,None,:]*z[candidates]).sum(-1)
        kernel = inputs['kernels'][which]
        if spec['phenotype']:
            features = torch.stack((sim,raw,kernel,sim*kernel,(sim-kernel).abs()),-1)
        else:
            zero = torch.zeros_like(sim)
            features = torch.stack((sim,raw,zero,zero,zero),-1)
        u = self.relation(features).squeeze(-1)
        centered = u-u.mean(1,keepdim=True)
        e = symmetric_edges(centered.tanh(),candidates,len(x))
        p = (sim[train]/self.cfg['tau_img']).softmax(-1)
        q = (kernel[train]/self.cfg['tau_ph']).softmax(-1)
        q_null = (inputs['null_kernels'][which]/self.cfg['tau_ph']).softmax(-1)
        jt = js_divergence(p,q,self.cfg['epsilon'])
        jn = js_divergence(p[None],q_null,self.cfg['epsilon']).mean(0)
        local = self.cfg['tau_margin']*F.softplus((jt-jn+self.cfg['margin'])/self.cfg['tau_margin'])
        return {'E':e,'Z':z,'P':p,'Q':q,'J_true':jt,'J_null':jn,'L_local':local.mean(),
            'centered':centered,'u':u,'kernel':kernel,'sim':sim}


def symmetric_edges(d,candidates,n):
    dense = d.new_zeros((n,n)).scatter(1,candidates,d)
    e = 0.5*(dense+dense.T)
    return e*(1-torch.eye(n,dtype=d.dtype,device=d.device))


def random_signs(seed,fold,n,device):
    h = np.ones((n,n),np.float32)
    for i in range(n):
        for j in range(i+1,n):
            bit = hashlib.sha256(f'LCRA9/random-sign/{seed}/{fold}/{i}/{j}'.encode()).digest()[0]&1
            h[i,j] = h[j,i] = 1.0 if bit else -1.0
    return torch.as_tensor(h,device=device)


class FrozenWrapper(nn.Module):
    def __init__(self,core,aligner,inputs,train,cfg,signs):
        super().__init__(); self.core=core; self.aligner=aligner
        self.inputs=inputs; self.train_indices=train; self.cfg=cfg; self.signs=signs

    @torch.no_grad()
    def forward(self,x,ph,affinity,mode='A1'):
        assert not self.training and not self.core.training and not self.aligner.training
        assert not torch.is_grad_enabled()
        core=self.core
        if core.load_pretrain: core.load_vae()
        ph,_=core.vae(ph)
        core.rp_graph,core.value=core.create_rp_graph(affinity)
        core.fused_sim=cal_feature_sim(torch.cat((x,ph),dim=1))
        base=core.fused_sim*core.rp_graph
        detail=self.aligner(x,self.inputs,self.train_indices,condition='shuffled' if mode in ('A4','replay_shuffled') else 'true')
        # Every condition uses the same scale from true TRAIN queries/anchors.
        if mode=='A0': e=detail['E']*0
        elif mode in ('A1','replay_shuffled'): e=detail['E']
        elif mode=='A6': e=detail['E']*0
        else:
            gamma=self.cfg['gamma_weak'] if mode=='A2' else self.cfg['gamma_strong']
            d=torch.tanh(gamma*detail['centered']/self.scale)
            if self.degenerate: d=d*0
            e=symmetric_edges(d,self.inputs['candidates'],len(x))
            if mode=='A5': e=e*self.signs
        graph=base*(1+self.cfg['alpha']*e)
        if mode=='A6':
            off=~torch.eye(len(x),device=x.device,dtype=torch.bool)
            graph=torch.where(off & (base>0),base*self.cfg['A6_off_diagonal_multiplier'],base)
        core.fused_graph=graph
        indices,attr=dense_to_sparse(graph); attr=attr.view(-1,1)
        img=core.img_unet(x,indices,attr)
        non=core.ph_unet(ph,indices,attr)
        core.joint_embed,core.img_weight,core.ph_weight=core.mm_attention(img,non)
        loss=core.cal_graph_loss(img,non)
        logits=core.clf(core.joint_embed)
        self.last={'base':base,'graph':graph,'E':e,'img':img,'ph':non,'joint':core.joint_embed,
            'logits':logits,'graph_loss':loss,'relation':detail,'sparse_index':indices,'sparse_attr':attr}
        return logits,loss
