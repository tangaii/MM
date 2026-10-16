"""Self-contained fold-safe PCA and cross-fitted TD reference implementation."""
import hashlib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold


def stable_seed(seed, fold, name):
    return int(hashlib.sha256(f'{seed}/{fold}/{name}'.encode()).hexdigest()[:8],16)


def make_shuffle(site, age, sex, seed, fold, split):
    site,age,sex=np.asarray(site),np.asarray(age),np.asarray(sex)
    p=np.arange(len(age))
    records={}
    for partition in ('train','val','test'):
        members=np.asarray(split[partition],dtype=np.int64)
        sites={}
        for s in np.unique(site[members]):
            ids=members[site[members]==s]
            rng=np.random.default_rng(stable_seed(seed,fold,f'PCND-partition-site-shuffle/{partition}/{s}'))
            ordered=rng.permutation(ids)
            if len(ids)>1:
                p[ordered]=np.roll(ordered,1)
                assert np.all(p[ids]!=ids)
            assert np.array_equal(np.sort(p[ids]),np.sort(ids))
            changed=(age[ids]!=age[p[ids]]) | (sex[ids]!=sex[p[ids]])
            sites[str(s)]={'n':int(len(ids)),'singleton':len(ids)==1,
                'changed_fraction':float(changed.mean()),'changed_count':int(changed.sum()),
                'indices':ids.tolist(),'source_indices':p[ids].tolist()}
        assert np.array_equal(np.sort(p[members]),np.sort(members))
        changed=(age[members]!=age[p[members]]) | (sex[members]!=sex[p[members]])
        records[partition]={'indices':members.tolist(),'source_indices':p[members].tolist(),
            'permutation_sha256':hashlib.sha256(p[members].astype('<i8').tobytes()).hexdigest(),
            'changed_fraction':float(changed.mean()),'changed_count':int(changed.sum()),
            'permuted_fraction':float(np.mean(p[members]!=members)),'sites':sites}
    assert np.all(site==site[p])
    assert np.array_equal(np.sort(p),np.arange(len(age)))
    changed=(age!=age[p]) | (sex!=sex[p])
    return p, {'scope':'partition_within_site','version':2,
        'all_changed':float(changed.mean()),'permuted':float(np.mean(p!=np.arange(len(p)))),
        'permutation':p.tolist(),'permutation_sha256':hashlib.sha256(p.astype('<i8').tobytes()).hexdigest(),
        'changed_mask':changed.tolist(),'partitions':records,'uses_diagnosis_labels':False}


def design(age, sex, kind, a0, a_scale, sex_mean):
    n=len(age)
    a=np.clip((age-a0)/a_scale,-3,3)
    cols=[np.ones(n,dtype=np.float64)]
    if kind in ('age','age_sex'):
        cols.extend([a,a*a])
    if kind=='age_sex':
        cols.append((sex==2).astype(float)-sex_mean)
    return np.stack(cols,axis=1)


def ridge_fit(d,z,lamb=10.0):
    penalty=np.eye(d.shape[1])*lamb
    penalty[0,0]=0.0
    return np.linalg.solve(d.T@d+penalty,d.T@z)


def density_gate(age, sex, reference_age, reference_sex):
    ref_age=np.asarray(reference_age,float)
    ref_sex=np.asarray(reference_sex,int)
    scale=max(float(np.subtract(*np.percentile(ref_age,[75,25]))),2.0)
    out=np.empty(len(age),float)
    k=min(10,len(ref_age))
    for i,(a,s) in enumerate(zip(age,sex)):
        dist=np.abs(ref_age-float(a))/scale+0.5*(ref_sex!=s)
        nearest=np.partition(dist,k-1)[k-1]
        out[i]=np.clip(np.exp(-.5*(nearest/.75)**2),.25,1.0)
    return out.astype('float32')


def prepare_fold(x, labels, age, sex, site, split, seed, fold, dims=32, groups_to_compute=None, pca=None):
    """x is the *historical* (871,500) RFE-selected matrix for this seed."""
    x=np.asarray(x,dtype=np.float64)
    y=np.asarray(labels,dtype=np.int64)
    age=np.asarray(age,dtype=np.float64)
    sex=np.asarray(sex,dtype=np.int64)
    site=np.asarray(site)
    tr=np.asarray(split['train'],dtype=np.int64)
    va=np.asarray(split['val'],dtype=np.int64)
    te=np.asarray(split['test'],dtype=np.int64)
    n=len(y)
    assert x.shape[0]==n==len(age)==len(sex)==len(site)
    assert set(np.unique(y))=={0,1} and set(np.unique(sex))<={1,2}
    assert len(np.unique(np.r_[tr,va,te]))==n
    td=tr[y[tr]==1]
    assert len(td)>=50
    if pca is None:
        pca=PCA(n_components=dims, svd_solver='full')
        pca.fit(x[tr])
    pc=pca.transform(x)
    pc_std=np.maximum(pc[tr].std(axis=0,ddof=1),1e-4)
    z=pc/pc_std
    a0=float(np.mean(age[td])); a_scale=max(float(np.std(age[td],ddof=1)),2.0)
    sm=float(np.mean(sex[td]==2))
    perm, shuffle_stats=make_shuffle(site,age,sex,seed,fold,split)
    age_perm,sex_perm=age[perm],sex[perm]
    out={}
    groups={'P1':'raw','P2':'global','P3':'age','P4':'age_sex','P5':'age_sex','P6':'age_sex'}
    if groups_to_compute is not None:
        groups={g:groups[g] for g in groups_to_compute}
    internal=list(StratifiedKFold(n_splits=5,shuffle=True,
                    random_state=stable_seed(seed,fold,'PCND-inner')).split(tr,y[tr]))
    for g,kind in groups.items():
        if g=='P1':
            out[g]={'features':np.clip(z,-4,4).astype('float32'),
                    'gate':np.ones(n,dtype='float32')}
            continue
        c_age=age_perm if g=='P6' else age
        c_sex=sex_perm if g=='P6' else sex
        design_kind='global' if kind=='global' else kind
        d=design(c_age,c_sex,design_kind,a0,a_scale,sm)
        predicted=np.empty_like(z)
        gate=np.empty(n,dtype='float32')
        for a,b in internal:
            fit_idx=tr[a]; held=tr[b]
            fit_td=fit_idx[y[fit_idx]==1]
            assert len(fit_td)>=50
            w=ridge_fit(d[fit_td],z[fit_td]); predicted[held]=d[held]@w
            gate[held]=density_gate(c_age[held],c_sex[held],c_age[fit_td],c_sex[fit_td])
        w_full=ridge_fit(d[td],z[td]); outside=np.r_[va,te]
        predicted[outside]=d[outside]@w_full
        gate[outside]=density_gate(c_age[outside],c_sex[outside],c_age[td],c_sex[td])
        scale=np.maximum((z[td]-predicted[td]).std(axis=0,ddof=1),.25)
        dev=np.clip((z-predicted)/scale,-4,4)
        out[g]={'features':dev.astype('float32'),
                'gate':gate if g in ('P5','P6') else np.ones(n,dtype='float32')}
        if g in ('P4','P5'):
            alt=design(age_perm,sex_perm,'age_sex',a0,a_scale,sm)
            counter=np.empty_like(z)
            for a,b in internal:
                fit_idx=tr[a]; held=tr[b]
                fit_td=fit_idx[y[fit_idx]==1]
                w=ridge_fit(d[fit_td],z[fit_td])
                counter[held]=alt[held]@w
            counter[outside]=alt[outside]@w_full
            alt_gate=np.empty(n,dtype='float32')
            for a,b in internal:
                fit_idx=tr[a]; held=tr[b]
                fit_td=fit_idx[y[fit_idx]==1]
                alt_gate[held]=density_gate(age_perm[held],sex_perm[held],age[fit_td],sex[fit_td])
            alt_gate[outside]=density_gate(age_perm[outside],sex_perm[outside],age[td],sex[td])
            out[g]['shuffled_features']=np.clip((z-counter)/scale,-4,4).astype('float32')
            out[g]['shuffled_gate']=alt_gate if g=='P5' else np.ones(n,dtype='float32')
    assert all(v['features'].shape==(n,dims) and np.isfinite(v['features']).all() for v in out.values())
    return out, {'train_td':len(td),'pca_explained_sum':float(pca.explained_variance_ratio_.sum()),
                 'shuffle':shuffle_stats, 'pca_training_indices':tr.tolist(),
                 'normative_reference_fit_only_td':True,
                 'normative_inner_splits':5,'pca_components':dims}


