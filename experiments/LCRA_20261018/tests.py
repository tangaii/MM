"""Concentrated synthetic numerical acceptance; no CPU neural training."""
import math
import numpy as np
import torch
from torch.nn import functional as F
from lcra import LCRA, prepare_inputs, phenotype_kernel, js_divergence


def synthetic_checks(device):
    from run import CFG, graph_check, state_hash, cpu_state
    names = []
    rng = np.random.default_rng(882381)
    x = rng.normal(size=(11,500)).astype(np.float32)
    meta = {'age':np.array([10.,12.,14.,16.,18.,np.nan,20.,22.,25.,30.,40.]),
        'sex':np.array([1,2,1,2,0,1,2,1,2,1,-1]),
        'site':np.array(['a','a','b','b','c','a','b','c','d','d','d'])}
    split = {'train':np.array([0,1,2,3,4]),'val':np.array([5,6,7]),'test':np.array([8,9,10])}
    raw = prepare_inputs(x,meta,split,911,0,CFG)
    assert raw['candidates'].shape == (11,4)
    assert np.isin(raw['candidates'],split['train']).all()
    assert (raw['candidates'][split['train']] != split['train'][:,None]).all()
    assert raw['info']['L4']['val']['singleton_subjects'] == 3
    names.append('train_only_distinct_candidates_self_exclusion_small_K_and_singletons')
    assert raw['info']['sex_unknown_count'] == 2 and raw['info']['age_missing_count']==1
    for part,row in raw['info']['L4'].items():
        a,b = np.asarray(row['indices']),np.asarray(row['sources'])
        assert np.array_equal(np.sort(a),np.sort(b)) and np.array_equal(meta['site'][a],meta['site'][b])
    names.append('joint_partition_site_derangement_and_unknown_imputation')
    kernel = phenotype_kernel(np.array([10.,10.,10.]),np.array([1,2,0]),np.full((3,1),10.),np.ones((3,1)),2.)
    assert np.array_equal(kernel[:,0],np.array([1.,.5,.75]))
    names.append('age_sex_kernel_closed_form')
    for v in list(raw['kernels'].values())+list(raw['null_kernels'].values()):
        assert np.isfinite(v).all() and np.all((v>=0)&(v<=1))
    names.append('true_null_kernel_bounds_finite')
    dtype = torch.float64 if device.type=='cpu' else torch.float32
    p = torch.tensor([[.2,.3,.5],[1.,0.,0.]],dtype=dtype,device=device,requires_grad=True)
    q = torch.tensor([[.4,.1,.5],[0.,1.,0.]],dtype=dtype,device=device)
    j = js_divergence(p,q)
    assert torch.allclose(j,js_divergence(q,p),rtol=0,atol=0)
    assert bool((j>=0).all()) and abs(float(j[1])-math.log(2))<2e-7
    assert torch.equal(js_divergence(q,q),torch.zeros(2,dtype=dtype,device=device))
    j.sum().backward(); assert torch.isfinite(p.grad).all()
    names.append('JS_symmetry_identical_disjoint_and_finite_gradient')
    if device.type=='cpu':
        logits = torch.tensor([[.1,.3,-.4]],dtype=torch.float64,requires_grad=True)
        target = torch.tensor([[.4,.35,.25]],dtype=torch.float64)
        assert torch.autograd.gradcheck(lambda v: js_divergence(v.softmax(-1),target),(logits,),eps=1e-6,atol=1e-6,rtol=1e-4)
        names.append('JS_double_precision_finite_difference_gradcheck')
    gap = torch.tensor([-2.,0.,2.],device=device,requires_grad=True)
    soft = CFG['tau_margin']*F.softplus((gap+CFG['margin'])/CFG['tau_margin'])
    soft.sum().backward()
    assert bool((gap.grad>0).all()) and torch.isfinite(gap.grad).all()
    names.append('softplus_margin_no_hard_zero_region')
    inputs = {'candidates':torch.as_tensor(raw['candidates'],device=device),
        'raw_cos':torch.as_tensor(raw['raw_cos'],device=device),
        'kernels':{k:torch.as_tensor(v,device=device) for k,v in raw['kernels'].items()},
        'null_kernels':{k:torch.as_tensor(v,device=device) for k,v in raw['null_kernels'].items()}}
    tensor = torch.as_tensor(x,device=device)
    train = torch.as_tensor(split['train'],device=device)
    model = LCRA.isolated(CFG,911,0,device)
    initial_hash = state_hash(cpu_state(model))
    for group in CFG['groups']:
        other = LCRA.isolated(CFG,911,0,device)
        assert state_hash(cpu_state(other))==initial_hash
        detail = other(tensor,inputs,train,group)
        assert torch.count_nonzero(detail['E'])==0
        assert torch.isfinite(detail['L_local'])
    names.append('all_groups_identical_private_initialization_zero_head_graph_exact')
    weights = torch.as_tensor(rng.uniform(.1,1.,(11,11)).astype(np.float32),device=device)
    weights = .5*(weights+weights.T)
    detail = model(tensor,inputs,train,'L3')
    loss = (detail['E']*weights).square().sum()+detail['E'].mul(weights.square()).sum()
    gs = torch.autograd.grad(loss,list(model.relation.parameters()),retain_graph=True)
    assert any(bool((v!=0).any()) for v in gs)
    local_grads = torch.autograd.grad(detail['L_local'],list(model.image.parameters()),retain_graph=True)
    assert all(torch.isfinite(v).all() for v in local_grads) and any(bool((v!=0).any()) for v in local_grads)
    names.append('zero_head_last_layer_native_surrogate_and_separate_local_projection_gradients')
    with torch.no_grad():
        model.relation[-1].weight.fill_(.02)
    detail = model(tensor,inputs,train,'L3')
    a = weights.clone(); a[0,1]=a[1,0]=0
    b = a*(1+CFG['alpha']*detail['E'])
    graph_check({'A_base':a,'A_new':b,'E':detail['E']})
    graph_check({'A_base':a,'A_new':a*(1+0*detail['E']),'E':detail['E']},alpha=0)
    graph_grads = torch.autograd.grad((b*weights.square()).sum(),list(model.image.parameters()),retain_graph=True)
    assert all(torch.isfinite(v).all() for v in graph_grads) and any(bool((v!=0).any()) for v in graph_grads)
    names.append('learned_symmetric_supported_nonnegative_bounded_graph_projection_gradient')
    names.append('no_CPU_optimizer_or_neural_training')
    return {'passed':True,'device':str(device),'checks':names,'n':len(names)}
