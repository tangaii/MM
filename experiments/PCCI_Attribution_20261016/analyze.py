"""Frozen layer diagnostics, paired OOF statistics and final Chinese report."""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
import time
import numpy as np
from scipy.special import softmax
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, recall_score, f1_score
import torch
from torch.nn import functional as F
import run as r
from controls import PCCI, Bridge, private_seed


def layers(core, value):
    first=F.linear(value,core.clf[0].weight,core.clf[0].bias)
    relu=F.relu(first)
    bn=core.clf[2]
    normalized=F.batch_norm(relu,bn.running_mean,bn.running_var,bn.weight,bn.bias,False,0,bn.eps)
    logits=F.linear(normalized,core.clf[3].weight,core.clf[3].bias)
    return {'input':value,'linear1':first,'relu':relu,'bn':normalized,'logits':logits}


def norms(value):
    return r.distribution(value.norm(dim=1))


def local_gradient_paths(loaded,method,group):
    image=loaded.bridge.imaging.detach().requires_grad_(True)
    phenotype=loaded.bridge.phenotype.detach().requires_grad_(True)
    joint=loaded.bridge.base.detach().requires_grad_(True)
    module=loaded.module
    if module is None or group=='R1':
        delta=torch.zeros_like(joint)
    else:
        delta=module.components(image.detach() if group=='R2' else image,loaded.conditions,
                                loaded.reference,method,permutation=loaded.permutation)['delta']
    loss=F.cross_entropy(layers(loaded.core,joint+delta)['logits'][loaded.train],loaded.labels[loaded.train])
    direct,=torch.autograd.grad(loss,image,allow_unused=True,retain_graph=True)
    parameters=list(module.parameters()) if module is not None else []
    parameter_grads=torch.autograd.grad(loss,parameters,allow_unused=True,retain_graph=True) if parameters else []
    classifier_grads=torch.autograd.grad(loss,list(loaded.core.clf.parameters()),allow_unused=True,retain_graph=True)
    native_joint,_,_=loaded.core.mm_attention(image,phenotype)
    graph=loaded.core.cal_graph_loss(image,phenotype)
    fusion_loss=F.cross_entropy(layers(loaded.core,native_joint+delta.detach())['logits'][loaded.train],loaded.labels[loaded.train])+graph
    native_image,native_ph=torch.autograd.grad(fusion_loss,(image,phenotype))
    norm=lambda gradients:float(torch.stack([v.detach().square().sum() for v in gradients if v is not None]).sum().sqrt()) if any(v is not None for v in gradients) else 0.0
    cosine=None
    if direct is not None and float(direct.norm())>0 and float(native_image.norm())>0:
        cosine=float(F.cosine_similarity(direct.flatten(),native_image.flatten(),dim=0))
    return {'direct_residual_imaging_output_gradient_l2':0.0 if direct is None else float(direct.norm()),
            'native_fusion_plus_graph_imaging_output_gradient_l2':float(native_image.norm()),
            'native_phenotype_output_gradient_l2':float(native_ph.norm()),
            'direct_to_native_ratio':0.0 if direct is None else float(direct.norm()/native_image.norm().clamp_min(1e-12)),
            'direct_vs_native_cosine':cosine,'adapter_CE_gradient_l2':norm(parameter_grads),
            'classifier_CE_gradient_l2':norm(classifier_grads),
            'scope':'Local eval-BN gradients at saved embeddings; not historical training-parameter gradients'}


def diagnose_one(unused,seed,fold,state):
    loaded=r.Loaded(seed,fold,state)
    diagnostic_source_sha256=r.sha_file(r.BASE/'analyze.py')
    affinity_input_sha256=r.array_hash(loaded.affinity_cpu)
    answer={}
    baseline_population_graph=None
    try:
        for group in ('C0','C1','C2','C3','C4','C5','C6',*r.GROUPS):
            if loaded.bridge is not None: loaded.bridge.close()
            loaded.frozen=None
            loaded.module=None
            loaded.group='R0'
            if group=='C0':
                saved=r.load_torch(r.ROOT/state['audit']['baseline'][f'{seed}/{fold}']['best_reference'])
                core_state=saved['core']; method=None
            elif group.startswith('C'):
                saved=r.load_torch(r.HIST/f'runs/{group}/s{seed}_f{fold}.pt',mmap=True)
                core_state=saved['core']; method=group
            else:
                saved=r.load_torch(r.result_path(group,seed,fold),mmap=True)
                core_state=saved['core']; method=r.config()['groups'][group]['adapter']
                loaded.group=group
            r.import_core(loaded.core,core_state)
            if method is not None:
                loaded.module=PCCI.isolated(r.config(),private_seed(seed,fold,'PCCI-initialization'),loaded.opt.device)
                loaded.module.load_state_dict(saved['pcci'])
                loaded.module.eval()
            loaded.bridge=Bridge(loaded.core,loaded.module,loaded.conditions,loaded.reference,loaded.permutation,
                                 method,inert=group=='R1',detach=group=='R2')
            core=loaded.core
            core.eval()
            formal_replay=None
            if group.startswith('R'):
                r.restore_rng(saved['formal_full_rng_before'])
                with torch.enable_grad():
                    replay_logits,_=loaded.forward()
                indices=loaded.split['test']
                expected=torch.as_tensor(np.asarray(saved['oof']['logits']),dtype=replay_logits.dtype,device=replay_logits.device)
                replay_error=r.compare(replay_logits[indices],expected,f'{group} saved formal logits',atol=2e-5,rtol=3e-5)
                replay_pred=replay_logits.detach().argmax(1).cpu().numpy()[indices]
                if not np.array_equal(replay_pred,saved['oof']['pred']):
                    raise AssertionError(f'{group} formal replay changed predicted labels')
                formal_replay={'max_logit_error':replay_error,'predictions_exact_equal':True,
                               'original_full_rng_restored':True,'saved_formal_result_not_modified':True}
                del replay_logits
                r.import_core(core,core_state)
            r.restore_rng(loaded.start['rng'])
            draw_seed=r.draw_rng(seed,fold,0)
            bn_before={k:r.array_hash(v) for k,v in core.clf[2].state_dict().items()}
            with torch.no_grad():
                logits,_=loaded.forward()
                if group=='C0':
                    baseline_population_graph=core.fused_graph.detach().clone()
                graph_error=(r.compare(core.fused_graph,baseline_population_graph,'R3 vs C0 frozen population graph') if group=='R3' else None)
                image,phenotype,base=loaded.bridge.imaging,loaded.bridge.phenotype,loaded.bridge.base
                delta=loaded.bridge.last['delta'] if loaded.module is not None else torch.zeros_like(base)
                test=loaded.split['test']
                a=layers(core,base)
                cases={}
                for condition in r.config()['counterfactuals']:
                    if loaded.module is None:
                        alternative=torch.zeros_like(base)
                    else:
                        alternative=loaded.module.components(image,loaded.conditions,loaded.reference,method,condition,loaded.permutation)['delta']
                        if group=='R1': alternative=torch.zeros_like(alternative)
                    b=layers(core,base+alternative)
                    p=softmax(b['logits'].cpu().numpy(),axis=1)[:,0]
                    cases[condition]={'layers_l2':{k:norms(v[test]) for k,v in b.items()},
                                      'difference_vs_off_l2':{k:norms((b[k]-a[k])[test]) for k in b},
                                      'relu_active_change_fraction':float(((b['linear1'][test]>0)!=(a['linear1'][test]>0)).float().mean()),
                                      'test_prob_asd':p[test].tolist()}
                off_p=softmax(a['logits'].cpu().numpy(),axis=1)[test,0]
                true_p=np.asarray(cases['true']['test_prob_asd'])
                previous=(old_fold(group,seed,fold,state) if group.startswith('C') else saved)
                expected=np.asarray(previous['frozen']['test_prob_asd'][0])
                replay_error=float(np.max(np.abs(true_p-expected)))
                replay_labels=np.where(true_p>=.5,0,1)
                replay_labels_equal=bool(np.array_equal(replay_labels,previous['frozen']['test_pred'][0]))
                replay_auc_difference=r.metrics(previous['oof']['labels'],replay_labels,true_p)['AUC']-previous['frozen']['draw_metrics'][0]['AUC']
                if replay_error>2e-5:
                    # An independently audited FP32 backend variation is not a new formal prediction.
                    if replay_error>1e-4 or not replay_labels_equal or abs(replay_auc_difference)>1e-12:
                        raise AssertionError(f'{group} diagnostic replay exceeds bounded numerical variation: {replay_error}')
                bn=core.clf[2]
                gain=bn.weight/(bn.running_var+bn.eps).sqrt()
                cos=F.cosine_similarity(base[test],delta[test],dim=1,eps=1e-12)
                valid_angle=(base[test].norm(dim=1)>1e-12)&(delta[test].norm(dim=1)>1e-12)
                info={'draw_seed':draw_seed,'diagnostic_source_sha256':diagnostic_source_sha256,
                      'formal_prediction_replay':formal_replay,
                      'affinity_input_sha256':affinity_input_sha256,
                      'R3_C0_population_graph_max_error':graph_error,
                      'imaging_l2':norms(image[test]),'phenotype_l2':norms(phenotype[test]),
                      'joint_l2':norms(base[test]),'delta_l2':norms(delta[test]),
                      'delta_joint_ratio':r.distribution(delta[test].norm(dim=1)/base[test].norm(dim=1).clamp_min(1e-12)),
                      'joint_delta_cosine':r.distribution(cos[valid_angle]),
                      'joint_delta_angle_degrees':r.distribution(torch.acos(cos[valid_angle].clamp(-1,1))*180/np.pi),
                      'angles_exclude_zero_vectors':True,
                      'BN_running_mean':r.distribution(bn.running_mean),'BN_running_var':r.distribution(bn.running_var),
                      'BN_gain_abs':r.distribution(gain.abs()),'BN_batches':int(bn.num_batches_tracked),
                      'first_linear_weight_spectral_norm':float(np.linalg.norm(core.clf[0].weight.detach().cpu().numpy(),ord=2)),
                      'last_linear_weight_spectral_norm':float(np.linalg.norm(core.clf[3].weight.detach().cpu().numpy(),ord=2)),
                      'cases':cases,'true_off_probability_MAD':float(np.abs(true_p-off_p).mean()),
                      'saved_draw0_max_probability_error':replay_error,
                      'saved_draw0_labels_equal':replay_labels_equal,
                      'saved_draw0_AUC_difference':replay_auc_difference,
                      'bounded_native_numerical_exception':replay_error>2e-5,
                      'fixed_eval_BN':True,'one_core_forward':True,
                      'core_forward_vs_manual_logit_max_error':float((logits-layers(core,base+delta)['logits']).abs().max())}
            # Exact chain-rule JVP for Linear -> ReLU -> fixed eval BN -> Linear.
            with torch.no_grad():
                pre=F.linear(base,core.clf[0].weight,core.clf[0].bias)
                tangent=F.linear(delta,core.clf[0].weight,None)*(pre>0)
                tangent=tangent*core.clf[2].weight/(core.clf[2].running_var+core.clf[2].eps).sqrt()
                jvp=F.linear(tangent,core.clf[3].weight,None)
            actual=(layers(core,(base+delta).detach())['logits']-layers(core,base.detach())['logits']).detach()
            info['JVP_logit_l2']=norms(jvp[test])
            info['actual_logit_change_l2']=norms(actual[test])
            info['JVP_relative_error']=float((jvp[test]-actual[test]).norm()/actual[test].norm().clamp_min(1e-12))
            info['local_gradient_paths']=local_gradient_paths(loaded,method,group)
            if {k:r.array_hash(v) for k,v in core.clf[2].state_dict().items()}!=bn_before:
                raise AssertionError('Counterfactual analysis changed BN statistics')
            if group=='C4':
                # Historical reference also replaces site; add the requested age/sex-only reference.
                reference_probs=[]
                condition=loaded.conditions.clone()
                condition[:,:3]=loaded.reference[:3]
                for draw in range(5):
                    r.import_core(core,core_state)
                    r.restore_rng(loaded.start['rng']);r.draw_rng(seed,fold,draw)
                    with torch.no_grad():
                        loaded.forward()
                        value=loaded.module.components(loaded.bridge.imaging,condition,loaded.reference,'C4',permutation=loaded.permutation)
                        probability=softmax(layers(core,loaded.bridge.base+value['delta'])['logits'].cpu().numpy(),axis=1)[:,0]
                        reference_probs.append(probability[test])
                info['age_sex_reference_keep_site_test_prob_asd']=np.stack(reference_probs).tolist()
                info['historical_reference_also_changes_site']=True
            answer[group]=info
        return answer
    finally:
        loaded.close()


def packed(rows):
    seeds={}
    for seed in r.config()['seeds']:
        selected=[v for v in rows if v['seed']==seed]
        if len(selected)!=10:
            raise AssertionError('Need ten folds per seed')
        seeds[str(seed)]={k:float(np.mean([v['metrics'][k] for v in selected])) for k in r.KEYS}
    return {'seed_means':seeds,
            'mean':{k:float(np.mean([v[k] for v in seeds.values()])) for k in r.KEYS},
            'seed_sd_ddof1':{k:float(np.std([v[k] for v in seeds.values()],ddof=1)) for k in r.KEYS}}


def old_fold(group,seed,fold,state):
    if group=='C0':
        b=state['audit']['baseline'][f'{seed}/{fold}']
        frozen=json.loads((r.HIST/'summary.json').read_text())['baseline_frozen'][f'{seed}/{fold}']
        return {**b,'oof':{k:b[k] for k in ('indices','labels','pred','prob_asd','ids','sites','metrics')},'frozen':frozen}
    return r.load_torch(r.HIST/f'runs/{group}/s{seed}_f{fold}.pt',mmap=True)


def independent_metrics(labels,pred,prob):
    y,p=np.asarray(labels)==0,np.asarray(pred)==0
    return {'ACC':float(accuracy_score(y,p)),'BA':float(balanced_accuracy_score(y,p)),
            'AUC':float(roc_auc_score(y,prob)),'SEN':float(recall_score(y,p)),
            'SPE':float(recall_score(~y,~p)),'F1':float(f1_score(y,p))}


def formal_signature(saved):
    fields=('group','seed','fold','training_code_sha256','config_digest','input_sha256',
            'best_core_sha256','best_pcci_sha256','best_epoch','best_validation_loss',
            'epochs_run','epochs','oof','formal_rng_before','formal_rng_after',
            'optimizer_core_parameter_order','parameter_changes','frozen_guard')
    return r.digest({**{k:saved[k] for k in fields},
                     'formal_full_rng_hash':r.rng_hash(saved['formal_full_rng_before'])})


def frozen_signature(frozen):
    def encode(value):
        if isinstance(value,np.ndarray) or torch.is_tensor(value):
            return {'array_sha256':r.array_hash(value)}
        if isinstance(value,dict):
            return {key:encode(item) for key,item in value.items()}
        if isinstance(value,(list,tuple)):
            return [encode(item) for item in value]
        return value
    return r.digest(encode(frozen))


def verified_counterfactual(saved):
    if saved.get('group') in ('R2','R3'):
        audit=saved.get('counterfactual_audit_v2')
        if audit is None or not audit['validated']:
            raise AssertionError('Missing fixed-base counterfactual audit')
        if audit['formal_signature_before']!=formal_signature(saved) or audit['original_frozen_signature']!=frozen_signature(saved['frozen']):
            raise AssertionError('Formal or original five-draw result changed')
        return audit
    return saved['frozen']


def repair_counterfactual_one(seed,fold,state):
    loaded=r.Loaded(seed,fold,state)
    results={}
    try:
        for group in ('R2','R3'):
            path=r.result_path(group,seed,fold)
            saved=r.load_torch(path,mmap=True)
            if 'counterfactual_audit_v2' in saved:
                verified_counterfactual(saved)
                results[group]={'skipped':True}
                continue
            if loaded.bridge is not None:
                loaded.bridge.close()
            loaded.frozen=None
            loaded.module=None
            # A fresh instance preserves R3's strict freeze checks and R2's native flags.
            if group=='R3':
                loaded.close()
                loaded=None
                loaded=r.Loaded(seed,fold,state)
            loaded.attach(group)
            loaded.module.load_state_dict(saved['pcci'])
            loaded.mode(False)
            r.import_core(loaded.core,saved['core'])
            before_formal=formal_signature(saved)
            before_frozen=frozen_signature(saved['frozen'])
            counter={name:[] for name in r.config()['counterfactuals']}
            errors=[];off_errors=[];manual_errors=[]
            bn_before={k:r.array_hash(v) for k,v in loaded.core.clf[2].state_dict().items()}
            base_reference=json.loads((r.HIST/'summary.json').read_text())['baseline_frozen'][f'{seed}/{fold}'] if group=='R3' else None
            for draw in range(5):
                r.import_core(loaded.core,saved['core'])
                r.restore_rng(loaded.start['rng']);r.draw_rng(seed,fold,draw)
                with torch.no_grad():
                    logits,_=loaded.forward()
                    image=loaded.bridge.imaging
                    base=loaded.bridge.base
                    base_hash=r.array_hash(base)
                    test=loaded.split['test']
                    for intervention in r.config()['counterfactuals']:
                        value=loaded.module.components(image,loaded.conditions,loaded.reference,
                                                       r.config()['groups'][group]['adapter'],intervention,loaded.permutation)
                        scores=layers(loaded.core,base+value['delta'])['logits']
                        probability=softmax(scores.cpu().numpy(),axis=1)[:,0]
                        counter[intervention].append(probability)
                        if intervention=='true':
                            manual_errors.append(r.compare(scores,logits,'fixed-base true versus original classifier'))
                    if r.array_hash(loaded.bridge.base)!=base_hash or r.array_hash(base)!=base_hash:
                        raise AssertionError('A counterfactual changed the original joint embedding')
                    actual=counter['true'][-1][test]
                    old=np.asarray(saved['frozen']['test_prob_asd'][draw])
                    error=float(abs(actual-old).max());errors.append(error)
                    if error>1e-4:
                        raise AssertionError('Corrected encoding does not replay original five-draw inference')
                    predicted=np.where(actual>=.5,0,1)
                    if not np.array_equal(predicted,saved['frozen']['test_pred'][draw]):
                        raise AssertionError('Corrected true condition changed original five-draw labels')
                    if abs(r.metrics(saved['oof']['labels'],predicted,actual)['AUC']-saved['frozen']['draw_metrics'][draw]['AUC'])>1e-12:
                        raise AssertionError('Corrected true condition changed original five-draw AUC')
                    if group=='R2':
                        for name in ('shuffled','reference'):
                            r.compare(torch.as_tensor(counter['true'][-1]),torch.as_tensor(counter[name][-1]),'C1 constant condition invariance')
                    else:
                        off=counter['off'][-1][test]
                        off_errors.append(float(abs(off-np.asarray(base_reference['test_prob_asd'][draw])).max()))
                        if off_errors[-1]>1e-4:
                            raise AssertionError('Frozen R3 off does not replay historical C0')
            if {k:r.array_hash(v) for k,v in loaded.core.clf[2].state_dict().items()}!=bn_before:
                raise AssertionError('Counterfactual repair changed BN')
            saved['counterfactual_audit_v2']={'validated':True,'counterfactual_prob_asd':{k:np.stack(v) for k,v in counter.items()},
                                            'same_original_joint_for_all_conditions':True,'manual_classifier_without_hooks':True,
                                            'original_frozen_signature':before_frozen,'formal_signature_before':before_formal,
                                            'true_replay_max_prob_error_by_draw':errors,
                                            'off_C0_max_prob_error_by_draw':off_errors,
                                            'manual_true_max_logit_error_by_draw':manual_errors,
                                            'BN_unchanged':True,'source_sha256':r.sha_file(r.BASE/'analyze.py'),
                                            'supersedes':'Only original frozen.counterfactual_prob_asd/base_test_prob_asd are invalid; original frozen object is preserved for provenance',
                                            'reason':'Disabled pre-hook still updated bridge.base; prior loop reused the mutable attribute rather than a fixed local base',
                                            'completed_utc':r.utc()}
            if formal_signature(saved)!=before_formal or frozen_signature(saved['frozen'])!=before_frozen:
                raise AssertionError('Auxiliary repair changed protected original results')
            r.atomic_torch(saved,path)
            verified_counterfactual(r.load_torch(path,mmap=True))
            results[group]={'validated':True,'max_true_replay_error':max(errors),'max_manual_logit_error':max(manual_errors),
                            'max_off_C0_error':max(off_errors,default=0),'formal_and_original_frozen_unchanged':True}
        return results
    finally:
        if loaded is not None:
            loaded.close()


def group_summary(group,state):
    rows,frozen_rows,epochs,seconds,changes,instability=[],[],[],[],[],[]
    draw_rows=[[] for _ in range(5)]
    for seed in r.config()['seeds']:
        coverage=np.zeros(871,dtype=int)
        for fold in range(10):
            if group.startswith('R'):
                if not r.verify_result(group,seed,fold,state,complete=True):
                    raise AssertionError('Missing final result')
                saved=r.load_torch(r.result_path(group,seed,fold),mmap=True)
            else:
                saved=old_fold(group,seed,fold,state)
            oof=saved['oof'];idx=np.asarray(oof['indices']);coverage[idx]+=1
            m=independent_metrics(oof['labels'],oof['pred'],oof['prob_asd'])
            if any(abs(m[k]-oof['metrics'][k])>1e-12 for k in r.KEYS):
                raise AssertionError('Independent sklearn formal recomputation failed')
            rows.append({'seed':seed,'fold':fold,'metrics':m,'best_epoch':saved['best_epoch']})
            frozen=saved['frozen'];prob=np.asarray(frozen['test_prob_asd']);pred=np.asarray(frozen['test_pred'])
            for d in range(5):
                m=independent_metrics(oof['labels'],pred[d],prob[d])
                if any(abs(m[k]-frozen['draw_metrics'][d][k])>1e-12 for k in r.KEYS):
                    raise AssertionError('Independent frozen recomputation failed')
                draw_rows[d].append({'seed':seed,'fold':fold,'metrics':m})
            mean=prob.mean(0)
            frozen_rows.append({'seed':seed,'fold':fold,'metrics':independent_metrics(oof['labels'],np.where(mean>=.5,0,1),mean)})
            instability.append(float(np.any(pred!=pred[0],axis=0).mean()))
            epochs.append({'seed':seed,'fold':fold,'best_epoch':saved['best_epoch'],'epochs_run':saved['epochs_run']})
            seconds.append(saved.get('classifier_seconds',0))
            if 'parameter_changes' in saved: changes.append(saved['parameter_changes'])
        if not np.all(coverage==1):
            raise AssertionError('OOF coverage is not exactly once per subject per seed')
    return {'formal':packed(rows),'formal_folds':rows,'frozen_probability_mean':packed(frozen_rows),
            'frozen_folds':frozen_rows,'draws':[packed(v) for v in draw_rows],
            'best_epoch_mean':float(np.mean([v['best_epoch'] for v in epochs])),
            'epochs_run_mean':float(np.mean([v['epochs_run'] for v in epochs])),
            'epochs':epochs,'classifier_seconds_total':float(sum(seconds)),
            'test_label_instability_mean':float(np.mean(instability)),
            'parameter_changes':changes,'r3_two_stage_diagnostic':group=='R3'}


def pair(a,b,state):
    answer={}
    for field in ('formal_folds','frozen_folds'):
        reference={(v['seed'],v['fold']):v for v in state['groups'][b][field]}
        rows=[{'seed':v['seed'],'fold':v['fold'],'metrics':{k:v['metrics'][k]-reference[v['seed'],v['fold']]['metrics'][k] for k in r.KEYS}}
              for v in state['groups'][a][field]]
        answer[field]={'aggregate':packed(rows),'paired_folds':rows,
                       'positive_seeds':{k:sum(v[k]>0 for v in packed(rows)['seed_means'].values()) for k in r.KEYS},
                       'positive_folds':{k:sum(v['metrics'][k]>1e-12 for v in rows) for k in r.KEYS}}
    answer['note']='Same 871 subjects in repeated CV; folds/seeds are correlated. Descriptive paired contrasts, not 30 independent cohorts.'
    if 'R3' in (a,b): answer['protocol_note']='R3 is a two-stage frozen adapter diagnostic; different training/validation selection protocol.'
    return answer


def ce(prob,labels):
    prob=np.clip(np.asarray(prob,dtype=np.float64),1e-7,1-1e-7)
    return -np.where(np.asarray(labels)==0,np.log(prob),np.log1p(-prob))


def conditional_audit(group,state):
    ctx=r.load_torch(r.OLD/'artifacts/context.pt');meta=r.metadata(ctx)
    records=[];per_seed={};strata={}
    for seed in r.config()['seeds']:
        fields={name:np.zeros((5,871)) for name in r.config()['counterfactuals']}
        for fold in range(10):
            saved=(r.load_torch(r.result_path(group,seed,fold),mmap=True) if group=='R3' else old_fold(group,seed,fold,state))
            test=np.asarray(saved['oof']['indices'])
            for name in fields:
                fields[name][:,test]=np.asarray(verified_counterfactual(saved)['counterfactual_prob_asd'][name])[:,test]
            if group=='C4':
                fields['reference'][:,test]=np.asarray(state['diagnostics'][f'{seed}/{fold}']['C4']['age_sex_reference_keep_site_test_prob_asd'])
        y=np.asarray(ctx['labels'])
        true_ce=ce(fields['true'],y[None,:])
        for other in ('shuffled','reference','off'):
            delta=ce(fields[other],y[None,:])-true_ce
            probability_mad=np.abs(fields['true']-fields[other])
            mean_true,mean_other=fields['true'].mean(0),fields[other].mean(0)
            actual=r.metrics(y,np.where(mean_true>=.5,0,1),mean_true)
            alternative=r.metrics(y,np.where(mean_other>=.5,0,1),mean_other)
            corr=float(spearmanr(mean_true,mean_other).statistic)
            per_seed.setdefault(str(seed),{})[other]={
                'CE_gain_true':float(delta.mean()),'probability_MAD':float(probability_mad.mean()),
                'benefited_fraction':float((delta.mean(0)>0).mean()),
                'stable_benefit_4of5_fraction':float(((delta>0).sum(0)>=4).mean()),
                'probability_rank_spearman':corr,
                'probability_mean_label_change_fraction':float(((mean_true>=.5)!=(mean_other>=.5)).mean()),
                'metric_delta_true_minus_other':{k:actual[k]-alternative[k] for k in r.KEYS}}
            names={'all':np.ones(871,dtype=bool),**{f'sex_{s}':meta['sex']==s for s in ('1','2')},
                   **{f'age_{lo}_{hi}':(meta['age']>=lo)&(meta['age']<hi) for lo,hi in zip(r.config()['age_bins'][:-1],r.config()['age_bins'][1:])},
                   **{f'site_{s}':meta['site']==s for s in sorted(set(meta['site']))}}
            for name,mask in names.items():
                s={'seed':seed,'n':int(mask.sum()),'CE_gain_true':float(delta[:,mask].mean()),
                   'probability_MAD':float(probability_mad[:,mask].mean()),
                   'benefited_fraction':float((delta[:,mask].mean(0)>0).mean())}
                if len(np.unique(y[mask]))==2:
                    a=r.metrics(y[mask],np.where(mean_true[mask]>=.5,0,1),mean_true[mask])
                    b=r.metrics(y[mask],np.where(mean_other[mask]>=.5,0,1),mean_other[mask])
                    s['metric_delta']={k:a[k]-b[k] for k in r.KEYS}
                strata.setdefault(other,{}).setdefault(name,[]).append(s)
    result={'per_seed':per_seed,'strata':strata,'scope':'same checkpoint, same stochastic encoder realization, fixed eval BN; test OOF only',
            'metric_unit':'Each seed pooled 871-subject OOF after five-probability mean; not the fold-mean main-table estimator',
            'C4_reference_holds_site_fixed':group=='C4'}
    result['mean']={other:{k:float(np.mean([v[other][k] for v in per_seed.values()]))
                          for k in ('CE_gain_true','probability_MAD','benefited_fraction','stable_benefit_4of5_fraction','probability_rank_spearman','probability_mean_label_change_fraction')}
                    for other in ('shuffled','reference','off')}
    result['seed_sd_ddof1']={other:{k:float(np.std([v[other][k] for v in per_seed.values()],ddof=1))
                                   for k in result['mean'][other]}
                            for other in ('shuffled','reference','off')}
    result['mean_metric_delta_true_minus_other']={other:{k:float(np.mean([v[other]['metric_delta_true_minus_other'][k] for v in per_seed.values()]))
                                                       for k in r.KEYS}
                                                for other in ('shuffled','reference','off')}
    return result


def optimization_audit(state):
    answer={}
    unique_core_parameter_names=set(r.load_torch(r.result_path('R0',r.config()['seeds'][0],0),mmap=True)['optimizer_core_parameter_order'])
    for group in ('C0','C1','C3','R0','R1','R2','R3'):
        fits=[]
        for seed in r.config()['seeds']:
            for fold in range(10):
                if group=='C0':
                    folder=r.OLD/f'outputs/E0/seed_{seed}/fold_{fold}'
                    saved=json.loads((folder/'metrics.json').read_text())
                    history=[json.loads(line) for line in (folder/'epochs.jsonl').read_text().splitlines() if line]
                    best=history[saved['best_epoch']]
                    history=[{'epoch':v['epoch'],'train_ce':v['train_ce'],'train_graph':v['train_graph'],
                              'val_ce':v['validation_ce'],'val_graph':v['validation_graph'],'val_loss':v['validation_original_loss']} for v in history]
                    best=history[saved['best_epoch']]
                    resolution='every epoch'
                    changes=r.parameter_changes(r.load_torch(folder/'best.pt')['core']['state'],r.load_torch(r.anchor_dir(seed,fold)/'train_start.pt')['core']['state'])
                else:
                    saved=(old_fold(group,seed,fold,state) if group.startswith('C') else r.load_torch(r.result_path(group,seed,fold),mmap=True))
                    history=saved['sampled_epochs'] if group.startswith('C') else saved['epochs']
                    best=saved['best_training_record']
                    resolution='every 25 epochs plus actual best record' if group.startswith('C') else 'every epoch'
                    changes=saved.get('parameter_changes')
                    if changes is None:
                        changes=r.parameter_changes(saved['core']['state'],r.load_torch(r.anchor_dir(seed,fold)/'train_start.pt')['core']['state'])
                if group=='C0':
                    best_core=r.load_torch(folder/'best.pt')['core']['state']
                else:
                    best_core=saved['core']['state']
                start_core=(r.load_torch(r.ROOT/state['audit']['baseline'][f'{seed}/{fold}']['best_reference'])['core']['state']
                            if group=='R3' else r.load_torch(r.anchor_dir(seed,fold)/'train_start.pt')['core']['state'])
                weights_only=lambda x:{k:v for k,v in x.items() if k in unique_core_parameter_names}
                actual_parameter_changes=r.parameter_changes(weights_only(best_core),weights_only(start_core))
                adapter_changes=None
                if group not in ('C0','R0'):
                    initial_adapter=PCCI.isolated(r.config(),private_seed(seed,fold,'PCCI-initialization'),'cpu').state_dict()
                    final_adapter=saved['pcci']
                    adapter_changes={'total_l2_change':sum(float((final_adapter[k]-initial_adapter[k]).double().square().sum()) for k in initial_adapter)**.5,
                                     'output_weight_l2':float(final_adapter['output.weight'].norm()),
                                     'output_bias_l2':float(final_adapter['output.bias'].norm()),
                                     'initial_sha256':r.state_hash(initial_adapter),'final_sha256':r.state_hash(final_adapter)}
                fits.append({'seed':seed,'fold':fold,'best_epoch':saved['best_epoch'],'epochs_run':saved['epochs_run'],
                             'best':{k:best[k] for k in ('train_ce','train_graph','val_ce','val_graph','val_loss')},
                             'best_train_val_CE_gap':best['val_ce']-best['train_ce'],
                             'history':[{k:v[k] for k in ('epoch','train_ce','train_graph','val_ce','val_graph','val_loss')} for v in history],
                             'observed_val_CE_min_epoch':min(history,key=lambda v:v['val_ce'])['epoch'],
                             'observed_val_Graph_min_epoch':min(history,key=lambda v:v['val_graph'])['epoch'],
                             'observed_val_CE_sd':float(np.std([v['val_ce'] for v in history])),
                             'observed_val_Graph_sd':float(np.std([v['val_graph'] for v in history])),
                             'resolution':resolution,'parameter_and_float_buffer_changes':changes,
                             'parameter_changes_without_BN_buffers':actual_parameter_changes,'adapter_changes':adapter_changes})
        answer[group]={'fits':fits,'mean':{k:float(np.mean([v['best'][k] for v in fits])) for k in ('train_ce','train_graph','val_ce','val_graph','val_loss')},
                       'mean_CE_gap':float(np.mean([v['best_train_val_CE_gap'] for v in fits])),
                       'best_epoch_mean':float(np.mean([v['best_epoch'] for v in fits]))}
    return answer


def matched_optimization(state):
    comparisons={}
    for a,b in (('C1','C0'),('C3','C0'),('R0','C0'),('R1','R0'),('R2','C1')):
        fits=[]
        reference={(v['seed'],v['fold']):v for v in state['optimization'][b]['fits']}
        for v in state['optimization'][a]['fits']:
            other=reference[v['seed'],v['fold']]
            aa={row['epoch']:row for row in v['history']}
            bb={row['epoch']:row for row in other['history']}
            shared=sorted(set(aa)&set(bb))
            diffs=[{'epoch':epoch,**{k:aa[epoch][k]-bb[epoch][k]
                                   for k in ('train_ce','train_graph','val_ce','val_graph','val_loss')}} for epoch in shared]
            fits.append({'seed':v['seed'],'fold':v['fold'],'matched_epoch_differences':diffs})
        snapshots={}
        for epoch in (0,25,50,75,100,125,150,200):
            rows=[row for fit in fits for row in fit['matched_epoch_differences'] if row['epoch']==epoch]
            if rows:
                snapshots[str(epoch)]={'paired_folds_available':len(rows),
                                      **{k:float(np.mean([row[k] for row in rows]))
                                         for k in ('train_ce','train_graph','val_ce','val_graph','val_loss')}}
        comparisons[f'{a}-{b}']={'fits':fits,'snapshots':snapshots}
    return {'comparisons':comparisons,
            'note':'Only actually recorded matching epochs; no interpolation. Later epochs exclude already-stopped folds, so survivor counts are explicit.'}


def replay_audit(state):
    records=[]
    for seed in r.config()['seeds']:
        for fold in range(10):
            a=r.load_torch(r.result_path('R0',seed,fold),mmap=True)
            b=r.load_torch(r.result_path('R1',seed,fold),mmap=True)
            common=min(len(a['epochs']),len(b['epochs']))
            records.append({'seed':seed,'fold':fold,'common_epochs':common,
                            'all_common_RNG_equal':all(x['rng_after_val']==y['rng_after_val'] for x,y in zip(a['epochs'],b['epochs'])),
                            'all_common_BN_counts_equal':all(x['bn_batches']==y['bn_batches'] for x,y in zip(a['epochs'],b['epochs'])),
                            'max_common_val_loss_difference':max(abs(x['val_loss']-y['val_loss']) for x,y in zip(a['epochs'],b['epochs'])),
                            'best_core_exact_equal':a['best_core_sha256']==b['best_core_sha256'],
                            'best_epoch_equal':a['best_epoch']==b['best_epoch'],
                            'R1_adapter_unchanged':b['r1_module_unchanged'],
                            'first_three_max_scalar_loss_abs_diff':max(abs(x[k]-y[k]) for x,y in zip(a['epochs'][:3],b['epochs'][:3]) for k in ('train_ce','train_graph','val_ce','val_graph','val_loss')),
                            'first_three_gradient_norm_abs_diff':{k:max(abs(x['early_trace']['gradient_norms'][k]-y['early_trace']['gradient_norms'][k]) for x,y in zip(a['epochs'][:3],b['epochs'][:3])) for k in ('imaging','phenotype','classifier')},
                            'first_three_gradient_sha_equal':[x['early_trace']['core_gradients_sha256']==y['early_trace']['core_gradients_sha256'] for x,y in zip(a['epochs'][:3],b['epochs'][:3])]})
    return {'folds':records,'all_RNG_trajectories_equal':all(v['all_common_RNG_equal'] for v in records),
            'all_BN_counts_equal':all(v['all_common_BN_counts_equal'] for v in records),
            'best_exact_equal_folds':sum(v['best_core_exact_equal'] for v in records),
            'native_backend_long_trajectory_not_assumed_deterministic':True}


def num(v,scale=100):
    return f'{v*scale:.2f}'


def execution_audit(state):
    events=[json.loads(line) for line in (r.BASE/'run.log').read_text().splitlines() if line]
    records={}
    snapshots=[event['resources'] for event in events if event.get('event')=='queue_status' and event.get('phase')=='train']
    for gpu in (0,1):
        starts=[event['time_utc'] for event in events if event.get('event')=='started' and event.get('action')=='train' and event.get('gpu')==gpu]
        ends=[event['time_utc'] for event in events if event.get('event')=='complete' and event.get('action')=='train' and event.get('gpu')==gpu]
        fits=[r.load_torch(r.result_path(group,seed,fold),mmap=True)
              for group in r.GROUPS for seed in r.config()['seeds'] for fold in range(10)
              if state['tasks'][f'train/{group}/{seed}/{fold}']['gpu']==gpu]
        memories=[snapshot['devices'][str(gpu)]['used_mib'] for snapshot in snapshots]
        utilizations=[snapshot['devices'][str(gpu)]['utilization_percent'] for snapshot in snapshots]
        records[str(gpu)]={'formal_tasks':len(fits),'first_started_utc':min(starts),'last_completed_utc':max(ends),
                           'wall_active_span_seconds':(datetime.fromisoformat(max(ends))-datetime.fromisoformat(min(starts))).total_seconds(),
                           'sum_classifier_training_seconds':float(sum(v['classifier_seconds'] for v in fits)),
                           'sum_worker_elapsed_seconds':float(sum(v['elapsed_seconds'] for k,v in state['tasks'].items() if k.startswith('train/') and v['gpu']==gpu)),
                           'sampled_peak_device_memory_mib':max(memories,default=0),
                           'sampled_utilization_mean_percent':float(np.mean(utilizations)) if utilizations else None}
    return {'per_gpu':records,'training_resource_samples':len(snapshots),
            'sampled_max_cpu_cores':max((v['cgroup_cpu_cores'] for v in snapshots),default=0),
            'sampled_min_ram_available_mib':min((v['ram_available_mib'] for v in snapshots),default=0),
            'sampling_note':'30-second queue snapshots; device memory/utilization include all processes, not a continuous peak measurement'}


def write_report(state):
    groups=state['groups'];pairs=state['comparisons']
    lines=['# Experiment 4：PCCI 性能退化归因与条件互补审计','',f"分析时间 UTC：{r.utc()}",'',
           '## 正式结果','',
           '每seed先平均10折，再报告3个seed的均值与样本标准差。ASD=0为阳性，AUC使用ASD概率。历史RFE跨折泄漏保留，仅用于源码行为控制归因，不能解释为无泄漏临床泛化。',
           '', '| Group | ACC | BA | AUC | SEN | SPE | F1 | Best Epoch | ΔACC vs C0 |',
           '|---|---|---|---|---|---|---|---|---|']
    names={'C0':'Historical C0','C1':'Historical C1','C3':'Historical C3','R0':'R0 Baseline Replay','R1':'R1 Inert Bridge','R2':'R2 Detached Residual','R3':'R3 Frozen Adapter [二阶段诊断]'}
    for group in names:
        g=groups[group];p=g['formal'];values=[f"{num(p['mean'][k])} +/- {num(p['seed_sd_ddof1'][k])}" for k in r.KEYS]
        values=[values[0],values[1],values[2],values[3],values[4],values[5]]
        lines.append('| '+' | '.join([names[group],*values,f"{g['best_epoch_mean']:.2f}",num(p['mean']['ACC']-groups['C0']['formal']['mean']['ACC'])])+' |')
    lines.extend(['','R3在已由验证集选择的C0核心上，再用同一验证集选择Adapter，是二阶段诊断；列于此仅便于归因，不参与同协议从头训练论文性能排名。','',
                  '## 配对比较','', '| Contrast | ΔACC pp | ΔBA pp | ΔAUC pp | ACC正向seed | ACC改善fold |','|---|---|---|---|---|---|'])
    for key,p in pairs.items():
        formal=p['formal_folds'];mean=formal['aggregate']['mean']
        lines.append(f"| {key} | {num(mean['ACC'])} | {num(mean['BA'])} | {num(mean['AUC'])} | {formal['positive_seeds']['ACC']}/3 | {formal['positive_folds']['ACC']}/30 |")
    lines.extend(['','这些是相关seed-fold配对描述，30折不是30个独立临床样本/外部队列。仅3个重复seed，不用伪独立fold检验制造显著性。','',
                  '## 五次固定RNG冻结评估','', '| Group | 概率均值ACC | BA | AUC | 单draw ACC范围 | 单draw AUC范围 | 患者标签变化比例 |','|---|---|---|---|---|---|---|'])
    for group in names:
        g=groups[group];p=g['frozen_probability_mean']['mean']
        ar=[x['mean']['ACC'] for x in g['draws']];au=[x['mean']['AUC'] for x in g['draws']]
        lines.append(f"| {group} | {num(p['ACC'])} | {num(p['BA'])} | {num(p['AUC'])} | {num(min(ar))}/{num(max(ar))} | {num(min(au))}/{num(max(au))} | {num(g['test_label_instability_mean'])} |")
    lines.extend(['','五次RNG严格复用实验3定义，不挑最好draw，不替代原始单次正式结果。','',
                  '## 数值与梯度验证','',f"30折三步确定性对照全部通过；历史E0参照、R0、R1逐步验证logits、CE/Graph、验证损失、核心梯度、Adam状态、BN、TopK集合与完整RNG。",
                  f"全程原生后端R0/R1：共同epoch RNG相同={state['replay_audit']['all_RNG_trajectories_equal']}，BN计数相同={state['replay_audit']['all_BN_counts_equal']}，最佳核心逐位相同={state['replay_audit']['best_exact_equal_folds']}/30。不能用三步确定性等价替代完整原生训练的比较。",
                  'R2 autograd确认新增残差对影像GTUNet的直接梯度为零，原融合影像/表型、Classifier与PCCI梯度仍存在。detach仍通过改变Classifier输入和CE间接影响后续主干优化，不能排除所有联合适应。',
                  'R3每次forward前后比较全部core参数与Buffer；仅允许恢复RP约束卷积的原位投影，其他漂移立即失败；BN固定eval，PCCI正常更新。初始零残差在同RNG下对照C0原checkpoint预测。完整证据见summary.checks和单fold frozen_guard。','',
                  '## D1/D2：尺度与Classifier/BN','', '| Group | Imaging L2 | Phenotype L2 | Joint L2 | Delta L2 | Delta/Joint | BN增益绝对值 | Logits变化L2 | 概率绝对变化 |','|---|---|---|---|---|---|---|---|---|'])
    for group in names:
        ds=[v[group] for v in state['diagnostics'].values()]
        vals=[np.mean([d[k]['mean'] for d in ds]) for k in ('imaging_l2','phenotype_l2','joint_l2','delta_l2','delta_joint_ratio','BN_gain_abs','actual_logit_change_l2')]
        mad=np.mean([d['true_off_probability_MAD'] for d in ds])
        lines.append('| '+group+' | '+' | '.join(f'{x:.6f}' for x in [*vals,mad])+' |')
    if state.get('numerical_replay_calibration'):
        lines.extend(['', '数值重放补充：seed912/fold6曾在2e-5绝对概率阈值下触发两次诊断失败，不是训练或五次正式冻结评估失败。追加16次同RNG原生/确定性前向后，最大偏差4.4465e-5，标签及AUC均不变。仅对此类诊断采用2e-5主阈值与1e-4上界，并要求超主阈值时标签和AUC严格一致；未改训练、Checkpoint或正式预测。完整校准、旧/新诊断源码指纹及恢复记录见summary.numerical_replay_calibration。不得将这些概率数值差异称为逐位等价。'])
    lines.extend(['','D1/D2对全部330个历史及新checkpoint各用预定draw0；同一次原编码的影像/joint上只改变条件残差。线性、ReLU、固定BN、最终logits逐层保存差异，并测局部JVP及ReLU激活跨界。残差绝对范数必须结合joint相对尺度、分类器权重与BN running_var解释；小范数不意味着小概率影响。','',
                  '| Group | ΔLinear1 L2 | ΔReLU L2 | ΔBN L2 | ΔLogits L2 | ReLU跨界比例 | JVP相对误差 |','|---|---|---|---|---|---|---|'])
    for group in names:
        ds=[v[group] for v in state['diagnostics'].values()]
        differences=[np.mean([d['cases']['true']['difference_vs_off_l2'][k]['mean'] for d in ds]) for k in ('linear1','relu','bn','logits')]
        switch=np.mean([d['cases']['true']['relu_active_change_fraction'] for d in ds])
        error=np.mean([d['JVP_relative_error'] for d in ds])
        lines.append('| '+group+' | '+' | '.join(f'{x:.6f}' for x in [*differences,switch,error])+' |')
    lines.extend(['','JVP是给定输入处沿delta的局部导数；完整delta跨越ReLU边界时不要求与有限变化完全相等。没有重新训练BN，也没有把其敏感性误称为BN单因素致退化。','',
                  '| Group | 直接残差影像输出梯度L2 | 原融合+图影像梯度L2 | 直接/原路径 | 梯度cos | Classifier CE梯度L2 |','|---|---|---|---|---|---|'])
    for group in names:
        gradients=[v[group]['local_gradient_paths'] for v in state['diagnostics'].values()]
        cols=[np.mean([v[k] for v in gradients]) for k in ('direct_residual_imaging_output_gradient_l2','native_fusion_plus_graph_imaging_output_gradient_l2','direct_to_native_ratio')]
        cosines=[v['direct_vs_native_cosine'] for v in gradients if v['direct_vs_native_cosine'] is not None]
        cosine=f'{np.mean(cosines):.6f}' if cosines else 'NA'
        classifier=np.mean([v['classifier_CE_gradient_l2'] for v in gradients])
        lines.append('| '+group+' | '+' | '.join(f'{x:.6f}' for x in cols)+f' | {cosine} | {classifier:.6f} |')
    lines.extend(['','这些是冻结checkpoint表示处、eval BN下的局部敏感性，不是历史训练时影像编码器参数梯度。R3列的Classifier梯度仅用于事后响应测量，其正式训练Classifier严格冻结；实际训练参数梯度见各fold记录及R2校验。','',
                  '## D3：优化与早停','', '| Group | Best Epoch | Train CE | Train Graph | Val CE | Val Graph | Val Total | CE gap |','|---|---|---|---|---|---|---|---|'])
    for group,o in state['optimization'].items():
        m=o['mean'];lines.append('| '+group+' | '+f"{o['best_epoch_mean']:.2f}"+' | '+' | '.join(f'{m[k]:.6f}' for k in ('train_ce','train_graph','val_ce','val_graph','val_loss'))+f" | {o['mean_CE_gap']:.6f} |")
    lines.extend(['','| Group | Imaging参数相对变化 | Phenotype参数相对变化 | Attention参数相对变化 | Classifier参数相对变化 | PCCI参数变化L2 |','|---|---|---|---|---|---|'])
    for group,o in state['optimization'].items():
        changes=[np.mean([fit['parameter_changes_without_BN_buffers'][prefix]['relative_l2_change'] for fit in o['fits']]) for prefix in ('img_unet','ph_unet','mm_attention','clf')]
        adapter=np.mean([fit['adapter_changes']['total_l2_change'] for fit in o['fits'] if fit['adapter_changes'] is not None]) if group not in ('C0','R0') else 0
        lines.append('| '+group+' | '+' | '.join(f'{x:.6f}' for x in [*changes,adapter])+' |')
    lines.extend(['','参数变化按原core.named_parameters唯一名称统计，排除BN running Buffer及state_dict中的共享模块重复别名；不同best epoch的参数变化并非等训练步数因果对照。历史C0逐epoch可读，历史C1/C3仅每25epoch抽样及实际best记录，新组逐epoch。summary保留真实时间点，不做插值。最佳时刻提前本身不能证明加速、过拟合或训练器缺陷；必须结合CE、Graph和参数变化。','',
                  '### 同epoch优化差异','', '| Contrast | Epoch | 配对fold数 | ΔTrain CE | ΔTrain Graph | ΔVal CE | ΔVal Graph | ΔVal Total |','|---|---|---|---|---|---|---|---|'])
    for key,comparison in state['matched_optimization']['comparisons'].items():
        for epoch in ('50','75','100'):
            if epoch not in comparison['snapshots']: continue
            row=comparison['snapshots'][epoch]
            lines.append('| '+key+' | '+epoch+f" | {row['paired_folds_available']} | "+' | '.join(f'{row[k]:.6f}' for k in ('train_ce','train_graph','val_ce','val_graph','val_loss'))+' |')
    lines.extend(['','仅比较实际共同记录的epoch；后期幸存fold数可能减少，不把这一选择效应当作全体fold的轨迹。R3冻结状态下Graph项对Adapter无梯度，但随机前向仍使其影响验证选模，本轮按规定保留；不能由早期best单独推断Adapter训练失败。','',
                  '## D4：真实表型增量','', '| Group | 真-错配 CE收益 | 真-错配概率MAD | 真-参考 CE收益 | 真-off CE收益 | 真比错配受益患者 |','|---|---|---|---|---|---|'])
    for group,d in state['conditional'].items():
        m=d['mean'];lines.append(f"| {group} | {m['shuffled']['CE_gain_true']:.7f} | {m['shuffled']['probability_MAD']:.7f} | {m['reference']['CE_gain_true']:.7f} | {m['off']['CE_gain_true']:.7f} | {num(m['shuffled']['benefited_fraction'])} |")
    lines.extend(['','| Group | 真-错配ΔACC pp | ΔBA pp | ΔAUC pp | 预测排序Spearman | 均值概率标签变化比例 |','|---|---|---|---|---|---|'])
    for group,d in state['conditional'].items():
        m=d['mean_metric_delta_true_minus_other']['shuffled'];v=d['mean']['shuffled']
        lines.append(f"| {group} | {num(m['ACC'])} | {num(m['BA'])} | {num(m['AUC'])} | {v['probability_rank_spearman']:.7f} | {num(v['probability_mean_label_change_fraction'])} |")
    lines.extend(['','正CE收益为CE(other)-CE(true)，先对每次随机前向计算CE再平均。上表反事实指标先在每seed汇集871名OOF、平均五次概率后计算，再平均seed；与主表逐折均值估计口径不同，不直接相减。同站点联合置换年龄/性别，表型参考由训练集定义。C4历史reference同时改变site，本轮额外计算保留site、仅替换年龄/性别的reference，避免把site干预混作年龄/性别证据。每seed的ACC/BA/AUC差、排序相关、年龄/性别/站点分层与稳定受益比例见summary.conditional。off仅是当前checkpoint关闭残差，不等于独立训练C0。','',
                  '辅助实现问题及处理：复核发现新运行器的旧反事实循环在禁用Hook时仍通过其更新bridge.base，后续条件累加残差，造成reference与off异常相同。历史PCCI评估使用局部base，不存在该问题；不能用它解释历史性能退化。本轮保留120折正式结果和600次原冻结主评估全部原始字段，在R2/R3每fold同一个pt内追加counterfactual_audit_v2。最终通过验证的重建含300次相同RNG编码，使用固定局部base与不触发Hook的eval分类器，验证R2常量条件不变、R3 off重现C0、BN不更新。原frozen内R2/R3的旧counterfactual_prob_asd/base_test_prob_asd标为无效，仅为溯源保留；D4与结论只用v2。修复不重新训练、不修改Checkpoint、不替换正式/五次主预测。早期27个修复尝试因BN比较锚点取于加载最佳Checkpoint之前而被拒绝，未写入结果；修正检查锚点后全部通过，不代表真实BN漂移。另2次失败属于上述有界数值重放，共29次诊断/修复失败均保留在run.log；正式训练失败和重训均为0。','',
                  '## 六个问题与研究决策','',*state['research_answers'],'',
                  '## 执行与保护','',f"120个正式训练任务、600次新checkpoint固定RNG评估、330个checkpoint逐层诊断；C0/C1-C6不重训。正式任务每卡完成数：{state['execution']['training_by_gpu']}；各组分卡：{state['execution']['groups_by_gpu']}。",
                  f"正式训练队列耗时 {state['scheduling']['phases']['train']['elapsed_seconds']:.2f} 秒，持久审计注册后的总耗时 {state['execution']['elapsed_seconds']:.2f} 秒；计时起点{state['created_utc']}，不包含此前交互式论文浏览和初始编码。并发峰值 {state['scheduling'].get('peak_workers')}。跨卡调度 {len(state['scheduling'].get('cross_gpu_assignments',[]))} 条，均有记录。",
                  f"每卡活动跨度、累计训练秒数、资源采样统计见summary.execution.resources。训练期CPU采样最大 {state['execution']['resources']['sampled_max_cpu_cores']:.2f} 核；资源监测未替代每fold自身的显存峰值记录。",
                  f"已完成{len(state.get('counterfactual_repairs',{}))}/30个辅助反事实修复任务，最终验证通过的重建包含300次只读编码；失败尝试的编码不混入该计数，亦未把这些编码计入600次主冻结评估。签名和严格不变性证据保存在每fold的counterfactual_audit_v2。",
                  f"只读文件最终复核：{state['readonly_final']}。临时目录为 {r.BASE}；指定迁移目标为 {r.ROOT/'experiments/PCCI_Attribution_20261016'}。未push。",
                  '', '## 论文与源码依据','',
                  '[MM-GTUNets论文](https://arxiv.org/html/2406.14455v3)；[作者代码](https://github.com/NZWANG/MM-GTUNets/tree/bb9351b931003535503694ad2f777b6cee1c151f)；[实验3固定提交](https://github.com/tangaii/MM/tree/e48c177dc64da4a91dfa44b939b61cb823b65d7b/experiments/PCCI_20261015)。原模型已包含表型图与非影像支路；源码特有行为、论文/源码差异及继承RFE泄漏见README和audit。'])
    if state.get('final_verification',{}).get('passed'):
        lines.extend(['','## 独立验收与存储状态','',
                      '120折正式指标与600次冻结指标已使用独立混淆矩阵及Mann-Whitney秩AUC复算，不依赖sklearn指标实现。每组每seed的OOF恰好覆盖871人一次，ASD概率与原logits一致；120折训练权重、正式预测及训练记录在追加辅助评估前后签名完全不变。全部120折正式训练均仅执行一次。',
                      '30折R3核心权重严格等同相应C0最佳Checkpoint；修复反事实的off重放未改变任何C0预测标签，最大概率误差约4.24e-5。所有60个v2辅助产物、完整Checkpoint指纹与最少文件结构均已验收。复查命令：`python -B analyze.py verify`，只读检查，不执行神经网络前向或训练。'])
        if state['execution'].get('final_verification_elapsed_seconds'):
            lines.append(f"最终验收及续跑检查完成时间 {state['final_verification']['completed_utc']}；自持久审计注册起计总墙钟 {state['execution']['final_verification_elapsed_seconds']:.2f} 秒（包含辅助修复及最终独立复核，不含迁移等待与此前浏览/初始编码）。验证续跑未重训、未执行模型前向，120个pt文件SHA256完全未变。")
    if state.get('storage_status',{}).get('state')=='pending':
        lines.extend(['',f"存储尚未恢复：最后检查 {state['storage_status']['last_checked_utc']}，创建迁移目标仍报Input/output error。研究计算与结果复核完成，但迁回原目录及根handoff.md一次性追加尚未完成；不能将整个Prompt标为最终交接完成。所有产物保留在`/tmp/PCCI_Attribution_20261016`，不要清理/tmp。待追加handoff正文保存在summary.pending_handoff.markdown，不另建文件。"])
    (r.BASE/'report.md').write_text('\n'.join(lines)+'\n')


def research_answers(state):
    g=state['groups'];p=state['comparisons']
    delta=lambda key,k='ACC':p[key]['formal_folds']['aggregate']['mean'][k]
    frozen_delta=lambda key,k='ACC':p[key]['frozen_folds']['aggregate']['mean'][k]
    r0=delta('R0-C0');r1=delta('R1-R0');r2=delta('R2-C1');r3=delta('R3-C0')
    c=state['conditional']['R3'];d=c['mean']['shuffled']
    positive=sum(v['shuffled']['CE_gain_true']>0 for v in c['per_seed'].values())
    ref_positive=sum(v['reference']['CE_gain_true']>0 for v in c['per_seed'].values())
    condition_promising=(positive==3 and ref_positive==3 and d['CE_gain_true']>.001
                         and c['mean_metric_delta_true_minus_other']['shuffled']['AUC']>0)
    conditional_seeds=', '.join(f"{seed}: {v['shuffled']['CE_gain_true']:.7f}" for seed,v in c['per_seed'].items())
    baseline_seeds=', '.join(f"{seed}: {num(v['ACC'])} pp" for seed,v in p['R0-C0']['formal_folds']['aggregate']['seed_means'].items())
    diagnostics=[v['C3'] for v in state['diagnostics'].values()]
    ds=lambda key:float(np.mean([v[key]['mean'] for v in diagnostics]))
    layer=lambda key:float(np.mean([v['cases']['true']['difference_vs_off_l2'][key]['mean'] for v in diagnostics]))
    matched=state['matched_optimization']['comparisons']['C1-C0']['snapshots']['75']
    native_loss_error=max(v['first_three_max_scalar_loss_abs_diff'] for v in state['replay_audit']['folds'])
    epochs=lambda group:g[group]['best_epoch_mean']
    trainer=('R0存在系统性下降，应优先调查训练器/运行环境；当前不能把历史下降全部归于残差结构。' if r0<-.015 else
             'R0总体接近历史C0，未重现历史新增分支约4个百分点的下降；不是统计等价性证明，也不能证明所有环境影响为零。')
    return [f"Q1 新训练器：R0-C0单次ACC {num(r0)} pp、AUC {num(delta('R0-C0','AUC'))} pp；固定五次概率均值ACC差 {num(frozen_delta('R0-C0'))} pp。逐seed单次ACC差为 {baseline_seeds}。{trainer}",'',
            f"Q2 闲置挂载：R1-R0单次ACC {num(r1)} pp、AUC {num(delta('R1-R0','AUC'))} pp，五次均值ACC差 {num(frozen_delta('R1-R0'))} pp。R1全程delta=0且闲置模块不更新；30折共同epoch完整RNG和BN计数一致。确定性三步完全相等；原生训练前三步标量损失最大差 {native_loss_error:.8g}，长程最佳核心并非逐位相同。"+('该控制未显示闲置Hook/参数列表是主要退化来源，不能把小幅原生差异说成绝对无影响。' if abs(r1)<.01 else '该差异需要继续作为实现/原生数值轨迹问题调查。'),'',
            f"Q3 新增直接反向通路：R2-C1单次ACC {num(r2)} pp、AUC {num(delta('R2-C1','AUC'))} pp，五次均值ACC差 {num(frozen_delta('R2-C1'))} pp；R2-R0单次ACC仍为 {num(delta('R2-R0'))} pp。autograd确认直接路径为零、原融合影像路径与PCCI/Classifier仍有梯度。"+('结果支持直接梯度可能有负面作用，但尚不能认定唯一原因。' if r2>.01 else '切断该路径没有恢复基线，因此不能把这条直接梯度认定为本轮主要退化原因。')+'detach仍改变Classifier输入、CE及原路径后续梯度，未消除所有联合适应。','',
            f"Q4 Classifier/BN：C3平均delta L2={ds('delta_l2'):.6f}、joint L2={ds('joint_l2'):.6f}，逐患者delta/joint比值均值={ds('delta_joint_ratio'):.3f}；绝对值小不等于相对扰动小。固定BN时Linear1/ReLU/BN/logits的变化L2依次为 {layer('linear1'):.5f}/{layer('relu'):.5f}/{layer('bn'):.5f}/{layer('logits'):.5f}，BN绝对增益均值={ds('BN_gain_abs'):.3f}。C3五次真-off概率MAD={state['conditional']['C3']['mean']['off']['probability_MAD']:.7f}。这些直接展示当前Classifier对残差的敏感性，并支持尺度、BN统计与分类器共同适应的解释；不是BN单独导致测试退化的因果消融。关闭残差也不是恢复独立训练C0。",'',
            f"Q5 提前最佳epoch：C0/R0/R1/C1/C3/R2平均best分别为 {epochs('C0'):.2f}/{epochs('R0'):.2f}/{epochs('R1'):.2f}/{epochs('C1'):.2f}/{epochs('C3'):.2f}/{epochs('R2'):.2f}。在实际共同记录的epoch75、{matched['paired_folds_available']}个配对fold上，C1-C0的Train CE/Train Graph/Val CE/Val Graph/Val Total差分别为 {matched['train_ce']:.6f}/{matched['train_graph']:.6f}/{matched['val_ce']:.6f}/{matched['val_graph']:.6f}/{matched['val_loss']:.6f}。"+('分类训练CE更快降低、验证图项更大，说明原CE+Graph优化及选模时刻被改变；不能简化成patience错误或仅分类验证CE变差。' if matched['train_ce']<0 and matched['val_graph']>0 else '分类与图项的实际方向见数值，不能仅从best更早推出训练更快或过拟合。')+'历史C1/C3只有25epoch采样，未插值。未执行CE-only选模或冻结BN单因素训练，因此尚未量化这两个因素各自的因果贡献。','',
            f"Q6 冻结核心：R3-C0单次ACC {num(r3)} pp，五次均值ACC差 {num(frozen_delta('R3-C0'))} pp；R3-C3单次ACC {num(delta('R3-C3'))} pp。原权重及BN状态全程不变，但R3是二阶段同验证集选模诊断，不能直接当作新主方法优势。R3真实对错配概率MAD={d['probability_MAD']:.7f}、CE收益={d['CE_gain_true']:.7f}（3seed样本SD {c['seed_sd_ddof1']['shuffled']['CE_gain_true']:.7f}），逐seed CE收益为 {conditional_seeds}；对参考条件CE收益={c['mean']['reference']['CE_gain_true']:.7f}。真-错配的OOF五次概率均值ACC/BA/AUC差为 {num(c['mean_metric_delta_true_minus_other']['shuffled']['ACC'])}/{num(c['mean_metric_delta_true_minus_other']['shuffled']['BA'])}/{num(c['mean_metric_delta_true_minus_other']['shuffled']['AUC'])} pp。"+('有值得独立验证的条件配对证据，但尚不是可靠互补的独立临床证明。' if condition_promising else '没有足够证据支持稳定且有实际幅度的患者条件对应收益；恢复基线不能自动归功于真实年龄/性别语义。'),'',
            '已直接证明：120折正式结果、数值/梯度连通性、零残差闲置约束、训练期间冻结约束、同模型同表示固定BN的干预响应。多项证据支持的解释：可训练残差改变原融合尺度、Classifier/BN与原图正则的共同优化；单纯挂载及新增直接影像梯度不足以解释全部下降。尚未排除或单独量化：非确定性长轨迹、BN与其他联合适应途径的各自贡献、CE+Graph选模效应、二阶段验证选择以及历史RFE泄漏。未发现可由现有结果支持的“PPU硬件把准确率必然拉低”结论。','',
            ('研究决策：优先修复/解释训练器或运行轨迹差异，不进入更复杂条件融合创新。' if r0<-.015 else
             '研究决策：暂不继续堆叠复杂表型Gate；优先检查独立、可辨识的表型诊断信号或表型介入人口图的阶段，或引入诊断前可获得的更丰富独立临床模态。若继续研究保护主干，应预注册BN/选模单因素对照，并在无泄漏、留站点条件下重新建立基线。' if not condition_promising else
             '研究决策：可以开展保护主干、有限条件调制的下一轮预注册实验；先做无泄漏与留站点验证，不能用本轮测试收益直接包装主方法。')]


def rank_metrics(labels,prediction,probability):
    positive=np.asarray(labels)==0
    predicted=np.asarray(prediction)==0
    probability=np.asarray(probability,dtype=float)
    tp=int(np.count_nonzero(positive&predicted));tn=int(np.count_nonzero(~positive&~predicted))
    fp=int(np.count_nonzero(~positive&predicted));fn=int(np.count_nonzero(positive&~predicted))
    n_positive=tp+fn;n_negative=tn+fp
    ranks=rankdata(probability,method='average')
    auc=(ranks[positive].sum()-n_positive*(n_positive+1)/2)/(n_positive*n_negative)
    return {'ACC':(tp+tn)/len(positive),'BA':.5*(tp/n_positive+tn/n_negative),
            'AUC':float(auc),'SEN':tp/n_positive,'SPE':tn/n_negative,'F1':2*tp/(2*tp+fp+fn)}


def verify_completed_bundle(state):
    """Independent rank-AUC/statistical checks; never constructs or forwards a model."""
    expected_roots={'README.md','config.json','controls.py','run.py','launch.py','analyze.py',
                    'summary.json','report.md','run.log','.gitignore'}
    files={str(path.relative_to(r.BASE)) for path in r.BASE.rglob('*') if path.is_file()}
    checkpoints=set(state['final_checkpoints_sha256'])
    if len(checkpoints)!=120 or files!=expected_roots|checkpoints:
        raise AssertionError('Unexpected or missing experiment files')
    for name,expected in {**state['final_code_sha256'],**state['final_checkpoints_sha256'],
                          'report.md':state['report_sha256']}.items():
        if r.sha_file(r.BASE/name)!=expected:
            raise AssertionError(f'Completed bundle file changed: {name}')
    context=r.load_torch(r.OLD/'artifacts/context.pt')
    labels=np.asarray(context['labels'])
    signatures=state['final_verification']['formal_signatures_before']
    coverage={};fold_metrics={};formal_count=0;draw_count=0;aux_count=0
    for name in sorted(checkpoints):
        saved=r.load_torch(r.BASE/name,mmap=True)
        if saved['state']!='complete' or formal_signature(saved)!=signatures[name]:
            raise AssertionError(f'Formal result changed: {name}')
        if saved['training_code_sha256']!=state['training_code_sha256'] or saved['config_digest']!=r.digest(r.config()):
            raise AssertionError(f'Protocol fingerprint mismatch: {name}')
        if r.core_hash(saved['core'])!=saved['best_core_sha256']:
            raise AssertionError(f'Core weight fingerprint mismatch: {name}')
        if saved['pcci'] is not None and r.state_hash(saved['pcci'])!=saved['best_pcci_sha256']:
            raise AssertionError(f'Adapter weight fingerprint mismatch: {name}')
        oof=saved['oof'];indices=np.asarray(oof['indices']);y=np.asarray(oof['labels'])
        if not np.array_equal(labels[indices],y):
            raise AssertionError(f'OOF subject order mismatch: {name}')
        if oof['ids']!=[context['ids'][i] for i in indices] or oof['sites']!=[context['site_names'][i] for i in indices]:
            raise AssertionError(f'OOF metadata order mismatch: {name}')
        if not np.allclose(softmax(np.asarray(oof['logits']),axis=1)[:,0],oof['prob_asd'],atol=1e-7,rtol=0):
            raise AssertionError(f'Wrong ASD probability: {name}')
        key=(saved['group'],saved['seed'])
        coverage.setdefault(key,[]).extend(indices.tolist())
        metrics=rank_metrics(y,oof['pred'],oof['prob_asd'])
        if any(abs(metrics[k]-oof['metrics'][k])>1e-12 for k in r.KEYS):
            raise AssertionError(f'Independent formal metric mismatch: {name}')
        fold_metrics.setdefault(key,[]).append(metrics);formal_count+=1
        frozen=saved['frozen'];p=np.asarray(frozen['test_prob_asd']);pred=np.asarray(frozen['test_pred'])
        if p.shape!=(5,len(y)) or not np.isfinite(p).all() or not np.array_equal(pred,np.where(p>=.5,0,1)):
            raise AssertionError(f'Incomplete frozen predictions: {name}')
        for draw in range(5):
            metrics=rank_metrics(y,pred[draw],p[draw])
            if any(abs(metrics[k]-frozen['draw_metrics'][draw][k])>1e-12 for k in r.KEYS):
                raise AssertionError(f'Independent frozen metric mismatch: {name}/{draw}')
            draw_count+=1
        p_mean=p.mean(0);metrics=rank_metrics(y,np.where(p_mean>=.5,0,1),p_mean)
        if any(abs(metrics[k]-frozen['mean_probability_metrics'][k])>1e-12 for k in r.KEYS):
            raise AssertionError(f'Independent probability-mean metric mismatch: {name}')
        if saved['group'] in ('R2','R3'):
            audit=verified_counterfactual(saved)
            if not audit['BN_unchanged'] or not audit['same_original_joint_for_all_conditions']:
                raise AssertionError(f'Invalid fixed-base intervention: {name}')
            aux_count+=1
        task=f"train/{saved['group']}/{saved['seed']}/{saved['fold']}"
        if state['attempts'][task]!=1:
            raise AssertionError(f'Unexpected training retry: {task}')
    for (group,seed),indices in coverage.items():
        if not np.array_equal(np.sort(indices),np.arange(871)):
            raise AssertionError(f'OOF coverage mismatch: {group}/{seed}')
        for metric in r.KEYS:
            value=float(np.mean([fold[metric] for fold in fold_metrics[(group,seed)]]))
            if abs(value-state['groups'][group]['formal']['seed_means'][str(seed)][metric])>1e-12:
                raise AssertionError(f'Seed aggregate mismatch: {group}/{seed}/{metric}')
    if any(state['tasks'][error['task']]['state']!='complete' for error in state['errors']):
        raise AssertionError('Unresolved failed diagnostic task')
    return {'passed':True,'time_utc':r.utc(),'formal_folds':formal_count,'frozen_draws':draw_count,
            'verified_auxiliary_artifacts':aux_count,'files':len(files),'checkpoint_files_unchanged':120,
            'formal_signatures_unchanged':120,'exact_OOF_coverage':True,'neural_forward_performed':False,
            'training_restarted':False,'method':'Independent confusion counts and Mann-Whitney rank AUC with average ties'}


def finalize(state):
    started=time.perf_counter()
    state['groups']={group:group_summary(group,state) for group in ('C0','C1','C2','C3','C4','C5','C6',*r.GROUPS)}
    state['comparisons']={f'{a}-{b}':pair(a,b,state) for a,b in r.config()['comparisons']}
    state['replay_audit']=replay_audit(state)
    state['optimization']=optimization_audit(state)
    state['matched_optimization']=matched_optimization(state)
    state['conditional']={group:conditional_audit(group,state) for group in ('C2','C3','C4','C5','R3')}
    state['research_answers']=research_answers(state)
    # Re-read all protected inputs after all training/diagnostics have finished.
    for name,saved in state['audit']['manifest'].items():
        if r.sha_file(r.ROOT/name)!=saved['sha256']:
            raise AssertionError(f'Read-only historical file changed: {name}')
    state['readonly_final']={'passed':True,'files':len(state['audit']['manifest']),'time_utc':r.utc()}
    train=[(key,value) for key,value in state['tasks'].items() if key.startswith('train/') and value['state']=='complete']
    if len(train)!=120:
        raise AssertionError(f'Expected 120 completed formal training tasks, got {len(train)}')
    by_gpu={str(g):sum(v['gpu']==g for _,v in train) for g in (0,1)}
    by_group={group:{str(g):sum(k.split('/')[1]==group and v['gpu']==g for k,v in train) for g in (0,1)} for group in r.GROUPS}
    state['execution']={'formal_fits':120,'frozen_draws':600,'layer_diagnostic_checkpoints':330,
                        'training_by_gpu':by_gpu,'groups_by_gpu':by_group,
                        'elapsed_seconds':(datetime.fromisoformat(r.utc())-datetime.fromisoformat(state['created_utc'])).total_seconds(),
                        'queue_and_analysis_seconds':sum(v['elapsed_seconds'] for v in state['scheduling']['phases'].values())+time.perf_counter()-started,
                        'elapsed_origin':'summary.created_utc, before persistent read-only audit'}
    state['execution']['resources']=execution_audit(state)
    state['final_code_sha256']={name:r.sha_file(r.BASE/name) for name in ('run.py','controls.py','launch.py','analyze.py','config.json')}
    state['final_checkpoints_sha256']={str(path.relative_to(r.BASE)):r.sha_file(path) for path in sorted((r.BASE/'runs').glob('R*/*.pt'))}
    if len(state['final_checkpoints_sha256'])!=120:
        raise AssertionError('Unexpected final checkpoint file count')
    state['phase']='research_complete_pending_migration_and_handoff'
    state['research_completed_utc']=r.utc()
    r.atomic_json(state)
    write_report(state)
    state['report_sha256']=r.sha_file(r.BASE/'report.md')
    r.atomic_json(state)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('repair','verify'))
    parser.add_argument('--group',default='R0')
    parser.add_argument('--seed',type=int)
    parser.add_argument('--fold',type=int)
    args=parser.parse_args()
    state=r.guard()
    if args.action=='verify':
        print(json.dumps(verify_completed_bundle(state),ensure_ascii=False),flush=True)
        raise SystemExit(0)
    if args.seed is None or args.fold is None:
        parser.error('repair requires --seed and --fold')
    r.require_ppu()
    with r.task_lock('R2',args.seed,args.fold,extra=400):
        r.emit('begin',action='repair',seed=args.seed,fold=args.fold)
        value=repair_counterfactual_one(args.seed,args.fold,state)
        r.emit('done',action='repair',result=value)
