"""Preregistered development gate, paired formal audit and Chinese research report."""
from __future__ import annotations
import hashlib
import csv
import json
from pathlib import Path
import os
import time
import numpy as np
from scipy.stats import spearmanr
import run as ex
BASE,CFG,ROOT = ex.BASE,ex.CFG,ex.ROOT


def development_gate(state):
    gate = CFG['scientific_gate']
    rows,failed_gradient,near_zero = [],[],[]
    no_positive_gap=True
    all_val_not_better=True
    for seed,fold in CFG['development_anchors']:
        values={g:ex.load_torch(ex.task_path('develop',g,seed,fold)) for g in CFG['groups']}
        assert len({v['private_init_sha256'] for v in values.values()})==1
        assert len({v['data_digest'] for v in values.values()})==1
        for g,v in values.items():
            assert not v['outer_test_metrics_computed'] and 'oof' not in v and v['epochs_run']==80
            assert len(v['curves'])==80 and len(v['diagnostics'])==8
            assert v['best_core_sha256']==ex.core_hash(v['best']['core'])
            assert v['best_aligner_sha256']==ex.state_hash(v['best']['aligner'])
        v=values['L3']
        late=[v['diagnostics'][str(e)] for e in CFG['late_diagnostic_epochs']]
        gaps=[r['mechanism']['true_vs_null_JS_gap']['mean'] for r in late]
        native_g=[r['gradients']['grad_from_native_to_relation'] for r in late]
        local_g=[r['gradients']['grad_from_local_loss_to_projection'] for r in late]
        graph=[r['mechanism']['relative_graph_l1'] for r in late]
        centered=[r['mechanism']['centered_score_max'] for r in late]
        lacks_native=all(x<=gate['gradient_norm_floor'] for x in native_g)
        lacks_local=all(x<=gate['gradient_norm_floor'] for x in local_g)
        negligible=all(a<gate['near_zero_relative_graph_l1'] and b<gate['near_zero_centered_score_max'] for a,b in zip(graph,centered))
        if lacks_native or lacks_local:failed_gradient.append(f'{seed}/{fold}')
        if negligible:near_zero.append(f'{seed}/{fold}')
        no_positive_gap=no_positive_gap and all(g<=0 for g in gaps)
        selected={g:r['best']['selection']['val_ce'] for g,r in values.items()}
        diffs={g:selected['L3']-selected[g] for g in ('L1','L2','L4')}
        all_val_not_better=all_val_not_better and all(d>=0 for d in diffs.values())
        rows.append({'seed':seed,'fold':fold,'JS_gap_late':gaps,'native_relation_grad_late':native_g,
            'weighted_local_projection_grad_late':local_g,'relative_graph_l1_late':graph,
            'centered_score_max_late':centered,'persistent_missing_gradient':lacks_native or lacks_local,
            'persistent_near_zero_graph':negligible,'selected_validation_ce':selected,'L3_selected_valCE_minus_control':diffs,
            'selected_epochs':{g:r['best_epoch'] for g,r in values.items()},
            'each_group_late':{g:{str(e):r['diagnostics'][str(e)] for e in CFG['late_diagnostic_epochs']} for g,r in values.items()}})
    reasons=[]
    threshold=gate['persistent_failure_anchor_threshold']
    if len(failed_gradient)>=threshold:reasons.append('persistent_missing_native_or_local_gradient_in_at_least_two_anchors')
    if len(near_zero)>=threshold:reasons.append('persistent_near_zero_graph_and_centered_scores_in_at_least_two_anchors')
    if no_positive_gap and all_val_not_better:reasons.append('no_true_null_training_advantage_and_no_validation_CE_advantage_in_all_five_anchors')
    return {'go':not reasons,'reasons':reasons,'failed_gradient_anchors':failed_gradient,'near_zero_graph_anchors':near_zero,
        'no_positive_training_gap':no_positive_gap,'all_five_val_ce_not_better_than_all_controls':all_val_not_better,
        'rows':rows,'development_only':True,'outer_test_performance_computed':False,'frozen_config_sha256':ex.sha_file(BASE/'config.json'),
        'criteria_frozen_before_training':True,'not_a_significance_test':True,'evaluated_utc':ex.utc()}


def verify_history(audit):
    for name,row in audit['manifest'].items():
        assert ex.sha_file(name)==row['sha256'],f'Protected file changed: {name}'
    for name,sha in audit['source_sha256'].items():assert ex.sha_file(ROOT/name)==sha
    return {'passed':True,'files':len(audit['manifest']),'utc':ex.utc()}


def aggregate(values):
    seed_means={str(seed):{m:float(np.mean([v['metrics'][m] for v in values if v['seed']==seed])) for m in ex.METRICS} for seed in CFG['seeds']}
    return {'seed_means':seed_means,'mean':{m:float(np.mean([v[m] for v in seed_means.values()])) for m in ex.METRICS},
        'seed_sd_ddof1':{m:float(np.std([v[m] for v in seed_means.values()],ddof=1)) for m in ex.METRICS}}


def baseline_rows(audit):
    # Only called after formal Go and all training; never selected using test performance.
    old=json.loads((ROOT/'experiments/PCND_20261016/summary.json').read_text())
    rows=[]
    for seed in CFG['seeds']:
        for fold in range(10):
            b=old['audit']['baseline'][f'{seed}/{fold}']
            path=ex.OLD/'outputs/E0'/f'seed_{seed}'/f'fold_{fold}'/'oof.csv'
            with path.open() as stream: source=list(csv.DictReader(stream))
            assert np.array_equal([int(r['index']) for r in source],b['indices'])
            assert np.array_equal([int(r['label_original']) for r in source],b['labels'])
            assert np.array_equal([int(r['pred_original']) for r in source],b['pred'])
            assert np.allclose([float(r['prob_asd']) for r in source],b['prob_asd'],rtol=0,atol=1e-12)
            rows.append({'seed':seed,'fold':fold,'metrics':b['metrics'],
                'oof':{key:np.asarray(b[key]) for key in ('indices','labels','prob_asd','pred','sites')},'best_epoch':b['best_epoch']})
    return rows


def formal_analysis(audit):
    groups={'R0':baseline_rows(audit)}
    registry={}
    for group in CFG['groups']:
        groups[group]=[]
        for seed in CFG['seeds']:
            seen=[]
            for fold in range(10):
                path=ex.task_path('train',group,seed,fold)
                v=ex.load_torch(path)
                assert v['state']=='complete' and v['code_sha256']==ex.code_hash()
                assert v['best_core_sha256']==ex.core_hash(v['best']['core'])
                assert v['best_aligner_sha256']==ex.state_hash(v['best']['aligner'])
                assert 'frozen' in v and len(v['frozen'])==5
                o=v['oof'];idx=np.asarray(o['indices']);seen.extend(idx.tolist())
                anchor=audit['anchors'][f'{seed}/{fold}']
                assert ex.array_hash(idx)==anchor['pairing']['split_fingerprints']['test']
                context=ex.load_torch(ex.OLD/'artifacts/context.pt')
                assert np.array_equal(o['labels'],context['labels'][idx])
                assert list(o['ids'])==[context['ids'][i] for i in idx]
                assert np.array_equal(o['prob_asd'],ex.softmax(o['logits'],axis=1)[:,0])
                computed=ex.metrics(o['labels'],o['prob_asd'],o['pred'])
                assert all(abs(computed[m]-v['metrics'][m])<1e-12 for m in ex.METRICS)
                groups[group].append(v)
                registry[str(path)]={'sha256':ex.sha_file(path),'bytes':path.stat().st_size}
            assert np.array_equal(np.sort(seen),np.arange(871))
    formal={g:aggregate(v) for g,v in groups.items()}
    expected={'ACC':.8188,'BA':.8168,'AUC':.8854,'SEN':.7879,'SPE':.8456,'F1':.7982}
    assert all(abs(formal['R0']['mean'][m]-expected[m])<.000051 for m in expected)
    for i in range(30):
        paired=[groups[g][i] for g in CFG['groups']]
        assert len({v['private_init_sha256'] for v in paired})==1
        assert len({v['data_digest'] for v in paired})==1
        assert len({v['input_sha256'] for v in paired})==1
        for v in paired:
            assert v['execution']['BN_updates']==v['epochs_run']
            assert all(n==2*v['epochs_run'] for n in v['execution']['native_submodule_calls'].values())
    paired={}
    for first,second in (('L1','R0'),('L2','L1'),('L3','L2'),('L3','L4'),('L3','R0')):
        a,b=groups[first],groups[second]
        rows=[]
        for av,bv in zip(a,b):
            assert (av['seed'],av['fold'])==(bv['seed'],bv['fold'])
            assert np.array_equal(av['oof']['indices'],bv['oof']['indices'])
            pa,pb=av['oof']['prob_asd'],bv['oof']['prob_asd'];y=av['oof']['labels']
            rows.append({'seed':av['seed'],'fold':av['fold'],**{m:av['metrics'][m]-bv['metrics'][m] for m in ex.METRICS},
                'probability_MAD':float(np.abs(pa-pb).mean()),'CE_difference':float((ex.ce_values(pa,y)-ex.ce_values(pb,y)).mean())})
        seed_delta={str(s):{m:float(np.mean([r[m] for r in rows if r['seed']==s])) for m in ex.METRICS+('CE_difference','probability_MAD')} for s in CFG['seeds']}
        paired[first+'-'+second]={'seed_differences':seed_delta,'mean':{m:float(np.mean([r[m] for r in seed_delta.values()])) for m in seed_delta[str(CFG['seeds'][0])]},
            'paired_fold_wins':{m:int(sum(r[m]>0 for r in rows)) for m in ex.METRICS},'folds':rows,
            'seed_difference_min_max':{m:[min(r[m] for r in seed_delta.values()),max(r[m] for r in seed_delta.values())] for m in ex.METRICS},
            'uncertainty_note':'three repeated seed differences are descriptive; correlated CV folds are not independent clinical cohorts'}
    frozen={}
    for group,values in groups.items():
        if group=='R0':continue
        frows=[]
        for v in values:
            o=v['oof'];y=o['labels']
            for record in v['frozen']:
                probs=record['prob_asd'];pt=probs['trained'];off=probs['off']
                row={'seed':v['seed'],'fold':v['fold'],'draw_seed':record['draw_seed'],
                    'trained_minus_off_CE':float((ex.ce_values(pt,y)-ex.ce_values(off,y)).mean()),
                    'trained_off_probability_MAD':float(np.abs(pt-off).mean()),
                    'rescued':int(np.sum(((pt>=.5)==(y==0))&((off>=.5)!=(y==0)))),
                    'harmed':int(np.sum(((pt>=.5)!=(y==0))&((off>=.5)==(y==0)))),
                    'metric_delta_trained_off':{m:ex.metrics(y,pt)[m]-ex.metrics(y,off)[m] for m in ex.METRICS}}
                if 'true' in probs:
                    true,wrong=probs['true'],probs['shuffled']
                    mapping=np.asarray(audit['anchors'][f'{v["seed"]}/{v["fold"]}']['inputs']['L4']['test']['sources'])
                    ctx=ex.load_torch(ex.OLD/'artifacts/context.pt');meta=ex.metadata(ctx);idx=np.asarray(o['indices'])
                    changed=(meta['age'][idx]!=meta['age'][mapping])|(meta['sex'][idx]!=meta['sex'][mapping])
                    row.update(true_minus_wrong_CE=float((ex.ce_values(true,y)-ex.ce_values(wrong,y)).mean()),
                        true_wrong_probability_MAD=float(np.abs(true-wrong).mean()),effective_shuffle_fraction=float(changed.mean()),
                        true_wrong_metric_delta={m:ex.metrics(y,true)[m]-ex.metrics(y,wrong)[m] for m in ex.METRICS},
                        effective_subset_CE=float((ex.ce_values(true[changed],y[changed])-ex.ce_values(wrong[changed],y[changed])).mean()) if changed.any() else None)
                frows.append(row)
        def hierarchy(field):
            return {str(s):float(np.mean([r[field] for r in frows if r['seed']==s])) for s in CFG['seeds']}
        frozen[group]={'rows':frows,'trained_off_probability_MAD_by_seed':hierarchy('trained_off_probability_MAD'),
            'trained_minus_off_CE_by_seed':hierarchy('trained_minus_off_CE')}
        if group!='L1':
            frozen[group].update(true_minus_wrong_CE_by_seed=hierarchy('true_minus_wrong_CE'),true_wrong_probability_MAD_by_seed=hierarchy('true_wrong_probability_MAD'))
    repeat={g:repeat_metrics(values) for g,values in groups.items() if g!='R0'}
    return {'formal':formal,'paired':paired,'frozen':frozen,'artifact_registry':registry,'groups':groups,'repeat_metrics':repeat}


def repeat_metrics(values):
    modes=tuple(values[0]['frozen'][0]['prob_asd'])
    result={}
    for mode in modes:
        draw=[]
        for j in range(5):
            rows=[{'seed':v['seed'],'metrics':ex.metrics(v['oof']['labels'],v['frozen'][j]['prob_asd'][mode])} for v in values]
            draw.append(aggregate(rows))
        ensemble=aggregate([{'seed':v['seed'],'metrics':ex.metrics(v['oof']['labels'],
            np.mean([r['prob_asd'][mode] for r in v['frozen']],axis=0))} for v in values])
        changes={};variability={}
        for seed in CFG['seeds']:
            p=np.concatenate([np.stack([r['prob_asd'][mode] for r in v['frozen']]) for v in values if v['seed']==seed],axis=1)
            changes[str(seed)]=float(np.any((p>=.5)!=(p[:1]>=.5),axis=0).mean())
            variability[str(seed)]=float(p.std(axis=0,ddof=1).mean())
        result[mode]={'five_draw_metrics':draw,'five_probability_mean_metrics':ensemble,
            'draw_mean_metrics':{m:float(np.mean([r['mean'][m] for r in draw])) for m in ex.METRICS},
            'draw_sd_ddof1_metrics':{m:float(np.std([r['mean'][m] for r in draw],ddof=1)) for m in ex.METRICS},
            'patient_any_label_change_by_seed':changes,'patient_probability_sd_by_seed':variability}
    return result


def correlation(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if len(a)<3 or np.ptp(a)==0 or np.ptp(b)==0:return None
    return float(spearmanr(a,b).statistic)


def mechanism_analysis(groups,audit):
    """Post-training, label-free best-projection audit; no additional GTUNet or optimization."""
    torch=ex.torch
    assert torch.cuda.is_available() and 'PPU' in torch.cuda.get_device_name(0).upper()
    device=torch.device('cuda:0')
    ctx=ex.load_torch(ex.OLD/'artifacts/context.pt');meta=ex.metadata(ctx)
    all_rows={g:[] for g in CFG['groups']}; patient_rows={'L3':[],'L4':[]}
    evolution={g:{} for g in CFG['groups']}
    scalar_fields=('relative_graph_l1','centered_score_max','P_entropy','Q_entropy','image_effective_rank',
        'train_true_js','train_null_js','within_site_relative_signed_change','cross_site_relative_signed_change')
    for group in CFG['groups']:
        for epoch in (0,20,40,60,79):
            rows=[v['diagnostics'][str(epoch)] for v in groups[group] if str(epoch) in v['diagnostics']]
            evolution[group][str(epoch)]={key:ex.describe([r['mechanism'][key] for r in rows]) for key in scalar_fields}
            evolution[group][str(epoch)]['gradients']={key:ex.describe([r['gradients'][key] for r in rows]) for key in
                ('grad_from_native_to_relation','grad_from_native_to_projection','grad_from_local_loss_to_projection','grad_total_relation')}
    for seed in CFG['seeds']:
        x=ex.load_torch(ex.anchor_dir(seed,0).parent/'selected_features.pt')
        for fold in range(10):
            split=ex.load_torch(ex.anchor_dir(seed,fold)/'splits.pt')
            inputs=ex.prepare_inputs(x.numpy(),meta,split,seed,fold,CFG)
            assert ex.digest(inputs['info'])==audit['anchors'][f'{seed}/{fold}']['inputs_digest']
            age=meta['age'].copy();age[~np.isfinite(age)]=inputs['info']['age_impute_train_median']
            sex=meta['sex'];candidates=inputs['candidates'];idx=np.asarray(split['test']);tr=np.asarray(split['train'])
            null=ex.np.stack([ex.phenotype_kernel(age,sex,age[p[candidates]],sex[p[candidates]],
                inputs['info']['age_bandwidth']) for p in inputs['nulls']])
            null_q=torch.as_tensor(null,device=device).div(CFG['tau_ph']).softmax(-1)
            x_gpu=x.to(device)
            tensors={'candidates':torch.as_tensor(candidates,device=device),'raw_cos':torch.as_tensor(inputs['raw_cos'],device=device),
                'kernels':{k:torch.as_tensor(v,device=device) for k,v in inputs['kernels'].items()},
                'null_kernels':{k:torch.as_tensor(v,device=device) for k,v in inputs['null_kernels'].items()}}
            for group in CFG['groups']:
                v=next(v for v in groups[group] if (v['seed'],v['fold'])==(seed,fold))
                module=ex.LCRA.isolated(CFG,seed,fold,device);module.load_state_dict(v['best']['aligner']);module.eval()
                with torch.no_grad():
                    d=module(x_gpu,tensors,torch.as_tensor(tr,device=device),group)
                    p=d['sim'].div(CFG['tau_img']).softmax(-1)
                    q=tensors['kernels']['true'].div(CFG['tau_ph']).softmax(-1)
                    gap=(ex.js_divergence(p[None],null_q,CFG['epsilon']).mean(0)-ex.js_divergence(p,q,CFG['epsilon'])).cpu().numpy()
                    ent=(-(p*p.clamp_min(1e-8).log()).sum(-1)).cpu().numpy()
                    rank=ex.effective_rank(d['Z'][tr].cpu().numpy())
                nearest=min((int(e) for e in v['diagnostics']),key=lambda e:abs(e-v['best_epoch']))
                all_rows[group].append({'seed':seed,'fold':fold,'best_epoch':v['best_epoch'],
                    'best_projection_rank':rank,'best_projection_relation_entropy_train':float(ent[tr].mean()),
                    'best_projection_true_null_gap_train':ex.describe(gap[tr]),'best_projection_true_null_gap_test':ex.describe(gap[idx]),
                    'sampled_graph_diagnostic_epoch':nearest,'sampled_epoch_distance_from_best':abs(nearest-v['best_epoch']),
                    'sampled_graph_mechanism':v['diagnostics'][str(nearest)]['mechanism'],
                    'sampled_gradients':v['diagnostics'][str(nearest)]['gradients']})
                if group not in patient_rows:continue
                true=np.stack([r['prob_asd']['true'] for r in v['frozen']])
                wrong=np.stack([r['prob_asd']['shuffled'] for r in v['frozen']])
                y=np.asarray(v['oof']['labels'])
                benefit=np.mean(ex.ce_values(wrong,y)-ex.ce_values(true,y),axis=0)
                a=next(r for r in groups['L3'] if (r['seed'],r['fold'])==(seed,fold))
                b=next(r for r in groups['L4'] if (r['seed'],r['fold'])==(seed,fold))
                cross=ex.ce_values(b['oof']['prob_asd'],y)-ex.ce_values(a['oof']['prob_asd'],y)
                cuts=np.quantile(gap[tr],[.25,.5,.75]);bins=np.searchsorted(cuts,gap[idx],side='right')
                moved=(age[idx]!=age[inputs['shuffle'][idx]])|(sex[idx]!=sex[inputs['shuffle'][idx]])
                kernel_changed=np.any(inputs['kernels']['true'][idx]!=inputs['kernels']['shuffled'][idx],axis=1)
                patient_rows[group].append({'seed':seed,'fold':fold,'indices':idx.tolist(),'reliability_JS_gap':gap[idx].tolist(),
                    'frozen_correct_condition_CE_benefit':benefit.tolist(),'cross_trained_L3_CE_benefit_vs_L4':cross.tolist(),
                    'train_calibrated_reliability_quartile':bins.tolist(),'query_own_condition_changed':moved.tolist(),
                    'candidate_kernel_changed':kernel_changed.tolist(),'train_quartile_cuts':cuts.tolist()})
                del module,d
            del x_gpu,tensors,null_q
    associations={}
    for group,rows in patient_rows.items():
        seeds={}
        for seed in CFG['seeds']:
            selected=[r for r in rows if r['seed']==seed]
            fields={k:np.concatenate([np.asarray(r[k]) for r in selected]) for k in
                ('indices','reliability_JS_gap','frozen_correct_condition_CE_benefit','cross_trained_L3_CE_benefit_vs_L4',
                 'train_calibrated_reliability_quartile','query_own_condition_changed','candidate_kernel_changed')}
            index=fields['indices'];reliability=fields['reliability_JS_gap'];benefit=fields['frozen_correct_condition_CE_benefit']
            cross=fields['cross_trained_L3_CE_benefit_vs_L4']
            strata={f'reliability_quartile/{i+1}':fields['train_calibrated_reliability_quartile']==i for i in range(4)}
            strata.update({f'site/{s}':meta['site'][index]==s for s in sorted(set(meta['site']))})
            strata.update({f'sex/{s}':meta['sex'][index]==s for s in sorted(set(meta['sex']))})
            strata.update({f'age/{lo}-{hi}':(meta['age'][index]>=lo)&(meta['age'][index]<hi) for lo,hi in ((0,12),(12,18),(18,30),(30,200))})
            strata['query_own_condition_changed']=fields['query_own_condition_changed']
            strata['candidate_kernel_changed']=fields['candidate_kernel_changed']
            seeds[str(seed)]={'reliability_vs_frozen_benefit_spearman':correlation(reliability,benefit),
                'reliability_vs_cross_trained_benefit_spearman':correlation(reliability,cross),
                'strata':{name:{'n':int(mask.sum()),'mean_reliability':float(reliability[mask].mean()),
                    'mean_frozen_CE_benefit':float(benefit[mask].mean()),'benefited_fraction':float((benefit[mask]>0).mean()),
                    'mean_cross_trained_CE_benefit':float(cross[mask].mean())} for name,mask in strata.items() if mask.any()}}
        associations[group]={'seeds':seeds,'fold_records':rows,
            'note':'Descriptive, post-training; correct-condition frozen benefit is paired within checkpoint; cross-trained benefit is not a single-module causal effect. Reliability quartiles use training queries only.'}
    return {'projection_checkpoint_rows':all_rows,'sampled_training_evolution':evolution,'patient_reliability':associations,
        'sample_resolution_note':'Best-projection rank/gap are recomputed exactly with no GTUNet. Graph/gradient rows are the nearest saved training-forward sample, not the exact saved validation-selected state.',
        'additional_training':False,'additional_GTUNet_forwards':0,'projection_audit_device':torch.cuda.get_device_name(0)}


def append_handoff(state,text,audit):
    path=ROOT/'handoff.md'
    prefix=path.read_bytes()
    marker=b'<!-- LCRA_20261018_START -->'
    if marker in prefix:
        assert text.encode() in prefix,'An existing different LCRA handoff must not be overwritten'
        return {'status':'already_appended','sha256':hashlib.sha256(prefix).hexdigest()}
    assert hashlib.sha256(prefix).hexdigest()==audit['handoff_before']['sha256'],'Root handoff changed externally'
    payload=prefix+b'\n\n'+text.encode()
    temporary=BASE/'handoff.atomic.pending'
    try:
        with temporary.open('xb') as stream:
            stream.write(payload);stream.flush();os.fsync(stream.fileno())
        assert temporary.read_bytes()==payload
        assert path.read_bytes()==prefix
        os.replace(temporary,path)
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        actual=path.read_bytes()
        assert actual==payload and actual[:len(prefix)]==prefix
        return {'status':'appended','before_sha256':hashlib.sha256(prefix).hexdigest(),'after_sha256':hashlib.sha256(actual).hexdigest(),
            'historical_prefix_unchanged':True,'atomic':True}
    except (OSError,AssertionError) as err:
        ex.atomic_bytes(text.encode(),BASE/'handoff_pending.md')
        return {'status':'pending_storage_or_integrity_error','error':repr(err),'append_text':text,'path':str(BASE/'handoff_pending.md')}


def verify_terminal(state,audit):
    assert ex.sha_file(BASE/'report.md')==state['report_sha256']
    assert ex.source_hashes()==state['new_source_sha256']
    verify_history(audit)
    for name,row in state['artifact_registry'].items():assert ex.sha_file(name)==row['sha256']
    if state['handoff']['status']=='appended':assert ex.sha_file(ROOT/'handoff.md')==state['handoff']['after_sha256']
    return True


def finish(state,audit,formal):
    gate=state['preflight']['scientific_gate']
    state['new_source_sha256']=ex.source_hashes()
    assert state['new_source_sha256']==audit['new_source_sha256']
    state['code_sha256']=ex.code_hash()
    state['protected_history_final']=verify_history(audit)
    registry={}
    for seed,fold in CFG['development_anchors']:
        for g in CFG['groups']:
            p=ex.task_path('develop',g,seed,fold)
            v=ex.load_torch(p)
            assert not v['outer_test_metrics_computed'] and 'oof' not in v
            registry[str(p)]={'sha256':ex.sha_file(p),'bytes':p.stat().st_size}
    state['development']={'tasks':20,'epochs':1600,'formal_tasks':0,'outer_test_evaluations':0,
        'gpu_tasks':{str(gpu):sum(r.get('state')=='complete' and r['gpu']==gpu and k.startswith('develop/') for k,r in state['tasks'].items()) for gpu in (0,1)},
        'gradient_and_graph_gate':gate}
    state['formal_completed']=120 if formal else 0
    state['formal_gpu_tasks']={str(g):sum(r.get('state')=='complete' and r['gpu']==g and k.startswith('train/') for k,r in state['tasks'].items()) for g in (0,1)}
    state['formal_group_completed']={g:30 if formal else 0 for g in CFG['groups']}
    if formal:
        assert state['formal_gpu_tasks']=={'0':60,'1':60}
        assert all(sum(k.startswith('train/'+g+'/') and r.get('state')=='complete' for k,r in state['tasks'].items())==30 for g in CFG['groups'])
        assert all(not r.get('result',{}).get('cpu_training',False) for r in state['tasks'].values())
    state['wall_seconds_to_acceptance']=(__import__('datetime').datetime.now(__import__('datetime').timezone.utc)-__import__('datetime').datetime.fromisoformat(state['started_utc'])).total_seconds()
    state['worker_train_seconds']=sum(r.get('result',{}).get('seconds',0) for k,r in state['tasks'].items() if k.startswith(('train/','develop/')) and r.get('state')=='complete')
    state['cpu_training']=False
    state['new_outer_test_evaluations']=120 if formal else 0
    state['official_single_test_forwards']=120 if formal else 0
    state['auxiliary_frozen_test_forwards']=1650 if formal else 0
    state['frozen_checkpoint_tasks']=120 if formal else 0
    state['artifact_registry']=registry
    baseline='|R0 历史C0|81.88 +/- 1.39|81.68 +/- 1.27|88.54 +/- 0.31|78.79 +/- 1.65|84.56 +/- 3.31|79.82 +/- 1.12|30只读复用|'
    report=[
        '# 实验8：LCRA 局部对比式跨模态关系对齐',
        f'验收UTC：{ex.utc()}。正式训练 {state["formal_completed"]}/120；开发检查20/20任务、1600个训练/验证epoch。',
        '## 结论与Go/No-Go',
        ('通过预注册开发门槛后执行全部正式任务，以下区分正式单次测试与冻结机制。' if formal else
         '按照预注册科学门槛停止，不产生L1-L4正式分类分数、正式OOF或正式冻结反事实。80轮开发筛选不是发表性能实验，也不能推断500轮最终性能。'),
        f'门槛理由：{json.dumps(gate["reasons"],ensure_ascii=False)}。配置在训练前冻结，未改变K、alpha、温度、margin或损失系数。',
        '## 方法与源码审计',
        '训练分区逐维标准化后构造固定影像Top64训练锚点；训练患者排除自身。关系投影500->32/LN/GELU->16；5->32->1关系MLP末层精确零初始化。L1把全部三个表型依赖输入通道置零，L2使用年龄/性别核，L3增加局部真实-vs-null平滑JS间隔，L4仅在新增分支内按分区/站点联合错配。四组结构、初始化、优化器及历史RNG相同。',
        '年龄带宽为训练非自身年龄差中位数下限2岁；同性/异性/未知核为1/0.5/0.75，缺失年龄只以训练中位数填补。8个训练站点内联合年龄/性别Null双射；L4的Null再次独立置换其正条件。置换、TopK索引及SHA见audit.json；训练拟合对held-out协变量替换保持不变，未改变原生表型。',
        'u行内中心化后tanh，D仅在候选边非零，E=(D+D.T)/2并清对角。A_new=A_base*(1+0.25E)，保持边支持、对称、非负和对角线，最大25%相对界限。没有q、硬门槛或warm-up。对称化并不严格保度，实际行度与站点内外边变化均记录。',
        'P=softmax(学习影像cos/0.15)，Q=softmax(表型核/0.20)，Llocal=mean[0.10*softplus((JStrue-meanJSnull+0.01)/0.10)]，总损失=原训练CE+原图正则+0.05*Llocal，仅L3/L4启用。验证选模始终为原val CE+对应新图原生Graph Loss。',
        '关系更新只插在原fused_sim*rp_graph后、dense_to_sparse前；每次只调用一次VAE、两条GTUNet各一次和Classifier一次。保留VAE反复加载和随机采样、eval Dropout、RP提前return、未注册RP权重、.data原位约束、detach权重及原始BN。未重新进行RFE或VAE预训练。',
        '原论文的RFE/VAE首先适配维度，不能等同患者级语义对齐；源码相似度为Euclidean cdist而非论文correlation表述。原双GTUNet各16维，ACMGL已经融合两分支及均值，分类器16->256/ReLU/BN->2。[原论文](https://arxiv.org/html/2406.14455v3)；[官方源码](https://github.com/NZWANG/MM-GTUNets)；[当前仓库](https://github.com/tangaii/MM)。',
        '历史实验1/2的高余弦与门控未证实患者配对；实验3/4说明融合残差相对尺度与联合适应风险；实验5/6的发育偏差未建立稳定额外诊断收益。实验4实际只读路径在/tmp。实验7源码实际位于/tmp/DPR_Align_20261017，并未假称存在于GitHub。',
        '实验7的5锚点/R3-R5在epoch40-44全部q=0，图与JS投影梯度为0；相对Null SD下界主导，epoch44全局最大z=0.81328，阈值1未越过。全局分布高熵使真实与错配趋近，硬截断形成当前窗口的失活，但不是已证明未来永远失活或影像表征完全秩坍塌。本轮没有降低其q阈值，而另行采用局部连续目标。',
        '## 严格数值验收',
        '5个固定锚点911/0、911/5、912/0、912/5、913/0：原模型与alpha=0 wrapper连续三步logits、CE/Graph、GTUNet输出、TopK、核心梯度、Adam状态/更新、BN和完整RNG严格一致，最大误差0。CPU仅合成单测，无CPU神经优化；实际训练均PPU。私有初始化不消费原生RNG，Checkpoint含未注册RP权重可完整恢复。',
        '单元测试涵盖JS闭式/对称/有限梯度与CPU数值差分、softplus、未知性别和单例、候选边界、零头图精确恢复、后续投影可微以及非负/对称/支持/边权范围。原生正式后端未额外启用确定性；短步确定性等价不保证PPU生产长程逐位重放。',
        '## 开发性筛选',
        '全部20任务各80轮，不提前早停以确保预定晚期窗口；选模仍为同一val CE+Graph规则。分别测量native->relation、加权local->projection及总梯度，不拿总梯度替代对齐梯度。未计算新模型外层测试分类指标。',
        '|锚点|L3晚期原生梯度范围|局部目标投影梯度范围|晚期相对图L1范围|中心分数最大范围|真实-null JS gap范围|',
        '|---|---|---|---|---|---|']
    def span(v):return f'{min(v):.6g} ~ {max(v):.6g}'
    for r in gate['rows']:
        report.append(f'|{r["seed"]}/{r["fold"]}|{span(r["native_relation_grad_late"])}|{span(r["weighted_local_projection_grad_late"])}|{span(r["relative_graph_l1_late"])}|{span(r["centered_score_max_late"])}|{span(r["JS_gap_late"])}|')
    report += ['晚期窗口固定epoch20/40/60/79；近零规则同时要求四点相对图L1<0.01%且中心分数最大值<0.001，并综合实际梯度及float32精度解释。初始零头的合法零效应不计入中止。>=2锚点持续无有效路径或持续近零图效果触发科学停止。',
        '|锚点|L1选中val CE|L2选中val CE|L3选中val CE|L4选中val CE|L3-L1|L3-L2|L3-L4|',
        '|---|---|---|---|---|---|---|---|']
    for r in gate['rows']:
        vals=r['selected_validation_ce'];d=r['L3_selected_valCE_minus_control']
        report.append(f'|{r["seed"]}/{r["fold"]}|'+ '|'.join(f'{vals[g]:.7f}' for g in CFG['groups'])+'|'+'|'.join(f'{d[g]:+.7f}' for g in ('L1','L2','L4'))+'|')
    report += ['各组选中epoch按同一规则在0-79内产生，而不是把某组最好CE与另一组任意点比较；每epoch训练CE、Graph、验证CE/Graph曲线在单任务pt。验证BA/AUC只作开发诊断，不能充当外层正式性能。',
        '## 正式结果','单位%，每seed先平均十折，再报三seed均值及样本SD；ASD=0，prob-AUC使用第0列连续概率。',
        '|组别|ACC|BA|prob-AUC|SEN-ASD|SPE-HC|F1-ASD|完成|','|---|---|---|---|---|---|---|---|',baseline]
    if formal:
        result=formal_analysis(audit)
        state['formal']=result['formal'];state['paired']=result['paired'];state['frozen']=result['frozen']
        state['repeat_metrics']=result['repeat_metrics']
        state['mechanisms']=mechanism_analysis(result['groups'],audit)
        state['artifact_registry'].update(result['artifact_registry'])
        for group in CFG['groups']:
            agg=result['formal'][group]
            report.append('|'+group+'|'+'|'.join(f'{agg["mean"][m]*100:.2f} +/- {agg["seed_sd_ddof1"][m]*100:.2f}' for m in ex.METRICS)+'|30/30|')
        report += ['## 正式配对比较','|对照|ΔACC pp|ΔBA pp|ΔAUC pp|CE差|ACC改善折数|','|---|---|---|---|---|---|']
        for name,row in result['paired'].items():
            report.append('|'+name+'|'+'|'.join(f'{row["mean"][m]*100:+.4f}' for m in ('ACC','BA','AUC'))+f'|{row["mean"]["CE_difference"]:+.7f}|{row["paired_fold_wins"]["ACC"]}/30|')
        a=result['formal'];p=result['paired']
        stable_seeds=all(all(row[m]>0 for m in ('ACC','BA','AUC')) for name in ('L3-R0','L3-L2','L3-L4')
            for row in p[name]['seed_differences'].values())
        measured_graph_effect=all(v>1e-5 for v in result['frozen']['L3']['trained_off_probability_MAD_by_seed'].values())
        if stable_seeds and measured_graph_effect and all(a['L3']['mean'][m]>a['L1']['mean'][m] for m in ('ACC','BA','AUC')):
            true_gains=[v<0 for v in result['frozen']['L3']['true_minus_wrong_CE_by_seed'].values()]
            decision='A' if all(true_gains) else 'B'
        elif any(a[g]['mean']['ACC']>a['R0']['mean']['ACC'] for g in ('L1','L2')):decision='B'
        elif a['L3']['mean']['ACC']<a['R0']['mean']['ACC']-.01:decision='D'
        else:decision='C'
        report += ['## 冻结反事实',
            '每个最佳Checkpoint用相同五个历史定义RNG，分别true/off/shuffled；L4的trained对应shuffled，true是该模型的反事实。每次恢复同一核心权重/BN/RNG，允许且恢复RP原位约束，不持续改变任何核心参数或BN。逐draw概率、CE、纠错/误伤、有效错配子集差及指标见summary.frozen。',
            '同Checkpoint关闭图修正不等于历史C0；不同训练权重间差异不是严格单模块因果效应。三seed差值范围是有限重复的不确定性描述，不对30个相关fold做伪独立检验。']
        for g,fr in result['frozen'].items():
            report.append(f'{g} trained-off概率MAD三seed：{fr["trained_off_probability_MAD_by_seed"]}；CE差：{fr["trained_minus_off_CE_by_seed"]}。')
            if g!='L1':report.append(f'{g} 真实-错配CE三seed：{fr["true_minus_wrong_CE_by_seed"]}；概率MAD：{fr["true_wrong_probability_MAD_by_seed"]}。')
        report += ['## 固定五次评估',
            '以下是五次概率平均的辅助结果，不替代正式单次随机测试，不参与Checkpoint或设计选择。跨draw波动与跨seed样本SD在summary.repeat_metrics分开保存。',
            '|组别|概率均值ACC|BA|AUC|五次内患者标签变化率（三seed）|',
            '|---|---|---|---|---|']
        for g,r in state['repeat_metrics'].items():
            v=r['trained'];agg=v['five_probability_mean_metrics']
            report.append('|'+g+'|'+'|'.join(f'{agg["mean"][m]*100:.2f} +/- {agg["seed_sd_ddof1"][m]*100:.2f}' for m in ('ACC','BA','AUC'))+
                '|'+','.join(f'{100*x:.2f}%' for x in v['patient_any_label_change_by_seed'].values())+'|')
        report += ['## 正式机制审计',
            '训练图/梯度采用距最佳epoch最近的已保存训练forward采样，不能伪称最佳Checkpoint的精确图状态；投影有效秩、真实-null JS间隔则由最佳投影独立重算，未增加GTUNet传播或训练。',
            '|组别|最佳投影有效秩中位数|训练真实-null JS gap中位数|测试gap中位数|采样相对图L1中位数|原生→关系梯度中位数|加权局部→投影梯度中位数|',
            '|---|---|---|---|---|---|---|']
        for g,rows in state['mechanisms']['projection_checkpoint_rows'].items():
            vals=[np.median([r['best_projection_rank'] for r in rows]),
                np.median([r['best_projection_true_null_gap_train']['mean'] for r in rows]),
                np.median([r['best_projection_true_null_gap_test']['mean'] for r in rows]),
                np.median([r['sampled_graph_mechanism']['relative_graph_l1'] for r in rows]),
                np.median([r['sampled_gradients']['grad_from_native_to_relation'] for r in rows]),
                np.median([r['sampled_gradients']['grad_from_local_loss_to_projection'] for r in rows])]
            report.append('|'+g+'|'+'|'.join(f'{v:.7g}' for v in vals)+'|')
            within=np.median([r['sampled_graph_mechanism']['within_site_relative_signed_change'] for r in rows])
            cross=np.median([r['sampled_graph_mechanism']['cross_site_relative_signed_change'] for r in rows])
            report.append(f'{g} 站点内/站点间边权有符号相对变化中位数：{within:+.7g}/{cross:+.7g}。未将站点输入关系核，也不据此排除全部站点混杂。')
        report += ['L1/L2没有将局部损失加入总目标；其local梯度读数仅为单独测量的反事实梯度，不能解释成该组被对齐监督训练。局部目标可学习与图传播带来诊断收益是两项不同命题。',
            '## 患者级可靠性与有效错配',
            '可靠性定义为最佳投影的JS(null)-JS(true)，零假设仍只置换训练站点内锚点，查询条件不动。分位阈值由训练患者确定。以下相关系数仅为描述性，不将同一队列的30折当独立队列；不同模型间的CE差不是单模块严格因果效应。']
        for g,r in state['mechanisms']['patient_reliability'].items():
            for seed,entry in r['seeds'].items():
                lo=entry['strata'].get('reliability_quartile/1',{}).get('mean_frozen_CE_benefit')
                hi=entry['strata'].get('reliability_quartile/4',{}).get('mean_frozen_CE_benefit')
                report.append(f'{g} seed{seed}：可靠性与同Checkpoint正确条件CE收益Spearman={entry["reliability_vs_frozen_benefit_spearman"]}；与L3相对L4正式CE收益相关={entry["reliability_vs_cross_trained_benefit_spearman"]}；低/高可靠组正确条件CE收益={lo}/{hi}。')
        fractions={part:[a['inputs']['L4'][part] for a in audit['anchors'].values()] for part in ex.PARTS}
        for part,rows in fractions.items():
            report.append(f'L4 {part}：实际年龄/性别改变比例范围{min(r["changed_fraction"] for r in rows):.4f}–{max(r["changed_fraction"] for r in rows):.4f}；单例人数每折范围{min(r["singleton_subjects"] for r in rows)}–{max(r["singleton_subjects"] for r in rows)}。年龄差和性别改变率逐折见audit。')
        report += ['有效错配子集同时区分“查询自身条件改变”和“其整行候选表型核改变”；即使查询自身未变，训练锚点的条件改变也可影响其关系。分站点、年龄、性别及这两类子集的CE收益见summary.mechanisms.patient_reliability。',
            '纠错与误伤数逐draw逐折保存在summary.frozen.rows，不能把不同随机draw重复计算为独立患者样本。']
    else:
        decision='E'
        for g in CFG['groups']:report.append('|'+g+'|未执行|未执行|未执行|未执行|未执行|未执行|0/30|')
        report += ['不存在L1-L4相对C0的正式ACC/BA/AUC差值、完整OOF、正式五次冻结或同Checkpoint测试反事实；不能由开发80轮验证CE排名替代。',
            '## 机制解释与证据边界',
            f'持续缺少梯度锚点：{gate["failed_gradient_anchors"]}；持续近零图锚点：{gate["near_zero_graph_anchors"]}。如局部投影梯度非零而图效果近零，应区分“辅助目标能优化”与“推理人口图产生足够扰动”；这不是实验7硬q截断的同一种失活。',
            '训练真实-null关系优势、局部熵与投影有效秩、站点/年龄/性别分层、图相对边权和行度均见audit与开发pt；不能仅凭正向训练关系差或者可微性就宣称诊断收益。即使有验证正向迹象，若触发事前图效应安全规则，也不能越过规则盲跑正式任务。',
            '本次只证明预定锚点和80轮窗口内的测量事实，不证明500轮永远无效、所有年龄/性别无生物学信息或方法必然降低最终ACC。未根据外层测试调参，未修改已注册阈值以放行。']
    state['research_decision']=decision
    decisions={'A':'种子方向、正确条件及图直接效应同时支持候选机制，值得另行预注册无泄漏、跨站点确认，仍不足以证明临床泛化。',
        'B':'局部图学习有正向迹象，但未同时建立对齐相对图容量、正确相对错配及跨seed稳定优势，不支持把跨模态对齐作为已验证创新点。',
        'C':'局部关系目标发生可测变化，但未建立稳定诊断增益，不建议继续叠加Gate；优先检验表型信息冗余与图传播敏感性。',
        'D':'整体出现明显分类退化，新增图及联合优化未获得收益；不能仅凭相关采样断定BN或某条梯度是唯一原因。',
        'E':'预检或证据不足，不能提出正式性能提升结论。'}
    report += ['## 研究判断',f'决策：{decision}。'+decisions[decision],
        '历史RFE跨折监督信息污染与全队列转导图仍保留；以后即使出现正向迹象，也需新的无泄漏、跨站点预注册确认。本轮不自动切换协议，也不自动push。',
        '## 执行审计',
        f'存储预检通过写入/fsync/关闭重读/逐字节一致/rename/目录fsync/锁排他。但ossfs不提供强POSIX原子可见性：正式调度出现过替换summary时名称短暂不可见，Worker在任何训练前失败，其余健康Worker正常完成；调度仅对这一精确启动错误有限重试，不吞掉EIO或模型错误。/data虚拟容量不能解释为物理预算。产物位于{BASE}。',
        f'Phase A 5/5锚点，开发20/20，正式{state["formal_completed"]}/120，开发PPU任务{state["development"]["gpu_tasks"]}，正式PPU任务{state["formal_gpu_tasks"]}。CPU线程每worker1，没有CPU训练或DDP。',
        f'累计Worker训练秒{state["worker_train_seconds"]:.3f}；队列启动至验收墙钟{state["wall_seconds_to_acceptance"]:.3f}秒。失败记录{len(state["failures"])}；OOM记录{sum(bool(r["oom_suspected"]) for r in state["failures"])}。',
        f'受保护历史文件SHA复核{len(audit["manifest"])}项通过。代码组合SHA256：{ex.code_hash()}。完整逐文件源SHA、输入、索引、映射、RNG、图/梯度及设备采样见audit.json/summary.json与runs。',
        '恢复机制：任务锁防重，结果以单文件原子保存并含数据/源码/最佳core和新增模块指纹，完成任务按SHA校验跳过。开发结果不复用于正式续训；Go后正式从历史起点重启，以避免不完整RNG/Adam恢复假等价。']
    audit['preflight_acceptance']={'checks':state['preflight']['checks'],'scientific_gate_go':gate['go'],
        'gate_evaluated_utc':gate['evaluated_utc'],'development_tasks':20,'formal_authorized':state['preflight']['formal_authorized']}
    audit['final_protected_history_verification']=state['protected_history_final']
    ex.atomic_json(audit,BASE/'audit.json')
    state['audit_sha256']=ex.sha_file(BASE/'audit.json')
    text='\n\n'.join(report)+'\n'
    ex.atomic_bytes(text.encode(),BASE/'report.md')
    state['report_sha256']=ex.sha_file(BASE/'report.md')
    state['finished_utc']=ex.utc()
    state['status']='research_complete' if formal else 'preflight_scientific_stop'
    state['resume_acceptance']={'passed':True,'artifacts':len(state['artifact_registry']),'all_checksums_valid':True,'neural_retraining_for_resume_test':False}
    handoff='<!-- LCRA_20261018_START -->\n\n'+text+'\n<!-- LCRA_20261018_END -->\n'
    state['handoff']=append_handoff(state,handoff,audit)
    ex.atomic_json(state,BASE/'summary.json')
    verify_terminal(state,audit)
    ex.log('final_acceptance',status=state['status'],formal_completed=state['formal_completed'],decision=decision,handoff=state['handoff']['status'])
    print(text,flush=True)
