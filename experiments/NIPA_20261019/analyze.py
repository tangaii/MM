"""Independent OOF acceptance, factorial contrasts and Chinese research report."""
from __future__ import annotations

import argparse
import collections
from datetime import datetime
import fcntl
import hashlib
import os

import numpy as np
from scipy.special import softmax
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix,roc_auc_score,log_loss,brier_score_loss
import run

KEYS=('ACC','BA','AUC','SEN','SPE','F1','CE','Brier')
PRIMARY=('ACC','BA','AUC','CE')
PAIRS=(('N0','N1'),('N0','N2'),('N0','N4'),('N2','N3'),('N1','N3'),
       ('N4','N5'),('N4','N6'),('N6','N7'),('N5','N7'),('N1','N5'),('N2','N6'),('N3','N7'),('N0','N3'),('N0','N7'))
WEIGHTS={
    'G': [.25,-.25,.25,-.25,.25,-.25,.25,-.25],
    'F': [.25,.25,-.25,-.25,.25,.25,-.25,-.25],
    'R': [.25,.25,.25,.25,-.25,-.25,-.25,-.25],
    'GxF': [.5,-.5,-.5,.5,.5,-.5,-.5,.5],
    'GxR': [.5,-.5,.5,-.5,-.5,.5,-.5,.5],
    'FxR': [.5,.5,-.5,-.5,-.5,-.5,.5,.5],
    'GxFxR': [1,-1,-1,1,-1,1,1,-1]}


def independent(y,p,pred):
    y,p,pred=np.asarray(y),np.asarray(p,np.float64),np.asarray(pred)
    tp,fn,fp,tn=confusion_matrix(y,pred,labels=[0,1]).astype(float).ravel()
    return {'ACC':float((tp+tn)/(tp+fn+fp+tn)), 'BA':float((tp/(tp+fn)+tn/(tn+fp))/2),
            'AUC':float(roc_auc_score(y==0,p)),'SEN':float(tp/(tp+fn)),'SPE':float(tn/(tn+fp)),
            'F1':float(2*tp/(2*tp+fn+fp)),
            'CE':float(log_loss(y==0,np.clip(p,1e-7,1-1e-7),labels=[False,True])),
            'Brier':float(brier_score_loss(y==0,p))}


def summarize(rows):
    per_seed={str(s):{k:float(np.mean([v['metrics'][k] for v in rows if v['seed']==s])) for k in KEYS} for s in run.CFG['seeds']}
    return {'seed_mean':per_seed,**{k:{'mean':float(np.mean([per_seed[str(s)][k] for s in run.CFG['seeds']])),
                                     'sample_sd':float(np.std([per_seed[str(s)][k] for s in run.CFG['seeds']],ddof=1))} for k in KEYS}}


def contrast(rows,weights):
    indexed={g:{(v['seed'],v['fold']):v for v in vs} for g,vs in rows.items()}
    output={}
    for k in PRIMARY:
        folds={f'{s}/{f}':float(sum(weights[j]*indexed[g][s,f]['metrics'][k] for j,g in enumerate(run.CFG['groups'])))
               for s in run.CFG['seeds'] for f in range(10)}
        seeds={str(s):float(np.mean([folds[f'{s}/{f}'] for f in range(10)])) for s in run.CFG['seeds']}
        direction=-1 if k=='CE' else 1
        output[k]={'mean':float(np.mean(list(seeds.values()))),'seed_differences':seeds,
            'seed_sample_sd':float(np.std(list(seeds.values()),ddof=1)),
            'observed_seed_range':[min(seeds.values()),max(seeds.values())],
            'positive_real_direction_seeds':sum(direction*d>1e-12 for d in seeds.values()),
            'positive_real_direction_folds':sum(direction*d>1e-12 for d in folds.values()),
            'fold_differences':folds,'ties_folds':sum(abs(d)<=1e-12 for d in folds.values())}
    return {'weights':dict(zip(run.CFG['groups'],weights)), 'metrics':output,
            'uncertainty':'Observed repeated-CV seed range/sample SD and paired fold distribution, not independent-cohort confidence intervals',
            'interaction_note':'Interaction is a difference of real-minus-null effects; positive is not automatically overall benefit'}


def ce_vector(y,p):
    p=np.clip(np.asarray(p,np.float64),1e-7,1-1e-7)
    return np.where(np.asarray(y)==0,-np.log(p),-np.log1p(-p))


def oof_by_seed(rows):
    output={}
    for s in run.CFG['seeds']:
        p=np.empty(871); pred=np.empty(871,int); y=np.empty(871,int); count=np.zeros(871,int)
        for v in rows:
            if v['seed']!=s:
                continue
            o=v['oof']; i=np.asarray(o['indices']); count[i]+=1
            p[i]=o['probability']; pred[i]=o['prediction']; y[i]=o['labels']
        assert np.all(count==1)
        output[str(s)]={'probability':p,'prediction':pred,'labels':y}
    return output


def patient_pair(aa,bb,meta=None):
    output={}
    for s in aa:
        a,b=aa[s],bb[s]; assert np.array_equal(a['labels'],b['labels'])
        y=a['labels']; ac=a['prediction']==y; bc=b['prediction']==y
        dce=ce_vector(y,a['probability'])-ce_vector(y,b['probability'])
        item={'probability_MAD':float(np.mean(abs(a['probability']-b['probability']))),
              'probability_max_change':float(np.max(abs(a['probability']-b['probability']))),
              'paired_CE_difference':float(dce.mean()), 'rescued':int(np.sum(ac&~bc)),
              'harmed':int(np.sum(~ac&bc)), 'NetRescue':int(np.sum(ac&~bc)-np.sum(~ac&bc)),
              'prediction_changed_fraction':float(np.mean(a['prediction']!=b['prediction']))}
        if meta is not None:
            for name,values in (('site',meta['site']),('sex',meta['sex']),('age_band',np.digitize(meta['age'],[12,18,30]))):
                item[f'by_{name}']={str(v):{'n':int(np.sum(values==v)),
                    'CE_difference':float(dce[values==v].mean()),
                    'NetRescue':int(np.sum(ac[values==v])-np.sum(bc[values==v]))}
                    for v in sorted(np.unique(values))}
        output[s]=item
    return output


def frozen_rows(rows,draw_index):
    groups={g:[] for g in run.CFG['groups']}
    for v in rows['N0']:
        draw=v['frozen']['rows'][draw_index]
        for g in groups:
            condition=draw['conditions'][g]
            groups[g].append({'group':g,'seed':v['seed'],'fold':v['fold'],'metrics':condition['metrics'],
                'oof':{**v['oof'],'probability':condition['probability'],'prediction':condition['prediction'],
                       'logits':condition['logits']}})
    return groups


def frozen_analysis(rows,meta):
    draws={}; contrasts={}; mechanisms={}; comparisons={}
    for j,base in enumerate(run.CFG['frozen_draw_bases']):
        groups=frozen_rows(rows,j)
        draws[str(base)]={g:summarize(vs) for g,vs in groups.items()}
        contrasts[str(base)]={name:contrast(groups,weights) for name,weights in WEIGHTS.items()}
        comparisons[str(base)]={}
        oof={g:oof_by_seed(vs) for g,vs in groups.items()}
        for a,b in PAIRS:
            w=[0.]*8; w[int(a[1])]=1; w[int(b[1])]=-1
            comparisons[str(base)][f'{a}-{b}']={'contrast':contrast(groups,w),'patients':patient_pair(oof[a],oof[b],meta)}
    # Average metrics over all three draws at fold level; do not select the best draw.
    means={g:[] for g in run.CFG['groups']}
    mean_probability={g:[] for g in run.CFG['groups']}
    for v in rows['N0']:
        for g in means:
            conditions=[r['conditions'][g] for r in v['frozen']['rows']]
            m={k:float(np.mean([c['metrics'][k] for c in conditions])) for k in KEYS}
            p=np.stack([c['probability'] for c in conditions]).mean(0)
            pred=np.where(p>=.5,0,1)
            means[g].append({'seed':v['seed'],'fold':v['fold'],'metrics':m})
            mean_probability[g].append({'seed':v['seed'],'fold':v['fold'],
                'metrics':run.binary_metrics(v['oof']['labels'],p,pred),
                'oof':{**v['oof'],'probability':p,'prediction':pred}})
    for g in run.CFG['groups']:
        values=[r['conditions'][g] for v in rows['N0'] for r in v['frozen']['rows']]
        mechanism={}
        for key in ('ph_z','rp_graph','fused_sim','fused_graph','img','ph','joint','graph_loss','graph_attention_weights'):
            mechanism[key]={field:run.describe([v['mechanism'][key][field] for v in values])
                            for field in ('mean_absolute','max_absolute','relative_L1','changed_fraction')}
        mechanism['edge_index_equal_count']=sum(v['mechanism']['actual_edges']['same_index'] for v in values)
        mechanism['intermediate_rng_all_equal_count']=sum(all(v['mechanism']['intermediate_rng_equal'].values()) for v in values)
        mechanism['attention_RNG_unchanged_all']=all(v['mechanism']['rng_extra_attention_has_no_consumption'] for v in values)
        mechanism['core_inplace_drift_forwards']=sum(bool(v['native_inplace_drift']) for v in values)
        mechanism['topk']={name:{'same_global_order_fraction':float(np.mean([p['global_order_equal'] for v in values for p in v['mechanism']['topk'][name]])),
            'mean_global_set_overlap':float(np.mean([p['global_set_overlap'] for v in values for p in v['mechanism']['topk'][name]]))}
            for name in ('img','ph')}
        mechanism['graph_partition_relative_L1']={part:run.describe([v['mechanism']['graph_by_partition'][part]['relative_L1'] for v in values]) for part in run.SUBSETS}
        y=np.concatenate([v['oof']['labels'] for v in rows['N0'] for _ in v['frozen']['rows']])
        p=np.concatenate([r['conditions'][g]['probability'] for v in rows['N0'] for r in v['frozen']['rows']])
        bp=np.concatenate([r['conditions']['N0']['probability'] for v in rows['N0'] for r in v['frozen']['rows']])
        effect=ce_vector(y,p)-ce_vector(y,bp)
        associations={}
        for key in ('img','ph','joint','graph_mean_absolute'):
            x=np.concatenate([v['mechanism']['per_patient_test'][key] for v in values])
            associations[key]=float(spearmanr(x,effect).statistic) if np.std(x)>0 and np.std(effect)>0 else None
        mechanism['pooled_descriptive_path_change_vs_CE_change_spearman']=associations
        mechanism['association_scope']='Repeated subjects/draws, post-hoc descriptive association only, not independent evidence or causal mediation'
        mechanisms[g]=mechanism
    noise={}
    for g in run.CFG['groups']:
        oofs=[oof_by_seed(frozen_rows(rows,j)[g]) for j in range(3)]
        noise[g]={}
        for s in map(str,run.CFG['seeds']):
            probabilities=np.stack([o[s]['probability'] for o in oofs]); predictions=np.stack([o[s]['prediction'] for o in oofs])
            noise[g][s]={'probability_SD_mean':float(probabilities.std(0).mean()),
                'label_changed_fraction':float(np.mean(predictions.min(0)!=predictions.max(0)))}
    return {'draws':draws,'draw_contrasts':contrasts,'draw_pair_comparisons':comparisons,
        'average_draw_metrics':{g:summarize(v) for g,v in means.items()},
        'mean_probability_metrics':{g:summarize(v) for g,v in mean_probability.items()},
        'average_draw_contrasts':{name:contrast(means,w) for name,w in WEIGHTS.items()},
        'mechanisms':mechanisms,'within_checkpoint_draw_noise':noise,
        'same_checkpoint_pairs_mean_probability':{f'{a}-{b}':patient_pair(oof_by_seed(mean_probability[a]),oof_by_seed(mean_probability[b]),meta) for a,b in PAIRS},
        'definition':'30 N0 checkpoints x 3 fixed draws x 8 full forwards; no optimization or checkpoint selection; not retrained N1-N7 model performance'}


def decisions(formal_contrasts,frozen):
    result={}
    for factor,target,group in (('G','fused_graph','N1'),('F','joint','N2'),('R','rp_graph','N4')):
        f=formal_contrasts[factor]['metrics']; z=frozen['average_draw_contrasts'][factor]['metrics']
        tests={'formal_AUC':f['AUC']['mean']>0 and f['AUC']['positive_real_direction_seeds']>=2,
               'formal_BA':f['BA']['mean']>=0 and f['BA']['positive_real_direction_seeds']>=2,
               'formal_CE':f['CE']['mean']<0 and f['CE']['positive_real_direction_seeds']>=2,
               'frozen_AUC':z['AUC']['mean']>0 and z['AUC']['positive_real_direction_seeds']>=2,
               'frozen_CE':z['CE']['mean']<0 and z['CE']['positive_real_direction_seeds']>=2,
               'target_changed':frozen['mechanisms'][group][target]['max_absolute']['max']>0}
        if all(tests.values()):
            status='Evidence-supported'
        elif f['AUC']['mean']<=0 and f['BA']['mean']<=0 and f['CE']['mean']>=0:
            status='Unsupported'
        else:
            status='Weak / Inconclusive'
        result[factor]={'status':status,'registered_descriptive_checks':tests,
            'scope':'Internal historical protocol only; not a p-value, clinical validation, or proof of unique causal mechanism'}
    supported=[k for k in ('G','F','R') if result[k]['status']=='Evidence-supported']
    if 'G' in supported and 'F' not in supported:
        priority='疾病判别驱动的表型辅助患者关系建模，首先在无RFE泄漏和未见站点协议确认'
    elif 'F' in supported and 'G' not in supported:
        priority='保留模态独有信息的选择性融合，首先在无RFE泄漏和未见站点协议确认'
    elif supported==['R']:
        priority='站点监督关系的混杂与跨站点泛化审计，而非继续年龄/性别对齐'
    elif 'G' in supported and 'F' in supported:
        priority='先验证原生图与融合路径协同在无RFE泄漏、未见站点条件下是否保留'
    else:
        priority='影像疾病表示的跨站点稳健性；暂停对当前弱表型继续叠加复杂对齐'
    return {'factors':result,'Go_NoGo':'Exploratory mechanism Go; external confirmation required' if supported else 'No-Go for new complex phenotype alignment',
            'supported_factors':supported,'single_next_priority':priority,
            'rule':run.CFG['evidence_rule'],'interaction_limitation':'Interactions reported descriptively; no significance claim from 30 correlated folds'}


def verify_protected(a):
    changed=[p for p,h in a['protected_sha256'].items() if run.sha_file(p)!=h]
    assert not changed, f'Protected historical inputs changed: {changed}'
    return {'files':len(a['protected_sha256']),'unchanged':True,'utc':run.utc()}


def append_handoff(summary,report):
    path=run.ROOT/'handoff.md'; start='<!-- NIPA_20261019_START -->'; end='<!-- NIPA_20261019_END -->'
    with (run.BASE/'analyze.py').open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        before=run.read_bytes(path)
        if start.encode() in before:
            assert end.encode() in before
            return {'appended':True,'idempotent_skip':True,'after_sha256':run.sha_file(path)}
        a=run.guard()
        assert len(before)==a['handoff_before_bytes'] and run.sha_file(path)==a['handoff_before_sha256'], 'Concurrent handoff edit; preserve it and inspect'
        text='\n\n'+start+'\n\n## 实验13 NIPA 原生非影像路径归因\n\n'+report+'\n'+end+'\n'
        combined=before+text.encode(); pending=run.BASE/'.handoff_pending'
        run.atomic_bytes(combined,pending); os.replace(pending,path)
        fd=os.open(path.parent,os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        after=run.read_bytes(path)
        assert after==combined and after[:len(before)]==before
        return {'appended':True,'time':run.utc(),'before_bytes_preserved':len(before),
                'before_sha256':hashlib.sha256(before).hexdigest(),'after_sha256':hashlib.sha256(after).hexdigest(),
                'old_prefix_identical':True,'marker':start}


def analyze(final=False,stop_reason=None):
    a=run.guard(); ctx=run.load_torch(run.OLD/'artifacts/context.pt'); meta=run.metadata(ctx)
    rows={g:[] for g in run.CFG['groups']}; manifest={}; max_error=0.; frozen_count=0
    for g in rows:
        for s in run.CFG['seeds']:
            coverage=np.zeros(871,int)
            for f in range(10):
                v=run.valid_result(g,s,f,a,required=final and not stop_reason)
                if v is None:
                    continue
                o=v['oof']; idx=o['indices']; coverage[idx]+=1
                assert np.array_equal(o['labels'],ctx['labels'][idx])
                assert o['ids']==[ctx['ids'][int(i)] for i in idx]
                assert np.array_equal(o['prediction'],o['logits'].argmax(1))
                assert np.allclose(softmax(o['logits'].astype(np.float64),axis=1)[:,0],o['probability'],atol=1e-7,rtol=1e-6)
                m=independent(o['labels'],o['probability'],o['prediction'])
                max_error=max(max_error,max(abs(m[k]-v['metrics'][k]) for k in KEYS))
                assert v['new_trainable_parameters']==v['rfe_fit_count']==v['vae_train_count']==0
                assert v['formal_test_grad_enabled'] and v['bn_best_batches']==v['best_epoch']+1
                assert v['formal_start_rng_sha256']==a['anchors'][f'{s}/{f}']['pairing']['classifier_start_rng']
                if g=='N0' and 'frozen' in v:
                    assert v['frozen']['formal_fields_sha256_before']==v['formal_sha256']
                    assert len(v['frozen']['rows'])==3
                    for j,r in enumerate(v['frozen']['rows']):
                        assert r['base']==run.CFG['frozen_draw_bases'][j] and set(r['conditions'])==set(run.CFG['groups'])
                        common=None
                        for condition,c in r['conditions'].items():
                            frozen_count+=1
                            assert c['restored_core_sha256']==v['best_core_sha256']
                            assert common is None or common==c['initial_rng']; common=c['initial_rng']
                            expected=independent(o['labels'],c['probability'],c['prediction'])
                            max_error=max(max_error,max(abs(expected[k]-c['metrics'][k]) for k in KEYS))
                            assert np.array_equal(c['prediction'],c['logits'].argmax(1))
                            assert c['calls']['clf']==c['calls']['vae']==c['calls']['img']==c['calls']['ph']==1
                            assert c['calls']['attention']==(2 if run.CFG['groups'][condition]['F']=='null' else 1)
                            assert c['mechanism']['ph_z']['max_absolute']==0
                elif g=='N0' and final and not stop_reason:
                    raise AssertionError('Missing N0 frozen interventions')
                rows[g].append(v); manifest[f'{g}/{s}/{f}']=run.sha_file(run.result_path(g,s,f))
            if len([v for v in rows[g] if v['seed']==s])==10:
                assert np.all(coverage==1)
    assert max_error<1e-12
    full={g:vs for g,vs in rows.items() if len(vs)==30}; total=sum(map(len,rows.values()))
    assert not final or stop_reason or (total==240 and frozen_count==720)
    if len(full)==8:
        for s in run.CFG['seeds']:
            for f in range(10):
                paired=[next(v for v in full[g] if (v['seed'],v['fold'])==(s,f)) for g in full]
                assert len({v['data_input_digest'] for v in paired})==1
                assert len({tuple(v['optimizer_parameter_order']) for v in paired})==1
                assert len({v['parameter_count'] for v in paired})==1
                assert len({run.digest(v['permutation_plan']) for v in paired})==1
    formal={g:summarize(v) for g,v in full.items()}
    contrasts={name:contrast(full,w) for name,w in WEIGHTS.items()} if len(full)==8 else {}
    oof={g:oof_by_seed(v) for g,v in full.items()}
    pairs={}
    for x,y in PAIRS:
        if len(full)!=8:
            break
        w=[0.]*8; w[int(x[1])]=1; w[int(y[1])]=-1
        pairs[f'{x}-{y}']={'contrast':contrast(full,w),'patients':patient_pair(oof[x],oof[y],meta)}
    frozen=frozen_analysis(full,meta) if frozen_count==720 else {}
    decision=decisions(contrasts,frozen) if contrasts and frozen else {'Go_NoGo':'Incomplete or scientific stop','single_next_priority':'先解决未通过的有效性门槛'}
    protected=verify_protected(a)
    execution=a['execution']; events=execution.get('events',[])
    counts=dict(collections.Counter(str(v['physical_gpu']) for vs in full.values() for v in vs))
    timings={}
    for gpu in (0,1):
        timings[str(gpu)]={}
        for mode in ('train','frozen'):
            e=[v for v in events if v['gpu']==gpu and v['action']==mode]
            if e:
                first=min(datetime.fromisoformat(v['started']) for v in e); last=max(datetime.fromisoformat(v['finished']) for v in e)
                timings[str(gpu)][mode]={'tasks':len(e),'by_group':dict(collections.Counter(v['task'][0] for v in e)),
                    'span_seconds':(last-first).total_seconds(),'first':first.isoformat(),'last':last.isoformat()}
    diagnostics={}
    for g,vs in full.items():
        diagnostics[g]={'best_epoch':run.describe([v['best_epoch'] for v in vs]),'actual_epochs':run.describe([v['epochs_run'] for v in vs]),
            'best_losses':{k:run.describe([v['best_record'][k] for v in vs]) for k in ('train_ce','train_graph','val_ce','val_graph','val_loss')},
            'best_core_change':{prefix:run.describe([v['best_core_change'][prefix]['relative_l2'] for v in vs]) for prefix in ('img_unet.','ph_unet.','mm_attention.','clf.','rp_attention.')},
            'gradient_norms':{key:run.describe([e['gradient_norms'][key] for v in vs for e in v['epochs'] if 'gradient_norms' in e]) for key in ('img','ph','attention','clf','RP')},
            'RP_background':run.describe([v['formal_test_path']['constant_RP_background'] for v in vs]),
            'sample_resolution':'First three and every 25 epochs; complete scalar CE/Graph/validation curves retained in per-fold artifact, no interpolation'}
    def serializable(v):
        if isinstance(v,np.ndarray): return v.tolist()
        if isinstance(v,np.generic): return v.item()
        if isinstance(v,dict): return {str(k):serializable(x) for k,x in v.items()}
        if isinstance(v,(tuple,list)): return [serializable(x) for x in v]
        return v
    protected_manifest_before=manifest.copy()
    # Exercise the same completed-result validator used by the scheduler, with no neural work.
    for g,vs in full.items():
        for v in vs:
            run.valid_result(g,v['seed'],v['fold'],a,True)
    for key,h in protected_manifest_before.items():
        group,seed,fold=key.split('/')
        assert run.sha_file(run.result_path(group,int(seed),int(fold)))==h
    summary={'experiment':run.CFG['experiment'],'status':'stopped' if stop_reason else 'complete' if total==240 and frozen_count==720 else 'partial',
        'stop_reason':stop_reason,'completed_training_tasks':total,'frozen_full_forwards':frozen_count,
        'groups':{g:len(v) for g,v in rows.items()},'formal':formal,'pairs':pairs,'factorial':contrasts,'frozen':frozen,
        'decision':decision,'training_diagnostics':diagnostics,'baseline_gate':a.get('baseline_gate'),
        'formal_OOF':serializable(oof),'fold_results':{g:[{'seed':v['seed'],'fold':v['fold'],'metrics':v['metrics'],
             'best_epoch':v['best_epoch'],'epochs_run':v['epochs_run'],'training_seconds':v['training_seconds']} for v in vs] for g,vs in full.items()},
        'permutation_plans':{k:v['permutation_plan'] for k,v in a['anchors'].items()},
        'raw_RP_diagnostics':{f'{v["seed"]}/{v["fold"]}':v['raw_RP_intervention'] for v in full.get('N0',[])},
        'execution':{'physical_gpu_tasks':counts,'timing':timings,'queues':execution.get('queues',[]),
            'worker_training_seconds':sum(v['training_seconds'] for vs in full.values() for v in vs),
            'formal_epochs':sum(v['epochs_run'] for vs in full.values() for v in vs),
            'single_worker_peak_reserved_GiB':max([v['peak_reserved_bytes']/1024**3 for vs in full.values() for v in vs],default=0),
            'resource_samples':execution.get('resources',[]),'failures':execution.get('failures',[]),
            'RFE_fits':0,'VAE_pretrains':0,'new_parameters':0,'CPU_neural_training':0,'DDP':False,
            'audit_to_analysis_seconds':(datetime.now().astimezone()-datetime.fromisoformat(a['started_utc'])).total_seconds()},
        'audit':{'protected_verification':protected,'independent_metric_max_error':max_error,'complete_OOF_per_seed':871,
            'all_paired_inputs_and_parameter_orders_equal':len(full)==8,'strict_preflight_anchors':len(a['preflight']),
            'resume':{'validated_completed_tasks':total,'artifact_SHA_unchanged':True,'retrained':0,'neural_forwards':0},
            'frozen_formal_fields_preserved':frozen_count==720,'source_and_config_SHA256':a['code_sha256'],
            'source_files':a['source_sha256']},'artifact_SHA256':manifest,'created_utc':run.utc(),'handoff':{'appended':False}}
    report=make_report(summary,a)
    run.atomic_json(summary,run.BASE/'summary.json'); run.atomic_bytes(report.encode(),run.BASE/'report.md')
    if final:
        try:
            summary['handoff']=append_handoff(summary,report)
        except Exception as e:
            summary['handoff']={'appended':False,'error':repr(e),'pending_content':report}
            run.atomic_json(summary,run.BASE/'summary.json'); raise
        summary['final_verified_utc']=run.utc(); summary['report_SHA256']=run.sha_file(run.BASE/'report.md')
        run.atomic_json(summary,run.BASE/'summary.json')
    run.log('analysis_complete',state=summary['status'],training=total,frozen=frozen_count,handoff=summary['handoff'].get('appended'))
    print(report,flush=True)


def make_report(s,a):
    lines=['# 实验13 NIPA：原生非影像路径归因','',f"状态：{s['status']}；正式训练{s['completed_training_tasks']}/240；冻结完整前向{s['frozen_full_forwards']}/720。",'',
        '## 问题与源码审计','',
        '本轮为三因素机制研究，不新增可训练层、对齐Loss、Gate或语义模型。实验1–12没有建立稳定的当前弱表型新增互补，故先定位原生非影像作用路径，而不是以最高ACC设计下一网络。',
        '原论文Table5的联合输入收益包含图、编码与融合，不能代替分路径患者对应检验。[原论文](https://arxiv.org/html/2406.14455v3)；[官方源码](https://github.com/NZWANG/MM-GTUNets)；[当前仓库](https://github.com/tangaii/MM)。',
        '实际原生VAE以SITE/SEX/AGE输出500维并重复加载、重参数采样；RFE500+VAE500以Torch Euclidean cdist核构图。双GTUNet各输出871x16受试者表示；Attention融合影像、表型及均值；Classifier为16->256/ReLU/BN->2。',
        'RP在首个SITE循环内提前return；SEX/AGE原始图为零，但sigmoid(0)=0.5，常量背景是0.5*(wSEX+wAGE)。reward/penalty仅训练标签参与，heldout相关motivation没有站点过滤。因此R不是完整三维人口统计关系，也不自动意味着生物学互补。',
        '保留.data共享卷积约束、未注册RP权重、torch.tensor重建detach、评估Dropout和BN；仅有此前已验收的CustomConv2d末行PPU兼容改动，本轮不修改根源码。',
        f"官方8文件固定commit：{a['official_source_comparison']['commit']}；完整差异在audit。",'',
        '## 设计与一致性','',
        'G-null只将同一次ph_z按perm_G置换用于图相似度，ph_unet仍接收真实ph_z；F-null先计算真实Attention权重供原图正则，再用ph_embed[perm_F]计算分类joint，Classifier一次；R-null只置换原RP builder接收的SITE_ID，真实训练标签、VAE和其他输入不变。',
        'G/F分别在train/val/test内部同站点循环错位，非单例无自映射、单例不动；R在各分区按站点块最大数量循环移位，完全标签盲、双射、频数不变并最大化站点改变。三套私有SHA随机源不消费全局RNG，全epoch固定，全部组共享同一seed/fold映射。',
        'N0直接调用原core.forward。N1 G-null；N2 F-null；N3 G/F-null；N4 R-null；N5 G/R-null；N6 F/R-null；N7全null。N7仍保留非影像边际信息，绝非纯影像模型。',
        '复用DRPA历史30份split/RFE500/VAE/train_start/core/RP/RNG，不重做RFE或VAE；ASD=0/TD=1；Adam1e-4、wd5e-4；最多500epoch、patience100；训练CE+原Graph，val CE+原Graph严格下降选best、相等重置早停计数。正式test恢复best权重但使用训练末RNG，grad-enabled单次随机前向，prob-AUC取softmax第0列。',
        '五固定锚点各连续三步对照官方train AST、N0和identity all_real wrapper：VAE/RP/相似度/人口图/稀疏边/两路GTUNet/Attention/logits/CE/Graph/梯度/Adam/BN/TopK/完整RNG严格等价。8组因素隔离及heldout标签不进入RP的边界检查通过；F-only在确定性诊断中所有上游和Graph Loss严格不变。',
        '确定性仅测试进程使用，正式PPU保留历史原生后端；短步严格等价不能宣称500epoch逐位重放。图干预会同时改变传播和Graph Loss，不是单条边独立因果效应。', '',
        '## 正式八组结果','',
        '单位%；每seed先平均10折，再报告三个seed均值±样本SD。CE/Brier为原量纲。30相关fold、3次重复871人不是独立临床队列。', '',
        '|组别|G/F/R|ACC|BA|prob-AUC|SEN-ASD|SPE-TD|F1-ASD|CE|Brier|Best Epoch|',
        '|---|---|---|---|---|---|---|---|---|---|---|']
    def fmt(v,k):
        x=v[k]; scale=1 if k in ('CE','Brier') else 100; precision=6 if scale==1 else 2
        return f"{scale*x['mean']:.{precision}f} +/- {scale*x['sample_sd']:.{precision}f}"
    for g,spec in run.CFG['groups'].items():
        if g in s['formal']:
            v=s['formal'][g]; epoch=s['training_diagnostics'][g]['best_epoch']['mean']
            lines.append('|'+g+'|'+('/'.join(spec.values()))+'|'+'|'.join(fmt(v,k) for k in KEYS)+f'|{epoch:.2f}|')
        else:
            lines.append(f'|{g}|'+('/'.join(spec.values()))+'|未完成|||||||||')
    if s['baseline_gate']:
        lines+=['','### 新鲜基线核查','',f"N0：{s['baseline_gate']['fresh']}；相对历史C0与实验12 D0：{s['baseline_gate']['comparisons']}。未挑分重跑。"]
    lines+=['','## 正式配对比较','', '|真实方向对照|ΔACC pp|ΔBA pp|ΔAUC pp|ΔCE|AUC胜出seed/折|三seed ΔAUC pp|', '|---|---|---|---|---|---|---|']
    for name,p in s['pairs'].items():
        m=p['contrast']['metrics']; z=m['AUC']
        lines.append('|'+name+'|'+'|'.join(f"{100*m[k]['mean']:+.4f}" for k in ('ACC','BA','AUC'))+
            f"|{m['CE']['mean']:+.6f}|{z['positive_real_direction_seeds']}/3;{z['positive_real_direction_folds']}/30|"+
            '/'.join(f'{100*d:+.4f}' for d in z['seed_differences'].values())+'|')
    lines+=['','全部患者概率MAD、配对CE、rescued/harmed/NetRescue和年龄/性别/站点分层在summary.pairs，属于固定模型事后诊断，不训练患者Gate。','',
        '## 完整析因','',
        '主效应G/F/R是四条对应real-minus-null差的平均。GxF为R两个水平下(N0-N1-N2+N3)对应差的平均；GxR、FxR同理。GxFxR=(N0-N1-N2+N3)-(N4-N5-N6+N7)。主效应ACC/BA/AUC正、CE负支持真实对应；交互符号只是差异中的差异，不直接等于总收益。', '',
        '|效应|重训ΔACC pp|ΔBA pp|ΔAUC pp|ΔCE|三seed ΔAUC pp|冻结ΔAUC pp|冻结ΔCE|','|---|---|---|---|---|---|---|---|']
    for name,p in s['factorial'].items():
        m=p['metrics']; z=s['frozen']['average_draw_contrasts'][name]['metrics']
        lines.append('|'+name+'|'+'|'.join(f"{100*m[k]['mean']:+.4f}" for k in ('ACC','BA','AUC'))+
            f"|{m['CE']['mean']:+.6f}|"+'/'.join(f'{100*d:+.4f}' for d in m['AUC']['seed_differences'].values())+
            f"|{100*z['AUC']['mean']:+.4f}|{z['CE']['mean']:+.6f}|")
    lines+=['','逐fold/seed效应、观察范围及样本SD均在summary.factorial；不是30个独立队列的显著性检验，也不将N0-N7冒充所有交互。','',
        '## 同Checkpoint冻结归因','',
        '30个N0最佳Checkpoint，各3个预注册历史定义RNG draw、8条件，共720完整前向。每条件恢复相同core/BN/未注册RP/完整RNG；没有backward/optimizer，不重复VAE或GTUNet。原生RP前向约束允许瞬时变化但随后严格恢复，不留跨条件污染。正式训练字段SHA保持不变。',
        '冻结均值以下是三个draw各指标的平均，不替代正式单次成绩；同Checkpoint敏感性和重新训练效应分别解释。共享初始RNG不能无条件保证相同随机数落在相同节点：图/TopK顺序或形状可能改变。保存中间RNG、边结构及TopK集合/顺序作为证据。', '',
        '|条件|冻结ACC|BA|AUC|相似度相对L1均值|RP相对L1均值|图相对L1均值|影像表示相对L1|表型表示相对L1|joint相对L1|','|---|---|---|---|---|---|---|---|---|']
    if s['frozen']:
        for g,v in s['frozen']['average_draw_metrics'].items():
            m=s['frozen']['mechanisms'][g]
            lines.append('|'+g+'|'+'|'.join(fmt(v,k) for k in ('ACC','BA','AUC'))+'|'+
                '|'.join(f"{m[k]['relative_L1']['mean']:.7g}" for k in ('fused_sim','rp_graph','fused_graph','img','ph','joint'))+'|')
        lines+=['','### 冻结单路径患者变化','', '|真实-Null|每seed概率MAD|每seed CE差（均值概率）|每seed净救回|','|---|---|---|---|']
        for name in ('N0-N1','N0-N2','N0-N4','N0-N7'):
            p=s['frozen']['same_checkpoint_pairs_mean_probability'][name]
            lines.append('|'+name+'|'+'/'.join(f"{v['probability_MAD']:.6g}" for v in p.values())+'|'+
                         '/'.join(f"{v['paired_CE_difference']:+.6f}" for v in p.values())+'|'+
                         '/'.join(str(v['NetRescue']) for v in p.values())+'|')
        lines+=['','所有draw而非最好draw的指标、配对CE/AUC、纠错误伤、TopK集合/顺序、边权、分区图幅度及重复患者描述性相关均保存于summary.frozen。F-only在生产中若有微小上游数值差，应与原生后端抖动区分；不得以放宽确定性测试掩盖结构差异。',
            f"N0三draw的患者概率/标签噪声：{s['frozen']['within_checkpoint_draw_noise']['N0']}。"]
    lines+=['','## 错配、RP与优化诊断','']
    for factor in ('G','F','R'):
        plans=[v['permutation_plan'][factor]['partitions'] for v in a['anchors'].values()]
        for partition in run.SUBSETS:
            cols={key:run.describe([p[partition][key] for p in plans])['mean'] for key in ('moved_fraction','condition_changed_fraction','mean_absolute_age_change','sex_changed_fraction','site_changed_fraction')}
            singles=sum(p[partition]['singleton_groups'] for p in plans)
            lines.append(f'- {factor}/{partition}：30折均值{cols}，单例组累计{singles}（组次非独立患者）。')
    if s['raw_RP_diagnostics']:
        raw=list(s['raw_RP_diagnostics'].values())
        lines+=['',f"原始RP train-train reward/penalty改变比例折均值：{np.mean([v['train-train']['reward']['changed_fraction'] for v in raw]):.6f}/{np.mean([v['train-train']['penalty']['changed_fraction'] for v in raw]):.6f}；所有heldout相关motivation变化均为0，是原生缺失站点过滤的真实零效应，未修复。",'',
            '|组别|实际epoch均值|best train CE|best train Graph|best val CE|best val Graph|img参数相对变化|Classifier相对变化|','|---|---|---|---|---|---|---|---|']
        for g,d in s['training_diagnostics'].items():
            b=d['best_losses']
            lines.append(f"|{g}|{d['actual_epochs']['mean']:.2f}|{b['train_ce']['mean']:.6f}|{b['train_graph']['mean']:.6f}|{b['val_ce']['mean']:.6f}|{b['val_graph']['mean']:.6f}|{d['best_core_change']['img_unet.']['mean']:.6f}|{d['best_core_change']['clf.']['mean']:.6f}|")
        lines+=['','这些为描述性轨迹，不是隔离后因果参数更新；train/val模式及随机性不同，不能简单把其差视为纯泛化差距。完整标量每epoch保存，图/梯度仅前3及每25epoch采样，不插值伪装完整。']
    lines+=['','## Q1–Q7与研究决策','',f"**{s['decision']['Go_NoGo']}**。"]
    for factor,label in (('G','Q1 图相似度患者配对'),('F','Q2 最终融合患者配对'),('R','Q3 站点RP关系')):
        if factor not in s['decision'].get('factors',{}):
            continue
        m=s['factorial'][factor]['metrics']; z=s['frozen']['average_draw_contrasts'][factor]['metrics']; d=s['decision']['factors'][factor]
        lines.append(f"{label}：{d['status']}。重训real-minus-null AUC {100*m['AUC']['mean']:+.4f}pp（{m['AUC']['positive_real_direction_seeds']}/3 seed支持）、BA {100*m['BA']['mean']:+.4f}pp、CE {m['CE']['mean']:+.6f}；冻结AUC {100*z['AUC']['mean']:+.4f}pp、CE {z['CE']['mean']:+.6f}。")
    lines+=['',
        'Q3解释边界：R把站点与真实训练标签共同构造reward/penalty；即使有效也首先是监督人口关系和站点内协议贡献，不能称可迁移年龄/性别疾病机制。',
        'Q4：以完整主效应与GxF/GxR/FxR/GxFxR表判断数值协同或抵消；符号随seed变化时不宣称稳定协同。原生三路径并非互相隔绝，图干预同时作用于两分支和图损失。',
        'Q5：冻结只度量固定已学模型对对应破坏的直接敏感性，重训允许主干适应和选模改变；方向不一致提示适应/随机轨迹/表示替代，不能唯一断言某条路径的因果重要性。',
        'Q6：只有正式配对多指标、跨seed支持和冻结诊断方向共同成立，才标探索性支持；不能用某组最高ACC或冻结大幅扰动直接证明真实临床互补。三因素判定规则在训练前config冻结，所有弱/负证据保留。',
        f"Q7 唯一下一优先方向：{s['decision']['single_next_priority']}。",'',
        '无论本轮是否见正信号，历史跨折监督RFE泄漏、转导图、原生随机推理和站点内重复CV全部限制临床解释。当前三种null仅测试对应关系而非完全删去非影像，不能将弱效应推广为所有临床模态都无价值。MICCAI正式创新成立仍需独立无泄漏和跨站点确认。', '',
        '## 执行与验收','',f"各组完成数：{s['groups']}；两卡正式任务：{s['execution']['physical_gpu_tasks']}。", 
        f"双卡分阶段实际墙钟及逐组任务：{s['execution']['timing']}。两卡并行时长不可相加当总墙钟。", 
        f"队列墙钟：{s['execution']['queues']}；累计worker训练{s['execution']['worker_training_seconds']:.2f}秒；审计登记至分析{s['execution']['audit_to_analysis_seconds']/60:.2f}分钟（不含此前阅读/编码）。", 
        f"单worker最高保留显存{s['execution']['single_worker_peak_reserved_GiB']:.3f}GiB，整卡峰值由summary资源采样复核；物理每卡98304MiB。失败/重试记录{len(s['execution']['failures'])}；详情audit.execution/run.log。每worker1CPU线程，PPU神经训练，CPU训练/VAE预训练/RFE拟合/新增参数/DDP均0。", 
        f"受保护SHA {s['audit']['protected_verification']['files']}项不变；独立float64指标最大误差{s['audit']['independent_metric_max_error']:.3g}；全部完整组每seed871名患者各一次OOF。240份正式字段SHA在冻结追加前后保持，去重验收不触发神经前向或重训。", 
        f"源码/配置组合SHA256：`{s['audit']['source_and_config_SHA256']}`。输入、排列、源码、环境、PPU资源与PID记录均在audit/summary/单fold产物。", 
        '单fold一个内容SHA封装产物，唯一锁、原子发布/fsync/关闭读回；完成校验后跳过，未完成从历史锚点重启；短暂ENOENT有限重试，EIO不吞掉。N0冻结结果合并保存，没有720份重复模型。', 
        '仅在全部训练、720冻结前向、OOF/统计/源文件保护及报告验收完成后，才幂等安全追加handoff。写入前后SHA和旧字节前缀核对记录在summary.handoff；若失败保留待追加正文并明确失败，不伪报完成。', 
        f"产物：`{run.BASE}`；报告report.md，结构化summary.json，audit.json，config.json，单一run.log和runs。未自动Git commit或push。",'']
    if s['stop_reason']:
        lines+=['## 科学/技术停止','',s['stop_reason'],'未执行任务不填造成绩。']
    return '\n'.join(lines)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--final',action='store_true'); p.add_argument('--stop-reason')
    args=p.parse_args(); analyze(args.final,args.stop_reason)
