"""Independent complete-OOF acceptance, paired evidence and Chinese report."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import time

import numpy as np
from scipy.special import softmax
from sklearn.metrics import confusion_matrix, roc_auc_score, log_loss, brier_score_loss
import run

KEYS=('ACC','BA','AUC','SEN','SPE','F1','CE','Brier')
PAIRS=(('D1','D0'),('D2','D1'),('D3','D1'),('D4','D1'),('D4','D2'),('D4','D3'),
       ('D4','D5'),('D6','D4'),('D6','D7'),('D6','D0'))


def independent(y,p,pred):
    cm=confusion_matrix(y,pred,labels=[0,1]).astype(float)
    tp,fn,fp,tn=cm.ravel()
    sen=tp/(tp+fn);spe=tn/(tn+fp)
    return {'ACC':(tp+tn)/cm.sum(),'BA':(sen+spe)/2,'AUC':roc_auc_score(np.asarray(y)==0,p),
            'SEN':sen,'SPE':spe,'F1':2*tp/(2*tp+fn+fp),
            'CE':log_loss(np.asarray(y)==0,np.clip(p,1e-7,1-1e-7),labels=[False,True]),
            'Brier':brier_score_loss(np.asarray(y)==0,p)}


def summarize(rows,field='metrics'):
    per_seed={str(s):{k:float(np.mean([v[field][k] for v in rows if v['seed']==s])) for k in KEYS} for s in run.CFG['seeds']}
    return {'seed_mean':per_seed, **{k:{'mean':float(np.mean([per_seed[str(s)][k] for s in run.CFG['seeds']])),
                                      'sample_sd':float(np.std([per_seed[str(s)][k] for s in run.CFG['seeds']],ddof=1))} for k in KEYS}}


def paired(rows,a,b):
    aa={(v['seed'],v['fold']):v for v in rows[a]};bb={(v['seed'],v['fold']):v for v in rows[b]}
    out={}
    for k in ('ACC','BA','AUC','CE'):
        seed={str(s):float(np.mean([aa[s,f]['metrics'][k]-bb[s,f]['metrics'][k] for f in range(10)])) for s in run.CFG['seeds']}
        diffs=[aa[s,f]['metrics'][k]-bb[s,f]['metrics'][k] for s in run.CFG['seeds'] for f in range(10)]
        direction=-1 if k=='CE' else 1
        out[k]={'mean':float(np.mean(list(seed.values()))),'seed_differences':seed,
                'observed_seed_range':[min(seed.values()),max(seed.values())],
                'sample_sd_seed_difference':float(np.std(list(seed.values()),ddof=1)),
                'improved_seeds':sum(direction*x>1e-12 for x in seed.values()),
                'improved_folds':sum(direction*x>1e-12 for x in diffs), 'ties_folds':sum(abs(x)<=1e-12 for x in diffs)}
    patient={}
    for s in run.CFG['seeds']:
        av=[aa[s,f] for f in range(10)];bv=[bb[s,f] for f in range(10)]
        ap=np.zeros(871);bp=ap.copy();y=np.zeros(871,int)
        for x,z in zip(av,bv):
            i=x['oof']['indices'];assert np.array_equal(i,z['oof']['indices'])
            ap[i]=x['oof']['probability'];bp[i]=z['oof']['probability'];y[i]=x['oof']['labels']
        ac=(np.where(ap>=.5,0,1)==y);bc=(np.where(bp>=.5,0,1)==y)
        patient[str(s)]={'probability_mad':float(np.mean(abs(ap-bp))),
                         'paired_oof_CE':independent(y,ap,np.where(ap>=.5,0,1))['CE']-independent(y,bp,np.where(bp>=.5,0,1))['CE'],
                         'rescued':int((ac&~bc).sum()),'harmed':int((~ac&bc).sum())}
    return {'metrics':out,'patients':patient,'uncertainty':'observed three repeated-CV seed range and sample SD, not confidence intervals or independent external cohorts'}


def mechanism(rows):
    out={}
    for g,values in rows.items():
        sampled=[e for v in values for e in v['epochs'] if 'prototype' in e]
        late=[e for e in sampled if e['epoch']>=9]
        proto=[e['prototype'] for e in late]
        stats={}
        for k in ('conditional_global_l2','class_delta_l2','ess_min','beta','r','r_site'):
            stats[k+'_sampled_means']=run.describe([p[k]['mean'] for p in proto])
        for k in ('r_fallback_fraction','beta_zero_fraction','class_delta_query_sd_l2'):
            stats[k]=run.describe([p[k] for p in proto])
        ratios=[e['weighted_z_gradient_ratio'] for v in values for e in v['epochs'] if 'weighted_z_gradient_ratio' in e and e['epoch']>=9]
        stats['weighted_z_gradient_ratio']=run.describe(ratios)
        stats['image_parameter_gradient_ratio']=run.describe([e['image_parameter_gradients']['ratio'] for e in late if 'image_parameter_gradients' in e])
        stats['image_parameter_aux_norm']=run.describe([e['image_parameter_gradients']['aux_weighted'] for e in late if 'image_parameter_gradients' in e])
        stats['r_CV']=run.describe([p['r']['cv'] for p in proto])
        stats['best_epoch']=run.describe([v['best_epoch'] for v in values])
        stats['epochs_run']=run.describe([v['epochs_run'] for v in values])
        stats['posthoc_test_image_margin']=run.describe(np.concatenate([v['posthoc_image_margin'] for v in values]))
        stats['posthoc_test_image_prototype_AUC']=run.describe([roc_auc_score(v['oof']['labels']==0,v['posthoc_image_prototype_probability']) for v in values])
        stats['image_parameter_change']=run.describe([v['best_core_change']['img_unet.']['relative_l2'] for v in values])
        stats['sampling_resolution']='prototype and parameter gradients first 3 epochs then every 25 epochs; not interpolated; z-gradient ratio recorded every auxiliary epoch'
        stats['effective_condition_changes']=run.describe([v['prototype_preparation']['shuffle_changed_fraction'] for v in values])
        # Save group-level class/sex/site/age reliability, preserving the per-fold values in artifacts.
        for field in ('class','sex','site','age'):
            keys=sorted(set(k for p in proto for k in p[f'r_by_{field}']))
            stats[f'r_by_{field}']={key:run.describe([p[f'r_by_{field}'][key]['mean'] for p in proto if key in p[f'r_by_{field}']]) for key in keys}
        out[g]=stats
    return out


def verify_protected(a):
    failures=[]
    for p,h in a['protected_sha256'].items():
        if run.sha_file(p)!=h:
            failures.append(p)
    if failures:
        raise AssertionError(f'Protected historical files changed: {failures}')
    return {'files':len(a['protected_sha256']),'unchanged':True,'time':run.utc()}


def append_handoff(summary,report):
    start='<!-- PCDCA_20261019_START -->';end='<!-- PCDCA_20261019_END -->'
    path=run.ROOT/'handoff.md'
    with (run.BASE/'analyze.py').open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        before=run.read_bytes(path)
        if start.encode() in before:
            assert end.encode() in before
            return {'appended':True,'idempotent_skip':True,'after_sha256':run.sha_file(path)}
        a=run.read_json(run.BASE/'audit.json')
        assert len(before)==a['handoff_before_bytes'] and run.sha_file(path)==a['handoff_before_sha256'], 'handoff changed concurrently; preserve it and review before append'
        text='\n\n'+start+'\n\n## Experiment 12 PC-DCA / PCDCA_20261019\n\n'+report+'\n\n'+end+'\n'
        data=before+text.encode()
        pending=run.BASE/'.handoff_pending'
        run.atomic_bytes(data,pending)
        os.replace(pending,path)
        fd=os.open(path.parent,os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        after=run.read_bytes(path)
        assert after==data and after[:len(before)]==before
        import hashlib
        assert hashlib.sha256(after[:len(before)]).hexdigest()==a['handoff_before_sha256']
        return {'appended':True,'before_sha256':a['handoff_before_sha256'],'before_bytes_preserved':len(before),
                'after_sha256':hashlib.sha256(after).hexdigest(),'marker':start,'time':run.utc()}


def main(final=False,stop_reason=None):
    a=run.guard()
    rows={g:[] for g in run.CFG['groups']}
    independent_max=0.;counts={0:0,1:0};peak=0;seconds=0.
    ctx=run.load_torch(run.OLD/'artifacts/context.pt')
    artifact_manifest={}
    for g in rows:
        for s in run.CFG['seeds']:
            coverage=np.zeros(871,int)
            for f in range(10):
                v=run.valid_result(g,s,f,a,required=final and stop_reason is None)
                if v is None:
                    continue
                if final and stop_reason is None and g in run.CFG['frozen_groups']:
                    assert 'frozen' in v and len(v['frozen']['rows'])==5
                idx=v['oof']['indices'];coverage[idx]+=1
                assert np.array_equal(v['oof']['labels'],ctx['labels'][idx])
                assert v['oof']['ids']==[ctx['ids'][int(i)] for i in idx]
                assert np.array_equal(v['oof']['prediction'],v['oof']['logits'].argmax(1))
                assert np.allclose(softmax(v['oof']['logits'].astype(np.float64),1)[:,0],v['oof']['probability'],atol=1e-7,rtol=1e-6)
                m=independent(v['oof']['labels'],v['oof']['probability'],v['oof']['prediction'])
                independent_max=max(independent_max,max(abs(m[k]-v['metrics'][k]) for k in KEYS))
                assert v['bn_best_batches']==v['best_epoch']+1
                assert v['physical_gpu']==f%2 and v['new_trainable_parameters']==0
                assert v['rfe_fit_count']==v['vae_train_count']==0
                assert v['formal_test_grad_enabled']
                assert v['formal_start_rng_sha256']==a['anchors'][f'{s}/{f}']['pairing']['classifier_start_rng']
                counts[v['physical_gpu']]+=1;seconds+=v['training_seconds'];peak=max(peak,v['peak_reserved_bytes'])
                rows[g].append(v)
                artifact_manifest[f'{g}/{s}/{f}']=run.sha_file(run.result_path(g,s,f))
            if len([v for v in rows[g] if v['seed']==s])==10:
                assert np.all(coverage==1)
            elif final and stop_reason is None:
                raise AssertionError(f'Incomplete OOF {g}/{s}')
    assert independent_max<1e-12
    full={g:v for g,v in rows.items() if len(v)==30}
    formal={g:summarize(v) for g,v in full.items()}
    pairs={f'{x}-{y}':paired(full,x,y) for x,y in PAIRS if x in full and y in full}
    mechanisms=mechanism(full)
    frozen={}
    for g,v in full.items():
        if all('frozen' in x for x in v):
            temp=[{**x,'mean_metrics':x['frozen']['mean_probability_metrics']} for x in v]
            frozen[g]={'mean_probability':summarize(temp,'mean_metrics'),
                       'label_flip_fraction':run.describe([x['frozen']['label_flip_fraction'] for x in v]),
                       'draws':{str(b):summarize([{**x,'draw_metrics':x['frozen']['rows'][j]['metrics']} for x in v],'draw_metrics')
                                for j,b in enumerate(run.CFG['frozen_draw_bases'])},
                       'selection':'all five fixed historical RNG draws; no auxiliary mechanism at inference, no best-draw selection'}
    go={}
    for g,other in (('D4','D5'),('D6','D7')):
        if all(k in formal for k in (g,other,'D0','D1')):
            p0=paired(full,g,'D0');p1=paired(full,g,'D1');pn=paired(full,g,other)
            # Direction must hold in the same primary AUC metric, not cherry-picked ACC.
            rules={'auc_vs_D0_at_least_0_5pp':p0['metrics']['AUC']['mean']>=.005,
                   'auc_vs_D0_two_seeds':p0['metrics']['AUC']['improved_seeds']>=2,
                   'auc_vs_global_positive_two_seeds':p1['metrics']['AUC']['mean']>0 and p1['metrics']['AUC']['improved_seeds']>=2,
                   'auc_vs_shuffle_positive_two_seeds':pn['metrics']['AUC']['mean']>0 and pn['metrics']['AUC']['improved_seeds']>=2,
                   'CE_not_worse':p0['metrics']['CE']['mean']<=0,
                   'BA_not_worse':p0['metrics']['BA']['mean']>=0,
                   'conditional_not_global':mechanisms[g]['conditional_global_l2_sampled_means']['mean']>1e-8,
                   'aux_parameter_gradient_present':mechanisms[g]['image_parameter_aux_norm']['min']>0}
            go[g]={'rules':rules,'exploratory_continue':all(rules.values())}
    decision='Scientific/technical stop' if stop_reason else ('Exploratory Go' if any(v['exploratory_continue'] for v in go.values()) else 'No-Go')
    protected=verify_protected(a)
    total=sum(map(len,rows.values()))
    summary={'experiment':run.CFG['experiment'],'status':'complete' if total==240 and not stop_reason else 'stopped' if stop_reason else 'partial',
             'completed_tasks':total,'group_counts':{g:len(v) for g,v in rows.items()},'formal':formal,'pairs':pairs,
             'mechanisms':mechanisms,'frozen':frozen,'decision':decision,'decision_checks':go,'stop_reason':stop_reason,
             'baseline_gate':a.get('baseline_gate'),'protected_verification':protected,'independent_metric_max_error':independent_max,
             'artifact_sha256':artifact_manifest,'code_sha256':a['code_sha256'],'input_context_sha256':a['context_sha256'],
             'execution':{'physical_gpu_tasks':{str(k):v for k,v in counts.items()},'total_worker_training_seconds':seconds,
                          'single_worker_peak_reserved_gib':peak/1024**3,'queue_walls':a['execution'].get('queues',[]),
                          'failures':a['execution'].get('failures',[]),'resource_samples':a['execution'].get('resources',[]),
                          'formal_neural_training':'PPU only','RFE_fits':0,'VAE_pretrains':0,'new_parameters':0},
             'OOF':{g:{str(s):{'subjects':871,'each_once':True} for s in run.CFG['seeds']} for g in full},
             'statistics':'10-fold mean per seed, then three-seed mean and sample SD; paired folds and seeds are correlated repeated CV',
             'created_utc':run.utc(),'handoff':{'appended':False}}
    report=make_report(summary,a)
    run.atomic_bytes(report.encode(),run.BASE/'report.md')
    run.atomic_json(summary,run.BASE/'summary.json')
    if final:
        try:
            summary['handoff']=append_handoff(summary,report)
        except Exception as e:
            summary['handoff']={'appended':False,'error':repr(e),'pending_append_content':report}
            run.atomic_json(summary,run.BASE/'summary.json')
            raise
        summary['final_verified_utc']=run.utc()
        summary['report_sha256']=run.sha_file(run.BASE/'report.md')
        run.atomic_json(summary,run.BASE/'summary.json')
    run.log('analysis_complete',status=summary['status'],tasks=total,decision=decision,handoff=summary['handoff'].get('appended'))
    print(report,flush=True)


def make_report(s,a):
    lines=['# 实验12：PC-DCA疾病判别驱动的表型条件监督','',
           f"状态：{s['status']}；正式任务 {s['completed_tasks']}/240；研究判断：**{s['decision']}**。",'',
           '## 问题、历史证据与创新边界','',
           '本轮只检验疾病类别原型与正确年龄/性别条件结合，能否超越全局类别原型及错误条件。原始分类路径完全不变，没有新网络、残差头、文本模型或图边修正。',
           'DRPA强制InfoNCE及门控未确立稳定收益；PCCI残差明显退化且正确对应证据弱；PCND发育偏差未证明增量；LCRA学到局部关系但推理图效应极弱；CMCV轻量探针没有稳定AUC/CE互补，但不证明深度条件互补不存在；GBSA身份InfoNCE未形成可靠患者匹配。',
           '原论文RFE/VAE首先统一维度，AMRS结合表型与影像构图，双GTUNet各输出871×16受试者表示，注意力融合两模态及算术均值，再经16→256/ReLU/BN→2分类。Table 5的ABIDE影像/联合ACC为76.49/82.92%，不等于独立对齐增益。[MM-GTUNets](https://arxiv.org/html/2406.14455v3)。',
           'PhenoBrain使用表型条件长程通信与注意力，不等于本轮训练期疾病原型。[PhenoBrain ICML2026](https://proceedings.mlr.press/v306/meng26f.html)。MGDR分离共享/私有多模态表示并融合，不是本轮条件类别均值。[MGDR MICCAI2024](https://papers.miccai.org/miccai-2024/paper/2494_paper.pdf)。',
           'To Align or Not to Align研究不同冗余/独有信息结构下的对齐强度，并不支持普遍强制对齐。[AAAI2026论文](https://arxiv.org/html/2511.12121v1)。WSSADN讨论结构MRI中的年龄/疾病解耦，区别于fMRI人口图训练期条件原型。[WSSADN MICCAI2024](https://papers.miccai.org/miccai-2024/paper/0336_paper.pdf)。',
           '类别均值及距离softmax已有经典先例，不能声称首次提出类别原型或表型条件化。[Prototypical Networks](https://papers.nips.cc/paper/2017/file/cb8da6767461f2812ae4290eac7cbc42-Paper.pdf)。本轮不证明严格共享/私有因子可识别。','',
           '## 实现、协议与验收','',
           '独立运行器直接调用原core；一次只读img_unet hook捕获当次影像输出，训练期间一次分类器调用和一次BN更新。原生VAE每次重载并随机采样、eval Dropout、RP循环内提前return、共享卷积.data投影、未注册RP权重及tensor重建detach均保留。与官方8份源码比对，只有既有CustomConv2d末行PPU算子替换。',
           '871人、ASD=0/TD=1、3seed×10折；精确复用历史划分/RFE500/VAE/训练起点/core/RP/RNG；Adam lr1e-4、wd5e-4，500epoch，patience100。val仅原CE+Graph选模，严格下降更新best、相等重置早停计数。正式测试恢复best core，不恢复best RNG，用训练末尾RNG做一次grad-enabled随机前向。',
           '原型只读取train。私有3折近似类别平衡且兼顾站点；每查询参考为另两子折，无自身参考。teacher/stat/kernel/prototype/ESS/beta/r全部detach；只沿查询z反传。这不是独立OOF深度Embedding，也没有消除历史RFE泄漏或转导图。',
           '年龄带宽max(2,IQR(train_age)/2)；缺失/无效年龄仅训练中位数填补，全部缺失固定17岁并记录；CSV性别1男2女与context及论文性别计数核对，同/异/未知核1/.5/.75。年龄性别联合错配只在train内同站点双射，非单例全错位、单例不动，完全不读取标签决定排列。',
           '每类分别归一核权重，ESS控制beta=clip((minESS-8)/32,0,1)，局部原型向同一参考集全局原型收缩。辅助坐标用train detach均值及总体SD下限.05；tau精确定义为标准化教师到训练中心的平均平方L2，下限1，所有组一致。',
           '可靠性按整个站点划分两侧；每侧每类至少2人且ESS≥2、方向范数>1e-8，否则r=.25。有效时r=clip((1+cos)/2*min(1,minESS/20),.25,1)。站点划分不交叉，权重detach；近常数不得称选择成功。',
           'lambda=clip(.05*||grad(native,z)||/(||grad(aux,z)||+1e-12),0,.5)*min(epoch/9,1)，epoch0为0，epoch9全量。每轮记录实际加权z梯度比，另按25epoch采样影像参数梯度，不把两者等同。',
           f"五锚点通过数：{sum(v.get('passed',False) for v in a['preflight'].values())}/5；官方train函数AST逐字执行与D0/aux=0连续三步比较Logits/CE/Graph/梯度/Adam/BN/TopK/完整RNG。测试专用确定性不进入正式训练。已保存持出标签/表型/Embedding扰动、无自参考、未知性别/单例/空支持/常量Embedding/零梯度回退与数值差分证据。",'',
           '## 正式结果','',
           '单位%；每seed先平均十折，再三个seed均值±样本SD。正式主表为单次原生随机测试，不使用五次均值替换。','',
           '|组别|做法|完成|ACC|BA|prob-AUC|SEN-ASD|SPE-TD|F1-ASD|CE|Brier|',
           '|---|---|---|---|---|---|---|---|---|---|---|']
    names={'D0':'新鲜原生基线','D1':'全局类别原型','D2':'年龄条件','D3':'性别条件','D4':'年龄+性别条件','D5':'D4同站点错配','D6':'D4+站点可靠性','D7':'D6同站点错配'}
    def fmt(v,k):
        x=v[k];factor=100 if k not in ('CE','Brier') else 1
        return f"{factor*x['mean']:.2f} +/- {factor*x['sample_sd']:.2f}" if factor==100 else f"{x['mean']:.6f} +/- {x['sample_sd']:.6f}"
    for g in run.CFG['groups']:
        v=s['formal'].get(g)
        lines.append(f"|{g}|{names[g]}|{s['group_counts'][g]}/30|"+'|'.join(fmt(v,k) if v else '未完成' for k in KEYS)+'|')
    lines+=['','### 相对新鲜D0','', '|组别|ΔACC pp|ΔBA pp|ΔAUC pp|','|---|---|---|---|']
    if 'D0' in s['formal']:
        for g,v in s['formal'].items():
            lines.append('|'+g+'|'+'|'.join(f"{100*(v[k]['mean']-s['formal']['D0'][k]['mean']):+.4f}" for k in ('ACC','BA','AUC'))+'|')
    lines+=['','### 关键配对','', '|比较|ΔACC pp|ΔBA pp|ΔAUC pp|ΔCE|AUC改善seed/折|三seed ΔAUC pp|','|---|---|---|---|---|---|---|']
    for name,p in s['pairs'].items():
        m=p['metrics'];au=m['AUC']
        lines.append('|'+name+'|'+'|'.join(f"{100*m[k]['mean']:+.4f}" for k in ('ACC','BA','AUC'))+
                     f"|{m['CE']['mean']:+.7f}|{au['improved_seeds']}/3; {au['improved_folds']}/30|"+
                     '/'.join(f'{100*x:+.4f}' for x in au['seed_differences'].values())+'|')
    lines+=['','每项ACC/BA/AUC/CE逐seed、改善折数、seed差值范围及样本SD、逐患者概率MAD/配对CE/纠错误伤保存在summary.pairs；三seed观察范围不是置信区间，30相关折不是30独立临床队列。','',
            '## 机制','', '|组别|条件-全局原型L2|条件类别差异query SD L2|ESS min均值|beta均值|r均值/CV|r回退比例|实测加权z梯度比|影像参数梯度比|','|---|---|---|---|---|---|---|---|---|']
    def val(v):
        return 'NA' if v is None else f'{v:.6g}'
    for g,m in s['mechanisms'].items():
        lines.append('|'+g+'|'+'|'.join(val(m[k]['mean']) for k in ('conditional_global_l2_sampled_means','class_delta_query_sd_l2','ess_min_sampled_means','beta_sampled_means'))+
                     '|'+val(m['r_sampled_means']['mean'])+'/'+val(m['r_CV']['mean'])+'|'+val(m['r_fallback_fraction']['mean'])+
                     '|'+val(m['weighted_z_gradient_ratio']['mean'])+'|'+val(m['image_parameter_gradient_ratio']['mean'])+'|')
    lines+=['','这些是训练采样分布的汇总，不冒充每epoch完整原型轨迹。条件原型相对全局的改变、类差异随query变化、真实/错配对照、ESS/beta/r、分年龄/性别/站点/诊断可靠性及影像参数变化均可在单fold产物与summary复核。',
            '最终测试影像类别间隔与训练类中心距离排序为事后诊断，未用于选模或可靠性。未保存大型Embedding/图副本；原型测量只访问train，测试标签仅用于最终指标和事后间隔。','',
            '## 固定五次推理','', '|组别|均值概率ACC|BA|AUC|患者标签变化比例|','|---|---|---|---|---|']
    for g,v in s['frozen'].items():
        lines.append('|'+g+'|'+'|'.join(fmt(v['mean_probability'],k) for k in ('ACC','BA','AUC'))+f"|{100*v['label_flip_fraction']['mean']:.3f}%|")
    lines+=['','五个历史固定RNG全部使用，不挑最好draw。跨draw指标与概率SD保存在summary/单fold；每次恢复同一best core、BN和未注册RP，检查只允许原生RP前向投影瞬时变化。PC-DCA不参与推理，同Checkpoint切换辅助条件不会产生合法预测反事实；真实条件贡献必须看D4-D5、D6-D7配对训练，且保留原生随机轨迹限制。','',
            '## 研究判断','',f"**{s['decision']}**。逐项预注册Go规则见summary.decision_checks。必须同时超过新鲜D0、全局D1、对应错配组，并有AUC/CE/BA与机制支持，不以单一最高ACC判成功。"]
    if s['stop_reason']:
        lines+=['',f"停止原因：{s['stop_reason']}。剩余任务未运行，不填造成绩。"]
    elif s['decision']=='No-Go':
        lines+=['','当前设计不满足预注册条件互补继续门槛，不建议直接作为MICCAI 2027第一个创新点扩展，也不自动增加Gate或调lambda。若仅D1有收益，应限定为类别原型正则化，而非正确年龄/性别条件跨模态作用。下一步唯一优先方向是独立无监督RFE污染、含未见站点的疾病判别监督确认，而非继续增加当前条件融合复杂度。']
    else:
        lines+=['','仅属探索性继续信号，不是临床显著性或创新成立证明。下一步优先在独立无RFE跨折泄漏、未见站点协议中确认本轮固定方法和条件负对照，不能在本轮改协议追求更高成绩。']
    e=s['execution']
    lines+=['','## 执行与保护','',f"实际正式任务{s['completed_tasks']}/240，两卡{e['physical_gpu_tasks']}；累计worker训练{e['total_worker_training_seconds']:.2f}秒；队列墙钟明细{e['queue_walls']}。单worker最高保留显存{e['single_worker_peak_reserved_gib']:.3f}GiB，不等于多进程整卡峰值。", 
            f"正式失败记录{len(e['failures'])}；资源、PID、命令、重试/故障明细见audit.execution/run.log。CPU每worker1线程，神经训练只在PPU；RFE拟合0、VAE预训练0、新可训练参数0，没有DDP。", 
            f"受保护文件{s['protected_verification']['files']}项SHA保持，独立指标最大差{s['independent_metric_max_error']:.3g}；完整组每seed871人恰好一次OOF。代码组合SHA `{s['code_sha256']}`。", 
            '完成产物先校验内容SHA、输入/代码指纹、core/RP与OOF再跳过；未完成任务从原始锚点重启，没有不完整的epoch续训。单文件原子保存、fsync、读回验证、唯一任务锁；仅暂时ENOENT有限重试，EIO不当作成功。',
            '全部验收后才幂等追加handoff；准确状态与前后SHA见summary.handoff。保留历史字节前缀，不自动git提交或push。',
            f"产物目录：`{run.BASE}`。",'',
            '本轮保留历史监督RFE跨折泄漏、871节点转导式图、随机VAE/eval Dropout及PPU原生非逐位重放；只属于同协议内部比较，绝不是无泄漏临床泛化性能。','']
    return '\n'.join(lines)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');parser.add_argument('--stop-reason')
    args=parser.parse_args();main(args.final,args.stop_reason)
