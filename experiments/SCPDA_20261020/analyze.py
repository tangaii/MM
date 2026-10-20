"""Independent fold acceptance, paired comparisons and safe Chinese handoff."""
from __future__ import annotations
import argparse
import csv
import fcntl
import hashlib
import io
import os
from pathlib import Path
import numpy as np
from scipy.special import softmax
from sklearn.metrics import confusion_matrix,roc_auc_score,log_loss,brier_score_loss
import run

KEYS=('ACC','SEN','SPE','AUC','BA','F1','CE','Brier')
PAIRS=(('E1','E0'),('E2','E1'),('E3','E1'),('E4','E1'),('E4','E2'),('E4','E3'),('E4','E5'),('E4','E0'))


def independent(y,p,pred):
    y=np.asarray(y);p=np.asarray(p,np.float64);pred=np.asarray(pred)
    tp,fn,fp,tn=confusion_matrix(y,pred,labels=[0,1]).astype(float).ravel()
    sen=tp/(tp+fn);spe=tn/(tn+fp)
    return {'ACC':(tp+tn)/(tp+fn+fp+tn),'SEN':sen,'SPE':spe,'BA':(sen+spe)/2,
            'AUC':roc_auc_score(y==0,p),'F1':2*tp/(2*tp+fn+fp),
            'CE':log_loss(y==0,np.clip(p,1e-7,1-1e-7),labels=[False,True]),
            'Brier':brier_score_loss(y==0,p)}


def verify_protected(a):
    for path,sha in a['protected_sha256'].items():
        assert run.sha_file(path)==sha,f'Historical file changed: {path}'
    return {'files':len(a['protected_sha256']),'unchanged':True,'utc':run.utc()}


def paired(rows,a,b):
    aa={v['fold']:v for v in rows[a]};bb={v['fold']:v for v in rows[b]}
    out={}
    for k in KEYS:
        d=[aa[f]['metrics'][k]-bb[f]['metrics'][k] for f in range(10)]
        out[k]={'mean':float(np.mean(d)),'sample_sd':float(np.std(d,ddof=1)),'fold_differences':d,
                'improved_folds':sum(x>1e-12 for x in d) if k not in ('CE','Brier') else sum(x < -1e-12 for x in d),
                'tied_folds':sum(abs(x)<=1e-12 for x in d)}
    return out


def mechanism(values):
    sampled=[r['prototype'] for v in values for r in v['epochs'] if 'prototype' in r]
    ratios=[r['weighted_z_gradient_ratio'] for v in values for r in v['epochs'] if 'weighted_z_gradient_ratio' in r and r['epoch']>=9]
    return {'ema_updates_by_fold':{str(v['fold']):v['ema_updates'] for v in values},
            'ema_current_l2':run.describe([p['ema_current_l2'] for p in sampled]),
            'beta_by_class_sampled_mean':None if not sampled else np.mean([p['beta_by_class'] for p in sampled],axis=0).tolist(),
            'ess_by_class_sampled_mean':None if not sampled else np.mean([p['ess_by_class'] for p in sampled],axis=0).tolist(),
            'weighted_query_gradient_ratio':run.describe(ratios),
            'lambda_eff':run.describe([r['lambda_eff'] for v in values for r in v['epochs'] if 'lambda_eff' in r]),
            'invalid_query_fraction':run.describe([p['invalid_prototype_fraction'] for p in sampled]),
            'sampling':'first 3 epochs and every 25 epochs; gradient scaling recorded each auxiliary epoch',
            'epochs_run':run.describe([v['epochs_run'] for v in values]),
            'best_epoch':run.describe([v['best_epoch'] for v in values])}


def decision(pairs):
    if 'E4-E1' not in pairs:return {'outcome':'Incomplete','criteria':{}}
    d=pairs['E4-E1']
    rules={'ACC_vs_D4_positive':d['ACC']['mean']>0,'AUC_vs_D4_positive':d['AUC']['mean']>0,
           'ACC_improved_at_least_6_folds':d['ACC']['improved_folds']>=6,
           'AUC_improved_at_least_6_folds':d['AUC']['improved_folds']>=6,
           'SEN_not_lower_vs_D4':d['SEN']['mean']>=0,
           'ACC_true_vs_shuffle_positive':pairs['E4-E5']['ACC']['mean']>0,
           'AUC_true_vs_shuffle_positive':pairs['E4-E5']['AUC']['mean']>0,
           'combined_AUC_not_lower_than_EMA':pairs['E4-E2']['AUC']['mean']>=0,
           'combined_AUC_not_lower_than_ESS':pairs['E4-E3']['AUC']['mean']>=0}
    outcome='Positive' if all(rules.values()) else 'Negative' if d['ACC']['mean']<=0 and d['AUC']['mean']<=0 else 'Mixed'
    return {'outcome':outcome,'criteria':rules,'interpretation':'Pre-registered descriptive decision, not statistical significance or clinical generalization'}


def report(s):
    lines=['# SCPDA_20261020：EMA稳定原型与类别独立支持度扩展','',
           f"状态：{s['status']}；完成{s['completed_tasks']}/60；研究判断：**{s['decision']['outcome']}**。",'',
           '## 研究背景与真实基线','',
           '实验12 seed911 D4的ACC/AUC为82.5588/88.6897%，D2为82.5601/88.5632%；D4没有建立实质ACC优势，AUC只有小幅正向。实验12三seed总体No-Go和NIPA弱患者对应证据均保留。本轮依据用户要求进行两项固定消融，不预设有效。',
           '[原论文](https://arxiv.org/html/2406.14455v3)首先以RFE/VAE统一维度，再经AMRS人口图、双GTUNet和ACMGL融合分类；本轮原始分类路径和图不变。[官方代码](https://github.com/NZWANG/MM-GTUNets)及历史本地路径已审阅。VAE每前向重载并随机采样、eval Dropout、RP循环内提前返回、共享卷积.data约束、未注册RP权重、tensor重建detach和全节点BN均保留。',
           '原Classifier每次核心前向只调用一次；验证CE+Graph严格下降选模，early stopping相等损失重置计数。正式测试用最佳权重与训练结束RNG做一次grad-enabled前向。没有多次测试择优、重新预训练VAE或重新拟合RFE。','',
           '## 六组与实现','',
           'E0原生MM-GTUNets；E1原始D4；E2仅EMA；E3仅类别独立ESS；E4两项组合；E5仅辅助原型年龄/性别在outer-train同站点联合错配。原生Non-imaging输入和人口图始终真实。',
           'D4参考是当前query之外的其他两个训练私有子折。年龄带宽max(2,IQR(train)/2)，合法(0,120]否则训练中位数填补，性别同/异/未知核1/.5/.75；联合核相乘。私有随机种子保留SHA256(seed/fold/PCDCA/purpose)，不因目录名变化。',
           'EMA只保存outer-train的16维影像输出，M0=detach(z0)，以后M=.9*M+.1*detach(z)。每epoch仅更新一次、no_grad、无第二个模型或核心前向；以当前EMA训练统计同时标准化teacher和当前query。E1/E3仍用当前detach影像teacher。std总体SD下限.05；tau沿用D4的标准化teacher平均平方中心距离，下限1。',
           '原D4共同beta=clip((minESS-8)/32,0,1)。新beta逐类为ESS_c/(ESS_c+32)，ESS=(sum w)^2/sum(w^2)，无局部支持回退对应全局原型；无全局类支持则屏蔽query。它是有效支持度启发式收缩，不是已校准的概率置信度。',
           '距离logits=-||query-prototype||²/tau；teacher/原型/坐标统计/核/ESS/beta均detach，仅当前影像query反传。lambda=clip(.05*||grad_z(native)||/(||grad_z(proto)||+1e-12),0,.5)*min(epoch/9,1)，epoch0为0。无新参数；原始验证选模不含辅助损失。','',
           '## 正式协议与结果','',
           'ABIDE/AAL116，871人，ASD=0/TD=1。仅seed911原始10折及独立val，历史RFE500/VAE/core/RP/RNG锚点；Adam lr1e-4/wd5e-4，最多500epoch/patience100。以下为10折指标均值±样本SD(ddof=1)，不是三个seed均值SD，也不是方差。',
           '|组别|方法|完成|ACC %|SEN-ASD %|SPE-TD %|prob-AUC %|','|---|---|---|---|---|---|---|']
    for g,m in s['formal'].items():
        vals=['%.4f ± %.4f'%(100*m[k]['mean'],100*m[k]['sample_sd']) for k in ('ACC','SEN','SPE','AUC')]
        lines.append('|'+g+'|'+run.CFG['groups'][g]['name']+'|10/10|'+'|'.join(vals)+'|')
    lines+=['','完整60折指标见fold_metrics.csv，患者级预测见oof.csv；每组871人各一次。ASD概率为softmax第0列，最终类别仍按原始logits argmax，不根据测试调阈值。','',
            '## 对应折配对差','',
            '|比较|ΔACC pp|ΔSEN pp|ΔSPE pp|ΔAUC pp|ACC改善折|AUC改善折|','|---|---|---|---|---|---|---|']
    for pair,p in s['pairs'].items():
        lines.append('|'+pair+'|'+'|'.join(f"{100*p[k]['mean']:+.4f}" for k in ('ACC','SEN','SPE','AUC'))+f"|{p['ACC']['improved_folds']}/10|{p['AUC']['improved_folds']}/10|")
    if s['completed_tasks']==60:
        pairs=s['pairs'];p=pairs['E4-E1'];sh=pairs['E4-E5']
        lines+=['','## 机制是否有效','',
          f"完整SCPDA相对原D4：ACC {100*p['ACC']['mean']:+.4f}pp，AUC {100*p['AUC']['mean']:+.4f}pp，SEN {100*p['SEN']['mean']:+.4f}pp；ACC/AUC改善{p['ACC']['improved_folds']}/{p['AUC']['improved_folds']}折。两指标是否同向提高：{p['ACC']['mean']>0 and p['AUC']['mean']>0}。",
          f"EMA单因素相对D4的ACC/AUC为{100*pairs['E2-E1']['ACC']['mean']:+.4f}/{100*pairs['E2-E1']['AUC']['mean']:+.4f}pp；ESS单因素为{100*pairs['E3-E1']['ACC']['mean']:+.4f}/{100*pairs['E3-E1']['AUC']['mean']:+.4f}pp。",
          f"组合相对EMA的ACC/AUC为{100*pairs['E4-E2']['ACC']['mean']:+.4f}/{100*pairs['E4-E2']['AUC']['mean']:+.4f}pp；相对ESS为{100*pairs['E4-E3']['ACC']['mean']:+.4f}/{100*pairs['E4-E3']['AUC']['mean']:+.4f}pp。若组合没有优于单项，不称组合有效。",
          f"正确条件相对E5错配的ACC/AUC为{100*sh['ACC']['mean']:+.4f}/{100*sh['AUC']['mean']:+.4f}pp，改善{sh['ACC']['improved_folds']}/{sh['AUC']['improved_folds']}折；这是分别重训的条件负对照，不是同Checkpoint直接因果效应。"]
    lines+=['','|组别|EMA更新数/折|EMA-current L2采样均值|beta ASD/TD采样均值|ESS ASD/TD采样均值|热身后加权query梯度比|',
            '|---|---|---|---|---|---|']
    def fmt(v):
        return 'NA' if v is None else f'{v:.6g}'
    for g,m in s['mechanisms'].items():
        beta=m['beta_by_class_sampled_mean'];ess=m['ess_by_class_sampled_mean']
        lines.append('|'+g+'|'+','.join(str(v) for v in m['ema_updates_by_fold'].values())+'|'+fmt(m['ema_current_l2']['mean'])+
                     '|'+('NA' if beta is None else '/'.join(map(fmt,beta)))+'|'+('NA' if ess is None else '/'.join(map(fmt,ess)))+
                     '|'+fmt(m['weighted_query_gradient_ratio']['mean'])+'|')
    lines+=['','这些统计只证明机制实际运行，不自动等于诊断收益。EMA不包含val/test，query保留梯度；新beta两个类别分别计算；E5仅训练内同站点错配，实际改变比例及索引SHA见summary.shuffle和manifest锚点。',
            '注意：训练原型不拟合heldout标签/表型/参考表示，但原生转导图和全节点BN仍可让heldout输入影响train表示，不声称整个模型是归纳式或无泄漏。','',
            '## 研究决策与下一步','',f"**{s['decision']['outcome']}**。描述性判定规则和逐项条件在summary.decision及冻结config中。只有一个seed，相关10折没有构成独立多中心证据，不进行额外bootstrap或显著性宣称。"]
    if s['decision']['outcome']=='Positive':
        lines+=['','仅建议将本轮固定E4及条件负对照带入多seed、独立RFE与未见站点确认；当前尚不足以直接声称MICCAI创新或临床收益。']
    elif s['decision']['outcome'] in ('Mixed','Negative'):
        lines+=['','当前证据不足以直接将完整SCPDA发展为MICCAI 2027第一个创新模块。唯一优先后续是无跨折RFE泄漏和未见站点下的固定疾病原型监督确认；不根据本轮测试调整EMA、ESS常数或辅助损失。']
    else:lines+=['','尚未完成正式实验，不能评价方法效果。恢复命令：README中的launch --stage all；完成且验收的fold会跳过。']
    e=s['execution']
    lines+=['','## 验收、执行和存储','',
            f"正式任务{s['completed_tasks']}/60，各组{s['group_counts']}，两卡{e['physical_gpu_tasks']}。正式首个任务到最后完成墙钟{e['training_span_seconds']/60:.2f}分钟；累计worker训练{e['worker_training_seconds']/60:.2f}分钟。二者不混用。",
            f"分组墙钟/累计训练及每卡时间见summary.execution。单worker最高保留显存{e['peak_reserved_GiB']:.3f}GiB，整卡采样峰值MiB {e['sampled_card_peak_mib']}。正式失败{len(e['failures'])}次、OOM {sum(x.get('oom',False) for x in e['failures'])}次；日志记录全部恢复。",
            f"原始和历史受保护文件{s['protected_verification']['files']}项SHA保持；独立指标最大误差{s['independent_metric_max_error']:.3g}。原始输入、源码、锚点、原VAE、全部历史PCDCA Checkpoint只读。神经训练仅PPU，每worker1CPU线程；新增参数、RFE拟合、VAE训练、DDP均0。两锚点原生/D4三步严格等价和EMA/ESS边界验证通过。",
            f"代码/配置SHA256：`{s['code_sha256']}`。结果内容SHA、core/RP SHA、输入指纹、正式RNG、BN及训练/验证调用数、严格最佳epoch、EMA更新范围、组别开关和OOF覆盖均自动验收；断点去重再次校验结果不执行训练。",
            f"本次运行目录：`{run.BASE}`。未写入/data的新实验目录；根handoff仅在最终追加，旧字节前缀不变。handoff状态及失败原因见summary.handoff；若存储写入失败，待追加内容保存在本目录handoff_pending.md，恢复后运行analyze.py --append-handoff。",
            '仍有历史RFE跨折监督泄漏、871节点转导人口图、原生随机VAE/eval Dropout、PPU长轨迹非逐位一致及单seed局限。本轮是同协议配对扩展，不是无泄漏临床泛化评估。不自动git commit或push。','']
    if s['stop_reason']:lines+=['停止原因：'+s['stop_reason'],'未完成部分不编造成绩。']
    return '\n'.join(lines)


def append_handoff(s,text):
    path=run.ROOT/'handoff.md';start=b'<!-- SCPDA_20261020_START -->';end=b'<!-- SCPDA_20261020_END -->'
    payload='\n\n'+start.decode()+'\n\n'+text+'\n\n'+end.decode()+'\n'
    run.atomic_bytes(payload.encode(),run.BASE/'handoff_pending.md')
    with (run.BASE/'analyze.py').open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        before=run.read_bytes(path)
        if start in before:
            assert end in before and before.count(start)==before.count(end)==1
            return {'appended':True,'idempotent_skip':True,'after_sha256':hashlib.sha256(before).hexdigest()}
        a=run.read_json(run.BASE/'manifest.json')
        assert hashlib.sha256(before).hexdigest()==a['handoff_before_sha256'] and len(before)==a['handoff_before_bytes'],'Concurrent handoff update; preserve and inspect'
        # Prepare the combined replacement only in this experiment directory.
        combined=before+payload.encode();temp=run.BASE/'.handoff_combined'
        run.atomic_bytes(combined,temp)
        try:
            # os.replace cannot cross /tmp and /data filesystems; append with rollback
            # and full readback, holding a lock on the actual handoff inode.
            with path.open('r+b') as dest:
                fcntl.flock(dest,fcntl.LOCK_EX)
                current=dest.read();assert current==before
                try:
                    dest.seek(0,os.SEEK_END);dest.write(payload.encode());dest.flush();os.fsync(dest.fileno())
                    dest.seek(0);after=dest.read()
                    assert after==combined and after[:len(before)]==before
                except Exception:
                    try:dest.truncate(len(before));dest.flush();os.fsync(dest.fileno())
                    except Exception:pass
                    raise
            assert run.read_bytes(path)==combined
        finally:temp.unlink(missing_ok=True)
        return {'appended':True,'before_sha256':a['handoff_before_sha256'],'old_prefix_bytes_preserved':len(before),
                'after_sha256':hashlib.sha256(combined).hexdigest(),'utc':run.utc(),'marker':start.decode()}


def main(final=False,stop_reason=None):
    a=run.guard();ctx=run.load_torch(run.OLD/'artifacts/context.pt')
    rows={g:[] for g in run.CFG['groups']};artifacts={};fold_csv=[];oof_csv=[];maximum=0
    counts={0:0,1:0}
    for g in rows:
        coverage=np.zeros(871,int)
        for f in range(10):
            v=run.valid_result(g,911,f,a,required=final and stop_reason is None)
            if v is None:continue
            idx=v['oof']['indices'];coverage[idx]+=1
            assert v['seed']==911 and v['fold']==f and v['physical_gpu']==f%2
            assert v['oof']['ids']==[ctx['ids'][int(i)] for i in idx]
            assert np.array_equal(v['oof']['labels'],ctx['labels'][idx])
            assert np.array_equal(v['oof']['prediction'],v['oof']['logits'].argmax(1))
            assert np.allclose(softmax(v['oof']['logits'].astype(float),1)[:,0],v['oof']['probability'],rtol=1e-6,atol=1e-7)
            m=independent(v['oof']['labels'],v['oof']['probability'],v['oof']['prediction'])
            maximum=max(maximum,max(abs(m[k]-v['metrics'][k]) for k in KEYS))
            assert v['configuration']==run.CFG
            assert v['new_trainable_parameters']==v['rfe_fit_count']==v['vae_train_count']==0
            assert v['class_calibration']==run.CFG['groups'][g]['class_calibration']
            assert v['ema_updates']==(v['epochs_run'] if run.CFG['groups'][g]['ema'] else 0)
            assert v['ema_shape']==([len(v['prototype_preparation']['train_indices']),16] if run.CFG['groups'][g]['ema'] else None)
            assert all(n==2*v['epochs_run']+1 for n in v['forward_counts'].values())
            assert v['bn_best_batches']==v['best_epoch']+1
            assert v['formal_start_rng_sha256']==a['anchors'][f'911/{f}']['pairing']['classifier_start_rng']
            assert v['formal_test_grad_enabled']
            losses=[e['val_loss'] for e in v['epochs']]
            assert v['best_epoch']==int(np.argmin(losses)) and v['best_val_loss']==min(losses)
            best_score=None;counter=0
            for i,loss in enumerate(losses):
                score=-loss
                if best_score is None:best_score=score
                elif score<best_score:counter+=1
                else:best_score=score;counter=0
                assert counter<100 or i==len(losses)-1
            assert len(losses)==500 or counter>=100
            for e in v['epochs']:
                if g!='E0' and e['epoch']==0:assert e['lambda_eff']==0
                if 'prototype' in e:
                    assert e['prototype']['ema_updates']==(e['epoch']+1 if run.CFG['groups'][g]['ema'] else 0)
                    assert e['prototype']['invalid_prototype_fraction']==0
            rows[g].append(v);counts[f%2]+=1;artifacts[f'{g}/911/{f}']=run.sha_file(run.result_path(g,911,f))
            fold_csv.append({'group':g,'seed':911,'fold':f,**m,'best_epoch':v['best_epoch'],'epochs_run':v['epochs_run'],
                             'training_seconds':v['training_seconds'],'physical_gpu':f%2})
            for j,i in enumerate(idx):
                oof_csv.append({'group':g,'seed':911,'fold':f,'index':int(i),'subject_id':v['oof']['ids'][j],
                                'label':int(v['oof']['labels'][j]),'prob_ASD':float(v['oof']['probability'][j]),
                                'prediction':int(v['oof']['prediction'][j]),'logit_0':float(v['oof']['logits'][j,0]),'logit_1':float(v['oof']['logits'][j,1])})
        if len(rows[g])==10:assert np.all(coverage==1)
    assert maximum<1e-12
    total=sum(map(len,rows.values()));full={g:v for g,v in rows.items() if len(v)==10}
    if final:assert total==60 and counts=={0:30,1:30}
    formal={g:{k:{'mean':float(np.mean([v['metrics'][k] for v in values])),
                   'sample_sd':float(np.std([v['metrics'][k] for v in values],ddof=1)),
                   'fold_values':[v['metrics'][k] for v in values]} for k in KEYS} for g,values in full.items()}
    pairs={f'{x}-{y}':paired(full,x,y) for x,y in PAIRS if x in full and y in full}
    values=[v for vs in rows.values() for v in vs]
    from datetime import datetime
    span=(max(datetime.fromisoformat(v['completed_utc']) for v in values)-min(datetime.fromisoformat(v['started_utc']) for v in values)).total_seconds() if values else 0
    timing={}
    for label,predicate in [(g,lambda v,g=g:v['group']==g) for g in rows]+[(f'GPU{k}',lambda v,k=k:v['physical_gpu']==k) for k in (0,1)]:
        vs=[v for v in values if predicate(v)]
        timing[label]={'tasks':len(vs),'worker_seconds':sum(v['training_seconds'] for v in vs),
                      'span_seconds':(max(datetime.fromisoformat(v['completed_utc']) for v in vs)-min(datetime.fromisoformat(v['started_utc']) for v in vs)).total_seconds() if vs else 0}
    execution=a.get('execution',{});samples=execution.get('resources',[])
    s={'experiment':run.CFG['experiment'],'status':'complete' if total==60 else 'stopped' if stop_reason else 'partial',
       'completed_tasks':total,'group_counts':{g:len(v) for g,v in rows.items()},'formal':formal,'pairs':pairs,
       'mechanisms':{g:mechanism(vs) for g,vs in full.items()},'decision':decision(pairs),'stop_reason':stop_reason,
       'statistics':run.CFG['statistics'],'code_sha256':a['code_sha256'],'artifact_sha256':artifacts,
       'protected_verification':verify_protected(a),'independent_metric_max_error':maximum,
       'OOF':{g:{'n':871,'each_once':True} for g in full},'historical_replay_checks':a.get('historical_replay_checks',{}),
       'shuffle':{k:v['prototype'] for k,v in a['anchors'].items()},
       'execution':{'physical_gpu_tasks':{str(k):v for k,v in counts.items()},'training_span_seconds':span,
                    'worker_training_seconds':sum(v['training_seconds'] for v in values),'timing':timing,
                    'peak_reserved_GiB':max([v['peak_reserved_bytes'] for v in values]+[0])/1024**3,
                    'sampled_card_peak_mib':{str(k):max([r['gpus'][str(k)]['used_mib'] for r in samples if r and 'gpus' in r]+[0]) for k in (0,1)},
                    'queues':execution.get('queues',[]),'failures':execution.get('failures',[]),
                    'RFE_fits':0,'VAE_pretrains':0,'new_trainable_parameters':0,'CPU_neural_training':0,'extra_frozen_inference':0},
       'handoff':{'appended':False},'created_utc':run.utc()}
    for name,data in [('fold_metrics.csv',fold_csv),('oof.csv',oof_csv)]:
        if data:
            buffer=io.StringIO();w=csv.DictWriter(buffer,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
            run.atomic_bytes(buffer.getvalue().encode(),run.BASE/name)
    text=report(s);run.atomic_bytes(text.encode(),run.BASE/'report.md')
    s['report_sha256']=run.sha_file(run.BASE/'report.md')
    run.atomic_json(s,run.BASE/'summary.json')
    if final or stop_reason:
        try:s['handoff']=append_handoff(s,text)
        except Exception as e:s['handoff']={'appended':False,'error':repr(e),'pending_append':str(run.BASE/'handoff_pending.md'),'recovery':'python -B /tmp/SCPDA_20261020/analyze.py --append-handoff'}
        s['final_verified_utc']=run.utc();run.atomic_json(s,run.BASE/'summary.json')
    run.log('analysis_complete',tasks=total,status=s['status'],decision=s['decision']['outcome'],handoff=s['handoff'])
    print(text,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');p.add_argument('--stop-reason');p.add_argument('--append-handoff',action='store_true')
    args=p.parse_args()
    if args.append_handoff:
        s=run.read_json(run.BASE/'summary.json')
        assert s['status']=='complete' or s.get('stop_reason')
        s['handoff']=append_handoff(s,run.read_bytes(run.BASE/'report.md').decode())
        run.atomic_json(s,run.BASE/'summary.json');print(s['handoff'])
    else:main(args.final,args.stop_reason)

