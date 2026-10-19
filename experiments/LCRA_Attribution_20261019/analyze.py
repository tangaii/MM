"""Hierarchical paired diagnostics, independent acceptance and Chinese report."""
from __future__ import annotations
import csv
from datetime import datetime,timezone
import hashlib
import os
from pathlib import Path
import time
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score,accuracy_score,balanced_accuracy_score,recall_score,f1_score,log_loss,brier_score_loss
import run as ex
BASE,ROOT,CFG=ex.BASE,ex.ROOT,ex.CFG
METRICS=('ACC','BA','AUC','SEN','SPE','F1','CE','Brier')


def ce(p,y):
    p=np.clip(np.asarray(p,np.float64),1e-7,1-1e-7)
    return -np.log(np.where(np.asarray(y)==0,p,1-p))


def evaluate(y,p):
    y=np.asarray(y)==0;p=np.asarray(p,np.float64);h=p>=.5
    assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
    tp=np.sum(y&h);tn=np.sum(~y&~h);fp=np.sum(~y&h);fn=np.sum(y&~h)
    return {'ACC':float(np.mean(y==h)),'BA':float(.5*(tp/max(tp+fn,1)+tn/max(tn+fp,1))),
        'AUC':float(roc_auc_score(y,p)) if len(set(y))==2 else None,
        'SEN':float(tp/max(tp+fn,1)),'SPE':float(tn/max(tn+fp,1)),
        'F1':float(2*tp/max(2*tp+fp+fn,1)),
        'CE':float(ce(p,np.where(y,0,1)).mean()),'Brier':float(np.mean((p-y)**2))}


def aggregate(rows):
    seeds={str(s):{m:float(np.mean([r[m] for r in rows if r['seed']==s and r[m] is not None])) for m in METRICS} for s in CFG['seeds']}
    return {'seed_means':seeds,'mean':{m:float(np.mean([r[m] for r in seeds.values()])) for m in METRICS},
        'seed_sd_ddof1':{m:float(np.std([r[m] for r in seeds.values()],ddof=1)) for m in METRICS},
        'definition':'average ten folds within seed, then three seed means and sample SD; no independence claim'}


def paired(a,b,values,draw=None):
    rows=[];total_rescue=total_harm=0
    for v in values:
        idx=v['split']['test'];y=v['labels'][idx]
        if draw is None:
            pa=np.mean([r['modes'][a]['prob_asd'][idx] for r in v['draws']],axis=0)
            pb=np.mean([r['modes'][b]['prob_asd'][idx] for r in v['draws']],axis=0)
        else:
            pa=v['draws'][draw]['modes'][a]['prob_asd'][idx];pb=v['draws'][draw]['modes'][b]['prob_asd'][idx]
        ma,mb=evaluate(y,pa),evaluate(y,pb)
        rescue=int(np.sum(((pa>=.5)==(y==0))&((pb>=.5)!=(y==0))))
        harm=int(np.sum(((pa>=.5)!=(y==0))&((pb>=.5)==(y==0))))
        total_rescue+=rescue;total_harm+=harm
        rows.append({'seed':v['seed'],'fold':v['fold'],**{m:ma[m]-mb[m] for m in METRICS},
            'probability_MAD':float(np.mean(np.abs(pa-pb))),'rescued':rescue,'harmed':harm})
    out=aggregate(rows)
    out['probability_MAD_by_seed']={str(s):float(np.mean([r['probability_MAD'] for r in rows if r['seed']==s])) for s in CFG['seeds']}
    out['rescued_occurrences']=total_rescue;out['harmed_occurrences']=total_harm
    out['folds']=rows
    out['three_seed_delta_range']={m:[min(r[m] for r in out['seed_means'].values()),max(r[m] for r in out['seed_means'].values())] for m in METRICS}
    out['note']='same subjects recur in three seeds; rescue counts are subject-seed occurrences; CE(a)-CE(b)<0 favors a'
    return out


def correlate(a,b):
    a,b=np.asarray(a),np.asarray(b);mask=np.isfinite(a)&np.isfinite(b);a,b=a[mask],b[mask]
    if len(a)<3 or np.ptp(a)==0 or np.ptp(b)==0:return None
    return float(spearmanr(a,b).statistic)


def stage_a(values):
    modes={};comparisons={}
    for mode in ex.MODES:
        by_draw=[]
        for j in range(5):
            rows=[{'seed':v['seed'],**evaluate(v['labels'][v['split']['test']],v['draws'][j]['modes'][mode]['prob_asd'][v['split']['test']])} for v in values]
            by_draw.append(aggregate(rows))
        rows=[{'seed':v['seed'],**evaluate(v['labels'][v['split']['test']],np.mean([r['modes'][mode]['prob_asd'][v['split']['test']] for r in v['draws']],axis=0))} for v in values]
        graph={};prop={}
        fields=('relative_l1','modified_edge_fraction','modified_base_weight_fraction','within_site_relative_l1','between_site_relative_l1')
        for key in fields:graph[key]=ex.describe([np.mean([r['modes'][mode]['graph'][key] for r in v['draws']]) for v in values])
        for key in ('mean','p50','max'):
            graph['selected_edge_relative_'+key]=ex.describe([np.mean([r['modes'][mode]['graph']['modified_relative'][key] for r in v['draws']]) for v in values])
        graph['partition_median']={p:float(np.median([np.mean([r['modes'][mode]['graph']['by_partition'][p] for r in v['draws']]) for v in values])) for p in ex.PARTS}
        for layer in ('img','ph','joint','logits'):
            prop[layer]={key:ex.describe([np.mean([r['modes'][mode]['propagation']['activations'][layer][key] for r in v['draws']]) for v in values]) for key in ('mean_l2','relative_l2','max_abs','mean_one_minus_cosine')}
        prop['graph_loss_delta']=ex.describe([np.mean([r['modes'][mode]['propagation']['graph_loss_delta'] for r in v['draws']]) for v in values])
        prop['pooling']={branch:[{'jaccard':ex.describe([r['modes'][mode]['propagation']['pooling'][branch][level]['jaccard'] for v in values for r in v['draws']]),
            'order_changed_occurrences':sum(not r['modes'][mode]['propagation']['pooling'][branch][level]['ordered_equal'] for v in values for r in v['draws'])}
            for level in range(2 if branch=='img' else 3)] for branch in ('img','ph')}
        # Keep only aligned convolution responses; mismatched pooled node order is explicitly excluded.
        conv={}
        n=len(values[0]['draws'][0]['modes'][mode]['propagation']['conv'])
        for k in range(n):
            rows=[r['modes'][mode]['propagation']['conv'][k] for v in values for r in v['draws']]
            valid=[r['difference']['relative_l2'] for r in rows if r['row_comparison_valid']]
            conv[f'{k}/{rows[0]["layer"]}']={'relative_l2':ex.describe(valid),'aligned':len(valid),'unaligned_excluded':len(rows)-len(valid),'nodes':rows[0]['nodes']}
        prop['convolutions']=conv
        modes[mode]={'by_draw':by_draw,'five_probability_mean':aggregate([
            {'seed':v['seed'],**evaluate(v['labels'][v['split']['test']],np.mean([r['modes'][mode]['prob_asd'][v['split']['test']] for r in v['draws']],axis=0))} for v in values]),
            'graph':graph,'propagation':prop}
    for a,b in [(m,'A0') for m in ex.MODES[1:]]+[('A3','A4'),('A3','A5')]:
        comparisons[a+'-'+b]={'by_draw':[paired(a,b,values,j) for j in range(5)],'five_probability_mean':paired(a,b,values)}
    return {'modes':modes,'paired':comparisons}


def stage_b(values):
    results={};paired_results={};strata={};residuals={}
    for part in ('val','test'):
        results[part]={};paired_results[part]={};strata[part]={}
        for name in ('B0','B1','B2','B3'):
            rows=[];null_draws=[]
            for v in values:
                idx=v['split'][part];p=v['stage_B'][name] if name!='B3' else v['stage_B']['B3_null'].mean(0)
                rows.append({'seed':v['seed'],**evaluate(v['labels'][idx],p[idx])})
            results[part][name]=aggregate(rows)
            if name=='B3':
                for j in range(8):
                    null_draws.append(aggregate([{'seed':v['seed'],**evaluate(v['labels'][v['split'][part]],v['stage_B']['B3_null'][j][v['split'][part]])} for v in values]))
                results[part][name]['eight_nulls']=null_draws
        for other in ('B0','B2','B3'):
            aa,bb=results[part]['B1'],results[part][other]
            paired_results[part]['B1-'+other]={'seed_differences':{s:{m:aa['seed_means'][s][m]-bb['seed_means'][s][m] for m in METRICS} for s in aa['seed_means']},
                'mean':{m:aa['mean'][m]-bb['mean'][m] for m in METRICS}}
        for seed in CFG['seeds']:
            subset=[v for v in values if v['seed']==seed]
            arrays={k:[] for k in ('index','label','age','sex','site','gap','B0','B1','B2','B3')}
            for v in subset:
                idx=v['split'][part]
                arrays['index'].extend(idx);arrays['label'].extend(v['labels'][idx])
                for k in ('age','sex','site'):arrays[k].extend(v['meta'][k][idx])
                arrays['gap'].extend(v['stage_B']['relation_gap'][idx])
                for k in ('B0','B1','B2'):arrays[k].extend(v['stage_B'][k][idx])
                arrays['B3'].extend(v['stage_B']['B3_null'].mean(0)[idx])
            d={k:np.asarray(v) for k,v in arrays.items()};y=d['label'];site=d['site']
            masks={'site/'+str(s):site==s for s in sorted(set(site))}
            masks.update({'sex/'+str(s):d['sex']==s for s in sorted(set(d['sex']))})
            masks.update({'diagnosis/'+str(s):y==s for s in (0,1)})
            for low,high in zip(CFG['stage_B']['age_bins'][:-1],CFG['stage_B']['age_bins'][1:]):
                masks[f'age/{low}-{high}']=(d['age']>=low)&(d['age']<high)
            strata[part][str(seed)]={name:{'n_occurrences':int(mask.sum()),'unique_subjects':len(set(d['index'][mask])),
                'B0':evaluate(y[mask],d['B0'][mask]),'B1':evaluate(y[mask],d['B1'][mask]),'B2':evaluate(y[mask],d['B2'][mask]),
                'CE_B1_minus_B0':float((ce(d['B1'][mask],y[mask])-ce(d['B0'][mask],y[mask])).mean()),
                'CE_B1_minus_B2':float((ce(d['B1'][mask],y[mask])-ce(d['B2'][mask],y[mask])).mean())}
                for name,mask in masks.items() if mask.any()}
            if part!='test':continue
            native=np.concatenate([v['native_official_test_probability'] for v in subset])
            c0=[]
            for v in subset:
                p=ex.OLD/'outputs/E0'/f'seed_{seed}'/f'fold_{v["fold"]}'/'oof.csv'
                with p.open() as f:rows=list(csv.DictReader(f))
                assert np.array_equal([int(r['index']) for r in rows],v['split']['test'])
                c0.extend(float(r['prob_asd']) for r in rows)
            extra=d['B1']-d['B0'];benefit=ce(d['B0'],y)-ce(d['B1'],y)
            mean_extra=extra.mean();between=sum(np.sum(site==s)*(extra[site==s].mean()-mean_extra)**2 for s in set(site))
            total=float(np.sum((extra-mean_extra)**2))
            record={'extra_vs_L3_residual_spearman':correlate(extra,(y==0)-native),
                'extra_vs_C0_residual_spearman':correlate(extra,(y==0)-np.asarray(c0)),
                'relation_gap_vs_B1_CE_benefit_spearman':correlate(d['gap'],benefit),
                'extra_site_between_variance_fraction':float(between/total) if total>0 else None,
                'site_group_count':len(set(site)),'site_fraction_is_descriptive_not_causal':True}
            centered_extra=extra.copy();centered_res=(y==0)-native
            for s in set(site):
                mask=site==s;centered_extra[mask]-=extra[mask].mean();centered_res[mask]-=centered_res[mask].mean()
            record['within_site_centered_extra_vs_L3_residual_spearman']=correlate(centered_extra,centered_res)
            high_error=(np.maximum(native,1-native)>=CFG['stage_B']['high_confidence'])&((native>=.5)!=(y==0))
            low=np.maximum(native,1-native)<=CFG['stage_B']['low_confidence']
            record['confidence_subsets']={name:{'n':int(mask.sum()),'CE_B1_minus_B0':float((ce(d['B1'][mask],y[mask])-ce(d['B0'][mask],y[mask])).mean()),
                'CE_B1_minus_B2':float((ce(d['B1'][mask],y[mask])-ce(d['B2'][mask],y[mask])).mean()),
                'benefited_B1_vs_B0_fraction':float(np.mean(benefit[mask]>0))} for name,mask in [('native_high_confidence_error',high_error),('native_low_confidence',low)] if mask.any()}
            residuals[str(seed)]=record
    return {'metrics':results,'paired':paired_results,'strata':strata,'redundancy':residuals,
        'validation_note':'Validation has repeated subject occurrences across folds; report counts and unique IDs, not an OOF cohort.',
        'test_note':'871 unique test subjects per seed; repeated seeds are same subjects; diagnostic post-hoc analysis, not conditional mutual information.'}


def verify_history(a):
    ex.verify_manifest(a['protected'])
    return {'passed':True,'files':len(a['protected']),'utc':ex.utc()}


def independent_acceptance(values):
    maximum=0.;count=0
    for seed in CFG['seeds']:
        rows=[v for v in values if v['seed']==seed]
        assert np.array_equal(np.sort(np.concatenate([v['split']['test'] for v in rows])),np.arange(871))
    for v in values:
        assert v['state']=='complete' and len(v['draws'])==5 and v['stage_B']['uses_train_anchor_labels_only']
        for r in v['draws']:
            assert r['derived_seed']==ex.draw_seed(v['seed'],v['fold'],r['draw_base'])
            for mode in ex.MODES:
                y=v['labels'][v['split']['test']]==0;p=r['modes'][mode]['prob_asd'][v['split']['test']]
                m=evaluate(np.where(y,0,1),p);h=p>=.5
                independent={'ACC':accuracy_score(y,h),'BA':balanced_accuracy_score(y,h),
                    'SEN':recall_score(y,h),'SPE':recall_score(~y,~h),'F1':f1_score(y,h),
                    'AUC':roc_auc_score(y,p),'CE':log_loss(y,np.clip(p.astype(np.float64),1e-7,1-1e-7)),
                    'Brier':brier_score_loss(y,p.astype(np.float64))}
                maximum=max(maximum,max(abs(m[k]-independent[k]) for k in METRICS));count+=1
                graph=r['modes'][mode]['graph'];assert graph['new_edges']==graph['lost_edges']==0
                assert r['modes'][mode]['audit']['BN_unchanged'] and r['modes'][mode]['audit']['first_transformer_edge_attr_exact']
    assert maximum<1e-12
    return {'passed':True,'sklearn_metric_sets':count,'maximum_absolute_metric_difference':maximum,'per_seed_unique_OOF':871,
        'training_count':0,'neural_CPU_forward_count':0,'state_RNG_graph_checks':True}


def format_score(agg,m):
    mul=100 if m in ('ACC','BA','AUC','SEN','SPE','F1') else 1
    digits=2 if mul==100 else 6
    return f'{agg["mean"][m]*mul:.{digits}f} +/- {agg["seed_sd_ddof1"][m]*mul:.{digits}f}'


def render_report(state,a,values):
    A,B=state['stage_A'],state['stage_B'];off=A['modes']['A0']['five_probability_mean']
    lines=['# 实验9：LCRA双路径归因与诊断互补性审计',
        '## 研究背景与协议',
        '实验1/2的强制对齐和高余弦未建立可靠患者对应；实验3/4发现新增融合残差及联合训练会影响分类；实验5/6的年龄规范偏差未显示稳定增量；实验7的硬选择出现失活；实验8局部JS学习成功，原生图直接效应极弱。因此本轮冻结全部30个L3最佳Checkpoint，神经网络训练次数为0。',
        '本轮是在已经观察实验8成绩之后提出的预注册机制分析。保留871人、3 seeds、10折、历史监督RFE跨折泄漏、全队列转导人口图、ASD=0、VAE抽样和eval Dropout。不能解释成新的独立临床确认或无泄漏泛化结果。A2-A6均为未重训的人工诊断干预，不是新方法成绩。',
        '## 论文与源码审计',
        '论文MRRL的RFE/VAE首先适配维度，并非显式语义对齐；原代码人口图为fused_sim*rp_graph，Tensor版本cal_feature_sim采用Euclidean cdist，区别于论文的correlation描述。RP图在SITE循环内提前返回，奖励卷积共享并在forward用.data投影，RP权重在PPU未注册，value/模态贡献权重存在tensor重建的detach。上述原行为逐项保留。',
        'LCRA的局部JS及softplus间隔直接作用于影像关系投影；RelationMLP另将学习cos、原始cos、表型核、乘积及差值转为中心化c，再经tanh进入图。两路径共享影像投影，但JS不直接约束RelationMLP输出幅度。图干预位于dense_to_sparse之前，原图正则随A_new计算。',
        'TransformerConv把lin_edge(edge_attr)同时加到注意力key和value，边权不是简单归一化邻接系数；原生残差/skip与beta门、TopK下采样/上采样会进一步决定扰动传播。双GTUNet各输出16维，ACMGL融合两模态与算术均值，分类器16->256/ReLU/BN->2。BN保持eval，functional dropout仍随机。',
        '[原论文](https://arxiv.org/html/2406.14455v3)、[官方实现](https://github.com/NZWANG/MM-GTUNets)、[项目仓库](https://github.com/tangaii/MM)。本轮新增实现与实验8冻结数学在5个锚点的确定性环境逐值比较，详情见审计。',
        '## 冻结设计与数值验收',
        '每个条件恢复同一完整核心/BN/未注册RP/关系参数与历史完整RNG，然后按sha256(seed/fold/PCCI-repeat-base)的原规则派生五个draw。不会直接使用51001等作为最终PPU种子。原生RP投影在前向后检查并恢复，其余参数与Buffer严格不变。训练优化器创建、backward、autograd.grad和RFE拟合/变换被运行期禁止。',
        'A0为关闭图修正；A1为原生真实LCRA；A2/A3为gamma=0.25/1的tanh(gamma*c/s)；A4为历史分区内同站点年龄/性别联合错配且共用真实训练尺度；A5为A3对称E的等幅哈希随机符号；A6为全部非对角正边统一*1.05。所有组原始表型、VAE和RP输入相同。',
        '尺度s取真实条件训练查询至训练锚点的|c|第75百分位。与1e-8以及8*float32_eps*max|u_train|比较，退化则停止校准放大并报告零校准图。真实/错配共用s。校准不使用标签或held-out指标；未从A2/A3中选择最优倍率。',
        f'训练尺度s中位数={np.median([v["scale"]["s"] for v in values]):.9g}，范围[{min(v["scale"]["s"] for v in values):.9g},{max(v["scale"]["s"] for v in values):.9g}]；退化锚点={sum(v["scale"]["degenerate"] for v in values)}/30。',
        f'历史逐受试者五draw重放：最大概率绝对差={state["replay"]["max_abs"]:.9g}，完全一致条件={state["replay"]["exact"]}/450，标签分歧={state["replay"]["label_disagreements"]}。该误差与确定性实现比较分开呈现，不通过放宽容差伪称逐位重放。',
        f'同draw末尾完整RNG一致条件={state["rng_after_equal"]}/1050；如TopK顺序/集合变化，分别统计，不能以顺序变化等同集合变化。',
        '## Stage A：图变化与传播',
        '下表为30锚点各先平均5draw再取中位数；百分比用于图相对变化，概率MAD为原始0-1量纲。图边按有向矩阵项计数，对称边会计两次。',
        '|条件|全图相对L1 %|修改边比例 %|承载原图质量 %|已修改边平均相对变化 %|影像表示相对L2|表型表示相对L2|融合相对L2|概率MAD|',
        '|---|---|---|---|---|---|---|---|---|']
    for mode in ex.MODES:
        entry=A['modes'][mode];g=entry['graph'];p=entry['propagation']
        mad=0 if mode=='A0' else np.mean(list(A['paired'][mode+'-A0']['five_probability_mean']['probability_MAD_by_seed'].values()))
        lines.append('|'+mode+'|'+ '|'.join(f'{g[k]["p50"]*100:.6f}' for k in ('relative_l1','modified_edge_fraction','modified_base_weight_fraction','selected_edge_relative_mean'))+'|'+
            '|'.join(f'{p[k]["relative_l2"]["p50"]:.6g}' for k in ('img','ph','joint'))+f'|{mad:.8g}|')
    lines+=['局部支持覆盖率与已选边修改强度必须分别解读。全图变化小可以由支持稀疏及低权重质量占比造成，也可以由MLP的c本身小造成。逐节点行度、分区、站点内/间边变化和绝对变化分位数保存在每折文件，summary给出分级统计。对A6单独执行5%规则，不把它视为和A3等幅。',
        '|条件|ACC|BA|prob-AUC|SEN-ASD|SPE-HC|F1-ASD|CE|Brier|',
        '|---|---|---|---|---|---|---|---|---|']
    for mode in ex.MODES:lines.append('|'+mode+'|'+'|'.join(format_score(A['modes'][mode]['five_probability_mean'],m) for m in METRICS)+'|')
    lines+=['上表为每fold五次概率平均之后计算指标，每seed平均十折，最后三个seed均值±样本SD；它是冻结稳定性辅助指标，不与实验8单次正式成绩混列。',
        '|配对（前者−后者）|ΔACC pp|ΔBA pp|ΔAUC pp|ΔCE|ΔBrier|救回/误伤（subject-seed）|',
        '|---|---|---|---|---|---|---|']
    for name,e in A['paired'].items():
        q=e['five_probability_mean'];m=q['mean']
        lines.append('|'+name+'|'+'|'.join(f'{m[k]*100:+.5f}' for k in ('ACC','BA','AUC'))+f'|{m["CE"]:+.8f}|{m["Brier"]:+.8f}|{q["rescued_occurrences"]}/{q["harmed_occurrences"]}|')
    lines+=['CE差负值表示前者更好；CE(A3)-CE(A0)、CE(A3)-CE(A4)、CE(A3)-CE(A5)是三个不同问题。救回/误伤跨seed重复同一受试者，不是独立病例总数。',
        '### 三seed分别结果与配对方向',
        '|条件|seed|ACC %|BA %|AUC %|CE|Brier|','|---|---|---|---|---|---|---|']
    for mode,e in A['modes'].items():
        for seed,m in e['five_probability_mean']['seed_means'].items():
            lines.append(f'|{mode}|{seed}|{m["ACC"]*100:.5f}|{m["BA"]*100:.5f}|{m["AUC"]*100:.5f}|{m["CE"]:.8f}|{m["Brier"]:.8f}|')
    lines+=['### 五draw逐次配对',
        '每格列出三个seed依次的差值，draw仍是重复推理，不能计为独立队列；完整各组每draw六指标及CE/Brier见summary.stage_A.modes。',
        '|配对|draw base|ΔACC pp（911/912/913）|ΔBA pp|ΔAUC pp|ΔCE|','|---|---|---|---|---|---|']
    for name,e in A['paired'].items():
        for j,q in enumerate(e['by_draw']):
            columns=[]
            for metric in ('ACC','BA','AUC','CE'):
                mul=1 if metric=='CE' else 100
                columns.append(' / '.join(f'{q["seed_means"][str(s)][metric]*mul:+.7g}' for s in CFG['seeds']))
            lines.append(f'|{name}|{CFG["draw_bases"][j]}|'+'|'.join(columns)+'|')
    lines+=['### 传播层与TopK',
        '|条件|影像TopK集合Jaccard最低值|表型TopK集合Jaccard最低值|Logits平均L2中位数|图正则差中位数|',
        '|---|---|---|---|---|']
    for mode,e in A['modes'].items():
        p=e['propagation'];ji=min(r['jaccard']['min'] for r in p['pooling']['img']);jp=min(r['jaccard']['min'] for r in p['pooling']['ph'])
        lines.append(f'|{mode}|{ji:.8f}|{jp:.8f}|{p["logits"]["mean_l2"]["p50"]:.9g}|{p["graph_loss_delta"]["p50"]:+.9g}|')
    lines+=['下表逐层相对L2取中位数；只比较相同患者顺序的激活。池化后节点集合或顺序变化的中间张量不进行伪配对，排除次数记录在summary；最终恢复全871节点的表示仍可比较。',
        '|条件|TransformerConv层（执行顺序）|相对L2中位数|有效/排除比较数|','|---|---|---|---|']
    for mode in ('A1','A3','A4','A5','A6'):
        for key,row in A['modes'][mode]['propagation']['convolutions'].items():
            lines.append(f'|{mode}|{key}|{row["relative_l2"]["p50"]:.9g}|{row["aligned"]}/{row["unaligned_excluded"]}|')
    lines+=['## Stage B：固定邻居疾病得分',
        '固定最佳L3投影和历史Top64；B0=Pimg，B1=normalize(Pimg*Qtrue)，B2=normalize(Pimg*Qshuffle)，B3为8种训练同站点Null的疾病得分均值。投票只访问当前训练锚点ASD标签。8种Null各自指标和分布均保留，均值概率指标不冒充均值指标。没有拟合分类器或调权重。',
        '|分区|关系|ACC|BA|AUC|SEN|SPE|F1|CE|Brier|','|---|---|---|---|---|---|---|---|---|---|']
    for part,entries in B['metrics'].items():
        for name,e in entries.items():lines.append('|'+part+'|'+name+'|'+'|'.join(format_score(e,m) for m in METRICS)+'|')
    lines+=['Validation的同一患者可能在不同fold重复出现，不能称为871人完整OOF。Test每seed恰871人；重复seed和Null不独立。',
        '|分区|配对|ΔAUC pp 三seed|ΔCE 三seed|ΔBrier 三seed|','|---|---|---|---|---|']
    for part,entries in B['paired'].items():
        for name,q in entries.items():
            columns=[' / '.join(f'{q["seed_differences"][str(s)][m]*(100 if m=="AUC" else 1):+.8f}' for s in CFG['seeds']) for m in ('AUC','CE','Brier')]
            lines.append('|'+part+'|'+name+'|'+'|'.join(columns)+'|')
    lines+=['### 冗余与患者分层',
        'extra_score=B1-B0；native残差=1(ASD)-历史原生单次OOF概率。以下是描述性Spearman和站点方差分量，不能解释为严格条件互信息或独立因果增量。',
        '|seed|extra与L3残差|extra与C0残差|站点内中心化后|关系优势与邻居CE收益|extra站点间方差占比|','|---|---|---|---|---|---|']
    for seed,q in B['redundancy'].items():
        lines.append('|'+seed+'|'+'|'.join(f'{q[k]:.7g}' if q[k] is not None else 'NA' for k in ('extra_vs_L3_residual_spearman','extra_vs_C0_residual_spearman',
            'within_site_centered_extra_vs_L3_residual_spearman','relation_gap_vs_B1_CE_benefit_spearman','extra_site_between_variance_fraction'))+'|')
    lines+=['|seed|原生置信度子集|人数|B1−B0 CE|B1−B2 CE|B1较B0受益比例|','|---|---|---|---|---|---|']
    for seed,q in B['redundancy'].items():
        for name,r in q['confidence_subsets'].items():lines.append(f'|{seed}|{name}|{r["n"]}|{r["CE_B1_minus_B0"]:+.8f}|{r["CE_B1_minus_B2"]:+.8f}|{r["benefited_B1_vs_B0_fraction"]:.6f}|')
    lines+=['高置信度为max(p,1-p)>=0.8且预测错误；低置信度<=0.6，阈值在新结果前固定。年龄分层为[0,12)、[12,18)、[18,30)、[30,1000)；性别、诊断、站点完整分层均在summary.stage_B.strata，单类分层AUC记为NA。',
        '## 研究判断',state['interpretation'],
        '这只定位固定Checkpoint下的推理路径。实验8不同训练组的差异仍混合联合优化、随机训练轨迹和随机选模，不能由本轮关闭模块直接证明某一训练梯度是唯一原因。年龄/性别已在原主干中使用，当前局部关系没有额外价值也不等同这些协变量完全没有医学信息。',
        '唯一下一步：'+state['next_direction'],
        '## 执行审计',
        f'完成30/30锚点，两卡任务数{state["gpu_tasks"]}；正式A0-A6条件前向1050次，历史shuffled原生重放额外150次（A0/A1的300次已复用），5锚点确定性对照30次；实际总前向{state["forward_count"]}。预检前两draw直接续作正式，不重复执行。',
        f'模型训练0、optimizer创建0、backward0、CPU神经推理0；累计Worker秒{state["worker_seconds"]:.3f}，审计启动至验收墙钟{state["wall_seconds"]:.3f}秒；各PPU用时{state["device_seconds"]}。显存峰值见summary资源采样和各折执行记录。',
        f'受保护历史文件{state["history_verification"]["files"]}项SHA全部一致。独立sklearn指标{state["independent_acceptance"]["sklearn_metric_sets"]}套复算最大差{state["independent_acceptance"]["maximum_absolute_metric_difference"]:.3g}。代码SHA={ex.code_hash()}。',
        '任务锁防止重复，单fold合并产物按draw原子保存，严格fingerprint与Checkpoint SHA匹配才跳过。ossfs读取仅对暂时FileNotFoundError有限重试，EIO直接报错。实际断点续跑验收不执行网络前向；历史模型只读引用，无模型副本。全部诊断和最终SHA验收后才追加handoff，不自动push。']
    return '\n\n'.join(lines)+'\n'


def append_handoff(text,a):
    path=ROOT/'handoff.md';marker=b'<!-- LCRA_Attribution_20261019_START -->'
    original=ex.read_bytes(path)
    if marker in original:
        assert hashlib.sha256(original[:a['handoff_before_bytes']]).hexdigest()==a['handoff_before_sha256']
        return {'status':'already_appended','after_sha256':hashlib.sha256(original).hexdigest(),'historical_prefix_preserved':True}
    assert hashlib.sha256(original).hexdigest()==a['handoff_before_sha256']
    addition='<!-- LCRA_Attribution_20261019_START -->\n\n'+text+'\n<!-- LCRA_Attribution_20261019_END -->\n'
    payload=original+b'\n\n'+addition.encode()
    pending=BASE/'handoff_pending.md';ex.atomic_bytes(addition.encode(),pending)
    temporary=BASE/'handoff.atomic.pending'
    try:
        ex.atomic_bytes(payload,temporary)
        assert ex.read_bytes(path)==original
        os.replace(temporary,path)
        fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
        assert ex.read_bytes(path)==payload
        pending.unlink()
        return {'status':'appended','before_sha256':a['handoff_before_sha256'],'after_sha256':hashlib.sha256(payload).hexdigest(),
            'historical_prefix_preserved':True,'idempotent_marker':marker.decode()}
    except OSError as error:
        return {'status':'pending','error':repr(error),'path':str(pending)}


def verify_terminal(state,a):
    assert state['code_sha256']==ex.code_hash()
    for path,sha in state['artifact_sha256'].items():assert ex.sha_file(path)==sha
    assert ex.sha_file(BASE/'report.md')==state['report_sha256']
    verify_history(a)
    assert ex.sha_file(ROOT/'handoff.md')==state['handoff']['after_sha256']
    return True


def finish(state,a):
    values=[ex.load_torch(ex.result_path(s,f)) for s in CFG['seeds'] for f in range(10)]
    assert all(v['code_sha256']==ex.code_hash() for v in values)
    state['independent_acceptance']=independent_acceptance(values)
    state['stage_A']=stage_a(values);state['stage_B']=stage_b(values)
    state['gpu_tasks']={g:sum(v['execution']['gpu']==g for v in values) for g in ('0','1')}
    assert state['gpu_tasks']=={'0':15,'1':15}
    state['worker_seconds']=sum(v['execution']['seconds'] for v in values)
    state['device_seconds']={g:sum(v['execution']['seconds'] for v in values if v['execution']['gpu']==g) for g in ('0','1')}
    state['forward_count']=sum(v['execution']['forwards'] for v in values)
    assert state['forward_count']==1230
    replay=[e for v in values for r in v['draws'] for e in r['historical_replay'].values()]
    state['replay']={'exact':sum(r['exact'] for r in replay),'total':len(replay),
        'max_abs':max(r['maximum_absolute_difference'] for r in replay),'label_disagreements':sum(r['label_disagreements'] for r in replay)}
    state['rng_after_equal']=sum(r['modes'][m]['propagation']['rng_after_equal'] for v in values for r in v['draws'] for m in ex.MODES)
    state['history_verification']=verify_history(a)
    state['artifact_sha256']={str(ex.result_path(v['seed'],v['fold'])):ex.sha_file(ex.result_path(v['seed'],v['fold'])) for v in values}
    state['resume_acceptance']={'all_30_skipped':all(ex.valid_result(v['seed'],v['fold'],a) for v in values),'new_forward':0,'training':0}
    A=state['stage_A'];B=state['stage_B']
    gains=A['paired']['A3-A4']['five_probability_mean']['seed_means'];random=A['paired']['A3-A5']['five_probability_mean']['seed_means']
    bsignal=B['paired']['test']['B1-B0']['seed_differences']
    true_better=all(v['CE']<0 for v in gains.values()) and all(v['CE']<0 for v in random.values())
    b_better=all(v['CE']<0 and v['AUC']>0 for v in bsignal.values())
    changed=A['paired']['A3-A0']['five_probability_mean'];global_changed=A['paired']['A6-A0']['five_probability_mean']
    if true_better and b_better:
        decision='A';reason='校准后正确关系在配对CE及固定邻居得分中出现方向一致的支持；仍需新的训练期方案及无泄漏独立验证，不能把A3当正式方法。'
    elif changed['rescued_occurrences']+changed['harmed_occurrences']==0 and global_changed['rescued_occurrences']+global_changed['harmed_occurrences']>0:
        decision='B';reason='校准局部图的标签效应仍弱，全图正控制产生响应；当前局部关系支持及稠密图质量分配构成更直接的传播瓶颈。'
    elif changed['rescued_occurrences']+changed['harmed_occurrences']>0 and not true_better and not b_better:
        decision='C';reason='GTUNet能感知校准局部关系，但正确关系没有同时优于错配和等幅随机方向；邻居得分也未建立跨seed一致新增诊断支持。'
    else:
        decision='D';reason='原生图推理效应极弱，校准及邻居对照未共同建立可靠条件增量；实验8小幅改善更接近训练期辅助正则化迹象，但本轮没有直接证明联合优化的唯一因果机制。'
    state['research_decision']=decision
    state['interpretation']=f'选择{decision}：'+reason
    state['next_direction']='在独立预注册、无RFE跨折泄漏并含站点外验证的协议中，先检验一个固定影像关系基准上的表型邻居疾病增量；在确认非冗余信息之前暂停为当前年龄/性别关系继续增加对齐或Gate复杂度。'
    if decision=='B':state['next_direction']='另行预注册训练锚点局部支持的图质量重分配对照，先用影像关系检验GTUNet可用的传播支持，再以等质量真实/错配表型验证增量；本轮不调alpha或重新训练。'
    state['wall_seconds']=time.time()-a['started_epoch'];state['finished_utc']=ex.utc()
    report=render_report(state,a,values);ex.atomic_bytes(report.encode(),BASE/'report.md')
    state['report_sha256']=ex.sha_file(BASE/'report.md')
    # Append a concise but complete research handoff once, after all acceptance.
    handoff='## 实验9：LCRA双路径归因与诊断互补性审计\n\n'+state['interpretation']+'\n\n'
    handoff+='冻结L3 30/30，A0-A6共1050次，历史重放额外150次，确定性等价30次；训练0。两卡15/15。\n\n'
    handoff+='|组别|ACC|BA|AUC|CE|\n|---|---|---|---|---|\n'
    for mode,e in A['modes'].items():handoff+='|'+mode+'|'+'|'.join(format_score(e['five_probability_mean'],m) for m in ('ACC','BA','AUC','CE'))+'|\n'
    handoff+='\n图与传播、真实/错配/等幅随机、B0-B3验证/测试及患者分层完整证据：`experiments/LCRA_Attribution_20261019/report.md`、`summary.json`。\n\n'
    handoff+='主干、BN、未注册RP与完整RNG每条件恢复，RP合法原位约束检查后恢复；无optimizer/backward/RFE/VAE训练。保留历史跨折RFE泄漏与转导图，所有A组是诊断干预，不能当作新模型临床成绩。\n\n'
    handoff+=f'历史SHA验收{state["history_verification"]["files"]}项；历史冻结重放最大概率差{state["replay"]["max_abs"]:.9g}，独立实现确定性逐值等价；sklearn指标最大差{state["independent_acceptance"]["maximum_absolute_metric_difference"]:.3g}；墙钟{state["wall_seconds"]:.3f}秒；各卡Worker秒{state["device_seconds"]}。\n\n'
    handoff+='唯一下一步：'+state['next_direction']+'\n'
    state['handoff']=append_handoff(handoff,a)
    assert state['handoff']['status'] in ('appended','already_appended'),state['handoff']
    state['history_final_after_handoff']=verify_history(a)
    state['status']='complete';ex.atomic_json(state,BASE/'summary.json')
    verify_terminal(state,a);ex.log('complete',decision=decision,anchors=30,training=0,forwards=state['forward_count'],gpu=state['gpu_tasks'])
    print(report,flush=True)
