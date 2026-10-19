"""Independent acceptance, paired results, Chinese report and safe handoff."""
from __future__ import annotations
import fcntl
import hashlib
import json
import os
import numpy as np
from scipy.special import softmax
from sklearn import metrics as sk
from run import BASE,ROOT,OLD,CFG,load,save_json,sha_file,history_load,array_sha,tree_hash,code_hash,protected_guard,completed,utc,log

METRICS=('ACC','BA','AUC','SEN','SPE','F1','CE','Brier')


def independent_metrics(y,p):
    y=(np.asarray(y)==0).astype(int)
    p=np.asarray(p,dtype=np.float64)
    pred=p>=.5
    tp=np.sum(pred&(y==1)); tn=np.sum(~pred&(y==0)); fp=np.sum(pred&(y==0)); fn=np.sum(~pred&(y==1))
    return {'ACC':float((tp+tn)/len(y)),'BA':float(.5*(tp/(tp+fn)+tn/(tn+fp))),
        'AUC':float(sk.roc_auc_score(y,p)),'SEN':float(tp/(tp+fn)),'SPE':float(tn/(tn+fp)),
        'F1':float(2*tp/max(2*tp+fp+fn,1)), 'CE':float(sk.log_loss(y,p,labels=[0,1])),
        'Brier':float(np.mean((y-p)**2))}


def results_table(groups):
    lines=['|组别|ACC %|BA %|prob-AUC %|SEN-ASD %|SPE-HC %|F1-ASD %|CE|Brier|',
           '|---|---|---|---|---|---|---|---|---|']
    for group,entry in groups.items():
        row=[]
        for key in METRICS:
            factor=100 if key not in ('CE','Brier') else 1
            mean=entry['mean'][key]*factor; sd=entry['fold_sample_sd'][key]*factor
            row.append(f'{mean:.2f} +/- {sd:.2f}' if factor==100 else f'{mean:.6f} +/- {sd:.6f}')
        lines.append('|'+group+'|'+'|'.join(row)+'|')
    return lines


def append_handoff(report):
    start='<!-- GBSA_Pilot_20261019_START -->'; end='<!-- GBSA_Pilot_20261019_END -->'
    addition='\n\n'+start+'\n\n'+report+'\n'+end+'\n'
    path=ROOT/'handoff.md'
    try:
        with path.open('r+b',buffering=0) as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            old=f.read()
            if start.encode() in old:
                assert old.count(start.encode())==old.count(end.encode())==1
                return {'status':'already_appended_verified','after_sha256':hashlib.sha256(old).hexdigest()}
            f.seek(0,os.SEEK_END)
            assert f.write(addition.encode())==len(addition.encode())
            os.fsync(f.fileno())
            f.seek(0); new=f.read()
            assert new==old+addition.encode()
            assert new.count(start.encode())==new.count(end.encode())==1
            return {'status':'appended_verified','before_sha256':hashlib.sha256(old).hexdigest(),
                'after_sha256':hashlib.sha256(new).hexdigest(),'prior_bytes_preserved':len(old),
                'method':'exclusive existing-inode lock, single append, fsync, exact readback; historical bytes preserved'}
    except (OSError,AssertionError) as exc:
        from descriptors import atomic_bytes
        atomic_bytes(BASE/'handoff_pending.md',addition.encode())
        return {'status':'pending','error':repr(exc),'path':str(BASE/'handoff_pending.md')}


def analyze():
    audit=json.loads((BASE/'audit.json').read_text())
    assert audit['preflight']['0']['passed'] and audit['preflight']['1']['passed']
    protected_guard(audit)
    ctx=history_load(OLD/'artifacts/context.pt'); y=ctx['labels']
    cache=load(BASE/'runs/descriptors.bin')
    assert cache['code_sha256']==code_hash() and cache['ids']==ctx['ids']
    groups={}; artifacts={}; rows={}; max_error=0.
    for group in CFG['groups']:
        folds=[]; oof=np.full(871,np.nan); coverage=np.zeros(871,int)
        for fold in range(10):
            assert completed(group,fold)
            path=BASE/f'runs/{group}_f{fold}.bin'; r=load(path)
            artifacts[str(path)]=sha_file(path)
            idx=r['indices']; coverage[idx]+=1; oof[idx]=r['prob_asd']
            assert r['ids']==[ctx['ids'][i] for i in idx] and np.array_equal(r['labels'],y[idx])
            assert np.array_equal(softmax(r['logits'],axis=1)[:,0],r['prob_asd'])
            m=independent_metrics(y[idx],r['prob_asd'])
            error=max(abs(m[k]-r['metrics'][k]) for k in METRICS)
            max_error=max(max_error,error); assert error<1e-12
            assert r['best']['epoch']==int(np.argmin([h['val_original'] for h in r['history']]))
            assert r['best']['validation_loss']==min(h['val_original'] for h in r['history'])
            assert r['BN_batch_count']==int(r['best']['core']['state']['clf.2.num_batches_tracked'])
            assert r['single_native_test_forward'] and r['test_no_text_required']
            assert r['physical_gpu']==str(fold%2)
            split=history_load(OLD/f'artifacts/seed_911/fold_{fold}/splits.pt')
            ref=cache['folds'][fold]['reference']
            assert ref['train_sha']==array_sha(split['train'])
            np.testing.assert_array_equal(ref['mean'],cache['raw_stats'][split['train']].mean(0))
            mechanisms=[h for h in r['history'] if 'mechanism' in h]
            if group=='S1':
                assert all(h['mechanism']['semantic_to_img_grad']>0 for h in mechanisms)
                assert any(h['semantic_coefficient']>0 for h in mechanisms)
            folds.append({'fold':fold,'metrics':m,'best_epoch':r['best']['epoch'],'epochs':r['epochs'],
                'seconds':r['classifier_seconds'],'physical_gpu':r['physical_gpu'],
                'peak_allocated_bytes':r['peak_allocated_bytes'],'peak_reserved_bytes':r['peak_reserved_bytes'],
                'early_mechanism':mechanisms[0],'late_mechanism':mechanisms[-1],
                'mechanism_samples':mechanisms,'pairing':r['pairing']})
            rows[group,fold]=r
        assert np.all(coverage==1) and np.isfinite(oof).all()
        groups[group]={'mean':{k:float(np.mean([f['metrics'][k] for f in folds])) for k in METRICS},
            'fold_sample_sd':{k:float(np.std([f['metrics'][k] for f in folds],ddof=1)) for k in METRICS},
            'OOF_metrics':independent_metrics(y,oof),'OOF_probability':oof.tolist(),'folds':folds}
    delta={k:groups['S1']['mean'][k]-groups['C0']['mean'][k] for k in METRICS}
    wins={k:int(sum(rows['S1',f]['metrics'][k]>rows['C0',f]['metrics'][k] for f in range(10))) for k in ('ACC','BA','AUC')}
    wins['CE']=int(sum(rows['S1',f]['metrics']['CE']<rows['C0',f]['metrics']['CE'] for f in range(10)))
    comparisons=[]
    for f in range(10):
        a,b=rows['C0',f],rows['S1',f]
        assert a['pairing']==b['pairing'],'Paired original initialization/inputs/RNG differ'
        comparisons.append({'fold':f,'delta':{k:b['metrics'][k]-a['metrics'][k] for k in METRICS},
            'probability_MAD':float(np.abs(a['prob_asd']-b['prob_asd']).mean()),
            'native_inference_RNG_equal':tree_hash(a['test_rng'])==tree_hash(b['test_rng']),
            'image_embedding_relative_l2':norm(b['test_image_embedding']-a['test_image_embedding'])/max(norm(a['test_image_embedding']),1e-12),
            'image_encoder_best_state_equal':tree_hash({k:v for k,v in a['best']['core']['state'].items() if k.startswith('img_unet.')})==tree_hash({k:v for k,v in b['best']['core']['state'].items() if k.startswith('img_unet.')})})
    gradient_nonzero=all(f['late_mechanism']['mechanism']['semantic_to_img_grad']>0 for f in groups['S1']['folds'])
    go=delta['AUC']>=CFG['go_auc_delta'] and wins['AUC']>=CFG['go_auc_wins'] and delta['CE']<=0 and gradient_nonzero
    execution=audit['execution'].copy()
    execution.update({'tasks':20,'task_counts_by_gpu':{str(g):sum(r['physical_gpu']==str(g) for r in rows.values()) for g in (0,1)},
        'task_counts_by_group_gpu':{group:{str(g):sum(rows[group,f]['physical_gpu']==str(g) for f in range(10)) for g in (0,1)} for group in CFG['groups']},
        'sum_training_seconds':sum(r['classifier_seconds'] for r in rows.values()),
        'training_seconds_by_gpu':{str(g):sum(r['classifier_seconds'] for r in rows.values() if r['physical_gpu']==str(g)) for g in (0,1)},
        'peak_worker_allocated_bytes':max(r['peak_allocated_bytes'] for r in rows.values()),
        'peak_worker_reserved_bytes':max(r['peak_reserved_bytes'] for r in rows.values()),
        'CPU_neural_training':0,'VAE_pretraining':0,'RFE_fitting':0})
    summary={'experiment':CFG['experiment'],'completed':True,'seed':CFG['seed'],'groups':groups,
        'paired_delta_S1_minus_C0':delta,'wins_out_of_10':wins,'paired_folds':comparisons,
        'go':go,'semantic_gradient_to_original_image_encoder_verified':gradient_nonzero,'execution':execution,
        'code_sha256':code_hash(),'artifact_SHA256':artifacts,
        'acceptance':{'tasks':20,'OOF_subjects_per_group':871,'probability_column':'ASD label0',
            'independent_metrics_max_error':max_error,'protected_files_SHA_unchanged':len(audit['protected_files']),
            'all_paired_origins_identical':True,'reference_train_only':True,'checkpoint_selection_verified':True,
            'single_native_test_verified':True,'text_frozen_verified':cache['text_model']['frozen']},
        'statistics':'One seed. Mean and sample SD across ten folds, not three-seed SD or independent cohorts.',
        'limitations':CFG['protocol_limitations'],'handoff':{'status':'not_yet_appended'}}
    lines=['# 实验11：全局脑网络语义对齐Pilot','','## 方法与审计','',
        '只比较C0原始MM-GTUNets与S1语义辅助监督，seed911各10折。全部复用历史train/val/test、500维RFE、VAE和训练起点；C0重新配对训练，不使用实验10轻量探针成绩。',
        '原img_unet在871名受试者人口图输出16维；捕获一次原生输出，经16→128→384训练投影头与冻结MiniLM文本做对称InfoNCE。仅训练患者构成正/负样本，Loss除log(N_train)，epoch0至9系数由0线性升至.05。验证仍仅CE+Graph，测试不调用文本或投影头。',
        'FC来自每人的原始AAL116相关矩阵，与原Fisher-Z缓存逐值核对。描述含全脑平均/绝对连接、最强两条正/负连接及全局图指标。正相关Top10%固定667条无向边；加权最短路径距离1/r，全局效率包括断连零贡献，确定性greedy模块度及NetworkX加权聚类，另有节点强度均值/SD。脑区用一基ROI编号，未臆造解剖语义。',
        '训练折独立拟合描述参考均值/SD，仅用固定英文模板；不含诊断、年龄、性别、站点。修改held-out统计不改变训练参考的断言通过。文本由同一FC导出，属于影像结构辅助监督，不是额外独立临床模态，也未证明疾病语义。',
        f'编码器`{CFG["text_model"]}`固定revision`{CFG["text_revision"]}`，官方masked mean pooling与L2归一化。全部参数冻结，缓存批量前向后权重SHA不变；CPU/PPU参考最大差{cache["text_model"]["CPU_PPU_max_error"]:.3g}。最长描述{cache["text_model"]["max_tokens"]} tokens，无截断。',
        '[fMRI-LM 3.1/3.2及Fig.2/4](https://arxiv.org/html/2511.21760v3)提供描述构造与冻结文本空间思路；官方实现包含多种contrastive选项及量化/域混淆。本轮仅采用用户固定的全局InfoNCE辅助Loss，并非复现fMRI-LM完整框架。[源码](https://github.com/yuxiangwei0808/fMRI-LM)。',
        '[MM-GTUNets论文](https://arxiv.org/html/2406.14455v3)及真实源码仍保留VAE随机采样、评估Dropout、RFE跨折覆盖、RP循环提前return、.data约束、未注册RP及tensor重建的特殊梯度路径。实验8/9未确认可靠关系增量，实验10轻量探针未建立稳定表型互补，因此本轮仅检验FC衍生辅助结构监督。','',
        '## 完整正式结果','',
        '均值与样本SD均来自单seed的10个fold；不是三seed标准差，不把相关fold当独立外部队列。主结果为每个最佳Checkpoint的一次原生随机测试。','']
    lines+=results_table(groups)
    lines+=['','|fold|C0 ACC %|S1 ACC %|C0 AUC %|S1 AUC %|ΔAUC pp|C0 CE|S1 CE|C0/S1最佳epoch|','|---|---|---|---|---|---|---|---|---|']
    for f in range(10):
        a,b=rows['C0',f],rows['S1',f]
        lines.append(f'|{f}|{a["metrics"]["ACC"]*100:.2f}|{b["metrics"]["ACC"]*100:.2f}|{a["metrics"]["AUC"]*100:.2f}|{b["metrics"]["AUC"]*100:.2f}|{(b["metrics"]["AUC"]-a["metrics"]["AUC"])*100:+.3f}|{a["metrics"]["CE"]:.6f}|{b["metrics"]["CE"]:.6f}|{a["best"]["epoch"]}/{b["best"]["epoch"]}|')
    lines+=['',f'S1−C0：ACC {delta["ACC"]*100:+.3f}pp，BA {delta["BA"]*100:+.3f}pp，AUC {delta["AUC"]*100:+.3f}pp，CE {delta["CE"]:+.6f}。ACC/BA/AUC/CE改善折数分别为{wins["ACC"]}/{wins["BA"]}/{wins["AUC"]}/{wins["CE"]}（分母10）。',
        '','## 机制诊断','','|fold|早/晚语义Loss(logN归一化)|早/晚语义→img梯度|晚期原生→img梯度|晚期语义→投影头梯度|早/晚影像有效秩|测试概率MAD|','|---|---|---|---|---|---|---|']
    for f in range(10):
        early=groups['S1']['folds'][f]['early_mechanism']; late=groups['S1']['folds'][f]['late_mechanism']
        a,b=early['mechanism'],late['mechanism']
        lines.append(f'|{f}|{early["semantic_normalized"]:.5f}/{late["semantic_normalized"]:.5f}|{a["semantic_to_img_grad"]:.5g}/{b["semantic_to_img_grad"]:.5g}|{b["native_to_img_grad"]:.5g}|{b["semantic_to_head_grad"]:.5g}|{a["image_effective_rank"]:.3f}/{b["image_effective_rank"]:.3f}|{comparisons[f]["probability_MAD"]:.6f}|')
    lines+=['','上表语义梯度未乘系数；乘实际epoch系数才是其当轮贡献，epoch0系数0。早/晚梯度在同一次训练forward分别用autograd.grad测量，不用总梯度冒充语义梯度。完整采样、原生/语义Loss、影像方差、匹配/错配余弦均保存在单fold产物history中。',
        '原img_unet与投影头均收到非零语义梯度；这证明路径生效，不能单由Loss下降证明诊断收益。两组最终参数及预测之差同时包含优化、随机前向及选模差异，不能当作严格因果效果。','',
        '|fold|文本中心化能量|文本有效秩|非同人余弦P05/P50/P95|重复描述率|相似度>0.99比例|','|---|---|---|---|---|---|']
    for f in range(10):
        d=cache['folds'][f]['diagnostics']; q=d['cosine_quantiles_0_05_50_95_100']
        lines.append(f'|{f}|{d["centered_energy"]:.6g}|{d["effective_rank"]:.3f}|{q[1]:.5f}/{q[2]:.5f}/{q[3]:.5f}|{d["duplicate_description_fraction"]:.4f}|{d["fraction_training_pairs_cosine_above_099"]:.4f}|')
    lines+=['','固定模板及相近脑网络统计仍可能产生语义重复和假负样本；重复率低也不保证患者级语义可识别。本轮未据测试调模板、温度或Loss权重。','',
        '## Go/No-Go与解释','',f'预注册结论：**{"Go（可考虑扩大至3seed确认）" if go else "No-Go（当前配置不建议直接扩大）"}**。门槛为平均AUC至少+0.5pp、至少6折AUC胜出、平均CE不恶化、原影像编码器语义梯度非零。各项逐一判断，不以单项最高ACC代替。',
        '只有一个seed，这只是探索性筛选；无论是否Go都未证明方法有效。历史监督RFE跨折泄漏和转导人口图限制临床解释。扩展需要另行授权，本轮没有追加seed、Loss或超参数搜索。','',
        '## 执行验收','',
        f'20/20训练完成，C0/S1各10折；PPU任务数{execution["task_counts_by_gpu"]}，组别/卡{execution["task_counts_by_group_gpu"]}。每worker CPU线程1，无DDP、CPU神经训练、VAE预训练或RFE拟合。',
        f'正式队列墙钟{execution["wall_seconds"]:.2f}秒，累计worker训练{execution["sum_training_seconds"]:.2f}秒，两卡训练秒{execution["training_seconds_by_gpu"]}。单worker分配/保留显存峰值分别{execution["peak_worker_allocated_bytes"]/2**30:.3f}/{execution["peak_worker_reserved_bytes"]/2**30:.3f}GiB；并发进程峰值与单worker峰值分开解读。失败/重试记录{len(execution["failures"])}，详细见audit.execution。',
        '两卡确定性诊断各在固定fold0连续三步验证无Hook/惰性Hook的logits、图损失、梯度、Adam、BN、TopK及RNG逐值一致；Checkpoint包含未注册RP权重。生产执行保留历史后端设置，短步确定性等价不保证原生长程逐位复现。',
        f'20份产物内部SHA及输入指纹、两组各871人恰好一次OOF、训练参考、原生单次测试、验证选模全部自动通过；独立指标最大差{max_error:.3g}；{len(audit["protected_files"])}项历史输入与源码SHA不变。代码SHA`{code_hash()}`。',
        f'完整产物：`{BASE}`。配对患者预测见summary.json和runs。handoff最终写入状态在summary.handoff记录，不自动Git提交或push。','']
    report='\n'.join(lines)
    from descriptors import atomic_bytes
    save_json(BASE/'summary.json',summary)
    atomic_bytes(BASE/'report.md',report.encode())
    summary['handoff']=append_handoff(report.replace('# 实验11：','## 实验11：',1))
    summary['report_sha256']=sha_file(BASE/'report.md')
    save_json(BASE/'summary.json',summary)
    audit['acceptance']=summary['acceptance']; audit['handoff']=summary['handoff']; audit['completed_utc']=utc()
    save_json(BASE/'audit.json',audit)
    for path,expected in artifacts.items(): assert sha_file(path)==expected
    protected_guard(audit)
    log(f'FINAL 20/20 Go={go} delta={delta} wins={wins} handoff={summary["handoff"]}')
    return summary


def norm(value):
    return float(value.detach().double().square().sum().sqrt())


if __name__=='__main__':
    analyze()
