"""Paired-card resource-aware queue; N0 gate precedes mixed N1-N7 training."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode=True
import run


def resources():
    try:
        raw=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free,memory.used,utilization.gpu',
                            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3,check=True).stdout
        gpus={}
        for line in raw.strip().splitlines():
            index,free,used,util=map(int,line.split(','))
            gpus[index]={'free_mib':free,'used_mib':used,'utilization':util}
    except (subprocess.SubprocessError,ValueError):
        return None
    c=Path('/sys/fs/cgroup'); mi=1024**2
    if (c/'memory/memory.limit_in_bytes').exists():
        limit=int((c/'memory/memory.limit_in_bytes').read_text()); current=int((c/'memory/memory.usage_in_bytes').read_text())
        stats=dict(line.split() for line in (c/'memory/memory.stat').read_text().splitlines())
        inactive=int(stats.get('total_inactive_file',stats.get('inactive_file',0)))
        quota=int((c/'cpu/cpu.cfs_quota_us').read_text()); period=int((c/'cpu/cpu.cfs_period_us').read_text())
    else:
        limit=int((c/'memory.max').read_text()); current=int((c/'memory.current').read_text())
        stats=dict(line.split() for line in (c/'memory.stat').read_text().splitlines()); inactive=int(stats.get('inactive_file',0))
        q,p=(c/'cpu.max').read_text().split(); quota=int(q) if q!='max' else -1; period=int(p)
    return {'utc':run.utc(),'gpus':gpus,'host_limit_mib':limit/mi,'host_current_mib':current/mi,
            'host_available_mib':(limit-current+inactive)/mi,'cpu_quota':quota/period if quota>0 else os.cpu_count(),
            'load1':os.getloadavg()[0]}


def command(action,group,seed,fold,gpu):
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONDONTWRITEBYTECODE':'1','NIPA_LOG_STDOUT':'1',
         'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
    if action=='preflight':
        env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    else:
        env.pop('CUBLAS_WORKSPACE_CONFIG',None)
    cmd=[sys.executable,'-B','-u',str(run.BASE/'run.py'),action,'--seed',str(seed),'--fold',str(fold)]
    if group:
        cmd+=['--group',group]
    return cmd,env


def preflight():
    for s,f in run.CFG['preflight_anchors']:
        a=run.guard()
        if a['preflight'].get(f'{s}/{f}',{}).get('passed'):
            continue
        cmd,env=command('preflight',None,s,f,f%2)
        run.log('preflight_dispatch',seed=s,fold=f,gpu=f%2,command=cmd)
        with (run.BASE/'run.log').open('a') as out:
            p=subprocess.Popen(cmd,cwd=run.BASE,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            rc=p.wait()
        if rc:
            raise RuntimeError(f'Preflight failed {s}/{f}; formal training blocked; inspect run.log')
        print(f'Preflight passed: {s}/{f}',flush=True)
    a=run.guard(); assert len(a['preflight'])==5 and all(x['passed'] for x in a['preflight'].values())
    a['formal_authorized']=True; a['preflight_completed_utc']=run.utc(); run.atomic_json(a,run.BASE/'audit.json')


def task_busy(group,seed,fold):
    slot=int(group[1:])*30+run.CFG['seeds'].index(seed)*10+fold
    with (run.BASE/'run.py').open('r+') as f:
        try:
            fcntl.lockf(f,fcntl.LOCK_EX|fcntl.LOCK_NB,1,1000+slot)
        except BlockingIOError:
            return True
        fcntl.lockf(f,fcntl.LOCK_UN,1,1000+slot)
    return False


def queue(action,tasks):
    a=run.guard(); pending=[]; skipped=0
    def finished(g,s,f):
        v=run.valid_result(g,s,f,a)
        return v is not None and (action!='frozen' or 'frozen' in v)
    for task in tasks:
        if finished(*task):
            skipped+=1
        else:
            pending.append(task)
    active=[]; attempts={}; failed=[]; selector=selectors.DefaultSelector()
    caps={0:run.CFG['workers_per_gpu'],1:run.CFG['workers_per_gpu']}
    execution=a.setdefault('execution',{})
    for k in ('events','failures','resources','queues'):
        execution.setdefault(k,[])
    started=time.perf_counter(); last_sample=0; last_progress=0; idle_since=None
    run.log('queue_begin',action=action,pending=len(pending),skipped=skipped)
    with (run.BASE/'run.log').open('a',buffering=1) as sink:
        while pending or active:
            for event,_ in selector.select(timeout=.2):
                line=event.fileobj.readline()
                if line:
                    sink.write(line)
                    item=next((x for x in active if x['process'].stdout is event.fileobj),None)
                    if item is not None:
                        item['tail']=(item['tail']+line)[-14000:]
                else:
                    selector.unregister(event.fileobj); event.fileobj.close()
            for item in active[:]:
                rc=item['process'].poll()
                if rc is None:
                    continue
                active.remove(item); g,s,f=item['task']; gpu=item['gpu']
                if rc==0:
                    assert finished(g,s,f), f'Exit zero without valid completed result: {item["task"]}'
                    execution['events'].append({'action':action,'task':[g,s,f],'gpu':gpu,'pid':item['process'].pid,
                        'started':item['utc'],'finished':run.utc(),'seconds':time.perf_counter()-item['started']})
                else:
                    tail=item['tail']; oom='out of memory' in tail.lower()
                    missing='FileNotFoundError' in tail and 'Input/output error' not in tail
                    execution['failures'].append({'action':action,'task':[g,s,f],'gpu':gpu,'returncode':rc,
                        'utc':run.utc(),'tail':tail[-5000:],'oom':oom})
                    if oom:
                        caps[gpu]=max(1,caps[gpu]-1)
                    if (oom or missing) and attempts[(g,s,f)]<run.CFG['max_attempts']:
                        pending.append((g,s,f)); run.log('retry_incomplete',task=[g,s,f],oom=oom,attempts=attempts[(g,s,f)])
                    else:
                        failed.append((g,s,f))
                run.atomic_json(a,run.BASE/'audit.json')
            if failed:
                if not active:
                    raise RuntimeError(f'Failed tasks, never silently skipped: {failed}')
                continue
            now=time.perf_counter()
            if now-last_sample>=30:
                execution['resources'].append(resources() or {'utc':run.utc(),'query_timeout':True}); last_sample=now
            for gpu in (0,1):
                own=[x for x in active if x['gpu']==gpu]
                candidates=[x for x in pending if x[2]%2==gpu]
                if not candidates or len(own)>=caps[gpu]:
                    continue
                task=next((x for x in candidates if not task_busy(*x)),None)
                if task is None:
                    continue
                if finished(*task):
                    pending.remove(task); skipped+=1; continue
                r=resources()
                if r is None or any(now-x['started']<8 for x in own):
                    continue
                mem=r['gpus'][gpu]
                growth=max(0,len(own)*run.CFG['gpu_worker_budget_mib']-mem['used_mib'])
                need=run.CFG['gpu_reserve_mib']+run.CFG['gpu_worker_budget_mib']+growth
                if mem['free_mib']<need or r['host_available_mib']<run.CFG['host_reserve_mib']+run.CFG['host_worker_budget_mib']:
                    continue
                if len(active)>=max(1,int(r['cpu_quota'])) or r['load1']>max(2*r['cpu_quota'],64):
                    continue
                g,s,f=task; pending.remove(task); cmd,env=command(action,g,s,f,gpu)
                p=subprocess.Popen(cmd,cwd=run.BASE,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                   text=True,bufsize=1,start_new_session=True)
                selector.register(p.stdout,selectors.EVENT_READ)
                active.append({'task':task,'gpu':gpu,'process':p,'started':time.perf_counter(),'utc':run.utc(),'tail':''})
                attempts[task]=attempts.get(task,0)+1; idle_since=None
                run.log('dispatch',action=action,group=g,seed=s,fold=f,gpu=gpu,pid=p.pid,command=cmd)
            if pending and not active:
                idle_since=idle_since or time.perf_counter()
                if time.perf_counter()-idle_since>600:
                    raise RuntimeError('Resource/active-task gate blocked for ten minutes; no healthy job interrupted')
            if now-last_progress>=60:
                run.log('queue_status',action=action,pending=len(pending),active=len(active),caps=caps)
                print(f'{action}: pending={len(pending)}, active={len(active)}, skipped={skipped}',flush=True)
                last_progress=now
            time.sleep(.3)
    execution['queues'].append({'action':action,'tasks':len(tasks),'skipped':skipped,'wall_seconds':time.perf_counter()-started,'finished_utc':run.utc()})
    run.atomic_json(a,run.BASE/'audit.json')
    run.log('queue_complete',action=action,tasks=len(tasks),skipped=skipped,seconds=time.perf_counter()-started)
    print(f'{action} queue complete: {len(tasks)} tasks, {skipped} skipped',flush=True)


def baseline_gate():
    import numpy as np
    a=run.guard()
    rows=[run.valid_result('N0',s,f,a,True) for s in run.CFG['seeds'] for f in range(10)]
    fresh={k:float(np.mean([v['metrics'][k] for v in rows])) for k in ('ACC','AUC')}
    comparisons={}
    for name,field,ef in (('historical_C0','historical_C0_metrics','historical_C0_best_epoch'),
                          ('previous_D0','previous_D0_metrics','previous_D0_best_epoch')):
        old={k:float(np.mean([v[field][k] for v in a['anchors'].values()])) for k in fresh}
        delta={k:100*(fresh[k]-old[k]) for k in fresh}
        epoch=[v['best_epoch']-a['anchors'][f'{v["seed"]}/{v["fold"]}'][ef] for v in rows]
        comparisons[name]={'metrics':old,'delta_pp':delta,'within_threshold':all(abs(d)<=run.CFG['baseline_drift_pp'] for d in delta.values()),
            'best_epoch_difference':run.describe(epoch),'equal_best_epoch_folds':sum(x==0 for x in epoch)}
    gate={'fresh':fresh,'comparisons':comparisons,'threshold_pp':run.CFG['baseline_drift_pp'],
          'no_high_score_retries':True,'native_nondeterminism_not_structural_equivalence':'Five anchors official AST/N0/all_real exact three-step test; original production backend retained'}
    a['baseline_gate']=gate; a['baseline_accepted']=all(v['within_threshold'] for v in comparisons.values())
    run.atomic_json(a,run.BASE/'audit.json'); run.log('baseline_gate',**gate)
    print('Baseline gate:',gate,flush=True)
    if not a['baseline_accepted']:
        raise RuntimeError('Fresh baseline drift exceeds 0.5pp; inspect saved trajectories and protocol before methods; no favorable reruns')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--stage',choices=['preflight','baseline','methods','frozen','analyze','all'],default='all')
    args=p.parse_args()
    with (run.BASE/'launch.py').open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        run.audit()
        if args.stage in ('all','preflight'):
            preflight()
        if args.stage in ('all','baseline'):
            assert run.guard()['formal_authorized']
            queue('train',[('N0',s,f) for s in run.CFG['seeds'] for f in range(10)]); baseline_gate()
        if args.stage in ('all','methods'):
            a=run.guard(); assert a['formal_authorized'] and a['baseline_accepted']
            tasks=[]; groups=list(run.CFG['groups'])[1:]
            for s in run.CFG['seeds']:
                for f in range(10):
                    order=groups[f%7:]+groups[:f%7]; tasks.extend((g,s,f) for g in order)
            queue('train',tasks)
        if args.stage in ('all','frozen'):
            a=run.guard()
            assert all(run.valid_result(g,s,f,a) is not None for g in run.CFG['groups'] for s in run.CFG['seeds'] for f in range(10))
            queue('frozen',[('N0',s,f) for s in run.CFG['seeds'] for f in range(10)])
        if args.stage in ('all','analyze'):
            subprocess.run([sys.executable,'-B',str(run.BASE/'analyze.py'),'--final'],cwd=run.BASE,check=True)


if __name__=='__main__':
    try:
        main()
    except Exception as e:
        run.log('coordinator_error',message=str(e),traceback=traceback.format_exc())
        raise
