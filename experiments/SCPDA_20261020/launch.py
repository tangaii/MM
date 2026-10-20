"""Resource-aware, paired-card supervisor; D0 gate before mixed D1-D7 queue."""
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
    mi=1024**2
    try:
        raw=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free,memory.used,utilization.gpu',
                            '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=3,check=True).stdout
        gpus={}
        for line in raw.strip().splitlines():
            index,free,used,util=map(int,line.split(','))
            gpus[index]={'free_mib':free,'used_mib':used,'utilization':util}
    except (subprocess.SubprocessError,ValueError):
        return None
    c=Path('/sys/fs/cgroup')
    if (c/'memory/memory.limit_in_bytes').exists():
        limit=int((c/'memory/memory.limit_in_bytes').read_text())
        current=int((c/'memory/memory.usage_in_bytes').read_text())
        stats=dict(line.split() for line in (c/'memory/memory.stat').read_text().splitlines())
        inactive=int(stats.get('total_inactive_file',stats.get('inactive_file',0)))
        quota=int((c/'cpu/cpu.cfs_quota_us').read_text());period=int((c/'cpu/cpu.cfs_period_us').read_text())
    else:
        limit=int((c/'memory.max').read_text());current=int((c/'memory.current').read_text())
        stats=dict(line.split() for line in (c/'memory.stat').read_text().splitlines())
        inactive=int(stats.get('inactive_file',0))
        q,p=(c/'cpu.max').read_text().split();quota=int(q) if q!='max' else -1;period=int(p)
    return {'utc':run.utc(),'gpus':gpus,'host_limit_mib':limit/mi,'host_current_mib':current/mi,
            'host_available_mib':(limit-current+inactive)/mi,'cpu_quota':quota/period if quota>0 else os.cpu_count(),
            'load1':os.getloadavg()[0]}


def worker_command(action,group,seed,fold,gpu):
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'PYTHONDONTWRITEBYTECODE':'1','SCPDA_LOG_STDOUT':'1',
         'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
    if action=='preflight':
        env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    else:
        env.pop('CUBLAS_WORKSPACE_CONFIG',None)
    command=[sys.executable,'-B','-u',str(run.BASE/'run.py'),action,'--seed',str(seed),'--fold',str(fold)]
    if group:
        command+=['--group',group]
    return command,env


def preflight(a):
    for s,f in run.CFG['preflight_anchors']:
        a=run.read_json(run.BASE/'manifest.json')
        if a['preflight'].get(f'{s}/{f}',{}).get('passed'):
            continue
        command,env=worker_command('preflight',None,s,f,f%2)
        run.log('preflight_dispatch',seed=s,fold=f,gpu=f%2,command=command)
        with (run.BASE/'run.log').open('a') as out:
            proc=subprocess.Popen(command,cwd=run.BASE,env=env,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
            rc=proc.wait()
        if rc:
            raise RuntimeError(f'Preflight failed {s}/{f}; inspect run.log; formal training blocked')
    a=run.read_json(run.BASE/'manifest.json')
    assert len(a['preflight'])==2 and all(v['passed'] for v in a['preflight'].values())
    a['formal_authorized']=True
    a['preflight_completed_utc']=run.utc()
    run.atomic_json(a,run.BASE/'manifest.json')


def queue(action,tasks,a):
    """Only coordinator reads finished artifacts and writes shared log/manifest."""
    pending=[];skipped=0
    for g,s,f in tasks:
        v=run.valid_result(g,s,f,a)
        done=(v is not None) and (action!='frozen' or 'frozen' in v)
        if done:
            skipped+=1
        else:
            pending.append((g,s,f))
    active=[]; attempts={};failed=[];selector=selectors.DefaultSelector()
    caps={0:run.CFG['workers_per_gpu'],1:run.CFG['workers_per_gpu']}
    execution=a.setdefault('execution',{})
    execution.setdefault('events',[]);execution.setdefault('failures',[]);execution.setdefault('resources',[])
    started=time.perf_counter();last_sample=0;last_progress=0;idle_since=None
    run.log('queue_begin',action=action,pending=len(pending),skipped=skipped)
    with (run.BASE/'run.log').open('a',buffering=1) as sink:
        while pending or active:
            for event,_ in selector.select(timeout=.2):
                line=event.fileobj.readline()
                if line:
                    sink.write(line)
                    item=next((x for x in active if x['process'].stdout is event.fileobj),None)
                    if item is not None:
                        item['tail']=(item['tail']+line)[-12000:]
                else:
                    selector.unregister(event.fileobj);event.fileobj.close()
            for item in active[:]:
                rc=item['process'].poll()
                if rc is None:
                    continue
                active.remove(item)
                g,s,f=item['task'];gpu=item['gpu']
                if rc==0:
                    v=run.valid_result(g,s,f,a,True)
                    assert action!='frozen' or 'frozen' in v
                    execution['events'].append({'action':action,'task':[g,s,f],'gpu':gpu,'pid':item['process'].pid,
                                                'started':item['utc'],'finished':run.utc(),'seconds':time.perf_counter()-item['started']})
                else:
                    tail=item['tail']
                    record={'action':action,'task':[g,s,f],'gpu':gpu,'returncode':rc,'utc':run.utc(),'tail':tail[-5000:]}
                    oom='out of memory' in tail.lower()
                    missing='FileNotFoundError' in tail and 'Input/output error' not in tail
                    record['oom']=oom;execution['failures'].append(record)
                    caps[gpu]=max(1,caps[gpu]-1) if oom else caps[gpu]
                    key=(g,s,f)
                    if (oom or missing) and attempts[key]<run.CFG['max_attempts']:
                        pending.append(key)
                        run.log('retry_incomplete',task=key,gpu=gpu,oom=oom,attempts=attempts[key])
                    else:
                        failed.append(key)
                a['execution']=execution;run.atomic_json(a,run.BASE/'manifest.json')
            if failed:
                # Drain already healthy jobs, stop new dispatch. No healthy worker is killed.
                if not active:
                    raise RuntimeError(f'{len(failed)} tasks failed; no silent skip: {failed}')
                continue
            now=time.perf_counter()
            if now-last_sample>=30:
                sample=resources()
                execution['resources'].append(sample or {'utc':run.utc(),'query_timeout':True})
                last_sample=now
            for gpu in (0,1):
                own=[x for x in active if x['gpu']==gpu]
                candidates=[x for x in pending if x[2]%2==gpu]
                if not candidates or len(own)>=caps[gpu]:
                    continue
                r=resources()  # Fresh query for every single dispatch.
                if r is None:
                    continue
                # Account for not-yet-visible allocations and additional growth of existing workers.
                if any(now-x['started']<8 for x in own):
                    continue
                mem=r['gpus'][gpu]
                growth=max(0,len(own)*run.CFG['gpu_worker_budget_mib']-mem['used_mib'])
                need=run.CFG['gpu_reserve_mib']+run.CFG['gpu_worker_budget_mib']+growth
                if mem['free_mib']<need or r['host_available_mib']<run.CFG['host_reserve_mib']+run.CFG['host_worker_budget_mib']:
                    continue
                if len(active)>=max(1,int(r['cpu_quota'])) or r['load1']>max(2*r['cpu_quota'],64):
                    continue
                g,s,f=candidates[0];pending.remove((g,s,f))
                command,env=worker_command(action,g,s,f,gpu)
                p=subprocess.Popen(command,cwd=run.BASE,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                   text=True,bufsize=1,start_new_session=True)
                selector.register(p.stdout,selectors.EVENT_READ)
                active.append({'task':(g,s,f),'gpu':gpu,'process':p,'started':time.perf_counter(),'utc':run.utc(),'tail':''})
                attempts[(g,s,f)]=attempts.get((g,s,f),0)+1
                run.log('dispatch',action=action,group=g,seed=s,fold=f,gpu=gpu,pid=p.pid,command=command)
                idle_since=None
            if pending and not active:
                idle_since=idle_since or time.perf_counter()
                if time.perf_counter()-idle_since>300:
                    raise RuntimeError('Resource gate blocked dispatch for 5 minutes; healthy external jobs untouched')
            if now-last_progress>60:
                run.log('queue_status',action=action,pending=len(pending),active=len(active),finished_this_queue=len(tasks)-len(pending)-len(active),caps=caps)
                last_progress=now
            time.sleep(.3)
    execution.setdefault('queues',[]).append({'action':action,'tasks':len(tasks),'skipped':skipped,'wall_seconds':time.perf_counter()-started,'finished_utc':run.utc()})
    a['execution']=execution;run.atomic_json(a,run.BASE/'manifest.json')
    run.log('queue_complete',action=action,tasks=len(tasks),skipped=skipped,seconds=time.perf_counter()-started)



def baseline_gate(a,group,historical):
    import numpy as np
    values=[run.valid_result(group,911,f,a,True) for f in range(10)]
    fresh={k:float(np.mean([v['metrics'][k] for v in values])) for k in ('ACC','SEN','SPE','AUC')}
    old=a['historical_seed911'][historical]
    delta={k:100*(fresh[k]-old[k]) for k in fresh}
    old_rows=[run.load_torch(run.HIST/'runs'/historical/f's911_f{f}.pt',artifact=True) for f in range(10)]
    epoch_delta=[v['best_epoch']-w['best_epoch'] for v,w in zip(values,old_rows)]
    assert all(v['formal_start_rng_sha256']==a['anchors'][f'911/{v["fold"]}']['pairing']['classifier_start_rng'] for v in values)
    assert all(v['physical_gpu']==v['fold']%2 and v['formal_test_grad_enabled'] for v in values)
    assert all(x['passed'] and x['maximum_native_error']==0 and x['maximum_D4_error']==0 for x in a['preflight'].values())
    gate={'fresh':fresh,'historical':old,'delta_pp':delta,'threshold_for_review_pp':run.CFG['baseline_drift_pp'],
          'within_expected_drift':all(abs(delta[k])<=run.CFG['baseline_drift_pp'] for k in ('ACC','AUC')),
          'best_epoch_differences':epoch_delta,'equal_best_epoch_folds':sum(x==0 for x in epoch_delta),
          'accepted_after_exact_short_trajectory_and_input_protocol_checks':True,
          'production_long_trajectory':'native PPU not guaranteed bitwise reproducible; no favorable-score reruns'}
    a.setdefault('historical_replay_checks',{})[group]=gate
    a['baseline_accepted' if group=='E0' else 'd4_accepted']=True
    run.atomic_json(a,run.BASE/'manifest.json')
    run.log('historical_replay_check',group=group,**gate)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['preflight','baseline','d4','methods','all'],default='all')
    args=p.parse_args()
    with (run.BASE/'launch.py').open('r+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        a=run.audit()
        if args.stage in ('all','preflight'):
            preflight(a);a=run.guard()
        if args.stage in ('all','baseline'):
            queue('train',[('E0',911,f) for f in range(10)],a)
            baseline_gate(a,'E0','D0')
        if args.stage in ('all','d4'):
            a=run.guard();assert a['baseline_accepted']
            queue('train',[('E1',911,f) for f in range(10)],a)
            baseline_gate(a,'E1','D4')
        if args.stage in ('all','methods'):
            a=run.guard();assert a['d4_accepted']
            groups=['E2','E3','E4','E5'];tasks=[]
            for f in range(10):
                order=groups[f%4:]+groups[:f%4]
                tasks.extend((g,911,f) for g in order)
            queue('train',tasks,a)
        if args.stage=='all':
            subprocess.run([sys.executable,'-B',str(run.BASE/'analyze.py'),'--final'],cwd=run.BASE,check=True)


if __name__=='__main__':
    try: main()
    except Exception as e:
        run.log('coordinator_error',message=str(e),traceback=traceback.format_exc())
        raise

