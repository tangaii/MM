"""Single-owner resource-aware queue; training scripts remain independent."""
import fcntl
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time
import run as r


def log(event, **values):
    line=json.dumps({'time_utc':r.utc(),'event':event,**values},ensure_ascii=False,allow_nan=False)
    with (r.BASE/'run.log').open('a') as f:
        f.write(line+'\n')
    print(line,flush=True)


class Resources:
    def __init__(self):
        self.budget={0:16384,1:16384}
        self.penalty={0:0,1:0}
        self.previous_cpu=None

    def sample(self,active):
        output=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True,timeout=10)
        devices={}
        for row in output.splitlines():
            gpu,total,used,util=[int(x.strip()) for x in row.split(',')]
            devices[gpu]={'total_mib':total,'used_mib':used,'utilization_percent':util}
        if not all(g in devices for g in (0,1)):
            raise RuntimeError('Two PPU devices are required')
        memory={row.split(':')[0]:int(row.split()[1])//1024 for row in Path('/proc/meminfo').read_text().splitlines() if len(row.split())>=2}
        cg=Path('/sys/fs/cgroup/cpuacct/cpuacct.usage')
        now=time.monotonic()
        cpu_usage=int(cg.read_text())/1e9 if cg.exists() else 0.0
        cpu_cores=0.0
        if self.previous_cpu:
            cpu_cores=(cpu_usage-self.previous_cpu[1])/max(now-self.previous_cpu[0],.01)
        self.previous_cpu=(now,cpu_usage)
        caps={}
        for g in (0,1):
            n=sum(w['gpu']==g for w in active)
            used=devices[g]['used_mib']
            # Reserve 16 GiB for every started process, including not-yet-initialized ones.
            reserved=max(used,n*self.budget[g])
            slots=max(0,(devices[g]['total_mib']-8192-reserved)//self.budget[g])
            caps[g]=min(4-self.penalty[g],n+slots)
            devices[g]['active']=n
            devices[g]['limit']=caps[g]
        limit=8
        if memory['MemAvailable']<16384 or cpu_cores>15.5:
            limit=len(active)
        return {'time_utc':r.utc(),'devices':devices,'ram_available_mib':memory['MemAvailable'],
                'cgroup_cpu_cores':cpu_cores,'cpu_budget':16},caps,limit


def identity(action,group,seed,fold):
    return f'{action}/{group}/{seed}/{fold}'


def attempt_budget(task,state):
    extra=state.get('recovery_authorizations',{}).get(identity(**task),{}).get('additional_attempts',0)
    if task['action']!='diagnose':
        extra=0
    return r.config()['max_attempts']+extra


def finished(action,group,seed,fold,state):
    key=f'{seed}/{fold}'
    if action=='check':
        return state.get('checks',{}).get(key,{}).get('passed',False)
    if action=='diagnose':
        return key in state.get('diagnostics',{})
    if action=='repair':
        if key not in state.get('counterfactual_repairs',{}):
            return False
        import analyze
        for method in ('R2','R3'):
            analyze.verified_counterfactual(r.load_torch(r.result_path(method,seed,fold),mmap=True))
        return True
    return r.verify_result(group,seed,fold,state,complete=action=='frozen')


def queue(action,state):
    groups=r.GROUPS if action in ('train','frozen') else ('R0',)
    pending=[]
    for seed in r.config()['seeds']:
        for fold in range(10):
            for group in groups:
                if not finished(action,group,seed,fold,state):
                    pending.append({'action':action,'group':group,'seed':seed,'fold':fold})
                elif action in ('train','frozen'):
                    key=identity(action,group,seed,fold)
                    saved=r.load_torch(r.result_path(group,seed,fold),mmap=True)
                    state.setdefault('tasks',{}).setdefault(key,{})
                    state['tasks'][key].update(state='complete',gpu=saved['physical_gpu'],verified_resume_utc=r.utc())
    if not pending:
        log('phase_verified_skip',action=action)
        state.setdefault('scheduling',{}).setdefault('phases',{}).setdefault(action,{})['verified_resume_utc']=r.utc()
        r.atomic_json(state)
        return
    selector=selectors.DefaultSelector()
    active=[]
    resources=Resources()
    started=time.perf_counter()
    latest_resource=0.0
    state['phase']=action
    phases=state.setdefault('scheduling',{}).setdefault('phases',{})
    previous=phases.get(action)
    previous_elapsed=previous.get('elapsed_seconds',0.0) if previous else 0.0
    phases[action]={'started_utc':previous['started_utc'] if previous else r.utc(),
                    'pending_at_start':previous['pending_at_start'] if previous else len(pending),
                    'pending_this_invocation':len(pending),'this_invocation_started_utc':r.utc()}
    if previous:
        phases[action]['previous_invocations']=previous.get('previous_invocations',[])+[
            {k:v for k,v in previous.items() if k!='previous_invocations'}]
    r.atomic_json(state)
    while pending or active:
        now=time.monotonic()
        if now-latest_resource>5 or not active:
            snapshot,caps,limit=resources.sample(active)
            state['scheduling']['latest_resources']=snapshot
            latest_resource=now
        for gpu in (0,1):
            while sum(w['gpu']==gpu for w in active)<caps[gpu] and len(active)<limit and pending:
                candidates=[(i,t) for i,t in enumerate(pending) if t['fold']%2==gpu]
                if not candidates:
                    candidates=list(enumerate(pending))
                selected,task=candidates[0]
                key=identity(**task)
                attempts=state.setdefault('attempts',{}).get(key,0)
                if attempts>=attempt_budget(task,state):
                    raise RuntimeError(f'Failure budget exhausted: {key}')
                pending.pop(selected)
                env=os.environ.copy()
                env['CUDA_VISIBLE_DEVICES']=str(gpu)
                env['PYTHONDONTWRITEBYTECODE']='1'
                for n in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
                    env[n]='1'
                if action=='check':
                    env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
                else:
                    env.pop('CUBLAS_WORKSPACE_CONFIG',None)
                entry='analyze.py' if action=='repair' else 'run.py'
                cmd=[sys.executable,'-u','-B',str(r.BASE/entry),action,'--group',task['group'],
                     '--seed',str(task['seed']),'--fold',str(task['fold'])]
                proc=subprocess.Popen(cmd,cwd=r.BASE,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                worker={'proc':proc,'gpu':gpu,'task':task,'key':key,'result':None,'tail':[],'started':time.perf_counter()}
                active.append(worker)
                selector.register(proc.stdout,selectors.EVENT_READ,worker)
                state['attempts'][key]=attempts+1
                state.setdefault('tasks',{})[key]={'state':'running','gpu':gpu,'pid':proc.pid,'started_utc':r.utc()}
                if gpu!=task['fold']%2:
                    state['scheduling'].setdefault('cross_gpu_assignments',[]).append({**task,'gpu':gpu,'reason':'preferred card queue empty'})
                peaks=state['scheduling'].setdefault('peak_workers',{'0':0,'1':0})
                peaks[str(gpu)]=max(peaks[str(gpu)],sum(w['gpu']==gpu for w in active))
                log('started',**task,gpu=gpu,pid=proc.pid)
                r.atomic_json(state)
        for key,_ in selector.select(timeout=1):
            worker=key.data
            line=key.fileobj.readline()
            if not line:
                selector.unregister(key.fileobj)
                key.fileobj.close()
                continue
            worker['tail'].append(line.rstrip())
            worker['tail']=worker['tail'][-12:]
            try:
                message=json.loads(line)
            except json.JSONDecodeError:
                log('worker_output',task=worker['key'],text=line.rstrip()[:1500])
                continue
            if message.get('kind')=='done':
                worker['result']=message['result']
            elif message.get('kind')=='progress':
                log('progress',task=worker['key'],epoch=message['epoch'],validation_loss=message['validation_loss'])
        for worker in list(active):
            proc=worker['proc']
            if proc.poll() is None or proc.stdout in selector.get_map().values():
                continue
            # Drain the final event even if process exit and stdout readiness coincide.
            if not proc.stdout.closed:
                remaining=proc.stdout.read()
                for line in remaining.splitlines():
                    try:
                        message=json.loads(line)
                        if message.get('kind')=='done': worker['result']=message['result']
                    except json.JSONDecodeError:
                        worker['tail'].append(line)
                try: selector.unregister(proc.stdout)
                except (KeyError,ValueError): pass
                proc.stdout.close()
            active.remove(worker)
            task,key=worker['task'],worker['key']
            if proc.returncode!=0 or worker['result'] is None:
                error={'task':key,'gpu':worker['gpu'],'exit_code':proc.returncode,'tail':worker['tail'][-12:],'time_utc':r.utc()}
                state['errors'].append(error)
                state['tasks'][key]={'state':'failed',**error}
                log('failed',**error)
                if any('out of memory' in s.lower() for s in worker['tail']):
                    resources.penalty[worker['gpu']]=min(3,resources.penalty[worker['gpu']]+1)
                    resources.budget[worker['gpu']]=int(resources.budget[worker['gpu']]*1.25)
                if state['attempts'][key]<attempt_budget(task,state):
                    pending.append(task)
                else:
                    r.atomic_json(state)
                    # Let healthy tasks finish before reporting a failed phase.
                    state['scheduling'].setdefault('exhausted',[]).append(key)
            else:
                result=worker['result']
                state['tasks'][key]={'state':'complete','gpu':worker['gpu'],'completed_utc':r.utc(),
                                     'elapsed_seconds':time.perf_counter()-worker['started']}
                if action=='check': state['checks'][f"{task['seed']}/{task['fold']}"]=result
                elif action=='diagnose': state.setdefault('diagnostics',{})[f"{task['seed']}/{task['fold']}"]=result
                elif action=='repair': state.setdefault('counterfactual_repairs',{})[f"{task['seed']}/{task['fold']}"]=result
                else:
                    if not finished(action,task['group'],task['seed'],task['fold'],state):
                        raise RuntimeError('Worker artifact validation failed')
                log('complete',**task,gpu=worker['gpu'],result=result if action in ('train','frozen') else {'passed':True})
            r.atomic_json(state)
        if time.monotonic()-state['scheduling'].get('_last_log_monotonic',0)>30:
            log('queue_status',phase=action,pending=len(pending),active=len(active),resources=snapshot)
            state['scheduling']['_last_log_monotonic']=time.monotonic()
    selector.close()
    state['scheduling']['phases'][action].update(completed_utc=r.utc(),elapsed_seconds=previous_elapsed+time.perf_counter()-started)
    r.atomic_json(state)
    unresolved=[key for key in state['scheduling'].get('exhausted',[]) if state['tasks'][key]['state']!='complete']
    if unresolved:
        raise RuntimeError('One or more tasks exhausted recovery budget')
    if state['scheduling'].get('exhausted'):
        state['scheduling'].setdefault('recovered_exhausted',[]).extend(state['scheduling'].pop('exhausted'))
        r.atomic_json(state)


def main():
    with (r.BASE/'launch.py').open('r+') as owner:
        fcntl.flock(owner.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (r.BASE/'summary.json').exists():
            state=r.guard()
        else:
            log('audit_start')
            state={'experiment':r.config()['experiment'],'created_utc':r.utc(),'config':r.config(),
                   'training_code_sha256':r.code_hash(),'audit':r.audit(),'checks':{},'tasks':{},'attempts':{},'errors':[],'scheduling':{}}
            r.atomic_json(state)
        if state.get('final_verification',{}).get('passed'):
            import analyze
            result=analyze.verify_completed_bundle(state)
            log('verified_complete_resume',phase=state['phase'],verification=result)
            return
        if state.get('phase')=='complete':
            log('verified_complete_resume')
            return
        queue('check',state)
        if len(state['checks'])!=30 or not all(v['passed'] for v in state['checks'].values()):
            raise AssertionError('All 30-fold checks must pass before formal training')
        queue('train',state)
        queue('frozen',state)
        queue('repair',state)
        queue('diagnose',state)
        import analyze
        analyze.finalize(state)
        log('all_research_complete',phase=state['phase'])


if __name__=='__main__':
    main()
