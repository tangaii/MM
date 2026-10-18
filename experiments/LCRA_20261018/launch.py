"""Resource-guarded, paired two-PPU acceptance/development/formal task queue."""
from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time
import traceback
sys.dont_write_bytecode = True
BASE = Path(__file__).resolve().parent
sys.path.insert(0,str(BASE))
import run as ex
CFG = ex.CFG


def resources():
    output = subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True,timeout=20)
    gpu = {}
    for line in output.strip().splitlines():
        index,total,used,util = [int(v.strip()) for v in line.split(',')]
        gpu[index] = {'total_mib':total,'used_mib':used,'utilization':util}
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key,value = line.split(':',1)
        memory[key] = int(value.strip().split()[0])*1024
    cgroup = Path('/sys/fs/cgroup/memory')
    if cgroup.exists():
        limit = int((cgroup/'memory.limit_in_bytes').read_text())
        used = int((cgroup/'memory.usage_in_bytes').read_text())
        oom = int((cgroup/'memory.failcnt').read_text())
    else:
        cg = Path('/sys/fs/cgroup')
        raw = (cg/'memory.max').read_text().strip()
        limit = int(raw) if raw!='max' else memory['MemTotal']
        used = int((cg/'memory.current').read_text())
        oom = int(dict(line.split() for line in (cg/'memory.events').read_text().splitlines())['oom'])
    return {'utc':ex.utc(),'gpu':gpu,'ram_available_mib':min(memory['MemAvailable'],limit-used)/2**20,
        'cgroup_limit_mib':limit/2**20,'cgroup_used_mib':used/2**20,'cgroup_failcnt':oom,'load1':os.getloadavg()[0]}


def gpu_for(seed,fold):
    return (CFG['seeds'].index(seed)*10+fold)%2


def key_for(action,group,seed,fold):
    return f'{action}/{group}/{seed}/{fold}'


class Scheduler:
    def __init__(self):
        self.path = BASE/'summary.json'
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            'experiment':CFG['experiment'],'status':'auditing','started_utc':ex.utc(),'tasks':{},'failures':[],
            'preflight':{'checks':{},'formal_authorized':False},'resource_samples':[],
            'wall_seconds_accumulated':0.0,'formal_target':120,'cpu_training':False}
        self.start_time = time.monotonic()
        self.active = {}; self.selector = selectors.DefaultSelector()
        self.last_sample = self.last_dispatch = self.last_save = 0.0
        self.stop_dispatch = False
        self.caps = {0:2,1:2}
        self.host_budget = CFG['initial_host_worker_budget_mib']
        self.gpu_budget = CFG['initial_gpu_worker_budget_mib']
        for row in self.state['tasks'].values():
            if row.get('state')=='running':
                try: os.kill(row['pid'],0)
                except ProcessLookupError: row['state']='interrupted'
                else: raise RuntimeError(f'Prior worker still alive: {row["pid"]}')

    def save(self):
        self.state['updated_utc'] = ex.utc()
        self.state['wall_seconds_current_session'] = time.monotonic()-self.start_time
        self.state['concurrency_caps'] = self.caps
        ex.atomic_json(self.state,self.path)
        self.last_save = time.monotonic()

    def valid(self,action,group,seed,fold):
        key = key_for(action,group,seed,fold)
        row = self.state['tasks'].get(key,{})
        if action=='check':
            value = self.state['preflight']['checks'].get(f'{seed}/{fold}')
            return bool(value and value['passed'] and value['code_sha256']==ex.code_hash())
        path = ex.task_path('train' if action=='frozen' else action,group,seed,fold)
        if not path.exists(): return False
        value = ex.load_torch(path)
        assert value['state']=='complete' and value['code_sha256']==ex.code_hash()
        assert (value['group'],value['seed'],value['fold'])==(group,seed,fold)
        assert value['best_core_sha256']==ex.core_hash(value['best']['core'])
        assert value['best_aligner_sha256']==ex.state_hash(value['best']['aligner'])
        assert value['input_sha256']==self.audit['anchors'][f'{seed}/{fold}']['inputs_digest']
        current = ex.sha_file(path)
        registry = self.state['tasks'].get(key_for('frozen',group,seed,fold),row) if action=='train' else row
        if registry.get('state')=='complete': assert current==registry['sha256'],f'Artifact checksum changed: {path}'
        if action=='frozen' and 'frozen' not in value: return False
        if action=='develop':
            assert value['epochs_run']==80 and not value['outer_test_metrics_computed'] and 'oof' not in value
        row.update(state='complete',sha256=current,verified_skip=True,gpu=gpu_for(seed,fold))
        if 'result' not in row:
            row['result'] = {'path':str(path),'sha256':current,**value['execution']}
        self.state['tasks'][key]=row
        return True

    def start(self,task,gpu):
        action,group,seed,fold = task
        key = key_for(*task)
        attempts = self.state['tasks'].get(key,{}).get('attempts',0)+1
        assert attempts<=CFG['max_attempts'],f'Too many failures: {key}'
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        if action=='check': env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        else: env.pop('CUBLAS_WORKSPACE_CONFIG',None)
        command = [sys.executable,'-u','-B',str(BASE/'run.py'),action,'--group',group,'--seed',str(seed),'--fold',str(fold)]
        process = subprocess.Popen(command,cwd=BASE,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,start_new_session=True)
        os.set_blocking(process.stdout.fileno(),False)
        self.selector.register(process.stdout,selectors.EVENT_READ,key)
        self.active[key]={'task':task,'gpu':gpu,'process':process,'buffer':b'','tail':'','result':None,'started':time.monotonic()}
        self.state['tasks'][key]={'state':'running','pid':process.pid,'gpu':gpu,'attempts':attempts,'started_utc':ex.utc(),'command':command}
        self.last_dispatch = time.monotonic()
        ex.log('worker_start',task=key,gpu=gpu,pid=process.pid)
        print(f'START {key} PPU{gpu} PID{process.pid}',flush=True)
        self.save()

    def consume(self,key,chunk):
        item = self.active[key]; item['buffer'] += chunk
        while b'\n' in item['buffer']:
            raw,item['buffer']=item['buffer'].split(b'\n',1)
            line = raw.decode(errors='replace'); item['tail']=(item['tail']+'\n'+line)[-15000:]
            try: event=json.loads(line)
            except ValueError: continue
            if not isinstance(event,dict): continue
            if event.get('kind')=='done': item['result']=event['result']
            elif event.get('kind')=='progress':
                ex.log('progress',task=key,epoch=event['epoch'],val_ce=event['val_ce'],val_graph=event['val_graph'])
                print(f'{key} epoch={event["epoch"]} valCE={event["val_ce"]:.6f}',flush=True)

    def tick(self):
        for event,_ in self.selector.select(timeout=2):
            chunk = os.read(event.fileobj.fileno(),65536)
            if chunk: self.consume(event.data,chunk)
        for key,item in list(self.active.items()):
            process=item['process']; rc=process.poll()
            if rc is None: continue
            while True:
                chunk=os.read(process.stdout.fileno(),65536)
                if not chunk: break
                self.consume(key,chunk)
            self.selector.unregister(process.stdout); process.stdout.close()
            row=self.state['tasks'][key]
            row.update(returncode=rc,ended_utc=ex.utc(),wall_seconds=time.monotonic()-item['started'])
            if rc==0 and item['result'] is not None:
                result=item['result']; row.update(state='complete',result=result)
                if 'sha256' in result: row['sha256']=result['sha256']
                if item['task'][0]=='check': self.state['preflight']['checks'][f'{item["task"][2]}/{item["task"][3]}']=result
                self.gpu_budget=max(self.gpu_budget,result.get('peak_allocated_mib',0)*1.4)
                self.host_budget=max(self.host_budget,result.get('host_peak_rss_mib',0)*1.3)
                ex.log('worker_complete',task=key,gpu=item['gpu'],wall_seconds=row['wall_seconds'])
                print(f'DONE {key} {row["wall_seconds"]:.1f}s',flush=True)
            else:
                oom=rc==-9 or any(s in item['tail'].lower() for s in ('out of memory','bad_alloc','cannot allocate memory'))
                # ossfs rename can briefly hide the live status name. This exact failure
                # precedes model construction/optimization; a bounded new admission is safe.
                visibility=(item['task'][0]=='train' and 'FileNotFoundError: [Errno 2]' in item['tail']
                    and str(BASE/'summary.json') in item['tail'] and 'state = json.loads' in item['tail']
                    and '"kind": "progress"' not in item['tail'])
                row.update(state='failed',oom_suspected=oom,startup_state_visibility=visibility,error_tail=item['tail'])
                self.state['failures'].append({'task':key,'gpu':item['gpu'],'returncode':rc,'oom_suspected':oom,
                    'startup_state_visibility':visibility,'tail':item['tail']})
                ex.log('worker_failed',task=key,gpu=item['gpu'],returncode=rc,oom_suspected=oom,
                    startup_state_visibility=visibility,error_tail=item['tail'])
                print(item['tail'],flush=True)
                if oom:
                    self.caps[item['gpu']]=max(1,self.caps[item['gpu']]-1)
                    self.host_budget*=1.25;self.gpu_budget*=1.25
                elif not visibility: self.stop_dispatch=True
            del self.active[key]
            self.save()
        now=time.monotonic()
        if now-self.last_sample>=30:
            try:
                sample=resources(); sample['active_workers']=len(self.active)
                self.state['resource_samples'].append(sample)
            except (OSError,subprocess.SubprocessError) as err:
                ex.log('resource_query_paused',error=repr(err))
            self.last_sample=now
        if now-self.last_save>=30:self.save()

    def queue(self,tasks,maximum):
        pending=[t for t in tasks if not self.valid(*t)]
        while pending or self.active:
            if pending and not self.stop_dispatch and time.monotonic()-self.last_dispatch>=3:
                try: sample=resources()
                except (OSError,subprocess.SubprocessError): sample=None
                if sample:
                    for gpu in (0,1):
                        count=sum(v['gpu']==gpu for v in self.active.values())
                        cap=min(maximum,self.caps[gpu])
                        if maximum>2 and time.monotonic()-self.start_time>60 and not any(
                            not r.get('startup_state_visibility',False) for r in self.state['failures']):
                            if sample['ram_available_mib']>CFG['host_reserve_mib']+2*self.host_budget and sample['load1']<16:
                                cap=min(maximum,cap+1);self.caps[gpu]=cap
                        index=next((i for i,t in enumerate(pending) if gpu_for(t[2],t[3])==gpu),None)
                        if index is None or count>=cap:continue
                        if sample['ram_available_mib']<CFG['host_reserve_mib']+self.host_budget:continue
                        if sample['gpu'][gpu]['total_mib']-sample['gpu'][gpu]['used_mib']<CFG['gpu_reserve_mib']+self.gpu_budget:continue
                        if sample['load1']>32 and self.active:continue
                        task=pending.pop(index)
                        if not self.valid(*task):self.start(task,gpu)
                        sample['ram_available_mib']-=self.host_budget
            self.tick()
            if self.stop_dispatch and not self.active:raise RuntimeError('Structural failure; healthy workers drained; fix then reaccept')
            for t in tasks:
                key=key_for(*t);row=self.state['tasks'].get(key,{})
                if row.get('state')=='failed' and (row.get('oom_suspected') or row.get('startup_state_visibility')) and t not in pending and key not in self.active:
                    if row['attempts']>=CFG['max_attempts']:self.stop_dispatch=True
                    else:pending.append(t);row['state']='retry_queued'
        self.save()

    def main(self):
        self.audit=ex.audit()
        assert self.audit['new_source_sha256']==ex.source_hashes(), 'Accepted experiment source changed'
        import analyze
        if self.state.get('status') in ('preflight_scientific_stop','research_complete') and self.state.get('finished_utc'):
            analyze.verify_terminal(self.state,self.audit)
            print('Terminal result verified; no training or duplicate handoff append.',flush=True)
            return
        self.state['launcher']={'pid':os.getpid(),'command':sys.argv,'utc':ex.utc()}
        self.state['audit_sha256']=ex.sha_file(BASE/'audit.json')
        self.state['status']='phase_A';self.save()
        self.queue([('check','L3',s,f) for s,f in CFG['development_anchors']],maximum=1)
        assert len(self.state['preflight']['checks'])==5
        self.state['status']='phase_B';self.save()
        # Mixed methods, same seed/fold physical card; all twenty tasks are required.
        dev=[('develop',g,s,f) for s,f in CFG['development_anchors'] for g in CFG['groups']]
        self.queue(dev,maximum=CFG['workers_per_gpu_max'])
        gate=analyze.development_gate(self.state)
        self.state['preflight']['scientific_gate']=gate
        self.state['preflight']['formal_authorized']=gate['go']
        self.save()
        if not gate['go']:
            self.state['status']='preflight_scientific_stop'
            analyze.finish(self.state,self.audit,formal=False)
            return
        self.state['status']='formal_training'
        self.state['formal_registration']={'code_sha256':ex.code_hash(),'source_sha256':ex.source_hashes(),'utc':ex.utc()}
        self.save()
        formal=[('train',g,s,f) for s in CFG['seeds'] for f in range(10) for g in CFG['groups']]
        self.queue(formal,maximum=CFG['workers_per_gpu_max'])
        self.state['status']='frozen_counterfactuals';self.save()
        self.queue([('frozen',g,s,f) for _,g,s,f in formal],maximum=CFG['workers_per_gpu_max'])
        self.state['status']='analysis';self.save()
        analyze.finish(self.state,self.audit,formal=True)


def main():
    with (BASE/'launch.py').open('r+') as lock:
        fcntl.lockf(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB,1,0)
        scheduler=Scheduler()
        try: scheduler.main()
        except BaseException:
            scheduler.state['status']='interrupted_or_failed'
            scheduler.state['launcher_error']=traceback.format_exc()
            scheduler.stop_dispatch=True
            while scheduler.active:scheduler.tick()
            scheduler.save()
            ex.log('launcher_stopped',error=scheduler.state['launcher_error'])
            raise
        finally:
            if not scheduler.state.get('finished_utc'):
                scheduler.state['wall_seconds_accumulated']+=time.monotonic()-scheduler.start_time
                scheduler.save()


if __name__=='__main__': main()
