"""Unified, single-owner PPU queue with cgroup-aware admission and resumability."""
import argparse
import errno
import fcntl
import json
import os
from pathlib import Path
import selectors
import subprocess
import sys
import time
import run as r

OWNED = []


def log(event, **values):
    line = json.dumps({'time_utc': r.utc(), 'event': event, **values}, ensure_ascii=False, allow_nan=False)
    with (r.BASE / 'run.log').open('a') as stream:
        stream.write(line + '\n')
        stream.flush()
    print(line, flush=True)


def storage_probe():
    path = r.BASE / '.storage_probe'
    moved = r.BASE / '.storage_probe_atomic'
    try:
        with path.open('wb') as stream:
            stream.write(b'PCND-storage-probe')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(path, moved)
        if moved.read_bytes() != b'PCND-storage-probe':
            raise OSError('Storage read-after-write failed')
    finally:
        path.unlink(missing_ok=True)
        moved.unlink(missing_ok=True)


class Resources:
    def __init__(self):
        self.previous_cpu = None
        self.penalty = {0: 0, 1: 0}
        self.budget = {0: r.config()['per_worker_budget_mib'], 1: r.config()['per_worker_budget_mib']}

    def sample(self, active):
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.total,memory.used,utilization.gpu',
                                          '--format=csv,noheader,nounits'], text=True, timeout=15)
        devices = {}
        for row in output.splitlines():
            gpu, total, used, util = [int(x.strip()) for x in row.split(',')]
            devices[gpu] = {'total_mib': total, 'used_mib': used, 'utilization_percent': util}
        if not all(g in devices for g in (0, 1)):
            raise RuntimeError('Two physical PPU devices required')
        mem = {row.split(':')[0]: int(row.split()[1]) // 1024
               for row in Path('/proc/meminfo').read_text().splitlines() if len(row.split()) >= 2}
        ram = mem['MemAvailable']
        cg = {}
        for folder, current, limit in (('/sys/fs/cgroup/memory', 'memory.usage_in_bytes', 'memory.limit_in_bytes'),
                                       ('/sys/fs/cgroup', 'memory.current', 'memory.max')):
            root = Path(folder)
            if (root / current).exists():
                raw_limit = (root / limit).read_text().strip()
                if raw_limit != 'max':
                    usage, maximum = int((root / current).read_text()), int(raw_limit)
                    if maximum < 2**60:
                        free = max(0, maximum - usage) // 1048576
                        ram = min(ram, free)
                        cg = {'usage_mib': usage // 1048576, 'limit_mib': maximum // 1048576,
                              'free_mib': free, 'admission_uses_cgroup_not_host_RAM': True}
                break
        now = time.monotonic()
        if Path('/sys/fs/cgroup/cpuacct/cpuacct.usage').exists():
            usage = int(Path('/sys/fs/cgroup/cpuacct/cpuacct.usage').read_text()) / 1e9
        else:
            rows = dict(line.split() for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())
            usage = int(rows['usage_usec']) / 1e6
        cores = 0.0
        if self.previous_cpu:
            cores = (usage - self.previous_cpu[1]) / max(now - self.previous_cpu[0], .01)
        self.previous_cpu = now, usage
        caps = {}
        for gpu in (0, 1):
            n = sum(w['gpu'] == gpu for w in active)
            reserved = max(devices[gpu]['used_mib'], n * self.budget[gpu])
            slots = max(0, (devices[gpu]['total_mib'] - r.config()['gpu_reserve_mib'] - reserved) // self.budget[gpu])
            caps[gpu] = max(n, min(r.config()['workers_per_gpu'] - self.penalty[gpu], n + slots))
            devices[gpu].update(active=n, limit=caps[gpu], budget_mib=self.budget[gpu])
        limit = 8
        if ram < r.config()['ram_reserve_mib'] or cores > r.config()['cpu_budget'] - .5:
            limit = len(active)
        return {'time_utc': r.utc(), 'devices': devices, 'host_RAM_available_mib': mem['MemAvailable'],
                'effective_RAM_available_mib': ram, 'cgroup_memory': cg,
                'cgroup_cpu_cores': cores, 'cpu_budget': r.config()['cpu_budget']}, caps, limit


def identity(task):
    return '/'.join(str(task[k]) for k in ('action', 'group', 'seed', 'fold'))


def finished(task, state):
    action, group, seed, fold = [task[k] for k in ('action', 'group', 'seed', 'fold')]
    key = f'{seed}/{fold}'
    if action == 'feature':
        return r.verify_cache(seed, fold, state)
    if action == 'check':
        return state.get('checks', {}).get(key, {}).get('passed', False)
    return r.verify_result(group, seed, fold, state, complete=action == 'frozen')


def task_plan(action):
    if action == 'check':
        return [{'action': action, 'group': 'P1', 'seed': seed, 'fold': fold}
                for seed, fold in r.config()['parity_folds']]
    groups = r.GROUPS if action == 'train' else tuple(r.config()['repeat_groups']) if action == 'frozen' else ('P1',)
    return [{'action': action, 'group': group, 'seed': seed, 'fold': fold}
            for seed in r.config()['seeds'] for fold in range(10) for group in groups]


def queue(action, state):
    pending = [t for t in task_plan(action) if not finished(t, state)]
    if not pending:
        log('phase_verified_skip', action=action)
        return
    selector, active, resources = selectors.DefaultSelector(), [], Resources()
    phases = state.setdefault('scheduling', {}).setdefault('phases', {})
    phase = phases.setdefault(action, {'started_utc': r.utc(), 'elapsed_seconds': 0.0})
    state['phase'] = action
    start, latest_resource, last_log = time.perf_counter(), 0.0, 0.0
    fatal = None
    r.atomic_json(state)
    while (pending and not fatal) or active:
        now = time.monotonic()
        if now - latest_resource >= 5 or not active:
            snapshot, caps, limit = resources.sample(active)
            state['scheduling']['latest_resources'] = snapshot
            samples = state['scheduling'].setdefault('resource_extrema', {})
            for key, value in (('effective_RAM_available_mib', snapshot['effective_RAM_available_mib']),
                               ('cgroup_cpu_cores', snapshot['cgroup_cpu_cores'])):
                v = samples.setdefault(key, {'min': value, 'max': value})
                v['min'], v['max'] = min(v['min'], value), max(v['max'], value)
            for gpu in (0, 1):
                d = snapshot['devices'][gpu]
                for key in ('used_mib', 'utilization_percent'):
                    v = samples.setdefault(f'GPU{gpu}_{key}', {'min': d[key], 'max': d[key]})
                    v['min'], v['max'] = min(v['min'], d[key]), max(v['max'], d[key])
            latest_resource = now
        if not fatal:
            for gpu in (0, 1):
                while sum(w['gpu'] == gpu for w in active) < caps[gpu] and len(active) < limit:
                    if action == 'feature' and len(active) >= r.config()['feature_workers']:
                        break
                    candidates = [(i, t) for i, t in enumerate(pending) if t['fold'] % 2 == gpu]
                    if not candidates:
                        break
                    index, task = candidates[0]
                    key = identity(task)
                    attempts = state.setdefault('attempts', {}).get(key, 0)
                    if attempts >= r.config()['max_attempts']:
                        fatal = f'Attempt budget exhausted: {key}'
                        break
                    pending.pop(index)
                    env = os.environ.copy()
                    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
                    env['PYTHONDONTWRITEBYTECODE'] = '1'
                    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
                        env[name] = '1'
                    if action == 'check':
                        env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
                    else:
                        env.pop('CUBLAS_WORKSPACE_CONFIG', None)
                    cmd = [sys.executable, '-u', '-B', str(r.BASE / 'run.py'), action,
                           '--group', task['group'], '--seed', str(task['seed']), '--fold', str(task['fold'])]
                    proc = subprocess.Popen(cmd, cwd=r.BASE, env=env, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True, bufsize=1)
                    OWNED.append(proc)
                    worker = {'proc': proc, 'gpu': gpu, 'task': task, 'key': key,
                              'result': None, 'tail': [], 'started': time.perf_counter()}
                    active.append(worker)
                    selector.register(proc.stdout, selectors.EVENT_READ, worker)
                    state['attempts'][key] = attempts + 1
                    state.setdefault('tasks', {})[key] = {'state': 'running', 'gpu': gpu, 'pid': proc.pid, 'started_utc': r.utc()}
                    peaks = state['scheduling'].setdefault('peak_workers', {'0': 0, '1': 0})
                    peaks[str(gpu)] = max(peaks[str(gpu)], sum(w['gpu'] == gpu for w in active))
                    log('started', **task, gpu=gpu, pid=proc.pid)
                    r.atomic_json(state)
        for event, _ in selector.select(timeout=1):
            worker = event.data
            line = event.fileobj.readline()
            if not line:
                selector.unregister(event.fileobj)
                event.fileobj.close()
                continue
            worker['tail'] = (worker['tail'] + [line.rstrip()])[-14:]
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                log('worker_output', task=worker['key'], text=line.rstrip()[:1500])
                continue
            if message.get('kind') == 'done':
                worker['result'] = message['result']
            elif message.get('kind') == 'progress':
                log('progress', task=worker['key'], epoch=message['epoch'], validation_loss=message['validation_loss'])
        for worker in list(active):
            proc = worker['proc']
            if proc.poll() is None:
                continue
            if not proc.stdout.closed:
                for line in proc.stdout.read().splitlines():
                    try:
                        message = json.loads(line)
                        if message.get('kind') == 'done':
                            worker['result'] = message['result']
                    except json.JSONDecodeError:
                        worker['tail'] = (worker['tail'] + [line])[-14:]
                try:
                    selector.unregister(proc.stdout)
                except (KeyError, ValueError):
                    pass
                proc.stdout.close()
            active.remove(worker)
            OWNED.remove(proc)
            task, key = worker['task'], worker['key']
            if proc.returncode != 0 or worker['result'] is None:
                error = {'task': key, 'gpu': worker['gpu'], 'exit_code': proc.returncode,
                         'tail': worker['tail'][-14:], 'time_utc': r.utc()}
                state.setdefault('errors', []).append(error)
                state['tasks'][key] = {'state': 'failed', **error}
                log('failed', **error)
                if any('Input/output error' in line or 'Errno 5' in line for line in worker['tail']):
                    fatal = 'Storage EIO; no new starts, draining healthy tasks, no /tmp fallback'
                elif state['attempts'][key] < r.config()['max_attempts']:
                    if any('out of memory' in line.lower() for line in worker['tail']):
                        resources.penalty[worker['gpu']] = min(3, resources.penalty[worker['gpu']] + 1)
                        resources.budget[worker['gpu']] = int(resources.budget[worker['gpu']] * 1.25)
                    pending.append(task)
                else:
                    fatal = f'Failed after maximum attempts: {key}'
            else:
                if not finished(task, state) and action != 'check':
                    fatal = f'Worker artifact validation failed: {key}'
                result = worker['result']
                state['tasks'][key] = {'state': 'complete', 'gpu': worker['gpu'], 'completed_utc': r.utc(),
                                      'elapsed_seconds': time.perf_counter() - worker['started']}
                if action == 'feature':
                    state.setdefault('feature_tasks', {})[f"{task['seed']}/{task['fold']}"] = result
                elif action == 'check':
                    state.setdefault('checks', {})[f"{task['seed']}/{task['fold']}"] = result
                log('complete', **task, gpu=worker['gpu'], result=result)
            r.atomic_json(state)
        if now - last_log > 30:
            log('queue_status', phase=action, pending=len(pending), active=len(active), resources=snapshot, draining=fatal)
            last_log = now
    selector.close()
    phase['elapsed_seconds'] += time.perf_counter() - start
    phase['completed_utc'] = r.utc()
    r.atomic_json(state)
    if fatal:
        raise RuntimeError(fatal)
    if not all(finished(t, state) for t in task_plan(action)):
        raise AssertionError('Phase did not finish all valid artifacts')


def preflight(state):
    if len(list((r.BASE / 'features').glob('*.pt'))) != 30:
        raise AssertionError('Exactly thirty feature caches required')
    if len(state['checks']) != 5 or not all(v['passed'] for v in state['checks'].values()):
        raise AssertionError('All five deterministic three-step parity checks required')
    boundary = [r.load_torch(r.feature_path(seed, fold))['boundary_checks']
                for seed in r.config()['seeds'] for fold in range(10)]
    if 'data_input_digests' not in state:
        ctx = r.load_torch(r.OLD / 'artifacts/context.pt')
        previous_argv = sys.argv
        try:
            sys.argv = ['audit_official_options']
            args = r.OptInit(model='MM_GTUNets', dataset='ABIDE', atlas='aal').args
        finally:
            sys.argv = previous_argv
        expected = {}
        for seed in r.config()['seeds']:
            x = r.load_torch(r.anchor_dir(seed, 0).parent / 'selected_features.pt')
            for fold in range(10):
                split = r.load_torch(r.anchor_dir(seed, fold) / 'splits.pt')
                affinity = r.create_reward_penalty_graph(ctx['ph_dict'], ctx['labels'],
                    *[split[k] for k in r.SUBSETS], args).astype('float32')
                expected[f'{seed}/{fold}'] = r.digest({'x': r.array_hash(x),
                    'ph': r.array_hash(ctx['ph_data'].astype('float32')), 'affinity': r.array_hash(affinity),
                    'ids': ctx['ids'], 'labels': ctx['labels'].tolist()})
        state['data_input_digests'] = expected
    conflicts = sum(v['P6_boundary_conflict_observed'] for v in boundary)
    state['preflight'].update(feature_caches_verified=30, deterministic_folds_verified=5,
        full_optimizer_steps_compared=15, deterministic_max_error=max(v['deterministic_max_error']
        for v in state['checks'].values()))
    authorization = state['preflight'].get('P6_boundary_policy')
    if conflicts or authorization != 'user_authorized_partition_within_site':
        raise AssertionError('All thirty revised P6 caches must satisfy the authorized partition-site boundary')
    if any(v['P6_heldout_covariate_transfers_to_train_TD'] for v in boundary):
        raise AssertionError('Heldout covariates reached the P6 training reference')
    state['preflight'].update(formal_authorized=True, passed_utc=r.utc(),
        feature_caches_verified=30, deterministic_folds_verified=5,
        full_optimizer_steps_compared=15, boundary_conflict_folds=conflicts,
        P6_exception_explicit=False, P6_revision_version=2,
        P1_P5_formal_features_bitwise_preserved=True,
        reason='Authorized revision validated: separate train/val/test within-site joint age-sex permutations')
    state.setdefault('formal_registered_utc', r.utc())
    state['formal_source_sha256'] = {n: r.sha_file(r.BASE / n) for n in
                                   ('run.py', 'normative.py', 'pcnd.py', 'config.json', 'launch.py', 'analyze.py')}
    r.atomic_json(state)
    return True


def register_p6_revision(state):
    if any((r.BASE / 'runs').glob('P*/*.pt')):
        raise RuntimeError('P6 preregistration may only change before formal training')
    if state.get('p6_revision'):
        raise RuntimeError('P6 revision already registered; resume without registering twice')
    algorithm = r.verify_authorized_algorithm()
    previous = {}
    for seed in r.config()['seeds']:
        for fold in range(10):
            path = r.feature_path(seed, fold)
            cache = r.load_torch(path)
            if cache['features_sha256'] != r.feature_digest(cache['features']):
                raise AssertionError('Legacy cache checksum failed before migration')
            previous[f'{seed}/{fold}'] = {'file_sha256': r.sha_file(path),
                'features_sha256': cache['features_sha256'], 'fingerprint': cache['fingerprint'],
                'P1_P5_formal_sha256': {g: {k: r.array_hash(cache['features'][g][k]) for k in ('features', 'gate')}
                                      for g in r.GROUPS[:-1]},
                'previous_P6_conflict': cache['boundary_checks']['P6_boundary_conflict_observed'],
                'previous_heldout_to_training_TD': cache['boundary_checks']['P6_heldout_covariate_transfers_to_train_TD']}
    state['p6_revision'] = {'version': 2, 'registered_utc': r.utc(),
        'authorization': 'User explicitly authorized scheme 2 before any formal task; no further confirmation needed',
        'previous_definition': 'cohort_within_site', 'final_definition': 'partition_within_site',
        'previous_training_code_sha256': state['training_code_sha256'], 'final_training_code_sha256': r.code_hash(),
        'formal_tasks_before_revision': 0, 'previous_caches': previous,
        'previous_preflight': state['preflight'], 'previous_checks': state['checks'],
        'previous_feature_tasks': state.get('feature_tasks', {}), 'previous_attempts': state['attempts'],
        'authorized_algorithm_audit': algorithm}
    state['config'], state['training_code_sha256'] = r.config(), r.code_hash()
    state['checks'], state['feature_tasks'], state['attempts'] = {}, {}, {}
    state['preflight'] = {'formal_authorized': False, 'P6_boundary_policy': 'user_authorized_partition_within_site',
        'P6_policy_resolution': '全队列同站点错配在正式训练前由用户授权修订为Train/Val/Test分区内部同站点联合错配；旧冲突完整保留在p6_revision。'}
    state['phase'] = 'revised_preflight'
    r.atomic_json(state)
    log('P6_preregistration_revision_registered', version=2, caches_to_migrate=30, formal_tasks=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('all', 'features', 'checks', 'train', 'frozen', 'analyze'), default='all')
    parser.add_argument('--refresh-preflight', action='store_true')
    parser.add_argument('--register-p6-revision', action='store_true')
    args = parser.parse_args()
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:r.config()['cpu_budget']])
    with (r.BASE / 'launch.py').open('r+') as owner:
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        storage_probe()
        if (r.BASE / 'summary.json').exists():
            if args.register_p6_revision:
                register_p6_revision(json.loads((r.BASE / 'summary.json').read_text()))
            if args.refresh_preflight:
                if any((r.BASE / 'runs').glob('P*/*.pt')):
                    raise RuntimeError('Never re-register source after any formal training')
                state = json.loads((r.BASE / 'summary.json').read_text())
                state.setdefault('preflight_source_revisions', []).append({'time_utc': r.utc(),
                    'previous': state['training_code_sha256'], 'new': r.code_hash(), 'checks_before': state.get('checks'),
                    'attempts_before': state.get('attempts')})
                state['training_code_sha256'] = r.code_hash()
                state['checks'] = {}
                state['attempts'] = {k: v for k, v in state['attempts'].items() if not k.startswith('check/')}
                state['preflight']['formal_authorized'] = False
                r.atomic_json(state)
            state = r.guard()
        else:
            log('audit_start')
            state = {'experiment': r.config()['experiment'], 'created_utc': r.utc(), 'config': r.config(),
                'training_code_sha256': r.code_hash(), 'audit': r.audit(), 'phase': 'audited',
                'feature_tasks': {}, 'checks': {}, 'tasks': {}, 'attempts': {}, 'errors': [], 'scheduling': {},
                'preflight': {'formal_authorized': False, 'P6_boundary_policy': 'awaiting_user_clarification'}}
            r.atomic_json(state)
        if state.get('final_verification', {}).get('passed'):
            import analyze
            log('verified_complete_resume', verification=analyze.verify_completed_bundle(state))
            return
        log('execution_start', stage=args.stage)
        if args.stage in ('all', 'features', 'checks', 'train'):
            queue('feature', state)
        if args.stage == 'features':
            return
        if args.stage in ('all', 'checks', 'train'):
            queue('check', state)
            if not preflight(state):
                return
        if args.stage == 'checks':
            return
        if args.stage in ('all', 'train'):
            queue('train', state)
            state['all_formal_training_complete'] = all(finished(t, state) for t in task_plan('train'))
            state.setdefault('all_formal_training_completed_utc', r.utc())
            signatures = {f'{group}/{seed}/{fold}': r.load_torch(r.result_path(group, seed, fold), mmap=True)['formal_signature']
                for group in r.GROUPS for seed in r.config()['seeds'] for fold in range(10)}
            if state.get('formal_signatures_after_training', signatures) != signatures:
                raise AssertionError('A previously completed formal result changed')
            state.setdefault('formal_signatures_after_training', signatures)
            r.atomic_json(state)
        if args.stage == 'train':
            return
        if args.stage in ('all', 'frozen'):
            queue('frozen', state)
        if args.stage == 'frozen':
            return
        import analyze
        analyze.finalize(state)
        log('all_research_complete', phase=state['phase'])


if __name__ == '__main__':
    try:
        main()
    finally:
        # Admission/logging errors must not orphan or interrupt healthy workers.
        for proc in OWNED:
            if not proc.stdout.closed:
                for line in proc.stdout:
                    print(line.rstrip(), flush=True)
                proc.stdout.close()
            proc.wait()

