"""Fixed 40-task, parity-bound, resource-aware PPU scheduler."""
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
import run

CFG = run.CFG


def resources():
    mib = 1024 ** 2
    raw = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.free,memory.used,utilization.gpu',
         '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5, check=True
    ).stdout
    gpus = {}
    for line in raw.strip().splitlines():
        index, free, used, util = (int(x.strip()) for x in line.split(','))
        gpus[index] = {'free_mib': free, 'used_mib': used, 'utilization': util}
    cg = Path('/sys/fs/cgroup')
    if (cg / 'memory.max').exists():
        raw_limit = (cg / 'memory.max').read_text().strip()
        limit = int(raw_limit) if raw_limit != 'max' else 2 ** 62
        current = int((cg / 'memory.current').read_text())
        stats = dict(line.split() for line in (cg / 'memory.stat').read_text().splitlines())
        inactive = int(stats.get('inactive_file', 0))
        quota_s, period_s = (cg / 'cpu.max').read_text().split()
        quota = int(quota_s) if quota_s != 'max' else -1
        period = int(period_s)
    else:
        limit = int((cg / 'memory/memory.limit_in_bytes').read_text())
        current = int((cg / 'memory/memory.usage_in_bytes').read_text())
        stats = dict(line.split() for line in (cg / 'memory/memory.stat').read_text().splitlines())
        inactive = int(stats.get('total_inactive_file', stats.get('inactive_file', 0)))
        quota = int((cg / 'cpu/cpu.cfs_quota_us').read_text())
        period = int((cg / 'cpu/cpu.cfs_period_us').read_text())
    return {'utc': run.utc(), 'gpus': gpus, 'host_limit_mib': limit / mib,
            'host_current_mib': current / mib,
            'host_available_mib': max(0, limit - current + inactive) / mib,
            'cpu_quota': quota / period if quota > 0 else os.cpu_count(),
            'load1': os.getloadavg()[0]}


def worker_command(group, seed, fold, gpu, action='train'):
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu),
           'PYTHONDONTWRITEBYTECODE': '1', 'PCHRRA_LOG_STDOUT': '1',
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
    command = [sys.executable, '-B', '-u', str(run.BASE / 'run.py'), action,
               '--seed', str(seed), '--fold', str(fold)]
    if group is not None:
        command += ['--group', group]
    return command, env


def freeze_manifest(audit):
    tasks = [{'group': group, 'seed': 911, 'fold': fold, 'physical_gpu': fold % 2}
             for group in run.GROUPS for fold in range(10)]
    assert len(tasks) == 40
    assert {item['group'] for item in tasks} == set(run.GROUPS)
    assert not any(item['group'] in {'D0', 'D4', 'E0', 'E1', 'F0', 'F1', 'G1', 'G2', 'G3', 'G4'}
                   for item in tasks)
    assert sum(item['physical_gpu'] == 0 for item in tasks) == 20
    assert sum(item['physical_gpu'] == 1 for item in tasks) == 20
    manifest = {'experiment': run.CFG['experiment'],
                'config_sha256': run.sha_file(run.BASE / 'config.json'),
                'code_sha256': audit['code_sha256'], 'source_sha256': audit['source_sha256'],
                'input_context_sha256': audit['context_sha256'], 'tasks': tasks,
                'created_utc': run.utc()}
    path = run.BASE / 'manifest.json'
    if path.exists():
        old = json.loads(run.read_bytes(path))
        for key in ('experiment', 'config_sha256', 'code_sha256', 'source_sha256',
                    'input_context_sha256', 'tasks'):
            if old.get(key) != manifest[key]:
                raise RuntimeError(f'Frozen manifest mismatch in {key}')
        return old
    run.atomic_json(manifest, path)
    return manifest


def run_cpu_tests(audit):
    prior = audit.get('preflight', {}).get('math', {})
    if prior.get('passed') and prior.get('code_sha256') == audit['code_sha256']:
        return audit
    command = [sys.executable, '-B', str(run.BASE / 'run.py'), 'math-test']
    proc = subprocess.run(command, cwd=run.BASE, capture_output=True, text=True)
    if proc.returncode:
        run.log('math_test_failed', returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr)
        raise RuntimeError(f'CPU mathematical checks failed: {proc.stderr[-4000:]}')
    value = json.loads(proc.stdout)
    assert value['passed'] and value['checks'] == 20
    audit = json.loads(run.read_bytes(run.BASE / 'audit.json'))
    audit['preflight']['math'] = {'passed': True, 'code_sha256': audit['code_sha256'],
                                  'completed_utc': run.utc(), 'result': value}
    run.atomic_json(audit, run.BASE / 'audit.json')
    run.log('math_test_passed', checks=value['checks'], q_bounds=value['q_bounds'])
    return audit


def run_preflight(audit):
    current = audit.get('preflight', {}).get('real_forward', {})
    if current.get('result', {}).get('passed'):
        return audit
    assert audit['preflight']['math']['passed']
    gpu = 0
    command, env = worker_command(None, 911, 0, gpu, action='preflight')
    current.setdefault('attempt_count', 0)
    current.setdefault('attempts', [])
    attempt = {'attempt': current['attempt_count'] + 1, 'started_utc': run.utc(),
               'physical_gpu': gpu, 'command': command, 'status': 'running',
               'reason': ('initial real-forward preflight' if current['attempt_count'] == 0 else
                          'rerun after prior preflight failure and source correction')}
    current['attempt_count'] += 1
    current['attempts'].append(attempt)
    audit['preflight']['real_forward'] = current
    run.atomic_json(audit, run.BASE / 'audit.json')
    run.log('preflight_dispatch', attempt=attempt['attempt'], seed=911, fold=0, gpu=gpu, command=command)
    proc = subprocess.run(command, cwd=run.BASE, env=env, capture_output=True, text=True)
    events = []
    for line in proc.stdout.splitlines():
        try:
            row = json.loads(line)
            if isinstance(row, dict) and 'event' in row:
                events.append(row)
        except json.JSONDecodeError:
            pass
    for row in events:
        run.log('preflight_worker_event', worker_event=row)
    audit = json.loads(run.read_bytes(run.BASE / 'audit.json'))
    current = audit['preflight']['real_forward']
    attempt = current['attempts'][-1]
    attempt.update({'finished_utc': run.utc(), 'returncode': proc.returncode,
                    'model_forward_started_count': sum(row.get('event') == 'preflight_model_forward_started'
                                                       for row in events),
                    'model_forward_completed_count': sum(row.get('event') == 'preflight_model_forward_completed'
                                                         for row in events),
                    'stdout_tail': proc.stdout[-8000:], 'stderr_tail': proc.stderr[-4000:]})
    if proc.returncode:
        attempt['status'] = 'failed'
        audit['preflight']['real_forward'] = current
        run.atomic_json(audit, run.BASE / 'audit.json')
        raise RuntimeError(f'Real one-forward preflight failed; formal manifest was not frozen: {proc.stderr[-2000:]}')
    audit['preflight']['real_forward'] = current
    audit['preflight']['real_forward']['result'] = audit['preflight']['real_forward'].get('result') or {
        'passed': True, 'seed': 911, 'fold': 0}
    attempt['status'] = 'passed'
    audit['preflight']['real_forward']['result']['passed'] = True
    audit['formal_authorized'] = True
    audit['formal_authorized_utc'] = run.utc()
    audit['manifest'] = freeze_manifest(audit)
    run.atomic_json(audit, run.BASE / 'audit.json')
    run.log('formal_authorized', tasks=len(audit['manifest']['tasks']),
            manifest_sha256=run.sha_file(run.BASE / 'manifest.json'))
    return audit


def retryable_failure(failure):
    text = (failure.get('tail', '') + '\n' + failure.get('stderr_tail', '')).lower()
    allowed = ('resource temporarily unavailable', 'cuda error: out of memory',
               'cuda out of memory', 'ppu out of memory', 'temporarily unavailable')
    return any(token in text for token in allowed)


def queue(audit):
    manifest_tasks = audit['manifest']['tasks']
    tasks = [(item['group'], item['seed'], item['fold'], item['physical_gpu']) for item in manifest_tasks]
    assert len(tasks) == 40
    assert all(group in run.GROUPS and seed == 911 and fold in range(10) and gpu == fold % 2
               for group, seed, fold, gpu in tasks)
    assert not any(group in {'D0', 'D4', 'E0', 'E1', 'F0', 'F1', 'G1', 'G2', 'G3', 'G4'}
                   for group, _, _, _ in tasks)
    pending, skipped = [], []
    for group, seed, fold, gpu in tasks:
        value = run.valid_result(group, seed, fold, audit)
        if value is None:
            pending.append((group, seed, fold, gpu))
        else:
            assert value['physical_gpu'] == gpu
            skipped.append((group, seed, fold, gpu))
    active, attempts, failures = [], {}, []
    selector = selectors.DefaultSelector()
    caps = {0: run.CFG['workers_per_gpu'], 1: run.CFG['workers_per_gpu']}
    execution = audit.setdefault('execution', {'events': [], 'failures': [], 'resources': []})
    execution.setdefault('events', [])
    execution.setdefault('failures', [])
    execution.setdefault('resources', [])
    started = time.perf_counter()
    last_sample = last_status = 0.0
    run.log('queue_begin', formal_tasks=40, pending=len(pending), skipped=len(skipped), worker_caps=caps)
    with (run.BASE / 'run.log').open('a', buffering=1) as sink:
        while pending or active:
            for event, _ in selector.select(timeout=0.2):
                line = event.fileobj.readline()
                if line:
                    sink.write(line)
                    item = next((candidate for candidate in active
                                 if candidate['process'].stdout is event.fileobj), None)
                    if item is not None:
                        item['tail'] = (item['tail'] + line)[-16000:]
                else:
                    selector.unregister(event.fileobj)
                    event.fileobj.close()
            for item in active[:]:
                rc = item['process'].poll()
                if rc is None:
                    continue
                active.remove(item)
                group, seed, fold, gpu = item['task']
                value = run.valid_result(group, seed, fold, audit)
                if value is not None:
                    assert value['physical_gpu'] == gpu
                    execution['events'].append({'task': [group, seed, fold], 'gpu': gpu,
                                                'pid': item['process'].pid, 'returncode': rc,
                                                'attempt': item['attempt'],
                                                'started_utc': item['started_utc'], 'finished_utc': run.utc(),
                                                'wall_seconds': time.perf_counter() - item['started']})
                    run.log('task_accepted', group=group, seed=seed, fold=fold, gpu=gpu,
                            returncode=rc, seconds=time.perf_counter() - item['started'])
                elif rc != 0:
                    failure = {'task': [group, seed, fold], 'gpu': gpu, 'returncode': rc,
                               'utc': run.utc(), 'tail': item['tail'][-6000:]}
                    execution['failures'].append(failure)
                    key = (group, seed, fold)
                    if retryable_failure(failure) and attempts.get(key, 0) < CFG['max_attempts']:
                        pending.insert(0, item['task'])
                        caps[gpu] = 1
                        run.log('retry_transient_resource_failure', group=group, seed=seed, fold=fold,
                                gpu=gpu, attempt=attempts.get(key, 0), lowered_gpu_cap=1)
                    else:
                        failures.append(key)
                        run.log('formal_task_failed_no_retry', group=group, seed=seed, fold=fold,
                                gpu=gpu, retryable=retryable_failure(failure))
                else:
                    raise RuntimeError(f'Worker exited successfully without valid artifact: {group}/911/{fold}')
                run.atomic_json(audit, run.BASE / 'audit.json')
            if failures:
                pending.clear()
            now = time.perf_counter()
            if now - last_sample >= 20:
                try:
                    execution['resources'].append(resources())
                    run.atomic_json(audit, run.BASE / 'audit.json')
                except Exception as exc:
                    run.log('resource_sample_error', message=str(exc), launching_paused=True)
                last_sample = now
            current = None
            if pending and not failures:
                try:
                    current = resources()
                except Exception as exc:
                    run.log('resource_check_error', message=str(exc), launching_paused=True)
            if current is not None:
                for gpu in (0, 1):
                    own = [item for item in active if item['gpu'] == gpu]
                    candidates = [task for task in pending if task[3] == gpu]
                    if not candidates or len(own) >= caps[gpu] or gpu not in current['gpus']:
                        continue
                    mem = current['gpus'][gpu]
                    growth = max(0, len(own) * CFG['gpu_worker_budget_mib'] - mem['used_mib'])
                    need = CFG['gpu_reserve_mib'] + CFG['gpu_worker_budget_mib'] + growth
                    host_need = CFG['host_reserve_mib'] + CFG['host_worker_budget_mib']
                    if mem['free_mib'] < need or current['host_available_mib'] < host_need:
                        continue
                    if len(active) >= max(1, int(current['cpu_quota'])):
                        continue
                    task = candidates[0]
                    pending.remove(task)
                    group, seed, fold, _ = task
                    command, env = worker_command(group, seed, fold, gpu)
                    proc = subprocess.Popen(command, cwd=run.BASE, env=env, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                                            start_new_session=True)
                    selector.register(proc.stdout, selectors.EVENT_READ)
                    attempts[(group, seed, fold)] = attempts.get((group, seed, fold), 0) + 1
                    active.append({'task': task, 'gpu': gpu, 'process': proc,
                                   'attempt': attempts[(group, seed, fold)],
                                   'started': time.perf_counter(), 'started_utc': run.utc(), 'tail': ''})
                    run.log('dispatch', group=group, seed=seed, fold=fold, gpu=gpu,
                            attempt=attempts[(group, seed, fold)], pid=proc.pid, free_mib=mem['free_mib'],
                            command=command)
            if now - last_status > 60:
                run.log('queue_status', pending=len(pending), active=len(active),
                        accepted=len(execution['events']), skipped=len(skipped), failures=len(failures), caps=caps)
                last_status = now
            time.sleep(0.2)

    if failures:
        raise RuntimeError(f'Formal tasks failed without safe retry: {failures}')
    results = [run.valid_result(group, seed, fold, audit, required=True) for group, seed, fold, _ in tasks]
    unique = {(value['group'], value['seed'], value['fold']) for value in results}
    gpu_counts = {str(gpu): sum(value['physical_gpu'] == gpu for value in results) for gpu in (0, 1)}
    assert len(unique) == 40 and gpu_counts == {'0': 20, '1': 20}
    execution['queue'] = {'formal_tasks': 40, 'completed_unique_tasks': len(unique),
                          'accepted_this_queue': len(execution['events']), 'skipped_valid': len(skipped),
                          'wall_seconds': time.perf_counter() - started, 'completed_utc': run.utc(),
                          'gpu_completed': gpu_counts}
    run.atomic_json(audit, run.BASE / 'audit.json')
    run.log('queue_complete', **execution['queue'])


def main():
    parser = __import__('argparse').ArgumentParser()
    parser.add_argument('--stage', choices=('preflight', 'train', 'all'), default='all')
    args = parser.parse_args()
    with (run.BASE / 'launch.py').open('r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        audit = run.audit()
        audit = run_cpu_tests(audit)
        if args.stage in ('preflight', 'all'):
            audit = run_preflight(audit)
        if args.stage in ('train', 'all'):
            audit = run.guard()
            if not audit.get('formal_authorized') or not audit.get('manifest'):
                raise RuntimeError('Formal queue requires successful CPU and one-forward PPU preflight')
            queue(audit)
        if args.stage == 'all':
            subprocess.run([sys.executable, '-B', str(run.BASE / 'analyze.py'), '--final'],
                           cwd=run.BASE, check=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        try:
            run.log('coordinator_error', message=str(exc), traceback=traceback.format_exc())
        finally:
            raise
