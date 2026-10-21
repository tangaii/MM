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
    c = Path('/sys/fs/cgroup')
    if (c / 'memory.max').exists():
        raw_limit = (c / 'memory.max').read_text().strip()
        limit = int(raw_limit) if raw_limit != 'max' else 2 ** 62
        current = int((c / 'memory.current').read_text())
        stats = dict(x.split() for x in (c / 'memory.stat').read_text().splitlines())
        inactive = int(stats.get('inactive_file', 0))
        quota_s, period_s = (c / 'cpu.max').read_text().split()
        quota = int(quota_s) if quota_s != 'max' else -1
        period = int(period_s)
    else:
        limit = int((c / 'memory/memory.limit_in_bytes').read_text())
        current = int((c / 'memory/memory.usage_in_bytes').read_text())
        stats = dict(x.split() for x in (c / 'memory/memory.stat').read_text().splitlines())
        inactive = int(stats.get('total_inactive_file', stats.get('inactive_file', 0)))
        quota = int((c / 'cpu/cpu.cfs_quota_us').read_text())
        period = int((c / 'cpu/cpu.cfs_period_us').read_text())
    return {'utc': run.utc(), 'gpus': gpus, 'host_limit_mib': limit / mib,
            'host_current_mib': current / mib,
            'host_available_mib': max(0, limit - current + inactive) / mib,
            'cpu_quota': quota / period if quota > 0 else os.cpu_count(),
            'load1': os.getloadavg()[0]}


def worker_command(group, seed, fold, gpu, preflight=False):
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu),
           'PYTHONDONTWRITEBYTECODE': '1', 'PCFRA_LOG_STDOUT': '1',
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
    action = 'preflight' if preflight else 'train'
    command = [sys.executable, '-B', '-u', str(run.BASE / 'run.py'), action,
               '--seed', str(seed), '--fold', str(fold)]
    if not preflight:
        command += ['--group', group]
    return command, env


def freeze_manifest(audit):
    groups, seeds, folds = ('J1', 'J2', 'J3', 'J4'), (911,), tuple(range(10))
    tasks = [{'group': group, 'seed': seed, 'fold': fold, 'physical_gpu': fold % 2}
             for group in groups for seed in seeds for fold in folds]
    assert len(tasks) == 40
    assert len(set((x['group'], x['seed'], x['fold']) for x in tasks)) == 40
    assert {item['group'] for item in tasks} == set(groups)
    assert not any(item['group'] in {'D0', 'D4', 'G1', 'G2', 'G3', 'G4', 'H1', 'H2', 'H3', 'H4'} for item in tasks)
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


def run_preflight(audit):
    old = audit.get('preflight', {}).get('911/0')
    if not (old and old.get('passed')):
        gpu = 0
        while True:
            sample = resources()
            audit.setdefault('execution', {}).setdefault('resources', []).append(sample)
            run.atomic_json(audit, run.BASE / 'audit.json')
            mem = sample['gpus'].get(gpu)
            gpu_ready = mem is not None and mem['free_mib'] >= (
                run.CFG['gpu_reserve_mib'] + run.CFG['gpu_worker_budget_mib'])
            host_ready = sample['host_available_mib'] >= (
                run.CFG['host_reserve_mib'] + run.CFG['host_worker_budget_mib'])
            if gpu_ready and host_ready and mem['utilization'] <= 15:
                break
            run.log('preflight_wait_resources', gpu=gpu, gpu_resources=mem,
                    host_available_mib=sample['host_available_mib'])
            time.sleep(10)
        command, env = worker_command(None, 911, 0, gpu, preflight=True)
        run.log('preflight_dispatch', seed=911, fold=0, gpu=gpu, command=command)
        with (run.BASE / 'run.log').open('a') as sink:
            proc = subprocess.run(command, cwd=run.BASE, env=env, stdout=sink, stderr=subprocess.STDOUT)
        if proc.returncode:
            raise RuntimeError('Single-forward preflight failed; formal queue remains stopped')
    audit = run.guard()
    assert audit['preflight']['911/0']['passed']
    if not audit.get('formal_authorized'):
        audit['formal_authorized'] = True
        audit['formal_authorized_utc'] = run.utc()
        audit['manifest'] = freeze_manifest(audit)
        run.atomic_json(audit, run.BASE / 'audit.json')
        run.log('formal_authorized', tasks=len(audit['manifest']['tasks']),
                manifest_sha256=run.sha_file(run.BASE / 'manifest.json'))
    return audit


def queue(audit):
    manifest_tasks = audit['manifest']['tasks']
    tasks = [(item['group'], item['seed'], item['fold'], item['physical_gpu']) for item in manifest_tasks]
    assert len(tasks) == 40
    assert all(g in run.GROUPS and s == 911 and f in range(10) and gpu == f % 2
               for g, s, f, gpu in tasks)
    assert not any(g in {'D0', 'D4', 'G1', 'G2', 'G3', 'G4', 'H1', 'H2', 'H3', 'H4'}
                   for g, _, _, _ in tasks)
    pending, skipped = [], []
    for group, seed, fold, gpu in tasks:
        value = run.valid_result(group, seed, fold, audit)
        if value is None:
            pending.append((group, seed, fold, gpu))
        else:
            if value['physical_gpu'] != gpu:
                raise AssertionError(f'GPU mapping mismatch in {group}/911/{fold}')
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
                    item = next((x for x in active if x['process'].stdout is event.fileobj), None)
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
                    execution['events'].append({
                        'task': [group, seed, fold], 'gpu': gpu, 'pid': item['process'].pid,
                        'returncode': rc, 'started_utc': item['started_utc'],
                        'finished_utc': run.utc(), 'wall_seconds': time.perf_counter() - item['started'],
                    })
                    run.log('task_accepted', group=group, seed=seed, fold=fold, gpu=gpu,
                            returncode=rc, seconds=time.perf_counter() - item['started'])
                elif rc != 0:
                    failure = {'task': [group, seed, fold], 'gpu': gpu, 'returncode': rc,
                               'utc': run.utc(), 'tail': item['tail'][-6000:]}
                    execution['failures'].append(failure)
                    key = (group, seed, fold)
                    if attempts.get(key, 0) < run.CFG['max_attempts']:
                        pending.append((group, seed, fold, gpu))
                        run.log('retry_incomplete', group=group, seed=seed, fold=fold, gpu=gpu,
                                attempt=attempts.get(key, 0))
                    else:
                        failures.append(key)
                else:
                    raise RuntimeError(f'Worker exited successfully without valid artifact: {group}/911/{fold}')
                run.atomic_json(audit, run.BASE / 'audit.json')
            if failures and not active:
                raise RuntimeError(f'Formal tasks failed after one retry: {failures}')

            now = time.perf_counter()
            if now - last_sample >= 20:
                try:
                    execution['resources'].append(resources())
                    run.atomic_json(audit, run.BASE / 'audit.json')
                except Exception as exc:
                    run.log('resource_sample_error', message=str(exc), launching_paused=True)
                last_sample = now

            current = None
            if pending:
                try:
                    current = resources()
                except Exception as exc:
                    run.log('resource_check_error', message=str(exc), launching_paused=True)
            if current is not None:
                for gpu in (0, 1):
                    own = [x for x in active if x['gpu'] == gpu]
                    candidates = [x for x in pending if x[3] == gpu]
                    if not candidates or len(own) >= caps[gpu] or gpu not in current['gpus']:
                        continue
                    mem = current['gpus'][gpu]
                    growth = max(0, len(own) * run.CFG['gpu_worker_budget_mib'] - mem['used_mib'])
                    need = run.CFG['gpu_reserve_mib'] + run.CFG['gpu_worker_budget_mib'] + growth
                    host_need = run.CFG['host_reserve_mib'] + run.CFG['host_worker_budget_mib']
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
                    active.append({'task': task, 'gpu': gpu, 'process': proc,
                                   'started': time.perf_counter(), 'started_utc': run.utc(), 'tail': ''})
                    attempts[(group, seed, fold)] = attempts.get((group, seed, fold), 0) + 1
                    run.log('dispatch', group=group, seed=seed, fold=fold, gpu=gpu, pid=proc.pid,
                            free_mib=mem['free_mib'], command=command)
            if now - last_status > 60:
                run.log('queue_status', pending=len(pending), active=len(active),
                        accepted=len(execution['events']), skipped=len(skipped), worker_caps=caps)
                last_status = now
            time.sleep(0.2)

    values = [run.valid_result(g, s, f, audit, required=True) for g, s, f, _ in tasks]
    unique = {(v['group'], v['seed'], v['fold']) for v in values}
    gpu_counts = {str(gpu): sum(v['physical_gpu'] == gpu for v in values) for gpu in (0, 1)}
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
        if args.stage in ('preflight', 'all'):
            audit = run_preflight(audit)
        if args.stage in ('train', 'all'):
            audit = run.guard()
            if not audit.get('formal_authorized'):
                raise RuntimeError('Formal queue requires the one-pass preflight to succeed')
            print(json.dumps(audit['manifest'], ensure_ascii=False, indent=2), flush=True)
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
