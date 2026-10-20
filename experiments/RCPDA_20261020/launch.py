"""Fixed RCPDA 60-task PPU scheduler."""
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
    mi = 1024 ** 2
    raw = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,memory.free,memory.used,utilization.gpu',
         '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5, check=True
    ).stdout
    gpus = {}
    for line in raw.strip().splitlines():
        idx, free, used, util = (int(x.strip()) for x in line.split(','))
        gpus[idx] = {'free_mib': free, 'used_mib': used, 'utilization': util}
    c = Path('/sys/fs/cgroup')
    if (c / 'memory.max').exists():
        limit = int((c / 'memory.max').read_text())
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
    return {'utc': run.utc(), 'gpus': gpus, 'host_limit_mib': limit / mi,
            'host_current_mib': current / mi,
            'host_available_mib': max(0, limit - current + inactive) / mi,
            'cpu_quota': quota / period if quota > 0 else os.cpu_count(),
            'load1': os.getloadavg()[0]}


def worker_command(group, seed, fold, gpu, preflight=False):
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu),
           'PYTHONDONTWRITEBYTECODE': '1', 'RCPDA_LOG_STDOUT': '1',
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1'}
    if preflight:
        env['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    else:
        env.pop('CUBLAS_WORKSPACE_CONFIG', None)
    action = 'preflight' if preflight else 'train'
    command = [sys.executable, '-B', '-u', str(run.BASE / 'run.py'), action,
               '--seed', str(seed), '--fold', str(fold)]
    if not preflight:
        command += ['--group', group]
    return command, env


def freeze_manifest(a):
    path = run.BASE / 'manifest.json'
    tasks = [{'group': g, 'seed': s, 'fold': f, 'physical_gpu': f % 2}
             for f in range(run.CFG['folds']) for g in run.CFG['groups']
             for s in run.CFG['seeds']]
    expected = {'experiment': run.CFG['experiment'], 'config_sha256': run.sha_file(run.BASE / 'config.json'),
                'code_sha256': a['code_sha256'], 'source_sha256': a['source_sha256'],
                'input_context_sha256': a['context_sha256'], 'tasks': tasks,
                'created_utc': run.utc()}
    if path.exists():
        old = run.read_json(path)
        for key in ('experiment', 'config_sha256', 'code_sha256', 'source_sha256', 'input_context_sha256', 'tasks'):
            if old.get(key) != expected[key]:
                raise RuntimeError(f'Frozen manifest mismatch in {key}')
        return old
    run.atomic_json(expected, path)
    return expected


def run_preflight(a):
    for seed, fold in run.CFG['preflight_anchors']:
        current = run.read_json(run.BASE / 'audit.json')
        if current['preflight'].get(f'{seed}/{fold}', {}).get('passed'):
            continue
        gpu = fold % 2
        command, env = worker_command(None, seed, fold, gpu, preflight=True)
        run.log('preflight_dispatch', seed=seed, fold=fold, gpu=gpu, command=command)
        with (run.BASE / 'run.log').open('a') as out:
            proc = subprocess.run(command, cwd=run.BASE, env=env, stdout=out, stderr=subprocess.STDOUT)
        if proc.returncode:
            raise RuntimeError(f'Preflight failed for seed={seed}, fold={fold}; formal training blocked')
    a = run.guard()
    assert all(a['preflight'].get(f'{s}/{f}', {}).get('passed') for s, f in run.CFG['preflight_anchors'])
    a['formal_authorized'] = True
    a['formal_authorized_utc'] = run.utc()
    a['manifest'] = freeze_manifest(a)
    run.atomic_json(a, run.BASE / 'audit.json')
    run.log('formal_authorized', tasks=len(a['manifest']['tasks']), manifest_sha256=run.sha_file(run.BASE / 'manifest.json'))
    return a


def queue(a):
    tasks = [(item['group'], item['seed'], item['fold'], item['physical_gpu'])
             for item in a['manifest']['tasks']]
    pending = []
    skipped = []
    for g, s, f, gpu in tasks:
        v = run.valid_result(g, s, f, a)
        if v is None:
            pending.append((g, s, f, gpu))
        else:
            if v['physical_gpu'] != gpu:
                raise AssertionError(f'GPU mapping mismatch in existing result {g}/{s}/{f}')
            skipped.append((g, s, f, gpu))
    assert len(tasks) == 60
    active = []
    attempts = {}
    failures = []
    selector = selectors.DefaultSelector()
    caps = {0: run.CFG['workers_per_gpu'], 1: run.CFG['workers_per_gpu']}
    execution = a.setdefault('execution', {})
    execution.setdefault('events', [])
    execution.setdefault('failures', [])
    execution.setdefault('resources', [])
    started = time.perf_counter()
    last_sample = 0.0
    last_status = 0.0
    run.log('queue_begin', total_tasks=60, pending=len(pending), skipped=len(skipped), worker_caps=caps)
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
                g, s, f, gpu = item['task']
                if rc == 0:
                    value = run.valid_result(g, s, f, a, required=True)
                    assert value['physical_gpu'] == gpu
                    execution['events'].append({'task': [g, s, f], 'gpu': gpu, 'pid': item['process'].pid,
                                                 'started_utc': item['started_utc'], 'finished_utc': run.utc(),
                                                 'wall_seconds': time.perf_counter() - item['started']})
                    run.log('task_accepted', group=g, seed=s, fold=f, gpu=gpu,
                            seconds=time.perf_counter() - item['started'])
                else:
                    rec = {'task': [g, s, f], 'gpu': gpu, 'returncode': rc, 'utc': run.utc(),
                           'tail': item['tail'][-6000:]}
                    rec['oom'] = 'out of memory' in rec['tail'].lower()
                    execution['failures'].append(rec)
                    key = (g, s, f)
                    if attempts.get(key, 0) < run.CFG['max_attempts']:
                        pending.append((g, s, f, gpu))
                        run.log('retry_incomplete', group=g, seed=s, fold=f, gpu=gpu,
                                attempt=attempts.get(key, 0), oom=rec['oom'])
                    else:
                        failures.append(key)
                a['execution'] = execution
                run.atomic_json(a, run.BASE / 'audit.json')
            if failures and not active:
                raise RuntimeError(f'Formal tasks failed after one retry: {failures}')
            now = time.perf_counter()
            if now - last_sample >= 20:
                execution['resources'].append(resources())
                last_sample = now
                a['execution'] = execution
                run.atomic_json(a, run.BASE / 'audit.json')
            current = resources() if pending else None
            if current is not None:
                for gpu in (0, 1):
                    own = [x for x in active if x['gpu'] == gpu]
                    candidates = [x for x in pending if x[3] == gpu]
                    if not candidates or len(own) >= caps[gpu]:
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
                    g, s, f, _ = task
                    command, env = worker_command(g, s, f, gpu)
                    proc = subprocess.Popen(command, cwd=run.BASE, env=env, stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
                    selector.register(proc.stdout, selectors.EVENT_READ)
                    active.append({'task': task, 'gpu': gpu, 'process': proc, 'started': time.perf_counter(),
                                   'started_utc': run.utc(), 'tail': ''})
                    attempts[(g, s, f)] = attempts.get((g, s, f), 0) + 1
                    run.log('dispatch', group=g, seed=s, fold=f, gpu=gpu, pid=proc.pid,
                            free_mib=mem['free_mib'], command=command)
            if now - last_status > 60:
                run.log('queue_status', pending=len(pending), active=len(active),
                        accepted=len(execution['events']), skipped=len(skipped), worker_caps=caps)
                last_status = now
            time.sleep(0.2)
    execution['queue'] = {'formal_tasks': 60, 'accepted_this_queue': len(execution['events']),
                          'skipped_valid': len(skipped), 'wall_seconds': time.perf_counter() - started,
                          'completed_utc': run.utc(), 'gpu_counts': {
                              str(gpu): sum(e['gpu'] == gpu for e in execution['events']) for gpu in (0, 1)}}
    a['execution'] = execution
    run.atomic_json(a, run.BASE / 'audit.json')
    run.log('queue_complete', **execution['queue'])


def main():
    parser = __import__('argparse').ArgumentParser()
    parser.add_argument('--stage', choices=['preflight', 'train', 'all'], default='all')
    args = parser.parse_args()
    with (run.BASE / 'launch.py').open('r+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        a = run.audit()
        if args.stage in ('preflight', 'all'):
            a = run_preflight(a)
        if args.stage in ('train', 'all'):
            a = run.guard()
            if not a.get('formal_authorized'):
                raise RuntimeError('Formal run requires successful preflight and frozen manifest')
            queue(a)
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
