"""Single-writer, completed-task-only supervisor for ossfs checkpoint visibility."""
import argparse
import os
import selectors
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.dont_write_bytecode = True
import run


def worker(group, fold):
    # Keep the frozen scientific runner unchanged; serialize only its logging.
    run.log = lambda message: print(f'{run.utc()} {message}', flush=True)
    try:
        run.train(group, fold)
    except Exception:
        traceback.print_exc()
        raise


def supervise():
    base = run.BASE
    with run.locked(base/'runs/.coordinator.lock'):
        audit = run.audit_inputs()
        assert all(audit['preflight'][str(g)]['passed'] for g in (0, 1))
        execution = audit['execution']
        launcher_sha = run.sha_file(__file__)
        if not execution.get('single_writer_recovery'):
            execution['failures'].append({
                'kind': 'coordinator_only', 'time': run.utc(),
                'error': 'Transient ENOENT while reading S1_f1.bin during an active atomic replacement',
                'workers_interrupted': 0, 'scientific_runner_changed': False})
            execution['single_writer_recovery'] = {
                'launcher_sha256': launcher_sha, 'frozen_scientific_code_sha256': run.code_hash(),
                'reason': 'Never read an active checkpoint; funnel worker stdout to one log writer.',
                'original_four_workers_finished_without_interruption': True}
            path = base/'run.log'
            raw = path.read_bytes()
            execution['single_writer_recovery']['prior_log_sha256'] = run.sha_file(path)
            execution['single_writer_recovery']['prior_log_nul_bytes'] = raw.count(b'\0')
            if b'\0' in raw:
                run.atomic_bytes(path, raw.replace(b'\0', b''))
        assert execution['single_writer_recovery']['launcher_sha256'] == launcher_sha
        run.save_json(base/'audit.json', audit)
        finished = {(g, f) for f in range(10) for g in run.CFG['groups'] if run.completed(g, f)}
        pending = [(g, f) for f in range(10) for g in run.CFG['groups'] if (g, f) not in finished]
        active = []
        retry = {}
        failed = []
        cap = {0: 2, 1: 2}
        selector = selectors.DefaultSelector()
        with (base/'run.log').open('a', buffering=1) as sink:
            def emit(message):
                line = f'{run.utc()} {message}'
                print(line, flush=True)
                sink.write(line+'\n')
                sink.flush()

            run.log = emit
            emit(f'RECOVER completed={len(finished)}/20; frozen scientific runner unchanged')
            while pending or active:
                for event, _ in selector.select(timeout=0.1):
                    line = event.fileobj.readline()
                    if line:
                        sink.write(line)
                    else:
                        selector.unregister(event.fileobj)
                        event.fileobj.close()
                for item in active[:]:
                    rc = item['process'].poll()
                    if rc is None:
                        continue
                    active.remove(item)
                    key = (item['group'], item['fold'])
                    if rc == 0:
                        assert run.completed(*key)
                        finished.add(key)
                    else:
                        record = {k: v for k, v in item.items() if k != 'process'}
                        record['returncode'] = rc
                        execution['failures'].append(record)
                        cap[item['gpu']] = 1
                        if retry.get(key, 0) < 1:
                            retry[key] = 1
                            pending.append(key)
                        else:
                            failed.append(key)
                    run.save_json(base/'audit.json', audit)
                for gpu in (0, 1):
                    own = [a for a in active if a['gpu'] == gpu]
                    if cap[gpu] > 1 and sum(f % 2 == gpu for _, f in finished) >= 2:
                        cap[gpu] = run.CFG['max_workers_per_gpu']
                    candidates = [t for t in pending if t[1] % 2 == gpu]
                    if not candidates or len(own) >= cap[gpu] or not run.resource_capacity(gpu, own):
                        continue
                    group, fold = candidates[0]
                    pending.remove((group, fold))
                    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONDONTWRITEBYTECODE': '1'}
                    cmd = [sys.executable, '-B', '-u', str(Path(__file__).resolve()),
                           'worker', '--group', group, '--fold', str(fold)]
                    process = subprocess.Popen(cmd, cwd=base, env=env, stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
                    selector.register(process.stdout, selectors.EVENT_READ)
                    item = {'process': process, 'group': group, 'fold': fold, 'gpu': gpu,
                            'pid': process.pid, 'started': run.utc(), 'allocated': 0}
                    active.append(item)
                    execution['attempts'].append({k: v for k, v in item.items() if k != 'process'})
                    run.save_json(base/'audit.json', audit)
                    emit(f'DISPATCH {group}/f{fold} PPU{gpu} pid={process.pid} active={len(own)+1}/{cap[gpu]}')
            selector.close()
            if failed:
                raise RuntimeError(f'Persistent failed folds: {failed}')
            execution['completed_utc'] = run.utc()
            execution['completed_epoch'] = time.time()
            execution['wall_seconds'] = time.time()-execution['started_epoch']
            run.save_json(base/'audit.json', audit)
            from analyze import analyze
            analyze()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['all', 'worker'], default='all', nargs='?')
    parser.add_argument('--group', choices=['C0', 'S1'])
    parser.add_argument('--fold', type=int)
    args = parser.parse_args()
    if args.command == 'worker':
        worker(args.group, args.fold)
    else:
        supervise()
