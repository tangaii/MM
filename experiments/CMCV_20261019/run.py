"""CMCV audit, bounded CPU scheduling, atomic restart and final acceptance."""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import argparse
import concurrent.futures
import csv
import datetime as dt
import fcntl
import json
import pickle
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import psutil
from scipy.io import loadmat
from threadpoolctl import threadpool_limits

from data_protocol import (BASE, ROOT, CFG, apply_preprocessing, array_sha, atomic_bytes,
    available_ram, canonical_age, cgroup_resources, file_sha, fit_phenotype, fit_preprocessing, fit_rfe,
    load_artifact, make_nulls, save_artifact, save_json, source_fingerprints, stable_seed,
    task_definitions, task_lock, transform_phenotype, validate_split)
from probes import (group_matrix, metrics, neighbor_inputs, phenotype_kernel, run_all)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def log(message):
    with open(BASE / 'run.log', 'a') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(f'{now()} pid={os.getpid()} {message}\n')
        f.flush()
    print(message, flush=True)


def storage_checks():
    path = BASE / '.storage-test'
    value = b'CMCV atomic local storage verification\x00' * 31
    atomic_bytes(path, value)
    assert path.read_bytes() == value
    path.unlink()
    lock = BASE / '.storage-lock'
    with task_lock(lock):
        check = subprocess.run([sys.executable, '-B', '-c',
            'import os,fcntl,sys; f=open(sys.argv[1],"a"); '\
            '\ntry: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)'\
            '\nexcept BlockingIOError: sys.exit(0)'\
            '\nelse: sys.exit(9)', str(lock)], capture_output=True)
        assert check.returncode == 0, check.stderr.decode()
    lock.unlink()
    return {'fsync_reopen_replace_directory_fsync': True, 'cross_process_lock_exclusion': True,
            'path': str(BASE), 'filesystem': subprocess.check_output(['stat', '-f', '-c', '%T', str(BASE)], text=True).strip()}


def hash_record(path):
    return str(path), {'sha256': file_sha(path), 'bytes': Path(path).stat().st_size}


def initialize():
    if (BASE / 'audit.json').exists():
        audit = json.loads((BASE / 'audit.json').read_text())
        assert audit['code_sha256'] == source_fingerprints()[1], 'Code changed after freeze'
        inputs = load_artifact(BASE / 'inputs.bin')
        assert array_sha(inputs['x']) == audit['input']['raw_feature_sha256']
        return audit
    import scipy
    import sklearn
    import torch
    started = time.time()
    log('INPUT_AUDIT start; historical inputs are read-only')
    old = json.loads((ROOT / 'experiments/LCRA_Attribution_20261019/audit.json').read_text())
    protected = old['protected'].copy()
    # The prior audit already covers raw FC files, original source and experiment 1-8.
    for directory in (ROOT / 'experiments/LCRA_Attribution_20261019',):
        for path in directory.rglob('*'):
            if path.is_file() and '__pycache__' not in str(path) and not path.name.endswith('.lock'):
                protected[str(path)] = hash_record(path)[1]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        current = dict(pool.map(hash_record, protected))
    for p, expected in protected.items():
        assert current[p] == expected, f'Historical hash drift: {p}'
    data = ROOT / 'dataset/ABIDE_aal'
    context = torch.load(ROOT / 'experiments/DRPA_20261014/artifacts/context.pt', map_location='cpu', weights_only=False)
    ids = np.array(context['ids'], dtype=str)
    y = np.asarray(context['labels'], np.int64)
    with open(data / 'Phenotypic_V1_0b_preprocessed1.csv', newline='') as f:
        rows = {str(int(float(r['SUB_ID']))): r for r in csv.DictReader(f)}
    ordered = [rows[v] for v in ids]
    assert np.array_equal(y, [int(r['DX_GROUP']) - 1 for r in ordered])
    assert np.array_equal(ids, np.loadtxt(data / 'id.txt', dtype=str))
    assert y.shape == (871,) and np.bincount(y).tolist() == [403, 468]
    def number(value):
        try:
            return float(value)
        except (ValueError, TypeError):
            return float('nan')
    age = np.array([number(r['AGE_AT_SCAN']) for r in ordered])
    sex = np.array([number(r['SEX']) for r in ordered])
    site = np.array([r['SITE_ID'] for r in ordered])
    assert len(set(site)) == 20
    x = np.asarray(torch.load(data / 'save_tensor/up_triangles.pt', map_location='cpu', weights_only=False), np.float64)
    assert x.shape == (871, 6670)
    upper = np.triu_indices(116, 1)
    def verify_fc(pair):
        i, sid = pair
        raw = loadmat(data / f'{sid}_aal_correlation.mat')['connectivity']
        assert raw.shape == (116, 116)
        with np.errstate(divide='ignore', invalid='ignore'):
            fc = np.arctanh(raw)
        fc[fc == np.inf] = 0
        assert np.array_equal(x[i], fc[upper], equal_nan=True), f'FC cache mismatch: {sid}'
        return True
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        assert all(pool.map(verify_fc, enumerate(ids)))
    splits = {}
    for seed in CFG['seeds']:
        coverage = np.zeros(871, int)
        for fold in range(10):
            path = ROOT / f'experiments/DRPA_20261014/artifacts/seed_{seed}/fold_{fold}/splits.pt'
            split = validate_split(torch.load(path, map_location='cpu', weights_only=False))
            coverage[split['test']] += 1
            splits[f'{seed}/{fold}'] = split
            if str(path) not in protected:
                protected[str(path)] = hash_record(path)[1]
        assert np.all(coverage == 1)
    inputs = {'x': x, 'y': y, 'ids': ids, 'age': age, 'sex': sex, 'site': site, 'splits': splits}
    save_artifact(BASE / 'inputs.bin', inputs)
    fp, code_sha = source_fingerprints()
    handoff = (ROOT / 'handoff.md').read_bytes()
    input_info = {'raw_feature_sha256': array_sha(x), 'raw_cache_file_sha256': file_sha(data / 'save_tensor/up_triangles.pt'),
        'shape': x.shape, 'finite': bool(np.isfinite(x).all()), 'subject_sha256': array_sha(ids), 'label_sha256': array_sha(y),
        'ASD': int((y == 0).sum()), 'HC': int((y == 1).sum()), 'sites': {s: int((site == s).sum()) for s in sorted(set(site))},
        'all_871_mat_uppertriangles_exactly_equal': True, 'original_mat_transform': 'Fisher arctanh, positive infinity to zero, upper triangle k=1',
        'age_missing': int((~np.isfinite(canonical_age(age))).sum()), 'sex_unknown': int((~np.isin(sex, [1, 2])).sum()),
        'history_split_files': 30, 'history_rfe': 'First fold raw6670->500; subsequent folds overwrite x and keep the earlier supervised selection; not used here',
        'local_cache_sha256': file_sha(BASE / 'inputs.bin')}
    audit = {'experiment': CFG['experiment'], 'started_utc': now(), 'started_epoch': started,
        'code_files': fp, 'code_sha256': code_sha, 'storage': storage_checks(), 'protected': protected,
        'input': input_info, 'handoff_before_sha256': __import__('hashlib').sha256(handoff).hexdigest(),
        'handoff_before_bytes': len(handoff), 'software': {'python': sys.version, 'numpy': np.__version__,
            'sklearn': sklearn.__version__, 'scipy': scipy.__version__, 'torch': torch.__version__},
        'resources': resources(), 'references': {'paper': 'https://arxiv.org/html/2406.14455v3',
            'official': 'https://github.com/NZWANG/MM-GTUNets', 'current': 'https://github.com/tangaii/MM'},
        'failures': [{'phase': 'input_audit_before_training', 'cause': 'Resource inspector initially assumed cgroup v2; this host uses v1',
                      'repair': 'Detect v1/v2 explicitly; no RFE or model fit had started', 'scientific_definition_changed': False}]
                    if 'FileNotFoundError' in (BASE / 'run.log').read_text() else [],
        'preflight': {}, 'neural_model_instantiations': 0, 'audit_seconds': time.time() - started}
    save_json(BASE / 'audit.json', audit)
    log(f'INPUT_AUDIT passed raw={x.shape}, protected={len(protected)}, seconds={audit["audit_seconds"]:.2f}')
    return audit


def resources():
    cgroup = cgroup_resources()
    r = {'affinity_cpus': len(os.sched_getaffinity(0)), 'ram_available_reclaimable_bytes': available_ram(),
         'load_average': os.getloadavg(), 'cgroup': cgroup, 'memory_max': cgroup['memory_max'],
         'memory_current': cgroup['memory_current'], 'cpu_max': cgroup['cpu_max']}
    cmd = ['ppu-smi']
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    r['ppu_smi'] = result.stdout if result.returncode == 0 else result.stderr
    return r


def concentrated_tests(inputs):
    names = []
    # Synthetic missing/unseen conditions: held-out inputs never refit encoders.
    a = np.array([10., 20., -1., np.nan])
    s = np.array([1., 2., 1., 2.])
    sites = np.array(['A', 'A', 'B', 'B'])
    state = fit_phenotype(a, s, sites)
    ph, sh = transform_phenotype(np.array([-1., 30.]), np.array([7., np.nan]), np.array(['C', 'A']), state)
    assert np.isfinite(ph).all() and np.all(sh[0] == 0) and np.all(ph[:, -1] > 0)
    assert np.all(ph[0, 1] > 0)
    names += ['missing_age_train_imputation', 'unseen_sex_explicit_unknown', 'unseen_site_safe_allzero']
    # Label convention and fixed .5 decisions, including a deliberately reversed class column.
    m = metrics(np.array([0, 0, 1, 1]), np.array([.9, .8, .2, .1]))
    assert m['ACC'] == m['AUC'] == m['BA'] == 1
    assert metrics(np.array([1, 1]), np.array([.2, .1]))['AUC'] is None
    assert metrics(np.array([1, 1]), np.array([.2, .1]))['BA'] is None
    names += ['ASD_zero_probability', 'single_class_NA_not_zero']
    k = phenotype_kernel(np.full(3, 10.), np.array([1, 2, 0]), np.full((3, 1), 10.), np.ones((3, 1)), 2.)
    assert np.array_equal(k[:, 0], [1., .5, .75])
    names += ['historical_age_sex_kernel_closed_form']
    for task in task_definitions(inputs):
        maps, rec = make_nulls(inputs, task)
        assert np.array_equal(maps, make_nulls(inputs, task)[0])
        changed_labels = dict(inputs, y=1 - inputs['y'])
        assert np.array_equal(maps, make_nulls(changed_labels, task)[0])
        if task['protocol'] == 'S':
            assert not set(inputs['site'][task['split']['test']]) & set(inputs['site'][task['split']['train']])
    names += ['all_400_null_maps_bijective_partition_site_safe', 'null_label_independence', 'LOSO_no_site_overlap']
    # Checksum tampering and atomic restart storage.
    temp = BASE / '.artifact-test'
    save_artifact(temp, {'status': 'rfe_ready', 'selected': np.arange(500)})
    assert load_artifact(temp)['status'] == 'rfe_ready'
    atomic_bytes(temp, temp.read_bytes()[:-1] + bytes([temp.read_bytes()[-1] ^ 1]))
    try:
        load_artifact(temp)
    except AssertionError:
        pass
    else:
        raise AssertionError('Tampered artifact accepted')
    temp.unlink()
    names += ['atomic_RFE_stage_resume', 'content_SHA_rejects_corruption']
    return {'passed': True, 'checks': names, 'count': len(names)}


def matrix_benchmark(inputs):
    """Measure one relevant non-neural cosine operation, including transfers."""
    import torch
    torch.set_num_threads(1)
    x = inputs['x'][:, :500].copy()
    x = (x - x[:704].mean(0)) / np.maximum(x[:704].std(0), 1e-12)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    ref = x @ x[:704].T
    cpu_times = []
    for _ in range(3):
        t = time.monotonic()
        result = x @ x[:704].T
        cpu_times.append(time.monotonic() - t)
        assert np.array_equal(ref, result)
    result = {'cpu_median_seconds': float(np.median(cpu_times)), 'PPUs': [], 'formal_route': 'CPU sklearn and NumPy',
              'formal_PPU_training_tasks': 0, 'rationale': 'sklearn LR/SVC/RFE are CPU algorithms; do not transfer models or occupy devices pointlessly'}
    assert torch.cuda.is_available() and torch.cuda.device_count() == 2
    for i in range(2):
        times, errors = [], []
        for repeat in range(4):
            torch.cuda.synchronize(i)
            t = time.monotonic()
            tx = torch.as_tensor(x, dtype=torch.float64, device=f'cuda:{i}')
            out = (tx @ tx[:704].T).cpu().numpy()
            torch.cuda.synchronize(i)
            elapsed = time.monotonic() - t
            error = float(np.max(np.abs(ref - out)))
            assert error < 1e-10, f'CPU/PPU cosine mismatch {error}'
            if repeat:
                times.append(elapsed)
                errors.append(error)
            del tx, out
        result['PPUs'].append({'device': i, 'name': torch.cuda.get_device_name(i),
                              'end_to_end_median_seconds': float(np.median(times)), 'max_abs_error': max(errors)})
        torch.cuda.empty_cache()
    # A PPU route is unnecessary even when one matmul is marginally faster: the
    # measured workload is tiny relative to mandatory sklearn fits and CPU sorting.
    result['matrix_speedups'] = [result['cpu_median_seconds'] / p['end_to_end_median_seconds'] for p in result['PPUs']]
    return result


def validate_complete(result, inputs, task, code_sha):
    assert result['status'] == 'complete' and result['code_sha256'] == code_sha
    assert result['input_sha256'] == array_sha(inputs['x'])
    assert result['task']['id'] == task['id']
    for part, indices in task['split'].items():
        assert np.array_equal(indices, result['task']['split'][part])
    tr = task['split']['train']
    r = result['rfe']
    assert len(r['selected']) == len(set(r['selected'])) == 500
    assert r['selected_sha256'] == array_sha(r['selected'])
    assert r['train_labels_sha256'] == array_sha(inputs['y'][tr])
    assert r['train_ids_sha256'] == array_sha(inputs['ids'][tr])
    expected = 44 if task['protocol'] == 'R' else 41
    assert len(result['output']['probabilities']) == expected
    for key, fit in result['output']['fits'].items():
        assert fit['train_labels_sha256'] == array_sha(inputs['y'][tr])
        assert fit['train_ids_sha256'] == array_sha(inputs['ids'][tr])
        assert fit['ASD_probability_column'] == 0 and fit['classes'] == [0, 1]
    for part, idx in task['split'].items():
        if len(idx):
            for key, probability in result['output']['probabilities'].items():
                a, b = metrics(inputs['y'][idx], probability[idx]), result['output']['metrics'][part][key]
                assert all((a[m] is None and b[m] is None) or abs(a[m] - b[m]) < 1e-14 for m in a)
    assert np.isin(result['output']['neighbors']['candidates'], tr).all()
    assert np.array_equal(result['nulls'], make_nulls(inputs, task)[0])
    return True


def worker(task_id):
    threadpool_limits(1)
    start = time.time()
    inputs = load_artifact(BASE / 'inputs.bin')
    task = next(t for t in task_definitions(inputs) if t['id'] == task_id)
    code_sha = source_fingerprints()[1]
    audit = json.loads((BASE / 'audit.json').read_text())
    assert audit['code_sha256'] == code_sha
    path = BASE / 'runs' / (task_id + '.bin')
    with task_lock(BASE / 'runs' / (task_id + '.lock')):
        if path.exists():
            saved = load_artifact(path)
            assert saved['code_sha256'] == code_sha and saved['input_sha256'] == array_sha(inputs['x'])
            if saved['status'] == 'complete':
                validate_complete(saved, inputs, task, code_sha)
                log(f'SKIP_VALID {task_id}')
                return
            assert saved['status'] == 'rfe_ready'
            rfe = saved['rfe']
            log(f'RESUME_RFE {task_id}; no refitting')
        else:
            tr = task['split']['train']
            rfe = fit_rfe(inputs['x'][tr], inputs['y'][tr], inputs['ids'][tr])
            saved = {'status': 'rfe_ready', 'task': task, 'code_sha256': code_sha,
                     'input_sha256': array_sha(inputs['x']), 'rfe': rfe, 'started_epoch': start}
            save_artifact(path, saved)
            log(f'RFE_READY {task_id} seconds={rfe["fit_seconds"]:.3f}')
        prep = fit_preprocessing(inputs, task, rfe)
        zimg, ph, site = apply_preprocessing(inputs, rfe, prep)
        tr = task['split']['train']
        assert abs(np.sum(ph[tr] ** 2, axis=1).mean() - 1) < 1e-12
        assert abs(np.sum(zimg[tr] ** 2, axis=1).mean() - 1) < 1e-9
        # Fitted statistics cannot depend on held-out metadata/features/labels.
        # Replay the fit boundary using identical train rows after mutating all held-out rows.
        changed = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in inputs.items()}
        held = np.concatenate((task['split']['val'], task['split']['test']))
        changed['x'][held] = 1e6
        changed['age'][held] = -999
        changed['sex'][held] = 999
        changed['site'][held] = 'UNSEEN'
        changed['y'][held] = 1 - changed['y'][held]
        prep2 = fit_preprocessing(changed, task, rfe)
        assert pickle.dumps(prep, protocol=5) == pickle.dumps(prep2, protocol=5)
        assert array_sha(inputs['x'][tr]) == rfe['train_raw_features_sha256']
        del changed, prep2
        nulls, null_info = make_nulls(inputs, task)
        output = run_all(inputs, task, rfe, prep, zimg, ph, site, nulls)
        saved.update({'status': 'complete', 'preprocessing': prep, 'nulls': nulls, 'null_info': null_info,
                      'output': output, 'completed_epoch': time.time(), 'seconds': time.time() - start,
                      'resources': {'peak_RSS_bytes': __import__('resource').getrusage(__import__('resource').RUSAGE_SELF).ru_maxrss * 1024,
                                    'lane': task['lane'], 'actual_compute': 'CPU single-thread sklearn/NumPy'},
                      'checks': {'train_only_fit_boundary_heldout_mutation_invariant': True,
                                 'RFE_original_6670dim_train_only': True, 'all_nulls_partition_site_safe': True}})
        validate_complete(saved, inputs, task, code_sha)
        save_artifact(path, saved)
        log(f'COMPLETE {task_id} lane={task["lane"]} seconds={saved["seconds"]:.2f} LR={output["fit_counts"]["LogisticRegression"]} SVC=10')


def invoke(task):
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(task['lane'])
    env['PYTHONHASHSEED'] = str(stable_seed(task['id'], 'hash'))
    result = subprocess.run([sys.executable, '-B', str(BASE / 'run.py'), '--worker', task['id']], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        log(f'WORKER_FAILURE {task["id"]} exit={result.returncode}\n{result.stdout}')
        raise RuntimeError(f'{task["id"]} failed; see run.log')
    return task['id']


def schedule(tasks, per_lane):
    pending = list(tasks)
    active = {}
    running = [0, 0]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2 * per_lane) as pool:
        while pending or active:
            for task in list(pending):
                lane = task['lane']
                if running[lane] < per_lane and available_ram() > CFG['resources']['minimum_available_ram_GiB'] * 2 ** 30:
                    active[pool.submit(invoke, task)] = task
                    running[lane] += 1
                    pending.remove(task)
            if not active and pending:
                raise RuntimeError('Insufficient RAM to launch a safe task')
            done, _ = concurrent.futures.wait(active, timeout=10, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                task = active.pop(future)
                running[task['lane']] -= 1
                future.result()
            if done:
                log(f'QUEUE remaining={len(pending)} active={running} available_GiB={available_ram()/2**30:.1f}')


def execute():
    with task_lock(BASE / '.coordinator.lock'):
        audit = initialize()
        inputs = load_artifact(BASE / 'inputs.bin')
        if 'technical_tests' not in audit:
            audit['technical_tests'] = concentrated_tests(inputs)
            audit['matrix_benchmark'] = matrix_benchmark(inputs)
            save_json(BASE / 'audit.json', audit)
            log(f'TECHNICAL_TESTS passed; CPU/PPU benchmark={audit["matrix_benchmark"]}')
        tasks = task_definitions(inputs)
        preflight = [next(t for t in tasks if t['id'] == tid) for tid in CFG['preflight']]
        # Measure an actual complete first fold before increasing concurrency.
        invoke(preflight[0])
        schedule(preflight[1:], 1)
        for task in preflight:
            path = BASE / 'runs' / (task['id'] + '.bin')
            result = load_artifact(path)
            validate_complete(result, inputs, task, audit['code_sha256'])
            before_sha = file_sha(path)
            invoke(task)
            assert file_sha(path) == before_sha, 'Completed task overwritten during resume'
            audit['preflight'][task['id']] = {'passed': True, 'seconds': result['seconds'],
                'RFE_seconds': result['rfe']['fit_seconds'], 'peak_RSS_bytes': result['resources']['peak_RSS_bytes'],
                'resume_no_retrain_no_overwrite': True}
        max_rss = max(v['peak_RSS_bytes'] for v in audit['preflight'].values())
        quota = audit['resources']['cpu_max'].split()
        cpus = float(quota[0]) / float(quota[1]) if quota[0] != 'max' else len(os.sched_getaffinity(0))
        per_lane = 2 if available_ram() > max(8 * max_rss, 8 * 2 ** 30) and cpus >= 4 else 1
        audit['scheduler'] = {'initial_workers_per_lane': 1, 'formal_max_workers_per_lane': per_lane,
            'peak_task_RSS_bytes_from_preflight': max_rss, 'logic': 'Fixed 25 tasks per lane, live RAM admission, one BLAS thread per worker',
            'queue_started_epoch': time.time(), 'resources_before_queue': resources()}
        save_json(BASE / 'audit.json', audit)
        log(f'PREFLIGHT 4/4 accepted; formal per_lane={per_lane}; no result-based method selection')
        remaining = [t for t in tasks if t['id'] not in CFG['preflight']]
        schedule(remaining, per_lane)
        audit = json.loads((BASE / 'audit.json').read_text())
        audit['scheduler']['queue_completed_epoch'] = time.time()
        audit['scheduler']['resources_after_queue'] = resources()
        save_json(BASE / 'audit.json', audit)
        import analyze
        analyze.complete()
        # All workers have exited, so stale lock files can be removed safely.
        for lock in (BASE / 'runs').glob('*.lock'):
            lock.unlink()
        log('EXPERIMENT_COMPLETE; 50 tasks and independent audits accepted')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    try:
        if args.worker:
            worker(args.worker)
        elif args.audit_only:
            initialize()
        elif args.execute:
            execute()
        else:
            parser.print_help()
    except Exception:
        log('FATAL\n' + traceback.format_exc())
        raise
