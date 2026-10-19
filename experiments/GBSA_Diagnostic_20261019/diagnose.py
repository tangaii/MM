"""Frozen GBSA mechanisms, train-only lightweight probes and ID retrieval."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.request
import warnings

sys.dont_write_bytecode = True
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
BASE = Path(__file__).resolve().parent
CFG = json.loads((BASE / 'config.json').read_text())
ROOT = Path(CFG['root'])
PILOT = ROOT / CFG['pilot']
OLD = ROOT / CFG['history']
os.environ['HF_HOME'] = str(BASE / 'runs' / '.hf')

import numpy as np
import torch
from torch.nn import functional as F
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, log_loss, r2_score, mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

torch.set_num_threads(1)
CODE = ('config.json', 'diagnose.py', 'analyze.py')
STAT_NAMES = ('mean_fc', 'mean_absolute_fc', 'global_efficiency', 'modularity', 'mean_clustering', 'mean_strength', 'sd_strength')


def utc():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    print(f'{utc()} {message}', flush=True)


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_sha(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def tree_hash(value):
    if torch.is_tensor(value) or isinstance(value, np.ndarray):
        return array_sha(value)
    if isinstance(value, dict):
        value = {str(k): tree_hash(v) for k, v in sorted(value.items(), key=lambda x: str(x[0]))}
    elif isinstance(value, (tuple, list)):
        value = [tree_hash(v) for v in value]
    return hashlib.sha256(repr(value).encode()).hexdigest()


def code_hash():
    return hashlib.sha256(json.dumps({n: sha_file(BASE / n) for n in CODE}, sort_keys=True).encode()).hexdigest()


def atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.atomic-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        assert path.read_bytes() == payload, f'Atomic readback differs: {path}'
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def save_json(path, value):
    atomic_bytes(path, (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode())


def save(path, value):
    stream = io.BytesIO()
    torch.save(value, stream)
    data = stream.getvalue()
    atomic_bytes(path, b'GBSAD1 ' + hashlib.sha256(data).hexdigest().encode() + b'\n' + data)


def load(path, historical=False):
    header, data = Path(path).read_bytes().split(b'\n', 1)
    prefix = b'GBSA1 ' if historical else b'GBSAD1 '
    assert header == prefix + hashlib.sha256(data).hexdigest().encode(), f'Content SHA mismatch: {path}'
    return torch.load(io.BytesIO(data), weights_only=False, map_location='cpu')


def history_load(path):
    return torch.load(path, weights_only=False, map_location='cpu')


@contextmanager
def locked(path):
    with Path(path).open('a+b') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def historical_module(name):
    spec = importlib.util.spec_from_file_location('frozen_pilot_' + name, PILOT / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DESC = historical_module('descriptors')
SEM = historical_module('semantic_model')


def deny_training(*args, **kwargs):
    raise RuntimeError('Neural training/backward is forbidden in this frozen diagnosis')


torch.optim.Optimizer.__init__ = deny_training
torch.Tensor.backward = deny_training
torch.autograd.backward = deny_training


def private_seed(*parts):
    return int.from_bytes(hashlib.sha256('/'.join(map(str, parts)).encode()).digest()[:8], 'little')


def storage_check():
    path = BASE / 'runs' / '.storage-check'
    payload = b'GBSA-Diagnostic storage probe\n' + bytes(range(256))
    atomic_bytes(path, payload)
    with path.open('r+b') as a, path.open('rb') as b:
        fcntl.flock(a, fcntl.LOCK_EX | fcntl.LOCK_NB)
        blocked = False
        try:
            fcntl.flock(b, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            blocked = True
        assert blocked, 'Storage does not enforce flock exclusion'
        fcntl.flock(a, fcntl.LOCK_UN)
    path.unlink()
    return {'write_fsync_rename_readback_directory_fsync': True, 'exclusive_flock': True}


def protected_guard(audit):
    assert code_hash() == audit['code_sha256'], 'Frozen diagnosis code/config changed'
    for name, expected in audit['protected_files'].items():
        assert sha_file(name) == expected, 'Historical input changed: ' + name


def read_inputs():
    context = history_load(OLD / 'artifacts/context.pt')
    cache = load(PILOT / 'runs/descriptors.bin', historical=True)
    splits = {f: history_load(OLD / f'artifacts/seed_911/fold_{f}/splits.pt') for f in range(10)}
    rows = {(g, f): load(PILOT / f'runs/{g}_f{f}.bin', historical=True) for f in range(10) for g in ('C0', 'S1')}
    return context, cache, splits, rows


def audit_inputs(context, cache, splits, rows):
    path = BASE / 'audit.json'
    if path.exists():
        audit = json.loads(path.read_text())
        protected_guard(audit)
        return audit
    start = time.time()
    historical_audit = json.loads((PILOT / 'audit.json').read_text())
    historical_summary = json.loads((PILOT / 'summary.json').read_text())
    protected = {}
    changes = []
    for name, expected in historical_audit['protected_files'].items():
        actual = sha_file(name)
        if actual != expected:
            assert Path(name) == ROOT / 'prompt', 'Unexpected change relative to Experiment 11: ' + name
            changes.append({'path': name, 'old_sha256': expected, 'current_sha256': actual,
                            'reason': 'User replaced the task prompt for this frozen diagnosis; not a scientific input/source change'})
        protected[name] = actual
    for p in PILOT.rglob('*'):
        if p.is_file():
            protected[str(p)] = sha_file(p)
    for p, expected in historical_summary['artifact_SHA256'].items():
        assert protected[p] == expected
    assert sha_file(PILOT / 'runs/descriptors.bin') == historical_audit['descriptors']['cache_sha256']
    for name, expected in cache['text_model']['files_sha256'].items():
        assert sha_file(PILOT / 'runs/text_model' / name) == expected
    for p in (ROOT / 'handoff.md', ROOT / 'experiments/LCRA_Attribution_20261019/report.md', Path('/tmp/CMCV_20261019/report.md')):
        if p.name != 'handoff.md':
            protected[str(p)] = sha_file(p)
        assert len(p.read_text()) > 0
    assert cache['complete'] and cache['ids'] == context['ids']
    assert len(set(context['ids'])) == 871
    assert np.bincount(context['labels']).tolist() == [403, 468]
    assert cache['raw_stats'].shape == (871, 7) and np.isfinite(cache['raw_stats']).all()
    coverage = np.zeros(871, int)
    metadata = {}
    for f in range(10):
        split = splits[f]
        combined = np.concatenate([split[k] for k in ('train', 'val', 'test')])
        np.testing.assert_array_equal(np.sort(combined), np.arange(871))
        coverage[split['test']] += 1
        d = cache['folds'][f]
        assert d['subject_ids'] == context['ids']
        assert d['embedding'].shape == (871, 384) and d['embedding'].dtype == torch.float32
        assert array_sha(d['embedding']) == d['embedding_sha256']
        assert tree_hash(split) == d['split_sha256']
        assert tree_hash(DESC.reference(cache['raw_stats'], split['train'])) == tree_hash(d['reference'])
        assert DESC.captions(cache['raw_stats'], cache['pairs'], d['reference']) == d['captions']
        assert rows['C0', f]['pairing'] == rows['S1', f]['pairing']
        for g in ('C0', 'S1'):
            r = rows[g, f]
            assert r['status'] == 'complete' and r['seed'] == 911 and r['fold'] == f and r['group'] == g
            assert r['cache_sha256'] == sha_file(PILOT / 'runs/descriptors.bin')
            assert np.array_equal(r['indices'], split['test'])
            assert r['ids'] == [context['ids'][i] for i in split['test']]
            np.testing.assert_array_equal(r['labels'], context['labels'][split['test']])
            assert r['test_image_embedding'].shape == (871, 16) and torch.isfinite(r['test_image_embedding']).all()
            assert r['test_image_embedding'].dtype == torch.float32
            assert r['best']['epoch'] == int(np.argmin([h['val_original'] for h in r['history']]))
            assert int(r['best']['core']['state']['clf.2.num_batches_tracked']) == r['best']['epoch'] + 1
            metadata[f'{g}_f{f}'] = {'head_sha256': tree_hash(r['best']['head']),
                'core_sha256': tree_hash(r['best']['core']), 'best_epoch': r['best']['epoch'],
                'test_rng_sha256': tree_hash(r['test_rng']), 'image_sha256': array_sha(r['test_image_embedding']),
                'image_shape': list(r['test_image_embedding'].shape), 'image_dtype': str(r['test_image_embedding'].dtype)}
    assert np.all(coverage == 1)
    sources = {}
    for name in ('nbs_data/get_fmri_discriptor.py', 'train_quantizer_contr.py', 'utils_loss.py'):
        url = 'https://raw.githubusercontent.com/yuxiangwei0808/fMRI-LM/main/' + name
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read()
        sources[name] = {'url': url, 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}
    audit = {'experiment': CFG['experiment'], 'started_utc': utc(), 'started_epoch': start,
        'preregistered_before_analysis': True, 'code_sha256': code_hash(), 'config_sha256': sha_file(BASE / 'config.json'),
        'protected_files': protected, 'historical_prompt_revision': changes,
        'storage': storage_check(), 'historical_artifacts': metadata,
        'historical_summary_sha256': sha_file(PILOT / 'summary.json'),
        'historical_11_results': {g: historical_summary['groups'][g]['mean'] for g in ('C0', 'S1')},
        'historical_11_delta': historical_summary['paired_delta_S1_minus_C0'],
        'handoff_before_sha256': sha_file(ROOT / 'handoff.md'), 'official_sources': sources,
        'paper': 'https://arxiv.org/html/2511.21760v3',
        'architecture_audit': {
            'our_11': 'single 16D native imaging output, 16-128-384 auxiliary head, frozen MiniLM FC+graph captions; symmetric row/column InfoNCE / log(Ntrain), tau .07; no text at diagnosis inference',
            'official': 'four imaging descriptor families; tokenizer reconstruction/VQ + domain-adversarial + contrastive, followed by LLM and instruction stages; not reproduced by our pilot',
            'paper_code_loss_difference': 'Paper calls loss SigLIP-style but writes row-softmax Eq3; utils_loss.py separates clip, soft_clip, siglip BCE and soft_siglip; do not conflate them',
            'descriptor_ablation_boundary': 'Paper removes both pretraining paired contrastive and F2T; not an isolated demonstration that this pilot auxiliary objective improves ASD',
            'roi_semantics': 'pilot uses one-based ROI numbers, not verified anatomical language'},
        'execution': {'neural_training': 0, 'GTUNet_training': 0, 'GTUNet_forward': 0, 'VAE_training': 0,
                      'semantic_head_training': 0, 'text_encoder_training': 0, 'RFE_fitting': 0,
                      'Ridge_fits': 0, 'Logistic_fits': 0, 'resumed_stages': []}}
    save_json(path, audit)
    log(f'Frozen audit passed: {len(protected)} protected files, 20 checkpoints, 10 splits; code={code_hash()}')
    return audit


def build_plan(context, cache, splits):
    strings = []
    lookup = {}
    initial = []
    def add(text):
        if text not in lookup:
            lookup[text] = len(strings)
            strings.append(text)
        return lookup[text]
    for f in range(10):
        for text, embedding in zip(cache['folds'][f]['captions'], cache['folds'][f]['embedding']):
            if text not in lookup:
                add(text)
                initial.append(embedding)
            else:
                assert torch.equal(initial[lookup[text]], embedding)
    n_initial = len(strings)
    stats, pairs, ids = cache['raw_stats'], cache['pairs'], context['ids']
    plan = {'controls': {}, 'A3': {}, 'B': {}, 'ids': ids, 'initial_count': n_initial}
    controlled_count = 0
    for f in range(10):
        outer = np.asarray(splits[f]['train'])
        ref = cache['folds'][f]['reference']
        selected = sorted(outer, key=lambda i: hashlib.sha256(f'911/{f}/{ids[i]}/GBSA-Diagnostic-controls'.encode()).hexdigest())[:CFG['A2']['patients_per_fold']]
        records = []
        for i in selected:
            row = stats[i].copy()
            record = {'subject_index': int(i), 'base': add(cache['folds'][f]['captions'][i]), 'concepts': {}}
            controlled_count += 1
            for name, col in CFG['A2']['concepts'].items():
                values = np.quantile(stats[outer, col], CFG['A2']['low_high_quantiles'])
                if col == 0:
                    bound = np.floor(row[1] * 1000) / 1000
                    values = np.clip(values, -bound, bound)
                values = np.round(values, 3)
                assert values[0] < values[1], 'Control interval is degenerate'
                change = {'low_high_values': values, 'low_high': []}
                for v in values:
                    x = row.copy()
                    x[col] = v
                    text = DESC.captions(x[None], [pairs[i]], ref)[0]
                    change['low_high'].append(add(text))
                    controlled_count += 1
                z0 = (row[col] - ref['mean'][col]) / ref['sd'][col]
                chosen = None
                for step in (CFG['A2']['small_numeric_step'], -CFG['A2']['small_numeric_step']):
                    x = row.copy()
                    x[col] += step
                    valid = abs(x[col]) <= row[1] if col == 0 else 0 <= x[col] <= 1
                    same_category = DESC.category((x[col] - ref['mean'][col]) / ref['sd'][col]) == DESC.category(z0)
                    text = DESC.captions(x[None], [pairs[i]], ref)[0]
                    if valid and same_category and text != cache['folds'][f]['captions'][i]:
                        chosen = {'index': add(text), 'step': step, 'value': x[col]}
                        controlled_count += 1
                        break
                change['small'] = chosen
                record['concepts'][name] = change
            p = {k: list(v) for k, v in pairs[i].items()}
            a, b, v = p['positive'][0]
            new_a = a % 116 + 1
            if new_a == b:
                new_a = new_a % 116 + 1
            assert new_a != a and new_a != b
            p['positive'][0] = (new_a, b, v)
            record['roi'] = add(DESC.captions(row[None], [p], ref)[0])
            record['roi_substitution'] = [a, new_a, b]
            controlled_count += 1
            records.append(record)
        plan['controls'][f] = records
        for stage in ('A3', 'B'):
            splitter = KFold(3, shuffle=True, random_state=911+f) if stage == 'A3' else StratifiedKFold(3, shuffle=True, random_state=911+f)
            iterator = splitter.split(outer) if stage == 'A3' else splitter.split(outer, context['labels'][outer])
            entries = []
            for k, (train_local, holdout_local) in enumerate(iterator):
                train, holdout = outer[train_local], outer[holdout_local]
                assert not np.intersect1d(train, holdout).size
                assert set(train) | set(holdout) == set(outer)
                ref_inner = DESC.reference(stats, train)
                combined = np.concatenate([train, holdout])
                captions = DESC.captions(stats[combined], [pairs[i] for i in combined], ref_inner)
                positions = np.asarray([add(t) for t in captions])
                entries.append({'inner_fold': k, 'train': train, 'holdout': holdout,
                    'reference': ref_inner, 'text_train': positions[:len(train)], 'text_holdout': positions[len(train):]})
                if stage == 'B':
                    for key, members in (('train', train), ('holdout', holdout)):
                        rng = np.random.default_rng(private_seed(911, f, k, key, 'GBSA-Diagnostic-probe-null'))
                        perm = rng.permutation(len(members))
                        assert np.array_equal(np.sort(perm), np.arange(len(members)))
                        entries[-1][key + '_permutation'] = perm
            plan[stage][f] = entries
    assert controlled_count <= CFG['A2']['max_controlled_sentences']
    plan['controlled_sentence_count_upper_bound'] = controlled_count
    plan['texts'] = strings
    plan['initial_embedding'] = torch.stack(initial)
    plan['plan_sha256'] = tree_hash({k: v for k, v in plan.items() if k != 'initial_embedding'})
    plan['code_sha256'] = code_hash()
    return plan


def encode(model, tokenizer, texts, device):
    lengths = [len(x) for x in tokenizer(texts, add_special_tokens=True)['input_ids']]
    assert max(lengths) <= 256, 'A caption would be truncated'
    rows = []
    with torch.inference_mode():
        for begin in range(0, len(texts), CFG['batch_size']):
            inp = tokenizer(texts[begin:begin+CFG['batch_size']], padding=True, truncation=False, return_tensors='pt')
            inp = {k: v.to(device) for k, v in inp.items()}
            hidden = model(**inp).last_hidden_state
            mask = inp['attention_mask'].unsqueeze(-1).to(hidden.dtype)
            result = F.normalize((hidden*mask).sum(1)/mask.sum(1).clamp_min(1), dim=1)
            assert torch.isfinite(result).all()
            rows.append(result.cpu())
    return torch.cat(rows), max(lengths)


def encode_shard(gpu):
    from transformers import AutoModel, AutoTokenizer
    from transformers.utils import logging
    logging.disable_progress_bar()
    start = time.perf_counter()
    plan = load(BASE / 'runs/text_cache.bin')
    assert plan['code_sha256'] == code_hash()
    use_ppu = CFG['diagnostic_encoding_backend'] != 'cpu'
    if use_ppu:
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1
        assert 'PPU' in torch.cuda.get_device_name(0)
    device = torch.device('cuda:0' if use_ppu else 'cpu')
    folder = PILOT / 'runs/text_model'
    tokenizer = AutoTokenizer.from_pretrained(folder, local_files_only=True)
    model = AutoModel.from_pretrained(folder, local_files_only=True, attn_implementation='eager').eval()
    for p in model.parameters():
        p.requires_grad_(False)
    before = tree_hash(model.state_dict())
    indices = np.arange(plan['initial_count'] if use_ppu else 0, len(plan['texts']))
    indices = indices[indices % 2 == gpu]
    texts = [plan['texts'][i] for i in indices]
    cpu, _ = encode(model, tokenizer, texts[:4], 'cpu')
    model.to(device)
    ppu, _ = encode(model, tokenizer, texts[:4], device)
    torch.testing.assert_close(cpu, ppu, rtol=CFG['numerical_checks']['text_CPU_PPU_rtol'], atol=CFG['numerical_checks']['text_CPU_PPU_atol'])
    repeat_texts = texts[:CFG['A2']['repeat_sentences']]
    r1, _ = encode(model, tokenizer, repeat_texts, device)
    r2, _ = encode(model, tokenizer, repeat_texts, device)
    assert torch.equal(r1, r2), 'Identical repeated batch encoding is not bitwise deterministic'
    existing_texts = plan['texts'][:16]
    existing, _ = encode(model, tokenizer, existing_texts, device)
    if use_ppu:
        torch.testing.assert_close(existing, plan['initial_embedding'][:16], rtol=1e-5, atol=2e-6)
        torch.cuda.reset_peak_memory_stats()
    embedding, max_tokens = encode(model, tokenizer, texts, device)
    after = tree_hash(model.state_dict())
    assert before == after and not any(p.requires_grad for p in model.parameters())
    assert before == json.loads((PILOT / 'audit.json').read_text())['descriptors']['text_model']['state_sha256_before_after']
    result = {'code_sha256': code_hash(), 'plan_sha256': plan['plan_sha256'], 'gpu': gpu,
        'indices': indices, 'embedding': embedding, 'max_tokens': max_tokens,
        'metadata': {'logical_shard': gpu, 'backend': str(device), 'physical_gpu': gpu if use_ppu else None,
            'unique_texts_encoded': len(indices), 'seconds': time.perf_counter()-start,
            'repeat_exact': True, 'repeat_max_abs_error': float((r1-r2).abs().max()),
            'CPU_PPU_max_abs_error': float((cpu-ppu).abs().max()) if use_ppu else None,
            'historical_embedding_max_abs_error': float((existing-plan['initial_embedding'][:16]).abs().max()),
            'model_state_sha256_before_after': before, 'model_frozen': True,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated() if use_ppu else 0, 'max_tokens': max_tokens}}
    save(BASE / f'runs/.encoded_gpu{gpu}.bin', result)
    log(f'Frozen text shard{gpu} backend={device}: {len(indices)} unique sentences')


def prepare_text(context, cache, splits, audit):
    path = BASE / 'runs/text_cache.bin'
    if path.exists():
        plan = load(path)
        assert plan['code_sha256'] == code_hash()
        if plan.get('complete'):
            assert array_sha(plan['embedding']) == plan['embedding_sha256']
            audit['execution']['resumed_stages'].append('text_cache')
            return plan
    else:
        plan = build_plan(context, cache, splits)
        save(path, plan)
    log(f'Text plan frozen: {len(plan["texts"])} unique texts, {plan["initial_count"]} historical cached; controls <= {plan["controlled_sentence_count_upper_bound"]}')
    children = []
    for gpu in (0, 1):
        shard = BASE / f'runs/.encoded_gpu{gpu}.bin'
        if shard.exists():
            r = load(shard)
            assert r['plan_sha256'] == plan['plan_sha256'] and r['code_sha256'] == code_hash()
            continue
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu)}
        child = subprocess.Popen([sys.executable, '-B', '-u', str(BASE / 'diagnose.py'), 'encode', '--gpu', str(gpu)], env=env, cwd=BASE)
        children.append(child)
    returncodes = [child.wait() for child in children]
    assert all(rc == 0 for rc in returncodes), 'Frozen text encoding worker failed; inspect stderr'
    embeddings = torch.empty(len(plan['texts']), 384)
    embeddings[:plan['initial_count']] = plan['initial_embedding']
    counts = np.zeros(len(plan['texts']), int)
    if CFG['diagnostic_encoding_backend'] != 'cpu':
        counts[:plan['initial_count']] = 1
    metadata = []
    for gpu in (0, 1):
        r = load(BASE / f'runs/.encoded_gpu{gpu}.bin')
        assert r['plan_sha256'] == plan['plan_sha256']
        embeddings[r['indices']] = r['embedding']
        counts[r['indices']] += 1
        metadata.append(r['metadata'])
    assert np.all(counts == 1) and torch.isfinite(embeddings).all()
    plan.pop('initial_embedding')
    plan.update({'embedding': embeddings, 'embedding_sha256': array_sha(embeddings), 'complete': True, 'encoding': metadata})
    save(path, plan)
    for gpu in (0, 1):
        (BASE / f'runs/.encoded_gpu{gpu}.bin').unlink()
    audit['execution']['text_encoding'] = metadata
    save_json(BASE / 'audit.json', audit)
    return plan


def effective_rank(singular, squared=False):
    values = singular**2 if squared else singular
    values = values[values > 0]
    if not values.size:
        return 0.
    p = values/values.sum()
    return float(np.exp(-np.sum(p*np.log(p))))


def geometry(cache, splits):
    result = {}
    for f in range(10):
        idx = splits[f]['train']
        a = cache['folds'][f]['embedding'][idx].double().numpy()
        norm = np.linalg.norm(a, axis=1)
        centered = a-a.mean(0)
        s = np.linalg.svd(centered, compute_uv=False)
        cosine = (a/norm[:, None]) @ (a/norm[:, None]).T
        off = cosine[np.triu_indices(len(a), 1)]
        dist = np.sqrt(np.maximum(0., 2-2*off))
        historical = cache['folds'][f]['diagnostics']
        raw_off = (a @ a.T)[np.triu_indices(len(a), 1)]
        np.testing.assert_allclose(np.quantile(raw_off, [0, .05, .5, .95, 1]), historical['cosine_quantiles_0_05_50_95_100'], rtol=0, atol=1e-12)
        assert abs(effective_rank(s)-historical['effective_rank']) < 1e-8
        texts = [cache['folds'][f]['captions'][i] for i in idx]
        result[f] = {'n_train': len(a), 'distinct_pairs': len(off), 'mean_norm': float(norm.mean()),
            'centered_energy': float(np.square(centered).sum(1).mean()),
            'covariance_effective_rank': effective_rank(s, squared=True), 'historical_singular_value_effective_rank': effective_rank(s),
            'cosine_quantiles_P05_P25_P50_P75_P95': np.quantile(off, [.05, .25, .5, .75, .95]),
            'fraction_cosine_above': {str(t): float((off > t).mean()) for t in CFG['A1']['cosine_thresholds']},
            'duplicate_text_fraction': 1-len(set(texts))/len(texts),
            'euclidean_distance_min_mean_max': [float(dist.min()), float(dist.mean()), float(dist.max())],
            'historical_diagnostics_reproduced': True, 'embedding_sha256': array_sha(a)}
    return result


def distance(a, b):
    a = a/np.linalg.norm(a, axis=-1, keepdims=True)
    b = b/np.linalg.norm(b, axis=-1, keepdims=True)
    return 1-np.sum(a*b, axis=-1)


def controlled_analysis(plan):
    emb = plan['embedding'].double().numpy()
    output = {}
    for f, records in plan['controls'].items():
        base = emb[[r['base'] for r in records]]
        roi = emb[[r['roi'] for r in records]]
        row = {'n_subjects': len(records), 'roi_cosine_distance': distance(base, roi), 'concepts': {}}
        directions = []
        for name in CFG['A2']['concepts']:
            low = emb[[r['concepts'][name]['low_high'][0] for r in records]]
            high = emb[[r['concepts'][name]['low_high'][1] for r in records]]
            delta = high-low
            unit = delta/np.linalg.norm(delta, axis=1, keepdims=True)
            directions.append(unit.mean(0))
            correct = []
            for i in range(len(records)):
                other = np.arange(len(records)) != i
                lo, hi = low[other].mean(0), high[other].mean(0)
                direction = hi-lo
                midpoint = (hi+lo)/2
                correct.extend([np.dot(low[i]-midpoint, direction) < 0, np.dot(high[i]-midpoint, direction) > 0])
            valid_small = [i for i, r in enumerate(records) if r['concepts'][name]['small'] is not None]
            small = emb[[records[i]['concepts'][name]['small']['index'] for i in valid_small]]
            similarities = unit @ unit.T
            row['concepts'][name] = {
                'low_to_high_cosine_distance': distance(low, high), 'base_to_low_cosine_distance': distance(base, low),
                'base_to_high_cosine_distance': distance(base, high),
                'small_numeric_cosine_distance': distance(base[valid_small], small),
                'small_valid_cases': len(valid_small), 'leave_one_patient_out_balanced_accuracy': float(np.mean(correct)),
                'mean_direction_cosine': float(similarities[np.triu_indices(len(records), 1)].mean()),
                'low_high_values': np.stack([r['concepts'][name]['low_high_values'] for r in records])}
        directions = np.stack(directions)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        row['between_concept_mean_direction_cosine'] = directions @ directions.T
        output[f] = row
    return output


def diagnostic_metrics(y, probability):
    asd = (np.asarray(y) == 0).astype(int)
    p = np.asarray(probability, dtype=np.float64)
    assert np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
    return {'ACC': float(accuracy_score(asd, p >= .5)), 'BA': float(balanced_accuracy_score(asd, p >= .5)),
            'AUC': float(roc_auc_score(asd, p)), 'CE': float(log_loss(asd, p, labels=[0, 1])),
            'Brier': float(np.mean((asd-p)**2))}


def lightweight_probes(context, cache, splits, plan):
    embeddings = plan['embedding'].double().numpy()
    raw = cache['raw_stats']
    result = {'A3': {}, 'B': {}, 'fit_counts': {'Ridge': 0, 'Logistic': 0}}
    for f in range(10):
        start = time.perf_counter()
        outer = np.asarray(splits[f]['train'])
        local = {int(g): i for i, g in enumerate(outer)}
        regression = np.full((len(outer), 7), np.nan)
        baseline = np.full_like(regression, np.nan)
        coverage = np.zeros(len(outer), int)
        a3_fits = []
        for part in plan['A3'][f]:
            tr, va = part['train'], part['holdout']
            assert tree_hash(DESC.reference(raw, tr)) == tree_hash(part['reference'])
            xtr, xva = embeddings[part['text_train']], embeddings[part['text_holdout']]
            scaler = StandardScaler().fit(xtr)
            model = Ridge(alpha=1.0).fit(scaler.transform(xtr), raw[tr])
            loc = [local[int(i)] for i in va]
            regression[loc] = model.predict(scaler.transform(xva))
            baseline[loc] = raw[tr].mean(0)
            coverage[loc] += 1
            result['fit_counts']['Ridge'] += 1
            a3_fits.append({'inner_fold': part['inner_fold'], 'train_sha256': array_sha(tr),
                'reference_sha256': tree_hash(part['reference']), 'scaler_sha256': tree_hash([scaler.mean_, scaler.scale_]),
                'coefficient_sha256': tree_hash([model.coef_, model.intercept_]), 'labels_accessed': False})
        assert np.all(coverage == 1) and np.isfinite(regression).all()
        y = raw[outer]
        result['A3'][f] = {'indices': outer, 'prediction': regression, 'baseline_prediction': baseline,
            'R2': r2_score(y, regression, multioutput='raw_values'), 'MAE': mean_absolute_error(y, regression, multioutput='raw_values'),
            'baseline_R2': r2_score(y, baseline, multioutput='raw_values'),
            'baseline_MAE': mean_absolute_error(y, baseline, multioutput='raw_values'), 'fits': a3_fits}
        probability = {g: np.full(len(outer), np.nan) for g in CFG['B']['groups']}
        coverage = np.zeros(len(outer), int)
        fits = []
        for part in plan['B'][f]:
            tr, va = part['train'], part['holdout']
            assert set(tr).issubset(set(outer)) and set(va).issubset(set(outer))
            assert tree_hash(DESC.reference(raw, tr)) == tree_hash(part['reference'])
            text_tr, text_va = embeddings[part['text_train']], embeddings[part['text_holdout']]
            for group in CFG['B']['groups']:
                if group == 'B0_raw7':
                    xtr, xva = raw[tr], raw[va]
                elif group == 'B1_text384':
                    xtr, xva = text_tr, text_va
                else:
                    xtr, xva = text_tr[part['train_permutation']], text_va[part['holdout_permutation']]
                scaler = StandardScaler().fit(xtr)
                model = LogisticRegression(C=1., solver='lbfgs', max_iter=2000, tol=1e-6, class_weight=None)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    model.fit(scaler.transform(xtr), context['labels'][tr])
                p = model.predict_proba(scaler.transform(xva))[:, list(model.classes_).index(0)]
                loc = [local[int(i)] for i in va]
                probability[group][loc] = p
                result['fit_counts']['Logistic'] += 1
                fits.append({'group': group, 'inner_fold': part['inner_fold'], 'train_indices': tr,
                    'holdout_indices': va, 'train_labels_sha256': array_sha(context['labels'][tr]),
                    'train_features_sha256': array_sha(xtr), 'scaler_sha256': tree_hash([scaler.mean_, scaler.scale_]),
                    'coefficient_sha256': tree_hash([model.coef_, model.intercept_]), 'classes': model.classes_,
                    'n_iter': model.n_iter_, 'warnings': [str(w.message) for w in caught],
                    'reference_sha256': tree_hash(part['reference']), 'inner_metrics': diagnostic_metrics(context['labels'][va], p)})
            coverage[[local[int(i)] for i in va]] += 1
        assert np.all(coverage == 1) and all(np.isfinite(v).all() for v in probability.values())
        result['B'][f] = {'indices': outer, 'probability': probability, 'fits': fits,
            'metrics': {g: diagnostic_metrics(context['labels'][outer], p) for g, p in probability.items()},
            'seconds': time.perf_counter()-start,
            'null_actual_subject_change_fraction': [{k: float((part[k + '_permutation'] != np.arange(len(part[k]))).mean())
                                                     for k in ('train', 'holdout')} for part in plan['B'][f]]}
        log(f'Fold{f}: inner Ridge and diagnostic LR finished; outer held-out labels not used')
    assert result['fit_counts'] == {'Ridge': 30, 'Logistic': 90}
    return result


def rank_metrics(ranks, n_candidates):
    ranks = np.asarray(ranks)
    return {'R1': float((ranks <= 1).mean()), 'R5': float((ranks <= 5).mean()),
            'MRR': float((1./ranks).mean()), 'median_rank': float(np.median(ranks)),
            'mean_rank_percentile': float((100*(ranks-1)/(n_candidates-1)).mean())}


def retrieval(scores, query_ids, candidate_ids, fold, setting):
    assert len(set(candidate_ids)) == len(candidate_ids)
    mapping = {sid: i for i, sid in enumerate(candidate_ids)}
    truth = np.asarray([mapping[sid] for sid in query_ids])
    n, m = scores.shape
    order = np.argsort(-scores, axis=1, kind='stable')
    all_ranks = np.empty_like(order)
    np.put_along_axis(all_ranks, order, np.broadcast_to(np.arange(1, m+1), (n, m)), axis=1)
    q = np.arange(n)
    matched = scores[q, truth]
    unmatched = (scores.sum(1)-matched)/(m-1)
    ranks = all_ranks[q, truth]
    nulls = []
    for rep in range(64):
        rng = np.random.default_rng(private_seed(911, fold, setting, rep, 'GBSA-Diagnostic-retrieval-null'))
        perm = rng.permutation(m)
        inverse = np.empty_like(perm)
        inverse[perm] = np.arange(m)
        positions = inverse[truth]
        rr = all_ranks[q, positions]
        actual_cosine = scores[q, positions]
        gap = actual_cosine-(scores.sum(1)-actual_cosine)/(m-1)
        nulls.append({'metrics': {**rank_metrics(rr, m), 'cosine_advantage': float(gap.mean())},
                      'permutation_sha256': array_sha(perm), 'ranks': rr.astype(np.int16), 'paired_cosine': actual_cosine})
    tie_count = int(np.sum(np.sum(scores == matched[:, None], axis=1)-1))
    random_baseline = {'R1': 1/m, 'R5': min(5, m)/m, 'MRR': float((1./np.arange(1, m+1)).mean()),
                       'median_rank': (m+1)/2, 'mean_rank_percentile': 50.}
    return {'n_queries': n, 'n_candidates': m, 'query_ids': query_ids, 'candidate_ids': candidate_ids,
        'truth_positions': truth, 'metrics': rank_metrics(ranks, m), 'random_baseline': random_baseline,
        'ranks': ranks, 'paired_cosine': matched, 'unpaired_mean_cosine': unmatched,
        'cosine_advantage': matched-unmatched, 'matched_above_mean_unmatched_fraction': float((matched > unmatched).mean()),
        'advantage_quantiles_P05_P25_P50_P75_P95': np.quantile(matched-unmatched, [.05, .25, .5, .75, .95]),
        'exact_score_ties_at_truth': tie_count, 'nulls': nulls,
        'score_sha256': array_sha(scores), 'labels_accessed': False}


def state_difference(a, b):
    keys = sorted(set(a) & set(b))
    difference = 0.
    norm = 0.
    max_abs = 0.
    equal = 0
    for k in keys:
        av, bv = a[k].double(), b[k].double()
        d = bv-av
        difference += float(d.square().sum())
        norm += float(av.square().sum())
        max_abs = max(max_abs, float(d.abs().max()))
        equal += torch.equal(a[k], b[k])
    return {'tensor_count': len(keys), 'equal_tensor_count': equal, 'relative_L2': (difference/max(norm, 1e-30))**.5, 'max_abs': max_abs}


def frozen_checkpoints(context, cache, splits, rows):
    result = {'retrieval': {}, 'checkpoint_audit': {}, 'projection_checks': {}, 'late_training': {}}
    for f in range(10):
        r = rows['S1', f]
        with torch.random.fork_rng(devices=[]):
            head = SEM.SemanticHead().eval()
        head.load_state_dict(r['best']['head'], strict=True)
        for p in head.parameters():
            p.requires_grad_(False)
        before = tree_hash(head.state_dict())
        with torch.no_grad():
            projected = F.normalize(head.project(r['test_image_embedding']), dim=1)
            text = F.normalize(cache['folds'][f]['embedding'], dim=1)
        assert torch.isfinite(projected).all() and torch.isfinite(text).all()
        assert before == tree_hash(head.state_dict()) == tree_hash(r['best']['head'])
        x = r['test_image_embedding'].numpy().astype(np.float64)
        state = r['best']['head']
        first = x @ state['project.0.weight'].numpy().T + state['project.0.bias'].numpy()
        from scipy.special import ndtr
        hidden = first*ndtr(first)
        reference = hidden @ state['project.2.weight'].numpy().T + state['project.2.bias'].numpy()
        reference /= np.linalg.norm(reference, axis=1, keepdims=True)
        error = float(np.abs(reference-projected.double().numpy()).max())
        assert error < 1e-6, 'Independent frozen head definition differs'
        z, t = projected.double().numpy(), text.double().numpy()
        result['projection_checks'][f] = {'head_sha256_before_after': before, 'independent_float64_max_abs_error': error,
                                          'requires_grad': False, 'GTUNet_forward': 0}
        result['retrieval'][f] = {}
        for setting in CFG['C']['settings']:
            query = np.asarray(splits[f]['train'] if setting == 'train_to_train' else splits[f]['test'])
            candidate = np.sort(np.asarray(splits[f]['train'] if setting == 'train_to_train' else splits[f]['test'] if setting == 'test_to_test' else np.arange(871)))
            scores = z[query] @ t[candidate].T
            result['retrieval'][f][setting] = retrieval(scores, [context['ids'][i] for i in query],
                [context['ids'][i] for i in candidate], f, setting)
        a, b = rows['C0', f], r
        st_a, st_b = a['best']['core']['state'], b['best']['core']['state']
        bn = lambda state: {k: v for k, v in state.items() if 'running_' in k or k.endswith('num_batches_tracked')}
        core_count = len(a['pairing']['original_optimizer_parameter_order'])
        def core_adam(obj):
            adam = obj['best']['optimizer']
            order = adam['param_groups'][0]['params'][:core_count]
            return {name: adam['state'].get(key) for name, key in zip(a['pairing']['original_optimizer_parameter_order'], order)}
        entry = {'best_epochs': [a['best']['epoch'], b['best']['epoch']], 'actual_training_epochs': [a['epochs'], b['epochs']],
            'same_initial_pairing': a['pairing'] == b['pairing'],
            'core_sha256_C0_S1': [tree_hash(a['best']['core']), tree_hash(b['best']['core'])],
            'BN_sha256_C0_S1': [tree_hash(bn(st_a)), tree_hash(bn(st_b))], 'BN_difference': state_difference(bn(st_a), bn(st_b)),
            'BN_batch_counts_C0_S1': [a['BN_batch_count'], b['BN_batch_count']],
            'core_Adam_sha256_C0_S1': [tree_hash(core_adam(a)), tree_hash(core_adam(b))],
            'optimizer_sha256_C0_S1': [tree_hash(a['best']['optimizer']), tree_hash(b['best']['optimizer'])],
            'probability_MAD': float(np.abs(a['prob_asd']-b['prob_asd']).mean()),
            'state_difference_by_part': {prefix: state_difference({k: v for k, v in st_a.items() if k.startswith(prefix)},
                {k: v for k, v in st_b.items() if k.startswith(prefix)}) for prefix in ('img_unet.', 'ph_unet.', 'clf.')},
            'metrics_C0_S1': [a['metrics'], b['metrics']]}
        for key in ('test_rng', 'end_rng'):
            entry[key + '_sha256_C0_S1'] = [tree_hash(a[key]), tree_hash(b[key])]
            entry[key + '_component_equal'] = {component: tree_hash(a[key][component]) == tree_hash(b[key][component]) for component in a[key]}
        entry['best_rng_sha256_C0_S1'] = [tree_hash(a['best']['rng']), tree_hash(b['best']['rng'])]
        entry['test_rng_equal'] = tree_hash(a['test_rng']) == tree_hash(b['test_rng'])
        result['checkpoint_audit'][f] = entry
        mechanisms = [h for h in r['history'] if 'mechanism' in h]
        last = mechanisms[-1]
        result['late_training'][f] = {'sampled_epoch': last['epoch'], 'best_epoch': r['best']['epoch'],
            'matched_minus_unmatched_cosine': last['mechanism']['matched_cosine']-last['mechanism']['unmatched_cosine'],
            'semantic_normalized': last['semantic_normalized'], 'mechanism': last['mechanism']}
        log(f'Fold{f}: frozen head, three retrieval settings and 64 nulls each complete')
    return result


def run_all():
    (BASE / 'runs').mkdir(exist_ok=True)
    with locked(BASE / 'runs/.coordinator.lock'):
        start = time.perf_counter()
        context, cache, splits, rows = read_inputs()
        audit = audit_inputs(context, cache, splits, rows)
        output_path = BASE / 'runs/diagnostics.bin'
        if output_path.exists():
            output = load(output_path)
            assert output['code_sha256'] == code_hash()
            if output.get('complete'):
                audit['execution']['resumed_stages'].append('all_diagnostics')
                save_json(BASE / 'audit.json', audit)
                log('SKIP verified complete diagnosis; no fitting or encoding repeated')
                return
        else:
            output = {'code_sha256': code_hash(), 'started_utc': utc()}
        if 'A1' not in output:
            output['A1'] = geometry(cache, splits)
            save(output_path, output)
        plan = prepare_text(context, cache, splits, audit)
        if 'A2' not in output:
            output['A2'] = controlled_analysis(plan)
            save(output_path, output)
        if 'probes' not in output:
            output['probes'] = lightweight_probes(context, cache, splits, plan)
            save(output_path, output)
        if 'frozen' not in output:
            output['frozen'] = frozen_checkpoints(context, cache, splits, rows)
            save(output_path, output)
        protected_guard(audit)
        output.update({'complete': True, 'completed_utc': utc(), 'analysis_seconds': time.perf_counter()-start})
        save(output_path, output)
        audit['execution'].update({'analysis_seconds': output['analysis_seconds'],
            'Ridge_fits': output['probes']['fit_counts']['Ridge'], 'Logistic_fits': output['probes']['fit_counts']['Logistic'],
            'text_encoding': plan['encoding'], 'complete_folds': 10,
            'retrieval_settings': 30, 'retrieval_null_evaluations': 1920,
            'controlled_sentence_count_upper_bound': plan['controlled_sentence_count_upper_bound'],
            'unique_text_cache_rows': len(plan['texts'])})
        audit['diagnostics_sha256'] = sha_file(output_path)
        audit['text_cache_sha256'] = sha_file(BASE / 'runs/text_cache.bin')
        audit['diagnostics_completed_utc'] = utc()
        save_json(BASE / 'audit.json', audit)
        log(f'Frozen diagnosis complete: 10/10 folds, neural training 0, seconds={output["analysis_seconds"]:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['all', 'encode'])
    parser.add_argument('--gpu', type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.command == 'encode':
        encode_shard(args.gpu)
    else:
        run_all()
