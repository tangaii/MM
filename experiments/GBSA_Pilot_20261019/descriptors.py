"""Deterministic, diagnosis-free FC captions and frozen MiniLM caches."""
from __future__ import annotations
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import urllib.request
import numpy as np
import torch

BASE = Path(__file__).resolve().parent
CFG = json.loads((BASE / 'config.json').read_text())
ROOT = Path(CFG['root'])
OLD = ROOT / CFG['history']


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


def atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.artifact-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        if path.read_bytes() != payload:
            raise OSError('Atomic artifact readback mismatch: ' + str(path))
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


def pack(value):
    stream = io.BytesIO()
    torch.save(value, stream)
    payload = stream.getvalue()
    return b'GBSA1 ' + hashlib.sha256(payload).hexdigest().encode() + b'\n' + payload


def unpack(payload):
    header, data = payload.split(b'\n', 1)
    assert header == b'GBSA1 ' + hashlib.sha256(data).hexdigest().encode(), 'Artifact SHA mismatch'
    return torch.load(io.BytesIO(data), weights_only=False, map_location='cpu')


def load(path):
    return unpack(Path(path).read_bytes())


def save(path, value):
    atomic_bytes(path, pack(value))


def history_load(path):
    return torch.load(path, weights_only=False, map_location='cpu')


def reference(stats, train):
    mean = stats[train].mean(0)
    sd = stats[train].std(0)
    return {'mean': mean, 'sd': np.where(sd > 1e-12, sd, 1.), 'train_sha': array_sha(train)}


def category(z):
    return ('markedly low', 'low', 'typical', 'high', 'markedly high')[int(np.searchsorted([-2., -1., 1., 2.], z, side='left'))]


def raw_statistics(fc):
    import networkx as nx
    from scipy.sparse.csgraph import shortest_path
    from scipy.sparse import csr_matrix
    n = fc.shape[0]
    assert fc.shape == (116, 116) and np.isfinite(fc).all()
    assert np.allclose(fc, fc.T, rtol=0, atol=1e-10)
    assert np.abs(fc).max() <= 1 + 1e-10
    u, v = np.triu_indices(n, 1)
    values = fc[u, v]
    ranked = np.argsort(values, kind='stable')
    pos = ranked[values[ranked] > 0][-CFG['top_connections']:][::-1]
    neg = ranked[values[ranked] < 0][:CFG['top_connections']]
    chosen = ranked[values[ranked] > 0][-int(CFG['graph_density'] * len(values)):]
    assert len(chosen) == 667, 'Too few positive connections for registered density'
    a = np.zeros_like(fc)
    a[u[chosen], v[chosen]] = values[chosen]
    a += a.T
    distance = np.zeros_like(a)
    distance[a > 0] = 1 / a[a > 0]
    d = shortest_path(csr_matrix(distance), directed=False)
    np.fill_diagonal(d, np.inf)
    efficiency = np.divide(1., d, where=np.isfinite(d), out=np.zeros_like(d)).sum() / (n * (n - 1))
    g = nx.from_numpy_array(a)
    communities = nx.community.greedy_modularity_communities(g, weight='weight', resolution=1)
    modularity = nx.community.modularity(g, communities, weight='weight', resolution=1)
    clustering = nx.average_clustering(g, weight='weight')
    strength = a.sum(1)
    stats = np.array([values.mean(), np.abs(values).mean(), efficiency, modularity, clustering, strength.mean(), strength.std()])
    pairs = {'positive': [(int(u[k] + 1), int(v[k] + 1), float(values[k])) for k in pos],
             'negative': [(int(u[k] + 1), int(v[k] + 1), float(values[k])) for k in neg]}
    assert np.isfinite(stats).all()
    return stats, pairs


def captions(stats, pairs, fitted):
    z = (stats - fitted['mean']) / fitted['sd']
    texts = []
    for row, zz, p in zip(stats, z, pairs):
        def fmt(key):
            return ', '.join(f'ROI {a}-{b} ({v:+.3f})' for a, b, v in p[key]) or 'none'
        texts.append(
            f'Functional connectivity. Whole-brain mean {row[0]:.3f} ({category(zz[0])}); '
            f'mean absolute connectivity {row[1]:.3f} ({category(zz[1])}). '
            f'Strongest positive pairs: {fmt("positive")}. Strongest negative pairs: {fmt("negative")}. '
            f'Graph metrics from a ten-percent positive weighted graph. '
            f'Global efficiency {row[2]:.3f} ({category(zz[2])}); modularity {row[3]:.3f} ({category(zz[3])}); '
            f'average clustering {row[4]:.3f} ({category(zz[4])}); '
            f'mean node strength {row[5]:.2f} ({category(zz[5])}); strength variability {row[6]:.2f}.')
    import re
    forbidden = r'\b(ASD|HC|autism|diagnosis|diagnostic|patient|male|female|sex|age|site)\b'
    assert not any(re.search(forbidden, text, re.I) for text in texts)
    return texts


def download_text_model():
    folder = BASE / 'runs' / 'text_model'
    names = ('config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json',
             'special_tokens_map.json', 'vocab.txt', 'sentence_bert_config.json', 'modules.json', '1_Pooling/config.json')
    manifests = {}
    for name in names:
        dest = folder / name
        if not dest.exists():
            url = f'https://huggingface.co/{CFG["text_model"]}/resolve/{CFG["text_revision"]}/{name}'
            with urllib.request.urlopen(url, timeout=120) as response:
                content = response.read()
            assert len(content) > 0
            atomic_bytes(dest, content)
        manifests[name] = sha_file(dest)
    pooling = json.loads((folder / '1_Pooling/config.json').read_text())
    assert pooling['pooling_mode_mean_tokens'] and not pooling['pooling_mode_cls_token']
    assert json.loads((folder / 'sentence_bert_config.json').read_text())['max_seq_length'] == CFG['text_max_tokens']
    return folder, manifests


def encode_texts(model, tokenizer, texts, device):
    import torch.nn.functional as F
    lengths = [len(x) for x in tokenizer(texts, add_special_tokens=True)['input_ids']]
    assert max(lengths) <= CFG['text_max_tokens'], 'Caption would be truncated'
    out = []
    with torch.no_grad():
        for begin in range(0, len(texts), CFG['text_batch_size']):
            inputs = tokenizer(texts[begin:begin+CFG['text_batch_size']], padding=True, truncation=False, return_tensors='pt')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            hidden = model(**inputs).last_hidden_state
            mask = inputs['attention_mask'].unsqueeze(-1).to(hidden.dtype)
            emb = F.normalize((hidden * mask).sum(1) / mask.sum(1).clamp_min(1), dim=1)
            out.append(emb.cpu())
    return torch.cat(out), max(lengths)


def text_diagnostics(embedding, texts, train):
    a = embedding[train].double().numpy()
    energy = float(np.square(a - a.mean(0)).sum(1).mean())
    similarity = a @ a.T
    off = similarity[np.triu_indices(len(a), 1)]
    quantiles = np.quantile(off, [0, .05, .5, .95, 1]).tolist()
    assert energy > CFG['text_collapse_max_centered_energy'] and quantiles[1] < CFG['text_collapse_min_pair_p05'], 'Text embeddings effectively collapsed'
    s = np.linalg.svd(a-a.mean(0), compute_uv=False)
    p = s / s.sum()
    return {'centered_energy': energy, 'cosine_quantiles_0_05_50_95_100': quantiles,
            'effective_rank': float(np.exp(-(p[p>0]*np.log(p[p>0])).sum())),
            'duplicate_description_fraction': 1-len(set(texts))/len(texts),
            'fraction_training_pairs_cosine_above_099': float((off>.99).mean())}
