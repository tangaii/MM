"""Parameter-free native-path interventions; label-blind private permutations."""
from __future__ import annotations

import hashlib
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse, dropout_edge
from utils.tools import cal_feature_sim

SUBSETS = ('train', 'val', 'test')


def private_seed(seed, fold, purpose):
    return int(hashlib.sha256(f'{seed}/{fold}/NIPA/{purpose}'.encode()).hexdigest()[:15], 16)


def array_hash(value):
    a = np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()


def describe(value):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    a = np.asarray(value, np.float64).reshape(-1)
    if not len(a):
        return {'n': 0, 'mean': None, 'sd': None, 'min': None, 'max': None, 'p10': None, 'p50': None, 'p90': None}
    if not np.isfinite(a).all():
        raise FloatingPointError('Nonfinite diagnostic')
    return {'n': len(a), 'mean': float(a.mean()), 'sd': float(a.std()), 'min': float(a.min()),
            'max': float(a.max()), **dict(zip(('p10','p50','p90'), map(float,np.quantile(a,[.1,.5,.9]))))}


def make_plan(meta, splits, seed, fold):
    """No diagnosis argument exists; all permutations are fixed before training."""
    n = len(meta['site'])
    identity = np.arange(n, dtype=np.int64)
    site, age, sex = (np.asarray(meta[k]) for k in ('site','age','sex'))
    assert np.array_equal(np.sort(np.concatenate([splits[k] for k in SUBSETS])), identity)
    permutations, records = {}, {}
    for factor in ('G','F','R'):
        p = identity.copy()
        partition_records = {}
        for partition in SUBSETS:
            idx = np.asarray(splits[partition], np.int64)
            rng = np.random.default_rng(private_seed(seed,fold,f'{factor}/{partition}'))
            singles = []
            if factor in ('G','F'):
                for s in sorted(np.unique(site[idx])):
                    members = idx[site[idx] == s]
                    order = rng.permutation(members)
                    if len(order) > 1:
                        p[order] = np.roll(order, 1)
                        assert np.all(p[members] != members)
                    else:
                        singles.extend(members.tolist())
                assert np.array_equal(site[idx], site[p[idx]])
            else:
                sites = rng.permutation(np.unique(site[idx]))
                blocks = [rng.permutation(idx[site[idx] == s]) for s in sites]
                order = np.concatenate(blocks)
                largest = max(map(len, blocks))
                p[order] = np.roll(order, largest)
                achievable = max(0, 2*largest-len(idx))
                assert int(np.sum(site[idx] == site[p[idx]])) == achievable
            assert np.array_equal(np.sort(p[idx]), np.sort(idx)), 'Partition boundary or bijection violation'
            changed = (age[idx] != age[p[idx]]) | (sex[idx] != sex[p[idx]])
            partition_records[partition] = {
                'n': len(idx), 'mapping': p[idx].tolist(), 'query_indices': idx.tolist(),
                'sha256': array_hash(p[idx]), 'moved_fraction': float(np.mean(p[idx] != idx)),
                'condition_changed_fraction': float(changed.mean()),
                'mean_absolute_age_change': float(np.mean(np.abs(age[idx]-age[p[idx]]))),
                'sex_changed_fraction': float(np.mean(sex[idx] != sex[p[idx]])),
                'site_changed_fraction': float(np.mean(site[idx] != site[p[idx]])),
                'singleton_subjects': singles, 'singleton_groups': len(singles),
                'private_seed': private_seed(seed,fold,f'{factor}/{partition}'),
                'site_counts_preserved': bool(np.array_equal(np.sort(site[idx]),np.sort(site[p[idx]]))),
                'per_site': {str(s): {'n': int(np.sum(site[idx] == s)),
                    'condition_changed_fraction': float(changed[site[idx] == s].mean()),
                    'site_changed_fraction': float(np.mean(site[p[idx]][site[idx] == s] != s))}
                    for s in sorted(np.unique(site[idx]))}}
        assert np.array_equal(np.sort(p), identity)
        permutations[factor] = p
        records[factor] = {'sha256': array_hash(p), 'mapping': p.tolist(), 'partitions': partition_records,
                           'uses_diagnosis_labels': False}
    assert not np.array_equal(permutations['G'], permutations['F'])
    return {'permutations': permutations, 'records': records}


def rp_metadata(original, permutation):
    result = {k: np.array(v,copy=True) for k,v in original.items()}
    result['SITE_ID'] = result['SITE_ID'][permutation].copy()
    for k in original:
        if k != 'SITE_ID':
            assert np.array_equal(result[k], original[k])
    return result


def raw_rp_diagnostic(real, null, splits):
    out = {}
    for i,a in enumerate(SUBSETS):
        for b in SUBSETS[i:]:
            block = np.ix_(np.asarray(splits[a]),np.asarray(splits[b]))
            channels = {}
            for c,name in enumerate(('reward','penalty','motivation')):
                x,y = real[0,c][block],null[0,c][block]
                channels[name] = {'changed_fraction': float(np.mean(x != y)),
                                  'absolute_difference_mean': float(np.mean(abs(x-y))),
                                  'real_nonzero': int(np.count_nonzero(x)), 'null_nonzero': int(np.count_nonzero(y))}
            out[f'{a}-{b}'] = channels
    assert np.array_equal(real[1:],np.zeros_like(real[1:]))
    assert np.array_equal(null[1:],np.zeros_like(null[1:]))
    assert not np.array_equal(real[0,:2],null[0,:2]), 'R-null failed to change train RP target'
    assert np.array_equal(real[:,2],null[:,2]), 'Native motivation must stay site-independent'
    out['zero_channels_sigmoid_background'] = '0.5*(rp_weights[SEX]+rp_weights[AGE]), not zero contribution'
    return out


def intervened_forward(core, img_features, ph_features, affinity_graphs, perm_G, perm_F, spec):
    """Mirror native order; real attention weights always govern graph loss."""
    img_features = img_features.to(core.device)
    ph_features = ph_features.to(core.device)
    affinity_graphs = affinity_graphs.to(core.device)
    if core.load_pretrain:
        core.load_vae()
    ph_features, _ = core.vae(ph_features)
    core.rp_graph, core.value = core.create_rp_graph(affinity_graphs)
    graph_ph = ph_features[perm_G] if spec['G'] == 'null' else ph_features
    fused_embed = torch.cat((img_features, graph_ph), dim=1)
    core.fused_sim = cal_feature_sim(fused_embed)
    core.fused_graph = core.fused_sim * core.rp_graph
    fused_index, fused_attr = dense_to_sparse(core.fused_graph)
    fused_attr = fused_attr.view(-1,1)
    if core.training and core.edge_drop > 0:
        fused_index, fused_mask = dropout_edge(fused_index)
        fused_attr = fused_attr[fused_mask]
    img_embed = core.img_unet(img_features, fused_index, fused_attr)
    ph_embed = core.ph_unet(ph_features, fused_index, fused_attr)
    core.joint_embed, core.img_weight, core.ph_weight = core.mm_attention(img_embed,ph_embed)
    if spec['F'] == 'null':
        # Do not assign the second call's weights to core: they must not change graph loss.
        core.joint_embed, _, _ = core.mm_attention(img_embed,ph_embed[perm_F])
    graph_loss = core.cal_graph_loss(img_embed,ph_embed)
    outputs = core.clf(core.joint_embed)
    return outputs, graph_loss
