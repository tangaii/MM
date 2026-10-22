"""Train-only, stop-gradient, cross-fitted conditional disease prototypes."""
from __future__ import annotations

import hashlib
import numpy as np
import torch
from torch.nn import functional as F


def private_seed(seed, fold, purpose):
    return int(hashlib.sha256(f'{seed}/{fold}/PCDCA/{purpose}'.encode()).hexdigest()[:15], 16)


def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def describe(a):
    if torch.is_tensor(a):
        a = a.detach().cpu().numpy()
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    if not len(a):
        return {'n': 0, 'mean': None, 'sd': None, 'cv': None, 'min': None, 'max': None,
                'p10': None, 'p50': None, 'p90': None}
    if not np.isfinite(a).all():
        raise FloatingPointError('Nonfinite diagnostic distribution')
    m, s = float(a.mean()), float(a.std())
    return {'n': len(a), 'mean': m, 'sd': s, 'cv': s / max(abs(m), 1e-12),
            'min': float(a.min()), 'max': float(a.max()),
            **dict(zip(('p10', 'p50', 'p90'), map(float, np.quantile(a, [.1, .5, .9]))))}


def prepare(labels, meta, train, seed, fold, cfg):
    """Never reads heldout labels, embeddings, or covariates for fitted objects."""
    tr = np.asarray(train, np.int64)
    y = np.asarray(labels, np.int64)[tr].copy()
    age = np.asarray(meta['age'], float)[tr].copy()
    sex = np.asarray(meta['sex'], float)[tr].copy()
    site = np.asarray(meta['site'], str)[tr].copy()
    bad_age = ~np.isfinite(age) | (age <= 0) | (age > cfg['age_valid_range'][1])
    median = float(np.median(age[~bad_age])) if (~bad_age).any() else 17.0
    age[bad_age] = median
    sex = np.where(np.isin(sex, [1, 2]), sex, -1).astype(np.int64)
    site = np.where(np.isin(site, ['', 'nan', 'None', '-1']), '__MISSING_SITE__', site)
    bandwidth = max(cfg['age_bandwidth_floor'], float(np.subtract(*np.percentile(age, [75, 25]))) / 2)
    n, nf = len(tr), cfg['prototype_folds']
    if n < nf or not np.isin(y, [0, 1]).all():
        raise ValueError('Invalid training subjects/classes')
    rng = np.random.default_rng(private_seed(seed, fold, 'prototype-folds'))
    pf = np.full(n, -1, np.int64)
    class_count, site_count = np.zeros((nf, 2), int), {s: np.zeros(nf, int) for s in np.unique(site)}
    for k in (0, 1):
        sites = sorted(np.unique(site[y == k]), key=lambda s: (-np.sum((site == s) & (y == k)), s))
        for s in sites:
            for i in rng.permutation(np.flatnonzero((site == s) & (y == k))):
                choices = np.flatnonzero(class_count[:, k] == class_count[:, k].min())
                choices = choices[site_count[s][choices] == site_count[s][choices].min()]
                g = int(rng.choice(choices))
                pf[i] = g
                class_count[g, k] += 1
                site_count[s][g] += 1
    p, singles = np.arange(n), []
    for s in sorted(set(site)):
        members = np.flatnonzero(site == s)
        srng = np.random.default_rng(private_seed(seed, fold, f'condition-shuffle/{s}'))
        order = srng.permutation(members)
        if len(order) > 1:
            p[order] = np.roll(order, 1)
            assert np.all(p[members] != members)
        else:
            singles.extend(members.tolist())
    assert np.array_equal(np.sort(p), np.arange(n)) and np.array_equal(site, site[p])
    halves = np.full((nf, n), -1, np.int64)
    half_records = []
    for g in range(nf):
        ref = pf != g
        hrng = np.random.default_rng(private_seed(seed, fold, f'site-halves/{g}'))
        order = list(np.unique(site[ref]))
        tie = {s: float(hrng.random()) for s in order}
        order.sort(key=lambda s: (-np.sum(ref & (site == s)), tie[s]))
        counts = np.zeros((2, 2), int)
        assignment = {}
        for s in order:
            mask = ref & (site == s)
            cnt = np.bincount(y[mask], minlength=2)
            scores = []
            for h in (0, 1):
                trial = counts.copy()
                trial[h] += cnt
                scores.append(float(np.square((trial[0] - trial[1]) / np.maximum(np.bincount(y[ref], minlength=2), 1)).sum()))
            best = np.flatnonzero(np.asarray(scores) == min(scores))
            h = int(hrng.choice(best))
            counts[h] += cnt
            halves[g, mask] = h
            assignment[str(s)] = h
        assert not (set(site[halves[g] == 0]) & set(site[halves[g] == 1]))
        half_records.append({'sites': assignment, 'class_counts': counts.tolist()})
    changed = (age != age[p]) | (sex != sex[p])
    info = {'train_indices': tr.tolist(), 'train_label_sha256': array_hash(y),
            'prototype_fold': pf.tolist(), 'prototype_fold_sha256': array_hash(pf),
            'prototype_class_counts': class_count.tolist(), 'age_bandwidth': bandwidth,
            'age_fill_train_median': median, 'age_missing_count': int(bad_age.sum()),
            'age_all_missing_fallback': bool(bad_age.all()), 'unknown_sex_count': int((sex == -1).sum()),
            'site_halves': half_records, 'site_halves_sha256': array_hash(halves),
            'shuffle_sources': tr[p].tolist(), 'shuffle_sha256': array_hash(p),
            'shuffle_moved_fraction': float(np.mean(p != np.arange(n))),
            'shuffle_changed_fraction': float(changed.mean()), 'singleton_subjects': tr[singles].tolist(),
            'mean_abs_age_change': float(np.abs(age - age[p]).mean()),
            'sex_changed_fraction': float(np.mean(sex != sex[p])),
            'per_site_changes': {s: {'n': int((site == s).sum()), 'changed_fraction': float(changed[site == s].mean())}
                                 for s in sorted(set(site))},
            'uses_heldout_labels_or_covariates': False}
    return {'train': tr, 'y': y, 'age': age, 'sex': sex, 'site': site, 'folds': pf,
            'halves': halves, 'permutation': p, 'bandwidth': bandwidth, 'info': info}


class PrototypeObjective:
    def __init__(self, prepared, group, cfg, device):
        self.data, self.spec, self.cfg = prepared, cfg['groups'][group], cfg
        self.group = group
        self.train = torch.as_tensor(prepared['train'], device=device)
        self.y = torch.as_tensor(prepared['y'], device=device)
        self.pf = torch.as_tensor(prepared['folds'], device=device)
        self.halves = torch.as_tensor(prepared['halves'], device=device)
        p = prepared['permutation'] if self.spec['shuffled'] else np.arange(len(prepared['train']))
        age = torch.as_tensor(prepared['age'][p], dtype=torch.float32, device=device)
        sex = torch.as_tensor(prepared['sex'][p], device=device)
        ka = torch.exp(-.5 * ((age[:, None] - age) / prepared['bandwidth']).square())
        known = (sex[:, None] > 0) & (sex > 0)
        ks = torch.where(known, torch.where(sex[:, None] == sex, 1., .5), .75)
        kind = self.spec['kernel']
        self.kernel = (torch.ones_like(ka) if kind == 'global' else ka if kind == 'age'
                       else ks if kind == 'sex' else ka * ks).detach()
        self.reference = (self.pf[:, None] != self.pf[None, :]).detach()
        assert not self.reference.diagonal().any()
        self.weights = torch.stack([self.kernel * self.reference * (self.y == k)[None] for k in (0, 1)], 1)
        self.global_weights = torch.stack([self.reference * (self.y == k)[None] for k in (0, 1)], 1).float()
        self.half_weights = []
        for h in (0, 1):
            half_mask = self.halves[self.pf] == h
            self.half_weights.append(self.weights * half_mask[:, None])

    def means(self, weights, teacher):
        eps = self.cfg['epsilon']
        den = weights.sum(-1)
        mu = torch.matmul(weights, teacher) / den.clamp_min(eps)[..., None]
        ess = den.square() / weights.square().sum(-1).clamp_min(eps)
        return mu, ess, den

    def __call__(self, z):
        cfg = self.cfg
        query = z[self.train]
        with torch.no_grad():
            teacher = query.detach()
            if not torch.isfinite(teacher).all():
                raise FloatingPointError('Nonfinite native image embedding')
            mean = teacher.mean(0)
            std = teacher.std(0, unbiased=False).clamp_min(cfg['embedding_std_floor'])
            t = (teacher - mean) / std
            tau = (t - t.mean(0)).square().sum(-1).mean().clamp_min(cfg['tau_floor'])
            global_mu, _, global_den = self.means(self.global_weights, t)
            local_mu, ess, local_den = self.means(self.weights, t)
            valid = (global_den > 0).all(-1)
            beta = ((ess.min(-1).values - cfg['shrink_ess_offset']) / cfg['shrink_ess_range']).clamp(0, 1)
            beta = torch.where((local_den > cfg['epsilon']).all(-1), beta, torch.zeros_like(beta))
            if self.spec['kernel'] == 'global':
                beta = torch.zeros_like(beta)
            mu = global_mu if self.spec['kernel'] == 'global' else beta[:, None, None] * local_mu + (1-beta[:, None, None]) * global_mu
            half_mu, half_ess, half_count = [], [], []
            for weights in self.half_weights:
                m, e, _ = self.means(weights, t)
                half_mu.append(m)
                half_ess.append(e)
                half_count.append((weights > 0).sum(-1))
            da, db = [m[:, 0] - m[:, 1] for m in half_mu]
            na, nb = da.norm(dim=-1), db.norm(dim=-1)
            support = ((torch.stack(half_count).amin((0, 2)) >= cfg['reliability_min_class_count']) &
                       (torch.stack(half_ess).amin((0, 2)) >= cfg['reliability_min_class_ess']) &
                       (na > cfg['reliability_direction_floor']) & (nb > cfg['reliability_direction_floor']))
            cos = (da * db).sum(-1) / (na * nb).clamp_min(cfg['epsilon'])
            r_site = ((1 + cos.clamp(-1, 1)) / 2).clamp(0, 1)
            r = (r_site * (ess.min(-1).values / cfg['reliability_support_scale']).clamp_max(1)).clamp(cfg['reliability_floor'], 1)
            r = torch.where(support, r, torch.full_like(r, cfg['reliability_floor']))
            if not self.spec['reliable']:
                r = torch.ones_like(r)
            r = r * valid
        u = (query - mean) / std
        logits = -(u[:, None] - mu).square().sum(-1) / tau
        ce = F.cross_entropy(logits, self.y, reduction='none')
        loss = (r * ce).sum() / r.sum().clamp_min(cfg['epsilon'])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite prototype loss')
        detail = {'mu': mu, 'global_mu': global_mu, 'teacher': t, 'mean': mean, 'std': std,
                  'tau': tau, 'ess': ess, 'beta': beta, 'r': r, 'r_site': r_site,
                  'support': support, 'valid': valid, 'logits': logits.detach(), 'ce': ce.detach(),
                  'local_den': local_den}
        return loss, detail

    def diagnostics(self, detail):
        delta = detail['mu'][:, 0] - detail['mu'][:, 1]
        result = {'tau': float(detail['tau']), 'ess': describe(detail['ess']),
                  'ess_min': describe(detail['ess'].min(-1).values), 'beta': describe(detail['beta']),
                  'r': describe(detail['r']), 'r_site': describe(detail['r_site']),
                  'r_fallback_fraction': float((~detail['support']).float().mean()),
                  'invalid_prototype_fraction': float((~detail['valid']).float().mean()),
                  'beta_zero_fraction': float((detail['beta'] == 0).float().mean()),
                  'conditional_global_l2': describe((detail['mu'] - detail['global_mu']).norm(dim=-1)),
                  'class_delta_l2': describe(delta.norm(dim=-1)),
                  'class_delta_query_sd_l2': float(delta.std(0, unbiased=False).norm()),
                  'prototype_acc': float((detail['logits'].argmax(-1) == self.y).float().mean())}
        for key in ('class', 'sex', 'site', 'age'):
            a = (self.data['y'] if key == 'class' else np.digitize(self.data['age'], [12,18,30])
                 if key == 'age' else self.data[key])
            result[f'r_by_{key}'] = {str(v): describe(detail['r'][torch.as_tensor(a == v, device=self.y.device)]) for v in np.unique(a)}
            if key in ('age','sex'):
                result[f'class_delta_by_{key}'] = {str(v): delta[torch.as_tensor(a == v, device=self.y.device)].mean(0).cpu().tolist() for v in np.unique(a)}
        return result


def gradient_scale(native, auxiliary, z, epoch, cfg):
    gn = torch.autograd.grad(native, z, retain_graph=True)[0]
    ga = torch.autograd.grad(auxiliary, z, retain_graph=True)[0]
    n, a = gn.detach().norm(), ga.detach().norm()
    weight = (cfg['auxiliary_gradient_ratio'] * n / (a + cfg['epsilon'])).clamp(0, cfg['lambda_cap']).detach()
    weight = weight * min(max(epoch, 0) / cfg['warmup_last_epoch'], 1)
    ratio = float((weight * ga.detach()).norm() / n.clamp_min(cfg['epsilon']))
    return weight, {'native_z_grad': float(n), 'aux_z_grad': float(a), 'lambda_eff': float(weight),
                    'weighted_z_gradient_ratio': ratio,
                    'z_gradient_cosine': float((gn.detach()*ga.detach()).sum() / (n*a).clamp_min(cfg['epsilon']))}
