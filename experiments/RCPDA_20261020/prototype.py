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
            # RCPDA does not use the historical D6 direction gate.  The
            # original D4 objective has unit weight on valid queries only;
            # the new gates are applied outside this reference objective.
            r = valid.to(dtype=t.dtype)
        u = (query - mean) / std
        logits = -(u[:, None] - mu).square().sum(-1) / tau
        ce = F.cross_entropy(logits, self.y, reduction='none')
        loss = (r * ce).sum() / r.sum().clamp_min(cfg['epsilon'])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite prototype loss')
        detail = {'mu': mu, 'global_mu': global_mu, 'teacher': t, 'mean': mean, 'std': std,
                  'tau': tau, 'ess': ess, 'beta': beta, 'r': r, 'r_site': r_site,
                  'support': support, 'valid': valid, 'logits': logits.detach(), 'ce': ce.detach(),
                  'ce_graph': ce, 'local_den': local_den}
        return loss, detail

    @torch.no_grad()
    def cross_site_gain_gate(self, teacher, tau, return_details=False):
        """Compare conditional and global prototypes across disjoint site halves."""
        t = teacher.detach()
        n = t.shape[0]
        eps = self.cfg['epsilon']
        min_ess = self.cfg['reliability_min_class_ess']
        out = t.new_zeros(n)
        gain_ab_all = t.new_zeros(n)
        gain_ba_all = t.new_zeros(n)
        valid_all = torch.zeros(n, dtype=torch.bool, device=t.device)

        def make_prototypes(q, ref):
            qn = q.numel()
            yr = self.y[ref]
            k = self.kernel[q[:, None], ref[None, :]].to(dtype=t.dtype)
            local_w = torch.stack([k * (yr[None, :] == c).to(t.dtype) for c in (0, 1)], dim=1)
            global_w = torch.stack([
                (yr[None, :] == c).to(t.dtype).expand(qn, -1) for c in (0, 1)
            ], dim=1)
            local_mu, ess, local_den = self.means(local_w, t[ref])
            global_mu, _, global_den = self.means(global_w, t[ref])
            beta = ((ess.min(dim=-1).values - self.cfg['shrink_ess_offset']) /
                    self.cfg['shrink_ess_range']).clamp(0.0, 1.0)
            conditional_mu = beta[:, None, None] * local_mu + (1.0 - beta[:, None, None]) * global_mu
            supported = (
                (local_den > eps).all(dim=-1)
                & (global_den > eps).all(dim=-1)
                & (ess >= min_ess).all(dim=-1)
            )
            return conditional_mu, global_mu, supported

        def weighted_risk(q, evaluate, prototypes):
            ye = self.y[evaluate]
            logits = -(
                t[evaluate][None, :, None, :] - prototypes[:, None, :, :]
            ).square().sum(dim=-1) / tau
            log_probs = torch.log_softmax(logits, dim=-1)
            target = ye[None, :, None].expand(q.numel(), -1, 1)
            losses = -log_probs.gather(dim=-1, index=target).squeeze(-1)
            kernel = self.kernel[q[:, None], evaluate[None, :]].to(dtype=t.dtype)
            risks = []
            validity = []
            for c in (0, 1):
                weights = kernel * (ye[None, :] == c).to(t.dtype)
                den = weights.sum(dim=-1)
                ess = den.square() / weights.square().sum(dim=-1).clamp_min(eps)
                risks.append((weights * losses).sum(dim=-1) / den.clamp_min(eps))
                validity.append((den > eps) & (ess >= min_ess))
            return 0.5 * (risks[0] + risks[1]), validity[0] & validity[1]

        for g in range(self.cfg['prototype_folds']):
            q = torch.nonzero(self.pf == g, as_tuple=False).flatten()
            a = torch.nonzero((self.pf != g) & (self.halves[g] == 0), as_tuple=False).flatten()
            b = torch.nonzero((self.pf != g) & (self.halves[g] == 1), as_tuple=False).flatten()
            if q.numel() == 0 or a.numel() == 0 or b.numel() == 0:
                continue

            cond_a, glob_a, ok_a = make_prototypes(q, a)
            cond_b, glob_b, ok_b = make_prototypes(q, b)
            cond_ab, valid_ab = weighted_risk(q, b, cond_a)
            glob_ab, valid_ab_global = weighted_risk(q, b, glob_a)
            cond_ba, valid_ba = weighted_risk(q, a, cond_b)
            glob_ba, valid_ba_global = weighted_risk(q, a, glob_b)
            gain_ab = glob_ab - cond_ab
            gain_ba = glob_ba - cond_ba
            positive_gain = torch.relu(torch.minimum(gain_ab, gain_ba))
            scale = torch.maximum(torch.maximum(gain_ab.abs(), gain_ba.abs()),
                                  torch.full_like(gain_ab, 1e-6))
            gate = (positive_gain / scale).clamp(0.0, 1.0)
            valid = (ok_a & ok_b & valid_ab & valid_ab_global & valid_ba & valid_ba_global)
            gate = torch.where(valid, gate, torch.zeros_like(gate))
            out[q] = gate
            gain_ab_all[q] = gain_ab
            gain_ba_all[q] = gain_ba
            valid_all[q] = valid

        if not torch.isfinite(out).all() or not bool(((out >= 0) & (out <= 1)).all()):
            raise FloatingPointError('Invalid cross-site reliability gate')
        details = {'gain_ab': gain_ab_all, 'gain_ba': gain_ba_all, 'valid': valid_all}
        return (out.detach(), details) if return_details else out.detach()

    def gated_loss(self, detail, reliability):
        gate = reliability.detach().to(dtype=detail['ce_graph'].dtype)
        valid = detail['valid'].to(dtype=gate.dtype)
        if gate.shape != valid.shape:
            raise ValueError('Reliability gate does not match the prototype query order')
        denominator = valid.sum().clamp_min(1.0)
        return (gate * valid * detail['ce_graph']).sum() / denominator

    def diagnostics(self, detail):
        delta = detail['mu'][:, 0] - detail['mu'][:, 1]
        result = {'tau': float(detail['tau']), 'ess': describe(detail['ess']),
                  'ess_min': describe(detail['ess'].min(-1).values), 'beta': describe(detail['beta']),
                  'invalid_prototype_fraction': float((~detail['valid']).float().mean()),
                  'beta_zero_fraction': float((detail['beta'] == 0).float().mean()),
                  'conditional_global_l2': describe((detail['mu'] - detail['global_mu']).norm(dim=-1)),
                  'class_delta_l2': describe(delta.norm(dim=-1)),
                  'class_delta_query_sd_l2': float(delta.std(0, unbiased=False).norm())}
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
