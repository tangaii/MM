"""Train-only historical D4 preparation and stop-gradient PCFRA ranking loss."""
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


def prepare(labels, meta, train, seed, fold, cfg):
    """Historical D4 train-only private folds, site halves, and condition shuffle."""
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


class PCFRAObjective:
    def __init__(self, prepared, group, device):
        if group not in {'G1', 'G2', 'G3', 'G4'}:
            raise ValueError(group)
        self.group = group
        self.eps = 1e-12
        self.train = torch.as_tensor(prepared['train'], dtype=torch.long, device=device)
        self.y = torch.as_tensor(prepared['y'], dtype=torch.long, device=device)
        self.pf = torch.as_tensor(prepared['folds'], dtype=torch.long, device=device)
        self.asd = torch.where(self.y == 0)[0]
        self.td = torch.where(self.y == 1)[0]
        if not self.asd.numel() or not self.td.numel():
            raise ValueError('Outer train must contain both ASD and TD')

        perm = prepared['permutation'] if group == 'G4' else np.arange(len(prepared['train']))
        age = torch.as_tensor(prepared['age'][perm], dtype=torch.float32, device=device)
        sex = torch.as_tensor(prepared['sex'][perm], dtype=torch.long, device=device)
        site = prepared['site']
        site_equal = torch.as_tensor(np.asarray(site[:, None] == site[None, :]),
                                     dtype=torch.bool, device=device)
        bw = float(prepared['bandwidth'])
        ka = torch.exp(-0.5 * ((age[:, None] - age[None, :]) / bw).square())
        known = (sex[:, None] > 0) & (sex[None, :] > 0)
        ks = torch.where(known, torch.where(sex[:, None] == sex[None, :], 1.0, 0.5), 0.75)
        kernel = (ka * ks).detach()
        self.kernel = kernel
        self.condition = (kernel * site_equal.to(kernel.dtype)).index_select(0, self.asd).index_select(1, self.td)
        self.reference = (self.pf[:, None] != self.pf[None, :]).detach()
        if bool(self.reference.diagonal().any()):
            raise AssertionError('Prototype reference must exclude the query private fold')
        self.local_w = torch.stack(
            [kernel * self.reference.to(kernel.dtype) * (self.y == c)[None, :].to(kernel.dtype)
             for c in (0, 1)], dim=1).detach()
        self.global_w = torch.stack(
            [self.reference.to(kernel.dtype) * (self.y == c)[None, :].to(kernel.dtype)
             for c in (0, 1)], dim=1).detach()
        self.row_valid = self.condition.sum(dim=1) > 0
        self.col_valid = self.condition.sum(dim=0) > 0
        self.static_info = {
            'group': group,
            'n_asd': int(self.asd.numel()),
            'n_td': int(self.td.numel()),
            'n_all_pairs': int(self.asd.numel() * self.td.numel()),
            'n_conditional_pairs': int((self.condition > 0).sum().item()),
            'conditional_asd_anchor_fraction': float(self.row_valid.float().mean().item()),
            'conditional_td_anchor_fraction': float(self.col_valid.float().mean().item()),
        }
        self.prototype_calls = 0
        self.last_info = {}

    def _means(self, weights, features):
        den = weights.sum(dim=-1)
        mu = torch.matmul(weights, features) / den.clamp_min(self.eps)[..., None]
        ess = den.square() / weights.square().sum(dim=-1).clamp_min(self.eps)
        return mu, ess, den

    @torch.no_grad()
    def prototype_evidence(self, z_img):
        self.prototype_calls += 1
        query = z_img.index_select(0, self.train).detach()
        if query.ndim != 2 or query.shape[1] != 16 or not bool(torch.isfinite(query).all()):
            raise ValueError('Expected finite 871x16 imaging embeddings')
        mean = query.mean(dim=0)
        std = query.std(dim=0, unbiased=False).clamp_min(0.05)
        teacher = (query - mean) / std
        tau = (teacher - teacher.mean(0)).square().sum(dim=-1).mean().clamp_min(1.0)
        global_mu, _, global_den = self._means(self.global_w, teacher)
        local_mu, ess, local_den = self._means(self.local_w, teacher)
        valid = (global_den > 0).all(dim=-1)
        beta = ((ess.min(dim=-1).values - 8.0) / 32.0).clamp(0.0, 1.0)
        beta = torch.where((local_den > self.eps).all(dim=-1), beta, torch.zeros_like(beta))
        mu = beta[:, None, None] * local_mu + (1.0 - beta[:, None, None]) * global_mu
        q = -(teacher[:, None, :] - mu).square().sum(dim=-1) / tau
        evidence = q[:, 0] - q[:, 1]
        if not bool(valid.all()) or not bool(torch.isfinite(evidence).all()):
            raise FloatingPointError('Historical D4 prototype reference support is invalid')
        if evidence.requires_grad:
            raise AssertionError('Disease prototype teacher must be stop-gradient')
        self.last_info.update({
            'teacher_tau': float(tau),
            'teacher_beta_mean': float(beta.mean()),
            'teacher_beta_zero_fraction': float((beta == 0).float().mean()),
            'teacher_evidence_std': float(evidence.std(unbiased=False)),
        })
        return evidence

    @staticmethod
    def adaptive_margin(diff, supported):
        if not bool(supported.any()):
            return torch.ones_like(diff), diff.new_tensor(1.0)
        temperature = diff[supported].abs().median().clamp_min(1.0).detach()
        margin = (1.0 + 0.5 * torch.tanh(diff / temperature)).detach()
        return margin, temperature

    def _conditional_loss(self, pair_loss):
        w = self.condition
        row_den = w.sum(dim=1)
        col_den = w.sum(dim=0)
        row = (w * pair_loss).sum(dim=1) / row_den.clamp_min(self.eps)
        col = (w * pair_loss).sum(dim=0) / col_den.clamp_min(self.eps)
        terms = []
        if bool(self.row_valid.any()):
            terms.append(row[self.row_valid].mean())
        if bool(self.col_valid.any()):
            terms.append(col[self.col_valid].mean())
        row_loss = row[self.row_valid].mean() if bool(self.row_valid.any()) else pair_loss.sum() * 0.0
        col_loss = col[self.col_valid].mean() if bool(self.col_valid.any()) else pair_loss.sum() * 0.0
        return 0.5 * (row_loss + col_loss)

    def __call__(self, logits, z_img):
        if logits.ndim != 2 or logits.shape[1] != 2 or logits.shape[0] <= int(self.train.max()):
            raise ValueError('Expected all-subject logits [N,2] covering outer train')
        self.last_info = {'group': self.group, **self.static_info}
        train_logits = logits.index_select(0, self.train)
        score = train_logits[:, 0] - train_logits[:, 1]
        delta = score.index_select(0, self.asd)[:, None] - score.index_select(0, self.td)[None, :]
        global_loss = F.softplus(1.0 - delta).mean()
        if self.group == 'G1':
            self.last_info.update({'global_loss': float(global_loss.detach()), 'conditional_loss': None,
                                   'temperature': None, 'margin_mean': None, 'margin_std': None,
                                   'margin_abs_deviation': None, 'prototype_used': False})
            return global_loss

        if self.group == 'G2':
            margin = torch.ones_like(delta)
            temperature = delta.new_tensor(1.0)
        else:
            evidence = self.prototype_evidence(z_img)
            e_asd = evidence.index_select(0, self.asd)
            e_td = evidence.index_select(0, self.td)
            evidence_diff = (e_asd[:, None] - e_td[None, :]).detach()
            margin, temperature = self.adaptive_margin(evidence_diff, self.condition > 0)
        pair_loss = F.softplus(margin - delta)
        conditional = self._conditional_loss(pair_loss)
        auxiliary = 0.5 * global_loss + 0.5 * conditional
        if not bool(torch.isfinite(auxiliary)) or not auxiliary.requires_grad:
            raise FloatingPointError('Invalid PCFRA auxiliary objective')
        supported_margin = margin[self.condition > 0]
        self.last_info.update({
            'global_loss': float(global_loss.detach()),
            'conditional_loss': float(conditional.detach()),
            'temperature': float(temperature.detach()),
            'margin_mean': float(supported_margin.detach().mean()) if supported_margin.numel() else 1.0,
            'margin_std': float(supported_margin.detach().std(unbiased=False)) if supported_margin.numel() else 0.0,
            'margin_abs_deviation': float((supported_margin.detach() - 1.0).abs().mean())
            if supported_margin.numel() else 0.0,
            'prototype_used': self.group in {'G3', 'G4'},
        })
        return auxiliary
