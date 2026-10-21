"""Train-only hard-pair residual reweighting with historical D4 prototypes."""
from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch.nn import functional as F


def private_seed(seed, fold, purpose):
    return int(hashlib.sha256(f'{seed}/{fold}/PCDCA/{purpose}'.encode()).hexdigest()[:15], 16)


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes()).hexdigest()


def prepare(labels, meta, train, seed, fold, cfg):
    """Exact PCDCA preparation: all fitted quantities use outer-train only."""
    tr = np.asarray(train, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64)[tr].copy()
    age = np.asarray(meta['age'], dtype=float)[tr].copy()
    sex = np.asarray(meta['sex'], dtype=float)[tr].copy()
    site = np.asarray(meta['site'], dtype=str)[tr].copy()
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
    private_folds = np.full(n, -1, np.int64)
    class_count = np.zeros((nf, 2), int)
    site_count = {s: np.zeros(nf, int) for s in np.unique(site)}
    for cls in (0, 1):
        sites = sorted(np.unique(site[y == cls]), key=lambda s: (-np.sum((site == s) & (y == cls)), s))
        for site_id in sites:
            for i in rng.permutation(np.flatnonzero((site == site_id) & (y == cls))):
                choices = np.flatnonzero(class_count[:, cls] == class_count[:, cls].min())
                choices = choices[site_count[site_id][choices] == site_count[site_id][choices].min()]
                group = int(rng.choice(choices))
                private_folds[i] = group
                class_count[group, cls] += 1
                site_count[site_id][group] += 1

    permutation = np.arange(n)
    singleton = []
    for site_id in sorted(set(site)):
        members = np.flatnonzero(site == site_id)
        srng = np.random.default_rng(private_seed(seed, fold, f'condition-shuffle/{site_id}'))
        order = srng.permutation(members)
        if len(order) > 1:
            permutation[order] = np.roll(order, 1)
            assert np.all(permutation[members] != members)
        else:
            singleton.extend(members.tolist())
    assert np.array_equal(np.sort(permutation), np.arange(n))
    assert np.array_equal(site, site[permutation])

    halves = np.full((nf, n), -1, np.int64)
    half_records = []
    for private_group in range(nf):
        reference = private_folds != private_group
        hrng = np.random.default_rng(private_seed(seed, fold, f'site-halves/{private_group}'))
        site_order = list(np.unique(site[reference]))
        tie = {site_id: float(hrng.random()) for site_id in site_order}
        site_order.sort(key=lambda site_id: (-np.sum(reference & (site == site_id)), tie[site_id]))
        counts = np.zeros((2, 2), int)
        assignment = {}
        for site_id in site_order:
            mask = reference & (site == site_id)
            site_counts = np.bincount(y[mask], minlength=2)
            scores = []
            class_totals = np.maximum(np.bincount(y[reference], minlength=2), 1)
            for half in (0, 1):
                trial = counts.copy()
                trial[half] += site_counts
                scores.append(float(np.square((trial[0] - trial[1]) / class_totals).sum()))
            best = np.flatnonzero(np.asarray(scores) == min(scores))
            half = int(hrng.choice(best))
            counts[half] += site_counts
            halves[private_group, mask] = half
            assignment[str(site_id)] = half
        assert not (set(site[halves[private_group] == 0]) & set(site[halves[private_group] == 1]))
        half_records.append({'sites': assignment, 'class_counts': counts.tolist()})

    changed = (age != age[permutation]) | (sex != sex[permutation])
    info = {
        'train_indices': tr.tolist(),
        'train_label_sha256': array_hash(y),
        'prototype_fold': private_folds.tolist(),
        'prototype_fold_sha256': array_hash(private_folds),
        'prototype_class_counts': class_count.tolist(),
        'site_halves': half_records,
        'site_halves_sha256': array_hash(halves),
        'age_bandwidth': bandwidth,
        'age_fill_train_median': median,
        'age_missing_count': int(bad_age.sum()),
        'age_all_missing_fallback': bool(bad_age.all()),
        'unknown_sex_count': int((sex == -1).sum()),
        'shuffle_sources': tr[permutation].tolist(),
        'shuffle_sha256': array_hash(permutation),
        'shuffle_moved_fraction': float(np.mean(permutation != np.arange(n))),
        'shuffle_changed_fraction': float(changed.mean()),
        'singleton_subjects': tr[singleton].tolist(),
        'mean_abs_age_change': float(np.abs(age - age[permutation]).mean()),
        'sex_changed_fraction': float(np.mean(sex != sex[permutation])),
        'per_site_changes': {
            site_id: {'n': int((site == site_id).sum()),
                      'changed_fraction': float(changed[site == site_id].mean())}
            for site_id in sorted(set(site))
        },
        'uses_heldout_labels_or_covariates': False,
    }
    return {'train': tr, 'y': y, 'age': age, 'sex': sex, 'site': site,
            'folds': private_folds, 'halves': halves, 'permutation': permutation, 'bandwidth': bandwidth,
            'info': info}


def _tensor_hash(tensor):
    return array_hash(tensor.detach().cpu().numpy())


def _describe(tensor):
    x = tensor.detach().to(dtype=torch.float64).cpu().numpy().reshape(-1)
    if not len(x):
        return {'n': 0, 'mean': None, 'sd': None, 'min': None, 'p10': None, 'p50': None, 'p90': None, 'max': None}
    if not np.isfinite(x).all():
        raise FloatingPointError('Nonfinite mechanism diagnostic')
    return {'n': len(x), 'mean': float(x.mean()), 'sd': float(x.std()), 'min': float(x.min()),
            'p10': float(np.quantile(x, .1)), 'p50': float(np.quantile(x, .5)),
            'p90': float(np.quantile(x, .9)), 'max': float(x.max())}


def _correlation(left, right):
    left = left.detach().float().reshape(-1)
    right = right.detach().float().reshape(-1)
    if left.numel() < 2 or float(left.std(unbiased=False)) == 0 or float(right.std(unbiased=False)) == 0:
        return None
    value = torch.corrcoef(torch.stack([left, right]))[0, 1]
    return float(value) if bool(torch.isfinite(value)) else None


class PCHRRAObjective:
    def __init__(self, prepared, group, device, cfg):
        if group not in {'H1', 'H2', 'H3', 'H4'}:
            raise ValueError(group)
        self.group = group
        self.cfg = cfg
        self.eps = float(cfg['epsilon'])
        self.train = torch.as_tensor(prepared['train'], dtype=torch.long, device=device)
        self.y = torch.as_tensor(prepared['y'], dtype=torch.long, device=device)
        self.private_folds = torch.as_tensor(prepared['folds'], dtype=torch.long, device=device)
        self.asd = torch.where(self.y == 0)[0]
        self.td = torch.where(self.y == 1)[0]
        if not self.asd.numel() or not self.td.numel():
            raise ValueError('Outer train must contain both ASD and TD')
        if not torch.all((self.y == 0) | (self.y == 1)):
            raise ValueError('Labels must use ASD=0 and TD=1')

        self.site = np.asarray(prepared['site'], dtype=str)
        asd_sites = self.site[self.asd.cpu().numpy()]
        td_sites = self.site[self.td.cpu().numpy()]
        self.site_pair_masks = {
            site_id: torch.as_tensor((asd_sites[:, None] == site_id) & (td_sites[None, :] == site_id),
                                     dtype=torch.bool, device=device)
            for site_id in sorted(set(self.site))
        }
        self.age = torch.as_tensor(prepared['age'], dtype=torch.float32, device=device)
        self.sex = torch.as_tensor(prepared['sex'], dtype=torch.long, device=device)
        self.site_equal = torch.as_tensor(self.site[:, None] == self.site[None, :],
                                          dtype=torch.bool, device=device)
        site_kernel = self.site_equal.to(torch.float32)
        self.site_condition = site_kernel.index_select(0, self.asd).index_select(1, self.td).detach()
        self.true_kernel = None
        self.true_condition = self.site_condition if group == 'H1' else None
        self.teacher_kernel = self.reference = self.global_weights = self.local_weights = None
        if group != 'H1':
            self.true_kernel = self._phenotype_kernel(self.age, self.sex, prepared['bandwidth'])
            self.true_condition = (site_kernel * self.true_kernel).index_select(0, self.asd).index_select(1, self.td).detach()
        if group in {'H3', 'H4'}:
            permutation = np.asarray(prepared['permutation'], dtype=np.int64)
            if group == 'H4':
                teacher_age = self.age.index_select(0, torch.as_tensor(permutation, device=device))
                teacher_sex = self.sex.index_select(0, torch.as_tensor(permutation, device=device))
            else:
                teacher_age, teacher_sex = self.age, self.sex
            self.teacher_kernel = self._phenotype_kernel(teacher_age, teacher_sex, prepared['bandwidth'])
            self.reference = (self.private_folds[:, None] != self.private_folds[None, :]).detach()
            if bool(self.reference.diagonal().any()):
                raise AssertionError('Private-fold reference must exclude the query fold')
            labels = self.y
            self.global_weights = torch.stack([
                self.reference.to(torch.float32) * (labels == cls)[None, :].to(torch.float32)
                for cls in (0, 1)
            ], dim=1).detach()
            self.local_weights = torch.stack([
                self.reference.to(torch.float32) * self.teacher_kernel * (labels == cls)[None, :].to(torch.float32)
                for cls in (0, 1)
            ], dim=1).detach()

        group_condition = self.site_condition if group == 'H1' else self.true_condition
        row_valid = group_condition.sum(1) > 0
        col_valid = group_condition.sum(0) > 0
        self.static_info = {
            'group': group,
            'n_train': int(self.y.numel()),
            'n_asd': int(self.asd.numel()),
            'n_td': int(self.td.numel()),
            'n_all_pairs': int(self.asd.numel() * self.td.numel()),
            'n_condition_pairs': int((group_condition > 0).sum().item()),
            'asd_anchor_fraction': float(row_valid.float().mean().item()),
            'td_anchor_fraction': float(col_valid.float().mean().item()),
            'true_condition_sha256': _tensor_hash(self.true_condition) if group != 'H1' else None,
            'site_condition_sha256': _tensor_hash(self.site_condition),
            'teacher_kernel_sha256': _tensor_hash(self.teacher_kernel) if self.teacher_kernel is not None else None,
            'uses_prototype': group in {'H3', 'H4'},
            'uses_true_condition_for_hard_pairs': group != 'H1',
        }
        self.prototype_calls = 0
        self.last_info = {}
        self.last_weights = None

    @staticmethod
    def _phenotype_kernel(age, sex, bandwidth):
        ka = torch.exp(-0.5 * ((age[:, None] - age[None, :]) / float(bandwidth)).square())
        known = (sex[:, None] > 0) & (sex[None, :] > 0)
        ks = torch.where(known, torch.where(sex[:, None] == sex[None, :], 1.0, 0.5), 0.75)
        return (ka * ks).detach()

    def build_condition(self):
        """Return detached site-only and true site-age-sex ASD-by-TD weights."""
        return self.site_condition, self.true_condition

    def global_ranking_loss(self, delta):
        if delta.ndim != 2 or tuple(delta.shape) != (self.asd.numel(), self.td.numel()):
            raise ValueError('delta must have shape [n_ASD, n_TD]')
        return F.softplus(float(self.cfg['margin']) - delta).mean()

    def hard_pair_weights(self, delta, alpha=None):
        """Compute detached bounded difficulty weighting for H2-H4."""
        if tuple(delta.shape) != (self.asd.numel(), self.td.numel()):
            raise ValueError('delta shape does not match train ASD/TD pairs')
        alpha = float(self.cfg['hard_alpha'] if alpha is None else alpha)
        difficulty = torch.sigmoid((float(self.cfg['margin']) - delta.detach()) / 1.0).to(
            dtype=self.true_condition.dtype)
        base = self.true_condition * (1.0 + alpha * difficulty)
        return base.detach(), difficulty.detach()

    @torch.no_grad()
    def prototype_evidence(self, z_img):
        self.prototype_calls += 1
        if z_img.ndim != 2 or z_img.shape[1] != 16 or z_img.shape[0] <= int(self.train.max()):
            raise ValueError('Expected all-subject Imaging GTUNet output [N,16]')
        query = z_img.index_select(0, self.train).detach()
        if not bool(torch.isfinite(query).all()):
            raise FloatingPointError('Nonfinite Imaging GTUNet teacher')
        mean = query.mean(0)
        std = query.std(0, unbiased=False).clamp_min(float(self.cfg['embedding_std_floor']))
        teacher = (query - mean) / std
        tau = (teacher - teacher.mean(0)).square().sum(-1).mean().clamp_min(float(self.cfg['tau_floor']))

        def prototype(weights):
            den = weights.sum(-1)
            mu = torch.matmul(weights, teacher) / den.clamp_min(self.eps)[..., None]
            ess = den.square() / weights.square().sum(-1).clamp_min(self.eps)
            return mu, ess, den

        global_mu, global_ess, global_den = prototype(self.global_weights)
        local_mu, local_ess, local_den = prototype(self.local_weights)
        beta = ((local_ess.min(-1).values - float(self.cfg['shrink_ess_offset'])) /
                float(self.cfg['shrink_ess_range'])).clamp(0.0, 1.0)
        local_supported = (local_den > self.eps).all(-1)
        beta = torch.where(local_supported, beta, torch.zeros_like(beta)).detach()
        cond_mu = beta[:, None, None] * local_mu + (1.0 - beta[:, None, None]) * global_mu
        valid = (global_den > 0).all(-1)
        if not bool(valid.all()):
            raise FloatingPointError('A global prototype class has no cross-private-fold support')
        e_global = (-(teacher[:, None, :] - global_mu).square().sum(-1) / tau)
        e_cond = (-(teacher[:, None, :] - cond_mu).square().sum(-1) / tau)
        e_global = e_global[:, 0] - e_global[:, 1]
        e_cond = e_cond[:, 0] - e_cond[:, 1]
        if not bool(torch.isfinite(e_global).all() & torch.isfinite(e_cond).all()):
            raise FloatingPointError('Nonfinite prototype evidence')
        self._prototype_cache = {
            'teacher': teacher.detach(), 'mean': mean.detach(), 'std': std.detach(), 'tau': tau.detach(),
            'global_mu': global_mu.detach(), 'cond_mu': cond_mu.detach(), 'beta': beta.detach(),
            'local_ess': local_ess.detach(), 'global_ess': global_ess.detach(),
        }
        return e_cond.detach(), e_global.detach()

    def residual_evidence(self, e_cond, e_global):
        if e_cond.shape != e_global.shape or e_cond.ndim != 1 or e_cond.shape[0] != self.y.numel():
            raise ValueError('prototype evidence must have shape [n_train]')
        residual = (e_cond - e_global).detach()
        r_asd = residual.index_select(0, self.asd)
        r_td = residual.index_select(0, self.td)
        ec_asd = e_cond.index_select(0, self.asd)
        ec_td = e_cond.index_select(0, self.td)
        R = r_asd[:, None] - r_td[None, :]
        E = ec_asd[:, None] - ec_td[None, :]
        return R.detach(), E.detach()

    def residual_reweight(self, base, R, E, beta_residual=None):
        if base.shape != R.shape or base.shape != E.shape:
            raise ValueError('B, R, and E must have the same ASD-by-TD shape')
        supported = self.true_condition > 0
        if not bool(supported.any()):
            raise ValueError('Prototype residual temperature has no true-condition support')
        floor = float(self.cfg['temperature_floor'])
        t_res = R[supported].abs().median().clamp_min(floor).detach()
        t_cond = E[supported].abs().median().clamp_min(floor).detach()
        confidence = (torch.sigmoid(R / t_res) * torch.sigmoid(E / t_cond)).detach()
        b = base.detach()
        c_bar = ((b * confidence).sum() / b.sum().clamp_min(self.eps)).detach()
        beta = float(self.cfg['residual_beta'] if beta_residual is None else beta_residual)
        q = (1.0 + beta * (confidence - c_bar)).detach()
        if not bool(torch.isfinite(q).all()):
            raise FloatingPointError('Nonfinite residual reweight')
        if bool((q < 0.5 - 1e-6).any()) or bool((q > 1.5 + 1e-6).any()):
            raise AssertionError('Centered residual weight must remain within [0.5,1.5]')
        weight = (b * q).detach()
        conservation = float((weight.sum() - b.sum()).abs() / b.sum().clamp_min(self.eps))
        c_values = confidence[supported].float()
        q_values = q[supported].float()
        correlation = None
        if (c_values.numel() > 1 and float(c_values.std(unbiased=False)) > 0
                and float(q_values.std(unbiased=False)) > 0):
            correlation = float(torch.corrcoef(torch.stack([c_values, q_values]))[0, 1])
        info = {
            'T_residual': float(t_res), 'T_condition': float(t_cond),
            'c_bar': float(c_bar), 'residual_beta': beta,
            'residual_R': _describe(R[supported]), 'condition_E': _describe(E[supported]),
            'confidence_c': _describe(confidence[supported]), 'reweight_q': _describe(q[supported]),
            'base_weight': _describe(b[supported]), 'final_weight': _describe(weight[supported]),
            'q_abs_deviation_mean': float((q[supported] - 1.0).abs().mean()),
            'weight_conservation_relative_error': conservation,
            'confidence_q_correlation': correlation,
        }
        return weight, confidence, q, info

    def conditional_ranking_loss(self, pair_loss, weight):
        if pair_loss.shape != weight.shape or tuple(pair_loss.shape) != (self.asd.numel(), self.td.numel()):
            raise ValueError('pair_loss and weight must match ASD-by-TD support')
        w = weight.detach()
        row_valid = w.sum(1) > 0
        col_valid = w.sum(0) > 0
        row_den = w.sum(1)
        col_den = w.sum(0)
        row = (w * pair_loss).sum(1) / row_den.clamp_min(self.eps)
        col = (w * pair_loss).sum(0) / col_den.clamp_min(self.eps)
        row_loss = row[row_valid].mean() if bool(row_valid.any()) else pair_loss.sum() * 0.0
        col_loss = col[col_valid].mean() if bool(col_valid.any()) else pair_loss.sum() * 0.0
        return 0.5 * (row_loss + col_loss)

    def forward(self, logits, z_img):
        if logits.ndim != 2 or logits.shape[1] != 2 or logits.shape[0] <= int(self.train.max()):
            raise ValueError('Expected all-subject logits [N,2] covering outer train')
        if not logits.is_floating_point() or logits.device != self.train.device:
            raise ValueError('logits must be floating point on the objective device')
        train_logits = logits.index_select(0, self.train)
        score = train_logits[:, 0] - train_logits[:, 1]
        delta = score.index_select(0, self.asd)[:, None] - score.index_select(0, self.td)[None, :]
        pair_loss = F.softplus(float(self.cfg['margin']) - delta)
        global_loss = self.global_ranking_loss(delta)
        details = {'group': self.group, **self.static_info,
                   'global_loss': float(global_loss.detach()),
                   'difficulty': None, 'hard_weight': None, 'prototype_used': False}

        if self.group == 'H1':
            weight = self.site_condition
        else:
            base, difficulty = self.hard_pair_weights(delta)
            weight = base
            details.update({'difficulty': _describe(difficulty[self.true_condition > 0]),
                            'hard_multiplier': _describe((1.0 + float(self.cfg['hard_alpha']) * difficulty)[
                                self.true_condition > 0]),
                            'hard_weight': True})
            if self.group in {'H3', 'H4'}:
                e_cond, e_global = self.prototype_evidence(z_img)
                R, E = self.residual_evidence(e_cond, e_global)
                weight, confidence, q, residual_info = self.residual_reweight(base, R, E)
                details.update(residual_info)
                details.update({
                    'prototype_used': True,
                    'prototype_calls': self.prototype_calls,
                    'prototype_e_cond': _describe(e_cond),
                    'prototype_e_global': _describe(e_global),
                    'prototype_e_increment': _describe(e_cond - e_global),
                    'prototype_beta': _describe(self._prototype_cache['beta']),
                    'prototype_tau': float(self._prototype_cache['tau']),
                    'prototype_global_mu_sha256': _tensor_hash(self._prototype_cache['global_mu']),
                    'prototype_cond_mu_sha256': _tensor_hash(self._prototype_cache['cond_mu']),
                    'confidence_sha256': _tensor_hash(confidence),
                    'q_sha256': _tensor_hash(q),
                    'difficulty_q_correlation': _correlation(
                        difficulty[self.true_condition > 0], q[self.true_condition > 0]),
                    'difficulty_c_correlation': _correlation(
                        difficulty[self.true_condition > 0], confidence[self.true_condition > 0]),
                })

        cond_loss = self.conditional_ranking_loss(pair_loss, weight)
        auxiliary = 0.5 * global_loss + 0.5 * cond_loss
        if not bool(torch.isfinite(auxiliary)) or not auxiliary.requires_grad:
            raise FloatingPointError('Invalid PC-HRRA auxiliary objective')
        self.last_weights = weight.detach()
        site_mass = {site_id: float(weight[mask].sum()) for site_id, mask in self.site_pair_masks.items()}
        site_total = sum(site_mass.values())
        details.update({'conditional_loss': float(cond_loss.detach()),
                        'auxiliary_loss': float(auxiliary.detach()),
                        'final_weight_sha256': _tensor_hash(weight),
                        'asd_anchor_weight': _describe(weight.sum(1)),
                        'td_anchor_weight': _describe(weight.sum(0)),
                        'site_weight_mass': site_mass,
                        'site_weight_share': {site_id: amount / max(site_total, self.eps)
                                              for site_id, amount in site_mass.items()},
                        'condition_gradient_norm': float(torch.autograd.grad(
                            auxiliary, logits, retain_graph=True)[0].norm())})
        self.last_info = details
        return auxiliary
