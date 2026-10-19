"""Fixed linear, nonlinear and untrained-neighbor diagnostic probes."""
from __future__ import annotations

import hashlib
import time
import warnings

import numpy as np
from scipy.special import softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, log_loss, recall_score, roc_auc_score)
from sklearn.svm import SVC

from data_protocol import CFG, array_sha, canonical_age, stable_seed, transform_phenotype

METRICS = CFG['metrics']


def metrics(labels, probability):
    labels, probability = np.asarray(labels), np.asarray(probability, np.float64)
    assert labels.shape == probability.shape and np.isfinite(probability).all()
    assert np.all((probability >= 0) & (probability <= 1))
    target = (labels == 0).astype(int)
    pred = probability >= .5
    both = len(np.unique(labels)) == 2
    return {'ACC': float(accuracy_score(target, pred)),
            'BA': float(balanced_accuracy_score(target, pred)) if both else None,
            'AUC': float(roc_auc_score(target, probability)) if both else None,
            'SEN': float(recall_score(target, pred, pos_label=1, zero_division=0)) if target.any() else None,
            'SPE': float(recall_score(target, pred, pos_label=0, zero_division=0)) if (~target.astype(bool)).any() else None,
            'F1': float(f1_score(target, pred, zero_division=0)) if target.any() else None,
            'CE': float(log_loss(target, probability, labels=[0, 1])),
            'Brier': float(brier_score_loss(target, probability))}


def per_patient_ce(labels, p):
    p = np.clip(p, CFG['probability_epsilon'], 1 - CFG['probability_epsilon'])
    return -np.where(np.asarray(labels) == 0, np.log(p), np.log1p(-p))


def rescue(labels, image, fusion):
    target = np.asarray(labels) == 0
    base_ok, fusion_ok = (image >= .5) == target, (fusion >= .5) == target
    rescued, harmed = ~base_ok & fusion_ok, base_ok & ~fusion_ok
    return {'rescued': int(rescued.sum()), 'harmed': int(harmed.sum()),
            'NetRescue': int(rescued.sum() - harmed.sum()),
            'both_correct': int((base_ok & fusion_ok).sum()), 'both_wrong': int((~base_ok & ~fusion_ok).sum()),
            'probability_MAD': float(np.mean(np.abs(fusion - image))),
            'CE_difference': float(np.mean(per_patient_ce(labels, fusion) - per_patient_ce(labels, image)))}


def group_matrix(zimg, ph, site, group):
    out = np.zeros((len(zimg), zimg.shape[1] + ph.shape[1] + site.shape[1]), np.float64)
    if group not in ('P1', 'P8'):
        out[:, :500] = zimg
    if group in ('P1', 'P2', 'P3', 'P7', 'K1', 'K2'):
        out[:, 500:500 + ph.shape[1]] = ph
    elif group == 'P4':
        out[:, 500:502] = ph[:, :2]
    elif group == 'P5':
        out[:, 502:500 + ph.shape[1]] = ph[:, 2:]
    if group in ('P6', 'P7', 'P8'):
        out[:, 500 + ph.shape[1]:] = site
    return out


def fit_probe(matrix_train, labels_train, matrix_all, ids_train, group, seed):
    """Only train labels enter estimator.fit or SVC's internal calibration."""
    is_svm = group.startswith('K')
    model = SVC(**{k: v for k, v in CFG['svm'].items() if k != 'random_state'}, random_state=seed) if is_svm else LogisticRegression(**CFG['logistic'])
    start = time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(matrix_train, labels_train)
    column = int(np.flatnonzero(model.classes_ == 0).item())
    p = model.predict_proba(matrix_all)[:, column]
    assert np.isfinite(p).all() and len(model.classes_) == 2
    info = {'algorithm': 'SVC' if is_svm else 'LogisticRegression', 'classes': model.classes_.tolist(),
            'ASD_probability_column': column, 'train_ids_sha256': array_sha(ids_train),
            'train_labels_sha256': array_sha(labels_train), 'train_design_sha256': array_sha(matrix_train),
            'train_shape': matrix_train.shape, 'seconds': time.monotonic() - start,
            'warnings': [str(w.message) for w in caught], 'n_iter': model.n_iter_.tolist(),
            'random_state': seed if is_svm else None, 'parameters': model.get_params()}
    if is_svm:
        assert model.fit_status_ == 0
        info.update({'fit_status': model.fit_status_, 'n_support': model.n_support_.tolist(),
                     'support_indices_sha256': array_sha(model.support_), 'support_vectors_sha256': array_sha(model.support_vectors_),
                     'dual_coef_sha256': array_sha(model.dual_coef_), 'probA': model.probA_.tolist(), 'probB': model.probB_.tolist(),
                     'calibration': 'libsvm internal train-only 5-fold probability calibration; fixed random_state'})
    else:
        assert int(np.max(model.n_iter_)) < CFG['logistic']['max_iter'], 'LR did not converge'
        info.update({'coef': model.coef_, 'intercept': model.intercept_})
    assert not info['warnings'], f'Fit warnings require investigation: {group} {info["warnings"]}'
    return p, info


def neighbor_inputs(zimg, train):
    normed = zimg / np.maximum(np.linalg.norm(zimg, axis=1, keepdims=True), 1e-12)
    cosine = normed @ normed[train].T
    cosine[train, np.arange(len(train))] = -np.inf
    k = min(CFG['neighbors']['k'], len(train) - 1)
    local = np.argsort(-cosine, axis=1, kind='stable')[:, :k]
    candidates = train[local]
    values = np.take_along_axis(cosine, local, axis=1)
    pimg = softmax(values / CFG['neighbors']['image_temperature'], axis=1)
    assert np.isfinite(pimg).all() and np.allclose(pimg.sum(1), 1., atol=1e-14)
    assert np.isin(candidates, train).all() and np.all(candidates[train] != train[:, None])
    return candidates, values, pimg


def phenotype_kernel(age_query, sex_query, age_anchor, sex_anchor, bandwidth):
    known = np.isin(sex_query[:, None], (1, 2)) & np.isin(sex_anchor, (1, 2))
    weight = np.where(known, np.where(sex_query[:, None] == sex_anchor, 1., .5), .75)
    return np.exp(-.5 * ((age_query[:, None] - age_anchor) / bandwidth) ** 2) * weight


def phenotype_vote(age_query, sex_query, age_anchor, sex_anchor, candidates, pimg, anchor_targets, bandwidth):
    kernel = phenotype_kernel(age_query, sex_query, age_anchor[candidates], sex_anchor[candidates], bandwidth)
    q = softmax(kernel / CFG['neighbors']['phenotype_temperature'], axis=1)
    weights = pimg * q
    weights /= weights.sum(1, keepdims=True)
    assert np.isfinite(weights).all() and np.allclose(weights.sum(1), 1., atol=1e-14)
    return np.sum(weights * anchor_targets, axis=1), {'kernel_sha256': array_sha(kernel),
            'weights_sha256': array_sha(weights), 'weight_entropy_mean': float(-(weights * np.log(weights)).sum(1).mean())}


def run_all(inputs, task, rfe, prep, zimg, ph, site, nulls):
    tr = task['split']['train']
    y_train, ids_train = inputs['y'][tr], inputs['ids'][tr]
    seed = stable_seed(task['id'], 'SVC-calibration')
    probabilities, fits = {}, {}
    groups = ['P0', 'P1', 'P2', 'P4', 'P5']
    if task['protocol'] == 'R':
        groups += ['P6', 'P7', 'P8']
    groups += ['K0', 'K1']
    for group in groups:
        matrix = group_matrix(zimg, ph, site, group)
        probabilities[group], fits[group] = fit_probe(matrix[tr], y_train, matrix, ids_train, group, seed)
    for b, mapping in enumerate(nulls):
        shuffled_ph, _ = transform_phenotype(inputs['age'][mapping], inputs['sex'][mapping], inputs['site'], prep['ph_state'])
        assert np.array_equal(site, transform_phenotype(inputs['age'], inputs['sex'], inputs['site'], prep['ph_state'])[1])
        for group in ('P3', 'K2'):
            matrix = group_matrix(zimg, shuffled_ph, site, group)
            assert np.array_equal(matrix[:, :500], zimg)
            key = f'{group}_{b}'
            probabilities[key], fits[key] = fit_probe(matrix[tr], y_train, matrix, ids_train, group, seed)
    start = time.monotonic()
    candidates, cosine, pimg = neighbor_inputs(zimg, tr)
    # Labels are gathered solely at verified training anchors, never at queries.
    targets = (inputs['y'][candidates] == 0).astype(float)
    age = canonical_age(inputs['age'])
    age = np.where(np.isfinite(age), age, prep['ph_state']['age_median'])
    sex = inputs['sex']
    width = prep['ph_state']['age_bandwidth']
    probabilities['N0'] = np.sum(pimg * targets, axis=1)
    probabilities['N1'], ntrue = phenotype_vote(age, sex, age, sex, candidates, pimg, targets, width)
    neighbor_stats = {'N1': ntrue}
    for b, mapping in enumerate(nulls):
        probabilities[f'N2a_{b}'], neighbor_stats[f'N2a_{b}'] = phenotype_vote(
            age, sex, age[mapping], sex[mapping], candidates, pimg, targets, width)
        probabilities[f'N2b_{b}'], neighbor_stats[f'N2b_{b}'] = phenotype_vote(
            age[mapping], sex[mapping], age[mapping], sex[mapping], candidates, pimg, targets, width)
    for p in probabilities.values():
        assert p.shape == (871,) and np.isfinite(p).all() and np.all((p >= 0) & (p <= 1))
    parts = {}
    for part, idx in task['split'].items():
        if len(idx):
            parts[part] = {key: metrics(inputs['y'][idx], p[idx]) for key, p in probabilities.items()}
    return {'probabilities': probabilities, 'fits': fits, 'metrics': parts,
            'neighbors': {'candidates': candidates, 'cosine_sha256': array_sha(cosine),
                          'pimg_sha256': array_sha(pimg), 'age_bandwidth': width,
                          'candidates_sha256': array_sha(candidates), 'train_anchor_ids_sha256': array_sha(inputs['ids'][tr]),
                          'stats': neighbor_stats, 'seconds': time.monotonic() - start},
            'image_train_mean_squared_norm': float(np.sum(zimg[tr] ** 2, axis=1).mean()),
            'image_preprocessing_sha256': array_sha(zimg), 'phenotype_preprocessing_sha256': array_sha(ph),
            'full_design_width': 500 + ph.shape[1] + site.shape[1],
            'fit_counts': {'RFE': 1, 'LogisticRegression': sum(not k.startswith('K') for k in fits),
                           'SVC': sum(k.startswith('K') for k in fits), 'GTUNet': 0, 'VAE': 0}}
