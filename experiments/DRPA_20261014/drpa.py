"""Training-only objectives; the official forward and fusion remain untouched."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class EmbeddingHooks:
    def __init__(self, core):
        self.img = None
        self.ph = None
        self.handles = [
            core.img_unet.register_forward_hook(self._img),
            core.ph_unet.register_forward_hook(self._ph),
        ]

    def _img(self, module, inputs, output):
        self.img = output

    def _ph(self, module, inputs, output):
        self.ph = output

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.img = self.ph = None


def finite(value, name):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"Non-finite {name}; stop the affected experiment")


def mlp(dims):
    return nn.Sequential(nn.Linear(dims[0], dims[1]), nn.GELU(),
                         nn.Linear(dims[1], dims[2]))


def disease_reliability(logits_i, logits_p, labels):
    with torch.no_grad():
        pi = logits_i.detach().softmax(dim=1)
        pp = logits_p.detach().softmax(dim=1)
        finite(pi, "image auxiliary probabilities")
        finite(pp, "phenotype auxiliary probabilities")
        mean = (pi + pp) * 0.5
        eps = 1e-8
        jsd = 0.5 * (
            (pi * (pi.clamp_min(eps).log() - mean.clamp_min(eps).log())).sum(1)
            + (pp * (pp.clamp_min(eps).log() - mean.clamp_min(eps).log())).sum(1)
        )
        support_i = pi.gather(1, labels[:, None]).squeeze(1)
        support_p = pp.gather(1, labels[:, None]).squeeze(1)
        return (torch.sqrt(support_i * support_p)
                * (1 - jsd / math.log(2)).clamp(0, 1)).clamp(0, 1)


def cross_site_reliability(si, sp, labels, sites, min_count=2):
    # The API receives only training rows. Both the query and prototypes detach.
    with torch.no_grad():
        si, sp = si.detach(), sp.detach()
        finite(si, "image shared representation")
        finite(sp, "phenotype shared representation")
        _, inverse = torch.unique(sites, sorted=True, return_inverse=True)
        n_sites = int(inverse.max().item()) + 1
        group = inverse * 2 + labels
        membership = F.one_hot(group, num_classes=n_sites * 2).to(si.dtype)
        counts = membership.sum(0).reshape(n_sites, 2)
        other_counts = counts.sum(0, keepdim=True) - counts
        valid_sites = (other_counts >= min_count).all(1)
        margins = []
        valid = valid_sites[inverse].clone()
        for shared in (si, sp):
            sums = (membership.T @ shared).reshape(n_sites, 2, -1)
            other = sums.sum(0, keepdim=True) - sums
            mu = other / other_counts.clamp_min(1).unsqueeze(-1)
            valid &= (mu.norm(dim=2) > 1e-8).all(1)[inverse]
            mu = F.normalize(mu, dim=2, eps=1e-8)
            positive = mu[inverse, labels]
            negative = mu[inverse, 1 - labels]
            margins.append((shared * positive).sum(1) - (shared * negative).sum(1))
        reliability = torch.sqrt(torch.sigmoid(2 * margins[0])
                                 * torch.sigmoid(2 * margins[1]))
        reliability = torch.where(valid, reliability, torch.zeros_like(reliability))
        return reliability, {
            "fallback": ~valid,
            "margin_img": margins[0],
            "margin_ph": margins[1],
            "other_class_counts": other_counts[inverse],
        }


def normalized_gate(q, threshold):
    with torch.no_grad():
        finite(q, "reliability gate")
        raw = (q.detach() - threshold).clamp_min(0)
        total = raw.sum()
        if total.item() == 0:
            return torch.zeros_like(raw), True
        return raw * (raw.numel() / total), False


def shuffled_gate(weights, labels, sites, generator):
    # Private CPU RNG: a negative control must not change official dropout RNG.
    labels_cpu = labels.detach().cpu()
    sites_cpu = sites.detach().cpu()
    permutation = torch.arange(len(weights))
    singletons = {0: [], 1: []}
    for cls in (0, 1):
        for site in torch.unique(sites_cpu):
            members = torch.where((labels_cpu == cls) & (sites_cpu == site))[0]
            if len(members) >= 2:
                permutation[members] = members[torch.randperm(len(members), generator=generator)]
            elif len(members) == 1:
                singletons[cls].append(int(members[0]))
    fallback_count = 0
    isolated_count = 0
    for members in singletons.values():
        if len(members) >= 2:
            members = torch.tensor(members, dtype=torch.long)
            permutation[members] = members[torch.randperm(len(members), generator=generator)]
            fallback_count += len(members)
        else:
            isolated_count += len(members)
    moved = permutation != torch.arange(len(weights))
    on_device = permutation.to(weights.device)
    assigned = weights[on_device].detach()
    return assigned, {
        "shuffle_moved_fraction": float(moved.float().mean()),
        "shuffle_changed_weight_fraction": float((assigned != weights).float().mean()),
        "shuffle_cross_site_singletons": fallback_count,
        "shuffle_isolated_singletons": isolated_count,
    }


def distribution(values):
    value = values.detach().float()
    if not value.numel():
        return {"n": 0}
    quantiles = torch.quantile(value, value.new_tensor([0.1, 0.5, 0.9])).cpu().tolist()
    return {
        "n": value.numel(), "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "p10": quantiles[0], "p50": quantiles[1], "p90": quantiles[2],
        "nonzero_fraction": float((value > 0).float().mean()),
    }


def gate_diagnostics(details, labels, sites):
    weights = details["weights"]
    result = {"weights": distribution(weights), "q_pred": distribution(details["q_pred"]),
              "q_site": distribution(details["q_site"]),
              "all_zero": details["all_zero"],
              "site_fallback_count": int(details["site_fallback"].sum()),
              "classes": {}, "sites": {}}
    for cls in (0, 1):
        mask = labels == cls
        result["classes"][str(cls)] = distribution(weights[mask])
    for site in torch.unique(sites):
        mask = sites == site
        result["sites"][str(int(site))] = {
            "weights": distribution(weights[mask]),
            "q_pred": distribution(details["q_pred"][mask]),
            "q_site": distribution(details["q_site"][mask]),
            "fallback_count": int(details["site_fallback"][mask].sum()),
        }
    for key in ("shuffle_moved_fraction", "shuffle_changed_weight_fraction",
                "shuffle_cross_site_singletons", "shuffle_isolated_singletons"):
        if key in details:
            result[key] = details[key]
    return result


class DRPA(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.project_img = mlp(config["projection"])
        self.project_ph = mlp(config["projection"])
        self.predict_img = mlp(config["predictor"])
        self.predict_ph = mlp(config["predictor"])
        self.head_img = nn.Linear(config["predictor"][-1], 2)
        self.head_ph = nn.Linear(config["predictor"][-1], 2)

    @classmethod
    def isolated(cls, config, seed, device):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            module = cls(config)
        return module.to(device)

    def forward(self, hi, hp):
        si = F.normalize(self.project_img(hi), dim=1, eps=1e-8)
        sp = F.normalize(self.project_ph(hp), dim=1, eps=1e-8)
        ti = self.predict_img(si)
        tp = self.predict_ph(sp)
        return {"si": si, "sp": sp, "ti": ti, "tp": tp,
                "logits_i": self.head_img(ti), "logits_p": self.head_ph(tp)}

    def objective(self, hi, hp, labels, sites, train_indices, method,
                  shuffle_generator=None, apply_shuffle=True):
        # Never pass held-out rows or labels to either the alignment or prototypes.
        y, site = labels[train_indices], sites[train_indices]
        value = self(hi[train_indices], hp[train_indices])
        for name, tensor in value.items():
            finite(tensor, name)
        aux = (F.cross_entropy(value["logits_i"], y)
               + F.cross_entropy(value["logits_p"], y)) * 0.5
        kind = self.config["methods"][method]["alignment"]
        si, sp, ti, tp = [value[k] for k in ("si", "sp", "ti", "tp")]
        q_pred = disease_reliability(value["logits_i"], value["logits_p"], y)
        q_site = torch.ones_like(q_pred)
        fallback = torch.zeros_like(q_pred, dtype=torch.bool)
        weights = torch.ones_like(q_pred)
        all_zero = False
        shuffle_info = {}
        if kind in ("site", "shuffled"):
            q_site, site_info = cross_site_reliability(
                si, sp, y, site, self.config["site_min_class_count"])
            fallback = site_info["fallback"]
        if kind in ("disease", "site", "shuffled"):
            weights, all_zero = normalized_gate(q_pred * q_site, self.config["gate_threshold"])
        if kind == "shuffled" and apply_shuffle:
            if shuffle_generator is None:
                raise ValueError("E7 requires an independent shuffle generator")
            weights, shuffle_info = shuffled_gate(weights, y, site, shuffle_generator)
        if kind == "none":
            align = None
        elif kind == "infonce":
            n = len(y)
            scores = (si @ sp.T) / self.config["temperature"]
            targets = torch.arange(n, device=si.device)
            align = ((F.cross_entropy(scores, targets) + F.cross_entropy(scores.T, targets))
                     * 0.5 / math.log(max(n, 2)))
        elif kind == "symmetric":
            align = ((1 - F.cosine_similarity(ti, sp.detach(), dim=1))
                     + (1 - F.cosine_similarity(tp, si.detach(), dim=1))).mean() * 0.5
        else:
            per_subject = 1 - F.cosine_similarity(tp, si.detach(), dim=1)
            align = (per_subject * weights).mean()
        details = {"weights": weights, "q_pred": q_pred, "q_site": q_site,
                   "site_fallback": fallback, "all_zero": all_zero, **shuffle_info}
        return aux, align, details, value
