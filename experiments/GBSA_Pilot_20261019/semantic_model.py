"""Observational image hook and train-only symmetric semantic supervision."""
import hashlib
import math
import torch
from torch import nn
from torch.nn import functional as F


class ImageHook:
    def __init__(self, core):
        self.image = None
        self.calls = 0
        self.handle = core.img_unet.register_forward_hook(self.capture)

    def capture(self, module, inputs, output):
        self.image = output
        self.calls += 1

    def close(self):
        self.handle.remove()


class SemanticHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(16,128), nn.GELU(), nn.Linear(128,384))

    @classmethod
    def isolated(cls, seed, fold, device):
        cpu = torch.get_rng_state().clone()
        cuda = torch.cuda.get_rng_state(0).clone()
        private = int.from_bytes(hashlib.sha256(f'{seed}/{fold}/GBSA-head'.encode()).digest()[:4], 'little')
        try:
            torch.manual_seed(private)
            head = cls().to(device)
        finally:
            torch.set_rng_state(cpu)
            torch.cuda.set_rng_state(cuda,0)
        assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state(0))
        return head

    def objective(self, imaging, text, train, temperature):
        z = F.normalize(self.project(imaging[train]), dim=1)
        target = F.normalize(text[train], dim=1)
        scores = z @ target.T / temperature
        labels = torch.arange(len(train), device=imaging.device)
        loss = .5*(F.cross_entropy(scores,labels)+F.cross_entropy(scores.T,labels))
        normalized = loss / math.log(len(train))
        return normalized, {'raw_loss':loss.detach(), 'matched_cosine':(z*target).sum(1).mean().detach(),
                            'unmatched_cosine':((z.sum(0)*target.sum(0)).sum()-(z*target).sum())/ (len(train)*(len(train)-1))}


def effective_rank(value):
    import numpy as np
    a = value.detach().cpu().double().numpy()
    s = np.linalg.svd(a-a.mean(0),compute_uv=False)
    if s.sum()==0:
        return 0.
    p = s/s.sum()
    return float(np.exp(-(p[p>0]*np.log(p[p>0])).sum()))


def gradient_norm(loss, params):
    grads = torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
    assert all(g is None or torch.isfinite(g).all() for g in grads)
    return float(sum(float(g.detach().square().sum()) for g in grads if g is not None)**.5)
