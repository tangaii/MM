import torch
from torch import nn

class PCNDEvidence(nn.Module):
    def __init__(self, dim=32, hidden=32, logit_cap=0.25):
        super().__init__()
        self.dim = dim
        self.logit_cap = float(logit_cap)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, base_logits, deviation, reliability):
        assert base_logits.ndim == 2 and base_logits.size(-1) == 2
        assert deviation.shape == (base_logits.size(0), self.dim)
        assert reliability.shape == (base_logits.size(0),)
        assert bool(torch.isfinite(deviation).all())
        assert bool(torch.isfinite(reliability).all())
        assert bool(torch.all((reliability >= .25) & (reliability <= 1)))
        signed = self.net(deviation).squeeze(-1)
        delta = self.logit_cap * torch.tanh(signed) * reliability
        corrected_logits = base_logits + torch.stack((delta, -delta), dim=-1)
        return corrected_logits, delta

