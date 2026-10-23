"""The sole trainable PEIM interaction module and amplitude-safe update."""
from __future__ import annotations

import torch
from torch import nn


class LowRankInteraction(nn.Module):
    def __init__(self):
        super().__init__()
        self.W_d = nn.Parameter(torch.zeros(16, 2))

    def apply(self, z: torch.Tensor, conditions: torch.Tensor, partitions: list[dict],
              epoch: int, cfg: dict, selective: bool) -> tuple[torch.Tensor, dict]:
        rho = float(cfg["rho_cap"]) * min(max(epoch, 0) / float(cfg["rho_warmup_last_epoch"]), 1.0)
        out = z
        raw_norms, bounded_norms, final_norms, gates, cap_flags, rel_l2, margins = [], [], [], [], [], [], []
        for part in partitions:
            idx = torch.as_tensor(part["indices"], dtype=torch.long, device=z.device)
            if idx.numel() == 0:
                continue
            mean = torch.as_tensor(part["mean"], dtype=z.dtype, device=z.device)
            std = torch.as_tensor(part["std"], dtype=z.dtype, device=z.device)
            projector = torch.as_tensor(part["P"], dtype=z.dtype, device=z.device)
            zq = z.index_select(0, idx)
            zstd = (zq - mean) / std
            d = conditions.index_select(0, idx)
            condition_signal = d @ self.W_d.T
            q = zstd * condition_signal
            delta_std = q @ projector.T
            delta = rho * delta_std * std
            delta_norm = delta.norm(dim=1)
            limit = float(cfg["relative_modulation_cap"]) * zq.norm(dim=1)
            scale = torch.minimum(torch.ones_like(delta_norm),
                                  limit / (delta_norm + float(cfg["epsilon"])))
            bounded = delta * scale[:, None]
            if selective:
                prototypes = torch.as_tensor(part["prototypes"], dtype=z.dtype, device=z.device)
                d0 = (zstd - prototypes[0]).square().sum(1)
                d1 = (zstd - prototypes[1]).square().sum(1)
                margin = (d0 - d1).abs()
                tau = max(float(part["tau"]), float(cfg["epsilon"]))
                gate = 1.0 / (1.0 + margin / tau)
            else:
                margin = torch.zeros_like(delta_norm)
                gate = torch.ones_like(delta_norm)
            final_delta = bounded * gate[:, None]
            out = out.index_copy(0, idx, zq + final_delta)
            raw_norms.append(delta_norm.detach())
            bounded_norms.append(bounded.norm(dim=1).detach())
            final_norms.append(final_delta.norm(dim=1).detach())
            gates.append(gate.detach())
            cap_flags.append((scale < 1).detach())
            rel_l2.append(final_delta.norm(dim=1).detach() / zq.norm(dim=1).clamp_min(float(cfg["epsilon"])))
            margins.append(margin.detach())

        if raw_norms:
            raw = torch.cat(raw_norms)
            bounded = torch.cat(bounded_norms)
            final = torch.cat(final_norms)
            gate_values = torch.cat(gates)
            cap = torch.cat(cap_flags)
            relative = torch.cat(rel_l2)
            margin_values = torch.cat(margins)
        else:
            raw = z.new_zeros((0,))
            bounded = z.new_zeros((0,))
            final = z.new_zeros((0,))
            gate_values = z.new_ones((0,))
            cap = torch.zeros((0,), dtype=torch.bool, device=z.device)
            relative = z.new_zeros((0,))
            margin_values = z.new_zeros((0,))
        stats = {"rho": rho, "raw_l2": raw, "bounded_l2": bounded, "final_l2": final,
                 "gate": gate_values, "cap_hit": cap, "relative_l2": relative,
                 "margin": margin_values}
        return out, stats

