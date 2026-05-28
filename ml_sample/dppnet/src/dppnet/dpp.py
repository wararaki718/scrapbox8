from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


def make_rbf_kernel(features: torch.Tensor, beta: float = 0.5) -> torch.Tensor:
    """Construct a PSD RBF kernel L_ij = exp(-beta * ||phi_i - phi_j||^2)."""
    # features: (N, D)
    sq_norm = (features**2).sum(dim=1, keepdim=True)
    dist_sq = sq_norm + sq_norm.transpose(0, 1) - 2.0 * (features @ features.transpose(0, 1))
    kernel = torch.exp(-beta * dist_sq.clamp_min(0.0))
    return kernel


def conditional_marginals(
    kernel: torch.Tensor,
    subset_indices: torch.Tensor,
    jitter: float = 1e-6,
) -> torch.Tensor:
    """Compute v_i = P(S U {i} subseteq Y | S subseteq Y) from Eq. (2)."""
    num_items = kernel.size(0)
    eye = torch.eye(num_items, device=kernel.device, dtype=kernel.dtype)

    outside_mask = torch.ones(num_items, device=kernel.device, dtype=kernel.dtype)
    if subset_indices.numel() > 0:
        outside_mask[subset_indices] = 0.0

    system_matrix = kernel + torch.diag(outside_mask) + jitter * eye
    inverse = torch.linalg.inv(system_matrix)

    marginals = 1.0 - torch.diag(inverse)
    marginals = marginals.clamp(0.0, 1.0)

    if subset_indices.numel() > 0:
        marginals = marginals.clone()
        marginals[subset_indices] = 0.0

    return marginals


def sample_subset(
    model: nn.Module,
    features: torch.Tensor,
    subset_size: int,
    conditioned_indices: Sequence[int] | None = None,
    greedy: bool = False,
) -> torch.Tensor:
    """Sample using DPPNet Algorithm 1; greedy=True returns mode-like subset."""
    selected = list(dict.fromkeys(conditioned_indices or []))

    while len(selected) < subset_size:
        subset_indices = torch.tensor(selected, dtype=torch.long, device=features.device)
        probs, _ = model(features=features, subset_indices=subset_indices)

        if probs.sum() <= 0:
            break

        if greedy:
            next_idx = int(torch.argmax(probs).item())
        else:
            normalized = probs / probs.sum()
            next_idx = int(torch.multinomial(normalized, num_samples=1).item())

        if next_idx in selected:
            break
        selected.append(next_idx)

    return torch.tensor(selected, dtype=torch.long, device=features.device)
