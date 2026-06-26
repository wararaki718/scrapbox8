from __future__ import annotations

from typing import Iterable, Sequence

import torch


def _elementary_symmetric_polynomial_k(eigenvalues: torch.Tensor, k: int) -> torch.Tensor:
    """Return the k-th elementary symmetric polynomial of eigenvalues."""
    if k < 0:
        raise ValueError("k must be non-negative")
    if k == 0:
        return torch.ones((), dtype=eigenvalues.dtype, device=eigenvalues.device)

    e: list[torch.Tensor] = [
        torch.ones((), dtype=eigenvalues.dtype, device=eigenvalues.device)
    ] + [torch.zeros((), dtype=eigenvalues.dtype, device=eigenvalues.device) for _ in range(k)]

    for i, lam in enumerate(eigenvalues):
        next_e = list(e)
        max_order = min(i + 1, k)
        for order in range(max_order, 0, -1):
            next_e[order] = e[order] + lam * e[order - 1]
        e = next_e

    return e[k]


def sum_k_dpp_partition(L: torch.Tensor, k: int) -> torch.Tensor:
    """Compute Z_k = e_k(lambda_1, ..., lambda_n) for a PSD kernel L."""
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError("L must be a square matrix")

    eigenvalues = torch.linalg.eigvalsh(L)
    clipped = torch.clamp(eigenvalues, min=0.0)
    return _elementary_symmetric_polynomial_k(clipped, k)


def log_k_dpp_probability(L: torch.Tensor, basket: Sequence[int], k: int) -> torch.Tensor:
    """Compute log P(Y=basket | L, |Y|=k) for a k-DPP."""
    if len(basket) != k:
        raise ValueError("basket size must equal k")

    if len(set(basket)) != len(basket):
        raise ValueError("basket must not contain duplicates")

    index = torch.tensor(basket, dtype=torch.long, device=L.device)
    sub = L[index][:, index]

    sign, log_det = torch.linalg.slogdet(sub)
    if torch.any(sign <= 0):
        raise ValueError("selected principal minor is not positive definite")

    z_k = sum_k_dpp_partition(L, k)
    return log_det - torch.log(z_k + 1e-12)


def _batched_tensor_indices(index_groups: Iterable[Sequence[int]], device: torch.device) -> torch.Tensor:
    """Convert index groups to a dense LongTensor."""
    return torch.tensor(list(index_groups), dtype=torch.long, device=device)
