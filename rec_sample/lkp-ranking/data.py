from __future__ import annotations

import itertools

import torch

from model import PersonalizedKDPP


def _sample_k_dpp_basket(L: torch.Tensor, k: int, rng: torch.Generator) -> list[int]:
    """Sample a fixed-size subset from a small k-DPP by explicit enumeration."""
    n_items = L.shape[0]
    subsets = list(itertools.combinations(range(n_items), k))

    scores: list[torch.Tensor] = []
    for subset in subsets:
        index = torch.tensor(subset, dtype=torch.long)
        sub = L[index][:, index]
        score = torch.det(sub)
        scores.append(torch.clamp(score, min=0.0))

    probs = torch.stack(scores)
    probs = probs / (probs.sum() + 1e-12)

    sampled_idx = torch.multinomial(probs, num_samples=1, replacement=False, generator=rng).item()
    return list(subsets[sampled_idx])


def build_synthetic_dataset(
    num_users: int,
    num_items: int,
    rank: int,
    k: int,
    num_samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate (user, basket) pairs from a hidden personalized k-DPP model."""
    rng = torch.Generator().manual_seed(seed)
    hidden = PersonalizedKDPP(num_users=num_users, num_items=num_items, rank=rank, k=k)

    with torch.no_grad():
        hidden.item_factors.copy_(torch.randn(num_items, rank, generator=rng) * 0.6)
        hidden.user_quality_logits.copy_(torch.randn(num_users, num_items, generator=rng) * 0.8)

    user_ids: list[int] = []
    baskets: list[list[int]] = []

    for _ in range(num_samples):
        user_id = int(torch.randint(0, num_users, (1,), generator=rng).item())
        L_u = hidden._kernel_for_user(user_id)
        basket = _sample_k_dpp_basket(L_u, k=k, rng=rng)

        user_ids.append(user_id)
        baskets.append(basket)

    users_tensor = torch.tensor(user_ids, dtype=torch.long)
    baskets_tensor = torch.tensor(baskets, dtype=torch.long)
    return users_tensor, baskets_tensor
