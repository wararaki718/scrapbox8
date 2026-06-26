from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class PersonalizedKDPP(nn.Module):
    """Low-rank personalized k-DPP model for ranking and reranking."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        rank: int,
        k: int,
        l2_weight: float = 1e-4,
    ) -> None:
        super().__init__()
        if k <= 0:
            raise ValueError("k must be positive")
        if k > num_items:
            raise ValueError("k must be <= num_items")

        self.num_users = num_users
        self.num_items = num_items
        self.rank = rank
        self.k = k
        self.l2_weight = l2_weight

        self.item_factors = nn.Parameter(torch.randn(num_items, rank) * 0.1)
        self.user_quality_logits = nn.Parameter(torch.zeros(num_users, num_items))

    def _kernel_for_user(self, user_id: int) -> torch.Tensor:
        if user_id < 0 or user_id >= self.num_users:
            raise IndexError("user_id out of range")

        quality = torch.nn.functional.softplus(self.user_quality_logits[user_id]) + 1e-6
        weighted_items = self.item_factors * quality.unsqueeze(1)
        L = weighted_items @ weighted_items.T

        jitter = 1e-6 * torch.eye(self.num_items, dtype=L.dtype, device=L.device)
        return L + jitter

    @staticmethod
    def _log_det_subset(L: torch.Tensor, subset: Sequence[int]) -> torch.Tensor:
        if not subset:
            return torch.zeros((), dtype=L.dtype, device=L.device)
        index = torch.tensor(subset, dtype=torch.long, device=L.device)
        sub = L[index][:, index]
        sign, log_det = torch.linalg.slogdet(sub + 1e-6 * torch.eye(len(subset), dtype=L.dtype, device=L.device))
        if torch.any(sign <= 0):
            return torch.tensor(float("-inf"), dtype=L.dtype, device=L.device)
        return log_det

    @torch.no_grad()
    def recommend_next_items(self, user_id: int, seen_items: Sequence[int], top_k: int) -> list[int]:
        if top_k <= 0:
            return []

        L_u = self._kernel_for_user(user_id)
        blocked = set(seen_items)
        candidates = [item for item in range(self.num_items) if item not in blocked]
        selected: list[int] = []

        while candidates and len(selected) < top_k:
            current_logdet = self._log_det_subset(L_u, selected)
            best_item = None
            best_gain = torch.tensor(float("-inf"), dtype=L_u.dtype, device=L_u.device)

            for candidate in candidates:
                candidate_subset = selected + [candidate]
                gain = self._log_det_subset(L_u, candidate_subset) - current_logdet
                if gain > best_gain:
                    best_gain = gain
                    best_item = candidate

            if best_item is None:
                break

            selected.append(best_item)
            candidates.remove(best_item)

        return selected
