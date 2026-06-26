from __future__ import annotations

import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss

from model import PersonalizedKDPP
from utils import log_k_dpp_probability


class PersonalizedKDPPNLLLoss(_Loss):
    """Negative log-likelihood loss for a personalized k-DPP model."""

    def __init__(self, l2_weight: float = 1e-4, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be one of {'mean', 'sum', 'none'}")

        self.l2_weight = l2_weight

    def forward(self, model: PersonalizedKDPP, user_ids: Tensor, baskets: Tensor) -> Tensor:
        if baskets.ndim != 2:
            raise ValueError("baskets must be [batch_size, k]")
        if baskets.shape[1] != model.k:
            raise ValueError("basket width must equal model k")
        if user_ids.ndim != 1:
            raise ValueError("user_ids must be [batch_size]")
        if user_ids.shape[0] != baskets.shape[0]:
            raise ValueError("user_ids and baskets batch size must match")

        log_probs: list[Tensor] = []
        for user_id, basket in zip(user_ids.tolist(), baskets.tolist()):
            L_u = model._kernel_for_user(user_id)
            log_probs.append(log_k_dpp_probability(L_u, basket=basket, k=model.k))

        nll_per_sample = -torch.stack(log_probs)

        if self.reduction == "mean":
            base_loss = nll_per_sample.mean()
        elif self.reduction == "sum":
            base_loss = nll_per_sample.sum()
        else:
            base_loss = nll_per_sample

        if self.reduction == "none":
            return base_loss

        reg = self.l2_weight * (
            model.item_factors.pow(2).mean() + model.user_quality_logits.pow(2).mean()
        )
        return base_loss + reg
