from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DPPNetConfig:
    feature_dim: int
    hidden_dims: tuple[int, ...] = (256, 128)


def inhibitive_attention(
    features: torch.Tensor,
    subset_indices: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute the paper's inhibitive attention vector (Eq. 4)."""
    # features: (N, D)
    num_items = features.size(0)

    if subset_indices.numel() == 0:
        return torch.full(
            (num_items,),
            fill_value=1.0 / float(num_items),
            device=features.device,
            dtype=features.dtype,
        )

    # query: (k, D)
    query = features.index_select(0, subset_indices)
    # logits: (k, N)
    logits = query @ features.transpose(0, 1) / math.sqrt(float(features.size(1)))
    dissimilarities = 1.0 - torch.softmax(logits / temperature, dim=1)
    # a_prime: (N,)
    a_prime = torch.prod(dissimilarities, dim=0)
    a = a_prime / a_prime.sum().clamp_min(eps)
    return a


class DPPNet(nn.Module):
    """DPPNet with inhibitive attention for fixed-size ground sets."""

    def __init__(self, config: DPPNetConfig) -> None:
        super().__init__()
        self.config = config

        input_dim = config.feature_dim * 2 + 2
        layers: list[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(
        self, features: torch.Tensor, subset_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # features: (N, D)
        # subset_indices: (k,)
        attention = inhibitive_attention(features=features, subset_indices=subset_indices)

        subset_mask = torch.zeros(features.size(0), device=features.device, dtype=features.dtype)
        if subset_indices.numel() > 0:
            subset_mask[subset_indices] = 1.0

        # context: (N, D)
        context = features * attention.unsqueeze(1)
        # network_input: (N, 2D+2)
        network_input = torch.cat(
            [
                features,
                context,
                attention.unsqueeze(1),
                subset_mask.unsqueeze(1),
            ],
            dim=1,
        )

        logits = self.mlp(network_input).squeeze(1)
        probs = torch.sigmoid(logits)

        if subset_indices.numel() > 0:
            probs = probs.clone()
            probs[subset_indices] = 0.0

        return probs, attention
