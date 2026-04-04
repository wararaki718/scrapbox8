from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DeepDPPConfig:
    num_items: int
    input_dim: int
    embedding_dim: int
    hidden_dims: tuple[int, ...] = (256, 128)


class DeepDPP(nn.Module):
    """Deep DPP model that maps item features to low-rank DPP embeddings."""

    def __init__(self, config: DeepDPPConfig) -> None:
        super().__init__()
        self.config = config
        self.mlp = self._build_mlp(
            input_dim=config.input_dim,
            hidden_dims=config.hidden_dims,
            output_dim=config.embedding_dim,
        )

    @staticmethod
    def _build_mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.SELU())
            prev = hidden
        layers.append(nn.Linear(prev, output_dim))
        return nn.Sequential(*layers)

    def forward(self, item_features: torch.Tensor) -> torch.Tensor:
        # item_features: (num_items, input_dim)
        embeddings = self.mlp(item_features)
        # embeddings (V): (num_items, embedding_dim)
        return embeddings

    @staticmethod
    def compute_kernel(embeddings: torch.Tensor) -> torch.Tensor:
        # embeddings: (num_items, embedding_dim)
        kernel = embeddings @ embeddings.transpose(0, 1)
        # kernel (L): (num_items, num_items)
        return kernel
