import torch
import torch.nn as nn

from .router import Router
from .nn import NeuralNetwork


class MixtureOfExperts(nn.Module):
    def __init__(self, n_experts: int, n_input: int, n_hidden: int, n_output: int) -> None:
        super().__init__()

        self._router = Router(n_experts, n_input, n_hidden)
        self._experts = nn.ModuleList(
            [NeuralNetwork(n_input, n_hidden, n_output) for _ in range(n_experts)]
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        routing_weights = self._router(x)
        expert_outputs = torch.stack([expert(x) for expert in self._experts], dim=-2)
        weighted_outputs = expert_outputs * routing_weights.unsqueeze(-1)
        return weighted_outputs.sum(dim=-2), routing_weights, expert_outputs
