import torch
import torch.nn as nn

from expert import Expert
from gate import TopKGate


class MixtureOfExperts(nn.Module):
    def __init__(self, n_experts: int, n_input: int, n_hidden: int, n_output: int, top_k: int=2) -> None:
        super().__init__()
        self._gate = TopKGate(n_experts, n_input, top_k)
        self._experts = nn.ModuleList(
            [Expert(n_input, n_hidden, n_output) for _ in range(n_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights, indices = self._gate(x)
        expert_outputs = torch.stack([
            self._experts[i](x) for i in range(len(self._experts))
        ], dim=1)
        output = torch.sum(weights.unsqueeze(-1) * expert_outputs, dim=1)
        return output
