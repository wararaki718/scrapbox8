import torch
import torch.nn as nn


class TopKGate(nn.Module):
    def __init__(self, n_experts: int, n_input: int, top_k: int=2) -> None:
        super().__init__()
        self._top_k = top_k
        self._model = nn.Sequential(
            nn.Linear(n_input, n_experts, bias=False),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self._model(x)
        top_k_weights, top_k_indices = torch.topk(logits, self._top_k, dim=-1)
        weights = torch.zeros_like(logits)
        weights.scatter_(1, top_k_indices, top_k_weights)

        return weights, top_k_indices
