import torch
import torch.nn as nn


class Router(nn.Module):
    def __init__(self, n_experts: int, n_input: int, n_hidden: int) -> None:
        super().__init__()

        self._model = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_experts),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)
