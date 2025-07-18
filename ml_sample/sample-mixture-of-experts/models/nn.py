import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    def __init__(self, n_input: int, n_hidden: int, n_output: int) -> None:
        super().__init__()

        self._model = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)
