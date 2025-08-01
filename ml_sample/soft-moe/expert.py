import einops
import torch
import torch.nn as nn


class Expert(nn.Module):
    def __init__(self, n_input: int, n_output: int, n_hidden: int=128, dropout: float = 0.) -> None:
        super().__init__()
        self._model = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, n_output)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


class Experts(nn.Module):
    def __init__(self, experts: list[Expert]) -> None:
        super().__init__()
        self.num_experts = len(experts)
        self.experts = nn.ModuleList(experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (b, e, s, d) -> batch, experts, slots, dim
        b, e, s, _ = x.shape
        x_reshaped = einops.rearrange(x, 'b e s d -> (b e s) d')

        outputs = [expert(x_reshaped[i*b*s:(i+1)*b*s]) for i, expert in enumerate(self.experts)]

        outputs_tensor = torch.cat(outputs, dim=0)
        return einops.rearrange(outputs_tensor, '(b e s) d -> b e s d', b=b, e=e, s=s)
