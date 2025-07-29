import torch
from deepspeed.moe.layer import MoE


class Model(torch.nn.Module):
    def __init__(self, n_input: int=10, n_hidden: int=10) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(n_input, n_hidden)
        self.moe = MoE(
            expert=torch.nn.Linear(n_hidden, n_hidden),
            num_experts=4,
            hidden_size=n_hidden,
            k=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.moe(x)
        return x
