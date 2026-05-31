import torch
import torch.nn as nn


class PoissonFactorization(nn.Module):
    def __init__(self, n_users, n_items, k):
        super().__init__()
        self.user_factors = nn.Parameter(
            torch.rand(n_users, k)
        )
        self.item_factors = nn.Parameter(
            torch.rand(n_items, k)
        )

    def forward(self):
        rate = torch.matmul(
            self.user_factors,
            self.item_factors.T
        )
        return torch.clamp(rate, min=1e-8)
