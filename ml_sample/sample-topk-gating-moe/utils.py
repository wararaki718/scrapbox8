import torch


def get_data(n_data: int, n_dim: int) -> torch.Tensor:
    return torch.randn(n_data, n_dim)
