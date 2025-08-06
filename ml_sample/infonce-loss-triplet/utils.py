import torch


def get_triplet(n_data: int, n_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    anchor = torch.randn(n_data, n_dim)
    positive = torch.randn(n_data, n_dim)
    negative = torch.randn(n_data, n_dim)

    return anchor, positive, negative
