import torch


def dpp_diversity_loss(item_embeddings: torch.Tensor, eps: float=1e-6) -> torch.Tensor:
    if item_embeddings.size(0) < 2:
        return torch.tensor(0.0)
    K = item_embeddings @ item_embeddings.T
    K = K + eps * torch.eye(K.size(0), device=K.device)
    L = torch.linalg.cholesky(K)
    log_det = 2 * torch.sum(torch.log(torch.diag(L)))
    return -log_det
