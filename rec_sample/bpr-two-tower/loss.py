import torch


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    if pos_scores.shape != neg_scores.shape:
        raise ValueError("pos_scores and neg_scores must have the same shape")
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-12).mean()
