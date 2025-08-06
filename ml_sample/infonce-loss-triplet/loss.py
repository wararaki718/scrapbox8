import torch


def info_nce(anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    pos_sim = torch.cosine_similarity(anchor, positive, dim=0)
    neg_sim = torch.cosine_similarity(anchor, negative, dim=0)
    return -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim)))
