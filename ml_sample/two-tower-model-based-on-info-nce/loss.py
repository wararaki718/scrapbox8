import torch
import torch.nn as nn


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float=1.0) -> None:
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.temperature = temperature

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        labels = torch.cat((
            torch.ones(anchor.size(0), device=anchor.device),
            torch.zeros(negative.size(0), device=negative.device),
        ))
        predictions = torch.cat((
            torch.cosine_similarity(anchor, positive, dim=1),
            torch.cosine_similarity(anchor, negative, dim=1),
        ))
        predictions /= self.temperature

        log_prob = nn.functional.log_softmax(predictions, dim=0).view(-1, 1)
        positive_scores = log_prob.diag()
        loss = - positive_scores.mean()

        return loss
