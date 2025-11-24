import torch
import torch.nn as nn


class BCETripletLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

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
        loss = self.criterion(predictions, labels)
        return loss
