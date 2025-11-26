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
        predictions = torch.cosine_similarity(
            torch.cat([anchor, anchor], dim=0),
            torch.cat([positive, negative], dim=0),
            dim=1,
        )
        loss = self.criterion(predictions, labels)
        return loss
