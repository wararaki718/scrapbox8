import torch
import torch.nn as nn


# dpp based loss
class DiversityAwareHingeBasedTripletLoss(nn.Module):
    def __init__(self, margin: float=1.0, alpha: float=0.5) -> None:
        super().__init__()
        self.margin = margin
        self.alpha = alpha
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0 - torch.cosine_similarity(x, y),
            margin=margin,
        )

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        # triplet loss
        basic_loss = self.triplet_loss(anchor, positive, negative)

        similarity = torch.cosine_similarity(positive, negative, dim=1)
        dynamic_margin: torch.Tensor = self.margin * (self.alpha - similarity)

        # total loss
        total_loss = torch.relu(dynamic_margin.unsqueeze(0) + basic_loss)
        return total_loss.sum()
