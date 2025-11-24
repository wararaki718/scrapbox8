import torch
import torch.nn as nn


class DiversityCorrelationPenaltyBasedTripletLoss(nn.Module):
    def __init__(self, margin: float=1.0, lambda_: float=0.001) -> None:
        super().__init__()
        self.lambda_ = lambda_
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

        if positive.size(0) <= 1:
            return basic_loss

        centered_embeddings = positive - positive.mean(dim=0, keepdim=True)
        correlation_matrix = torch.matmul(
            centered_embeddings,
            centered_embeddings.t()
        ) / (positive.size(1) - 1)

        diag_mask = ~ torch.eye(positive.size(0), dtype=torch.bool, device=positive.device)
        penalty = (correlation_matrix[diag_mask]).pow(2).sum()

        # total loss
        total_loss = basic_loss + self.lambda_ * penalty
        return total_loss
