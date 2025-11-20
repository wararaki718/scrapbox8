import torch
import torch.nn as nn


# dpp based loss
class DiversityAwareDPPBasedTripletLoss(nn.Module):
    def __init__(self, margin: float=1.0, dpp_lambda: float=0.5, epsilon: float=1e-5) -> None:
        super().__init__()
        self.margin = margin
        self.dpp_lambda = dpp_lambda
        self.epsilon = epsilon
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

        # Compute DPP-based diversity regularization
        pos_similarity = torch.matmul(positive, positive.t())

        # positive samples diversity
        identity = torch.eye(pos_similarity.size(0)).to(pos_similarity.device)
        dpp_loss = -torch.logdet(pos_similarity + self.epsilon * identity)

        # total loss
        total_loss = basic_loss + self.dpp_lambda * dpp_loss
        return total_loss
