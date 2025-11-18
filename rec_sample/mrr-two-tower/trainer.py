import torch
import torch.nn as nn

from loss import dpp_diversity_loss
from model import TwoTowerModel

class Trainer:
    def __init__(
        self, model: TwoTowerModel,
        lambda_div: float=0.01,
        lr: float=1e-3,
    ) -> None:
        self.model = model
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.lambda_div = lambda_div

    def train(self, dataloader: torch.utils.data.DataLoader, num_epochs: int=10) -> None:
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in dataloader:
                loss = self.step(batch)
                total_loss += loss
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    def step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
        x_user, x_positive, x_negatives = batch
        # Ensure batch dimensions
        x_user = x_user.unsqueeze(0) if x_user.dim() == 1 else x_user # (1, user_feat_dim)
        x_positive = x_positive.unsqueeze(0) if x_positive.dim() == 1 else x_positive # (1, item_feat_dim)

        # Positive score
        pos_score = self.model(x_user, x_positive)

        # Expand user features to match negative samples
        x_user_expand = x_user.expand(x_negatives.size(0), -1) # (num_neg, user_feat_dim)
        neg_score = self.model(x_user_expand, x_negatives)

        pos_score = pos_score.view(-1)
        neg_score = neg_score.view(-1)

        logits = torch.cat([pos_score, neg_score], dim=0)
        labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)], dim=0)
        rel_loss = self.loss_fn(logits, labels)

        div_loss = dpp_diversity_loss(self.model.forward_item(x_positive.unsqueeze(0)))
        loss: torch.Tensor = rel_loss + self.lambda_div * div_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())
