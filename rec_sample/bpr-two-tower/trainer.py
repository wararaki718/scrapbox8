from __future__ import annotations

import torch

from loss import bpr_loss
from model import TwoTowerModel


class Trainer:
    def __init__(self, model: TwoTowerModel, lr: float = 1e-3) -> None:
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    def train(
        self,
        dataloader: torch.utils.data.DataLoader,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
        num_epochs: int = 5,
    ) -> None:
        self.model.train()
        for epoch in range(num_epochs):
            running_loss = 0.0
            for user_idx, pos_idx, neg_idx in dataloader:
                loss_value = self.step(
                    user_idx=user_idx,
                    pos_idx=pos_idx,
                    neg_idx=neg_idx,
                    user_features=user_features,
                    item_features=item_features,
                )
                running_loss += loss_value

            avg_loss = running_loss / len(dataloader)
            print(f"epoch={epoch + 1:02d} loss={avg_loss:.4f}")

    def step(
        self,
        user_idx: torch.Tensor,
        pos_idx: torch.Tensor,
        neg_idx: torch.Tensor,
        user_features: torch.Tensor,
        item_features: torch.Tensor,
    ) -> float:
        x_user = user_features[user_idx]
        x_pos = item_features[pos_idx]
        x_neg = item_features[neg_idx]

        pos_scores = self.model(x_user, x_pos)
        neg_scores = self.model(x_user, x_neg)

        loss = bpr_loss(pos_scores=pos_scores, neg_scores=neg_scores)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())
