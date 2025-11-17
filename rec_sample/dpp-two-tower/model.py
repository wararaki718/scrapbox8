import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoTowerModel(nn.Module):
    def __init__(
        self,
        user_feat_dim: int,
        item_feat_dim: int,
        emb_dim: int=64,
        hidden_dim: int=128
    ):
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.Linear(user_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim)
        )
        self.item_tower = nn.Sequential(
            nn.Linear(item_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, emb_dim)
        )

    def forward_user(self, x_user: torch.Tensor) -> torch.Tensor:
        u = F.normalize(self.user_tower(x_user), dim=1)
        return u

    def forward_item(self, x_item: torch.Tensor) -> torch.Tensor:
        v = F.normalize(self.item_tower(x_item), dim=1)
        return v

    def forward(self, x_user: torch.Tensor, x_item: torch.Tensor) -> torch.Tensor:
        u = self.forward_user(x_user)
        v = self.forward_item(x_item)
        return torch.sum(u * v, dim=1)
