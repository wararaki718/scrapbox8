from __future__ import annotations

import torch

from model import TwoTowerModel


def recall_at_k(
    model: TwoTowerModel,
    user_features: torch.Tensor,
    item_features: torch.Tensor,
    positive_item_per_user: dict[int, int],
    k: int = 10,
) -> float:
    model.eval()
    hits = 0
    with torch.no_grad():
        item_emb = model.encode_item(item_features)
        for user_idx, pos_item_idx in positive_item_per_user.items():
            user_emb = model.encode_user(user_features[user_idx].unsqueeze(0))
            scores = torch.mv(item_emb, user_emb.squeeze(0))
            top_k_indices = torch.topk(scores, k=k).indices.tolist()
            if pos_item_idx in top_k_indices:
                hits += 1

    return hits / len(positive_item_per_user)
