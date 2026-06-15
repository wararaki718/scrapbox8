from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class SyntheticDataBundle:
    train_dataset: "BPRTripletDataset"
    eval_positive_item_per_user: dict[int, int]
    user_features: torch.Tensor
    item_features: torch.Tensor


class BPRTripletDataset(Dataset):
    def __init__(
        self,
        user_indices: list[int],
        pos_item_indices: list[int],
        neg_item_indices: list[int],
    ) -> None:
        if not (len(user_indices) == len(pos_item_indices) == len(neg_item_indices)):
            raise ValueError("All index lists must have the same length")
        self.user_indices = user_indices
        self.pos_item_indices = pos_item_indices
        self.neg_item_indices = neg_item_indices

    def __len__(self) -> int:
        return len(self.user_indices)

    def __getitem__(self, idx: int) -> tuple[int, int, int]:
        return (
            self.user_indices[idx],
            self.pos_item_indices[idx],
            self.neg_item_indices[idx],
        )


def build_synthetic_bpr_data(
    num_users: int = 32,
    num_items: int = 120,
    user_latent_dim: int = 8,
    triples_per_user: int = 30,
    seed: int = 42,
) -> SyntheticDataBundle:
    rng = random.Random(seed)
    torch.manual_seed(seed)

    user_latent = torch.randn(num_users, user_latent_dim)
    item_latent = torch.randn(num_items, user_latent_dim)
    scores = user_latent @ item_latent.T

    user_features = torch.eye(num_users, dtype=torch.float32)
    item_features = torch.eye(num_items, dtype=torch.float32)

    user_indices: list[int] = []
    pos_item_indices: list[int] = []
    neg_item_indices: list[int] = []
    eval_positive_item_per_user: dict[int, int] = {}

    for u in range(num_users):
        ranked_items = torch.argsort(scores[u], descending=True).tolist()
        eval_positive_item_per_user[u] = ranked_items[1]

        train_positive_pool = ranked_items[:16]
        negative_pool = ranked_items[60:]

        for _ in range(triples_per_user):
            user_indices.append(u)
            pos_item_indices.append(rng.choice(train_positive_pool))
            neg_item_indices.append(rng.choice(negative_pool))

    dataset = BPRTripletDataset(
        user_indices=user_indices,
        pos_item_indices=pos_item_indices,
        neg_item_indices=neg_item_indices,
    )

    return SyntheticDataBundle(
        train_dataset=dataset,
        eval_positive_item_per_user=eval_positive_item_per_user,
        user_features=user_features,
        item_features=item_features,
    )
