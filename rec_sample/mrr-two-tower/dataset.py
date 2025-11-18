import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class ML100kDataset(Dataset):
    def __init__(
        self,
        df_ratings: pd.DataFrame,
        user_features: np.ndarray,
        item_features: np.ndarray,
        num_neg: int=4,
    ) -> None:
        self.user_features = user_features
        self.item_features = item_features
        self.user_map = {uid: i for i, uid in enumerate(sorted(df_ratings.userId.unique()))}
        self.item_map = {iid: i for i, iid in enumerate(sorted(df_ratings.movieId.unique()))}
        self.user_positive_items: dict[int, set[int]] = df_ratings.groupby('userId')['movieId'].apply(set).to_dict()
        self.num_neg = num_neg
        self.users: list[int] = list(self.user_positive_items.keys())

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        uid = self.users[idx]
        pos_item_id = random.choice(list(self.user_positive_items[uid]))

        neg_items: list[int] = []
        negative_items: list[int] = list(self.item_map.keys())
        while len(neg_items) < self.num_neg:
            negative = random.choice(negative_items)
            if negative not in self.user_positive_items[uid]:
                neg_items.append(negative)

        x_user = torch.tensor(self.user_features[self.user_map[uid]], dtype=torch.float32)
        x_positive = torch.tensor(self.item_features[self.item_map[pos_item_id]], dtype=torch.float32)
        x_negatives = torch.tensor(self.item_features[[self.item_map[i] for i in neg_items]], dtype=torch.float32)
        return x_user, x_positive, x_negatives
