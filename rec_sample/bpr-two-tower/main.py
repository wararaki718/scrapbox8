from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from dataset import build_synthetic_bpr_data
from metrics import recall_at_k
from model import TwoTowerModel
from trainer import Trainer


def main() -> None:
    torch.manual_seed(42)

    bundle = build_synthetic_bpr_data(
        num_users=32,
        num_items=120,
        user_latent_dim=8,
        triples_per_user=30,
        seed=42,
    )

    dataloader = DataLoader(bundle.train_dataset, batch_size=256, shuffle=True)

    model = TwoTowerModel(
        user_feat_dim=bundle.user_features.shape[1],
        item_feat_dim=bundle.item_features.shape[1],
        emb_dim=32,
        hidden_dim=64,
    )
    trainer = Trainer(model=model, lr=1e-3)

    before = recall_at_k(
        model=model,
        user_features=bundle.user_features,
        item_features=bundle.item_features,
        positive_item_per_user=bundle.eval_positive_item_per_user,
        k=10,
    )
    print(f"recall@10 before training: {before:.4f}")

    trainer.train(
        dataloader=dataloader,
        user_features=bundle.user_features,
        item_features=bundle.item_features,
        num_epochs=8,
    )

    after = recall_at_k(
        model=model,
        user_features=bundle.user_features,
        item_features=bundle.item_features,
        positive_item_per_user=bundle.eval_positive_item_per_user,
        k=10,
    )
    print(f"recall@10 after training:  {after:.4f}")


if __name__ == "__main__":
    main()
