from __future__ import annotations

import torch

from data import build_synthetic_dataset
from loss import PersonalizedKDPPNLLLoss
from model import PersonalizedKDPP
from recommend import show_recommendation_example
from train import train_model


def _main() -> None:
    torch.manual_seed(42)

    num_users = 24
    num_items = 36
    rank = 8
    k = 3

    train_users, train_baskets = build_synthetic_dataset(
        num_users=num_users,
        num_items=num_items,
        rank=rank,
        k=k,
        num_samples=1200,
        seed=7,
    )

    model = PersonalizedKDPP(num_users=num_users, num_items=num_items, rank=rank, k=k)
    criterion = PersonalizedKDPPNLLLoss(l2_weight=model.l2_weight)

    with torch.no_grad():
        initial_nll = criterion(model, train_users[:64], train_baskets[:64]).item()
    print(f"initial_nll={initial_nll:.4f}")

    train_model(
        model=model,
        criterion=criterion,
        user_ids=train_users,
        baskets=train_baskets,
        epochs=80,
        batch_size=64,
        lr=1e-2,
        seed=23,
    )

    with torch.no_grad():
        final_nll = criterion(model, train_users[:64], train_baskets[:64]).item()
    print(f"final_nll={final_nll:.4f}")

    first_user = int(train_users[0].item())
    first_seen = train_baskets[0].tolist()
    show_recommendation_example(model, user_id=first_user, seen_items=first_seen)


if __name__ == "__main__":
    _main()
