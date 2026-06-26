from __future__ import annotations

import torch

from loss import PersonalizedKDPPNLLLoss
from model import PersonalizedKDPP


def train_model(
    model: PersonalizedKDPP,
    criterion: PersonalizedKDPPNLLLoss,
    user_ids: torch.Tensor,
    baskets: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    rng = torch.Generator().manual_seed(seed)

    n = len(user_ids)
    for epoch in range(1, epochs + 1):
        order = torch.randperm(n, generator=rng)
        total = 0.0

        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            batch_users = user_ids[idx]
            batch_baskets = baskets[idx]

            optimizer.zero_grad()
            loss = criterion(model, batch_users, batch_baskets)
            loss.backward()
            optimizer.step()

            total += loss.item() * len(idx)

        if epoch == 1 or epoch % 20 == 0:
            avg = total / n
            print(f"epoch={epoch:03d} nll={avg:.4f}")
