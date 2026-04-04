from __future__ import annotations

import torch

from .deep_dpp import DeepDPP, deep_dpp_loss, next_item_scores
from .deep_dpp.model import DeepDPPConfig
from .utils import build_synthetic_data, compute_item_counts


def main() -> None:
    torch.manual_seed(7)

    num_items = 80
    input_dim = 80
    embedding_dim = 24

    baskets = build_synthetic_data(num_items=num_items, num_baskets=400, min_basket_size=2, max_basket_size=8)
    item_counts = compute_item_counts(num_items=num_items, baskets=baskets)

    item_features = torch.eye(num_items, dtype=torch.float32)

    model = DeepDPP(
        DeepDPPConfig(
            num_items=num_items,
            input_dim=input_dim,
            embedding_dim=embedding_dim,
            hidden_dims=(128, 64),
        )
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(200):
        optimizer.zero_grad()
        embeddings = model(item_features)
        loss = deep_dpp_loss(
            embeddings=embeddings,
            baskets=baskets,
            item_counts=item_counts,
            alpha=0.05,
        )
        loss.backward()
        optimizer.step()

        if step % 40 == 0:
            print(f"step={step:03d} loss={loss.item():.4f}")

    observed = baskets[0][:2]
    final_embeddings = model(item_features).detach()
    scores = next_item_scores(final_embeddings, observed_items=observed)
    top5 = torch.topk(scores, k=5).indices.tolist()
    print(f"observed items: {observed.tolist()}")
    print(f"top-5 recommended next items: {top5}")


if __name__ == "__main__":
    main()
