from __future__ import annotations

import torch

from src.deep_dpp import DeepDPP, deep_dpp_loss, next_item_scores
from src.deep_dpp.model import DeepDPPConfig


def _toy_baskets() -> list[torch.Tensor]:
    return [
        torch.tensor([0, 1, 2], dtype=torch.long),
        torch.tensor([2, 3], dtype=torch.long),
        torch.tensor([1, 4], dtype=torch.long),
    ]


def _toy_counts(num_items: int, baskets: list[torch.Tensor]) -> torch.Tensor:
    counts = torch.zeros(num_items, dtype=torch.float32)
    for basket in baskets:
        counts[basket] += 1.0
    return counts


def test_forward_shape() -> None:
    model = DeepDPP(
        DeepDPPConfig(num_items=10, input_dim=10, embedding_dim=4, hidden_dims=(16, 8))
    )
    features = torch.eye(10)
    embeddings = model(features)
    assert embeddings.shape == (10, 4)


def test_loss_is_finite_scalar() -> None:
    num_items = 6
    model = DeepDPP(
        DeepDPPConfig(num_items=num_items, input_dim=num_items, embedding_dim=3, hidden_dims=(12,))
    )
    features = torch.eye(num_items)
    baskets = _toy_baskets()
    counts = _toy_counts(num_items=num_items, baskets=baskets)

    loss = deep_dpp_loss(model(features), baskets=baskets, item_counts=counts, alpha=0.01)
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_next_item_scores_mask_observed() -> None:
    model = DeepDPP(
        DeepDPPConfig(num_items=8, input_dim=8, embedding_dim=3, hidden_dims=(10,))
    )
    features = torch.eye(8)
    embeddings = model(features).detach()

    observed = torch.tensor([1, 3], dtype=torch.long)
    scores = next_item_scores(embeddings=embeddings, observed_items=observed)

    assert scores.shape == (8,)
    assert scores[1].item() == float("-inf")
    assert scores[3].item() == float("-inf")


def test_training_reduces_loss() -> None:
    torch.manual_seed(0)
    num_items = 10
    baskets = [
        torch.tensor([0, 1, 2], dtype=torch.long),
        torch.tensor([2, 3, 4], dtype=torch.long),
        torch.tensor([0, 4, 5], dtype=torch.long),
        torch.tensor([6, 7], dtype=torch.long),
        torch.tensor([7, 8, 9], dtype=torch.long),
    ]
    counts = _toy_counts(num_items=num_items, baskets=baskets)

    model = DeepDPP(
        DeepDPPConfig(num_items=num_items, input_dim=num_items, embedding_dim=4, hidden_dims=(16, 8))
    )
    features = torch.eye(num_items)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)

    first_loss = deep_dpp_loss(model(features), baskets=baskets, item_counts=counts, alpha=0.01).item()

    for _ in range(60):
        optimizer.zero_grad()
        loss = deep_dpp_loss(model(features), baskets=baskets, item_counts=counts, alpha=0.01)
        loss.backward()
        optimizer.step()

    last_loss = deep_dpp_loss(model(features), baskets=baskets, item_counts=counts, alpha=0.01).item()
    assert last_loss < first_loss
