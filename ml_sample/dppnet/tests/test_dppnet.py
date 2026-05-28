from __future__ import annotations

import torch
from torch import nn

from dppnet import (
    DPPNet,
    DPPNetConfig,
    conditional_marginals,
    inhibitive_attention,
    make_rbf_kernel,
)
from dppnet.dpp import sample_subset


def test_inhibitive_attention_shape_and_simplex() -> None:
    torch.manual_seed(0)
    features = torch.rand(10, 4)
    subset = torch.tensor([1, 3, 5], dtype=torch.long)

    attention = inhibitive_attention(features=features, subset_indices=subset)

    assert attention.shape == (10,)
    assert torch.all(attention >= 0.0)
    assert torch.isclose(attention.sum(), torch.tensor(1.0), atol=1e-6)


def test_conditional_marginals_observed_items_are_zero() -> None:
    torch.manual_seed(0)
    features = torch.rand(12, 3)
    kernel = make_rbf_kernel(features=features, beta=2.5)
    subset = torch.tensor([2, 7], dtype=torch.long)

    marginals = conditional_marginals(kernel=kernel, subset_indices=subset)

    assert marginals.shape == (12,)
    assert marginals[2].item() == 0.0
    assert marginals[7].item() == 0.0
    assert torch.all((marginals >= 0.0) & (marginals <= 1.0))


def test_forward_output_shape_and_masking() -> None:
    model = DPPNet(DPPNetConfig(feature_dim=5, hidden_dims=(16, 8)))
    features = torch.rand(9, 5)
    subset = torch.tensor([0, 4], dtype=torch.long)

    probs, attention = model(features=features, subset_indices=subset)

    assert probs.shape == (9,)
    assert attention.shape == (9,)
    assert probs[0].item() == 0.0
    assert probs[4].item() == 0.0


def test_training_reduces_supervised_marginal_loss() -> None:
    torch.manual_seed(42)

    num_items = 20
    feature_dim = 3
    features = torch.rand(num_items, feature_dim)
    kernel = make_rbf_kernel(features=features, beta=3.0)

    states = []
    for size in [0, 1, 2, 3, 4]:
        perm = torch.randperm(num_items)
        states.append(perm[:size].sort().values)

    model = DPPNet(DPPNetConfig(feature_dim=feature_dim, hidden_dims=(32, 16)))
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    criterion = nn.BCELoss()

    def mean_loss() -> torch.Tensor:
        losses = []
        for subset in states:
            target = conditional_marginals(kernel=kernel, subset_indices=subset)
            pred, _ = model(features=features, subset_indices=subset)
            losses.append(criterion(pred, target))
        return torch.stack(losses).mean()

    first = mean_loss().item()
    for _ in range(80):
        optimizer.zero_grad()
        loss = mean_loss()
        loss.backward()
        optimizer.step()
    last = mean_loss().item()

    assert last < first


def test_sampler_respects_conditioned_items_and_cardinality() -> None:
    torch.manual_seed(1)
    model = DPPNet(DPPNetConfig(feature_dim=2, hidden_dims=(8, 8)))
    features = torch.rand(15, 2)

    conditioned = [2, 9]
    subset = sample_subset(
        model=model,
        features=features,
        subset_size=6,
        conditioned_indices=conditioned,
        greedy=True,
    )

    assert subset.numel() == 6
    assert len(set(subset.tolist())) == 6
    assert 2 in subset.tolist()
    assert 9 in subset.tolist()
