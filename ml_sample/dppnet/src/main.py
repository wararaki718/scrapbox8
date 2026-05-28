from __future__ import annotations

import torch
from torch import nn

from dppnet import DPPNet, DPPNetConfig, conditional_marginals, make_rbf_kernel, sample_subset


def _build_training_states(
    num_items: int,
    max_subset_size: int,
    num_states: int,
    seed: int,
) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    states: list[torch.Tensor] = []
    for _ in range(num_states):
        size = int(
            torch.randint(low=0, high=max_subset_size, size=(1,), generator=generator).item()
        )
        perm = torch.randperm(num_items, generator=generator)
        states.append(perm[:size].sort().values)
    return states


def run_demo() -> None:
    torch.manual_seed(0)

    num_items = 64
    feature_dim = 2
    subset_size = 12

    features = torch.rand(num_items, feature_dim)
    kernel = make_rbf_kernel(features=features, beta=4.0)

    model = DPPNet(DPPNetConfig(feature_dim=feature_dim, hidden_dims=(128, 64)))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    criterion = nn.BCELoss()

    states = _build_training_states(
        num_items=num_items,
        max_subset_size=subset_size,
        num_states=256,
        seed=123,
    )

    def batch_loss() -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for subset in states:
            target = conditional_marginals(kernel=kernel, subset_indices=subset)
            pred, _ = model(features=features, subset_indices=subset)
            losses.append(criterion(pred, target))
        return torch.stack(losses).mean()

    initial = batch_loss().item()
    for _ in range(120):
        optimizer.zero_grad()
        loss = batch_loss()
        loss.backward()
        optimizer.step()
    final = batch_loss().item()

    conditioned = [0, 3]
    greedy_subset = sample_subset(
        model=model,
        features=features,
        subset_size=subset_size,
        conditioned_indices=conditioned,
        greedy=True,
    )
    sampled_subset = sample_subset(
        model=model,
        features=features,
        subset_size=subset_size,
        conditioned_indices=conditioned,
        greedy=False,
    )

    print(f"initial_loss={initial:.6f}")
    print(f"final_loss={final:.6f}")
    print(f"conditioned={conditioned}")
    print(f"greedy_subset={greedy_subset.tolist()}")
    print(f"sampled_subset={sampled_subset.tolist()}")


if __name__ == "__main__":
    run_demo()
