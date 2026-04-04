import random

import torch


def build_synthetic_data(
    num_items: int,
    num_baskets: int,
    min_basket_size: int,
    max_basket_size: int,
) -> list[torch.Tensor]:
    rng = random.Random(42)
    baskets: list[torch.Tensor] = []
    for _ in range(num_baskets):
        size = rng.randint(min_basket_size, max_basket_size)
        items = rng.sample(range(num_items), k=size)
        baskets.append(torch.tensor(items, dtype=torch.long))
    return baskets


def compute_item_counts(num_items: int, baskets: list[torch.Tensor]) -> torch.Tensor:
    counts = torch.zeros(num_items, dtype=torch.float32)
    for basket in baskets:
        counts[basket] += 1.0
    return counts
