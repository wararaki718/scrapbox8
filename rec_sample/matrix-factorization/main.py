from __future__ import annotations

import argparse
import numpy as np

from model import MatrixFactorization
from utils import (
    load_movielens_official_small,
    load_movielens_small_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Matrix Factorization on MovieLens")
    parser.add_argument(
        "--dataset",
        choices=["official", "sample"],
        default="official",
        help="利用するデータセット (default: official)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dataset == "official":
        data = load_movielens_official_small()
    else:
        data = load_movielens_small_sample()

    model = MatrixFactorization(
        num_users=data.num_users,
        num_items=data.num_items,
        n_factors=8,
        seed=0,
    )

    before = model.mse(data.user_indices, data.item_indices, data.ratings)
    history = model.fit(
        user_indices=data.user_indices,
        item_indices=data.item_indices,
        ratings=data.ratings,
        epochs=10 if args.dataset == "official" else 150,
        lr=0.03,
        reg=0.01,
    )
    after = history[-1]

    target_user = 0
    seen = data.item_indices[data.user_indices == target_user]
    rec_idx = model.recommend(target_user, seen_item_indices=seen, top_k=3)
    rec_movie_ids = data.raw_item_ids[np.array(rec_idx, dtype=np.int64)]

    print(f"dataset={args.dataset}")
    print(f"users={data.num_users}, items={data.num_items}, ratings={len(data.ratings)}")
    print(f"mse_before={before:.4f}, mse_after={after:.4f}")
    print(f"recommendations_for_user_{int(data.raw_user_ids[target_user])}: {rec_movie_ids.tolist()}")


if __name__ == "__main__":
    main()
