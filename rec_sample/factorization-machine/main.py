from __future__ import annotations

import argparse
import numpy as np

from model import FactorizationMachineRegressor
from utils import (
    load_movielens_official_small,
    make_movielens_user_item_features,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Factorization Machine on MovieLens")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--k", type=int, default=8)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20000,
        help="学習に使う最大レコード数 (default: 20000)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data = load_movielens_official_small()
    x, y = make_movielens_user_item_features(data)

    if args.max_samples > 0 and x.shape[0] > args.max_samples:
        x = x[: args.max_samples]
        y = y[: args.max_samples]

    fm = FactorizationMachineRegressor(
        n_features=x.shape[1],
        k=args.k,
        seed=0,
    )

    before = fm.mse(x, y)
    history = fm.fit(
        x,
        y,
        epochs=args.epochs,
        lr=args.lr,
        reg_w=1e-4,
        reg_v=1e-4,
    )
    after = history[-1]

    target_user = 0
    seen = data.item_indices[data.user_indices == target_user]
    rec_idx = fm.recommend(
        user_index=target_user,
        num_users=data.num_users,
        num_items=data.num_items,
        seen_item_indices=seen,
        top_k=5,
    )
    rec_movie_ids = data.raw_item_ids[np.array(rec_idx, dtype=np.int64)]

    print("dataset=movielens:ml-latest-small")
    print(f"users={data.num_users}, items={data.num_items}, ratings={len(data.ratings)}")
    print(f"mse_before={before:.4f}, mse_after={after:.4f}")
    print(
        f"recommendations_for_user_{int(data.raw_user_ids[target_user])}: "
        f"{rec_movie_ids.tolist()}"
    )


if __name__ == "__main__":
    main()
