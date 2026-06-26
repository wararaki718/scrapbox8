from __future__ import annotations

import numpy as np


class MatrixFactorization:
    def __init__(
        self,
        num_users: int,
        num_items: int,
        n_factors: int = 16,
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.user_factors = 0.1 * rng.standard_normal((num_users, n_factors))
        self.item_factors = 0.1 * rng.standard_normal((num_items, n_factors))
        self.user_bias = np.zeros(num_users, dtype=np.float64)
        self.item_bias = np.zeros(num_items, dtype=np.float64)
        self.global_bias = 0.0

    def predict(self, user_indices: np.ndarray, item_indices: np.ndarray) -> np.ndarray:
        interaction = np.sum(
            self.user_factors[user_indices] * self.item_factors[item_indices],
            axis=1,
        )
        return (
            self.global_bias
            + self.user_bias[user_indices]
            + self.item_bias[item_indices]
            + interaction
        )

    def mse(
        self,
        user_indices: np.ndarray,
        item_indices: np.ndarray,
        ratings: np.ndarray,
    ) -> float:
        pred = self.predict(user_indices, item_indices)
        return float(np.mean((ratings - pred) ** 2))

    def fit(
        self,
        user_indices: np.ndarray,
        item_indices: np.ndarray,
        ratings: np.ndarray,
        epochs: int = 100,
        lr: float = 0.01,
        reg: float = 0.01,
    ) -> list[float]:
        self.global_bias = float(np.mean(ratings))
        history: list[float] = []

        for _ in range(epochs):
            order = np.arange(len(ratings))
            np.random.shuffle(order)
            for idx in order:
                u = int(user_indices[idx])
                i = int(item_indices[idx])
                r = float(ratings[idx])

                pred = self.global_bias + self.user_bias[u] + self.item_bias[i] + float(
                    np.dot(self.user_factors[u], self.item_factors[i])
                )
                err = r - pred

                pu = self.user_factors[u].copy()
                qi = self.item_factors[i].copy()

                self.user_bias[u] += lr * (err - reg * self.user_bias[u])
                self.item_bias[i] += lr * (err - reg * self.item_bias[i])
                self.user_factors[u] += lr * (err * qi - reg * pu)
                self.item_factors[i] += lr * (err * pu - reg * qi)

            history.append(self.mse(user_indices, item_indices, ratings))

        return history

    def recommend(
        self,
        user_index: int,
        seen_item_indices: np.ndarray,
        top_k: int = 5,
    ) -> list[int]:
        all_items = np.arange(self.item_factors.shape[0], dtype=np.int64)
        user_vec = np.full_like(all_items, user_index)
        scores = self.predict(user_vec, all_items)

        scores[np.asarray(seen_item_indices, dtype=np.int64)] = -np.inf
        ranked = np.argsort(scores)[::-1]
        return ranked[:top_k].tolist()
