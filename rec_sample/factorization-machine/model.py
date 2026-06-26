from __future__ import annotations

import numpy as np


class FactorizationMachineRegressor:
    """2次の Factorization Machine (回帰) をSGDで学習する最小実装。"""

    def __init__(self, n_features: int, k: int = 8, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.w0 = 0.0
        self.w = np.zeros(n_features, dtype=np.float64)
        self.v = 0.01 * rng.standard_normal((n_features, k))

    def predict(self, x: np.ndarray) -> np.ndarray:
        linear = self.w0 + x @ self.w

        xv = x @ self.v
        x2v2 = (x * x) @ (self.v * self.v)
        interactions = 0.5 * np.sum(xv * xv - x2v2, axis=1)

        return linear + interactions

    def mse(self, x: np.ndarray, y: np.ndarray) -> float:
        pred = self.predict(x)
        return float(np.mean((y - pred) ** 2))

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        epochs: int = 30,
        lr: float = 0.02,
        reg_w: float = 1e-4,
        reg_v: float = 1e-4,
    ) -> list[float]:
        n_samples = x.shape[0]
        history: list[float] = []

        for _ in range(epochs):
            order = np.arange(n_samples)
            np.random.shuffle(order)

            for idx in order:
                xi = x[idx]
                yi = float(y[idx])

                # 予測
                linear = self.w0 + float(np.dot(self.w, xi))
                s = xi @ self.v
                interactions = 0.5 * float(np.sum(s * s - (xi * xi) @ (self.v * self.v)))
                pred = linear + interactions

                err = yi - pred

                # バイアス更新
                self.w0 += lr * err

                # 1次項更新
                self.w += lr * (err * xi - reg_w * self.w)

                # 2次項更新
                v_old = self.v.copy()
                for f in range(v_old.shape[1]):
                    sf = float(np.dot(xi, v_old[:, f]))
                    grad_vf = xi * (sf - v_old[:, f] * xi)
                    self.v[:, f] += lr * (err * grad_vf - reg_v * v_old[:, f])

            history.append(self.mse(x, y))

        return history

    def recommend(
        self,
        user_index: int,
        num_users: int,
        num_items: int,
        seen_item_indices: np.ndarray,
        top_k: int = 5,
    ) -> list[int]:
        x = np.zeros((num_items, num_users + num_items), dtype=np.float64)
        x[:, user_index] = 1.0
        x[np.arange(num_items), num_users + np.arange(num_items)] = 1.0

        scores = self.predict(x)
        seen = np.asarray(seen_item_indices, dtype=np.int64)
        if seen.size > 0:
            scores[seen] = -np.inf

        ranked = np.argsort(scores)[::-1]
        return ranked[:top_k].tolist()
