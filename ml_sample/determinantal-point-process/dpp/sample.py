from __future__ import annotations

import numpy as np

from .common import validate_l_kernel


def sample_dpp(l_kernel: np.ndarray, rng: np.random.Generator) -> list[int]:
    """Sample a subset from a DPP defined by an L-ensemble kernel."""
    validate_l_kernel(l_kernel)

    # Step 1: L = V diag(lambda) V^T を固有値分解する。
    eigvals, eigvecs = np.linalg.eigh(l_kernel)
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)

    # Step 2: 各固有ベクトルを lambda/(1+lambda) で Bernoulli 選択する。
    # これにより、以降の逐次サンプリングで使う部分空間 V を決める。
    include_prob = eigvals / (1.0 + eigvals)
    keep_mask = rng.random(eigvals.shape[0]) < include_prob
    v = eigvecs[:, keep_mask]

    selected: list[int] = []

    # Step 3: V が空になるまでアイテムを逐次サンプリングする。
    # 注: ここは「サンプリング」であり MAP 推論（argmax det(L_Y)）とは別問題。
    while v.shape[1] > 0:
        # 各アイテム i の選択確率は、V の i 行ノルム二乗に比例。
        row_norm_sq = np.sum(v * v, axis=1)
        total = float(row_norm_sq.sum())
        if total <= 1e-15:
            break

        probs = row_norm_sq / total
        item = int(rng.choice(v.shape[0], p=probs))
        selected.append(item)

        # 選ばれた item に対応する方向を消去するための pivot 列を選ぶ。
        pivot_row = v[item, :]
        pivot_idx = int(np.argmax(np.abs(pivot_row)))
        denom = v[item, pivot_idx]
        if np.abs(denom) <= 1e-12:
            break

        # pivot 列で張られる成分を落とし、残りを再直交化して次反復へ。
        basis = np.delete(v, pivot_idx, axis=1)
        correction = np.outer(v[:, pivot_idx], pivot_row / denom)
        basis = basis - np.delete(correction, pivot_idx, axis=1)

        if basis.size == 0:
            break
        q, _ = np.linalg.qr(basis)
        v = q

    selected.sort()
    return selected
