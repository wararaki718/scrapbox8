from __future__ import annotations

from collections.abc import Sequence

import torch


def _logdet_psd(matrix: torch.Tensor, jitter: float) -> torch.Tensor:
    eye = torch.eye(matrix.size(0), device=matrix.device, dtype=matrix.dtype)
    sign, logabsdet = torch.linalg.slogdet(matrix + jitter * eye)
    if torch.any(sign <= 0):
        raise RuntimeError("Encountered non-positive determinant while computing logdet.")
    return logabsdet

# 集合尤度を最大化する（品質と多様性の同時最適化が入る）。
# - log det(L_A) + log det(L + I) + regularizer
# 要は、DeepDPP は「MLPで特徴を作る」＋「DPPで集合分布を定義する」の組み合わせ
def deep_dpp_loss(
    embeddings: torch.Tensor,
    baskets: Sequence[torch.Tensor],
    item_counts: torch.Tensor,
    alpha: float = 0.0,
    jitter: float = 1e-6,
) -> torch.Tensor:
    """Compute negative regularized DeepDPP objective for minimization."""
    # embeddings (V): (num_items, embedding_dim)
    kernel = embeddings @ embeddings.transpose(0, 1)
    # kernel (L): (num_items, num_items)

    num_items = kernel.size(0)
    identity = torch.eye(num_items, device=kernel.device, dtype=kernel.dtype)
    global_logdet = _logdet_psd(kernel + identity, jitter=0.0)

    subset_logdets = torch.tensor(0.0, device=kernel.device, dtype=kernel.dtype)
    for basket in baskets:
        if basket.numel() == 0:
            continue
        sub_kernel = kernel.index_select(0, basket).index_select(1, basket)
        subset_logdets = subset_logdets + _logdet_psd(sub_kernel, jitter=jitter)

    # Popularity-aware regularizer from the paper.
    inv_counts = torch.reciprocal(item_counts.clamp_min(1.0)).unsqueeze(1)
    row_norm_sq = embeddings.pow(2).sum(dim=1, keepdim=True)
    regularizer = (inv_counts * row_norm_sq).sum()

    num_baskets = float(len(baskets))
    objective = subset_logdets - num_baskets * global_logdet - alpha * regularizer
    return -objective


def next_item_scores(
    embeddings: torch.Tensor,
    observed_items: torch.Tensor,
    jitter: float = 1e-6,
) -> torch.Tensor:
    """Compute DPP Schur-complement gains for next-item prediction."""
    # embeddings: (num_items, embedding_dim)
    kernel = embeddings @ embeddings.transpose(0, 1)
    # kernel: (num_items, num_items)

    num_items = embeddings.size(0)
    scores = torch.diag(kernel).clone()

    if observed_items.numel() == 0:
        return scores

    l_aa = kernel.index_select(0, observed_items).index_select(1, observed_items)
    eye_obs = torch.eye(l_aa.size(0), device=l_aa.device, dtype=l_aa.dtype)
    l_aa_inv = torch.linalg.inv(l_aa + jitter * eye_obs)

    for i in range(num_items):
        if torch.any(observed_items == i):
            scores[i] = float("-inf")
            continue
        l_i_a = kernel[i, observed_items].unsqueeze(0)
        gain = kernel[i, i] - l_i_a @ l_aa_inv @ l_i_a.transpose(0, 1)
        scores[i] = gain.squeeze()

    return scores
