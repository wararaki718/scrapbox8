from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from dpp import make_dpp_kernel
from mmr import mmr_select


def _mmr_select_trace(
    relevance: np.ndarray,
    sim: np.ndarray,
    k: int,
    lambda_rel: float,
) -> list[list[int]]:
    selected: list[int] = []
    candidates = set(range(len(relevance)))
    history: list[list[int]] = []

    for _ in range(min(k, len(relevance))):
        best_idx = None
        best_score = -np.inf

        for i in candidates:
            novelty_penalty = np.max(sim[i, selected]) if selected else 0.0
            score = lambda_rel * relevance[i] - (1.0 - lambda_rel) * novelty_penalty
            if score > best_score:
                best_score = score
                best_idx = i

        assert best_idx is not None
        selected.append(best_idx)
        candidates.remove(best_idx)
        history.append(selected.copy())

    return history


def _dpp_greedy_map_trace(l: np.ndarray, k: int) -> list[list[int]]:
    n = l.shape[0]
    cis = np.zeros((k, n))
    di2s = np.clip(np.diag(l).copy(), a_min=0.0, a_max=None)
    selected: list[int] = []
    history: list[list[int]] = []

    for it in range(min(k, n)):
        j = int(np.argmax(di2s))
        if di2s[j] <= 1e-12:
            break

        selected.append(j)
        history.append(selected.copy())

        if it == k - 1:
            break

        ci_opt = cis[:it, j]
        di_opt = np.sqrt(di2s[j])
        eis = (l[j, :] - ci_opt @ cis[:it, :]) / (di_opt + 1e-12)
        cis[it, :] = eis
        di2s = di2s - eis**2
        di2s[j] = -np.inf

    return history


def _plot_selection(
    x: np.ndarray,
    query: np.ndarray,
    selected: list[int],
    title: str,
    relevance: np.ndarray,
    ax: plt.Axes,
) -> None:
    ax.scatter(x[:, 0], x[:, 1], c=relevance, cmap="viridis", s=45, alpha=0.75, label="all items")

    if selected:
        s = np.array(selected)
        ax.scatter(
            x[s, 0],
            x[s, 1],
            facecolors="none",
            edgecolors="crimson",
            s=220,
            linewidths=2.0,
            label="selected",
        )
        for rank, idx in enumerate(selected, start=1):
            ax.text(x[idx, 0] + 0.05, x[idx, 1] + 0.05, str(rank), fontsize=9)

    ax.scatter(query[0], query[1], marker="*", s=300, c="black", label="query")
    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.legend(loc="best")


def export_selection_artifacts(
    x: np.ndarray,
    query: np.ndarray,
    relevance: np.ndarray,
    sim: np.ndarray,
    *,
    k: int,
    lambda_rel: float,
    quality_scale: float,
    output_dir: str = "outputs",
    gif_name: str = "selection_process.gif",
    fps: int = 1,
    interval_ms: int = 900,
) -> tuple[Path, int, list[int], list[int]]:
    """Export MMR/DPP selection process as GIF."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mmr_selected = mmr_select(relevance=relevance, sim=sim, k=k, lambda_rel=lambda_rel)
    l_kernel = make_dpp_kernel(relevance=relevance, sim=sim, quality_scale=quality_scale)
    dpp_history = _dpp_greedy_map_trace(l_kernel, k=k)
    mmr_history = _mmr_select_trace(relevance=relevance, sim=sim, k=k, lambda_rel=lambda_rel)

    num_frames = max(len(mmr_history), len(dpp_history))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

    def update(frame: int) -> None:
        for ax in axes:
            ax.clear()

        mmr_sel = mmr_history[min(frame, len(mmr_history) - 1)] if mmr_history else []
        dpp_sel = dpp_history[min(frame, len(dpp_history) - 1)] if dpp_history else []

        _plot_selection(
            x=x,
            query=query,
            selected=mmr_sel,
            title=f"MMR step {min(frame + 1, len(mmr_history))}/{k} (lambda={lambda_rel:.2f})",
            relevance=relevance,
            ax=axes[0],
        )
        _plot_selection(
            x=x,
            query=query,
            selected=dpp_sel,
            title=f"DPP step {min(frame + 1, len(dpp_history))}/{k} (quality_scale={quality_scale:.1f})",
            relevance=relevance,
            ax=axes[1],
        )
        fig.tight_layout()

    anim = FuncAnimation(fig, update, frames=num_frames, interval=interval_ms, repeat=True)
    gif_path = out_dir / gif_name
    anim.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)

    return gif_path, num_frames, mmr_selected, dpp_history[-1] if dpp_history else []
