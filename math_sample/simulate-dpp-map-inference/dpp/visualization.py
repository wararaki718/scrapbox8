"""Visualization and animation helpers for DPP MAP inference."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.figure import Figure


sns.set_theme(style="whitegrid", context="talk")

CHOL_CMAP = sns.color_palette("blend:#f7fbff,#6baed6,#08306b", as_cmap=True)
RANK1_CMAP = sns.color_palette("blend:#fff5f0,#fb6a4a,#67000d", as_cmap=True)


def render_dpp_frame(
    frame_idx: int,
    trace: list[dict],
    points: np.ndarray,
    ax_scatter: Axes,
    ax_bar: Axes,
    ax_chol: Axes,
    ax_rank1: Axes,
    fig: Figure,
    min_gain: float,
    max_gain: float,
) -> None:
    """Render one animation frame for a given DPP trace step.

    Args:
        frame_idx: Frame index corresponding to a trace step.
        trace: Per-step trace returned by greedy_map_with_trace.
        points: Point array of shape (N, 2) for geometric visualization.
        ax_scatter: Axis for point scatter plot.
        ax_bar: Axis for marginal-gain bar chart.
        ax_chol: Axis for Cholesky matrix heatmap.
        ax_rank1: Axis for residual matrix heatmap.
        fig: Figure object to update title and layout.
        min_gain: Global minimum gain across trace.
        max_gain: Global maximum gain across trace.
    """
    ax_scatter.clear()
    ax_bar.clear()
    ax_chol.clear()
    ax_rank1.clear()

    step_data = trace[frame_idx]
    selected = set(step_data["selected"])
    best_i = step_data["best_i"]
    gains = step_data["gains"]

    idx_all = np.arange(len(points))
    idx_sel = np.array([i for i in idx_all if i in selected], dtype=int)
    idx_non = np.array([i for i in idx_all if i not in selected], dtype=int)

    ax_scatter.scatter(
        points[idx_non, 0],
        points[idx_non, 1],
        s=70,
        alpha=0.65,
        c="#7f8c8d",
        label="not selected",
    )
    ax_scatter.scatter(
        points[idx_sel, 0],
        points[idx_sel, 1],
        s=140,
        alpha=0.95,
        c="#2c3e50",
        marker="o",
        label="selected",
    )
    ax_scatter.scatter(
        points[best_i, 0],
        points[best_i, 1],
        s=240,
        c="#e74c3c",
        marker="*",
        edgecolor="black",
        linewidth=0.8,
        label="chosen at this step",
    )

    for i, (x, y) in enumerate(points):
        if i in selected:
            ax_scatter.text(x + 0.01, y + 0.01, str(i), fontsize=9, color="#1b2631")

    ax_scatter.set_xlim(0, 1)
    ax_scatter.set_ylim(0, 1)
    ax_scatter.set_title(
        f"DPP Greedy MAP Step {step_data['step']}\n"
        f"objective = {step_data['objective']:.3f}, best gain = {step_data['best_gain']:.3f}"
    )
    ax_scatter.legend(loc="upper right", fontsize=9)

    valid_idx = np.where(~np.isnan(gains))[0]
    bar_colors = ["#95a5a6" for _ in valid_idx]
    if best_i in valid_idx:
        best_pos = list(valid_idx).index(best_i)
        bar_colors[best_pos] = "#e74c3c"

    ax_bar.bar(valid_idx, gains[valid_idx], color=bar_colors)
    ax_bar.axhline(0, color="black", linewidth=1)
    ax_bar.set_ylim(min(-0.05, min_gain * 1.15), max_gain * 1.15)
    ax_bar.set_title("Marginal gain: $\\Delta_i = \\log\\det(L_{S\\cup\\{i\\}})-\\log\\det(L_S)$")
    ax_bar.set_xlabel("candidate index")
    ax_bar.set_ylabel("gain")

    chol_sub = step_data["chol_sub"]
    sns.heatmap(
        chol_sub,
        ax=ax_chol,
        cmap=CHOL_CMAP,
        cbar=False,
        square=True,
        xticklabels=False,
        yticklabels=False,
    )
    ax_chol.set_title(
        f"Cholesky of selected submatrix: chol(L_S)\n"
        f"S size = {len(step_data['selected'])}"
    )

    rank1_mat = step_data["rank1_matrix"]
    if rank1_mat.size == 0:
        ax_rank1.text(0.5, 0.5, "No remaining items", ha="center", va="center", fontsize=12)
        ax_rank1.set_title("Rank-1 updated residual matrix")
        ax_rank1.set_axis_off()
    else:
        sns.heatmap(
            rank1_mat,
            ax=ax_rank1,
            cmap=RANK1_CMAP,
            cbar=False,
            square=True,
            xticklabels=False,
            yticklabels=False,
        )
        ax_rank1.set_title(
            "Residual after rank-1 update (Schur)\n"
            f"remaining size = {len(step_data['remaining_after'])}"
        )

    fig.suptitle("Greedy DPP MAP Inference: geometry, gain, Cholesky, and rank-1 updates", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])


def animate_dpp_trace(
    points: np.ndarray,
    trace: list[dict],
    out_dir: Path,
    stem: str = "dpp_map",
    interval_ms: int = 1000,
) -> Path:
    """Create and save DPP MAP trace animation as GIF.

    Args:
        points: Point array of shape (N, 2).
        trace: Per-step trace returned by greedy_map_with_trace.
        out_dir: Output directory for animation.
        stem: Output file stem.
        interval_ms: Frame interval in milliseconds.

    Returns:
        Path to the saved GIF file.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(16, 11))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.15, 1.0])
    ax_scatter = fig.add_subplot(grid[0, 0])
    ax_bar = fig.add_subplot(grid[0, 1])
    ax_chol = fig.add_subplot(grid[1, 0])
    ax_rank1 = fig.add_subplot(grid[1, 1])

    max_gain = max(np.nanmax(item["gains"]) for item in trace) if trace else 1.0
    min_gain = min(np.nanmin(item["gains"]) for item in trace) if trace else -0.1

    animation = FuncAnimation(
        fig,
        render_dpp_frame,
        frames=len(trace),
        interval=interval_ms,
        repeat=False,
        fargs=(trace, points, ax_scatter, ax_bar, ax_chol, ax_rank1, fig, min_gain, max_gain),
    )

    gif_path = out_dir / f"{stem}.gif"
    animation.save(gif_path, dpi=110, fps=1.0, writer="pillow")

    plt.close(fig)
    return gif_path
