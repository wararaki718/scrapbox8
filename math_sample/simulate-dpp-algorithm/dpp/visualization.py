"""Visualization helpers for DPP simulation outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap


Array = np.ndarray


def _sequential_gradient_cmap(name: str = "dpp_seq") -> LinearSegmentedColormap:
    """Build a smooth sequential gradient colormap."""
    return LinearSegmentedColormap.from_list(
        name,
        ["#f7fbff", "#6baed6", "#08519c"],
        N=256,
    )


def _diverging_gradient_cmap(name: str = "dpp_div") -> LinearSegmentedColormap:
    """Build a smooth diverging gradient colormap."""
    return LinearSegmentedColormap.from_list(
        name,
        ["#313695", "#74add1", "#ffffbf", "#f46d43", "#a50026"],
        N=256,
    )


def _gains_gradient_cmap(name: str = "dpp_gains") -> LinearSegmentedColormap:
    """Build a smooth gradient colormap for greedy gains."""
    return LinearSegmentedColormap.from_list(
        name,
        ["#f7fcf5", "#c7e9c0", "#41ab5d", "#00441b"],
        N=256,
    )


def _inverse_gradient_cmap(name: str = "dpp_inv") -> LinearSegmentedColormap:
    """Build a smooth gradient colormap for inverse matrix values."""
    return LinearSegmentedColormap.from_list(
        name,
        ["#fff5f0", "#fcbba1", "#fb6a4a", "#a50f15"],
        N=256,
    )


def save_matrix_heatmap(
    matrix: Array,
    output_path: str | Path,
    title: str,
    cmap: str | LinearSegmentedColormap = "auto",
    annotate: bool = True,
    fmt: str = ".2f",
) -> Path:
    """Save a matrix heatmap image to local storage.

    Args:
        matrix: Target matrix to visualize.
        output_path: Destination image path (e.g., PNG).
        title: Figure title.
        cmap: Matplotlib colormap name or colormap instance. Use "auto" for default gradient.
        annotate: If True, draw each matrix element value on the heatmap.
        fmt: Numeric format string used for annotations.

    Returns:
        Resolved path to the saved image file.
    """
    if matrix.ndim != 2:
        raise ValueError("matrix must be 2-dimensional")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if cmap == "auto":
        cmap = _sequential_gradient_cmap()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("column index")
    ax.set_ylabel("row index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if annotate:
        norm = plt.Normalize(vmin=float(np.min(matrix)), vmax=float(np.max(matrix)))
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                text_color = "black" if norm(value) > 0.55 else "white"
                ax.text(j, i, format(float(value), fmt), ha="center", va="center", color=text_color, fontsize=6)

    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)

    return output.resolve()


def save_assumption_gif(
    features: Array,
    qualities: Array,
    sampled_subsets: Iterable[list[int]],
    output_path: str | Path,
    fps: int = 2,
) -> Path:
    """Visualize simulation assumptions and save as GIF.

    The animation shows:
    - Candidate items as 2D points (feature space assumption).
    - Marker size proportional to quality score (quality assumption).
    - Highlighted sampled subset for each frame (diversity outcome).

    Args:
        features: 2D coordinates for plotting.
        qualities: Quality score for each item.
        sampled_subsets: Sequence of sampled subsets.
        output_path: GIF destination path.
        fps: Frames per second.

    Returns:
        Resolved path to the saved GIF file.
    """
    if features.shape[1] < 2:
        raise ValueError("features must have at least 2 dimensions for visualization")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    pts = features[:, :2]
    q = qualities / np.max(qualities)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_title("DPP Simulation Assumptions")
    ax.set_xlabel("Feature axis 1")
    ax.set_ylabel("Feature axis 2")

    x_pad = 0.8
    y_pad = 0.8
    ax.set_xlim(float(np.min(pts[:, 0]) - x_pad), float(np.max(pts[:, 0]) + x_pad))
    ax.set_ylim(float(np.min(pts[:, 1]) - y_pad), float(np.max(pts[:, 1]) + y_pad))

    sizes = 220.0 * (0.2 + q)

    base = ax.scatter(
        pts[:, 0],
        pts[:, 1],
        s=sizes,
        c="#8fb9a8",
        alpha=0.45,
        edgecolors="#203a2a",
        linewidths=0.8,
        label="Candidates (size = quality)",
    )

    highlight = ax.scatter(
        [],
        [],
        s=[],
        c="#ef6f6c",
        alpha=0.95,
        edgecolors="#5d1614",
        linewidths=1.0,
        label="Sampled by DPP",
    )

    ax.legend(loc="upper right")
    frame_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, va="top")
    subset_list = list(sampled_subsets)

    def _update(frame: int):
        subset = subset_list[frame]
        if subset:
            subset_pts = pts[subset]
            highlight.set_offsets(subset_pts)
            highlight.set_sizes(sizes[subset])
        else:
            highlight.set_offsets(np.empty((0, 2)))
            highlight.set_sizes([])

        frame_text.set_text(
            f"Frame {frame + 1}/{len(subset_list)} | subset={subset}"
        )
        return base, highlight, frame_text

    anim = FuncAnimation(
        fig,
        _update,
        frames=len(subset_list),
        interval=int(1000 / max(1, fps)),
        blit=False,
        repeat=False,
    )
    anim.save(output, writer=PillowWriter(fps=fps))
    plt.close(fig)

    return output.resolve()


def save_map_inference_trace_gif(
    features: Array,
    qualities: Array,
    map_history: Iterable[dict[str, object]],
    output_path: str | Path,
    fps: int = 1,
) -> Path:
    """Save a single GIF that tracks MAP inference and matrix evolution.

    Frame contents:
    - Candidate points, highlighting selected items.
    - Conditional kernel heatmap after each rank-1 update.
    - Gain vector heatmap used for greedy MAP selection.
    - Selected submatrix $L_{Y,Y}$ heatmap.
    - Updated inverse $(L_{Y,Y})^{-1}$ heatmap.
    - Extracted candidate matrix $C_{R,R}$ for remaining items.

    Args:
        features: Candidate features. First two dims are used for scatter.
        qualities: Quality scores for marker sizes.
        map_history: Step history from map_inference_rank1_updates.
        output_path: Destination GIF path.
        fps: Frames per second.

    Returns:
        Resolved path to the saved GIF.
    """
    if features.shape[1] < 2:
        raise ValueError("features must have at least 2 dimensions for visualization")

    history = list(map_history)
    if not history:
        raise ValueError("map_history is empty")

    for frame in history:
        if "selected_kernel" not in frame or "inv_selected_kernel" not in frame:
            raise ValueError(
                "map_history frames must include selected_kernel and inv_selected_kernel"
            )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    pts = features[:, :2]
    q = qualities / np.max(qualities)
    sizes = 220.0 * (0.2 + q)
    n_items = pts.shape[0]

    extracted_list: list[Array] = []
    remaining_sizes: list[int] = []
    for frame in history:
        selected = list(frame["selected"])
        cond = np.asarray(frame["conditional_kernel"], dtype=float)
        remaining = [i for i in range(n_items) if i not in selected]
        remaining_sizes.append(len(remaining))
        if remaining:
            extracted_list.append(cond[np.ix_(remaining, remaining)])
        else:
            extracted_list.append(np.zeros((0, 0), dtype=float))

    cond_min = min(float(np.min(frame["conditional_kernel"])) for frame in history)
    cond_max = max(float(np.max(frame["conditional_kernel"])) for frame in history)
    gain_min = min(
        float(np.min(frame["gains"][np.isfinite(frame["gains"])]))
        for frame in history
    )
    gain_max = max(
        float(np.max(frame["gains"][np.isfinite(frame["gains"])]))
        for frame in history
    )

    extracted_values = [mat for mat in extracted_list if mat.size > 0]
    ext_min = min(float(np.min(mat)) for mat in extracted_values)
    ext_max = max(float(np.max(mat)) for mat in extracted_values)

    fig, axes = plt.subplots(3, 2, figsize=(14, 18), constrained_layout=True)
    ax_scatter = axes[0, 0]
    ax_cond = axes[0, 1]
    ax_gain = axes[1, 0]
    ax_sel = axes[1, 1]
    ax_inv = axes[2, 0]
    ax_ext = axes[2, 1]

    kmax = max(len(frame["selected"]) for frame in history)
    rmax = max(remaining_sizes)

    def _pad_square(matrix: Array, target_size: int) -> Array:
        """Pad a square matrix to a fixed size for stable animation axes."""
        out = np.full((target_size, target_size), np.nan, dtype=float)
        if matrix.size == 0:
            return out
        k = matrix.shape[0]
        out[:k, :k] = matrix
        return out

    ax_scatter.set_title("MAP Inference (Selected Items)")
    ax_scatter.set_xlabel("Feature axis 1")
    ax_scatter.set_ylabel("Feature axis 2")
    x_pad = 0.8
    y_pad = 0.8
    ax_scatter.set_xlim(float(np.min(pts[:, 0]) - x_pad), float(np.max(pts[:, 0]) + x_pad))
    ax_scatter.set_ylim(float(np.min(pts[:, 1]) - y_pad), float(np.max(pts[:, 1]) + y_pad))

    ax_scatter.scatter(
        pts[:, 0],
        pts[:, 1],
        s=sizes,
        c="#adc2d1",
        alpha=0.45,
        edgecolors="#1e3140",
        linewidths=0.8,
        label="Candidates",
    )
    selected_plot = ax_scatter.scatter(
        [],
        [],
        s=[],
        c="#e85d75",
        alpha=0.95,
        edgecolors="#5b1520",
        linewidths=1.0,
        label="Selected",
    )
    ax_scatter.legend(loc="upper right")

    first_cond = np.asarray(history[0]["conditional_kernel"], dtype=float)
    first_gain = np.asarray(history[0]["gains"], dtype=float).reshape(1, -1)
    gain_init = np.where(np.isfinite(first_gain), first_gain, np.nan)
    first_sel = _pad_square(np.asarray(history[0]["selected_kernel"], dtype=float), kmax)
    first_inv = _pad_square(np.asarray(history[0]["inv_selected_kernel"], dtype=float), kmax)
    first_ext = _pad_square(extracted_list[0], rmax)

    cond_im = ax_cond.imshow(
        first_cond,
        cmap=_sequential_gradient_cmap("cond_seq"),
        vmin=cond_min,
        vmax=cond_max,
        aspect="auto",
    )
    ax_cond.set_title("Conditional Kernel C")
    ax_cond.set_xlabel("column")
    ax_cond.set_ylabel("row")
    fig.colorbar(cond_im, ax=ax_cond, fraction=0.046, pad=0.04)

    gain_im = ax_gain.imshow(
        gain_init,
        cmap=_gains_gradient_cmap("gain_grad"),
        vmin=gain_min,
        vmax=gain_max,
        aspect="auto",
    )
    ax_gain.set_title("Greedy Gains diag(C)")
    ax_gain.set_xlabel("item index")
    ax_gain.set_yticks([])
    fig.colorbar(gain_im, ax=ax_gain, fraction=0.046, pad=0.04)

    sel_values = [
        np.asarray(frame["selected_kernel"], dtype=float)
        for frame in history
        if np.asarray(frame["selected_kernel"], dtype=float).size > 0
    ]
    sel_min = min(float(np.min(mat)) for mat in sel_values)
    sel_max = max(float(np.max(mat)) for mat in sel_values)
    sel_im = ax_sel.imshow(
        first_sel,
        cmap=_sequential_gradient_cmap("sel_seq"),
        vmin=sel_min,
        vmax=sel_max,
        aspect="auto",
    )
    ax_sel.set_title("Selected Submatrix L_YY")
    ax_sel.set_xlabel("selected index")
    ax_sel.set_ylabel("selected index")
    fig.colorbar(sel_im, ax=ax_sel, fraction=0.046, pad=0.04)

    inv_values = [
        np.asarray(frame["inv_selected_kernel"], dtype=float)
        for frame in history
        if np.asarray(frame["inv_selected_kernel"], dtype=float).size > 0
    ]
    inv_min = min(float(np.min(mat)) for mat in inv_values)
    inv_max = max(float(np.max(mat)) for mat in inv_values)
    inv_im = ax_inv.imshow(
        first_inv,
        cmap=_inverse_gradient_cmap("inv_grad"),
        vmin=inv_min,
        vmax=inv_max,
        aspect="auto",
    )
    ax_inv.set_title("Updated Inverse (L_YY)^(-1)")
    ax_inv.set_xlabel("selected index")
    ax_inv.set_ylabel("selected index")
    fig.colorbar(inv_im, ax=ax_inv, fraction=0.046, pad=0.04)

    ext_im = ax_ext.imshow(
        first_ext,
        cmap=_sequential_gradient_cmap("ext_seq"),
        vmin=ext_min,
        vmax=ext_max,
        aspect="auto",
    )
    ax_ext.set_title("Extracted Candidate Matrix C_RR")
    ax_ext.set_xlabel("remaining index")
    ax_ext.set_ylabel("remaining index")
    fig.colorbar(ext_im, ax=ax_ext, fraction=0.046, pad=0.04)

    frame_text = fig.text(0.02, 0.98, "", va="top")

    def _update(frame_idx: int):
        frame = history[frame_idx]
        selected = list(frame["selected"])
        gains = np.asarray(frame["gains"], dtype=float)
        cond = np.asarray(frame["conditional_kernel"], dtype=float)
        remaining = [i for i in range(n_items) if i not in selected]
        if remaining:
            extracted = cond[np.ix_(remaining, remaining)]
        else:
            extracted = np.zeros((0, 0), dtype=float)

        sel = _pad_square(np.asarray(frame["selected_kernel"], dtype=float), kmax)
        inv = _pad_square(np.asarray(frame["inv_selected_kernel"], dtype=float), kmax)
        ext = _pad_square(extracted, rmax)
        chosen = int(frame["chosen"])
        schur = float(frame["schur"])

        if selected:
            selected_pts = pts[selected]
            selected_plot.set_offsets(selected_pts)
            selected_plot.set_sizes(sizes[selected])
        else:
            selected_plot.set_offsets(np.empty((0, 2)))
            selected_plot.set_sizes([])

        cond_im.set_data(cond)
        gain_im.set_data(np.where(np.isfinite(gains), gains, np.nan).reshape(1, -1))
        sel_im.set_data(sel)
        inv_im.set_data(inv)
        ext_im.set_data(ext)

        frame_text.set_text(
            "Step "
            f"{int(frame['step'])}/{len(history)} | "
            f"chosen={chosen} | schur={schur:.4e} | selected={selected}"
        )
        return selected_plot, cond_im, gain_im, sel_im, inv_im, ext_im, frame_text

    anim = FuncAnimation(
        fig,
        _update,
        frames=len(history),
        interval=int(1000 / max(1, fps)),
        blit=False,
        repeat=False,
    )
    anim.save(output, writer=PillowWriter(fps=fps))
    plt.close(fig)

    return output.resolve()
