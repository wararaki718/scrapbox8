from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class DPPResult:
    selected_indices: list[int]
    kernel: np.ndarray


def validate_l_kernel(l_kernel: np.ndarray, atol: float = 1e-8) -> None:
    """Validate that the input is a symmetric L-ensemble kernel."""
    if l_kernel.ndim != 2 or l_kernel.shape[0] != l_kernel.shape[1]:
        raise ValueError("l_kernel must be a square matrix")
    if not np.allclose(l_kernel, l_kernel.T, atol=atol, rtol=0.0):
        raise ValueError("l_kernel must be symmetric")


def build_psd_kernel(features: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Build a positive semidefinite L-ensemble kernel from feature vectors."""
    if features.ndim != 2:
        raise ValueError("features must be 2D (n_items, n_features)")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")

    scaled = features / np.sqrt(temperature)
    return scaled @ scaled.T


def generate_demo_kernel(
    n_items: int,
    n_features: int,
    temperature: float,
    seed: int,
) -> tuple[np.random.Generator, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = rng.normal(loc=0.0, scale=1.0, size=(n_items, n_features))
    l_kernel = build_psd_kernel(features=features, temperature=temperature)
    return rng, l_kernel


def load_matrix(matrix_path: str | Path) -> np.ndarray:
    """Load a 2D matrix from a .npy or delimited text file."""
    path = Path(matrix_path)
    suffix = path.suffix.lower()

    if suffix == ".npy":
        matrix = np.load(path)
    elif suffix in {".csv", ".txt", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        matrix = np.loadtxt(path, delimiter=delimiter)
    else:
        raise ValueError("matrix path must end with .npy, .csv, .tsv, or .txt")

    if matrix.ndim != 2:
        raise ValueError("loaded matrix must be 2D")
    return matrix
