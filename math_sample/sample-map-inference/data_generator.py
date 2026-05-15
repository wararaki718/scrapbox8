from __future__ import annotations

from numpy.random import default_rng
import numpy as np
from numpy.typing import NDArray


def generate_linear_regression_data(
    n_samples: int,
    n_features: int,
    noise_std: float,
    seed: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], float]:
    """Generate synthetic data for linear regression.

    Returns feature matrix X, target vector y, true weight vector w, and true bias b.
    """
    rng = default_rng(seed)

    x = rng.normal(loc=0.0, scale=1.0, size=(n_samples, n_features))
    true_w = rng.normal(loc=0.0, scale=1.0, size=n_features)
    true_b = float(rng.normal(loc=0.0, scale=0.5))

    noise = rng.normal(loc=0.0, scale=noise_std, size=n_samples)
    y = x @ true_w + true_b + noise

    return x, y, true_w, true_b
