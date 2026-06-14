from __future__ import annotations

import numpy as np

from dpp import DPPParams


def build_demo_inputs() -> tuple[np.ndarray, np.ndarray, DPPParams]:
    qualities = np.array(
        [1.00, 0.98, 0.96, 0.93, 0.90, 0.88, 0.84, 0.80, 0.77, 0.73],
        dtype=np.float64,
    )
    features = np.array(
        [
            [1, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 1, 0, 1, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 1, 1],
            [0, 1, 0, 1, 1, 0, 1, 0],
            [1, 1, 0, 0, 1, 0, 0, 1],
            [0, 0, 1, 1, 0, 1, 1, 0],
            [1, 0, 0, 1, 1, 0, 1, 0],
            [0, 1, 1, 0, 0, 1, 0, 1],
            [1, 1, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1, 1, 0, 1],
        ],
        dtype=np.int64,
    )
    params = DPPParams(alpha=0.95, sigma=0.25)
    return qualities, features, params
