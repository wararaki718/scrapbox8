import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).parent))

from dpp import DPPParams, build_kernel, jaccard_distance_matrix, rank_feed_via_dpp
import main
import utils


def test_jaccard_distance_matrix_basic() -> None:
    features = np.array(
        [
            [1, 0, 1],
            [1, 1, 0],
            [0, 0, 1],
        ],
        dtype=np.int64,
    )

    distances = jaccard_distance_matrix(features)

    assert distances.shape == (3, 3)
    assert np.allclose(np.diag(distances), 0.0)
    assert np.isclose(distances[0, 1], 2.0 / 3.0)
    assert np.isclose(distances[0, 2], 0.5)


def test_build_kernel_is_symmetric_with_quality_diagonal() -> None:
    qualities = np.array([0.9, 0.8, 0.7], dtype=np.float64)
    distances = np.array(
        [
            [0.0, 1.0, 0.8],
            [1.0, 0.0, 0.2],
            [0.8, 0.2, 0.0],
        ],
        dtype=np.float64,
    )
    params = DPPParams(alpha=0.6, sigma=0.5)

    kernel = build_kernel(qualities=qualities, distances=distances, params=params)

    assert kernel.shape == (3, 3)
    assert np.allclose(kernel, kernel.T)
    assert np.allclose(np.diag(kernel), qualities**2, atol=1e-8)
    eigenvalues = np.linalg.eigvalsh(kernel)
    assert np.min(eigenvalues) >= -1e-9


def test_rank_feed_via_dpp_prefers_diverse_pair() -> None:
    qualities = np.array([1.0, 0.95, 0.75], dtype=np.float64)
    features = np.array(
        [
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ],
        dtype=np.int64,
    )
    params = DPPParams(alpha=0.95, sigma=0.2)

    ranked = rank_feed_via_dpp(
        qualities=qualities,
        features=features,
        params=params,
        window_size=2,
    )

    assert len(ranked) == 3
    assert set(ranked) == {0, 1, 2}
    assert set(ranked[:2]) == {0, 2}


def test_main_has_entrypoint_function() -> None:
    assert hasattr(main, "main")


def test_utils_has_sample_data_builder() -> None:
    assert hasattr(utils, "build_demo_inputs")


def test_demo_inputs_are_larger_than_minimum_size() -> None:
    qualities, features, params = utils.build_demo_inputs()

    assert qualities.shape[0] >= 8
    assert features.shape[0] >= 8
    assert features.shape[1] >= 6
    assert qualities.shape[0] == features.shape[0]
    assert params.sigma > 0


def test_main_uses_adaptive_window_size_policy() -> None:
    assert main.choose_window_size(10) == 5
    assert main.choose_window_size(3) == 3
    assert main.choose_window_size(0) == 1
