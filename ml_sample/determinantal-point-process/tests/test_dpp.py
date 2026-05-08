from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np

from dpp import load_matrix, map_inference_dpp, run_dpp, run_from_features, sample_dpp


def brute_force_best_subset(l_kernel: np.ndarray, max_length: int) -> list[int]:
    n_items = l_kernel.shape[0]
    best_subset: tuple[int, ...] = ()
    best_score = 0.0

    for subset_size in range(1, max_length + 1):
        for subset in combinations(range(n_items), subset_size):
            submatrix = l_kernel[np.ix_(subset, subset)]
            score = float(np.linalg.det(submatrix))
            if score > best_score:
                best_score = score
                best_subset = subset

    return list(best_subset)


def test_map_inference_matches_bruteforce_on_diagonal_kernel() -> None:
    l_kernel = np.diag(np.array([3.0, 2.0, 0.8, 0.4]))

    selected = map_inference_dpp(l_kernel, max_length=4)

    assert selected == brute_force_best_subset(l_kernel, max_length=4)
    assert selected == [0, 1]


def test_map_inference_caps_length_to_number_of_items() -> None:
    l_kernel = np.diag(np.array([4.0, 2.5, 0.7]))

    selected = map_inference_dpp(l_kernel, max_length=100)

    assert len(selected) <= l_kernel.shape[0]
    assert selected == [0, 1]


def test_map_inference_returns_empty_when_all_gains_are_non_positive() -> None:
    l_kernel = np.diag(np.array([0.9, 0.8, 0.3]))

    selected = map_inference_dpp(l_kernel, max_length=3)

    assert selected == []


def test_map_inference_matches_bruteforce_on_small_dense_kernel() -> None:
    features = np.array(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [0.1, 0.1],
            [0.0, 0.8],
        ]
    )
    l_kernel = features @ features.T

    selected = map_inference_dpp(l_kernel, max_length=2)

    assert selected == brute_force_best_subset(l_kernel, max_length=2)
    assert selected == [0, 1]


def test_sample_dpp_returns_unique_valid_indices() -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(6, 3))
    l_kernel = features @ features.T

    selected = sample_dpp(l_kernel, rng=np.random.default_rng(11))

    assert len(selected) == len(set(selected))
    assert all(0 <= index < l_kernel.shape[0] for index in selected)


def test_algorithms_reject_non_symmetric_kernel() -> None:
    l_kernel = np.array([[1.0, 2.0], [0.0, 1.0]])

    for fn in (map_inference_dpp, lambda kernel: sample_dpp(kernel, np.random.default_rng(0))):
        try:
            fn(l_kernel)
        except ValueError as exc:
            assert "symmetric" in str(exc)
        else:
            raise AssertionError("expected ValueError for non-symmetric kernel")


def test_load_matrix_supports_csv_and_npy(tmp_path: Path) -> None:
    matrix = np.array([[1.0, 2.0], [3.0, 4.0]])
    csv_path = tmp_path / "matrix.csv"
    npy_path = tmp_path / "matrix.npy"
    np.savetxt(csv_path, matrix, delimiter=",")
    np.save(npy_path, matrix)

    loaded_csv = load_matrix(csv_path)
    loaded_npy = load_matrix(npy_path)

    assert np.allclose(loaded_csv, matrix)
    assert np.allclose(loaded_npy, matrix)


def test_run_from_features_matches_run_dpp_on_same_kernel() -> None:
    features = np.array(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [0.1, 0.1],
            [0.0, 0.8],
        ]
    )
    l_kernel = features @ features.T

    result_from_features = run_from_features(
        features=features,
        mode="map",
        max_length=2,
        temperature=1.0,
    )
    result_from_kernel = run_dpp(l_kernel=l_kernel, mode="map", max_length=2)

    assert result_from_features.selected_indices == result_from_kernel.selected_indices
    assert np.allclose(result_from_features.kernel, l_kernel)