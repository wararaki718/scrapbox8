from __future__ import annotations

import argparse

import numpy as np

from dpp import load_matrix, run_demo, run_dpp, run_from_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Determinantal Point Process sampling or MAP inference")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["sample", "map"],
        default="sample",
        help="Execution mode: stochastic sampling or greedy MAP inference",
    )
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--features-path",
        type=str,
        default=None,
        help="Path to a 2D feature matrix (.npy, .csv, .tsv, .txt)",
    )
    input_group.add_argument(
        "--kernel-path",
        type=str,
        default=None,
        help="Path to a symmetric L-kernel matrix (.npy, .csv, .tsv, .txt)",
    )
    parser.add_argument("--n-items", type=int, default=20, help="Number of candidate items")
    parser.add_argument("--n-features", type=int, default=8, help="Feature dimension per item")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Kernel temperature scaling (higher -> flatter kernel)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Max subset size (mainly for map mode)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if args.features_path is not None:
        features = load_matrix(args.features_path)
        result = run_from_features(
            features=features,
            mode=args.mode,
            rng=rng,
            max_length=args.max_length,
            temperature=args.temperature,
        )
        input_summary = f"features_path={args.features_path}, n_items={features.shape[0]}, n_features={features.shape[1]}"
    elif args.kernel_path is not None:
        kernel = load_matrix(args.kernel_path)
        result = run_dpp(
            l_kernel=kernel,
            mode=args.mode,
            rng=rng,
            max_length=args.max_length,
        )
        input_summary = f"kernel_path={args.kernel_path}, n_items={kernel.shape[0]}"
    else:
        result = run_demo(
            n_items=args.n_items,
            n_features=args.n_features,
            temperature=args.temperature,
            seed=args.seed,
            mode=args.mode,
            max_length=args.max_length,
        )
        input_summary = f"demo_input=True, n_items={args.n_items}, n_features={args.n_features}"

    print("=== Determinantal Point Process ===")
    print(
        f"mode={args.mode}, {input_summary}, temperature={args.temperature}, "
        f"max_length={args.max_length}"
    )
    print(f"selected_count={len(result.selected_indices)}")
    print(f"selected_indices={result.selected_indices}")


if __name__ == "__main__":
    main()
