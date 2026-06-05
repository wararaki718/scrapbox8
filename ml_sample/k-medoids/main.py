from __future__ import annotations

from k_medoids import k_medoids
from utils import build_sample_points, format_clusters


def main() -> None:
    seed = 42
    points_array = build_sample_points(seed=seed)
    k = 5

    medoid_indices, assignments, cost = k_medoids(points_array, k=k, seed=seed)

    print("k-medoids result")
    print(f"k={k}")
    print(f"medoid indices={medoid_indices}")
    print(f"total cost={cost:.4f}")
    print("clusters:")
    print(format_clusters(points_array, assignments, medoid_indices))


if __name__ == "__main__":
    main()
