import torch

from metrics import compute_intra_list_average_distances, compute_intra_list_minimal_distances


def main() -> None:
    # high diversity embeddings
    low_diversity_embeddings = torch.tensor([
        [1.0, 2.0, 3.0],
        [1.1, 2.1, 3.1],
        [0.9, 1.9, 2.9],
    ])
    # low diversity embeddings
    high_diversity_embeddings = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])

    ilad_score = compute_intra_list_average_distances(low_diversity_embeddings)
    print(f"Intra-list Average Diversity (low-diversity): {ilad_score}")

    ilad_score = compute_intra_list_average_distances(high_diversity_embeddings)
    print(f"Intra-list Average Diversity (high-diversity): {ilad_score}")
    print()

    ilmd_score = compute_intra_list_minimal_distances(low_diversity_embeddings)
    print(f"Intra-list Minimal Distances (low-diversity): {ilmd_score}")

    ilmd_score = compute_intra_list_minimal_distances(high_diversity_embeddings)
    print(f"Intra-list Minimal Distances (high-diversity): {ilmd_score}")
    print("DONE")


if __name__ == "__main__":
    main()
