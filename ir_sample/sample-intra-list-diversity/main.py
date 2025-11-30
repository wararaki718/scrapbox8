import torch

from metrics import compute_intra_list_diversity


def main() -> None:
    # high diversity embeddings
    low_diversity_embeddings = torch.tensor([
        [1.0, 2.0, 3.0],
        [1.1, 2.1, 3.1],
        [0.9, 1.9, 2.9],
    ])
    ild_score = compute_intra_list_diversity(low_diversity_embeddings)
    print(f"Intra-list Diversity (low-diversity): {ild_score}")

    # low diversity embeddings
    high_diversity_embeddings = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    ild_score = compute_intra_list_diversity(high_diversity_embeddings)
    print(f"Intra-list Diversity (high-diversity) : {ild_score}")
    print("DONE")

if __name__ == "__main__":
    main()
