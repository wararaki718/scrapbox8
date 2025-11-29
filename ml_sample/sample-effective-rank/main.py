import torch

from metrics import compute_effective_rank


def main() -> None:
    batch_size = 200
    embedding_dim = 256
    high_diversity_embeddings = torch.randn(batch_size, embedding_dim)

    low_diversity_embeddings = torch.randn(batch_size, embedding_dim) * 0.01
    low_diversity_embeddings[:, :5] += torch.randn(batch_size, 5) * 10.0

    effective_rank_high = compute_effective_rank(high_diversity_embeddings)
    print(f"Effective Rank (high): {effective_rank_high}")

    effective_rank_low = compute_effective_rank(low_diversity_embeddings)
    print(f"Effective Rank (low) : {effective_rank_low}")

if __name__ == "__main__":
    main()
