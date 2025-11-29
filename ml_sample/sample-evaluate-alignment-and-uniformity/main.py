import torch

from metrics import compute_alignment, compute_uniformity


def main() -> None:
    batch_size = 200
    embedding_dim = 256
    document_embeddings = torch.randn(batch_size, embedding_dim)
    query_embeddings = torch.randn(batch_size, embedding_dim) * 0.01



    effective_rank_high = compute_alignment(query_embeddings, document_embeddings)
    print(f"alignment: {effective_rank_high}")

    effective_rank_low = compute_uniformity(query_embeddings, document_embeddings)
    print(f"uniformity : {effective_rank_low}")

if __name__ == "__main__":
    main()
