import torch


def compute_intra_list_diversity(embeddings: torch.Tensor) -> float:
    """
    Compute intra-list diversity for a list of embeddings.

    Args:
        top-k item embeddings: A tensor of shape (n_items, embedding_dim).
    Returns:
        float: Intra-list diversity score.
    """
    if embeddings.size(0) <= 1:
        return 0.0

    # Normalize embeddings to unit vectors
    normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    # Compute cosine similarity matrix
    similarity_matrix = torch.matmul(
        normalized_embeddings,
        normalized_embeddings.t()
    )
    distance_matrix = 1.0 - similarity_matrix

    # Exclude self-similarities by creating a mask
    mask = ~ torch.eye(embeddings.size(0), dtype=torch.bool, device=embeddings.device)

    # calculate ild score
    ild_score = distance_matrix[mask].sum() / (distance_matrix.size(0) * (distance_matrix.size(0) - 1))

    return ild_score.item()
