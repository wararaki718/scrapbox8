import torch


def compute_intra_list_average_distances(embeddings: torch.Tensor) -> float:
    """
    Compute intra-list average distances for a list of embeddings.

    Args:
        top-k item embeddings: A tensor of shape (n_items, embedding_dim).

    Returns:
        The average pairwise distance between the embeddings.
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
    mask = ~torch.eye(embeddings.size(0), dtype=torch.bool, device=embeddings.device)

    # calculate ilad score
    ilad_score = distance_matrix[mask].sum() / (distance_matrix.size(0) * (distance_matrix.size(0) - 1))

    return ilad_score.item()


def compute_intra_list_minimal_distances(embeddings: torch.Tensor) -> float:
    """
    Compute intra-list minimal distances for a list of embeddings.

    Args:
        top-k item embeddings: A tensor of shape (n_items, embedding_dim).

    Returns:
        The minimal pairwise distance between the embeddings.
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

    # calculate ilmd score
    ilmd_score = distance_matrix[mask].min()

    return ilmd_score.item()
