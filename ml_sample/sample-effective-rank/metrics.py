import torch


def compute_effective_rank(embeddings: torch.Tensor, epsilon: float=1e-10) -> float:
    """
    Compute the effective rank of the given embeddings.

    Args:
        embeddings (torch.Tensor): A 2D tensor of shape (n_samples, n_features).
        epsilon (float): A small value to avoid log(0).

    Returns:
        float: The effective rank of the embeddings.
    """
    # Center the embeddings
    centered_embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)

    # Compute the covariance matrix
    covariance_matrix = torch.matmul(
        centered_embeddings.t(),
        centered_embeddings
    ) / (embeddings.size(0) - 1)

    # Compute eigenvalues
    eigenvalues = torch.linalg.svd(covariance_matrix, full_matrices=False).S

    # Normalize eigenvalues to form a probability distribution
    sigma_square = eigenvalues.pow(2)
    probabilities = sigma_square / sigma_square.sum()

    # Compute entropy
    entropy: torch.Tensor = - (probabilities * (probabilities + epsilon).log()).sum()

    # Effective rank
    effective_rank: float = torch.exp(entropy).item()
    return effective_rank