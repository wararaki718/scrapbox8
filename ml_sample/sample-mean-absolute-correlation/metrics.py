import torch

def compute_mean_absolute_correlation(embeddings: torch.Tensor) -> float:
    """
    Compute the correlation matrix of the given embeddings.

    Args:
        embeddings (torch.Tensor): A 2D tensor of shape (n_samples, n_features).

    Returns:
        torch.Tensor: A 2D tensor representing the correlation matrix of shape (n_samples, n_samples).
    """
    if embeddings.size(0) <= 1:
        raise ValueError("At least two samples are required to compute the correlation matrix.")

    centered_embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
    norm_embeddings = centered_embeddings / (centered_embeddings.std(dim=0, keepdim=True) + 1e-8)
    correlation_matrix = torch.matmul(
        norm_embeddings.t(),
        norm_embeddings
    ) / embeddings.size(0)

    I = torch.eye(correlation_matrix.size(0), dtype=torch.bool, device=embeddings.device)
    off_diag_elements = correlation_matrix[~I]
    mean_absolute_correlation = off_diag_elements.abs().mean()

    return mean_absolute_correlation.item()
