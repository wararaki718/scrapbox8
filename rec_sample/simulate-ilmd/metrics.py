import numpy as np
from sklearn.metrics.pairwise import cosine_distances


def ilmd_metric(embeddings: np.ndarray) -> float:
    """
    Compute the ILMD metric for a set of embeddings.

    Args:
        embeddings (np.ndarray): A 2D array of shape (n_samples, n_features) representing the embeddings.
    Returns:
        float: The computed ILMD metric value.
    """
    distances = cosine_distances(embeddings)
    triangular_indices = np.triu_indices_from(distances, k=1)
    ilmd_value = np.median(distances[triangular_indices])
    return ilmd_value
