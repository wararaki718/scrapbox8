import numpy as np
from sklearn.metrics.pairwise import cosine_distances


def ilad_metric(embeddings: np.ndarray) -> float:
    """
    Compute the ILAD metric for a set of embeddings.

    Args:
        embeddings (np.ndarray): A 2D array of shape (n_samples, n_features) representing the embeddings.
    Returns:
        float: The computed ILAD metric value.
    """
    n_samples = embeddings.shape[0]
    if n_samples < 2:
        return 0.0

    # Compute pairwise cosine distances between embeddings
    distances = cosine_distances(embeddings)

    # ILAD(L) = (1 / (|L|(|L|-1))) * sum_{i in L} sum_{j in L, j != i} d(i, j)
    off_diagonal_sum = float(np.sum(distances) - np.trace(distances))
    ilad_value = off_diagonal_sum / (n_samples * (n_samples - 1))

    return ilad_value
