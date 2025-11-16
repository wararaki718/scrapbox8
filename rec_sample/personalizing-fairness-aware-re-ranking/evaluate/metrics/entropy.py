from collections import Counter

import numpy as np


def entropy(movies: list[int], movie2provider: dict[int, str]) -> float:
    """
    Calculate the entropy of providers in the given list of movies.

    Args:
        movies (list[int]): List of movie IDs.
        movie2provider (dict[int, str]): Mapping from movie ID to provider name.

    Returns:
        float: Entropy value representing the diversity of providers.
    """
    if not movies:
        return 0.0

    provider_counts = Counter([movie2provider.get(movie_id, "Unknown") for movie_id in movies])
    p = np.array(list(provider_counts.values())) / sum(provider_counts.values())
    result = -np.sum(p * np.log(p + 1e-6))
    return float(result)
