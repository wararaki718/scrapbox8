from .metrics.coverage import coverage
from .metrics.entropy import entropy


def evaluate(movies: list[int], movie2provider: dict[int, str]) -> dict[str, float]:
    """
    Evaluate the reranked movies using coverage metric.

    Args:
        movies (list[int]): List of movie IDs.
        movie2provider (dict[int, str]): Dictionary mapping movie IDs to their providers.

    Returns:
        float: Coverage value representing the proportion of unique providers.
    """
    if not movies:
        return {"coverage": 0.0, "entropy": 0.0}

    return {
        "coverage": coverage(movies, movie2provider),
        "entropy": entropy(movies, movie2provider),
    }
