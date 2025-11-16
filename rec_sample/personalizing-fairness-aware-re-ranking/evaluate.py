def coverage(movies: list[int], movie2provider: dict[int, str]) -> float:
    """
    Calculate the coverage of providers in the given list of movies.

    Args:
        movies (list[int]): List of movie IDs.
        movie2provider (dict[int, str]): Dictionary mapping movie IDs to their providers.

    Returns:
        float: Coverage value representing the proportion of unique providers.
    """
    if not movies:
        return 0.0

    return len({movie2provider[movie_id] for movie_id in movies})
