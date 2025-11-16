from pathlib import Path

import pandas as pd


def load_ratings(
    data_path: Path=Path("./data/ml-100k/u.data"),
) -> pd.DataFrame:
    """
    Load MovieLens 100k dataset.

    Args:
        data_path (Path): Path to the MovieLens data file.

    Returns:
        pd.DataFrame: DataFrame with user-item ratings.
    """
    # Load ratings data
    ratings_path = data_path
    ratings_df = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )

    return ratings_df


def load_movies(movie_path: Path=Path("./data/ml-100k/u.item")) -> pd.DataFrame:
    """
    Load MovieLens 100k movies data.

    Args:
        movie_path (Path): Path to the MovieLens movies file.

    Returns:
        pd.DataFrame: DataFrame with movie information.
    """
    # Load movies data
    movies = pd.read_csv(
        movie_path,
        sep="|",
        encoding="latin-1",
        names=[
            "movie_id","title","release","video","url",
            "unknown","Action","Adventure","Animation","Childrens",
            "Comedy","Crime","Documentary","Drama","Fantasy",
            "FilmNoir","Horror","Musical","Mystery","Romance","SciFi",
            "Thriller","War","Western",
        ]
    )

    return movies


if __name__ == "__main__":
    ratings_df = load_ratings()
    movies_df = load_movies()
    print(ratings_df.head())
    print(ratings_df.shape)
    print(movies_df.head())
    print(movies_df.shape)
