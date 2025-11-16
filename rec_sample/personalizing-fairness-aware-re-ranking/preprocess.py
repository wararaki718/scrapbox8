import pandas as pd


def _get_first_genre(row: pd.Series, genre_cols: list[str]) -> str:
    for genre in genre_cols:
        if row[genre] == 1:
            return genre
    return "Unknown"


def preprocess_movie(movie_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess movie DataFrame to add 'provider' column based on genres.

    Args:
        movie_df (pd.DataFrame): DataFrame containing movie information.
    Returns:
        pd.DataFrame: Preprocessed DataFrame with 'provider' column.
    """
    genre_cols = [
        "Action","Adventure","Animation","Childrens","Comedy","Crime","Documentary",
        "Drama","Fantasy","FilmNoir","Horror","Musical","Mystery","Romance","SciFi",
        "Thriller","War","Western",
    ]

    movie_df["provider"] = movie_df.apply(
        lambda row: _get_first_genre(row, genre_cols),
        axis=1,
    )
    return movie_df[["movie_id","title","provider"]]
