import re
from pathlib import Path

import pandas as pd


def load_users(filepath: Path=Path("./data/ml-100k/u.user")) -> pd.DataFrame:
    # Load users
    df = pd.read_csv(
        filepath,
        sep="|",
        header=None,
        names=["userId","age","gender","occupation","zip"],
        encoding="latin-1",
    )
    return df


def load_items(filepath: Path=Path("./data/ml-100k/u.item")) -> pd.DataFrame:
    genre_columns = [f"g{i}" for i in range(19)]
    column_names = ["movieId","title","release","video","url"] + genre_columns
    df_items = pd.read_csv(
        filepath,
        sep="|",
        header=None,
        engine="python",
        names=column_names,
        encoding="latin-1",
    )
    def extract_year(title):
        m = re.search(r"(\d{4})", str(title))
        return float(m.group(1)) if m else 0.0

    df_items["year"] = df_items.title.apply(extract_year)
    df_items["year_norm"] = (df_items.year - df_items.year.min())/(df_items.year.max()-df_items.year.min()+1e-6)

    return df_items


def load_ratings(filepath: Path=Path("./data/ml-100k/u.data")) -> pd.DataFrame:
    df = pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        names=["userId", "movieId", "rating", "timestamp"],
    )
    return df
