from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np

MOVIELENS_SMALL_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


@dataclass(frozen=True)
class RatingsData:
    user_indices: np.ndarray
    item_indices: np.ndarray
    ratings: np.ndarray
    num_users: int
    num_items: int
    raw_user_ids: np.ndarray
    raw_item_ids: np.ndarray


def _build_ratings_data(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    ratings: np.ndarray,
) -> RatingsData:
    unique_users = np.unique(user_ids)
    unique_items = np.unique(item_ids)

    user_map = {u: idx for idx, u in enumerate(unique_users.tolist())}
    item_map = {m: idx for idx, m in enumerate(unique_items.tolist())}

    user_indices = np.array([user_map[u] for u in user_ids], dtype=np.int64)
    item_indices = np.array([item_map[m] for m in item_ids], dtype=np.int64)

    return RatingsData(
        user_indices=user_indices,
        item_indices=item_indices,
        ratings=ratings,
        num_users=len(unique_users),
        num_items=len(unique_items),
        raw_user_ids=unique_users,
        raw_item_ids=unique_items,
    )


def _load_ratings_csv(csv_path: Path) -> RatingsData:
    raw = np.genfromtxt(csv_path, delimiter=",", names=True)

    user_ids = np.atleast_1d(raw["userId"]).astype(np.int64)
    item_ids = np.atleast_1d(raw["movieId"]).astype(np.int64)
    ratings = np.atleast_1d(raw["rating"]).astype(np.float64)

    return _build_ratings_data(user_ids=user_ids, item_ids=item_ids, ratings=ratings)


def load_movielens_small_sample() -> RatingsData:
    csv_path = Path(__file__).parent / "data" / "ratings_sample.csv"
    return _load_ratings_csv(csv_path)


def ensure_official_movielens_small(
    data_dir: Path | None = None,
    source_url: str = MOVIELENS_SMALL_URL,
) -> Path:
    base_dir = data_dir or (Path(__file__).parent / "data")
    base_dir.mkdir(parents=True, exist_ok=True)

    zip_path = base_dir / "ml-latest-small.zip"
    extracted_dir = base_dir / "ml-latest-small"
    ratings_csv = extracted_dir / "ratings.csv"

    if not zip_path.exists():
        urlretrieve(source_url, zip_path)

    if not ratings_csv.exists():
        with ZipFile(zip_path, mode="r") as zf:
            zf.extractall(base_dir)

    return zip_path


def load_movielens_official_small(data_dir: Path | None = None) -> RatingsData:
    base_dir = data_dir or (Path(__file__).parent / "data")
    ensure_official_movielens_small(data_dir=base_dir)

    csv_path = base_dir / "ml-latest-small" / "ratings.csv"
    return _load_ratings_csv(csv_path)
