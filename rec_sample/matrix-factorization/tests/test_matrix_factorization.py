import shutil
import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model import MatrixFactorization
from utils import (
    ensure_official_movielens_small,
    load_movielens_official_small,
    load_movielens_small_sample,
)


def test_load_movielens_small_sample() -> None:
    data = load_movielens_small_sample()

    assert data.num_users >= 5
    assert data.num_items >= 6
    assert data.user_indices.shape == data.item_indices.shape == data.ratings.shape
    assert data.user_indices.max() < data.num_users
    assert data.item_indices.max() < data.num_items


def test_training_reduces_mse() -> None:
    data = load_movielens_small_sample()

    model = MatrixFactorization(
        num_users=data.num_users,
        num_items=data.num_items,
        n_factors=8,
        seed=0,
    )

    before = model.mse(data.user_indices, data.item_indices, data.ratings)
    model.fit(
        user_indices=data.user_indices,
        item_indices=data.item_indices,
        ratings=data.ratings,
        epochs=120,
        lr=0.03,
        reg=0.01,
    )
    after = model.mse(data.user_indices, data.item_indices, data.ratings)

    assert after < before


def test_recommend_excludes_seen_items() -> None:
    data = load_movielens_small_sample()
    model = MatrixFactorization(
        num_users=data.num_users,
        num_items=data.num_items,
        n_factors=8,
        seed=0,
    )
    model.fit(
        user_indices=data.user_indices,
        item_indices=data.item_indices,
        ratings=data.ratings,
        epochs=80,
        lr=0.03,
        reg=0.01,
    )

    target_user = 0
    seen = data.item_indices[data.user_indices == target_user]
    recommendations = model.recommend(
        user_index=target_user,
        seen_item_indices=seen,
        top_k=2,
    )

    assert len(recommendations) == 2
    assert len(set(recommendations)) == 2
    assert set(recommendations).isdisjoint(set(seen.tolist()))


def test_ensure_official_movielens_small_and_loadable() -> None:
    project_dir = Path(__file__).resolve().parents[1]
    data_dir = project_dir / "data"
    zip_path = data_dir / "ml-latest-small.zip"
    extracted_dir = data_dir / "ml-latest-small"

    if zip_path.exists():
        zip_path.unlink()
    if extracted_dir.exists():
        shutil.rmtree(extracted_dir)

    mirror_root = project_dir / "tests" / "_fixtures" / "ml-mirror"
    mirror_root.mkdir(parents=True, exist_ok=True)
    mirror_zip = mirror_root / "ml-latest-small.zip"

    with ZipFile(mirror_zip, mode="w", compression=ZIP_DEFLATED) as zf:
        zf.writestr(
            "ml-latest-small/ratings.csv",
            "userId,movieId,rating,timestamp\n1,10,4.0,1\n2,11,3.5,2\n",
        )

    downloaded_zip = ensure_official_movielens_small(
        data_dir=data_dir,
        source_url=mirror_zip.as_uri(),
    )

    assert downloaded_zip.exists()
    data = load_movielens_official_small(data_dir=data_dir)
    assert data.num_users == 2
    assert data.num_items == 2
    assert data.ratings.shape[0] == 2
