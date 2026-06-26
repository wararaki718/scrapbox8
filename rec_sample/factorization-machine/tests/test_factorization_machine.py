import sys
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model import FactorizationMachineRegressor
from utils import (
    ensure_official_movielens_small,
    load_movielens_official_small,
    make_movielens_user_item_features,
)


def _prepare_local_movielens_zip(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    mirror_zip = base_dir / "ml-latest-small.zip"
    with ZipFile(mirror_zip, mode="w", compression=ZIP_DEFLATED) as zf:
        zf.writestr(
            "ml-latest-small/ratings.csv",
            "userId,movieId,rating,timestamp\n"
            "1,10,4.0,1\n"
            "1,11,5.0,2\n"
            "2,10,3.5,3\n"
            "2,12,2.0,4\n"
            "3,11,4.5,5\n"
            "3,12,1.0,6\n",
        )
    return mirror_zip


def test_ensure_official_movielens_small_and_loadable_with_local_mirror(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    mirror_zip = _prepare_local_movielens_zip(tmp_path / "mirror")

    downloaded_zip = ensure_official_movielens_small(
        data_dir=data_dir,
        source_url=mirror_zip.as_uri(),
    )

    assert downloaded_zip.exists()
    data = load_movielens_official_small(data_dir=data_dir)
    assert data.num_users == 3
    assert data.num_items == 3
    assert data.ratings.shape[0] == 6


def test_make_features_shapes(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    mirror_zip = _prepare_local_movielens_zip(tmp_path / "mirror")
    ensure_official_movielens_small(data_dir=data_dir, source_url=mirror_zip.as_uri())

    data = load_movielens_official_small(data_dir=data_dir)
    x, y = make_movielens_user_item_features(data)

    assert x.shape[0] == y.shape[0] == data.ratings.shape[0]
    assert x.shape[1] == data.num_users + data.num_items
    assert np.all((y >= 0.5) & (y <= 5.0))


def test_factorization_machine_training_reduces_mse(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    mirror_zip = _prepare_local_movielens_zip(tmp_path / "mirror")
    ensure_official_movielens_small(data_dir=data_dir, source_url=mirror_zip.as_uri())

    data = load_movielens_official_small(data_dir=data_dir)
    x, y = make_movielens_user_item_features(data)

    fm = FactorizationMachineRegressor(n_features=x.shape[1], k=4, seed=0)
    before = fm.mse(x, y)
    history = fm.fit(x, y, epochs=60, lr=0.03, reg_w=1e-4, reg_v=1e-4)
    after = history[-1]

    assert after < before
