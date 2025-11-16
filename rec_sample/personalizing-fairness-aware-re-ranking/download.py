import urllib.request
from pathlib import Path
from zipfile import ZipFile


def download_movielens(
    url: str = "http://files.grouplens.org/datasets/movielens/ml-100k.zip",
    data_dir: Path = Path("./data"),
) -> None:
    """
    Download and extract the MovieLens 100k dataset.

    Args:
        url (str): URL to download the dataset from.
        data_dir (Path): Directory to save the dataset.
    """
    # Check if dataset already exists
    zip_path = data_dir / "tmp.zip"
    if (data_dir / "ml-100k").exists():
        print(f"{zip_path} already exists, skipping download.", flush=True)
        return

    # Create data directory if it doesn't exist
    data_dir.mkdir(parents=True, exist_ok=True)

    # donwload and extract dataset
    print("Downloading MovieLens dataset...", flush=True)
    urllib.request.urlretrieve(url, zip_path)

    print("Download complete. Extracting...", flush=True)
    with ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    print("Extraction complete.", flush=True)

    # Remove the zip file after extraction
    zip_path.unlink()


if __name__ == "__main__":
    download_movielens()
