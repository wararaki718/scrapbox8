import torch
from torch.utils.data import Dataset


class PairwiseDataset(Dataset):
    def __init__(self, X_query: torch.Tensor, X_document: torch.Tensor, y: torch.Tensor) -> None:
        self._X_query = X_query
        self._X_document = X_document
        self._y = y

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._X_query[index], self._X_document[index], self._y[index]

    def __len__(self) -> int:
        return len(self._X_query)
