import torch
from torch.utils.data import Dataset


class TripletDataset(Dataset):
    def __init__(self, X_q: torch.Tensor, X_pos: torch.Tensor, X_neg: torch.Tensor) -> None:
        self._X_q = X_q
        self._X_pos = X_pos
        self._X_neg = X_neg

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._X_q[index], self._X_pos[index], self._X_neg[index]

    def __len__(self) -> int:
        return len(self._X_q)
