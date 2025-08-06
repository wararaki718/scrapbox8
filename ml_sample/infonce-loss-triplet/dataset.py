import torch
from torch.utils.data import Dataset


class TripletDataset(Dataset):
    def __init__(self, anchors: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor) -> None:
        self._anchors = anchors
        self._positives = positives
        self._negatives = negatives

    def __len__(self) -> int:
        return len(self._anchors)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._anchors[index], self._positives[index], self._negatives[index]
