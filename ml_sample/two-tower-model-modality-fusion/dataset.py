import torch
from torch.utils.data import Dataset


class MultiModalDataset(Dataset):
    def __init__(
        self,
        query_modalities: dict[str, torch.Tensor],
        document_modalities: dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> None:
        self.query_modalities = query_modalities
        self.document_modalities = document_modalities
        self.labels = labels

    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        x_query = {name: tensor[idx] for name, tensor in self.query_modalities.items()}
        x_document = {name: tensor[idx] for name, tensor in self.document_modalities.items()}
        y = self.labels[idx]
        return x_query, x_document, y
