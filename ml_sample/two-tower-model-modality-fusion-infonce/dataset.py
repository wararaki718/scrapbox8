import torch
from torch.utils.data import Dataset


class MultiModalDataset(Dataset):
    def __init__(
        self,
        query_modalities: dict[str, torch.Tensor],
        document_modalities: dict[str, torch.Tensor],
        hard_negative_document_modalities: dict[str, torch.Tensor],
    ) -> None:
        self.query_modalities = query_modalities
        self.document_modalities = document_modalities
        self.hard_negative_document_modalities = hard_negative_document_modalities

    def __len__(self) -> int:
        return next(iter(self.query_modalities.values())).size(0)

    def __getitem__(self, idx: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        x_query = {name: tensor[idx] for name, tensor in self.query_modalities.items()}
        x_document = {name: tensor[idx] for name, tensor in self.document_modalities.items()}
        x_hard_negative_document = {name: tensor[idx] for name, tensor in self.hard_negative_document_modalities.items()}
        return x_query, x_document, x_hard_negative_document
