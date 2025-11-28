import torch
from torch.utils.data import DataLoader

from dataset import PairwiseDataset


def load_dummy_data(n_data: int, query_input_size: int, document_input_size: int) -> DataLoader:
    X_query = torch.randn((n_data, query_input_size), dtype=torch.float32)
    X_document = torch.randn((n_data, document_input_size), dtype=torch.float32)
    y = torch.randint(0, 2, (n_data,), dtype=torch.float32)

    data_loader = DataLoader(
        PairwiseDataset(X_query, X_document, y),
        batch_size=5,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
    )
    return data_loader
