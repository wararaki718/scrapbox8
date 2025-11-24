import torch
from torch.utils.data import DataLoader

from dataset import TripletDataset


def load_dummy_data(n_data: int, query_input_size: int, document_input_size: int) -> DataLoader:
    X_query = torch.randn((n_data, query_input_size), dtype=torch.float32)
    X_pos = torch.randn((n_data, document_input_size), dtype=torch.float32)
    X_neg = torch.randn((n_data, document_input_size), dtype=torch.float32)

    data_loader = DataLoader(
        TripletDataset(X_query, X_pos, X_neg),
        batch_size=5,
        shuffle=True,
        num_workers=2,
        persistent_workers=True,
    )
    return data_loader
