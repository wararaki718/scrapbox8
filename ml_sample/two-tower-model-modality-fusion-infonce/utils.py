import torch

from dataset import MultiModalDataset


def _get_query_modalities(n: int=100) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    modalities = {
        "keyword": torch.randn(n, 128),
        "condition": torch.randn(n, 128),
        "user_profile": torch.randn(n, 128),
        "user_action": torch.randn(n, 128),
    }
    modality_dims = {name: tensor.size(1) for name, tensor in modalities.items()}
    return modalities, modality_dims


def _get_document_modalities(n: int=100) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    modalities = {
        "title": torch.randn(n, 128),
        "body": torch.randn(n, 128),
        "metadata": torch.randn(n, 128),
        "image": torch.randn(n, 128),
        "interaction_history": torch.randn(n, 128),
        "tags": torch.randn(n, 128),
    }
    modality_dims = {name: tensor.size(1) for name, tensor in modalities.items()}
    return modalities, modality_dims


def load_dummy_data(n_data: int=100) -> tuple[
    MultiModalDataset,
    dict[str, int],
    dict[str, int],
]:
    query_modalities, query_modality_dims = _get_query_modalities(n_data)
    document_modalities, document_modality_dims = _get_document_modalities(n_data)
    hard_negative_document_modalities, _ = _get_document_modalities(n_data)

    dataset = MultiModalDataset(
        query_modalities=query_modalities,
        document_modalities=document_modalities,
        hard_negative_document_modalities=hard_negative_document_modalities,
    )

    return dataset, query_modality_dims, document_modality_dims
