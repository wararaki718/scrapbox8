import torch


def get_query_modalities(n: int=100) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    modalities = {
        "keyword": torch.randn(n, 128),
        "condition": torch.randn(n, 128),
        "user_profile": torch.randn(n, 128),
        "user_action": torch.randn(n, 128),
    }
    modality_dims = {name: tensor.size(1) for name, tensor in modalities.items()}
    return modalities, modality_dims


def get_document_modalities(n: int=100) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    modalities = {
        "title": torch.randn(n, 128),
        "body": torch.randn(n, 128),
        "metadata": torch.randn(n, 128),
    }
    modality_dims = {name: tensor.size(1) for name, tensor in modalities.items()}
    return modalities, modality_dims


def get_labels(n: int=100) -> torch.Tensor:
    return torch.randint(0, 2, (n,), dtype=torch.float32)


def load_dummy_data(n_data: int=100) -> tuple[
    tuple[dict[str, torch.Tensor], dict[str, int]],
    tuple[dict[str, torch.Tensor], dict[str, int]],
    torch.Tensor,
]:
    query_modalities, query_modality_dims = get_query_modalities(n_data)
    document_modalities, document_modality_dims = get_document_modalities(n_data)
    labels = get_labels(n_data)

    return query_modalities, query_modality_dims, document_modalities, document_modality_dims, labels
