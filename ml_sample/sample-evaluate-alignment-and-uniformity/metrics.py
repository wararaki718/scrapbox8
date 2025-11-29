import torch


def compute_alignment(queries: torch.Tensor, documents: torch.Tensor) -> float:
    """Compute alignment metrics for embeddings.

    Args:
        query (torch.Tensor): Query embeddings of shape (batch_size, embedding_dim).
        document (torch.Tensor): Document embeddings of shape (batch_size, embedding_dim).

    Returns:
        float: Alignment metric.
    """
    similarities = torch.cosine_similarity(queries, documents, dim=1)
    alignment = (2 * (1 - similarities)).mean()
    return alignment.item()


def compute_uniformity(queries: torch.Tensor, documents: torch.Tensor) -> float:
    """Compute uniformity metrics for embeddings.

    Args:
        queries (torch.Tensor): Query embeddings of shape (batch_size, embedding_dim).
        documents (torch.Tensor): Document embeddings of shape (batch_size, embedding_dim).

    Returns:
        float: Uniformity metric.
    """
    embeddings = torch.cat([queries, documents], dim=0)
    similarities = torch.matmul(
        embeddings,
        embeddings.t()
    ) / (embeddings.norm(dim=1, keepdim=True) * embeddings.norm(dim=1, keepdim=True).t() + 1e-8)

    distribution = 2 * (1 - similarities)
    mask = ~torch.eye(distribution.size(0), dtype=torch.bool, device=embeddings.device)

    uniformity = torch.log(torch.exp(-2 * distribution[mask]).mean())
    return uniformity.item()
