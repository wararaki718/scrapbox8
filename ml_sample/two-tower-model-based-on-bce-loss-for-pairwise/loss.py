import torch
import torch.nn as nn


class BCELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(
        self,
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        
        predictions = 1 - torch.cosine_similarity(query_embeddings, document_embeddings, dim=1)
        loss = self.criterion(predictions, labels)

        return loss
