import torch
import torch.nn as nn


class NPairLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(
        self,
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        
        predictions = torch.cosine_similarity(query_embeddings, document_embeddings, dim=1)

        log_prob = nn.functional.log_softmax(predictions, dim=0).view(-1, 1)
        positive_scores = log_prob.diag()
        loss = - positive_scores.mean()

        return loss
