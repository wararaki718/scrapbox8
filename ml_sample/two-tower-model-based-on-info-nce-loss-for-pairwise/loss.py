import torch
import torch.nn as nn


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float=1.0) -> None:
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
        self.temperature = temperature

    def forward(
        self,
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        
        predictions = torch.cosine_similarity(query_embeddings, document_embeddings, dim=1)
        predictions /= self.temperature

        log_prob = nn.functional.log_softmax(predictions, dim=0).view(-1, 1)
        positive_scores = log_prob.diag()
        loss = - positive_scores.mean()

        return loss
