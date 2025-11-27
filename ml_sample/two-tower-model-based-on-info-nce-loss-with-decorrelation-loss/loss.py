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


class DecorrelatonLoss(nn.Module):
    def __init__(self, lambda_: float=0.001) -> None:
        super().__init__()
        self.lambda_ = lambda_

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        centered_embeddings = embeddings - embeddings.mean(dim=0, keepdim=True)
        norm_embeddings = centered_embeddings / (
            centered_embeddings.pow(2).mean(dim=0) + 1e-6
        ).sqrt()
        correlation_matrix = torch.matmul(
            norm_embeddings,
            norm_embeddings.t()
        ) / embeddings.size(0)

        diag_mask = ~torch.eye(embeddings.size(0), dtype=torch.bool, device=embeddings.device)
        penalty = (correlation_matrix[diag_mask]).pow(2).sum()

        return self.lambda_ * penalty


class DiversityInfoNCELoss(nn.Module):
    def __init__(self, temperature: float=1.0, lambda_d: float=0.001, lambda_q: float=0.001) -> None:
        super().__init__()
        self.info_nce_loss = InfoNCELoss(temperature)
        self.document_decorrelation_loss = DecorrelatonLoss(lambda_d)
        self.query_decorrelation_loss = DecorrelatonLoss(lambda_q)

    def forward(
        self,
        query_embeddings: torch.Tensor,
        document_embeddings: torch.Tensor,
        labels: torch.Tensor, # unused
    ) -> torch.Tensor:
        info_nce_loss = self.info_nce_loss(
            query_embeddings,
            document_embeddings,
            labels, # unused
        )
        loss_document = self.document_decorrelation_loss(document_embeddings)
        loss_query = self.query_decorrelation_loss(query_embeddings)

        total_loss = info_nce_loss + loss_document + loss_query
        return total_loss
