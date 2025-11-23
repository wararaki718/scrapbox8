import torch
import torch.nn as nn
from lightning import LightningModule
from torch.optim.optimizer import Optimizer


class QueryEncoder(nn.Module):
    def __init__(self, input_size: int) -> None:
        super(QueryEncoder, self).__init__()
        layers = [
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 10),
            nn.ReLU(),
            nn.Linear(10, 8),
            nn.ReLU(),
        ]
        self._model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


class DocumentEncoder(nn.Module):
    def __init__(self, input_size: int) -> None:
        super(DocumentEncoder, self).__init__()
        layers = [
            nn.Linear(input_size, 12),
            nn.ReLU(),
            nn.Linear(12, 8),
            nn.ReLU(),
        ]
        self._model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._model(x)


class UnifiedEmbeddingModel(LightningModule):
    def __init__(
            self,
            query_encoder: QueryEncoder,
            document_encoder: DocumentEncoder,
            criterion: nn.TripletMarginWithDistanceLoss,
        ) -> None:
        super().__init__()
        self._query_encoder = query_encoder
        self._document_encoder = document_encoder
        self._criterion = criterion

        self.automatic_optimization = False

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x_q, x_pos, x_neg = batch
        anchor = self._query_encoder(x_q)
        positive = self._document_encoder(x_pos)
        negative = self._document_encoder(x_neg)

        query_optimizer, document_optimizer = self.optimizers()
        query_optimizer.zero_grad()
        document_optimizer.zero_grad()

        loss: torch.Tensor = self._criterion(anchor, positive, negative)
        self.manual_backward(loss)
        # self.log(f"{batch_idx}: train_loss", loss)

        query_optimizer.step()
        document_optimizer.step()

        return loss

    def configure_optimizers(self) -> tuple[Optimizer, Optimizer]:
        query_optimizer = torch.optim.Adam(self._query_encoder.parameters(), lr=1e-3)
        document_optimizer = torch.optim.Adam(self._document_encoder.parameters(), lr=1e-3)
        return query_optimizer, document_optimizer

    def estimate(
        self,
        x_q: torch.Tensor,
        x_d: torch.Tensor,
    ) -> torch.Tensor:
        self._query_encoder.eval()
        self._document_encoder.eval()
        with torch.no_grad():
            query_embedding = self._query_encoder(x_q)
            document_embedding = self._document_encoder(x_d)
        scores = torch.cosine_similarity(query_embedding, document_embedding, dim=-1)
        return scores
