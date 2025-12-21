import torch
import torch.nn as nn
from lightning import LightningModule
from torch.optim.optimizer import Optimizer


class TwoTowerModel(LightningModule):
    def __init__(
            self,
            query_encoder: nn.Module,
            document_encoder: nn.Module,
        ) -> None:
        super().__init__()
        self.query_encoder = query_encoder
        self.document_encoder = document_encoder
        self._criterion = nn.CosineEmbeddingLoss(
            margin=1.0,
        )
        self.automatic_optimization = False

    def training_step(
        self,
        batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x_query, x_document, y = batch
        query_embeddings = self.query_encoder(x_query)
        document_embeddings = self.document_encoder(x_document)

        query_optimizer, document_optimizer = self.optimizers()
        query_optimizer.zero_grad()
        document_optimizer.zero_grad()

        loss: torch.Tensor = self._criterion(query_embeddings, document_embeddings, y)
        self.manual_backward(loss)
        # self.log(f"{batch_idx}: train_loss", loss)

        query_optimizer.step()
        document_optimizer.step()

        return loss

    def configure_optimizers(self) -> tuple[Optimizer, Optimizer]:
        query_optimizer = torch.optim.Adam(self.query_encoder.parameters(), lr=1e-3)
        document_optimizer = torch.optim.Adam(self.document_encoder.parameters(), lr=1e-3)
        return query_optimizer, document_optimizer
