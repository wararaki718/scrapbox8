import torch
import torch.nn as nn
from lightning import LightningModule
from torch.optim.optimizer import Optimizer

from .loss import SymmetricInfoNCEWithHardNegativesLoss

class TwoTowerModel(LightningModule):
    def __init__(
            self,
            query_encoder: nn.Module,
            document_encoder: nn.Module,
            lr: float = 1e-3,
            tau: float = 0.07,
        ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["query_encoder", "document_encoder"])

        self.query_encoder = query_encoder
        self.document_encoder = document_encoder

        self._criterion = SymmetricInfoNCEWithHardNegativesLoss(
            tau=tau,
        )
    
    def forward(self, x_query: dict[str, torch.Tensor], x_document: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        query_embeddings = self.query_encoder(x_query, modal_mask_prob=self.modal_mask_prob)
        document_embeddings = self.document_encoder(x_document, modal_mask_prob=self.modal_mask_prob)
        return query_embeddings, document_embeddings

    def training_step(
        self,
        batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]],
        batch_idx: int,
    ) -> torch.Tensor:
        x_query, x_document, x_hard_negative_document = batch

        query_embeddings = self.query_encoder(x_query)
        document_embeddings = self.document_encoder(x_document)
        hard_negative_document_embeddings = self.document_encoder(x_hard_negative_document)

        loss: torch.Tensor = self._criterion(query_embeddings, document_embeddings, hard_negative_document_embeddings)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/tau", torch.exp(self._criterion.logit_scale).clamp(max=100), on_step=True, prog_bar=True, logger=True)

        return loss

    def validation_step(
        self,
        batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor]],
        batch_idx: int
    ) -> None:
        x_query, x_document, x_hard_negative_document = batch
        query_embeddings = self.query_encoder(x_query)
        document_embeddings = self.document_encoder(x_document)
        hard_negative_document_embeddings = self.document_encoder(x_hard_negative_document)

        val_loss = self._criterion(query_embeddings, document_embeddings, hard_negative_document_embeddings)
        self.log("val/loss", val_loss, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self) -> dict[str, Optimizer]:
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
