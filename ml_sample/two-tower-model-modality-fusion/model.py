import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch.optim.optimizer import Optimizer


class MultiModalTower(nn.Module):
    def __init__(self, input_dims: dict[str, int], output_dim: int=128) -> None:
        super().__init__()
        self.encoders = nn.ModuleDict({
            name: nn.Linear(dim, 64) for name, dim in input_dims.items()
        })
        self.fusion_layer = nn.Linear(64 * len(input_dims), output_dim)

    def forward(
        self,
        modality_dict: dict[str, torch.Tensor],
        active_modalities: list[str] | None=None,
    ) -> torch.Tensor:
        features = []
        for name, encoder in self.encoders.items():
            feature = modality_dict[name]
            if active_modalities is not None and name not in active_modalities:
                feature = torch.zeros_like(feature)
            features.append(encoder(feature))
        
        # 結合して融合層へ
        combined = torch.cat(features, dim=1)
        output = self.fusion_layer(combined)

        return F.normalize(output, p=2, dim=1)


class TwoTowerModel(LightningModule):
    def __init__(
            self,
            query_encoder: MultiModalTower,
            document_encoder: MultiModalTower,
        ) -> None:
        super().__init__()
        self._query_encoder = query_encoder
        self._document_encoder = document_encoder
        self._criterion = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0 - torch.cosine_similarity(x, y),
            margin=1.0,
        )

        self.automatic_optimization = False

    def training_step(
        self,
        batch: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x_query, x_document, y = batch
        query_embeddings = self._query_encoder(x_query)
        document_embeddings = self._document_encoder(x_document)

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
        query_optimizer = torch.optim.Adam(self._query_encoder.parameters(), lr=1e-3)
        document_optimizer = torch.optim.Adam(self._document_encoder.parameters(), lr=1e-3)
        return query_optimizer, document_optimizer
