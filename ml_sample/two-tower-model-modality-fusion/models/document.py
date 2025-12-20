import torch
import torch.nn as nn
import torch.nn.functional as F


class EarlyFusionDocumentTower(nn.Module):
    def __init__(self, input_dims: dict[str, int], output_dim: int=128) -> None:
        super().__init__()
        input_dim = sum(input_dims.values())
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(
        self,
        modality_dict: dict[str, torch.Tensor],
        active_modalities: list[str] | None=None,
    ) -> torch.Tensor:
        features = [
            modality_dict["title"],
            modality_dict["body"],
            modality_dict["metadata"],
            modality_dict["image"],
            modality_dict["interaction_history"],
            modality_dict["tags"],
        ]

        # ealry fusion: concatenate raw features
        x = torch.cat(features, dim=1)
        output = self.mlp(x)

        return F.normalize(output, p=2, dim=1)


class IntermediateFusionDocumentTower(nn.Module):
    def __init__(self, input_dims: dict[str, int], output_dim: int=128) -> None:
        super().__init__()
        self.encoders = nn.ModuleDict({
            "title": nn.Linear(input_dims["title"], 64),
            "body": nn.Linear(input_dims["body"], 64),
            "metadata": nn.Linear(input_dims["metadata"], 64),
            "image": nn.Linear(input_dims["image"], 64),
            "interaction_history": nn.Linear(input_dims["interaction_history"], 64),
            "tags": nn.Linear(input_dims["tags"], 64),
        })

        self.fusion_layer = nn.Sequential(
            nn.Linear(64 * len(input_dims), 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

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


class LateFusionDocumentTower(nn.Module):
    def __init__(self, input_dims: dict[str, int], output_dim: int=128) -> None:
        super().__init__()
        self.towers = nn.ModuleDict({
            "title": nn.Linear(input_dims["title"], output_dim),
            "body": nn.Linear(input_dims["body"], output_dim),
            "metadata": nn.Linear(input_dims["metadata"], output_dim),
            "image": nn.Linear(input_dims["image"], output_dim),
            "interaction_history": nn.Linear(input_dims["interaction_history"], output_dim),
            "tags": nn.Linear(input_dims["tags"], output_dim),
        })

    def forward(
        self,
        modality_dict: dict[str, torch.Tensor],
        active_modalities: list[str] | None=None,
    ) -> torch.Tensor:
        features = []
        for name, tower in self.towers.items():
            feature = modality_dict[name]
            if active_modalities is not None and name not in active_modalities:
                feature = torch.zeros_like(feature)
            features.append(tower(feature))
        
        # late fusion: average of modality-specific outputs
        combined = torch.stack(features, dim=0).mean(dim=0)

        return F.normalize(combined, p=2, dim=1)
