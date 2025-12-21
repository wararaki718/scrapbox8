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
