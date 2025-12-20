import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryTower(nn.Module):
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
