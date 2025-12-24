import torch
import torch.nn as nn
import torch.nn.functional as F


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
