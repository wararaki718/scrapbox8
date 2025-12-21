import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedQueryTower(nn.Module):
    def __init__(self, input_dims: dict[str, int], output_dim: int=128) -> None:
        super().__init__()
        self.encoders = nn.ModuleDict({
            "keyword": nn.Linear(input_dims["keyword"], output_dim),
            "condition": nn.Linear(input_dims["condition"], output_dim),
            "user_profile": nn.Linear(input_dims["user_profile"], output_dim),
            "user_action": nn.Linear(input_dims["user_action"], output_dim),
        })
        self.gate = nn.Sequential(
            nn.Linear(output_dim * len(input_dims), output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, len(input_dims)),
        )
        self.model = nn.Linear(output_dim, output_dim)

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

        # ゲートを計算
        stacked_features = torch.stack(features, dim=1)  # (batch_size, num_modalities, output_dim)
        gate_values = self.gate(stacked_features.reshape(stacked_features.size(0), -1))  # (batch_size, num_modalities)
        gate_weights = F.softmax(gate_values, dim=1).unsqueeze(-1)  # (batch_size, num_modalities, 1)

        # 加重和を計算
        gated_features = torch.sum(stacked_features * gate_weights, dim=1)  # (batch_size, output_dim)
        output = self.model(gated_features)
        return F.normalize(output, p=2, dim=1)

    def get_gate_weights(self, modality_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            features = {
                name: torch.relu(encoder(modality_dict[name]))
                for name, encoder in self.encoders.items()
            }
            x_combined = torch.cat(list(features.values()), dim=1)
            x_gate = torch.softmax(self.gate(x_combined), dim=1)
        return x_gate
