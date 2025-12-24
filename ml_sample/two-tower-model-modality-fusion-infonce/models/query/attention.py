import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionQueryTower(nn.Module):
    def __init__(self, input_dims: dict[str, int], output_dim: int=128) -> None:
        super().__init__()
        self.encoders = nn.ModuleDict({
            "keyword": nn.Linear(input_dims["keyword"], output_dim),
            "condition": nn.Linear(input_dims["condition"], output_dim),
            "user_profile": nn.Linear(input_dims["user_profile"], output_dim),
            "user_action": nn.Linear(input_dims["user_action"], output_dim),
        })
        self.attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=4,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(output_dim)
        self.fc = nn.Linear(output_dim, output_dim)

    def forward(
        self,
        modality_dict: dict[str, torch.Tensor],
        active_modalities: list[str] | None=None,
        return_attention: bool=False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        features = []
        for name, encoder in self.encoders.items():
            feature = modality_dict[name]
            if active_modalities is not None and name not in active_modalities:
                feature = torch.zeros_like(feature)
            features.append(encoder(feature))

        # attentionを計算
        x_stacked = torch.stack(features, dim=1)  # (batch_size, num_modalities, output_dim)
        attn_output, weights = self.attention(
            query=x_stacked,
            key=x_stacked,
            value=x_stacked,
        )  # (batch_size, num_modalities, output_dim)

        # 残差接続と正規化
        x = self.norm(x_stacked + attn_output)
        x_output = self.fc(torch.mean(x, dim=1))
        if return_attention:
            return F.normalize(x_output, p=2, dim=1), weights
        return F.normalize(x_output, p=2, dim=1)

    def get_attention_map(self, modality_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            _, weights = self.forward(modality_dict, return_attention=True)
        return weights
