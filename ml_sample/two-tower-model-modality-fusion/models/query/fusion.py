import torch
import torch.nn as nn


class GatedMultimodalFusionQueryTower(nn.Module):
    """
    Gated Multimodal Fusion (GMF) Module
    
    各モーダルの入力を共通次元に投影し、ゲート機構を用いて動的に重み付け合計を行う。
    
    Args:
        input_dims (Dict[str, int]): モーダル名をキー、入力次元数を値とする辞書。
        projection_dim (int): 投影後の共通次元数 (D)。
        dropout_rate (float): ドロップアウト率。
        activation (str): 投影後の活性化関数 ('relu', 'tanh', 'gelu' 等)。
    """
    def __init__(
        self, 
        input_dims: dict[str, int], 
        output_dim: int, 
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.modalities = list(input_dims.keys())
        self.projection_dim = output_dim
        
        # 1. Feature Projection Layers
        # 各モーダルを共通次元 D に投影
        self.projections = nn.ModuleDict({
            m: nn.Linear(dim, output_dim) 
            for m, dim in input_dims.items()
        })
        
        # 2. Gate Mechanism
        # 全モーダルの情報を統合してゲート値を算出するための統合層
        # 効率化のため、一括で計算する設計
        self.gate_layer = nn.Sequential(
            nn.Linear(len(self.modalities) * output_dim, len(self.modalities)),
            nn.Sigmoid()
        )
        
        # 3. Utilities
        self.dropout = nn.Dropout(dropout_rate)
        self.act = nn.Tanh()
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            inputs (dict[str, torch.Tensor]): 
                各モーダルの名前とTensor {modal_name: (batch_size, input_dim)}
        
        Returns:
            torch.Tensor: 融合されたベクトル (batch_size, output_dim)
        """
        projected_features = []
        
        # 各モーダルの投影 (Batch, D)
        for m in self.modalities:
            x = self.projections[m](inputs[m])
            x = self.act(x)
            projected_features.append(x)
        
        # スタックしてテンソル化: (Batch, Num_Modalities, D)
        # 後の加重平均計算をベクトル化するために保持
        feat_stack = torch.stack(projected_features, dim=1) 
        
        # ゲート値の計算
        # 全特徴量を結合してコンテキストを考慮したゲートを生成
        # (Batch, Num_Modalities * D)
        combined = feat_stack.flatten(start_dim=1)
        gates = self.gate_layer(combined) # (Batch, Num_Modalities)
        
        # 加重合計: (Batch, 1, Num_Modalities) @ (Batch, Num_Modalities, D)
        # -> (Batch, 1, D) -> (Batch, D)
        gates = gates.unsqueeze(1) 
        fusion_out = torch.bmm(gates, feat_stack).squeeze(1)
        fusion_out = self.layer_norm(fusion_out)
        
        return self.dropout(fusion_out)

    def get_gate_weights(self, modality_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        with torch.no_grad():
            features = {
                name: torch.relu(encoder(modality_dict[name]))
                for name, encoder in self.projections.items()
            }
            x_combined = torch.cat(list(features.values()), dim=1)
            x_gate = torch.softmax(self.gate_layer(x_combined), dim=1)
        return x_gate
