import torch
import torch.nn as nn
import torch.nn.functional as F


class SymmetricInfoNCEWithHardNegativesLoss(nn.Module):
    def __init__(self, tau: float = 0.07) -> None:
        super().__init__()
        # 1/tau を学習させる（CLIPの公式実装に準拠）
        self.logit_scale = nn.Parameter(torch.log(torch.tensor([1.0 / tau])))

    def forward(self, q_emb: torch.Tensor, d_pos: torch.Tensor, d_hard: torch.Tensor) -> torch.Tensor:
        q = F.normalize(q_emb, p=2, dim=1)
        dp = F.normalize(d_pos, p=2, dim=1)
        dh = F.normalize(d_hard, p=2, dim=1)
        
        # logit_scale = 1/tau. クランプ範囲は ln(1)〜ln(100) 程度が妥当
        scale = self.logit_scale.exp().clamp(max=100)
        
        # (B, B) 類似度行列
        logits_inbatch = torch.matmul(q, dp.T) * scale
        
        # (B, 1) ハード負例との類似度 (対角成分のみ抽出)
        # q_i と d_hard_i のペア
        logits_hard = (q * dh).sum(dim=1, keepdim=True) * scale
        
        # 合計 (B, B+1) のロジット。正解ラベルは対角 (0, 1, 2, ..., B-1)
        logits_q2d = torch.cat([logits_inbatch, logits_hard], dim=1)
        labels = torch.arange(q.size(0), device=q.device)
        
        loss_q2d = F.cross_entropy(logits_q2d, labels)
        
        # d2q側: 簡易化のため inbatch のみで行うか、
        # 正解ドキュメント dp から見て dh も負例として扱う設計にする
        loss_d2q = F.cross_entropy(logits_inbatch.T, labels)
        
        return (loss_q2d + loss_d2q) / 2
