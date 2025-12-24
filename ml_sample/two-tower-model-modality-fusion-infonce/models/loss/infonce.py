import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Two-towerモデル向けのInfoNCE Loss実装。
    学習可能な温度パラメータ (Learnable Temperature) を含む。
    """
    def __init__(self, tau: float = 0.07, learnable: bool = True) -> None:
        super().__init__()
        # 数値的安定性のために、log(1/tau) の形式で学習させることが多い（CLIP方式）
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1 / tau)))
        self.learnable = learnable

    def forward(self, query_embeds: torch.Tensor, doc_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_embeds: (Batch, D) - GMFを通した後のクエリベクトル
            doc_embeds: (Batch, D) - GMFを通した後のドキュメントベクトル
        """
        # 1. ベクトルを単位円上に正規化 (L2 Normalize)
        # Cosine Similarityを内積(Dot Product)で計算するための準備
        query_norm = F.normalize(query_embeds, p=2, dim=1)
        doc_norm = F.normalize(doc_embeds, p=2, dim=1)

        # 2. ロジットスケール（1/tau）の取得
        # スケールが大きくなりすぎないよう制限をかけるのが一般的
        logit_scale = self.logit_scale.exp()
        if self.learnable:
            logit_scale = torch.clamp(logit_scale, max=100)

        # 3. 全ペアの類似度行列を計算 (Batch, Batch)
        # logits[i, j] は クエリi と ドキュメントj の類似度
        logits = torch.matmul(query_norm, doc_norm.t()) * logit_scale

        # 4. ラベルの作成
        # 対角成分 (i == j) が正解ペア
        batch_size = query_embeds.size(0)
        labels = torch.arange(batch_size, device=query_embeds.device)

        # 5. Symmetric Cross Entropy (双方向から学習)
        # クエリから見たドキュメントの正解率と、ドキュメントから見たクエリの正解率を平均
        loss_q = F.cross_entropy(logits, labels)
        loss_d = F.cross_entropy(logits.t(), labels)

        return (loss_q + loss_d) / 2
    