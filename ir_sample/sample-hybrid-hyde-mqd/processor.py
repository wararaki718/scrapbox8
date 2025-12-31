import torch
import torch.nn.functional as F


class HQUProcessor(torch.nn.Module):
    """
    Geminiの推論結果をPyTorchで高速にベクトル合成するプロセッサ
    """
    def __init__(self, alpha: float = 0.35, device: str = "cpu") -> None:
        super().__init__()
        self.alpha = alpha
        self.device = device

    def fuse_embeddings(self, q_embeds: torch.Tensor, h_embeds: torch.Tensor) -> torch.Tensor:
        """
        数式: v_search = alpha * Embed(q) + (1 - alpha) * Embed(d_hyp)
        """
        q_embeds = q_embeds.to(self.device)
        h_embeds = h_embeds.to(self.device)
        
        # 加重平均とL2正規化 (バッチ処理対応)
        v_search = (self.alpha * q_embeds) + ((1 - self.alpha) * h_embeds)
        return F.normalize(v_search, p=2, dim=-1)
