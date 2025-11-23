import torch


class MRRReranker:
    def __init__(self, lambda_: float=0.5) -> None:
        self._lambda = lambda_

    def rerank(
        self,
        relevance_scores: torch.Tensor,
        item_embeddings: torch.Tensor, 
        top_k: int,
    ) -> list[int]:
        """
        Maximal Marginal Relevance

        Args:
            candidate_scores (torch.Tensor): 関連性スコアのテンソル (N,)
            item_embeddings : アイテムの埋め込みベクトル行列 (N x D)
            lambda_param (float): 再ランキングの多様性と関連性のトレードオフを調整するパラメータ (0 <= lambda <= 1)
            top_k (int): 選択するアイテム数
        """
        # 類似度行列の事前計算
        similarity_matrix = torch.cosine_similarity(
            item_embeddings.unsqueeze(1), 
            item_embeddings.unsqueeze(0),
            dim=2
        )
        N = relevance_scores.size(0)
        
        # 選択状態を管理するブールマスク
        is_selected = torch.zeros(N, dtype=torch.bool) 
        selected_indices = []

        for _ in range(min(top_k, N)):
            # 1. 類似度ペナルティの計算
            similarity_penalty = torch.zeros(N)
            
            if selected_indices:
                # 既に選択されたアイテムとの類似度を抽出 (O(N*M)で計算)
                # similarity_matrix[:, selected_indices] は (N, M) の行列
                # その各行の最大値（Max Similarity）を計算
                max_similarity_to_selected = torch.max(similarity_matrix[:, selected_indices], dim=1).values
                
                # 類似度の項 = (1 - lambda) * Max Similarity
                similarity_penalty = (1.0 - self._lambda) * max_similarity_to_selected

            # 2. MMRスコアの計算 (ベクトル化)
            # MMR(d_i) = lambda * Relevance(d_i) - Similarity_Penalty
            mmr_scores = (self._lambda * relevance_scores) - similarity_penalty
            
            # 3. 既に選択されたアイテムのスコアを負の無限大に設定（選択から除外）
            mmr_scores[is_selected] = -torch.inf
            
            # 4. 次に選択すべきアイテムのインデックスを決定
            best_index = torch.argmax(mmr_scores)
            
            if mmr_scores[best_index] == -torch.inf:
                # 候補が残っていない場合は終了
                break

            # 5. 選択リストを更新
            selected_indices.append(best_index.item())
            is_selected[best_index] = True

        return selected_indices
