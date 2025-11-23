import torch


class DPPMapUpdater:
    """
    DPPのGreedy MAP推定における逆行列の逐次更新 (O(N^2)) を管理するクラス。
    """
    def __init__(self, L_matrix: torch.Tensor) -> None:
        # L_matrix: 全アイテム間のカーネル行列
        self.L = L_matrix
        self.N = L_matrix.size(0)
        # 選択されたアイテムのインデックス
        self.S = [] 
        # 選択された集合Sのカーネル行列の逆行列 C = L_S^-1
        self.C = torch.Tensor([[]]) 

    def calculate_gain(self, index: int) -> float:
        """
        未選択アイテム i を追加した際の行列式の増加率（ゲイン）を計算する。
        """
        if len(self.S) == 0:
            # 最初のアイテムの場合、ゲインはL_ii
            return self.L[index, index]

        # L の i 行目から S に対応する部分ベクトル l_i を抽出
        l_i = self.L[index, self.S].view(-1, 1)
        
        # C @ l_i の計算
        Ci_li = self.C @ l_i
        
        # ゲイン = L_ii - l_i.T @ C @ l_i
        # L_ii は L の i 行 i 列の要素
        gain = self.L[index, index] - (l_i.T @ Ci_li)[0, 0]
        return gain.item()

    def update_set(self, q_index: int) -> None:
        """
        最適アイテム q_index を選択肢に追加し、逆行列 C を更新する。
        """
        k = len(self.S)
        if k == 0:
            # 最初のアイテムの場合、C_new は 1x1 の行列 [1/L_ii]
            self.C = torch.Tensor([[1.0 / self.L[q_index, q_index]]])
            self.S.append(q_index)
            return

        # q を追加した際の逆行列 C_new を更新 (Schur補行列の原理に基づく更新)
        l_q = self.L[q_index, self.S].view(-1, 1)
        C_old = self.C
        
        # ゲインを再計算 (数値安定性のため)
        d = 1.0 / self.calculate_gain(q_index)
        
        q_C_lq = C_old @ l_q
        
        # 新しい逆行列 C_new の計算
        C_new = torch.zeros((k + 1, k + 1))
        C_new[:k, :k] = C_old + d * (q_C_lq @ q_C_lq.t())
        C_new[:k, k] = -d * q_C_lq[:, 0]
        C_new[k, :k] = -d * q_C_lq[:, 0].t()
        C_new[k, k] = d
        self.C = C_new
        self.S.append(q_index)


class DPPReranker:
    def rerank(
        self,
        relevance_scores: torch.Tensor,
        item_embeddings: torch.Tensor,
        top_k: int=10,
    ) -> list[int]:
        """
        DPPのGreedy MAP推定に基づくリランキング（可読性と効率を両立）。

        Args:
            candidate_scores: torch.Tensor
            item_embeddings: torch.Tensor
            top_k (int): 最終的に選択するアイテム数
            
        Returns:
            list: リランキングされたアイテムIDのリスト
        """
        similarity_matrix = torch.cosine_similarity(
            item_embeddings.unsqueeze(1), 
            item_embeddings.unsqueeze(0),
            dim=2
        )
        N = relevance_scores.size(0)
        
        # 1. カーネル行列 L の構築 (これは必須)
        qualities = torch.clamp(relevance_scores, min=0.0)
        Q = torch.sqrt(qualities)
        L = torch.outer(Q, Q) * similarity_matrix 
        
        # 2. アップデートクラスを初期化
        updater = DPPMapUpdater(L)
        
        # 選択済みアイテムと未選択アイテムのインデックス管理
        selected_indices = []

        # (最適化のため、ブールマスクで追跡するのが理想だが、ここではリスト操作で簡潔に)
        is_selected = torch.zeros(N, dtype=torch.bool) 
        for _ in range(min(top_k, N)):
            best_gain = -torch.inf
            best_index = -1
            
            # --- 3. 次に最も行列式を増やすアイテムを見つける ---
            # 未選択のアイテムのみをループ (enumerateを使ってインデックス i と値 score を取得)
            for i in range(N):
                if is_selected[i]:
                    continue
                    
                # DPPMapUpdaterにゲイン計算を依頼
                gain = updater.calculate_gain(i)
                if gain > best_gain:
                    best_gain = gain
                    best_index = i
            
            # --- 4. 選択の実行とUpdaterの更新 ---
            if best_index != -1 and best_gain > 0:
                # 選択されたインデックスをUpdaterに渡し、逆行列Cを更新させる
                updater.update_set(best_index)
                
                # メインリストの更新
                selected_indices.append(best_index)
                is_selected[best_index] = True
            else:
                break

        return selected_indices
