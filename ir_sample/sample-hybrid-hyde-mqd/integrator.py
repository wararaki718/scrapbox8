from collections import defaultdict


class RRFIntegrator:
    """
    Reciprocal Rank Fusion (RRF) を実行するクラス
    """
    def __init__(self, k: int = 60) -> None:
        # kはハイパーパラメータ。標準的には60が推奨される（低い順位の影響度を調整）
        self.k = k

    def fuse(self, search_results_list: list[list[str]]) -> list[tuple[str, float]]:
        """
        複数のランキング結果を統合し、スコアの高い順に返す
        Args:
            search_results_list: 各視点の検索結果（Document IDのリスト）のリスト
        Returns:
            Sorted list of (doc_id, score)
        """
        rrf_score = defaultdict(float)
        
        for rank_list in search_results_list:
            for rank, doc_id in enumerate(rank_list):
                # RRFのコア数式: score = sum(1 / (k + rank + 1))
                rrf_score[doc_id] += 1.0 / (self.k + rank + 1)
        
        # スコア降順でソート
        return sorted(rrf_score.items(), key=lambda x: x[1], reverse=True)
