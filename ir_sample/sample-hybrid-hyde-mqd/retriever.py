import asyncio
from typing import Any

from db import FAISSVectorDB
from engine import HQUEngine
from schema import SearchResultWithMeta


class HQUFullStackRetriever:
    """
    視点メタデータを保持しつつ、最終生成へ橋渡しする統合リトリーバー
    """
    def __init__(
        self,
        hqu_engine: HQUEngine,
        vector_db: FAISSVectorDB,
        doc_store: dict[str, str],
    ) -> None:
        self.engine = hqu_engine
        self.db = vector_db
        self.doc_store = doc_store  # IDから本文を引くための辞書

    async def search_with_perspectives(self, user_query: str, top_k: int = 5) -> dict[str, Any]:
        # 1. Geminiでクエリ分解
        hqu_data, fused_vectors = await self.engine.generate_hqu_vectors(user_query)

        # 2. 視点ごとに検索を実行し、メタ情報を保持
        tasks = []
        for i, item in enumerate(hqu_data.hybrid_queries):
            tasks.append(self._search_single_perspective(
                fused_vectors[i].tolist(), 
                item.perspective
            ))

        # perspective_results: List[List[SearchResultWithMeta]]
        perspective_results: list[list[SearchResultWithMeta]] = await asyncio.gather(*tasks)

        # 3. RRFで統合ランキングを作成（提示用）
        flattened_results = [doc.doc_id for res in perspective_results for doc in res]
        # 簡易的な統合ロジック（実際には前述のRRFIntegratorを使用）
        unique_docs = list(dict.fromkeys(flattened_results))[:top_k]

        return {
            "original_query": user_query,
            "perspectives": hqu_data.hybrid_queries,
            "results_by_perspective": perspective_results,
            "final_docs": [self.doc_store.get(did, "") for did in unique_docs]
        }

    async def _search_single_perspective(
        self,
        vector: list[float],
        perspective: str,
    ) -> list[SearchResultWithMeta]:
        # スコア付きで検索を実行
        search_hits = await self.db.search_async_with_scores(vector, limit=3)
        
        return [
            SearchResultWithMeta(
                doc_id=did,
                score=score,  # ここにFAISSからのスコアを格納
                perspective=perspective,
                content=self.doc_store.get(did, "本文なし")
            ) for did, score in search_hits
        ]
