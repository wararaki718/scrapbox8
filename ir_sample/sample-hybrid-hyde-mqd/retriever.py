import asyncio
from typing import Any

from db import FAISSVectorDB
from engine import HQUEngine
from profiler import HQUProfiler
from schema import ProfileResult, SearchResultWithMeta


class HQUFullStackRetriever:
    """
    視点メタデータを保持しつつ、最終生成へ橋渡しする統合リトリーバー
    """
    def __init__(
        self,
        hqu_engine: HQUEngine,
        vector_db: FAISSVectorDB,
        doc_store: dict[str, str],
        threshold: float = 0.6,
    ) -> None:
        self.engine = hqu_engine
        self.db = vector_db
        self.doc_store = doc_store  # IDから本文を引くための辞書
        self.threshold = threshold  # 類似度のしきい値

    async def _search_single_perspective(self, vector: list[float], perspective: str) -> list[SearchResultWithMeta]:
        """
        特定の視点から検索を行い、しきい値でフィルタリングする
        """
        # スコア付きで検索実行 (limit は少し多めに取っておく)
        search_hits = await self.db.search_async_with_scores(vector, limit=5)
        
        filtered_results = []
        for did, score in search_hits:
            # しきい値判定: コサイン類似度が self.threshold 未満なら除外
            if score >= self.threshold:
                filtered_results.append(
                    SearchResultWithMeta(
                        doc_id=did,
                        score=score,
                        perspective=perspective,
                        content=self.doc_store.get(did, "本文なし")
                    )
                )
        
        # フィルタリング後の結果を返す
        if not filtered_results:
            print(f"[Warning] No documents met the threshold ({self.threshold}) for perspective: {perspective}")
            
        return filtered_results

    async def search_with_perspectives(self, user_query: str, top_k: int = 5) -> dict[str, Any]:
        """
        HQUプロセス全体を実行。全視点でしきい値を超えない場合のフォールバック付き。
        """
        # hqu_data, fused_vectors = await self.engine.generate_hqu_vectors(user_query)
        hqu_data, fused_vectors = await self.engine.generate_hqu_with_cache(user_query)

        tasks = [
            self._search_single_perspective(fused_vectors[i].tolist(), item.perspective)
            for i, item in enumerate(hqu_data.hybrid_queries)
        ]
        
        perspective_results = await asyncio.gather(*tasks)
        
        # 全視点で結果が空だった場合の処理
        total_hits = sum(len(res) for res in perspective_results)
        if total_hits == 0:
            # フォールバック: しきい値を下げて再試行、または「情報なし」として扱う
            print("[System] Zero results found above threshold. Returning empty list.")
            return {
                "original_query": user_query,
                "perspectives": hqu_data.hybrid_queries,
                "results_by_perspective": [],
                "final_docs": []
            }

        # 重複を除去して最終リストを作成 (RRFなどの統合ロジックへ)
        unique_docs = []
        seen = set()
        for res_list in perspective_results:
            for res in res_list:
                if res.doc_id not in seen:
                    unique_docs.append(res.content)
                    seen.add(res.doc_id)

        return {
            "original_query": user_query,
            "perspectives": hqu_data.hybrid_queries,
            "results_by_perspective": perspective_results,
            "final_docs": unique_docs[:top_k]
        }

    async def search_with_profiling(self, user_query: str) -> dict[str, Any]:
        profile = ProfileResult()
        profiler = HQUProfiler(profile)

        # 1. キャッシュ・ルックアップ (L1/L2)
        with profiler.measure("1_cache_lookup"):
            q_vector = await self.engine._get_embeddings([user_query])
            hqu_data, is_stale = await self.engine.cache.lookup_with_swr(
                user_query, q_vector[0], self.engine
            )

        # 2. クエリ分解 (Cache Miss 時のみ Gemini を実行)
        if not hqu_data:
            with profiler.measure("2_gemini_decomposition"):
                hqu_data, fused_vectors = await self.engine.generate_hqu_vectors(user_query)
        else:
            with profiler.measure("2_vector_fusion_from_cache"):
                fused_vectors = await self.engine.process_vectors_from_cache(hqu_data)

        # 3. 並列ベクトル検索 (FAISS)
        with profiler.measure("3_parallel_faiss_search"):
            tasks = [
                self._search_single_perspective(fused_vectors[i].tolist(), item.perspective)
                for i, item in enumerate(hqu_data.hybrid_queries)
            ]
            perspective_results = await asyncio.gather(*tasks)

        # 全視点で結果が空だった場合の処理
        total_hits = sum(len(res) for res in perspective_results)
        if total_hits == 0:
            # フォールバック: しきい値を下げて再試行、または「情報なし」として扱う
            print("[System] Zero results found above threshold. Returning empty list.")
            return {
                "original_query": user_query,
                "perspectives": hqu_data.hybrid_queries,
                "results_by_perspective": [],
                "final_docs": []
            }

        # 4. 最終ランキング統合 (RRF)
        with profiler.measure("4_rrf_fusion"):
            # 重複を除去して最終リストを作成 (RRFなどの統合ロジックへ)
            unique_docs = []
            seen = set()
            for res_list in perspective_results:
                for res in res_list:
                    if res.doc_id not in seen:
                        unique_docs.append(res.content)
                        seen.add(res.doc_id)

        return {
            "answer_data": hqu_data,
            "results": unique_docs,
            "performance": profile.durations, # ミリ秒単位の計測結果
            "cache_status": "hit" if hqu_data else "miss",
            "is_stale_triggered": is_stale
        }
