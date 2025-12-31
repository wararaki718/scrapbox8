import asyncio

from integrator import RRFIntegrator


class HQURetriever:
    """
    HQUエンジンとVector DBを接続するリトリーバー
    """
    def __init__(self, hqu_engine, vector_db_client) -> None:
        self.engine = hqu_engine
        self.db = vector_db_client
        self.rrf = RRFIntegrator(k=60)

    async def search(self, user_query: str, top_k: int = 10) -> list[tuple[str, float]]:
        # 1. Gemini + PyTorch で合成ベクトルを生成 (3つの視点)
        hqu_data, fused_vectors = await self.engine.generate_hqu_vectors(user_query)
        
        # 2. Vector DB への並列検索リクエスト
        # 3つのベクトルを非同期で同時に投げてLatencyを最小化
        tasks = [
            self.db.search_async(vec.tolist(), limit=top_k * 2) 
            for vec in fused_vectors
        ]
        
        # 全ての視点からの結果を待機
        multi_results: list[list[str]] = await asyncio.gather(*tasks)
        
        # 3. RRF によるランキング統合
        final_ranking = self.rrf.fuse(multi_results)
        
        return final_ranking[:top_k]
