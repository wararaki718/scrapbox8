import asyncio
from typing import Any

from schema import SearchResult


async def mock_search_provider(query: str) -> list[SearchResult]:
    """
    検索エンジンのシミュレーター (Async I/Oを想定)
    実際にはここでVector DB (Pinecone/Milvus) や Elasticsearch へリクエストを送る。
    """
    await asyncio.sleep(0.1)  # I/Oレイテンシのシミュレーション
    # ダミーの結果を返す
    return [
        SearchResult(doc_id=f"doc_{query[:2]}_{i}", score=1.0/(i+1), content=f"Content for {query}")
        for i in range(5)
    ]

def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60) -> list[dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF) アルゴリズムによるランキング統合。
    $RRFscore(d \in D) = \sum_{r \in R} \frac{1}{k + r(d)}$
    """
    fused_scores = {}
    
    for results in results_list:
        for rank, res in enumerate(results, start=1):
            if res.doc_id not in fused_scores:
                fused_scores[res.doc_id] = 0.0
            fused_scores[res.doc_id] += 1.0 / (k + rank)
            
    # スコアの高い順にソート
    sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [{"doc_id": doc_id, "rrf_score": score} for doc_id, score in sorted_results]
