import asyncio
import os

from engine import StepBackDecompositionEngine
from retriever import mock_search_provider, reciprocal_rank_fusion


async def run_step_back_pipeline(user_query: str, api_key: str) -> None:
    engine = StepBackDecompositionEngine(api_key=api_key)
    
    # クエリの抽象化と分解
    print(f"[*] Original Query: {user_query}")
    result = engine.generate_step_back_queries(user_query)
    
    print(f"\n[Step-back Query (Abstract)]")
    print(f" >> {result.step_back_query}")
    
    print(f"\n[Specific Sub-queries]")
    for q in result.specific_sub_queries:
        print(f" >> {q}")

    # 全クエリを統合して検索（抽象クエリ + 具体クエリ）
    all_queries = [result.step_back_query] + result.specific_sub_queries
    
    # 以降、前回実装した async_search_provider と RRF に渡す
    print(f"\n[*] Total {len(all_queries)} queries ready for parallel retrieval.")
    # search_tasks = [mock_search_provider(q) for q in all_queries] ...

    for q in all_queries:
        print(f" >> Ready to search with query: {q}")
    print("\n[*] Executing parallel searches...")

    search_tasks = [mock_search_provider(q) for q in all_queries]
    all_results = await asyncio.gather(*search_tasks)
    
    # Step 3: Result Fusion
    final_rankings = reciprocal_rank_fusion(all_results)
    
    print("\n[!] Top 3 Fused Results:")
    for i, res in enumerate(final_rankings[:3]):
        print(f"Rank {i+1}: {res['doc_id']} (Score: {res['rrf_score']:.5f})")


def main() -> None:
    API_KEY = os.environ.get("GEMINI_API_KEY", "YOUR_API_KEY")
    query = "Transformerの推論時にKVキャッシュを効率化してメモリ消費を抑える方法は？"
    asyncio.run(run_step_back_pipeline(query, API_KEY))
    print("\n[+] Pipeline execution completed.")


if __name__ == "__main__":
    main()
