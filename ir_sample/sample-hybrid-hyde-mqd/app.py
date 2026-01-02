import asyncio
import os
from typing import Any

# これまで実装してきたクラス群を想定（1つのファイルにまとめるかインポート）
# HQUEngine, FAISSVectorDB, HQUFullStackRetriever, HQUAnswerGenerator, 
# HQUSemanticCacheSWR, ProfileResult

from google import genai

from answer import HQUAnswerGenerator
from cache import HQUSemanticCacheSWR
from db import FAISSVectorDB
from engine import HQUEngine
from retriever import HQUFullStackRetriever

class HQUProductionSystem:
    """
    HQU RAG パイプラインの全機能を統括するメインシステム
    """
    def __init__(self, api_key: str) -> None:
        # 1. 基礎コンポーネントの初期化
        client = genai.Client(api_key=api_key)
        self.engine = HQUEngine(client=client)
        self.vector_db = FAISSVectorDB(dimension=768)
        self.doc_store: dict[str, str] = {}
        
        # 2. SWR キャッシュの統合 (TTL 7日, Stale 1時間)
        self.cache = HQUSemanticCacheSWR(
            dimension=768, 
            threshold=0.95, 
            ttl=86400 * 7, 
            stale_threshold=3600
        )
        self.engine.cache = self.cache # エンジンにキャッシュを紐付け
        
        # 3. リトリーバーとジェネレーターの統合
        self.retriever = HQUFullStackRetriever(
            self.engine, self.vector_db, self.doc_store, threshold=0.6
        )
        self.generator = HQUAnswerGenerator(self.engine.client)

    async def ingest_documents(self, documents: dict[str, str]):
        """ドキュメントをシステムに登録（インデクシング）"""
        self.doc_store.update(documents)
        ids = list(documents.keys())
        texts = list(documents.values())
        
        # 埋め込みの一括取得と登録
        embeddings = await self.engine._get_embeddings(texts)
        self.vector_db.add_documents(ids, embeddings)
        print(f"[System] Ingested {len(ids)} documents into FAISS.")

    async def chat(self, user_query: str) -> dict[str, Any]:
        """ユーザーからの問い合わせに対するエンドツーエンドの処理"""
        
        # A. プロファイリング付き検索の実行
        search_result = await self.retriever.search_with_profiling(user_query)
        
        # B. 検索結果に基づく回答生成
        # search_result["results"] にはフィルタリング済みのドキュメントが含まれる
        # 検索結果が空の場合は generator 側でハンドリング
        final_answer = await self.generator.generate_final_answer({
            "original_query": user_query,
            "results_by_perspective": search_result["answer_data"].hybrid_queries if search_result["answer_data"] else [],
            "final_docs": search_result["results"],
            "perspective_meta": search_result.get("perspective_results", [])
        })

        return {
            "answer": final_answer,
            "performance": search_result["performance"],
            "cache_status": search_result["cache_status"],
            "debug_info": {
                "is_stale_update": search_result["is_stale_triggered"],
                "num_docs_retrieved": len(search_result["results"])
            }
        }

# --- 実行エントリーポイント ---
async def start_application():
    # APIキーの取得
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY を環境変数に設定してください。")

    # システムの起動
    app = HQUProductionSystem(api_key)

    # テストデータの投入
    await app.ingest_documents({
        "iphone_sh01": "iPhoneの画面が真っ暗な場合、強制再起動を試してください。手順はボリューム上げ、下げ、サイドボタン長押しです。",
        "iphone_sh02": "充電不足により画面が映らないことがあります。純正アダプタで30分以上充電してください。",
        "iphone_sh03": "内部の基板故障の場合、Apple Storeでの修理が必要です。バックアップがない場合はデータ消失の可能性があります。"
    })

    # ユーザー問い合わせシミュレーション
    queries = [
        "iPhoneが反応しない、画面も真っ暗", # 初回実行 (Miss)
        "iPhoneの画面が映らない時の直し方", # 類似実行 (L2 Hit)
    ]

    for query in queries:
        print(f"\n>>> User: {query}")
        response = await app.chat(query)
        
        print(f"Gemini: {response['answer']}")
        print(f"Stats: {response['performance']} (Cache: {response['cache_status']})")

if __name__ == "__main__":
    asyncio.run(start_application())