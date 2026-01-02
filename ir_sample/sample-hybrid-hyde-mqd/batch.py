import asyncio
import os

from google import genai

from answer import HQUAnswerGenerator
from db import FAISSVectorDB
from engine import HQUEngine
from retriever import HQUFullStackRetriever


async def main() -> None:
    # 1. GenAI クライアントの初期化
    api_key = os.getenv("GEMINI_API_KEY", "your_api_key_here")
    client = genai.Client(api_key=api_key)
    dimension = 768  # text-embedding-004
    
    # 2. コンポーネントのインスタンス化
    engine = HQUEngine(client=client)
    faiss_db = FAISSVectorDB(dimension=dimension)
    
    # ドキュメント本文を保持するストア (ID -> Content)
    # 本番では Redis や DB になりますが、ここでは Dict でシミュレート
    doc_store = {
        "doc_tech_01": "iPhoneの強制再起動: 音量を上げるボタンを押して離し、下げるボタンを押して離し、サイドボタンを保持します。",
        "doc_user_01": "画面が真っ暗な時は、まず充電ケーブルを刺して15分待ち、充電マークが出るか確認してください。",
        "doc_bg_01": "Appleのサポート規約では、水没や物理的損傷がある場合は強制再起動よりも修理が推奨されます。"
    }

    # 3. データの事前登録 (Indexing Phase)
    # 実際には登録済みのインデックスをロードすることが多いです
    print("[System] Indexing documents...")
    sample_ids = list(doc_store.keys())
    # 実際には engine._get_embeddings() などを使ってベクトル化
    sample_texts = list(doc_store.values())
    sample_vecs = await engine._get_embeddings(sample_texts) 
    faiss_db.add_documents(sample_ids, sample_vecs)

    # 4. HQU パイプラインの構築
    retriever = HQUFullStackRetriever(engine, faiss_db, doc_store)
    generator = HQUAnswerGenerator(client)

    # 5. 実行 (Query Phase)
    user_input = "iPhoneの画面が真っ暗で反応しない時の対処法"
    print(f"\n[User Query] {user_input}")
    print("[System] Analyzing query with HQU and searching FAISS...")

    # A. 視点付き検索の実行
    search_data = await retriever.search_with_perspectives(user_input, top_k=3)

    # B. 最終回答の生成
    print("[System] Generating explainable answer...")
    final_answer = await generator.generate_final_answer(search_data)

    # 6. 結果の出力
    print("\n" + "="*50)
    print("FINAL ANSWER (HQU ENHANCED)")
    print("="*50)
    print(final_answer)

    result = await retriever.search_with_profiling(user_input)

    print("\n--- Performance Profile (ms) ---")
    for task, duration in result["performance"].items():
        print(f"{task:30}: {duration:>8} ms")


if __name__ == "__main__":
    asyncio.run(main())
