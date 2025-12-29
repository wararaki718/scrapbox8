import os
import asyncio

from db import VectorDBInitializer
from hyde import GeminiHyDEEngine
from rag import RAGPipeline
from search import VectorSearchEngine


def main():
    # 1. コンポーネントの初期化
    api_key = os.getenv("GEMINI_API_KEY")
    hyde_engine = GeminiHyDEEngine(api_key=api_key)
    vector_db = VectorSearchEngine(dimension=768)
    
    # 2. ダミーデータの作成
    raw_docs = [f"これはドキュメント番号 {i} の内容です。特定の技術トピックが含まれます。" for i in range(200)]
    
    # 3. 初期構築の実行
    initializer = VectorDBInitializer(hyde_engine, vector_db, batch_size=50)
    initializer.run(raw_docs)

    # 4. RAG パイプラインの実行例
    pipeline = RAGPipeline(hyde_engine, vector_db)
    answer = asyncio.run(pipeline.answer_question("PyTorchの分散学習におけるエラー対処法を教えて"))
    print(answer)


if __name__ == "__main__":
    main()
