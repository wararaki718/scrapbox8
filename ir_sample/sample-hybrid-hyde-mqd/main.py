import asyncio
import os

import torch

from db import FAISSVectorDB
from engine import HQUEngine
from retriever import HQURetriever


async def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY", "your_api_key_here")

    # サンプルデータをFAISSに登録 (実際は事前にベクトル化して保存しておく想定)
    dimension = 768
    db = FAISSVectorDB(dimension=dimension)

    sample_ids = [f"manual_doc_{i}" for i in range(100)]
    sample_vecs = torch.randn(100, dimension)
    sample_vecs = torch.nn.functional.normalize(sample_vecs, p=2, dim=-1)
    db.add_documents(sample_ids, sample_vecs)

    print("Querying: 'iPhoneの画面が真っ暗な時の対処法'...")
    engine = HQUEngine(api_key)
    retriever = HQURetriever(engine, db)
    results = await retriever.search("iPhoneの画面が真っ暗な時の対処法")

    print("\n--- Final Integrated Ranking (RRF) ---")
    for doc_id, score in results:
        print(f"Document ID: {doc_id} | RRF Score: {score:.5f}")

if __name__ == "__main__":
    asyncio.run(main())
