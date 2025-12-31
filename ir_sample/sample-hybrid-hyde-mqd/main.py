import os
import asyncio
import torch

# 1. データ構造の定義

# 2. HQU 統合エンジンの実装
from engine import HQUEngine

# 3. 実行デモ用のモック
async def mock_embedding_api(texts: list[str] ) -> list[list[float]]:
    # 実際のAPIコールの代わりにランダムベクトルを返す（dim=768）
    return torch.randn(len(texts), 768).tolist()

async def main() -> None:
    api_key = os.getenv("GEMINI_API_KEY", "your_api_key_here")
    engine = HQUEngine(api_key)
    
    user_input = "iPhoneの画面が真っ暗で反応しない時の対処法"
    
    # HQU実行 (生成 + ベクトル合成)
    hqu_result, vectors = await engine.generate_hqu_vectors(user_input, mock_embedding_api)

    print(f"Synthesized Vectors Shape: {vectors.shape}") # [3, 768]
    for i, item in enumerate(hqu_result.hybrid_queries):
        print(f"\nPerspective {i+1}: {item.perspective}")
        print(f"Vector (norm): {torch.norm(vectors[i]):.4f}") # 正規化の確認


if __name__ == "__main__":
    asyncio.run(main())
