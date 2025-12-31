import asyncio

import faiss
import numpy as np
import torch


class FAISSVectorDB:
    """
    FAISS インデックスを管理し、非同期検索インターフェースを提供するクラス
    """
    def __init__(self, dimension: int=768) -> None:
        # 高速な L2 または Inner Product 検索用インデックス
        # HQUではL2正規化済みベクトルを使うため、IndexFlatIP (内積) がコサイン類似度と等価になります
        self.index = faiss.IndexFlatIP(dimension)
        self.doc_map = {}  # index_id -> doc_id (実際のID) のマッピング

    def add_documents(self, doc_ids: list[str], embeddings: torch.Tensor) -> None:
        """
        ドキュメントをインデックスに追加する
        """
        # FAISSは float32 の numpy 配列を期待する
        vecs = embeddings.detach().cpu().numpy().astype('float32')
        
        # 既存のIDとの紐付けを保存
        start_idx = self.index.ntotal
        for i, doc_id in enumerate(doc_ids):
            self.doc_map[start_idx + i] = doc_id
        
        self.index.add(vecs)
        print(f"Added {len(doc_ids)} docs. Total: {self.index.ntotal}")

    async def search_async(self, vector: list[float], limit: int=60) -> list[str]:
        """
        HQURetriever から呼ばれる非同期検索インターフェース
        """
        # FAISSの検索自体はCPU計算（またはGPU）ですが、
        # asyncio.to_thread を使いブロッキングを回避して並列性を確保します
        query_vec = np.array([vector], dtype=np.float32)
        
        _, indices = await asyncio.to_thread(
            self.index.search, query_vec, limit
        )

        # FAISS のインデックスIDを実際の Doc ID に変換
        res = []
        for idx in indices[0]:
            if idx in self.doc_map:
                res.append(self.doc_map[idx])
        return res
    
    async def search_async_with_scores(self, vector: list[float], limit: int) -> list[tuple[str, float]]:
        """
        Doc ID とそのスコア（内積/類似度）をペアで返す
        """
        query_vec = np.array([vector]).astype('float32')
        
        # D: Distances (スコア), I: Indices (ID)
        # to_thread を使い、CPUヘビーなFAISS検索中もイベントループを止めない
        D, I = await asyncio.to_thread(self.index.search, query_vec, limit)

        results = []
        for score, idx in zip(D[0], I[0]):
            if idx in self.doc_map:
                # IndexFlatIP かつ正規化済みベクトルの場合、score はコサイン類似度(0~1)になる
                results.append((self.doc_map[idx], float(score)))
        return results
