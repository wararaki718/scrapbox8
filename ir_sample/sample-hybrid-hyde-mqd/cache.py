import time
from typing import Any

import faiss
import numpy as np
import torch

from schema import CacheEntry, HQUResponse


class HQUSemanticCache:
    """
    L1 (完全一致) と L2 (意味的類似性) を組み合わせた多段階キャッシュ
    """
    def __init__(self, dimension: int, threshold: float = 0.95, ttl: int = 86400 * 7) -> None:
        # L1: Exact Match (Memory Dict / Redis想定)
        self.l1_cache: dict[str, CacheEntry] = {}

        # L2: Semantic Match (FAISS)
        self.l2_index = faiss.IndexFlatIP(dimension)
        self.l2_metadata: list[CacheEntry] = []

        self.threshold = threshold
        self.ttl = ttl

    def _is_expired(self, entry: CacheEntry) -> bool:
        return (time.time() - entry.timestamp) > self.ttl

    async def lookup(self, query_text: str, query_vector: torch.Tensor) -> None | HQUResponse:
        """
        キャッシュ・ルックアップ・フローの実装
        """
        # --- L1: Exact Match ---
        if query_text in self.l1_cache:
            entry = self.l1_cache[query_text]
            if not self._is_expired(entry):
                print(f"[Cache] L1 Hit: {query_text}")
                return entry.hqu_response

        # --- L2: Semantic Match ---
        if self.l2_index.ntotal == 0:
            return None

        v_np = query_vector.cpu().numpy().astype(np.float32).reshape(1, -1)
        # 検索 (k=1)
        D, I = self.l2_index.search(v_np, 1)

        similarity = D[0][0]
        idx = I[0][0]

        if idx != -1 and similarity >= self.threshold:
            entry: CacheEntry = self.l2_metadata[idx]
            if not self._is_expired(entry):
                print(f"[Cache] L2 Hit: Similarity {similarity:.4f}")
                return entry.hqu_response
        return None

    def update(self, query_text: str, query_vector: torch.Tensor, hqu_response: HQUResponse) -> None:
        """
        キャッシュの更新 (L1 & L2 両方)
        """
        entry = CacheEntry(hqu_response=hqu_response, timestamp=time.time())

        # L1 更新
        self.l1_cache[query_text] = entry

        # L2 更新 (FAISS)
        v_np = query_vector.cpu().numpy().astype('float32').reshape(1, -1)
        self.l2_index.add(v_np)
        self.l2_metadata.append(entry)
