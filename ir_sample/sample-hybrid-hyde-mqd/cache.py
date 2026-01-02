import asyncio
import time
from typing import Any, Optional

import faiss
import numpy as np
import torch

# from engine import HQUEngine
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


class HQUSemanticCacheSWR(HQUSemanticCache):
    """
    Stale-While-Revalidate 戦略を搭載したセマンティック・キャッシュ
    """
    def __init__(self, *args, stale_threshold: int = 3600, **kwargs) -> None:
        """
        Args:
            stale_threshold: この時間を過ぎていたら「古い(Stale)」とみなし、バックグラウンド更新を開始する（秒）
        """
        super().__init__(*args, **kwargs)
        self.stale_threshold = stale_threshold
        self.updating_keys = set() # 重複更新防止用のロック

    def _internal_lookup(self, query_text: str, query_vector: torch.Tensor) -> tuple[Any, float] | None:
        """
        L1/L2キャッシュを検索し、(データ, タイムスタンプ) のペアを返す。
        判定ロジックを含まない「純粋な抽出」を行う内部メソッド。
        """
        # --- 1. L1: Exact Match (完全一致) ---
        if query_text in self.l1_cache:
            entry = self.l1_cache[query_text]
            return entry.hqu_response, entry.timestamp

        # --- 2. L2: Semantic Match (FAISS) ---
        if self.l2_index.ntotal == 0:
            return None

        # FAISS検索用にnumpy変換
        v_np = query_vector.cpu().numpy().astype('float32').reshape(1, -1)
        
        # 最も近い1件を検索
        D, I = self.l2_index.search(v_np, 1)
        
        similarity = D[0][0]
        idx = I[0][0]

        # 指示書の閾値 (0.95等) を超えているか確認
        if idx != -1 and similarity >= self.threshold:
            # L2用メタデータリストから該当エントリを抽出
            entry: CacheEntry = self.l2_metadata[idx]
            return entry.hqu_response, entry.timestamp
        
        return None

    async def lookup_with_swr(self, query_text: str, query_vector: torch.Tensor, engine_ref: Any) -> tuple[Optional[Any], bool]:
        """
        キャッシュを検索し、古い場合はバックグラウンド更新をスケジュールする
        Returns: (hqu_response, is_stale)
        """
        # 既存のロジックでキャッシュを検索
        # (内部で L1/L2 検索を行う lookup メソッドを呼び出す想定)
        entry_data = self._internal_lookup(query_text, query_vector)
        
        if not entry_data:
            return None, False

        hqu_res, timestamp = entry_data
        elapsed = time.time() - timestamp
        
        # 1. 完全に期限切れ(TTL)の場合は Miss 扱い
        if elapsed > self.ttl:
            return None, False

        # 2. Stale（古い）だが TTL 以内の場合：結果を返しつつ裏で更新
        if elapsed > self.stale_threshold:
            if query_text not in self.updating_keys:
                print(f"[SWR] Cache is stale ({elapsed:.0f}s). Background refresh started.")
                self.updating_keys.add(query_text)
                
                # メインフローを止めずにバックグラウンドタスクを生成
                asyncio.create_task(
                    self._background_refresh(query_text, engine_ref)
                )
            return hqu_res, True

        # 3. Fresh な場合
        return hqu_res, False

    async def _background_refresh(self, query_text: str, engine_ref: Any) -> None:
        """
        Gemini API を呼び出してキャッシュを最新化するバックグラウンドタスク
        """
        try:
            # 最新の Gemini 推論を実行
            # engine_ref は HQUEngine のインスタンス
            new_hqu_data, _ = await engine_ref.generate_hqu_vectors(query_text, use_cache=False)
            
            # キャッシュを更新 (最新のベクトルで再登録)
            q_vector = await engine_ref._get_embeddings([query_text])
            self.update(query_text, q_vector[0], new_hqu_data)
            
            print(f"[SWR] Cache refreshed for: {query_text}")
        except Exception as e:
            print(f"[SWR] Refresh failed: {e}")
        finally:
            self.updating_keys.discard(query_text)
