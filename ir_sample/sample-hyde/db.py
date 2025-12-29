import time
import math

import torch
from google.api_core import exceptions

from hyde import GeminiHyDEEngine
from search import VectorSearchEngine


class VectorDBInitializer:
    """
    大量のドキュメントをバッチ処理で Embedding 化し、
    VectorSearchEngine (FAISS) に登録するクラス。
    """
    def __init__(
        self,
        hyde_engine: GeminiHyDEEngine,
        vector_db: VectorSearchEngine,
        batch_size: int = 50,
    ) -> None:
        self.engine = hyde_engine  # 前述の GeminiHyDEEngine
        self.db = vector_db        # 前述の VectorSearchEngine
        self.batch_size = batch_size

    def _get_embeddings_with_retry(self, batch_texts: list[str]) -> torch.Tensor:
        """
        指数バックオフを用いた Embedding 取得 (429エラー対策)
        """
        max_retries = 5
        for i in range(max_retries):
            try:
                # 前述の get_embeddings メソッドを呼び出し
                return self.engine.get_embeddings(batch_texts)
            except exceptions.ResourceExhausted as e:
                wait_time = (2 ** i) * 5 + 5  # 10s, 15s, 25s...
                print(f"[Warn] Quota exceeded. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            except Exception as e:
                print(f"[Error] Unexpected error: {e}")
                raise e
        raise Exception("Max retries exceeded for Embedding API.")

    def run(self, all_documents: list[str]) -> None:
        """
        全ドキュメントをバッチ分割して処理を実行する
        """
        total_docs = len(all_documents)
        num_batches = math.ceil(total_docs / self.batch_size)

        print(f"[Start] Processing {total_docs} docs in {num_batches} batches.")
        start_time = time.perf_counter()

        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, total_docs)
            batch_texts = all_documents[start_idx:end_idx]

            print(f"  - Batch {i+1}/{num_batches} ({len(batch_texts)} docs)...")

            # API経由でベクトル取得（リトライ制御付き）
            embeddings = self._get_embeddings_with_retry(batch_texts)

            # Vector DB (FAISS) へ登録
            self.db.add_documents(batch_texts, embeddings)

            # 無料枠の場合、バッチ間に少し遊びを入れる（安全策）
            time.sleep(2)

        duration = time.perf_counter() - start_time
        print(f"[Completed] Indexing finished in {duration:.2f}s.")
