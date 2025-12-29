import torch
import faiss


class VectorSearchEngine:
    """
    FAISS を利用したベクトル検索エンジン。
    Gemini の Embedding (text-embedding-004) を効率的に管理する。
    """
    def __init__(self, dimension: int = 768) -> None:
        # text-embedding-004 のデフォルト次元数は 768
        # IndexFlatIP は「内積（Inner Product）」による類似度検索用。
        # 正規化済みベクトルを使用することで、コサイン類似度と同等になる。
        self.index = faiss.IndexFlatIP(dimension)
        self.documents = []  # インデックスに対応する元のテキストを保持

    def add_documents(self, texts: list[str], embeddings: torch.Tensor) -> None:
        """
        ドキュメントとそのベクトルを DB に登録する。
        """
        # PyTorch Tensor を numpy (float32) に変換
        vectors = embeddings.detach().cpu().numpy().astype('float32')
        
        # FAISS インデックスに追加
        self.index.add(vectors)
        self.documents.extend(texts)
        print(f"[Log] Added {len(texts)} documents to Vector DB.")

    def search(self, query_vector: torch.Tensor, top_k: int = 3) -> list[tuple[str, float]]:
        """
        クエリベクトルに類似するドキュメントを検索する。
        """
        # 検索クエリを numpy 形式に整形 [1, dimension]
        q_vec = query_vector.detach().cpu().numpy().astype('float32')
        if q_vec.ndim == 1:
            q_vec = q_vec.reshape(1, -1)

        # 検索実行: D は距離（内積スコア）、I はインデックス番号
        distances, indices = self.index.search(q_vec, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # 該当なしを除外
                results.append((self.documents[idx], distances[0][i]))
        
        return results
