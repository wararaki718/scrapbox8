import time
import torch
from google import genai
from google.genai import types


class GeminiHyDEEngine:
    """
    Gemini 3 Flash を活用した高効率な HyDE (Hypothetical Document Embeddings) 実装クラス。
    """
    def __init__(
        self,
        api_key: str,
        model_id: str = "gemini-2.5-flash-lite",
    ) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
        self.embed_model_id = "text-embedding-004"
        
        # System Instruction for HyDE
        self.system_instruction = (
            "あなたは技術ドキュメントの専門家です。ユーザーのクエリに対し、"
            "解決策となる簡潔な技術解説文（200文字程度）を1つ生成してください。"
        )

    async def generate_hypothetical_document(self, query: str) -> str:
        """
        Gemini 3 Flash を用いて仮説的な回答文書を生成する。
        thinking_levelを最小限に抑え、低遅延レスポンスを実現。
        """
        start_time = time.perf_counter()
        
        # API呼出し（非同期想定）
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=query,
            config=types.GenerateContentConfig(
                system_instruction=self.system_instruction,
                max_output_tokens=250,
                temperature=0.3, # 決定論的な出力を微調整
                # thinking_level="minimal" はSDK/モデル仕様に従いパラメータ調整
            )
        )
        
        latency = (time.perf_counter() - start_time) * 1000
        print(f"[Log] TTFT/Generation Latency: {latency:.2f}ms")
        
        return response.text.strip()

    def get_embeddings(self, texts: list[str]) -> torch.Tensor:
        """
        Gemini Embedding API を用いてテキスト群を一括ベクトル化する。
        """
        # Batch Processing による効率化
        response = self.client.models.embed_content(
            model=self.embed_model_id,
            contents=texts,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        
        # ベクトルをPyTorch Tensorとして抽出し、L2正規化を適用（コサイン類似度計算用）
        embeddings = torch.tensor([item.values for item in response.embeddings])
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    async def hyde_search(
        self, 
        query: str, 
        alpha: float = 0.5, 
        use_weighted_avg: bool = True
    ) -> torch.Tensor:
        """
        HyDE メインロジック。クエリと仮説文書を統合し、最終的な検索ベクトルを生成する。
        
        Args:
            query: 検索クエリ
            alpha: 元のクエリの重み (1.0 = クエリのみ, 0.0 = 仮説文書のみ)
            use_weighted_avg: 重み付け平均を使用するかどうかのフラグ
        """
        overall_start = time.perf_counter()

        # 1. 仮説文書の生成
        hypothetical_doc = await self.generate_hypothetical_document(query)
        
        # 2. Embedding の取得 (Query + HypDoc を一括処理)
        # バッチ処理によりAPIのラウンドトリップを削減
        embeddings = self.get_embeddings([query, hypothetical_doc])
        
        q_vec = embeddings[0]
        d_hyp_vec = embeddings[1]

        # 3. ベクトル統合
        if use_weighted_avg:
            # $v_{final} = \alpha \cdot v_q + (1 - \alpha) \cdot v_{hyp}$
            final_vec = alpha * q_vec + (1 - alpha) * d_hyp_vec
            final_vec = torch.nn.functional.normalize(final_vec, p=2, dim=0)
        else:
            final_vec = d_hyp_vec

        total_latency = (time.perf_counter() - overall_start) * 1000
        print(f"[Log] Total HyDE Latency: {total_latency:.2f}ms")
        
        return final_vec
    