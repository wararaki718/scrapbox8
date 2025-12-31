import torch
from google import genai
from google.genai import types

from processor import HQUProcessor
from schema import HQUResponse


class HQUEngine:
    def __init__(self, api_key: str, alpha: float = 0.35) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model_id = "gemini-2.5-flash-lite"
        self.processor = HQUProcessor(alpha=alpha)

    async def generate_hqu_vectors(self, user_query: str, embed_fn) -> tuple[HQUResponse, torch.Tensor]:
        """
        1. Geminiでクエリ分解 & 仮説生成
        2. テキストを一括でベクトル化
        3. PyTorchで合成
        """
        # --- Step 1: Geminiによる構造化生成 ---
        prompt = f"""
        あなたは検索エンジン最適化の専門家です。
        ユーザーのクエリを「技術的」「意図・悩み」「背景」の3つの視点に分解し、
        それぞれに対する仮説回答(150文字程度)をJSONで出力してください。
        Query: {user_query}
        """

        response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=HQUResponse,
                temperature=0.2,
            ),
        )
        hqu_data: HQUResponse = response.parsed

        # --- Step 2: テキストの抽出と一括ベクトル化 ---
        # 効率化のため、全テキストをフラットなリストにして1回のAPIコールで送るのが理想
        queries = [item.sub_query for item in hqu_data.hybrid_queries]
        hypos = [item.hypothetical_answer for item in hqu_data.hybrid_queries]

        # ※ embed_fn は外部のEmbeddingモデル(text-embedding-004等)を想定
        q_vectors = await embed_fn(queries) # Shape: [3, Dim]
        h_vectors = await embed_fn(hypos)   # Shape: [3, Dim]

        # --- Step 3: PyTorchによるベクトル合成 ---
        with torch.no_grad():
            fused_vectors = self.processor.fuse_embeddings(
                torch.tensor(q_vectors), 
                torch.tensor(h_vectors)
            )

        return hqu_data, fused_vectors
