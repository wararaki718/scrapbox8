import torch
from google import genai
from google.genai import types

from processor import HQUProcessor
from schema import HQUResponse


class HQUEngine:
    def __init__(self, client: genai.Client, alpha: float = 0.35) -> None:
        self.client = client
        self.model_id = "gemini-2.5-flash-lite"
        self.embed_model_id = "text-embedding-004" # 最新のEmbeddingモデル
        self.processor = HQUProcessor(alpha=alpha)

    async def _get_embeddings(self, texts: list[str]) -> torch.Tensor:
        """
        Google GenAI API を使用して一括でベクトルを取得する
        """
        # SDKの batch_embed_contents を使用
        response = self.client.models.embed_content(
            model=self.embed_model_id,
            contents=texts,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
        )
        # response.embeddings[i].values にベクトルが格納されている
        vectors = [e.values for e in response.embeddings]
        return torch.tensor(vectors, dtype=torch.float32)

    async def generate_hqu_vectors(self, user_query: str):
        """
        HQUプロセス全体を実行
        """
        # Step 1: Gemini によるクエリ分解と仮説生成
        prompt = f"ユーザーのクエリを3つの視点に分解し、仮説回答を生成してください: {user_query}"
        
        gen_response = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=HQUResponse,
                temperature=0.2,
            ),
        )
        hqu_data: HQUResponse = gen_response.parsed

        # Step 2: テキストの抽出 (Sub-Query と Hypothetical Answer)
        queries = [item.sub_query for item in hqu_data.hybrid_queries]
        hypos = [item.hypothetical_answer for item in hqu_data.hybrid_queries]

        # Step 3: 一括ベクトル化 (API Call を 1回に集約して高速化)
        # 全てのテキスト(計6個)を一度に送り、ネットワークRTTを削減
        all_texts = queries + hypos
        all_embeddings = await self._get_embeddings(all_texts)
        
        # 取得したテンソルを分割 [3, dim] ずつ
        q_embeds = all_embeddings[:len(queries)]
        h_embeds = all_embeddings[len(queries):]

        # Step 4: PyTorch によるベクトル合成
        with torch.no_grad():
            fused_vectors = self.processor.fuse_embeddings(
                q_embeds, 
                h_embeds
            )

        return hqu_data, fused_vectors
