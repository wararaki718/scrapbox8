import torch
import torch.nn.functional as F
from google import genai
from google.genai import types

from cache import HQUSemanticCache
from processor import HQUProcessor
from schema import HQUResponse


class HQUEngine:
    def __init__(self, client: genai.Client, alpha: float = 0.35) -> None:
        self.client = client
        self.model_id = "gemini-2.5-flash-lite"
        self.embed_model_id = "text-embedding-004" # 最新のEmbeddingモデル
        self.processor = HQUProcessor(alpha=alpha)
        self.cache = HQUSemanticCache(dimension=768, threshold=0.95, ttl=86400 * 7)

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

    async def generate_hqu_vectors(self, user_query: str) -> tuple[HQUResponse, torch.Tensor]:
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

    async def process_vectors_from_cache(self, cached_hqu: HQUResponse) -> torch.Tensor:
        """
        キャッシュされたHQUデータから、直接検索用ベクトルを合成する。
        Gemini APIの呼び出しをスキップし、Embedding APIと合成演算のみを実行。
        """
        # 1. キャッシュデータからテキストを抽出
        queries = [item.sub_query for item in cached_hqu.hybrid_queries]
        hypos = [item.hypothetical_answer for item in cached_hqu.hybrid_queries]

        # 2. テキストを一括でベクトル化 (ここだけはAPIを叩くが、Gemini生成より圧倒的に速い)
        all_texts = queries + hypos
        all_embeddings = await self._get_embeddings(all_texts)
        
        # 3. テンソルのスライシング
        q_embeds = all_embeddings[:len(queries)]
        h_embeds = all_embeddings[len(queries):]

        # 4. PyTorchによる合成演算 (既存のロジックを再利用)
        with torch.no_grad():
            fused_vectors = self.processor.fuse_embeddings(
                q_embeds, 
                h_embeds
            )
        return fused_vectors

    async def generate_hqu_with_cache(self, user_query: str) -> tuple[HQUResponse, torch.Tensor]:
        # 1. クエリのベクトル化 (キャッシュ判定用)
        q_vector = await self._get_embeddings([user_query])
        
        # 2. キャッシュ確認
        cached_res = await self.cache.lookup(user_query, q_vector[0])
        if cached_res:
            # キャッシュヒット時は、このデータを使ってそのままFAISS検索へ
            return cached_res, await self.process_vectors_from_cache(cached_res)

        # 3. キャッシュミス時は Gemini を実行
        hqu_data, fused_vectors = await self.generate_hqu_vectors(user_query)
        
        # 4. 結果をキャッシュに保存
        self.cache.update(user_query, q_vector[0], hqu_data)
        
        return hqu_data, fused_vectors
