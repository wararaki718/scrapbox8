from google.genai import types

from hyde import GeminiHyDEEngine
from search import VectorSearchEngine


class RAGPipeline:
    """
    HyDE検索とコンテキスト注入を組み合わせた RAG システム。
    """
    def __init__(
        self,
        hyde_engine: GeminiHyDEEngine,
        vector_db: VectorSearchEngine,
    ) -> None:
        self.engine = hyde_engine
        self.db = vector_db

        # 最終回答用のシステムプロンプト
        self.rag_system_instruction = (
            "あなたは誠実なアシスタントです。提供された【コンテキスト】の情報のみに基づいて、"
            "ユーザーの質問に正確に答えてください。コンテキストに答えが含まれていない場合は、"
            "「提供された資料の中には、その質問に対する答えが見つかりませんでした」と正直に伝えてください。"
            "回答は簡潔かつ構造的に（箇条書きなどを活用して）記述してください。"
        )

    async def answer_question(self, user_query: str, top_k: int = 3) -> str:
        """
        質問に対して検索を行い、根拠に基づいた回答を生成する。
        """
        # 1. HyDEベクトル生成 (query -> d_hyp -> vector)
        print(f"[Step 1] Generating HyDE vector for: {user_query[:30]}...")
        query_vec = await self.engine.hyde_search(user_query)

        # 2. Vector DB (FAISS) から検索
        print(f"[Step 2] Retrieving context from Vector DB...")
        search_results = self.db.search(query_vec, top_k=top_k)
        
        # 3. コンテキストの整形
        context_text = "\n---\n".join([doc for doc, score in search_results])
        
        # 4. 最終回答の生成 (RAG Prompt)
        prompt = f"""
【ユーザーの質問】:
{user_query}

【コンテキスト】:
{context_text}

上記【コンテキスト】の内容を元に、質問に答えてください。
"""

        print(f"[Step 3] Generating final answer using Gemini...")
        response = self.engine.client.models.generate_content(
            model=self.engine.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self.rag_system_instruction,
                temperature=0.0, # 忠実性を高めるためランダム性を排除
                max_output_tokens=800
            )
        )

        return response.text
