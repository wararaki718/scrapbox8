from typing import Any

from google import genai


class HQUAnswerGenerator:
    """
    メタ情報を活用して最終的な根拠付き回答を生成する
    """
    def __init__(self, client: genai.Client) -> None:
        self.client = client

    async def generate_final_answer(self, search_data: dict[str, Any]) -> str:
        # コンテキストの構築
        context_blocks = []
        for res_list in search_data["results_by_perspective"]:
            for res in res_list:
                block = f"【視点: {res.perspective}】\n内容: {res.content}"
                context_blocks.append(block)
        
        context_str = "\n\n".join(context_blocks)
        
        prompt = f"""
        あなたはユーザーの質問に対し、複数の視点から得られた情報を整理して回答するエキスパートです。
        以下の「検索コンテキスト」に基づき、ユーザーの質問に回答してください。
        
        回答の際は必ず：
        - どの視点（技術的、ユーザーの悩み、等）からの情報であるかを明記すること。
        - 視点間で情報が異なる場合は、多角的に説明すること。

        ユーザーの質問: {search_data['original_query']}

        検索コンテキスト:
        {context_str}
        """

        response = self.client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        return response.text
