from google import genai
from google.genai import types

from schema import StepBackQueryModel


class StepBackDecompositionEngine:
    def __init__(
        self,
        api_key: str,
        model_id: str = "gemini-2.5-flash-lite",
    ) -> None:
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id

    def generate_step_back_queries(self, user_input: str) -> StepBackQueryModel:
        """
        Step-back Promptingを実行し、抽象と具体のハイブリッドクエリを生成。
        """
        system_instruction = (
            "You are an expert search architect. Your task is 'Step-back Prompting'.\n"
            "1. Identify the core underlying concepts of the user's query.\n"
            "2. Create a 'Step-back Query' which is a more generic and high-level version of the query.\n"
            "3. Create a few specific sub-queries for technical retrieval.\n"
            "This allows for retrieving both high-level principles and low-level solutions."
        )

        # GeminiのJSON出力機能を活用
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=f"Original Query: {user_input}",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=StepBackQueryModel,
                temperature=0.2,
            ),
        )
        return response.parsed
