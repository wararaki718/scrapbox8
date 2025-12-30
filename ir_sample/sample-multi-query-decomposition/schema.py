from pydantic import BaseModel, Field


class StepBackQueryModel(BaseModel):
    """Step-back Prompting用の構造化出力スキーマ"""
    original_query: str = Field(..., description="The original specific user query.")
    step_back_query: str = Field(..., description="A high-level, abstract concept query derived from the original.")
    specific_sub_queries: list[str] = Field(..., description="2-3 detailed sub-queries focusing on specific terms.")
    reasoning: str = Field(..., description="Reasoning for why this abstraction was chosen.")


class SearchResult(BaseModel):
    """検索エンジンのモック用結果構造"""
    doc_id: str
    score: float
    content: str
