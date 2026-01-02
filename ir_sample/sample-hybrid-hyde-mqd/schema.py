from dataclasses import field

from pydantic import BaseModel


class HybridQuery(BaseModel):
    perspective: str
    sub_query: str
    hypothetical_answer: str


class HQUResponse(BaseModel):
    hybrid_queries: list[HybridQuery]


class SearchResultWithMeta(BaseModel):
    doc_id: str
    score: float
    perspective: str  # どの視点でヒットしたか
    content: str      # ドキュメント本文


class CacheEntry(BaseModel):
    hqu_response: HQUResponse  # HQUResponse object
    timestamp: float


class ProfileResult(BaseModel):
    """各プロセスの実行時間を保持するデータ構造"""
    durations: dict[str, float] = field(default_factory=dict)

    def log(self, task_name: str, duration: float) -> None:
        self.durations[task_name] = round(duration * 1000, 2)  # ミリ秒変換
