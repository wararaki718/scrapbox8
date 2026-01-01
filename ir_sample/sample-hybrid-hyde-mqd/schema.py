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
