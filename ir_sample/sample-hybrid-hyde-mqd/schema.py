from pydantic import BaseModel


class HybridQuery(BaseModel):
    perspective: str
    sub_query: str
    hypothetical_answer: str

class HQUResponse(BaseModel):
    hybrid_queries: list[HybridQuery]
