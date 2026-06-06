from typing import Any
from pydantic import BaseModel, Field

class DocumentChunk(BaseModel):
    chunk_id: str
    text: str
    source: str
    source_type: str
    page: int | None = None
    lecture_number: int | None = None
    topic: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

class RetrievedChunk(BaseModel):
    chunk: DocumentChunk
    semantic_score: float | None = None
    keyword_score: float | None = None
    metadata_score: float | None = None
    authority_score: float | None = None
    recency_score: float | None = None
    final_score: float = 0.0

class RouterDecision(BaseModel):
    intent: str
    retrieval_mode: str
    needs_quiz: bool = False
    needs_grading: bool = False
    needs_tool: bool = False
    needs_safety_check: bool = True
    reasoning: str = ""

class AgentTrace(BaseModel):
    user_query: str
    router_decision: RouterDecision
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    checker_feedback: dict[str, Any] = Field(default_factory=dict)
    final_answer: str = ""