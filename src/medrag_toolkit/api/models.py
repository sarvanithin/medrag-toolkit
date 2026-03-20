"""FastAPI request/response models."""
from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=50)


class CitationInfo(BaseModel):
    type: str
    id: str
    raw: str


class CitationReportInfo(BaseModel):
    grounded: int
    hallucinated: int
    coverage: float
    uncited_claims: list[str]
    is_well_cited: bool


class HallucinationFlagInfo(BaseModel):
    type: str
    text: str
    confidence: float
    explanation: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationInfo]
    retrieved_doc_count: int
    hallucination_score: float
    hallucination_flags: list[HallucinationFlagInfo]
    citation_report: CitationReportInfo
    confidence: float
    processing_time_ms: float
    model: str


class IndexBuildRequest(BaseModel):
    source: str = Field(..., pattern="^(pubmed|drug_kb|all)$")
    topics: list[str] = Field(..., min_length=1, max_items=50)


class IndexBuildResponse(BaseModel):
    status: str
    source: str
    topics_processed: int
    message: str


class HealthResponse(BaseModel):
    status: str
    ollama: str
    pubmed_index: str
    drug_kb_index: str
