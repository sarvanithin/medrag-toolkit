"""
FastAPI route handlers for medrag-toolkit.

Routes:
  POST /v1/query        — full Q&A with evidence
  POST /v1/query/stream — SSE streaming query
  POST /v1/index/build  — build/update KB index
  GET  /v1/health       — dependency health check
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from medrag_toolkit.api.models import (
    CitationInfo,
    CitationReportInfo,
    HallucinationFlagInfo,
    HealthResponse,
    IndexBuildRequest,
    IndexBuildResponse,
    QueryRequest,
    QueryResponse,
)
from medrag_toolkit.core import MedRAG

router = APIRouter()


def _get_medrag(request: Request) -> MedRAG:
    return request.app.state.medrag


@router.post("/v1/query", response_model=QueryResponse)
async def query(req: QueryRequest, medrag: MedRAG = Depends(_get_medrag)) -> QueryResponse:
    try:
        resp = await medrag.query(req.question)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}")

    return QueryResponse(
        answer=resp.answer,
        citations=[
            CitationInfo(type=c.type, id=c.id, raw=c.raw)
            for c in resp.citations
        ],
        retrieved_doc_count=len(resp.retrieved_docs),
        hallucination_score=resp.hallucination_score,
        hallucination_flags=[
            HallucinationFlagInfo(
                type=f.type.value,
                text=f.text[:100],
                confidence=f.confidence,
                explanation=f.explanation,
            )
            for f in resp.hallucination_flags
        ],
        citation_report=CitationReportInfo(
            grounded=len(resp.citation_report.grounded_citations),
            hallucinated=len(resp.citation_report.hallucinated_citations),
            coverage=resp.citation_report.coverage,
            uncited_claims=resp.citation_report.uncited_factual_claims[:5],
            is_well_cited=resp.citation_report.is_well_cited,
        ),
        confidence=resp.confidence,
        processing_time_ms=resp.processing_time_ms,
        model=resp.model,
    )


@router.post("/v1/query/stream")
async def query_stream(
    req: QueryRequest, medrag: MedRAG = Depends(_get_medrag)
) -> StreamingResponse:
    async def event_generator():
        try:
            async for token in medrag.stream_query(req.question):
                yield f"data: {token}\n\n"
        except RuntimeError as exc:
            yield f"data: [ERROR: {exc}]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.post("/v1/index/build", response_model=IndexBuildResponse)
async def build_index(
    req: IndexBuildRequest, medrag: MedRAG = Depends(_get_medrag)
) -> IndexBuildResponse:
    try:
        if req.source in ("pubmed", "all"):
            await medrag._pubmed_kb.build_index(req.topics)
        if req.source in ("drug_kb", "all"):
            await medrag._drug_kb.build_index(req.topics)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Index build failed: {exc}")

    return IndexBuildResponse(
        status="ok",
        source=req.source,
        topics_processed=len(req.topics),
        message=f"Index built for {len(req.topics)} topics from {req.source}",
    )


@router.get("/v1/health", response_model=HealthResponse)
async def health(medrag: MedRAG = Depends(_get_medrag)) -> HealthResponse:
    # Check Ollama connectivity
    ollama_status = "ok"
    try:
        r = await medrag._http.get(
            f"{medrag._config.ollama.base_url}/api/tags", timeout=3.0
        )
        r.raise_for_status()
    except Exception as exc:
        ollama_status = f"unavailable: {exc}"

    pubmed_status = "ready" if medrag._pubmed_kb._indexer.is_ready else "not indexed"
    drug_kb_status = "ready" if medrag._drug_kb._indexer.is_ready else "not indexed"

    overall = "ok" if ollama_status == "ok" else "degraded"
    return HealthResponse(
        status=overall,
        ollama=ollama_status,
        pubmed_index=pubmed_status,
        drug_kb_index=drug_kb_status,
    )
