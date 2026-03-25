"""
MedRAG orchestrator: query → retrieve → generate → cite → verify → return.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import AsyncIterator

import re

import httpx
import structlog

from medrag_toolkit.citation.extractor import Citation, extract
from medrag_toolkit.citation.verifier import CitationReport, verify
from medrag_toolkit.config import Settings
from medrag_toolkit.generation.ollama import OllamaGenerator
from medrag_toolkit.hallucination.detector import HallucinationDetector, HallucinationResult
from medrag_toolkit.knowledge.drug_kb import DrugKnowledgeBase
from medrag_toolkit.knowledge.pubmed import PubMedKnowledgeBase
from medrag_toolkit.retrieval.base import RetrievedDocument

log = structlog.get_logger(__name__)


@dataclass
class MedRAGResponse:
    answer: str
    citations: list[Citation]
    retrieved_docs: list[RetrievedDocument]
    hallucination_score: float
    hallucination_flags: list
    citation_report: CitationReport
    confidence: float
    processing_time_ms: float
    model: str = ""


_DRUG_SIGNALS_RE = re.compile(
    r"\b(drug|medication|interaction|dose|dosage|contraindication|"
    r"prescri|side effect|adverse|rxnorm|openfda)\b",
    re.IGNORECASE,
)


class MedRAG:
    def __init__(self, config: Settings) -> None:
        self._config = config
        self._http = httpx.AsyncClient(timeout=config.ollama.timeout)

        self._pubmed_kb = PubMedKnowledgeBase(
            http_client=self._http,
            index_dir=config.faiss.index_dir,
            embedding_model=config.faiss.embedding_model,
            max_results=config.pubmed.max_results,
            api_key=config.pubmed.api_key,
        )
        self._drug_kb = DrugKnowledgeBase(
            http_client=self._http,
            index_dir=config.faiss.index_dir,
            embedding_model=config.faiss.embedding_model,
            api_timeout=config.drug_kb.api_timeout_seconds,
            cache_ttl=config.drug_kb.cache_ttl_seconds,
        )
        self._generator = OllamaGenerator(
            http_client=self._http,
            base_url=config.ollama.base_url,
            model=config.ollama.model,
            temperature=config.ollama.temperature,
            timeout=config.ollama.timeout,
        )
        self._detector = HallucinationDetector(
            hallucination_threshold=config.rag.hallucination_threshold,
            embedding_model=config.faiss.embedding_model,
        )

    async def query(self, question: str) -> MedRAGResponse:
        t0 = time.monotonic()

        # 1. Route query to relevant KB(s) and retrieve
        retrieved_docs = await self._retrieve(question)

        if not retrieved_docs:
            log.warning("no_documents_retrieved", question=question[:80])

        # 2. Generate answer with Ollama
        gen_response = await self._generator.generate(question, retrieved_docs)

        # 3. Extract + verify citations
        citations = extract(gen_response.answer)
        citation_report = verify(gen_response.answer, citations, retrieved_docs)

        # 4. Run hallucination detector
        hall_result: HallucinationResult = await self._detector.check(
            gen_response.answer, context_docs=retrieved_docs
        )

        # 5. Compute confidence
        confidence = _compute_confidence(
            retrieved_docs, citation_report, hall_result
        )

        processing_ms = (time.monotonic() - t0) * 1000
        log.info(
            "medrag_query_complete",
            question=question[:60],
            n_docs=len(retrieved_docs),
            n_citations=len(citations),
            hallucination_score=hall_result.hallucination_score,
            confidence=confidence,
            processing_ms=round(processing_ms),
        )

        return MedRAGResponse(
            answer=gen_response.answer,
            citations=citations,
            retrieved_docs=retrieved_docs,
            hallucination_score=hall_result.hallucination_score,
            hallucination_flags=hall_result.flags,
            citation_report=citation_report,
            confidence=confidence,
            processing_time_ms=processing_ms,
            model=gen_response.model,
        )

    async def stream_query(self, question: str) -> AsyncIterator[str]:
        """
        Stream answer tokens. Emits citation and hallucination summary at the end.
        """
        retrieved_docs = await self._retrieve(question)

        full_answer = ""
        async for token in self._generator.stream_generate(question, retrieved_docs):
            full_answer += token
            yield token

        # Post-stream: emit citation and hallucination info
        citations = extract(full_answer)
        citation_report = verify(full_answer, citations, retrieved_docs)
        hall_result = await self._detector.check(full_answer, context_docs=retrieved_docs)

        yield f"\n\n---\n[Citations: {len(citations)} | Grounded: {len(citation_report.grounded_citations)} | Hallucination score: {hall_result.hallucination_score:.2f}]"

    async def _retrieve(self, question: str) -> list[RetrievedDocument]:
        """Route query to relevant KB(s) and retrieve top-k documents."""
        top_k = self._config.faiss.top_k
        is_drug = bool(_DRUG_SIGNALS_RE.search(question))

        # Always search PubMed; search Drug KB when drug signals present
        pubmed_task = self._pubmed_kb.search(question, top_k=top_k)
        drug_task = self._drug_kb.search(question, top_k=top_k // 2) if is_drug else None

        if drug_task:
            pubmed_docs, drug_docs = await asyncio.gather(pubmed_task, drug_task)
        else:
            pubmed_docs = await pubmed_task
            drug_docs = []

        # Pin drug docs for any drugs explicitly mentioned in the question
        # so the LLM always has their label data and can cite them correctly
        from medrag_toolkit.knowledge.drug_kb import _extract_drug_names
        mentioned_drugs = _extract_drug_names(question)
        pinned: list = []
        if mentioned_drugs and self._drug_kb._indexer.is_ready:
            for drug_name in mentioned_drugs[:5]:
                for meta, score in self._drug_kb._indexer.search(drug_name, top_k=2):
                    if meta.get("metadata", {}).get("drug", "").lower() == drug_name.lower():
                        pinned.append(meta)

        combined = pubmed_docs + drug_docs

        # Convert Document → RetrievedDocument
        from medrag_toolkit.retrieval.base import RetrievedDocument
        seen_ids: set[str] = set()
        results = []

        # Add pinned drug docs first (guaranteed in context)
        for meta in pinned:
            if meta["id"] not in seen_ids:
                seen_ids.add(meta["id"])
                results.append(RetrievedDocument(
                    id=meta["id"],
                    content=meta["content"],
                    source=meta["source"],
                    metadata=meta.get("metadata", {}),
                    score=1.0,  # pinned = highest priority
                ))

        for doc in combined:
            if doc.id not in seen_ids:
                seen_ids.add(doc.id)
                score = float(doc.metadata.get("score", 0.0))
                results.append(RetrievedDocument(
                    id=doc.id,
                    content=doc.content,
                    source=doc.source,
                    metadata=doc.metadata,
                    score=score,
                ))

        results.sort(key=lambda d: d.score, reverse=True)
        return results[:top_k]

    async def close(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "MedRAG":
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()


def _compute_confidence(
    docs: list[RetrievedDocument],
    citation_report: CitationReport,
    hall_result: HallucinationResult,
) -> float:
    if not docs:
        return 0.0

    # Average retrieval score (capped at 1.0)
    avg_score = min(sum(d.score for d in docs[:5]) / len(docs[:5]), 1.0) if docs else 0.0
    citation_score = citation_report.coverage
    hallucination_penalty = hall_result.hallucination_score

    confidence = (avg_score * 0.4 + citation_score * 0.4) * (1.0 - hallucination_penalty * 0.5)
    return round(max(0.0, min(confidence, 1.0)), 3)
