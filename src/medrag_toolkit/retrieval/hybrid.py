"""
Hybrid retriever combining BM25 sparse + FAISS dense via Reciprocal Rank Fusion.

RRF score = 1/(k + rank_dense) + 1/(k + rank_sparse), k=60
"""
from __future__ import annotations

import asyncio

import structlog

from medrag_toolkit.knowledge.base import Document
from medrag_toolkit.retrieval.base import RetrievedDocument

log = structlog.get_logger(__name__)

_RRF_K = 60


class HybridRetriever:
    def __init__(self, documents: list[Document], dense_indexer, top_k: int = 10) -> None:
        self._documents = documents
        self._dense_indexer = dense_indexer
        self._top_k = top_k
        self._bm25 = None
        self._corpus: list[str] = []
        self._build_bm25()

    def _build_bm25(self) -> None:
        if not self._documents:
            return
        try:
            from rank_bm25 import BM25Okapi
            self._corpus = [doc.content for doc in self._documents]
            tokenized = [text.lower().split() for text in self._corpus]
            self._bm25 = BM25Okapi(tokenized)
        except ImportError:
            log.warning("rank_bm25_not_available_falling_back_to_dense")

    async def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedDocument]:
        k = top_k or self._top_k

        # Run both retrievers concurrently
        dense_task = asyncio.create_task(self._dense_retrieve(query, k * 2))
        sparse_task = asyncio.create_task(self._sparse_retrieve(query, k * 2))
        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)

        return _reciprocal_rank_fusion(dense_results, sparse_results, top_k=k)

    async def _dense_retrieve(self, query: str, top_k: int) -> list[RetrievedDocument]:
        if not self._dense_indexer.is_ready:
            return []
        results = self._dense_indexer.search(query, top_k)
        return [
            RetrievedDocument(
                id=meta["id"],
                content=meta["content"],
                source=meta["source"],
                metadata=meta.get("metadata", {}),
                score=score,
            )
            for meta, score in results
        ]

    async def _sparse_retrieve(self, query: str, top_k: int) -> list[RetrievedDocument]:
        if self._bm25 is None or not self._documents:
            return []

        scores = self._bm25.get_scores(query.lower().split())
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        results = []
        for idx, score in ranked:
            doc = self._documents[idx]
            results.append(RetrievedDocument(
                id=doc.id,
                content=doc.content,
                source=doc.source,
                metadata=doc.metadata,
                score=float(score),
            ))
        return results


def _reciprocal_rank_fusion(
    dense: list[RetrievedDocument],
    sparse: list[RetrievedDocument],
    top_k: int = 10,
) -> list[RetrievedDocument]:
    scores: dict[str, float] = {}
    doc_map: dict[str, RetrievedDocument] = {}

    for rank, doc in enumerate(dense):
        scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (_RRF_K + rank + 1)
        doc_map[doc.id] = doc

    for rank, doc in enumerate(sparse):
        scores[doc.id] = scores.get(doc.id, 0.0) + 1.0 / (_RRF_K + rank + 1)
        doc_map[doc.id] = doc

    ranked_ids = sorted(scores, key=lambda x: scores[x], reverse=True)[:top_k]
    results = []
    for doc_id in ranked_ids:
        doc = doc_map[doc_id]
        doc.score = scores[doc_id]
        results.append(doc)
    return results
