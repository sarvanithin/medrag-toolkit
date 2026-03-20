"""Dense FAISS retriever."""
from __future__ import annotations

from medrag_toolkit.knowledge.indexer import FAISSIndexer
from medrag_toolkit.retrieval.base import RetrievedDocument


class FAISSRetriever:
    def __init__(self, indexer: FAISSIndexer) -> None:
        self._indexer = indexer

    async def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedDocument]:
        if not self._indexer.is_ready:
            return []

        results = self._indexer.search(query, top_k)
        docs = []
        for meta, score in results:
            docs.append(RetrievedDocument(
                id=meta["id"],
                content=meta["content"],
                source=meta["source"],
                metadata=meta.get("metadata", {}),
                score=score,
            ))
        return docs
