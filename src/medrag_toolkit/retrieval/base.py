"""Base types for the retrieval layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from medrag_toolkit.knowledge.base import Document


@dataclass
class RetrievedDocument:
    id: str
    content: str
    source: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0

    @classmethod
    def from_document(cls, doc: Document, score: float = 0.0) -> "RetrievedDocument":
        return cls(
            id=doc.id,
            content=doc.content,
            source=doc.source,
            metadata=doc.metadata,
            score=score,
        )


class Retriever(Protocol):
    async def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedDocument]: ...
