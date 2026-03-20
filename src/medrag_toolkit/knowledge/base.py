"""
Base types for the knowledge layer.

Document is the universal unit of knowledge. KnowledgeBase is a protocol
that all knowledge sources (PubMed, Drug KB) must implement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class Document:
    id: str
    content: str
    source: str  # "pubmed" | "drug_kb"
    metadata: dict = field(default_factory=dict)
    embedding: np.ndarray | None = field(default=None, repr=False)


class KnowledgeBase(Protocol):
    async def search(self, query: str, top_k: int = 10) -> list[Document]: ...
    async def build_index(self, topics: list[str]) -> None: ...
