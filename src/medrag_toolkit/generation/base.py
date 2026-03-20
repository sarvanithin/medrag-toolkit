"""Base types for the generation layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol

from medrag_toolkit.retrieval.base import RetrievedDocument


@dataclass
class GeneratorResponse:
    answer: str
    raw_context_used: list[str] = field(default_factory=list)
    model: str = ""


class Generator(Protocol):
    async def generate(
        self, question: str, context_docs: list[RetrievedDocument]
    ) -> GeneratorResponse: ...

    async def stream_generate(
        self, question: str, context_docs: list[RetrievedDocument]
    ) -> AsyncIterator[str]: ...
