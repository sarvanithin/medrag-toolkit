"""
Ollama LLM generator for medical RAG.

Enforces citation grounding via system prompt.
Supports both streaming and non-streaming generation.
"""
from __future__ import annotations

import json
from typing import AsyncIterator

import httpx
import structlog

from medrag_toolkit.generation.base import GeneratorResponse
from medrag_toolkit.retrieval.base import RetrievedDocument

log = structlog.get_logger(__name__)

_SYSTEM_PROMPT = """You are a medical AI assistant. Your role is to answer medical questions accurately based ONLY on the provided context documents.

CRITICAL RULES:
1. Answer using ONLY information from the provided context. Do not use external knowledge.
2. You MUST cite sources inline using these exact formats:
   - For PubMed articles: [PMID:12345678]
   - For drug information: [Drug:drug_name]
3. Every factual claim must have a citation.
4. If the context does not contain enough information to answer the question, say: "The provided context does not contain sufficient information to answer this question."
5. Never make up citations or PMIDs.
6. Acknowledge uncertainty — never use absolute terms like "always", "definitely", "guaranteed".
"""


class OllamaGenerator:
    def __init__(
        self,
        http_client: httpx.AsyncClient,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        temperature: float = 0.1,
        timeout: float = 60.0,
    ) -> None:
        self._http = http_client
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._timeout = timeout

    def _build_context(self, docs: list[RetrievedDocument]) -> str:
        sections = []
        for i, doc in enumerate(docs, 1):
            source_ref = _format_source_ref(doc)
            sections.append(f"[Document {i}] {source_ref}\n{doc.content[:800]}")
        return "\n\n---\n\n".join(sections)

    async def generate(
        self, question: str, context_docs: list[RetrievedDocument]
    ) -> GeneratorResponse:
        context = self._build_context(context_docs)
        user_msg = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer with citations:"

        try:
            r = await self._http.post(
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "stream": False,
                    "options": {"temperature": self._temperature},
                },
                timeout=self._timeout,
            )
            r.raise_for_status()
            data = r.json()
            answer = data.get("message", {}).get("content", "")
            return GeneratorResponse(
                answer=answer,
                raw_context_used=[doc.content for doc in context_docs],
                model=self._model,
            )
        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self._base_url}. "
                "Run: ollama serve && ollama pull llama3.2"
            )
        except Exception as exc:
            log.error("ollama_generate_failed", error=str(exc))
            raise

    async def stream_generate(
        self, question: str, context_docs: list[RetrievedDocument]
    ) -> AsyncIterator[str]:
        context = self._build_context(context_docs)
        user_msg = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer with citations:"

        try:
            async with self._http.stream(
                "POST",
                f"{self._base_url}/api/chat",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    "stream": True,
                    "options": {"temperature": self._temperature},
                },
                timeout=self._timeout,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue
        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self._base_url}. "
                "Run: ollama serve && ollama pull llama3.2"
            )


def _format_source_ref(doc: RetrievedDocument) -> str:
    if doc.source == "pubmed":
        pmid = doc.metadata.get("pmid", "")
        title = doc.metadata.get("title", "")
        year = doc.metadata.get("year", "")
        return f"[PMID:{pmid}] {title} ({year})" if pmid else title
    elif doc.source == "drug_kb":
        drug = doc.metadata.get("drug", "")
        dtype = doc.metadata.get("type", "")
        return f"[Drug:{drug}] {dtype}"
    return doc.id
