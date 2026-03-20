"""
PubMed knowledge base for medical RAG.

Adapted from medguard/knowledge/pubmed.py — adds PubMedKnowledgeBase wrapper
that builds and queries a FAISS index from PubMed abstracts.
"""
from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import structlog

from medrag_toolkit.knowledge.base import Document
from medrag_toolkit.knowledge.indexer import FAISSIndexer

log = structlog.get_logger(__name__)

_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"


@dataclass
class PubMedArticle:
    pmid: str
    title: str
    abstract: str
    journal: str = ""
    year: str = ""


class PubMedClient:
    """
    Lightweight async PubMed client.

    Rate limits:
      - Without API key: 2 req/s (conservative)
      - With NCBI_API_KEY env var: 8 req/s
    """

    def __init__(self, http_client: httpx.AsyncClient, max_results: int = 20) -> None:
        self._http = http_client
        self._max_results = max_results
        self._api_key = os.environ.get("NCBI_API_KEY", "")
        rps = 8 if self._api_key else 2
        self._semaphore = asyncio.Semaphore(int(rps))

    def _params(self, **kwargs) -> dict:
        p = {"db": "pubmed", **kwargs}
        if self._api_key:
            p["api_key"] = self._api_key
        return p

    async def search(self, query: str) -> list[str]:
        async with self._semaphore:
            try:
                r = await self._http.get(
                    _ESEARCH,
                    params=self._params(
                        term=query,
                        retmax=self._max_results,
                        retmode="json",
                        sort="relevance",
                    ),
                    timeout=8.0,
                )
                r.raise_for_status()
                return r.json().get("esearchresult", {}).get("idlist", [])
            except Exception as exc:
                log.debug("pubmed_search_failed", query=query[:60], error=str(exc))
                return []

    async def fetch_summaries(self, pmids: list[str]) -> list[PubMedArticle]:
        if not pmids:
            return []
        async with self._semaphore:
            try:
                r = await self._http.get(
                    _ESUMMARY,
                    params=self._params(id=",".join(pmids), version="2.0", retmode="json"),
                    timeout=8.0,
                )
                r.raise_for_status()
                data = r.json()
                articles = []
                result = data.get("result", {})
                for pmid in pmids:
                    doc = result.get(pmid, {})
                    if not doc:
                        continue
                    articles.append(PubMedArticle(
                        pmid=pmid,
                        title=doc.get("title", ""),
                        abstract="",
                        journal=doc.get("source", ""),
                        year=doc.get("pubdate", "")[:4],
                    ))
                return articles
            except Exception as exc:
                log.debug("pubmed_summary_failed", error=str(exc))
                return []

    async def fetch_abstracts(self, pmids: list[str]) -> list[PubMedArticle]:
        if not pmids:
            return []
        async with self._semaphore:
            try:
                r = await self._http.get(
                    _EFETCH,
                    params=self._params(id=",".join(pmids), rettype="abstract", retmode="text"),
                    timeout=10.0,
                )
                r.raise_for_status()
                return _parse_text_abstracts(pmids, r.text)
            except Exception as exc:
                log.debug("pubmed_fetch_failed", error=str(exc))
                return []


class PubMedKnowledgeBase:
    """
    FAISS-backed PubMed knowledge base.

    build_index(topics) fetches abstracts for each topic and indexes them.
    search(query) returns Document objects ranked by embedding similarity.
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        index_dir: Path,
        embedding_model: str = "all-MiniLM-L6-v2",
        max_results: int = 20,
        api_key: str = "",
    ) -> None:
        if api_key:
            os.environ.setdefault("NCBI_API_KEY", api_key)
        self._client = PubMedClient(http_client, max_results=max_results)
        self._indexer = FAISSIndexer(embedding_model)
        self._index_path = index_dir / "pubmed"
        self._documents: list[Document] = []

        if self._index_path.exists():
            try:
                self._indexer.load(self._index_path)
                log.info("pubmed_index_loaded_from_disk")
            except Exception as exc:
                log.warning("pubmed_index_load_failed", error=str(exc))

    async def build_index(self, topics: list[str]) -> None:
        """Fetch abstracts for each topic and build FAISS index."""
        all_docs: list[Document] = []

        async def fetch_topic(topic: str) -> list[Document]:
            pmids = await self._client.search(topic)
            if not pmids:
                return []
            summaries, articles = await asyncio.gather(
                self._client.fetch_summaries(pmids),
                self._client.fetch_abstracts(pmids[:10]),
            )
            abstract_map = {a.pmid: a.abstract for a in articles}
            docs = []
            for s in summaries:
                s.abstract = abstract_map.get(s.pmid, "")
                content = f"{s.title}\n\n{s.abstract}".strip()
                if not content:
                    continue
                docs.append(Document(
                    id=f"pubmed_{s.pmid}",
                    content=content,
                    source="pubmed",
                    metadata={
                        "pmid": s.pmid,
                        "title": s.title,
                        "journal": s.journal,
                        "year": s.year,
                        "topic": topic,
                    },
                ))
            return docs

        results = await asyncio.gather(*[fetch_topic(t) for t in topics])
        for docs in results:
            all_docs.extend(docs)

        # Deduplicate by PMID
        seen: set[str] = set()
        unique_docs = []
        for doc in all_docs:
            pmid = doc.metadata.get("pmid", doc.id)
            if pmid not in seen:
                seen.add(pmid)
                unique_docs.append(doc)

        self._documents = unique_docs
        self._indexer.build(unique_docs)
        self._indexer.save(self._index_path)
        log.info("pubmed_index_built", n_docs=len(unique_docs), topics=len(topics))

    async def search(self, query: str, top_k: int = 10) -> list[Document]:
        if not self._indexer.is_ready:
            log.warning("pubmed_index_not_ready")
            return []

        results = self._indexer.search(query, top_k)
        docs = []
        for meta, score in results:
            docs.append(Document(
                id=meta["id"],
                content=meta["content"],
                source=meta["source"],
                metadata={**meta["metadata"], "score": score},
            ))
        return docs


# ---------------------------------------------------------------------------
# Helpers (adapted from medguard)
# ---------------------------------------------------------------------------

def _parse_text_abstracts(pmids: list[str], raw: str) -> list[PubMedArticle]:
    articles = []
    blocks = re.split(r"\n\n\d+\.\s", "\n\n" + raw)[1:]
    for i, block in enumerate(blocks):
        pmid = pmids[i] if i < len(pmids) else str(i)
        lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
        title = lines[0] if lines else ""
        abstract = ""
        in_abstract = False
        for line in lines:
            if line.startswith("Abstract"):
                in_abstract = True
                abstract = line[len("Abstract"):].strip()
            elif in_abstract:
                if re.match(r"^(Author|PMID|DOI|Copyright|©)", line):
                    break
                abstract += " " + line
        articles.append(PubMedArticle(pmid=pmid, title=title, abstract=abstract.strip()))
    return articles
