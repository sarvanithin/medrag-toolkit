"""
Drug knowledge base combining RxNorm + OpenFDA for drug interaction retrieval.

Adapted from medguard/knowledge/openfda.py and medguard/knowledge/rxnorm.py.
Converts drug interaction data into Document objects suitable for RAG retrieval.
"""
from __future__ import annotations

import asyncio
import re
from enum import Enum
from pathlib import Path

import httpx
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from medrag_toolkit.knowledge.base import Document
from medrag_toolkit.knowledge.indexer import FAISSIndexer

log = structlog.get_logger(__name__)

_RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"
_LABEL_URL = "https://api.fda.gov/drug/label.json"
_EVENT_URL = "https://api.fda.gov/drug/event.json"

# Drug name extraction: match common drug suffix patterns + known drugs without drug-like suffixes
_DRUG_MENTION_RE = re.compile(
    r"\b([A-Za-z][a-z]{2,}(?:mycin|cillin|oxacin|olol|pril|sartan|statin|"
    r"azole|vir|mab|nib|tide|zide|olam|pam|zolam|phen|dine|mide|zone|farin|parin|"
    r"tran|xaban|gatran|oxone|lukast|terol|tropium|sonide|asone|"
    r"codone|morphine|codeine|methadone|metformin|insulin))\b",
    re.IGNORECASE,
)

_CONTRAINDICATED_RE = re.compile(
    r"\b(contraindicated|do not (use|administer|take)|must not (use|be used))\b",
    re.IGNORECASE,
)
_HIGH_RE = re.compile(
    r"\b(avoid|serious|life[- ]threatening|fatal|severe|significant(ly)? (increase|elevate))\b",
    re.IGNORECASE,
)
_MODERATE_RE = re.compile(
    r"\b(monitor|caution|adjust (dose|dosage)|use with caution|may (increase|decrease|affect))\b",
    re.IGNORECASE,
)


class InteractionSeverity(str, Enum):
    CONTRAINDICATED = "contraindicated"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    UNKNOWN = "unknown"


class RxNormClient:
    def __init__(self, http_client: httpx.AsyncClient, cache_ttl: int = 86400) -> None:
        self._http = http_client
        self._cache = _build_cache("rxnorm", cache_ttl)
        self._semaphore = asyncio.Semaphore(5)
        self._cache_ttl = cache_ttl

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=5))
    async def get_rxcui(self, drug_name: str) -> str | None:
        key = f"rxcui:{drug_name.lower().strip()}"
        if key in self._cache:
            return self._cache[key]
        async with self._semaphore:
            try:
                r = await self._http.get(
                    f"{_RXNORM_BASE}/rxcui.json",
                    params={"name": drug_name, "search": "1"},
                    timeout=10.0,
                )
                r.raise_for_status()
                rxnorm_ids = r.json().get("idGroup", {}).get("rxnormId", [])
                rxcui = rxnorm_ids[0] if rxnorm_ids else None
                self._cache.set(key, rxcui, expire=self._cache_ttl)
                return rxcui
            except httpx.HTTPStatusError:
                return None
            except Exception as exc:
                log.warning("rxnorm_error", error=str(exc), drug=drug_name)
                raise

    async def validate_drug_exists(self, drug_name: str) -> bool:
        return (await self.get_rxcui(drug_name)) is not None


class OpenFDAClient:
    def __init__(self, http_client: httpx.AsyncClient, api_timeout: float = 10.0, cache_ttl: int = 86400) -> None:
        self._http = http_client
        self._semaphore = asyncio.Semaphore(4)
        self._cache = _build_cache("openfda", cache_ttl)
        self._timeout = api_timeout
        self._cache_ttl = cache_ttl

    async def get_drug_label(self, drug_name: str) -> dict | None:
        """Fetch drug label data from OpenFDA."""
        cache_key = f"label:{drug_name.lower()}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        async with self._semaphore:
            try:
                r = await self._http.get(
                    _LABEL_URL,
                    params={"search": f'openfda.generic_name:"{drug_name}"', "limit": "1"},
                    timeout=self._timeout,
                )
                if r.status_code == 404:
                    self._cache.set(cache_key, None, expire=self._cache_ttl)
                    return None
                r.raise_for_status()
                results = r.json().get("results", [])
                data = results[0] if results else None
                self._cache.set(cache_key, data, expire=self._cache_ttl)
                return data
            except Exception as exc:
                log.debug("openfda_label_error", error=str(exc), drug=drug_name)
                return None

    async def get_drug_interactions(self, drug_a: str, drug_b: str) -> dict | None:
        """Get interaction text between two drugs from FDA labels."""
        cache_key = f"interaction:{drug_a.lower()}:{drug_b.lower()}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        async with self._semaphore:
            try:
                r = await self._http.get(
                    _LABEL_URL,
                    params={
                        "search": f'openfda.generic_name:"{drug_a}" AND drug_interactions:"{drug_b}"',
                        "limit": "3",
                    },
                    timeout=self._timeout,
                )
                if r.status_code == 404:
                    self._cache.set(cache_key, None, expire=self._cache_ttl)
                    return None
                r.raise_for_status()
                results = r.json().get("results", [])
                if not results:
                    self._cache.set(cache_key, None, expire=self._cache_ttl)
                    return None

                interaction_text = ""
                for result in results:
                    texts = result.get("drug_interactions", [])
                    if texts:
                        combined = " ".join(texts)
                        for sentence in combined.split("."):
                            if drug_b.lower() in sentence.lower():
                                interaction_text += sentence.strip() + ". "
                        if not interaction_text:
                            interaction_text = combined[:500]
                        break

                if not interaction_text:
                    self._cache.set(cache_key, None, expire=self._cache_ttl)
                    return None

                severity = _parse_severity(interaction_text)
                result_data = {
                    "drug_a": drug_a,
                    "drug_b": drug_b,
                    "severity": severity.value,
                    "description": interaction_text.strip()[:400],
                }
                self._cache.set(cache_key, result_data, expire=self._cache_ttl)
                return result_data
            except Exception as exc:
                log.debug("openfda_interaction_error", error=str(exc))
                return None


class DrugKnowledgeBase:
    """
    FAISS-backed drug knowledge base combining RxNorm + OpenFDA.

    build_index(topics) interprets topics as drug names and fetches label data.
    search(query) extracts drug names from query, fetches interactions, returns Documents.
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        index_dir: Path,
        embedding_model: str = "all-MiniLM-L6-v2",
        api_timeout: float = 10.0,
        cache_ttl: int = 86400,
    ) -> None:
        self._openfda = OpenFDAClient(http_client, api_timeout=api_timeout, cache_ttl=cache_ttl)
        self._rxnorm = RxNormClient(http_client, cache_ttl=cache_ttl)
        self._indexer = FAISSIndexer(embedding_model)
        self._index_path = index_dir / "drug_kb"
        self._documents: list[Document] = []

        if self._index_path.exists():
            try:
                self._indexer.load(self._index_path)
                log.info("drug_kb_index_loaded_from_disk")
            except Exception as exc:
                log.warning("drug_kb_index_load_failed", error=str(exc))

    async def build_index(self, topics: list[str]) -> None:
        """Fetch drug label data sequentially to respect OpenFDA rate limits."""
        docs: list[Document] = []

        for i, drug in enumerate(topics):
            if i > 0:
                await asyncio.sleep(0.3)  # OpenFDA: 240 req/min free tier
            log.info("drug_kb_fetching", drug=drug, n=f"{i+1}/{len(topics)}")

            label = await self._openfda.get_drug_label(drug)
            if not label:
                log.debug("drug_kb_no_label", drug=drug)
                continue

            interactions = label.get("drug_interactions", [])
            if interactions:
                content = f"Drug interactions for {drug}:\n" + " ".join(interactions)[:2000]
                docs.append(Document(
                    id=f"drug_interaction_{drug.lower().replace(' ', '_')}",
                    content=content,
                    source="drug_kb",
                    metadata={"drug": drug, "type": "interactions"},
                ))

            warnings = label.get("warnings", []) + label.get("contraindications", [])
            if warnings:
                content = f"Warnings and contraindications for {drug}:\n" + " ".join(warnings)[:2000]
                docs.append(Document(
                    id=f"drug_warnings_{drug.lower().replace(' ', '_')}",
                    content=content,
                    source="drug_kb",
                    metadata={"drug": drug, "type": "warnings"},
                ))

            dosage = label.get("dosage_and_administration", [])
            if dosage:
                content = f"Dosage information for {drug}:\n" + " ".join(dosage)[:2000]
                docs.append(Document(
                    id=f"drug_dosage_{drug.lower().replace(' ', '_')}",
                    content=content,
                    source="drug_kb",
                    metadata={"drug": drug, "type": "dosage"},
                ))

        if docs:
            self._documents = docs
            self._indexer.build(docs)
            self._indexer.save(self._index_path)
            log.info("drug_kb_index_built", n_docs=len(docs), drugs=len(topics))
        else:
            log.warning("drug_kb_no_documents_built")

    async def search(self, query: str, top_k: int = 10) -> list[Document]:
        """
        Search drug knowledge base.

        Extracts drug names from query, fetches live interaction data,
        and also searches the FAISS index for pre-built documents.
        """
        results: list[Document] = []

        # FAISS search if index is ready
        if self._indexer.is_ready:
            for meta, score in self._indexer.search(query, top_k):
                results.append(Document(
                    id=meta["id"],
                    content=meta["content"],
                    source=meta["source"],
                    metadata={**meta["metadata"], "score": score},
                ))

        # Live drug interaction lookup
        drug_names = _extract_drug_names(query)
        if len(drug_names) >= 2:
            for i, drug_a in enumerate(drug_names[:3]):
                for drug_b in drug_names[i + 1:i + 3]:
                    interaction = await self._openfda.get_drug_interactions(drug_a, drug_b)
                    if interaction:
                        content = (
                            f"Drug interaction between {drug_a} and {drug_b} "
                            f"(severity: {interaction['severity']}):\n{interaction['description']}"
                        )
                        results.append(Document(
                            id=f"live_interaction_{drug_a}_{drug_b}",
                            content=content,
                            source="drug_kb",
                            metadata={**interaction, "score": 0.9},
                        ))

        return results[:top_k]


def _extract_drug_names(text: str) -> list[str]:
    """Extract potential drug names from free text."""
    matches = _DRUG_MENTION_RE.findall(text)
    return list(dict.fromkeys(m.lower() for m in matches))  # dedup, preserve order


def _parse_severity(text: str) -> InteractionSeverity:
    if _CONTRAINDICATED_RE.search(text):
        return InteractionSeverity.CONTRAINDICATED
    if _HIGH_RE.search(text):
        return InteractionSeverity.HIGH
    if _MODERATE_RE.search(text):
        return InteractionSeverity.MODERATE
    return InteractionSeverity.UNKNOWN


def _build_cache(name: str, ttl_seconds: int):
    try:
        import diskcache
        cache_dir = Path.home() / ".medrag" / "cache" / name
        cache_dir.mkdir(parents=True, exist_ok=True)
        return diskcache.Cache(str(cache_dir))
    except ImportError:
        return _MemoryCache(ttl_seconds)


class _MemoryCache:
    def __init__(self, ttl: int) -> None:
        self._store: dict = {}

    def __contains__(self, key: str) -> bool:
        return key in self._store

    def __getitem__(self, key: str):
        return self._store[key]

    def set(self, key: str, value, expire: int = 0) -> None:
        self._store[key] = value
