"""
Citation verifier: maps citations → retrieved docs, computes coverage, flags uncited claims.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from medrag_toolkit.citation.extractor import Citation
from medrag_toolkit.retrieval.base import RetrievedDocument

# Sentences with dosages or medical terms that should have citations
_FACTUAL_CLAIM_RE = re.compile(
    r"\b(\d+\s*(?:mg|mcg|g|mEq|IU|units?)\b|"
    r"(?:contraindicated|increases?|decreases?|inhibits?|should|recommended|evidence|"
    r"study|trial|showed|demonstrated|associated with))\b",
    re.IGNORECASE,
)


@dataclass
class CitationReport:
    grounded_citations: list[Citation] = field(default_factory=list)
    hallucinated_citations: list[Citation] = field(default_factory=list)
    coverage: float = 0.0  # fraction of factual sentences with at least one citation
    uncited_factual_claims: list[str] = field(default_factory=list)
    total_citations: int = 0

    @property
    def is_well_cited(self) -> bool:
        return (
            len(self.hallucinated_citations) == 0
            and self.coverage >= 0.5
            and self.total_citations > 0
        )


def verify(
    answer: str,
    citations: list[Citation],
    retrieved_docs: list[RetrievedDocument],
) -> CitationReport:
    """
    Verify citations against retrieved documents.

    - grounded: citation id matches a retrieved doc's PMID or drug name
    - hallucinated: citation has no matching retrieved doc
    - coverage: fraction of factual sentences with >= 1 citation
    """
    # Build lookup sets from retrieved docs
    pmid_set: set[str] = set()
    drug_set: set[str] = set()
    # All retrieved content combined for drug name scanning
    all_content_lower = " ".join(doc.content.lower() for doc in retrieved_docs)

    for doc in retrieved_docs:
        if doc.source == "pubmed":
            pmid = doc.metadata.get("pmid", "")
            if pmid:
                pmid_set.add(pmid)
        elif doc.source == "drug_kb":
            drug = doc.metadata.get("drug", "")
            if drug:
                drug_set.add(drug.lower())

    grounded = []
    hallucinated = []
    for cite in citations:
        if cite.type == "pmid" and cite.id in pmid_set:
            grounded.append(cite)
        elif cite.type == "drug":
            drug_lower = cite.id.lower()
            # Grounded if drug has a drug_kb doc OR is mentioned in any retrieved content
            if drug_lower in drug_set or drug_lower in all_content_lower:
                grounded.append(cite)
            else:
                hallucinated.append(cite)
        else:
            hallucinated.append(cite)

    # Compute coverage over factual sentences
    sentences = [s.strip() for s in re.split(r"[.!?]+", answer) if s.strip()]
    cited_count = 0
    uncited_factual: list[str] = []

    for sent in sentences:
        has_citation = any(
            cite.raw in sent for cite in citations
        )
        is_factual = bool(_FACTUAL_CLAIM_RE.search(sent))

        if is_factual:
            if has_citation:
                cited_count += 1
            else:
                uncited_factual.append(sent[:120])

    factual_count = cited_count + len(uncited_factual)
    coverage = cited_count / factual_count if factual_count > 0 else 1.0

    return CitationReport(
        grounded_citations=grounded,
        hallucinated_citations=hallucinated,
        coverage=coverage,
        uncited_factual_claims=uncited_factual,
        total_citations=len(citations),
    )
