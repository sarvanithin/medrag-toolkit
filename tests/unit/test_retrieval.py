"""Unit tests for retrieval layer."""
from __future__ import annotations

import numpy as np
import pytest

from medrag_toolkit.knowledge.base import Document
from medrag_toolkit.retrieval.base import RetrievedDocument
from medrag_toolkit.retrieval.hybrid import _reciprocal_rank_fusion


def _make_doc(id: str, score: float = 0.5) -> RetrievedDocument:
    return RetrievedDocument(
        id=id,
        content=f"Content of {id}",
        source="pubmed",
        metadata={"pmid": id},
        score=score,
    )


def test_rrf_merges_dense_and_sparse():
    dense = [_make_doc("a", 0.9), _make_doc("b", 0.8), _make_doc("c", 0.7)]
    sparse = [_make_doc("b", 0.95), _make_doc("d", 0.85), _make_doc("a", 0.75)]

    result = _reciprocal_rank_fusion(dense, sparse, top_k=4)
    assert len(result) <= 4
    # "a" and "b" appear in both — should rank high
    ids = [r.id for r in result]
    assert "a" in ids
    assert "b" in ids


def test_rrf_handles_empty_sparse():
    dense = [_make_doc("a", 0.9), _make_doc("b", 0.8)]
    result = _reciprocal_rank_fusion(dense, [], top_k=5)
    assert len(result) == 2


def test_rrf_handles_empty_dense():
    sparse = [_make_doc("x", 0.7)]
    result = _reciprocal_rank_fusion([], sparse, top_k=5)
    assert len(result) == 1
    assert result[0].id == "x"


def test_retrieved_document_from_document():
    doc = Document(
        id="pubmed_123",
        content="test content",
        source="pubmed",
        metadata={"pmid": "123"},
    )
    rdoc = RetrievedDocument.from_document(doc, score=0.75)
    assert rdoc.id == "pubmed_123"
    assert rdoc.score == 0.75
    assert rdoc.source == "pubmed"
