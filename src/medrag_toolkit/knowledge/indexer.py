"""
FAISS index builder for dense retrieval.

Wraps faiss.IndexFlatIP (inner product on L2-normalized vectors = cosine sim)
with sentence-transformers for embedding.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import structlog

log = structlog.get_logger(__name__)


class FAISSIndexer:
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2") -> None:
        self._model_name = embedding_model
        self._model = None  # lazy load
        self._index = None
        self._metadata: list[dict] = []

    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts into L2-normalized vectors."""
        model = self._get_model()
        embeddings = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)
        return np.array(embeddings, dtype=np.float32)

    def build(self, documents: list) -> None:
        """Build FAISS index from Document objects."""
        import faiss

        if not documents:
            raise ValueError("Cannot build index from empty document list")

        texts = [doc.content for doc in documents]
        embeddings = self.embed(texts)

        dim = embeddings.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(embeddings)

        self._metadata = [
            {"id": doc.id, "source": doc.source, "metadata": doc.metadata, "content": doc.content}
            for doc in documents
        ]
        log.info("faiss_index_built", n_docs=len(documents), dim=dim)

    def save(self, path: Path) -> None:
        """Persist index and metadata to disk."""
        import faiss

        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "index.faiss"))
        with open(path / "metadata.pkl", "wb") as f:
            pickle.dump(self._metadata, f)
        log.info("faiss_index_saved", path=str(path))

    def load(self, path: Path) -> None:
        """Load index and metadata from disk."""
        import faiss

        self._index = faiss.read_index(str(path / "index.faiss"))
        with open(path / "metadata.pkl", "rb") as f:
            self._metadata = pickle.load(f)
        log.info("faiss_index_loaded", path=str(path), n_docs=len(self._metadata))

    def search(self, query: str, top_k: int = 10) -> list[tuple[dict, float]]:
        """Return (metadata, score) pairs for top_k nearest neighbors."""
        if self._index is None:
            raise RuntimeError("Index not built or loaded")

        q_emb = self.embed([query])
        scores, indices = self._index.search(q_emb, min(top_k, self._index.ntotal))

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx >= 0:
                results.append((self._metadata[idx], float(score)))
        return results

    @property
    def is_ready(self) -> bool:
        return self._index is not None and self._index.ntotal > 0
