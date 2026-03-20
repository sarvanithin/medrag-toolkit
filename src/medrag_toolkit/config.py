"""
Configuration for medrag-toolkit.

All settings load from environment variables or ~/.medrag/config.json.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OllamaConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OLLAMA_", extra="ignore")

    base_url: str = "http://localhost:11434"
    model: str = "llama3.2"
    temperature: float = 0.1
    timeout: float = 60.0


class FAISSConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FAISS_", extra="ignore")

    index_dir: Path = Path.home() / ".medrag" / "indices"
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k: int = 10


class PubMedConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="NCBI_", extra="ignore")

    api_key: str = Field(default="", alias="NCBI_API_KEY")
    rate_limit: float = Field(default=2.0)  # req/s without key, 8 with key
    max_results: int = 20
    cache_ttl_seconds: int = 86400  # 24h

    model_config = SettingsConfigDict(env_prefix="PUBMED_", extra="ignore", populate_by_name=True)


class DrugKBConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DRUG_", extra="ignore")

    openfda_api_key: str = ""
    rxnorm_base_url: str = "https://rxnav.nlm.nih.gov/REST"
    openfda_label_url: str = "https://api.fda.gov/drug/label.json"
    api_timeout_seconds: float = 10.0
    cache_ttl_seconds: int = 86400


class RAGConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", extra="ignore")

    min_confidence: float = 0.4
    max_context_tokens: int = 4000
    require_citations: bool = True
    hallucination_threshold: float = 0.3


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")

    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    faiss: FAISSConfig = Field(default_factory=FAISSConfig)
    pubmed: PubMedConfig = Field(default_factory=PubMedConfig)
    drug_kb: DrugKBConfig = Field(default_factory=DrugKBConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)

    @classmethod
    def from_file(cls, path: Path | None = None) -> "Settings":
        config_path = path or (Path.home() / ".medrag" / "config.json")
        overrides: dict[str, Any] = {}
        if config_path.exists():
            overrides = json.loads(config_path.read_text())
        return cls(**overrides)
