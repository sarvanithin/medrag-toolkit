"""Unit tests for configuration."""
from __future__ import annotations

from medrag_toolkit.config import Settings, OllamaConfig, FAISSConfig, RAGConfig


def test_default_ollama_config():
    config = OllamaConfig()
    assert config.base_url == "http://localhost:11434"
    assert config.model == "llama3.2"
    assert config.temperature == 0.1


def test_default_faiss_config():
    config = FAISSConfig()
    assert config.embedding_model == "all-MiniLM-L6-v2"
    assert config.top_k == 10


def test_default_rag_config():
    config = RAGConfig()
    assert config.min_confidence == 0.4
    assert config.require_citations is True
    assert config.hallucination_threshold == 0.3


def test_settings_from_file_missing_file(tmp_path):
    # Non-existent path should use defaults
    settings = Settings.from_file(tmp_path / "nonexistent.json")
    assert settings.ollama.model == "llama3.2"


def test_settings_from_file_with_overrides(tmp_path):
    import json
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"ollama": {"model": "mistral"}}))

    settings = Settings.from_file(config_path)
    assert settings.ollama.model == "mistral"
