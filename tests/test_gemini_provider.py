import pytest

from app.embeddings.gemini import GeminiEmbeddingProvider


def test_missing_api_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        GeminiEmbeddingProvider()


def test_accepts_google_api_key_env_var(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    provider = GeminiEmbeddingProvider()
    assert provider.model_name == "gemini-embedding-001"


def test_accepts_gemini_api_key_env_var_as_fallback(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    provider = GeminiEmbeddingProvider()
    assert provider.model_name == "gemini-embedding-001"


def test_defaults_to_768_dimensions(monkeypatch):
    monkeypatch.delenv("GEMINI_EMBEDDING_DIMENSION", raising=False)
    provider = GeminiEmbeddingProvider(api_key="test-key")
    assert provider.dimension == 768


def test_dimension_overridable_via_constructor_arg():
    provider = GeminiEmbeddingProvider(api_key="test-key", output_dimension=1536)
    assert provider.dimension == 1536


def test_dimension_overridable_via_env(monkeypatch):
    monkeypatch.setenv("GEMINI_EMBEDDING_DIMENSION", "3072")
    provider = GeminiEmbeddingProvider(api_key="test-key")
    assert provider.dimension == 3072


def test_embed_query_rejects_empty_string():
    provider = GeminiEmbeddingProvider(api_key="test-key")
    with pytest.raises(ValueError):
        provider.embed_query("   ")


def test_embed_documents_returns_empty_list_for_empty_input():
    provider = GeminiEmbeddingProvider(api_key="test-key")
    assert provider.embed_documents([]) == []
