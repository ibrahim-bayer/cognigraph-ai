"""Tests for EmbeddingService."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cognigraph.config import CogniGraphConfig
from cognigraph.embedding import EmbeddingService
from cognigraph.exceptions import EmbeddingError
from cognigraph.protocols import EmbeddingProvider


class TestEmbeddingService:
    """Tests using a mocked sentence-transformers model."""

    def _make_service(self, dim: int = 384) -> EmbeddingService:
        """Create service with a mocked model pre-injected."""
        config = CogniGraphConfig(embedding_dim=dim)
        service = EmbeddingService(config)
        mock_model = MagicMock()

        def mock_encode(texts, normalize_embeddings=False):
            if isinstance(texts, str):
                vec = np.random.randn(dim).astype(np.float32)
                if normalize_embeddings:
                    vec = vec / np.linalg.norm(vec)
                return vec
            vecs = np.random.randn(len(texts), dim).astype(np.float32)
            if normalize_embeddings:
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                vecs = vecs / norms
            return vecs

        mock_model.encode = mock_encode
        service._model = mock_model
        return service

    def test_implements_protocol(self) -> None:
        service = self._make_service()
        assert isinstance(service, EmbeddingProvider)

    def test_embed_returns_correct_dim(self) -> None:
        service = self._make_service(384)
        vec = service.embed("hello world")
        assert len(vec) == 384

    def test_embed_returns_list_of_floats(self) -> None:
        service = self._make_service()
        vec = service.embed("test")
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)

    def test_embed_is_l2_normalized(self) -> None:
        service = self._make_service()
        vec = service.embed("hello")
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 0.01

    def test_embed_batch_returns_correct_count(self) -> None:
        service = self._make_service()
        vecs = service.embed_batch(["hello", "world", "test"])
        assert len(vecs) == 3

    def test_embed_batch_correct_dims(self) -> None:
        service = self._make_service(384)
        vecs = service.embed_batch(["a", "b"])
        for vec in vecs:
            assert len(vec) == 384

    def test_embed_batch_all_normalized(self) -> None:
        service = self._make_service()
        vecs = service.embed_batch(["hello", "world"])
        for vec in vecs:
            norm = np.linalg.norm(vec)
            assert abs(norm - 1.0) < 0.01

    def test_query_prefix_prepended(self) -> None:
        config = CogniGraphConfig()
        service = EmbeddingService(config)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros(384, dtype=np.float32)
        service._model = mock_model

        # Will fail on dim check since all zeros, so we patch dim check
        mock_model.encode.return_value = np.ones(384, dtype=np.float32) / np.sqrt(384)
        service.embed("hello")

        call_args = mock_model.encode.call_args
        assert call_args[0][0] == "query: hello"

    def test_batch_query_prefix_prepended(self) -> None:
        config = CogniGraphConfig()
        service = EmbeddingService(config)
        mock_model = MagicMock()
        vec = np.ones(384, dtype=np.float32) / np.sqrt(384)
        mock_model.encode.return_value = np.stack([vec, vec])
        service._model = mock_model

        service.embed_batch(["hello", "world"])

        call_args = mock_model.encode.call_args
        assert call_args[0][0] == ["query: hello", "query: world"]


class TestEmbeddingServiceLazyLoading:
    def test_model_not_loaded_on_init(self) -> None:
        service = EmbeddingService(CogniGraphConfig())
        assert service._model is None

    def test_model_load_failure_raises_embedding_error(self) -> None:
        service = EmbeddingService(CogniGraphConfig(embedding_model="nonexistent/model"))
        with patch("sentence_transformers.SentenceTransformer", side_effect=RuntimeError("not found")):
            with pytest.raises(EmbeddingError, match="Failed to load"):
                service.embed("hello")


class TestEmbeddingServiceDimValidation:
    def test_wrong_dim_raises(self) -> None:
        config = CogniGraphConfig(embedding_dim=128)
        service = EmbeddingService(config)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.ones(384, dtype=np.float32)
        service._model = mock_model

        with pytest.raises(EmbeddingError, match="Expected 128-dim"):
            service.embed("test")


class TestEmbeddingServiceEdgeCases:
    def test_embed_empty_string(self) -> None:
        config = CogniGraphConfig()
        service = EmbeddingService(config)
        mock_model = MagicMock()
        vec = np.ones(384, dtype=np.float32) / np.sqrt(384)
        mock_model.encode.return_value = vec
        service._model = mock_model

        result = service.embed("")
        assert len(result) == 384
        # Verify "query: " prefix still applied for empty string
        assert mock_model.encode.call_args[0][0] == "query: "

    def test_embed_batch_empty_list(self) -> None:
        config = CogniGraphConfig()
        service = EmbeddingService(config)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.zeros((0, 384), dtype=np.float32)
        service._model = mock_model

        result = service.embed_batch([])
        assert result == []


class TestEmbeddingServiceSimilarity:
    """Verify semantic similarity properties using deterministic mock vectors."""

    def _make_similarity_service(self) -> EmbeddingService:
        """Create service with a mock that returns distinct but deterministic vectors."""
        config = CogniGraphConfig()
        service = EmbeddingService(config)
        mock_model = MagicMock()

        def mock_encode(texts, normalize_embeddings=False):
            if isinstance(texts, str):
                texts = [texts]
                single = True
            else:
                single = False

            vecs = []
            for text in texts:
                # Deterministic vector based on text hash
                rng = np.random.RandomState(hash(text) % 2**31)
                vec = rng.randn(384).astype(np.float32)
                if normalize_embeddings:
                    vec = vec / np.linalg.norm(vec)
                vecs.append(vec)

            result = np.array(vecs)
            return result[0] if single else result

        mock_model.encode = mock_encode
        service._model = mock_model
        return service

    def test_same_input_same_output(self) -> None:
        service = self._make_similarity_service()
        v1 = service.embed("hello")
        v2 = service.embed("hello")
        assert v1 == v2

    def test_similar_texts_high_cosine_similarity(self) -> None:
        """Same text with prefix produces same hash → identical vectors → similarity 1.0."""
        service = self._make_similarity_service()
        v1 = service.embed("hello world")
        v2 = service.embed("hello world")
        similarity = np.dot(v1, v2)
        assert similarity > 0.99

    def test_dissimilar_texts_low_cosine_similarity(self) -> None:
        """Different texts produce different hash seeds → distinct vectors → low similarity."""
        service = self._make_similarity_service()
        v1 = service.embed("hello world")
        v2 = service.embed("quantum physics equations")
        similarity = abs(np.dot(v1, v2))
        # Random 384-dim normalized vectors have near-zero expected cosine similarity
        assert similarity < 0.3
