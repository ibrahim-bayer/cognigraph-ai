"""Embedding service — wraps sentence-transformers for semantic matching."""

from __future__ import annotations

import numpy as np

from cognigraph.config import CogniGraphConfig
from cognigraph.exceptions import EmbeddingError
from cognigraph.types import EmbeddingVector


class EmbeddingService:
    """Produces L2-normalized embedding vectors using E5-Small.

    Implements the EmbeddingProvider protocol. Model is lazy-loaded
    on first call to avoid startup cost when not needed.
    """

    def __init__(self, config: CogniGraphConfig | None = None) -> None:
        cfg = config or CogniGraphConfig()
        self._model_name = cfg.embedding_model
        self._expected_dim = cfg.embedding_dim
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name)
        except Exception as e:
            raise EmbeddingError(f"Failed to load embedding model '{self._model_name}': {e}") from e

    def embed(self, text: str) -> EmbeddingVector:
        self._ensure_loaded()
        prefixed = f"query: {text}"
        try:
            vec = self._model.encode(prefixed, normalize_embeddings=True)
            result = vec.tolist()
            if len(result) != self._expected_dim:
                raise EmbeddingError(
                    f"Expected {self._expected_dim}-dim vector, got {len(result)}"
                )
            return result
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Embedding failed: {e}") from e

    def embed_batch(self, texts: list[str]) -> list[EmbeddingVector]:
        self._ensure_loaded()
        prefixed = [f"query: {t}" for t in texts]
        try:
            vecs = self._model.encode(prefixed, normalize_embeddings=True)
            results = vecs.tolist()
            for i, vec in enumerate(results):
                if len(vec) != self._expected_dim:
                    raise EmbeddingError(
                        f"Expected {self._expected_dim}-dim vector at index {i}, got {len(vec)}"
                    )
            return results
        except EmbeddingError:
            raise
        except Exception as e:
            raise EmbeddingError(f"Batch embedding failed: {e}") from e
