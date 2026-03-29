"""Tests for CogniGraphConfig."""

from __future__ import annotations

import pytest

from cognigraph.config import CogniGraphConfig


class TestConfigDefaults:
    def test_creates_with_defaults(self) -> None:
        cfg = CogniGraphConfig()
        assert cfg.confidence_threshold == 0.7
        assert cfg.similarity_threshold == 0.85
        assert cfg.max_graph_capacity == 10_000
        assert cfg.decay_interval_hours == 24.0
        assert cfg.learning_min_repetitions == 3
        assert cfg.max_sequence_depth == 10
        assert cfg.working_memory_window == 10
        assert cfg.embedding_model == "intfloat/e5-small-v2"
        assert cfg.embedding_dim == 384
        assert cfg.faiss_search_k == 5
        assert cfg.llm_model == "claude-sonnet-4-20250514"
        assert cfg.llm_max_tokens == 1024
        assert cfg.ambiguity_gap == 0.05
        assert cfg.blocklist_patterns == []
        assert cfg.max_response_length == 4096
        assert cfg.db_path == "cognigraph.db"
        assert cfg.faiss_index_path == "cognigraph.faiss"

    def test_accepts_overrides(self) -> None:
        cfg = CogniGraphConfig(
            confidence_threshold=0.9,
            max_graph_capacity=500,
            learning_min_repetitions=5,
        )
        assert cfg.confidence_threshold == 0.9
        assert cfg.max_graph_capacity == 500
        assert cfg.learning_min_repetitions == 5

    def test_strength_weight_defaults(self) -> None:
        cfg = CogniGraphConfig()
        assert cfg.strength_weight_frequency == 1.0
        assert cfg.strength_weight_recency == 1.0
        assert cfg.strength_weight_stability == 1.0
        assert cfg.strength_weight_acceptance == 1.0
        assert cfg.strength_weight_latency == 1.0
        assert cfg.strength_weight_conflict == 1.0
        assert cfg.strength_weight_decay == 1.0

    def test_frozen_prevents_mutation(self) -> None:
        cfg = CogniGraphConfig()
        with pytest.raises(AttributeError):
            cfg.confidence_threshold = 0.5  # type: ignore[misc]


class TestConfigValidation:
    @pytest.mark.parametrize("field,bad_value", [
        ("confidence_threshold", -0.1),
        ("confidence_threshold", 1.1),
        ("similarity_threshold", -0.5),
        ("similarity_threshold", 2.0),
        ("eviction_threshold", -0.01),
        ("eviction_threshold", 1.5),
        ("learning_stability_threshold", -1.0),
        ("learning_starting_confidence", 1.01),
        ("confidence_boost", -0.1),
        ("ambiguity_gap", 1.5),
        ("decay_rate_high", -0.001),
        ("decay_rate_medium", 1.1),
    ])
    def test_rejects_out_of_range_floats(self, field: str, bad_value: float) -> None:
        with pytest.raises(ValueError, match=f"{field} must be between"):
            CogniGraphConfig(**{field: bad_value})

    @pytest.mark.parametrize("field,bad_value", [
        ("max_input_length", 0),
        ("max_graph_capacity", 0),
        ("max_graph_capacity", -1),
        ("decay_interval_hours", 0),
        ("learning_min_repetitions", -1),
        ("max_sequence_depth", 0),
        ("working_memory_window", -5),
        ("embedding_dim", 0),
        ("faiss_search_k", -1),
        ("llm_max_tokens", 0),
        ("max_response_length", -10),
    ])
    def test_rejects_non_positive_values(self, field: str, bad_value: int) -> None:
        with pytest.raises(ValueError, match=f"{field} must be positive"):
            CogniGraphConfig(**{field: bad_value})

    def test_fallback_must_be_less_than_similarity(self) -> None:
        with pytest.raises(ValueError, match="fallback_similarity.*must be less than"):
            CogniGraphConfig(fallback_similarity=0.9, similarity_threshold=0.85)

    def test_fallback_equal_to_similarity_rejected(self) -> None:
        with pytest.raises(ValueError, match="fallback_similarity.*must be less than"):
            CogniGraphConfig(fallback_similarity=0.85, similarity_threshold=0.85)

    def test_stability_medium_must_be_less_than_high(self) -> None:
        with pytest.raises(ValueError, match="stability_medium_threshold.*must be less than"):
            CogniGraphConfig(stability_medium_threshold=20, stability_high_threshold=5)

    @pytest.mark.parametrize("field", [
        "embedding_model",
        "llm_model",
        "db_path",
        "faiss_index_path",
    ])
    def test_rejects_empty_strings(self, field: str) -> None:
        with pytest.raises(ValueError, match=f"{field} must not be empty"):
            CogniGraphConfig(**{field: ""})

    @pytest.mark.parametrize("field", [
        "embedding_model",
        "llm_model",
        "db_path",
        "faiss_index_path",
    ])
    def test_rejects_whitespace_only_strings(self, field: str) -> None:
        with pytest.raises(ValueError, match=f"{field} must not be empty"):
            CogniGraphConfig(**{field: "   "})

    def test_rejects_negative_strength_weight(self) -> None:
        with pytest.raises(ValueError, match="strength_weight_frequency must be between"):
            CogniGraphConfig(strength_weight_frequency=-0.1)

    def test_valid_edge_values(self) -> None:
        """Boundary values at exactly 0.0 and 1.0 should be accepted."""
        cfg = CogniGraphConfig(
            confidence_threshold=0.0,
            similarity_threshold=1.0,
            fallback_similarity=0.0,
        )
        assert cfg.confidence_threshold == 0.0
        assert cfg.similarity_threshold == 1.0
