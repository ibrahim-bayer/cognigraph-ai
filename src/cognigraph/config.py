"""Configuration for CogniGraph with all tunable parameters."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CogniGraphConfig:
    """Central configuration for all CogniGraph components.

    All thresholds are validated to be within sensible ranges on creation.
    """

    # --- Graph matching ---
    confidence_threshold: float = 0.7
    similarity_threshold: float = 0.85
    fallback_similarity: float = 0.6

    # --- Graph capacity ---
    max_graph_capacity: int = 10_000

    # --- Decay ---
    decay_interval_hours: float = 24.0
    decay_rate_high: float = 0.001
    decay_rate_medium: float = 0.005
    decay_rate_low: float = 0.01
    eviction_threshold: float = 0.1

    # --- Learning ---
    learning_min_repetitions: int = 3
    learning_stability_threshold: float = 0.9
    learning_starting_confidence: float = 0.5

    # --- Reinforcement ---
    confidence_boost: float = 0.02
    stability_medium_threshold: int = 5
    stability_high_threshold: int = 20

    # --- Habit strength formula weights ---
    # habit_strength = (freq * a) + (recency * b) + (stability * c)
    #                + (acceptance * d) + (latency_savings * e)
    #                - (conflict * f) - (decay * g)
    strength_weight_frequency: float = 1.0
    strength_weight_recency: float = 1.0
    strength_weight_stability: float = 1.0
    strength_weight_acceptance: float = 1.0
    strength_weight_latency: float = 1.0
    strength_weight_conflict: float = 1.0
    strength_weight_decay: float = 1.0

    # --- Traversal ---
    max_sequence_depth: int = 10

    # --- Working memory ---
    working_memory_window: int = 10

    # --- Embedding ---
    embedding_model: str = "intfloat/e5-small-v2"
    embedding_dim: int = 384

    # --- FAISS ---
    faiss_search_k: int = 5

    # --- LLM ---
    llm_model: str = "claude-sonnet-4-20250514"
    llm_max_tokens: int = 1024

    # --- Safety ---
    ambiguity_gap: float = 0.05
    blocklist_patterns: list[str] = field(default_factory=list)

    # --- Responder ---
    max_response_length: int = 4096

    # --- Persistence ---
    db_path: str = "cognigraph.db"
    faiss_index_path: str = "cognigraph.faiss"

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate that all parameters are within sensible ranges."""
        self._check_non_empty("embedding_model", self.embedding_model)
        self._check_non_empty("llm_model", self.llm_model)
        self._check_non_empty("db_path", self.db_path)
        self._check_non_empty("faiss_index_path", self.faiss_index_path)

        self._check_range("confidence_threshold", self.confidence_threshold, 0.0, 1.0)
        self._check_range("similarity_threshold", self.similarity_threshold, 0.0, 1.0)
        self._check_range("fallback_similarity", self.fallback_similarity, 0.0, 1.0)
        self._check_range("eviction_threshold", self.eviction_threshold, 0.0, 1.0)
        self._check_range("learning_stability_threshold", self.learning_stability_threshold, 0.0, 1.0)
        self._check_range("learning_starting_confidence", self.learning_starting_confidence, 0.0, 1.0)
        self._check_range("confidence_boost", self.confidence_boost, 0.0, 1.0)
        self._check_range("ambiguity_gap", self.ambiguity_gap, 0.0, 1.0)
        self._check_range("decay_rate_high", self.decay_rate_high, 0.0, 1.0)
        self._check_range("decay_rate_medium", self.decay_rate_medium, 0.0, 1.0)
        self._check_range("decay_rate_low", self.decay_rate_low, 0.0, 1.0)

        self._check_positive("max_graph_capacity", self.max_graph_capacity)
        self._check_positive("decay_interval_hours", self.decay_interval_hours)
        self._check_positive("learning_min_repetitions", self.learning_min_repetitions)
        self._check_positive("stability_medium_threshold", self.stability_medium_threshold)
        self._check_positive("stability_high_threshold", self.stability_high_threshold)
        self._check_positive("max_sequence_depth", self.max_sequence_depth)
        self._check_positive("working_memory_window", self.working_memory_window)
        self._check_positive("embedding_dim", self.embedding_dim)
        self._check_positive("faiss_search_k", self.faiss_search_k)
        self._check_positive("llm_max_tokens", self.llm_max_tokens)
        self._check_positive("max_response_length", self.max_response_length)

        for weight_name in (
            "strength_weight_frequency",
            "strength_weight_recency",
            "strength_weight_stability",
            "strength_weight_acceptance",
            "strength_weight_latency",
            "strength_weight_conflict",
            "strength_weight_decay",
        ):
            self._check_range(weight_name, getattr(self, weight_name), 0.0, 100.0)

        if self.fallback_similarity >= self.similarity_threshold:
            raise ValueError(
                f"fallback_similarity ({self.fallback_similarity}) must be less than "
                f"similarity_threshold ({self.similarity_threshold})"
            )

        if self.stability_medium_threshold >= self.stability_high_threshold:
            raise ValueError(
                f"stability_medium_threshold ({self.stability_medium_threshold}) must be less than "
                f"stability_high_threshold ({self.stability_high_threshold})"
            )

    @staticmethod
    def _check_range(name: str, value: float, low: float, high: float) -> None:
        if not (low <= value <= high):
            raise ValueError(f"{name} must be between {low} and {high}, got {value}")

    @staticmethod
    def _check_positive(name: str, value: int | float) -> None:
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")

    @staticmethod
    def _check_non_empty(name: str, value: str) -> None:
        if not value.strip():
            raise ValueError(f"{name} must not be empty")
