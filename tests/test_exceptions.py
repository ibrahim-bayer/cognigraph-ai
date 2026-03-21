"""Tests for CogniGraph exception hierarchy."""

from __future__ import annotations

import pytest

from cognigraph.exceptions import (
    CapacityExceededError,
    CogniGraphError,
    EmbeddingError,
    LLMError,
    NodeNotFoundError,
    PersistenceError,
    SafetyViolationError,
)


class TestExceptionHierarchy:
    @pytest.mark.parametrize("exc_class", [
        NodeNotFoundError,
        EmbeddingError,
        LLMError,
        SafetyViolationError,
        CapacityExceededError,
        PersistenceError,
    ])
    def test_all_inherit_from_base(self, exc_class: type) -> None:
        assert issubclass(exc_class, CogniGraphError)

    def test_base_inherits_from_exception(self) -> None:
        assert issubclass(CogniGraphError, Exception)

    def test_catchable_as_base(self) -> None:
        with pytest.raises(CogniGraphError):
            raise NodeNotFoundError("node-1")


class TestExceptionMessages:
    def test_node_not_found(self) -> None:
        exc = NodeNotFoundError("node-123")
        assert "node-123" in str(exc)
        assert exc.node_id == "node-123"

    def test_safety_violation(self) -> None:
        exc = SafetyViolationError("high risk topic")
        assert "high risk topic" in str(exc)
        assert exc.reason == "high risk topic"

    def test_capacity_exceeded(self) -> None:
        exc = CapacityExceededError(10000)
        assert "10000" in str(exc)
        assert exc.capacity == 10000

    def test_embedding_error(self) -> None:
        exc = EmbeddingError("model failed to load")
        assert "model failed to load" in str(exc)

    def test_llm_error(self) -> None:
        exc = LLMError("API timeout")
        assert "API timeout" in str(exc)

    def test_persistence_error(self) -> None:
        exc = PersistenceError("disk full")
        assert "disk full" in str(exc)
