"""Protocol ABCs defining component contracts for CogniGraph."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from cognigraph.models import (
    ChildLink,
    HabitNode,
    InteractionLog,
    LearningOutcome,
    MatchResult,
    SafetyDecision,
)
from cognigraph.types import EmbeddingVector, LLMResponse, NodeId


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Contract for embedding model implementations."""

    def embed(self, text: str) -> EmbeddingVector: ...

    def embed_batch(self, texts: list[str]) -> list[EmbeddingVector]: ...


@runtime_checkable
class LLMProvider(Protocol):
    """Contract for LLM implementations."""

    def generate(
        self,
        prompt: str,
        context: list[dict] | None = None,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse: ...


@runtime_checkable
class VectorIndexProtocol(Protocol):
    """Contract for vector index implementations."""

    def add(self, node_id: NodeId, vector: EmbeddingVector) -> None: ...

    def remove(self, node_id: NodeId) -> None: ...

    def search(
        self, query_vector: EmbeddingVector, k: int = 5
    ) -> list[tuple[NodeId, float]]: ...

    def count(self) -> int: ...

    def save(self, path: str) -> None: ...

    def load(self, path: str) -> None: ...


@runtime_checkable
class NodeMatcherProtocol(Protocol):
    """Contract for node matcher implementations."""

    def match(self, embedding: EmbeddingVector) -> MatchResult: ...


@runtime_checkable
class PersistenceProtocol(Protocol):
    """Contract for persistence backends (interaction log + graph snapshots).

    Only the methods the reinforcement logger actually uses are required;
    full graph save/load lives on the concrete SQLitePersistence class.
    """

    def log_interaction(self, log: InteractionLog) -> None: ...

    def get_interactions(
        self, limit: int = 100, offset: int = 0
    ) -> list[InteractionLog]: ...

    def get_interactions_for_node(
        self, node_id: NodeId
    ) -> list[InteractionLog]: ...


@runtime_checkable
class ReinforcementLoggerProtocol(Protocol):
    """Contract for reinforcement logger implementations."""

    def log_and_reinforce(self, interaction: InteractionLog) -> bool: ...

    def get_node_history(
        self, node_id: NodeId, limit: int | None = 100
    ) -> list[InteractionLog]: ...


@runtime_checkable
class SafetyBoundaryProtocol(Protocol):
    """Contract for the safety boundary.

    Gates graph routing decisions so the system never confidently
    serves a wrong, dangerous, ambiguous, or stale answer.
    """

    def check(
        self, match_result: MatchResult, input_text: str
    ) -> SafetyDecision: ...


@runtime_checkable
class LearnerProtocol(Protocol):
    """Contract for learner implementations.

    The learner consumes interaction logs (typically from the
    reinforcement logger) and decides whether the LLM's behavior on
    this turn is stable enough to deserve a new graph node.
    """

    def evaluate_for_learning(
        self, interaction: InteractionLog
    ) -> LearningOutcome: ...


@runtime_checkable
class GraphStoreProtocol(Protocol):
    """Contract for graph store implementations."""

    def get_node(self, node_id: NodeId) -> HabitNode: ...

    def put_node(self, node: HabitNode) -> None: ...

    def remove_node(self, node_id: NodeId) -> None: ...

    def get_children(self, node_id: NodeId) -> list[ChildLink]: ...

    def get_parents(self, node_id: NodeId) -> set[NodeId]: ...

    def add_link(self, parent_id: NodeId, child_link: ChildLink) -> None: ...

    def remove_link(self, parent_id: NodeId, child_id: NodeId, *, condition: str | None = ...) -> None: ...

    def all_nodes(self) -> list[HabitNode]: ...

    def node_count(self) -> int: ...
