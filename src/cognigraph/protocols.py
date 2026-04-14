"""Protocol ABCs defining component contracts for CogniGraph."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from cognigraph.models import ChildLink, HabitNode, MatchResult
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
