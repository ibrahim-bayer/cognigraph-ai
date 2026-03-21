"""Tests for CogniGraph protocol ABCs."""

from __future__ import annotations

from cognigraph.models import ChildLink, HabitNode
from cognigraph.protocols import EmbeddingProvider, GraphStoreProtocol, LLMProvider
from cognigraph.types import LLMResponse, NodeId, EmbeddingVector


class MockEmbeddingProvider:
    def embed(self, text: str) -> EmbeddingVector:
        return [0.1, 0.2, 0.3]

    def embed_batch(self, texts: list[str]) -> list[EmbeddingVector]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class MockLLMProvider:
    def generate(self, prompt: str, context: list[dict] | None = None) -> LLMResponse:
        return LLMResponse(text="response", model="test", latency_ms=10.0)


class MockGraphStore:
    def get_node(self, node_id: NodeId) -> HabitNode:
        return HabitNode(pattern_id=node_id)

    def put_node(self, node: HabitNode) -> None:
        pass

    def remove_node(self, node_id: NodeId) -> None:
        pass

    def get_children(self, node_id: NodeId) -> list[ChildLink]:
        return []

    def get_parents(self, node_id: NodeId) -> set[NodeId]:
        return set()

    def add_link(self, parent_id: NodeId, child_link: ChildLink) -> None:
        pass

    def remove_link(self, parent_id: NodeId, child_id: NodeId) -> None:
        pass

    def all_nodes(self) -> list[HabitNode]:
        return []

    def node_count(self) -> int:
        return 0


class TestProtocolConformance:
    def test_embedding_provider_isinstance(self) -> None:
        provider = MockEmbeddingProvider()
        assert isinstance(provider, EmbeddingProvider)

    def test_llm_provider_isinstance(self) -> None:
        provider = MockLLMProvider()
        assert isinstance(provider, LLMProvider)

    def test_graph_store_isinstance(self) -> None:
        store = MockGraphStore()
        assert isinstance(store, GraphStoreProtocol)


class TestMockImplementations:
    def test_embedding_provider_works(self) -> None:
        provider = MockEmbeddingProvider()
        vec = provider.embed("hello")
        assert len(vec) == 3
        batch = provider.embed_batch(["a", "b"])
        assert len(batch) == 2

    def test_llm_provider_works(self) -> None:
        provider = MockLLMProvider()
        resp = provider.generate("hello")
        assert resp.text == "response"
        assert resp.model == "test"
        assert resp.latency_ms == 10.0

    def test_graph_store_works(self) -> None:
        store = MockGraphStore()
        node = store.get_node("test-id")
        assert node.pattern_id == "test-id"
        assert store.node_count() == 0
        assert store.all_nodes() == []
        assert store.get_children("x") == []
        assert store.get_parents("x") == set()


class TestNonConformance:
    def test_missing_method_fails_isinstance(self) -> None:
        class Incomplete:
            def embed(self, text: str) -> list[float]:
                return []
            # missing embed_batch

        assert not isinstance(Incomplete(), EmbeddingProvider)
