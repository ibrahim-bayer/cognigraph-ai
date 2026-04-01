"""Tests for InMemoryGraphStore."""

from __future__ import annotations

import time

import pytest

from cognigraph.exceptions import NodeNotFoundError
from cognigraph.graph_store import InMemoryGraphStore
from cognigraph.models import ChildLink, HabitNode, Stability
from cognigraph.protocols import GraphStoreProtocol


# --- Fixtures ---


@pytest.fixture
def store() -> InMemoryGraphStore:
    return InMemoryGraphStore()


def _make_node(
    pattern_id: str = "node-1",
    confidence: float = 0.5,
    last_used_at: float | None = None,
    **kwargs,
) -> HabitNode:
    return HabitNode(
        pattern_id=pattern_id,
        confidence=confidence,
        last_used_at=last_used_at or time.time(),
        **kwargs,
    )


# --- Protocol conformance ---


class TestProtocolConformance:
    def test_implements_graph_store_protocol(self) -> None:
        assert isinstance(InMemoryGraphStore(), GraphStoreProtocol)


# --- CRUD ---


class TestNodeCRUD:
    def test_put_and_get(self, store: InMemoryGraphStore) -> None:
        node = _make_node("a")
        store.put_node(node)
        assert store.get_node("a") is node

    def test_get_missing_raises(self, store: InMemoryGraphStore) -> None:
        with pytest.raises(NodeNotFoundError) as exc_info:
            store.get_node("nonexistent")
        assert exc_info.value.node_id == "nonexistent"

    def test_put_overwrites_existing(self, store: InMemoryGraphStore) -> None:
        store.put_node(_make_node("a", confidence=0.3))
        store.put_node(_make_node("a", confidence=0.9))
        assert store.get_node("a").confidence == 0.9

    def test_remove_node(self, store: InMemoryGraphStore) -> None:
        store.put_node(_make_node("a"))
        store.remove_node("a")
        with pytest.raises(NodeNotFoundError):
            store.get_node("a")

    def test_remove_missing_raises(self, store: InMemoryGraphStore) -> None:
        with pytest.raises(NodeNotFoundError):
            store.remove_node("nonexistent")

    def test_put_after_remove(self, store: InMemoryGraphStore) -> None:
        store.put_node(_make_node("a"))
        store.remove_node("a")
        store.put_node(_make_node("a", confidence=0.8))
        assert store.get_node("a").confidence == 0.8


# --- Link operations ---


class TestLinkOperations:
    def test_add_link_and_get_children(self, store: InMemoryGraphStore) -> None:
        store.put_node(_make_node("parent"))
        store.put_node(_make_node("child"))
        store.add_link("parent", ChildLink(habit_id="child", order=0))
        children = store.get_children("parent")
        assert len(children) == 1
        assert children[0].habit_id == "child"

    def test_get_children_sorted_by_order(self, store: InMemoryGraphStore) -> None:
        store.put_node(_make_node("p"))
        store.put_node(_make_node("c1"))
        store.put_node(_make_node("c2"))
        store.put_node(_make_node("c3"))
        store.add_link("p", ChildLink(habit_id="c2", order=2))
        store.add_link("p", ChildLink(habit_id="c3", order=3))
        store.add_link("p", ChildLink(habit_id="c1", order=1))
        children = store.get_children("p")
        assert [c.habit_id for c in children] == ["c1", "c2", "c3"]

    def test_get_parents(self, store: InMemoryGraphStore) -> None:
        store.put_node(_make_node("p1"))
        store.put_node(_make_node("p2"))
        store.put_node(_make_node("child"))
        store.add_link("p1", ChildLink(habit_id="child"))
        store.add_link("p2", ChildLink(habit_id="child"))
        assert store.get_parents("child") == {"p1", "p2"}

    def test_remove_link(self, store: InMemoryGraphStore) -> None:
        store.put_node(_make_node("p"))
        store.put_node(_make_node("c"))
        store.add_link("p", ChildLink(habit_id="c"))
        store.remove_link("p", "c")
        assert store.get_children("p") == []
        assert store.get_parents("c") == set()

    def test_add_link_missing_parent_raises(self, store: InMemoryGraphStore) -> None:
        store.put_node(_make_node("child"))
        with pytest.raises(NodeNotFoundError):
            store.add_link("missing", ChildLink(habit_id="child"))

    def test_add_link_missing_child_raises(self, store: InMemoryGraphStore) -> None:
        store.put_node(_make_node("parent"))
        with pytest.raises(NodeNotFoundError):
            store.add_link("parent", ChildLink(habit_id="missing"))

    def test_get_children_missing_node_raises(self, store: InMemoryGraphStore) -> None:
        with pytest.raises(NodeNotFoundError):
            store.get_children("missing")

    def test_get_parents_missing_node_raises(self, store: InMemoryGraphStore) -> None:
        with pytest.raises(NodeNotFoundError):
            store.get_parents("missing")

    def test_remove_link_missing_parent_raises(self, store: InMemoryGraphStore) -> None:
        with pytest.raises(NodeNotFoundError):
            store.remove_link("missing", "c")

    def test_remove_link_missing_child_raises(self, store: InMemoryGraphStore) -> None:
        store.put_node(_make_node("p"))
        with pytest.raises(NodeNotFoundError):
            store.remove_link("p", "missing")

    def test_remove_nonexistent_link_is_noop(self, store: InMemoryGraphStore) -> None:
        store.put_node(_make_node("p"))
        store.put_node(_make_node("c"))
        store.remove_link("p", "c")  # no link exists — should not raise
        assert store.get_children("p") == []

    def test_add_link_with_condition(self, store: InMemoryGraphStore) -> None:
        store.put_node(_make_node("p"))
        store.put_node(_make_node("c"))
        store.add_link("p", ChildLink(habit_id="c", condition="if_raining", order=1))
        children = store.get_children("p")
        assert children[0].condition == "if_raining"

    def test_add_identical_duplicate_link_is_noop(self, store: InMemoryGraphStore) -> None:
        """Adding the exact same link twice should not create duplicates."""
        store.put_node(_make_node("p"))
        store.put_node(_make_node("c"))
        link = ChildLink(habit_id="c", condition="rain", order=1)
        store.add_link("p", link)
        store.add_link("p", ChildLink(habit_id="c", condition="rain", order=1))
        assert len(store.get_children("p")) == 1

    def test_same_child_and_condition_different_order_kept(
        self, store: InMemoryGraphStore
    ) -> None:
        """Links with same child+condition but different order are distinct."""
        store.put_node(_make_node("p"))
        store.put_node(_make_node("c"))
        store.add_link("p", ChildLink(habit_id="c", condition="rain", order=1))
        store.add_link("p", ChildLink(habit_id="c", condition="rain", order=2))
        assert len(store.get_children("p")) == 2


# --- Node removal cascades link cleanup ---


class TestRemoveNodeCascade:
    def test_remove_parent_cleans_up_children_reverse_index(
        self, store: InMemoryGraphStore
    ) -> None:
        store.put_node(_make_node("parent"))
        store.put_node(_make_node("child"))
        store.add_link("parent", ChildLink(habit_id="child"))
        store.remove_node("parent")
        # child should no longer list 'parent' as a parent
        assert store.get_parents("child") == set()

    def test_remove_child_cleans_up_parent_forward_links(
        self, store: InMemoryGraphStore
    ) -> None:
        store.put_node(_make_node("parent"))
        store.put_node(_make_node("child"))
        store.add_link("parent", ChildLink(habit_id="child"))
        store.remove_node("child")
        assert store.get_children("parent") == []

    def test_remove_shared_node_cleans_all_parents(
        self, store: InMemoryGraphStore
    ) -> None:
        store.put_node(_make_node("p1"))
        store.put_node(_make_node("p2"))
        store.put_node(_make_node("shared"))
        store.add_link("p1", ChildLink(habit_id="shared"))
        store.add_link("p2", ChildLink(habit_id="shared"))
        store.remove_node("shared")
        assert store.get_children("p1") == []
        assert store.get_children("p2") == []

    def test_remove_node_with_multiple_children(
        self, store: InMemoryGraphStore
    ) -> None:
        store.put_node(_make_node("p"))
        store.put_node(_make_node("c1"))
        store.put_node(_make_node("c2"))
        store.add_link("p", ChildLink(habit_id="c1"))
        store.add_link("p", ChildLink(habit_id="c2"))
        store.remove_node("p")
        assert store.get_parents("c1") == set()
        assert store.get_parents("c2") == set()


# --- Bulk queries ---


class TestBulkQueries:
    def test_all_nodes_empty(self, store: InMemoryGraphStore) -> None:
        assert store.all_nodes() == []

    def test_all_nodes(self, store: InMemoryGraphStore) -> None:
        store.put_node(_make_node("a"))
        store.put_node(_make_node("b"))
        ids = {n.pattern_id for n in store.all_nodes()}
        assert ids == {"a", "b"}

    def test_node_count_empty(self, store: InMemoryGraphStore) -> None:
        assert store.node_count() == 0

    def test_node_count(self, store: InMemoryGraphStore) -> None:
        store.put_node(_make_node("a"))
        store.put_node(_make_node("b"))
        assert store.node_count() == 2

    def test_node_count_after_remove(self, store: InMemoryGraphStore) -> None:
        store.put_node(_make_node("a"))
        store.put_node(_make_node("b"))
        store.remove_node("a")
        assert store.node_count() == 1


# --- Filtered queries ---


class TestFilteredQueries:
    def test_nodes_by_confidence(self, store: InMemoryGraphStore) -> None:
        store.put_node(_make_node("low", confidence=0.2))
        store.put_node(_make_node("mid", confidence=0.5))
        store.put_node(_make_node("high", confidence=0.9))
        result = store.nodes_by_confidence(0.5)
        ids = {n.pattern_id for n in result}
        assert ids == {"mid", "high"}

    def test_nodes_by_confidence_none_match(self, store: InMemoryGraphStore) -> None:
        store.put_node(_make_node("low", confidence=0.1))
        assert store.nodes_by_confidence(0.9) == []

    def test_nodes_by_confidence_empty_store(self, store: InMemoryGraphStore) -> None:
        assert store.nodes_by_confidence(0.0) == []

    def test_nodes_by_last_used(self, store: InMemoryGraphStore) -> None:
        old_time = 1000.0
        new_time = 9999999999.0
        store.put_node(_make_node("old", last_used_at=old_time))
        store.put_node(_make_node("new", last_used_at=new_time))
        result = store.nodes_by_last_used(before=5000.0)
        ids = {n.pattern_id for n in result}
        assert ids == {"old"}

    def test_nodes_by_last_used_none_match(self, store: InMemoryGraphStore) -> None:
        store.put_node(_make_node("recent", last_used_at=9999999999.0))
        assert store.nodes_by_last_used(before=1000.0) == []

    def test_nodes_by_last_used_empty_store(self, store: InMemoryGraphStore) -> None:
        assert store.nodes_by_last_used(before=9999999999.0) == []


# --- Edge cases ---


class TestEdgeCases:
    def test_put_upsert_preserves_links(self, store: InMemoryGraphStore) -> None:
        """Upserting a node should not destroy its existing links."""
        store.put_node(_make_node("p"))
        store.put_node(_make_node("c"))
        store.add_link("p", ChildLink(habit_id="c"))
        # upsert parent with new confidence
        store.put_node(_make_node("p", confidence=0.99))
        assert len(store.get_children("p")) == 1
        assert store.get_parents("c") == {"p"}

    def test_node_as_both_parent_and_child(self, store: InMemoryGraphStore) -> None:
        store.put_node(_make_node("a"))
        store.put_node(_make_node("b"))
        store.add_link("a", ChildLink(habit_id="b"))
        store.add_link("b", ChildLink(habit_id="a"))
        assert store.get_children("a")[0].habit_id == "b"
        assert store.get_children("b")[0].habit_id == "a"
        assert store.get_parents("a") == {"b"}
        assert store.get_parents("b") == {"a"}

    def test_self_link(self, store: InMemoryGraphStore) -> None:
        """A node linking to itself should not cause issues."""
        store.put_node(_make_node("a"))
        store.add_link("a", ChildLink(habit_id="a"))
        assert store.get_children("a")[0].habit_id == "a"
        assert "a" in store.get_parents("a")

    def test_remove_self_linked_node(self, store: InMemoryGraphStore) -> None:
        store.put_node(_make_node("a"))
        store.add_link("a", ChildLink(habit_id="a"))
        store.remove_node("a")
        assert store.node_count() == 0

    def test_multiple_links_same_child(self, store: InMemoryGraphStore) -> None:
        """Multiple links to same child (different conditions) are allowed."""
        store.put_node(_make_node("p"))
        store.put_node(_make_node("c"))
        store.add_link("p", ChildLink(habit_id="c", condition="rain", order=1))
        store.add_link("p", ChildLink(habit_id="c", condition="sun", order=2))
        children = store.get_children("p")
        assert len(children) == 2
        assert children[0].condition == "rain"
        assert children[1].condition == "sun"

    def test_get_children_returns_copy(self, store: InMemoryGraphStore) -> None:
        """Mutating the returned list must not affect internal state."""
        store.put_node(_make_node("p"))
        store.put_node(_make_node("c"))
        store.add_link("p", ChildLink(habit_id="c"))
        children = store.get_children("p")
        children.clear()
        assert len(store.get_children("p")) == 1


# --- Selective link removal ---


class TestSelectiveLinkRemoval:
    def test_remove_link_default_removes_all(self, store: InMemoryGraphStore) -> None:
        """Default remove_link removes all links from parent to child."""
        store.put_node(_make_node("p"))
        store.put_node(_make_node("c"))
        store.add_link("p", ChildLink(habit_id="c", condition="rain", order=1))
        store.add_link("p", ChildLink(habit_id="c", condition="sun", order=2))
        store.remove_link("p", "c")
        assert store.get_children("p") == []
        assert store.get_parents("c") == set()

    def test_remove_link_by_condition_selective(self, store: InMemoryGraphStore) -> None:
        """Passing condition= removes only the matching link."""
        store.put_node(_make_node("p"))
        store.put_node(_make_node("c"))
        store.add_link("p", ChildLink(habit_id="c", condition="rain", order=1))
        store.add_link("p", ChildLink(habit_id="c", condition="sun", order=2))
        store.remove_link("p", "c", condition="rain")
        children = store.get_children("p")
        assert len(children) == 1
        assert children[0].condition == "sun"
        # parent still in reverse index since one link remains
        assert store.get_parents("c") == {"p"}

    def test_remove_link_by_condition_none(self, store: InMemoryGraphStore) -> None:
        """condition=None removes only the unconditioned link."""
        store.put_node(_make_node("p"))
        store.put_node(_make_node("c"))
        store.add_link("p", ChildLink(habit_id="c", condition=None, order=1))
        store.add_link("p", ChildLink(habit_id="c", condition="rain", order=2))
        store.remove_link("p", "c", condition=None)
        children = store.get_children("p")
        assert len(children) == 1
        assert children[0].condition == "rain"

    def test_remove_link_by_condition_clears_reverse_when_last(
        self, store: InMemoryGraphStore
    ) -> None:
        """Reverse index is cleaned up when the last conditional link is removed."""
        store.put_node(_make_node("p"))
        store.put_node(_make_node("c"))
        store.add_link("p", ChildLink(habit_id="c", condition="rain", order=1))
        store.remove_link("p", "c", condition="rain")
        assert store.get_children("p") == []
        assert store.get_parents("c") == set()
