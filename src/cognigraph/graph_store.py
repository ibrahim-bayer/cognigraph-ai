"""In-memory graph store for habit nodes — the hot-path data structure."""

from __future__ import annotations

import bisect
from typing import Any

from cognigraph.exceptions import NodeNotFoundError
from cognigraph.models import ChildLink, HabitNode
from cognigraph.types import NodeId, Timestamp

# Sentinel for remove_link's default condition parameter.
# Distinguishes "remove all links" (default) from "remove where condition is None".
_REMOVE_ALL: Any = object()


class InMemoryGraphStore:
    """Primary graph store using Python dicts for ~200ns lookups.

    Maintains three synchronized structures:
      _nodes:    node_id -> HabitNode
      _children: parent_id -> list[ChildLink]  (forward links, kept sorted by order)
      _parents:  child_id -> set[NodeId]        (reverse index)

    Thread safety: not thread-safe. Callers must synchronize externally
    if concurrent access is needed.

    Returned HabitNode and ChildLink objects are direct references to stored
    objects. Do not mutate fields on returned objects — in particular,
    mutating pattern_id/habit_id breaks lookups, mutating order breaks the
    sorted invariant, and mutating condition breaks dedup.
    """

    def __init__(self) -> None:
        self._nodes: dict[NodeId, HabitNode] = {}
        self._children: dict[NodeId, list[ChildLink]] = {}
        self._parents: dict[NodeId, set[NodeId]] = {}

    def _validate_node_exists(self, node_id: NodeId) -> None:
        if node_id not in self._nodes:
            raise NodeNotFoundError(node_id)

    # --- CRUD ---

    def get_node(self, node_id: NodeId) -> HabitNode:
        try:
            return self._nodes[node_id]
        except KeyError:
            raise NodeNotFoundError(node_id)

    def put_node(self, node: HabitNode) -> None:
        self._nodes[node.pattern_id] = node
        if node.pattern_id not in self._children:
            self._children[node.pattern_id] = []
        if node.pattern_id not in self._parents:
            self._parents[node.pattern_id] = set()

    def remove_node(self, node_id: NodeId) -> None:
        self._validate_node_exists(node_id)

        # Remove all child links FROM this node
        for child_link in self._children.get(node_id, []):
            self._parents.get(child_link.habit_id, set()).discard(node_id)
        self._children.pop(node_id, None)

        # Remove all parent links TO this node (skip self — already cleaned above)
        for parent_id in self._parents.get(node_id, set()).copy():
            if parent_id == node_id:
                continue
            self._children[parent_id] = [
                cl for cl in self._children.get(parent_id, [])
                if cl.habit_id != node_id
            ]
        self._parents.pop(node_id, None)

        del self._nodes[node_id]

    # --- Link operations ---

    def add_link(self, parent_id: NodeId, child_link: ChildLink) -> None:
        self._validate_node_exists(parent_id)
        self._validate_node_exists(child_link.habit_id)

        children = self._children[parent_id]
        # Reject identical duplicate links
        for existing in children:
            if (
                existing.habit_id == child_link.habit_id
                and existing.condition == child_link.condition
                and existing.order == child_link.order
            ):
                return
        # Insert in sorted order by `order` to avoid sorting on every read
        bisect.insort(children, child_link, key=lambda cl: cl.order)
        self._parents[child_link.habit_id].add(parent_id)

    def remove_link(
        self, parent_id: NodeId, child_id: NodeId, *, condition: str | None = _REMOVE_ALL
    ) -> None:
        """Remove links from parent to child.

        By default removes ALL links to child_id. Pass condition=<str> or
        condition=None to remove only the link matching that condition value.
        """
        self._validate_node_exists(parent_id)
        self._validate_node_exists(child_id)

        if condition is _REMOVE_ALL:
            self._children[parent_id] = [
                cl for cl in self._children[parent_id]
                if cl.habit_id != child_id
            ]
        else:
            self._children[parent_id] = [
                cl for cl in self._children[parent_id]
                if not (cl.habit_id == child_id and cl.condition == condition)
            ]

        # Update reverse index: remove parent only if no links remain
        has_remaining = any(
            cl.habit_id == child_id for cl in self._children[parent_id]
        )
        if not has_remaining:
            self._parents[child_id].discard(parent_id)

    def get_children(self, node_id: NodeId) -> list[ChildLink]:
        self._validate_node_exists(node_id)
        return list(self._children.get(node_id, []))

    def get_parents(self, node_id: NodeId) -> set[NodeId]:
        self._validate_node_exists(node_id)
        return self._parents.get(node_id, set()).copy()

    # --- Bulk queries ---

    def all_nodes(self) -> list[HabitNode]:
        return list(self._nodes.values())

    def node_count(self) -> int:
        return len(self._nodes)

    def nodes_by_confidence(self, min_conf: float) -> list[HabitNode]:
        return [n for n in self._nodes.values() if n.confidence >= min_conf]

    def nodes_by_last_used(self, before: Timestamp) -> list[HabitNode]:
        return [n for n in self._nodes.values() if n.last_used_at < before]
