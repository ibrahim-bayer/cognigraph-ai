"""Reinforcement logger — strengthens nodes that were used successfully.

Sits on the back side of every pipeline turn: takes the InteractionLog,
persists it, and (when the graph route succeeded) updates the matched
node's reinforcement_count, confidence, last_used_at, decay_score, and
stability tier.

Non-graph routes (LLM_FALLBACK, LLM_ONLY) are still logged so the
learner can mine the interaction history later, but they do not
reinforce any existing node — that decision belongs to FlatNodeLearner
(#015) which has more context (response divergence, stability, dedup).
"""

from __future__ import annotations

import logging
import time

from cognigraph.config import CogniGraphConfig
from cognigraph.exceptions import NodeNotFoundError
from cognigraph.models import HabitNode, InteractionLog, RouteDecision, Stability
from cognigraph.protocols import GraphStoreProtocol, PersistenceProtocol
from cognigraph.types import NodeId

logger = logging.getLogger(__name__)


class ReinforcementLogger:
    """Logs every interaction and reinforces the matched node when the
    graph successfully handled the turn.

    Thread safety: not thread-safe. The reinforcement path is a
    read-modify-write on a HabitNode; concurrent calls against the same
    node_id can lose updates. The current single-threaded pipeline
    contract is fine; revisit when concurrency lands.
    """

    def __init__(
        self,
        graph_store: GraphStoreProtocol,
        persistence: PersistenceProtocol,
        config: CogniGraphConfig | None = None,
    ) -> None:
        self._store = graph_store
        self._persistence = persistence
        self._config = config or CogniGraphConfig()
        # Observability: count interactions whose matched_node_id no longer
        # resolves in the store. Mirrors NodeMatcher.stale_hit_count.
        # Per-instance — a fresh logger always starts at 0.
        self._stale_reinforcements: int = 0
        # Observability: count graph-route interactions that arrived
        # without a matched_node_id (pipeline bug — caller said GRAPH_*
        # but didn't tell us which node).
        self._missing_node_id_on_graph_route: int = 0

    @property
    def stale_reinforcement_count(self) -> int:
        return self._stale_reinforcements

    @property
    def missing_node_id_count(self) -> int:
        """Graph-route interactions that arrived without matched_node_id."""
        return self._missing_node_id_on_graph_route

    # --- Public API ---

    def log_and_reinforce(self, interaction: InteractionLog) -> bool:
        """Persist the interaction; reinforce the node if the graph won.

        Returns True iff a node was reinforced. False means the
        interaction was logged but no node mutation happened (non-graph
        route, no matched_node_id, or stale id).

        Order of operations: log first, then reinforce. The interaction
        log is the source of truth for the learner — never lose it. If
        the reinforcement leg fails (`get_node` or `put_node` raises),
        the exception propagates and the in-memory `node` may already be
        partially mutated by `_reinforce` (the store returns nodes by
        reference). The pipeline must treat the interaction log as the
        durable record; the in-memory store is best-effort.
        """
        # Always log first — never lose an interaction even if the
        # reinforcement path errors out.
        self._persistence.log_interaction(interaction)

        if not self._is_graph_route(interaction.route_decision):
            return False
        if not interaction.matched_node_id:
            # Graph route without a node id is a pipeline bug. Surface it
            # via counter + warning so a regression can't go silent.
            self._missing_node_id_on_graph_route += 1
            logger.warning(
                "graph route %s arrived without matched_node_id; pipeline "
                "bug? (total missing-id occurrences: %d)",
                interaction.route_decision.value,
                self._missing_node_id_on_graph_route,
            )
            return False

        try:
            node = self._store.get_node(interaction.matched_node_id)
        except NodeNotFoundError:
            self._stale_reinforcements += 1
            logger.warning(
                "stale matched_node_id %r in interaction; cannot reinforce "
                "(total stale reinforcements: %d)",
                interaction.matched_node_id,
                self._stale_reinforcements,
            )
            return False

        self._reinforce(node)
        # NOTE(W1): the in-memory `node` is already mutated by
        # `_reinforce` (the store returns by reference). If put_node
        # raises here, the interaction is logged AND the in-memory node
        # is reinforced, but downstream observers may see a "failed"
        # operation. The interaction log is authoritative.
        self._store.put_node(node)
        return True

    def get_node_history(
        self, node_id: NodeId, limit: int | None = 100
    ) -> list[InteractionLog]:
        """Return persisted interactions where this node was the match.

        `limit=None` returns the full history. `limit=0` returns an
        empty list. Negative values raise ValueError.
        """
        if limit is not None and limit < 0:
            raise ValueError(f"limit must be >= 0 or None, got {limit}")
        history = self._persistence.get_interactions_for_node(node_id)
        if limit is None:
            return history
        return history[:limit]

    # --- Internals ---

    @staticmethod
    def _is_graph_route(route: RouteDecision) -> bool:
        return route in (RouteDecision.GRAPH_DIRECT, RouteDecision.GRAPH_COMPOSED)

    def _reinforce(self, node: HabitNode) -> None:
        cfg = self._config
        # Detect pre-existing stability disagreement BEFORE incrementing
        # count. A normal tier promotion at the threshold count is fine
        # (the count just crossed it); but a stored stability that
        # disagrees with the OLD count signals a corrupted/migrated node.
        expected_old = self._stability_for_count(node.reinforcement_count)
        if expected_old != node.stability:
            logger.warning(
                "stability snapped %s→%s for node %r (count=%d); stored "
                "stability disagreed with reinforcement_count before this "
                "reinforcement",
                node.stability.value,
                expected_old.value,
                node.pattern_id,
                node.reinforcement_count,
            )

        node.reinforcement_count += 1
        node.last_used_at = time.time()
        node.confidence = min(1.0, node.confidence + cfg.confidence_boost)
        node.decay_score = 0.0
        node.stability = self._stability_for_count(node.reinforcement_count)

    def _stability_for_count(self, count: int) -> Stability:
        cfg = self._config
        if count >= cfg.stability_high_threshold:
            return Stability.HIGH
        if count >= cfg.stability_medium_threshold:
            return Stability.MEDIUM
        return Stability.LOW
