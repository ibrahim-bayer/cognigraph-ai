"""Node matcher — routes an incoming query to the best graph node.

Combines FAISS similarity search with graph-store confidence scoring to
pick a RouteDecision:

  GRAPH_DIRECT    — high similarity, high confidence, leaf node
  GRAPH_COMPOSED  — high similarity, high confidence, has children (skill chain)
  LLM_FALLBACK    — graph found something but not confident enough
  LLM_ONLY        — no viable match; genuinely novel input

Thresholds are configurable via CogniGraphConfig. The matcher is a pure
decision function: it does not mutate the graph or the index.
"""

from __future__ import annotations

import logging

from cognigraph.config import CogniGraphConfig
from cognigraph.exceptions import NodeNotFoundError
from cognigraph.models import HabitNode, MatchResult, RouteDecision
from cognigraph.protocols import GraphStoreProtocol, VectorIndexProtocol
from cognigraph.types import EmbeddingVector, NodeId

logger = logging.getLogger(__name__)


class NodeMatcher:
    """Decides routing for a query embedding.

    Thread safety: `match()` mutates no matcher state — it only reads
    through `GraphStoreProtocol` and `VectorIndexProtocol`. However, those
    underlying implementations (InMemoryGraphStore, FAISSIndex) are
    themselves not thread-safe. Concurrent `match()` calls are only safe
    when the backing store and index are independently thread-safe or
    externally synchronized. Wrapping the matcher in a lock does NOT
    make FAISS search reentrant.
    """

    def __init__(
        self,
        graph_store: GraphStoreProtocol,
        vector_index: VectorIndexProtocol,
        config: CogniGraphConfig | None = None,
    ) -> None:
        self._store = graph_store
        self._index = vector_index
        self._config = config or CogniGraphConfig()
        # Observability: surface FAISS / graph-store drift via a counter
        # that operators can sample. Logs a warning on each drift event.
        self._stale_hits: int = 0

    @property
    def stale_hit_count(self) -> int:
        """Number of FAISS hits that did not resolve in the graph store."""
        return self._stale_hits

    # --- Public API ---

    def match(self, embedding: EmbeddingVector) -> MatchResult:
        """Find the best node for `embedding` and decide the route."""
        k = self._config.faiss_search_k
        candidates = self._index.search(embedding, k=k)

        if not candidates:
            return MatchResult(
                node=None,
                score=0.0,
                similarity=0.0,
                route_decision=RouteDecision.LLM_ONLY,
                candidates=[],
                ambiguous=False,
            )

        # Rank candidates by combined score (similarity * confidence).
        # Only nodes that still exist in the graph store are considered;
        # a stale FAISS entry is logged and skipped rather than crashing.
        scored: list[tuple[HabitNode, float, float]] = []
        for node_id, similarity in candidates:
            try:
                node = self._store.get_node(node_id)
            except NodeNotFoundError:
                self._stale_hits += 1
                logger.warning(
                    "faiss-graph drift: node_id %r present in index but not "
                    "in graph store (total stale hits: %d)",
                    node_id,
                    self._stale_hits,
                )
                continue
            combined = self._compute_score(similarity, node)
            scored.append((node, similarity, combined))

        if not scored:
            # FAISS had hits but none survived in the graph store — treat
            # as genuinely novel from the router's perspective. `candidates`
            # is still populated so the learner can see the drift.
            return MatchResult(
                node=None,
                score=0.0,
                similarity=0.0,
                route_decision=RouteDecision.LLM_ONLY,
                candidates=candidates,
                ambiguous=False,
            )

        scored.sort(key=lambda t: t[2], reverse=True)
        best_node, best_similarity, best_combined = scored[0]
        # Clamp similarity for the MatchResult surface so downstream code
        # sees values in [0, 1], matching how FAISSIndex now clamps scores.
        best_similarity_clamped = max(0.0, min(1.0, best_similarity))

        # Ambiguity: close call between top-1 and top-2 by combined score.
        ambiguous = False
        if len(scored) >= 2:
            runner_up_combined = scored[1][2]
            if (best_combined - runner_up_combined) < self._config.ambiguity_gap:
                ambiguous = True

        # Route decision is re-evaluated from the winner's raw sim+conf,
        # NOT from the combined score. This keeps the routing independent
        # of the ranking heuristic.
        route = self._decide_route(best_similarity, best_node)

        return MatchResult(
            node=best_node,
            score=best_combined,
            similarity=best_similarity_clamped,
            route_decision=route,
            candidates=candidates,
            ambiguous=ambiguous,
        )

    # --- Scoring + routing ---

    def _compute_score(self, similarity: float, node: HabitNode) -> float:
        """Combined score = similarity × node.confidence.

        Both inputs clamped to [0, 1]. Negative similarities (anti-parallel
        vectors) collapse to 0 so they never beat a positive-similarity
        low-confidence node on the sort.

        Note: confidence=0 nodes produce combined=0 regardless of
        similarity, leading to unranked ties within the conf=0 pool. In
        practice unreachable since learning_starting_confidence=0.5, but
        the final route decision is re-evaluated against raw similarity
        and confidence, so routing stays correct even in that edge case.
        """
        sim = max(0.0, min(1.0, similarity))
        conf = max(0.0, min(1.0, node.confidence))
        return sim * conf

    def _decide_route(
        self, similarity: float, node: HabitNode
    ) -> RouteDecision:
        """Map (similarity, node.confidence, has-children) to a route.

        Thresholds (from config):
          similarity_threshold — min similarity for confident graph match
          confidence_threshold — min node.confidence for automatic exec
          fallback_similarity  — below this → LLM_ONLY (too weak to trust)

        The `sim < fallback_similarity` early return deliberately ignores
        confidence: a weak-similarity hit is not evidence of relevance even
        if the node is highly confident. This is a product decision that
        keeps novelty distinguishable from a bad match.
        """
        cfg = self._config

        if similarity < cfg.fallback_similarity:
            return RouteDecision.LLM_ONLY

        strong_similarity = similarity >= cfg.similarity_threshold
        strong_confidence = node.confidence >= cfg.confidence_threshold

        if strong_similarity and strong_confidence:
            # TOCTOU note: this second store call is safe under the
            # documented single-threaded contract. In a future concurrent
            # world this could race with a concurrent remove_node; the
            # matcher's thread-safety docstring covers that case.
            has_children = bool(self._store.get_children(node.pattern_id))
            return (
                RouteDecision.GRAPH_COMPOSED
                if has_children
                else RouteDecision.GRAPH_DIRECT
            )

        # In the [fallback_similarity, similarity_threshold) band, or strong
        # similarity but weak confidence — graph has a lead but not a
        # trustworthy answer. Fall back to the LLM using the graph hit as
        # context.
        return RouteDecision.LLM_FALLBACK
