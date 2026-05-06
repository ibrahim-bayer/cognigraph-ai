"""Flat node learner — turns repeated stable LLM responses into graph nodes.

Sits AFTER the reinforcement logger in the pipeline. The reinforcement
logger persists every interaction; the learner watches the resulting
history, looks for clusters of N similar inputs that consistently get
the same response, and creates a new HabitNode that captures the
pattern. Subsequent matching turns then route GRAPH_DIRECT instead of
calling the LLM.

Decision flow per LLM-route interaction:
  1. Skip if the interaction was a graph route (already handled by the
     reinforcement logger).
  2. Skip if response_text is empty.
  3. Find recent LLM-route interactions whose normalized_text is at
     least `learning_stability_threshold`-similar to this one.
  4. Skip if fewer than `learning_min_repetitions` similar interactions
     exist (not enough evidence yet).
  5. Skip if responses are not stable across the cluster (LLM
     disagreed, can't crystallize a habit).
  6. Skip if a node already covers this input + response (issue #22:
     dedup considers both embedding similarity AND response
     similarity — divergent answers for similar inputs DO get distinct
     nodes).
  7. Otherwise: create the node, add to graph + FAISS, return it.

Thread safety: not thread-safe (read-modify-write across store, FAISS,
and persistence). Single-threaded pipeline contract.
"""

from __future__ import annotations

import logging
import time
import uuid

from cognigraph.config import CogniGraphConfig
from cognigraph.exceptions import NodeNotFoundError
from cognigraph.models import (
    HabitNode,
    InteractionLog,
    LearningOutcome,
    ResponseForm,
    RiskLevel,
    RouteDecision,
    Stability,
)
from cognigraph.protocols import (
    EmbeddingProvider,
    GraphStoreProtocol,
    PersistenceProtocol,
    VectorIndexProtocol,
)
from cognigraph.types import EmbeddingVector

logger = logging.getLogger(__name__)


class FlatNodeLearner:
    """Detect repeated stable LLM responses and create graph nodes from them."""

    def __init__(
        self,
        graph_store: GraphStoreProtocol,
        vector_index: VectorIndexProtocol,
        embedder: EmbeddingProvider,
        persistence: PersistenceProtocol,
        config: CogniGraphConfig | None = None,
    ) -> None:
        self._store = graph_store
        self._faiss = vector_index
        self._embed = embedder
        self._persistence = persistence
        self._config = config or CogniGraphConfig()
        # Observability for FAISS/graph drift seen during dedup. Mirrors
        # NodeMatcher.stale_hit_count and ReinforcementLogger.stale_reinforcement_count.
        self._stale_dedup_hits: int = 0

    @property
    def stale_dedup_hit_count(self) -> int:
        return self._stale_dedup_hits

    # --- Public API ---

    def evaluate_for_learning(
        self, interaction: InteractionLog
    ) -> LearningOutcome:
        """Decide whether this interaction should produce a new node."""
        cfg = self._config

        if self._is_graph_route(interaction.route_decision):
            return LearningOutcome(
                created_node=None,
                reason="graph_route_already_handled",
            )
        # Whitespace-only text counts as "missing" — embedding it would
        # produce a low-information vector that collides with other
        # whitespace-only inputs across intents.
        if (
            not interaction.normalized_text
            or not interaction.normalized_text.strip()
            or not interaction.response_text
            or not interaction.response_text.strip()
        ):
            return LearningOutcome(
                created_node=None,
                reason="missing_text_or_response",
            )

        # Embed the trigger once; reused for similar-input search and dedup
        query_emb = self._embed.embed(interaction.normalized_text)

        # The reinforcement logger is expected to have already persisted
        # this interaction before the learner is called. Filter it out of
        # the similar set so we don't double-count when re-adding the
        # current interaction's data below. Match on the (timestamp,
        # normalized_text, response_text) triple — including response
        # ensures two rapid same-text interactions whose timestamps
        # collide at float precision do NOT both get filtered (B1).
        similar = [
            (past, emb)
            for past, emb in self._find_similar_interactions(query_emb)
            if not (
                past.timestamp == interaction.timestamp
                and past.normalized_text == interaction.normalized_text
                and past.response_text == interaction.response_text
            )
        ]

        # The current interaction always counts as part of its own cluster
        cluster_size = len(similar) + 1

        if cluster_size < cfg.learning_min_repetitions:
            return LearningOutcome(
                created_node=None,
                reason="insufficient_repetitions",
                similar_count=cluster_size,
            )

        # Append the current interaction's data to the cluster for stability
        # and centroid calculations. `similar` is [(InteractionLog, vec), ...]
        cluster_texts = [i.normalized_text for i, _ in similar] + [
            interaction.normalized_text
        ]
        cluster_responses = [i.response_text for i, _ in similar] + [
            interaction.response_text
        ]
        cluster_input_embs = [emb for _, emb in similar] + [query_emb]

        if not self._check_response_stability(cluster_responses):
            return LearningOutcome(
                created_node=None,
                reason="responses_unstable",
                similar_count=cluster_size,
            )

        if self._already_covered(query_emb, interaction.response_text):
            return LearningOutcome(
                created_node=None,
                reason="already_covered_by_existing_node",
                similar_count=cluster_size,
            )

        # All gates passed — create the node.
        # B3: Add to FAISS BEFORE the graph store. If FAISS fails, the
        # graph stays clean (no phantom node). If put_node fails after
        # FAISS succeeds, the matcher's stale_hit_count surfaces the
        # drift on next routing attempt — already-handled state.
        node = self._build_node(
            cluster_texts, cluster_input_embs, interaction.response_text
        )
        self._faiss.add(node.pattern_id, node.embedding_vector)
        try:
            self._store.put_node(node)
        except Exception:
            # Roll back the FAISS insert so the system stays consistent
            self._faiss.remove(node.pattern_id)
            raise
        logger.info(
            "learner created node %r from %d similar interactions",
            node.pattern_id,
            cluster_size,
        )
        return LearningOutcome(
            created_node=node,
            reason="created",
            similar_count=cluster_size,
        )

    # --- Internals ---

    @staticmethod
    def _is_graph_route(route: RouteDecision) -> bool:
        return route in (RouteDecision.GRAPH_DIRECT, RouteDecision.GRAPH_COMPOSED)

    def _find_similar_interactions(
        self, query_emb: EmbeddingVector
    ) -> list[tuple[InteractionLog, EmbeddingVector]]:
        """Return recent LLM-route interactions whose input is similar.

        Each tuple is (interaction, its_embedding). Embeddings are
        computed in one batched call to amortize the model overhead
        across the lookback window.
        """
        cfg = self._config
        recent = self._persistence.get_interactions(
            limit=cfg.learning_lookback_window
        )
        # Filter to LLM-route interactions with a response_text up front
        # so we don't waste embedding calls on noise.
        candidates = [
            i
            for i in recent
            if not self._is_graph_route(i.route_decision)
            and i.normalized_text
            and i.response_text
        ]
        if not candidates:
            return []

        embeddings = self._embed.embed_batch(
            [i.normalized_text for i in candidates]
        )
        # Looser input-clustering threshold (architect B2): real
        # paraphrases of the same intent score 0.85-0.92 in E5-Small.
        threshold = cfg.learning_input_cluster_threshold
        out: list[tuple[InteractionLog, EmbeddingVector]] = []
        for interaction, emb in zip(candidates, embeddings):
            if _cosine(query_emb, emb) >= threshold:
                out.append((interaction, emb))
        return out

    def _check_response_stability(
        self, responses: list[str]
    ) -> bool:
        """All pairwise response similarities must clear the threshold.

        Returns True for length-1 input (vacuously stable). The cluster
        gate ensures we only call this when the cluster size is at
        least learning_min_repetitions, so length-1 cannot reach here
        in normal flow — but we handle it defensively.
        """
        if len(responses) < 2:
            return True
        embeddings = self._embed.embed_batch(responses)
        # Tighter response-stability threshold (architect B2): the
        # cluster's responses must agree more strongly than its inputs
        # cluster, otherwise we'd crystallize a node from divergent
        # answers.
        threshold = self._config.learning_response_stability_threshold
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                if _cosine(embeddings[i], embeddings[j]) < threshold:
                    return False
        return True

    def _already_covered(
        self, query_emb: EmbeddingVector, response_text: str
    ) -> bool:
        """Is there a node whose input AND response both match closely?

        Issue #22: an existing node whose input embedding is similar but
        whose response is divergent is NOT a dup — it's a different
        intent that happens to embed near this one. Both conditions
        must hold for "already covered".
        """
        cfg = self._config
        candidates = self._faiss.search(
            query_emb, k=cfg.faiss_search_k
        )
        if not candidates:
            return False

        response_emb = self._embed.embed(response_text)
        for node_id, input_sim in candidates:
            if input_sim < cfg.learning_dedup_threshold:
                continue
            try:
                node = self._store.get_node(node_id)
            except NodeNotFoundError:
                self._stale_dedup_hits += 1
                logger.warning(
                    "learner dedup: stale FAISS hit %r not in graph store "
                    "(total stale dedup hits: %d)",
                    node_id,
                    self._stale_dedup_hits,
                )
                continue
            if not node.response:
                continue
            node_resp_emb = self._embed.embed(node.response)
            if _cosine(response_emb, node_resp_emb) >= cfg.learning_dedup_threshold:
                return True
        return False

    def _build_node(
        self,
        cluster_texts: list[str],
        cluster_input_embs: list[EmbeddingVector],
        response_text: str,
    ) -> HabitNode:
        """Construct the new HabitNode with averaged + L2-normalized
        input embedding.

        W4: re-normalize the centroid so the on-disk node.embedding_vector
        is a unit vector. FAISSIndex normalizes on add() anyway, but
        any code that reads node.embedding_vector and computes its own
        cosine (tests, audits, alternate matchers) sees a value in
        [-1, 1] without surprises.
        """
        cfg = self._config
        avg_emb = _normalize_l2(_average_vectors(cluster_input_embs))
        return HabitNode(
            pattern_id=str(uuid.uuid4()),
            trigger_patterns=list(cluster_texts),
            embedding_vector=avg_emb,
            confidence=cfg.learning_starting_confidence,
            reinforcement_count=0,
            last_used_at=time.time(),
            decay_score=0.0,
            stability=Stability.LOW,
            risk_level=RiskLevel.LOW,
            response_form=ResponseForm.FIXED,
            response=response_text,
        )


# --- Module helpers ---


def _cosine(a: EmbeddingVector, b: EmbeddingVector) -> float:
    """Cosine similarity for two equal-length lists.

    Both inputs are expected to be already L2-normalized (the embedding
    service does this). NaN/Inf inputs return 0.0 to fail-safe rather
    than propagating into threshold comparisons. We do one final clamp
    at 1.0 to absorb float round-off so callers can compare against
    thresholds like 0.85 without worrying about 1.000001-ulps overshooting.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    # N3: NaN/Inf guard — propagating a NaN through `>= threshold`
    # silently fails (NaN compares False) and would mask learning bugs.
    if s != s or s == float("inf") or s == float("-inf"):
        return 0.0
    if s > 1.0:
        return 1.0
    if s < -1.0:
        return -1.0
    return s


def _average_vectors(vectors: list[EmbeddingVector]) -> EmbeddingVector:
    """Element-wise mean. Caller is responsible for any normalization."""
    if not vectors:
        return []
    n = len(vectors)
    dim = len(vectors[0])
    out = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            out[i] += v[i]
    for i in range(dim):
        out[i] /= n
    return out


def _normalize_l2(vec: EmbeddingVector) -> EmbeddingVector:
    """Return `vec` scaled to unit L2 norm. Zero-vector input returns
    zero (no-op); NaN/Inf input is left as-is for caller to detect."""
    if not vec:
        return []
    s = 0.0
    for x in vec:
        s += x * x
    if s <= 0.0:
        return list(vec)
    norm = s ** 0.5
    return [x / norm for x in vec]
