"""Tests for NodeMatcher — real GraphStore + real FAISSIndex, synthetic vectors."""

from __future__ import annotations

import pytest

from cognigraph.config import CogniGraphConfig
from cognigraph.graph_store import InMemoryGraphStore
from cognigraph.matcher import NodeMatcher
from cognigraph.models import ChildLink, HabitNode, MatchResult, RouteDecision
from cognigraph.protocols import NodeMatcherProtocol
from cognigraph.vector_index import FAISSIndex


DIM = 4


# --- Fixtures ---


@pytest.fixture
def config() -> CogniGraphConfig:
    # similarity_threshold=0.85, confidence_threshold=0.7,
    # fallback_similarity=0.6 (defaults)
    return CogniGraphConfig(faiss_search_k=5)


@pytest.fixture
def store() -> InMemoryGraphStore:
    return InMemoryGraphStore()


@pytest.fixture
def index() -> FAISSIndex:
    return FAISSIndex(dimension=DIM)


@pytest.fixture
def matcher(
    store: InMemoryGraphStore,
    index: FAISSIndex,
    config: CogniGraphConfig,
) -> NodeMatcher:
    return NodeMatcher(store, index, config)


# --- Helpers ---


def _basis(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    return v


def _mix(a: int, b: int, frac: float) -> list[float]:
    """Return a unit vector mixing basis[a] and basis[b] by `frac`.

    frac=0 → all b, frac=1 → all a. Cosine similarity between the result
    and basis[a] is exactly `frac` (for 0 ≤ frac ≤ 1), because
      v · e_a = frac / sqrt(frac^2 + (1-frac)^2)  ... not quite.
    So instead build the mix with components (cos θ, sin θ) and report
    via cos θ.
    """
    import math

    # frac is the desired cosine to e_a
    theta = math.acos(max(-1.0, min(1.0, frac)))
    v = [0.0] * DIM
    v[a] = math.cos(theta)
    v[b] = math.sin(theta)
    return v


def _put(
    store: InMemoryGraphStore,
    index: FAISSIndex,
    pattern_id: str,
    vector: list[float],
    *,
    confidence: float = 0.9,
) -> HabitNode:
    node = HabitNode(
        pattern_id=pattern_id,
        trigger_patterns=[pattern_id],
        embedding_vector=vector,
        confidence=confidence,
        response=f"response-{pattern_id}",
    )
    store.put_node(node)
    index.add(pattern_id, vector)
    return node


# --- Empty / degenerate cases ---


class TestEmptyCases:
    def test_empty_index_returns_llm_only(self, matcher: NodeMatcher) -> None:
        result = matcher.match(_basis(0))
        assert result.node is None
        assert result.score == 0.0
        assert result.route_decision == RouteDecision.LLM_ONLY
        assert result.candidates == []

    def test_no_graph_node_for_faiss_hit_degrades_to_llm_only(
        self,
        store: InMemoryGraphStore,
        index: FAISSIndex,
        matcher: NodeMatcher,
    ) -> None:
        """FAISS has an entry but the graph store does not (stale index)."""
        index.add("ghost", _basis(0))
        result = matcher.match(_basis(0))
        assert result.node is None
        assert result.route_decision == RouteDecision.LLM_ONLY
        # candidates carries the FAISS hit for debugging
        assert len(result.candidates) == 1
        assert result.candidates[0][0] == "ghost"


# --- GRAPH_DIRECT ---


class TestGraphDirect:
    def test_high_sim_high_conf_no_children(
        self,
        store: InMemoryGraphStore,
        index: FAISSIndex,
        matcher: NodeMatcher,
    ) -> None:
        _put(store, index, "a", _basis(0), confidence=0.95)
        result = matcher.match(_basis(0))

        assert result.node is not None
        assert result.node.pattern_id == "a"
        assert result.route_decision == RouteDecision.GRAPH_DIRECT
        assert result.score == pytest.approx(0.95, abs=1e-5)

    def test_exactly_at_thresholds_routes_direct(
        self,
        store: InMemoryGraphStore,
        index: FAISSIndex,
        config: CogniGraphConfig,
    ) -> None:
        """sim == similarity_threshold and conf == confidence_threshold
        should still route as GRAPH_DIRECT (inclusive bounds)."""
        m = NodeMatcher(store, index, config)
        # Inject a vector at similarity exactly similarity_threshold=0.85
        _put(store, index, "a", _basis(0), confidence=config.confidence_threshold)
        probe = _mix(0, 1, config.similarity_threshold)

        result = m.match(probe)
        assert result.route_decision == RouteDecision.GRAPH_DIRECT


# --- GRAPH_COMPOSED ---


class TestGraphComposed:
    def test_high_sim_high_conf_with_children(
        self,
        store: InMemoryGraphStore,
        index: FAISSIndex,
        matcher: NodeMatcher,
    ) -> None:
        _put(store, index, "root", _basis(0), confidence=0.95)
        _put(store, index, "step1", _basis(1), confidence=0.9)
        _put(store, index, "step2", _basis(2), confidence=0.9)
        store.add_link("root", ChildLink(habit_id="step1", order=1))
        store.add_link("root", ChildLink(habit_id="step2", order=2))

        result = matcher.match(_basis(0))
        assert result.node is not None
        assert result.node.pattern_id == "root"
        assert result.route_decision == RouteDecision.GRAPH_COMPOSED

    def test_composed_root_still_chosen_over_leaf_siblings(
        self,
        store: InMemoryGraphStore,
        index: FAISSIndex,
        matcher: NodeMatcher,
    ) -> None:
        """A leaf sibling with equal similarity but lower confidence
        must NOT outrank the composed root."""
        _put(store, index, "root", _basis(0), confidence=0.95)
        _put(store, index, "leaf", _basis(0), confidence=0.7)  # same vec
        _put(store, index, "child", _basis(1), confidence=0.9)
        store.add_link("root", ChildLink(habit_id="child", order=1))

        result = matcher.match(_basis(0))
        assert result.node.pattern_id == "root"
        assert result.route_decision == RouteDecision.GRAPH_COMPOSED


# --- LLM_FALLBACK ---


class TestLLMFallback:
    def test_medium_similarity_routes_fallback(
        self,
        store: InMemoryGraphStore,
        index: FAISSIndex,
        config: CogniGraphConfig,
        matcher: NodeMatcher,
    ) -> None:
        _put(store, index, "a", _basis(0), confidence=0.95)
        # Probe at sim=0.7 (between fallback=0.6 and similarity_threshold=0.85)
        probe = _mix(0, 1, 0.7)
        result = matcher.match(probe)
        assert result.node is not None
        assert result.route_decision == RouteDecision.LLM_FALLBACK

    def test_high_similarity_low_confidence_routes_fallback(
        self,
        store: InMemoryGraphStore,
        index: FAISSIndex,
        matcher: NodeMatcher,
    ) -> None:
        """Strong semantic match but node hasn't earned enough confidence."""
        _put(store, index, "a", _basis(0), confidence=0.5)  # below 0.7
        result = matcher.match(_basis(0))
        assert result.node is not None
        assert result.route_decision == RouteDecision.LLM_FALLBACK
        # Combined score reflects the weak confidence
        assert result.score == pytest.approx(0.5, abs=1e-5)


# --- LLM_ONLY ---


class TestLLMOnly:
    def test_below_fallback_similarity_routes_llm_only(
        self,
        store: InMemoryGraphStore,
        index: FAISSIndex,
        matcher: NodeMatcher,
    ) -> None:
        _put(store, index, "a", _basis(0), confidence=0.95)
        probe = _mix(0, 1, 0.3)  # sim 0.3 < 0.6 fallback
        result = matcher.match(probe)
        assert result.route_decision == RouteDecision.LLM_ONLY

    def test_orthogonal_probe_routes_llm_only(
        self,
        store: InMemoryGraphStore,
        index: FAISSIndex,
        matcher: NodeMatcher,
    ) -> None:
        _put(store, index, "a", _basis(0), confidence=0.95)
        result = matcher.match(_basis(1))  # orthogonal → sim=0
        assert result.route_decision == RouteDecision.LLM_ONLY


# --- Ranking & candidates ---


class TestRankingAndCandidates:
    def test_multiple_candidates_ranked_by_combined_score(
        self,
        store: InMemoryGraphStore,
        index: FAISSIndex,
        matcher: NodeMatcher,
    ) -> None:
        # Three nodes. `b` has the highest similarity to the probe,
        # but `a` has the highest combined (sim*conf). Matcher must
        # pick `a`, not `b`.
        _put(store, index, "a", _basis(0), confidence=1.0)   # sim=0.9, combined=0.9
        _put(store, index, "b", _basis(1), confidence=0.3)   # sim=1.0, combined=0.3
        _put(store, index, "c", _basis(2), confidence=0.5)   # sim=0, combined=0

        probe = _mix(1, 0, 0.9)  # sim ~0.9 to basis(1)='b', ~0.435 to 'a'
        # Wait, with _mix(a=1, b=0, frac=0.9) → cos theta to e_1 = 0.9,
        # cos to e_0 = sin theta ≈ 0.436. combined(a) = 0.436,
        # combined(b) = 0.9 * 0.3 = 0.27, combined(c) = 0.
        # So `a` wins by combined score even though `b` has higher raw sim.
        result = matcher.match(probe)
        assert result.node is not None
        assert result.node.pattern_id == "a"
        # candidates are the raw FAISS hits (not rescored)
        assert len(result.candidates) == 3
        candidate_ids = {c[0] for c in result.candidates}
        assert candidate_ids == {"a", "b", "c"}

    def test_candidates_include_top_k(
        self,
        store: InMemoryGraphStore,
        index: FAISSIndex,
        matcher: NodeMatcher,
    ) -> None:
        _put(store, index, "a", _basis(0))
        _put(store, index, "b", _basis(1))
        _put(store, index, "c", _basis(2))
        result = matcher.match(_basis(0))
        ids = [c[0] for c in result.candidates]
        # k=5 default, so all 3 nodes returned
        assert len(ids) == 3
        assert "a" in ids


# --- Score behavior ---


class TestScoring:
    def test_negative_similarity_clamped_to_zero(
        self,
        store: InMemoryGraphStore,
        index: FAISSIndex,
        matcher: NodeMatcher,
    ) -> None:
        """Anti-parallel query must not produce a negative combined score."""
        _put(store, index, "a", _basis(0), confidence=0.95)
        antipode = [-1.0, 0.0, 0.0, 0.0]
        result = matcher.match(antipode)
        # Similarity ≈ -1 → clamped to 0 → combined 0 → LLM_ONLY
        assert result.score == 0.0
        assert result.route_decision == RouteDecision.LLM_ONLY

    def test_combined_score_is_sim_times_confidence(
        self,
        store: InMemoryGraphStore,
        index: FAISSIndex,
        matcher: NodeMatcher,
    ) -> None:
        _put(store, index, "a", _basis(0), confidence=0.8)
        probe = _mix(0, 1, 0.9)
        result = matcher.match(probe)
        # similarity ≈ 0.9, confidence = 0.8 → combined ≈ 0.72
        assert result.score == pytest.approx(0.9 * 0.8, abs=1e-4)


# --- Configurability ---


class TestConfigurableThresholds:
    def test_lowering_similarity_threshold_promotes_to_direct(
        self,
        store: InMemoryGraphStore,
        index: FAISSIndex,
    ) -> None:
        strict = CogniGraphConfig(similarity_threshold=0.95)
        relaxed = CogniGraphConfig(similarity_threshold=0.75)

        m_strict = NodeMatcher(store, index, strict)
        m_relaxed = NodeMatcher(store, index, relaxed)

        _put(store, index, "a", _basis(0), confidence=0.95)
        probe = _mix(0, 1, 0.8)  # similarity 0.8

        strict_result = m_strict.match(probe)
        relaxed_result = m_relaxed.match(probe)

        assert strict_result.route_decision == RouteDecision.LLM_FALLBACK
        assert relaxed_result.route_decision == RouteDecision.GRAPH_DIRECT

    def test_raising_confidence_threshold_forces_fallback(
        self,
        store: InMemoryGraphStore,
        index: FAISSIndex,
    ) -> None:
        loose = CogniGraphConfig(confidence_threshold=0.5)
        tight = CogniGraphConfig(confidence_threshold=0.95)

        _put(store, index, "a", _basis(0), confidence=0.8)

        m_loose = NodeMatcher(store, index, loose)
        m_tight = NodeMatcher(store, index, tight)

        assert m_loose.match(_basis(0)).route_decision == RouteDecision.GRAPH_DIRECT
        assert m_tight.match(_basis(0)).route_decision == RouteDecision.LLM_FALLBACK

    def test_lowering_fallback_similarity_pulls_from_llm_only(
        self,
        store: InMemoryGraphStore,
        index: FAISSIndex,
    ) -> None:
        _put(store, index, "a", _basis(0), confidence=0.95)
        probe = _mix(0, 1, 0.4)  # sim 0.4

        strict = CogniGraphConfig(fallback_similarity=0.6)  # default
        permissive = CogniGraphConfig(fallback_similarity=0.3)

        m_strict = NodeMatcher(store, index, strict)
        m_permissive = NodeMatcher(store, index, permissive)

        assert m_strict.match(probe).route_decision == RouteDecision.LLM_ONLY
        assert m_permissive.match(probe).route_decision == RouteDecision.LLM_FALLBACK

    def test_rejects_confidence_threshold_one(self) -> None:
        """Config must reject confidence_threshold=1.0 — a silent footgun."""
        with pytest.raises(ValueError, match="confidence_threshold"):
            CogniGraphConfig(confidence_threshold=1.0)


# --- Protocol conformance ---


class TestProtocolConformance:
    def test_implements_node_matcher_protocol(
        self, matcher: NodeMatcher
    ) -> None:
        assert isinstance(matcher, NodeMatcherProtocol)


# --- MatchResult surface ---


class TestMatchResultSurface:
    def test_result_carries_raw_similarity_separately_from_combined(
        self,
        store: InMemoryGraphStore,
        index: FAISSIndex,
        matcher: NodeMatcher,
    ) -> None:
        _put(store, index, "a", _basis(0), confidence=0.8)
        probe = _mix(0, 1, 0.9)  # similarity ≈ 0.9
        result = matcher.match(probe)

        # similarity is raw cosine, score is combined
        assert result.similarity == pytest.approx(0.9, abs=1e-4)
        assert result.score == pytest.approx(0.9 * 0.8, abs=1e-4)
        # The two should not be equal when confidence != 1.0
        assert result.similarity != result.score

    def test_empty_match_similarity_is_zero(self, matcher: NodeMatcher) -> None:
        result = matcher.match(_basis(0))
        assert result.similarity == 0.0
        assert result.score == 0.0

    def test_stale_all_match_similarity_is_zero(
        self,
        index: FAISSIndex,
        matcher: NodeMatcher,
    ) -> None:
        index.add("ghost", _basis(0))
        result = matcher.match(_basis(0))
        assert result.similarity == 0.0
        assert result.node is None

    def test_ambiguous_flag_set_on_close_top_two(
        self,
        store: InMemoryGraphStore,
        index: FAISSIndex,
    ) -> None:
        """Two near-duplicate candidates within ambiguity_gap → ambiguous=True."""
        cfg = CogniGraphConfig(ambiguity_gap=0.1)
        m = NodeMatcher(store, index, cfg)

        _put(store, index, "a", _basis(0), confidence=0.9)
        _put(store, index, "b", _mix(0, 1, 0.95), confidence=0.9)

        result = m.match(_basis(0))
        assert result.ambiguous is True

    def test_ambiguous_flag_false_on_clear_winner(
        self,
        store: InMemoryGraphStore,
        index: FAISSIndex,
        matcher: NodeMatcher,
    ) -> None:
        _put(store, index, "a", _basis(0), confidence=0.95)
        _put(store, index, "b", _basis(2), confidence=0.3)
        result = matcher.match(_basis(0))
        assert result.ambiguous is False

    def test_ambiguous_flag_false_with_single_candidate(
        self,
        store: InMemoryGraphStore,
        index: FAISSIndex,
        matcher: NodeMatcher,
    ) -> None:
        _put(store, index, "a", _basis(0), confidence=0.95)
        result = matcher.match(_basis(0))
        assert result.ambiguous is False


# --- Observability ---


class TestStaleHitCounter:
    def test_counter_increments_on_drift(
        self,
        store: InMemoryGraphStore,
        index: FAISSIndex,
        matcher: NodeMatcher,
    ) -> None:
        _put(store, index, "live", _basis(0), confidence=0.9)
        index.add("ghost1", _basis(1))
        index.add("ghost2", _basis(2))

        assert matcher.stale_hit_count == 0
        result = matcher.match(_basis(0))

        # The live node wins; the two ghosts increment the counter
        assert result.node is not None
        assert result.node.pattern_id == "live"
        assert matcher.stale_hit_count == 2


# --- Top-k + extra coverage ---


class TestTopKAndCoverage:
    def test_candidates_capped_at_faiss_search_k(
        self,
        store: InMemoryGraphStore,
        index: FAISSIndex,
    ) -> None:
        """Regression: FAISS search must honor configured k."""
        # Use a larger dim so we have 6 linearly independent directions
        idx = FAISSIndex(dimension=6)
        s = InMemoryGraphStore()
        for i in range(6):
            vec = [0.0] * 6
            vec[i] = 1.0
            node = HabitNode(
                pattern_id=f"n{i}",
                trigger_patterns=[f"n{i}"],
                embedding_vector=vec,
                confidence=0.9,
                response=f"r{i}",
            )
            s.put_node(node)
            idx.add(f"n{i}", vec)

        cfg = CogniGraphConfig(faiss_search_k=3)
        m = NodeMatcher(s, idx, cfg)
        query = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        result = m.match(query)

        assert len(result.candidates) == 3
        idx.close()

    def test_mixed_stale_and_live_candidates(
        self,
        store: InMemoryGraphStore,
        index: FAISSIndex,
        matcher: NodeMatcher,
    ) -> None:
        """Stale entries are skipped individually; live winner still wins."""
        _put(store, index, "live1", _basis(0), confidence=0.95)
        index.add("ghost", _basis(1))
        _put(store, index, "live2", _basis(2), confidence=0.9)

        result = matcher.match(_basis(0))
        assert result.node is not None
        assert result.node.pattern_id == "live1"
        # candidates carries all 3 FAISS hits (including the ghost)
        assert len(result.candidates) == 3
        assert matcher.stale_hit_count == 1

    def test_all_candidates_below_fallback_multi(
        self,
        store: InMemoryGraphStore,
        index: FAISSIndex,
        matcher: NodeMatcher,
    ) -> None:
        """Multiple nodes, probe orthogonal to all → LLM_ONLY."""
        _put(store, index, "a", _basis(0), confidence=0.9)
        _put(store, index, "b", _basis(1), confidence=0.9)
        _put(store, index, "c", _basis(2), confidence=0.9)
        result = matcher.match(_basis(3))  # orthogonal to all three
        assert result.route_decision == RouteDecision.LLM_ONLY

    def test_wrong_dim_embedding_propagates(
        self,
        store: InMemoryGraphStore,
        index: FAISSIndex,
        matcher: NodeMatcher,
    ) -> None:
        """Programmer error: wrong-dim embedding raises from FAISS."""
        _put(store, index, "a", _basis(0), confidence=0.9)
        with pytest.raises(ValueError, match="dim mismatch"):
            matcher.match([1.0, 0.0])
