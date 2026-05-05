"""Tests for ReinforcementLogger — real GraphStore + real SQLite via tmp_path."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from cognigraph.config import CogniGraphConfig
from cognigraph.graph_store import InMemoryGraphStore
from cognigraph.models import (
    HabitNode,
    InteractionLog,
    ResponseForm,
    RiskLevel,
    RouteDecision,
    Stability,
)
from cognigraph.persistence import SQLitePersistence
from cognigraph.protocols import ReinforcementLoggerProtocol
from cognigraph.reinforcement import ReinforcementLogger


# --- Fixtures ---


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "reinf.db")


@pytest.fixture
def store() -> InMemoryGraphStore:
    return InMemoryGraphStore()


@pytest.fixture
def persistence(db_path: str):
    p = SQLitePersistence(db_path)
    yield p
    p.close()


@pytest.fixture
def config() -> CogniGraphConfig:
    return CogniGraphConfig()  # defaults: boost=0.02, MEDIUM=5, HIGH=20


@pytest.fixture
def logger(
    store: InMemoryGraphStore,
    persistence: SQLitePersistence,
    config: CogniGraphConfig,
) -> ReinforcementLogger:
    return ReinforcementLogger(store, persistence, config)


def _put_node(
    store: InMemoryGraphStore,
    pattern_id: str,
    *,
    confidence: float = 0.5,
    reinforcement_count: int = 0,
    stability: Stability = Stability.LOW,
    decay_score: float = 0.5,
) -> HabitNode:
    node = HabitNode(
        pattern_id=pattern_id,
        trigger_patterns=[pattern_id],
        embedding_vector=[0.1, 0.2, 0.3],
        confidence=confidence,
        reinforcement_count=reinforcement_count,
        last_used_at=1000.0,  # known stale value so tests can detect updates
        decay_score=decay_score,
        stability=stability,
        risk_level=RiskLevel.LOW,
        response_form=ResponseForm.FIXED,
        response=f"response-{pattern_id}",
    )
    store.put_node(node)
    return node


def _interaction(
    *,
    route: RouteDecision,
    matched_node_id: str | None,
    text: str = "test",
    timestamp: float | None = None,
) -> InteractionLog:
    return InteractionLog(
        timestamp=timestamp or time.time(),
        input_text=text,
        normalized_text=text,
        route_decision=route,
        matched_node_id=matched_node_id,
        response_text=f"answer to {text}",
        latency_ms=1.0,
    )


# --- Protocol conformance ---


class TestProtocolConformance:
    def test_implements_reinforcement_logger_protocol(
        self, logger: ReinforcementLogger
    ) -> None:
        assert isinstance(logger, ReinforcementLoggerProtocol)


# --- Reinforcement happy path ---


class TestReinforcementHappyPath:
    def test_graph_direct_increments_count(
        self,
        store: InMemoryGraphStore,
        logger: ReinforcementLogger,
    ) -> None:
        _put_node(store, "a", reinforcement_count=3)
        result = logger.log_and_reinforce(
            _interaction(route=RouteDecision.GRAPH_DIRECT, matched_node_id="a")
        )
        assert result is True
        assert store.get_node("a").reinforcement_count == 4

    def test_graph_composed_also_reinforces(
        self,
        store: InMemoryGraphStore,
        logger: ReinforcementLogger,
    ) -> None:
        _put_node(store, "a", reinforcement_count=2)
        result = logger.log_and_reinforce(
            _interaction(route=RouteDecision.GRAPH_COMPOSED, matched_node_id="a")
        )
        assert result is True
        assert store.get_node("a").reinforcement_count == 3

    def test_last_used_at_updated_to_now(
        self,
        store: InMemoryGraphStore,
        logger: ReinforcementLogger,
    ) -> None:
        node = _put_node(store, "a")
        assert node.last_used_at == 1000.0  # stale sentinel
        before = time.time()
        logger.log_and_reinforce(
            _interaction(route=RouteDecision.GRAPH_DIRECT, matched_node_id="a")
        )
        updated = store.get_node("a").last_used_at
        assert before <= updated <= time.time()

    def test_decay_score_reset_to_zero(
        self,
        store: InMemoryGraphStore,
        logger: ReinforcementLogger,
    ) -> None:
        _put_node(store, "a", decay_score=0.8)
        logger.log_and_reinforce(
            _interaction(route=RouteDecision.GRAPH_DIRECT, matched_node_id="a")
        )
        assert store.get_node("a").decay_score == 0.0

    def test_returns_true_on_reinforcement(
        self,
        store: InMemoryGraphStore,
        logger: ReinforcementLogger,
    ) -> None:
        _put_node(store, "a")
        assert (
            logger.log_and_reinforce(
                _interaction(
                    route=RouteDecision.GRAPH_DIRECT, matched_node_id="a"
                )
            )
            is True
        )


# --- Confidence cap ---


class TestConfidenceCap:
    def test_boost_applied_normally(
        self,
        store: InMemoryGraphStore,
        logger: ReinforcementLogger,
        config: CogniGraphConfig,
    ) -> None:
        _put_node(store, "a", confidence=0.5)
        logger.log_and_reinforce(
            _interaction(route=RouteDecision.GRAPH_DIRECT, matched_node_id="a")
        )
        assert store.get_node("a").confidence == pytest.approx(
            0.5 + config.confidence_boost, abs=1e-9
        )

    def test_capped_at_one_when_close(
        self,
        store: InMemoryGraphStore,
        logger: ReinforcementLogger,
    ) -> None:
        _put_node(store, "a", confidence=0.99)
        logger.log_and_reinforce(
            _interaction(route=RouteDecision.GRAPH_DIRECT, matched_node_id="a")
        )
        assert store.get_node("a").confidence == 1.0

    def test_no_overshoot_at_one(
        self,
        store: InMemoryGraphStore,
        logger: ReinforcementLogger,
    ) -> None:
        _put_node(store, "a", confidence=1.0)
        logger.log_and_reinforce(
            _interaction(route=RouteDecision.GRAPH_DIRECT, matched_node_id="a")
        )
        assert store.get_node("a").confidence == 1.0


# --- Stability promotion ---


class TestStabilityPromotion:
    def test_stays_low_below_medium_threshold(
        self,
        store: InMemoryGraphStore,
        logger: ReinforcementLogger,
        config: CogniGraphConfig,
    ) -> None:
        # 4 reinforcements → still LOW (medium threshold is 5)
        _put_node(store, "a", reinforcement_count=3, stability=Stability.LOW)
        logger.log_and_reinforce(
            _interaction(route=RouteDecision.GRAPH_DIRECT, matched_node_id="a")
        )
        assert store.get_node("a").reinforcement_count == 4
        assert store.get_node("a").stability == Stability.LOW

    def test_promotes_to_medium_at_threshold(
        self,
        store: InMemoryGraphStore,
        logger: ReinforcementLogger,
        config: CogniGraphConfig,
    ) -> None:
        # Pre-load with count=4, this reinforcement makes it 5 = MEDIUM
        _put_node(store, "a", reinforcement_count=4, stability=Stability.LOW)
        logger.log_and_reinforce(
            _interaction(route=RouteDecision.GRAPH_DIRECT, matched_node_id="a")
        )
        assert store.get_node("a").reinforcement_count == 5
        assert store.get_node("a").stability == Stability.MEDIUM

    def test_stays_medium_between_thresholds(
        self,
        store: InMemoryGraphStore,
        logger: ReinforcementLogger,
    ) -> None:
        _put_node(store, "a", reinforcement_count=18, stability=Stability.MEDIUM)
        logger.log_and_reinforce(
            _interaction(route=RouteDecision.GRAPH_DIRECT, matched_node_id="a")
        )
        assert store.get_node("a").reinforcement_count == 19
        assert store.get_node("a").stability == Stability.MEDIUM

    def test_promotes_to_high_at_threshold(
        self,
        store: InMemoryGraphStore,
        logger: ReinforcementLogger,
    ) -> None:
        _put_node(store, "a", reinforcement_count=19, stability=Stability.MEDIUM)
        logger.log_and_reinforce(
            _interaction(route=RouteDecision.GRAPH_DIRECT, matched_node_id="a")
        )
        assert store.get_node("a").reinforcement_count == 20
        assert store.get_node("a").stability == Stability.HIGH

    def test_no_demotion_after_promotion(
        self,
        store: InMemoryGraphStore,
        logger: ReinforcementLogger,
    ) -> None:
        # Stability is recomputed from count, so a node already at HIGH
        # remains HIGH on subsequent reinforcements.
        _put_node(store, "a", reinforcement_count=25, stability=Stability.HIGH)
        for _ in range(3):
            logger.log_and_reinforce(
                _interaction(
                    route=RouteDecision.GRAPH_DIRECT, matched_node_id="a"
                )
            )
        assert store.get_node("a").stability == Stability.HIGH


# --- Non-graph routes log but don't reinforce ---


class TestNonGraphRoutes:
    def test_llm_only_does_not_reinforce(
        self,
        store: InMemoryGraphStore,
        logger: ReinforcementLogger,
    ) -> None:
        _put_node(store, "a", reinforcement_count=3)
        result = logger.log_and_reinforce(
            _interaction(
                route=RouteDecision.LLM_ONLY, matched_node_id=None
            )
        )
        assert result is False
        assert store.get_node("a").reinforcement_count == 3

    def test_llm_fallback_does_not_reinforce_even_with_node_id(
        self,
        store: InMemoryGraphStore,
        logger: ReinforcementLogger,
    ) -> None:
        # LLM_FALLBACK can have a matched_node_id (the matcher found
        # something but wasn't confident enough). The logger must NOT
        # reinforce in that case — that's the learner's job.
        _put_node(store, "a", reinforcement_count=3, confidence=0.5)
        result = logger.log_and_reinforce(
            _interaction(route=RouteDecision.LLM_FALLBACK, matched_node_id="a")
        )
        assert result is False
        assert store.get_node("a").reinforcement_count == 3
        assert store.get_node("a").confidence == 0.5

    def test_llm_only_still_logs(
        self,
        persistence: SQLitePersistence,
        logger: ReinforcementLogger,
    ) -> None:
        logger.log_and_reinforce(
            _interaction(route=RouteDecision.LLM_ONLY, matched_node_id=None)
        )
        assert len(persistence.get_interactions()) == 1

    def test_llm_fallback_still_logs(
        self,
        persistence: SQLitePersistence,
        logger: ReinforcementLogger,
    ) -> None:
        logger.log_and_reinforce(
            _interaction(route=RouteDecision.LLM_FALLBACK, matched_node_id="x")
        )
        assert len(persistence.get_interactions()) == 1


# --- Stale node handling ---


class TestStaleMatchedNodeId:
    def test_stale_node_id_skips_reinforcement(
        self,
        logger: ReinforcementLogger,
        persistence: SQLitePersistence,
    ) -> None:
        result = logger.log_and_reinforce(
            _interaction(
                route=RouteDecision.GRAPH_DIRECT, matched_node_id="ghost"
            )
        )
        assert result is False
        assert logger.stale_reinforcement_count == 1

    def test_stale_id_still_logs_interaction(
        self,
        logger: ReinforcementLogger,
        persistence: SQLitePersistence,
    ) -> None:
        logger.log_and_reinforce(
            _interaction(
                route=RouteDecision.GRAPH_DIRECT, matched_node_id="ghost"
            )
        )
        logs = persistence.get_interactions()
        assert len(logs) == 1
        assert logs[0].matched_node_id == "ghost"

    def test_missing_node_id_with_graph_route_skips(
        self,
        logger: ReinforcementLogger,
        store: InMemoryGraphStore,
    ) -> None:
        # A pipeline bug: claims GRAPH_DIRECT but didn't set a node_id.
        # Logger should not crash, decline to reinforce, AND bump the
        # observability counter so the regression doesn't go silent.
        result = logger.log_and_reinforce(
            _interaction(
                route=RouteDecision.GRAPH_DIRECT, matched_node_id=None
            )
        )
        assert result is False
        assert logger.stale_reinforcement_count == 0  # not stale, missing
        assert logger.missing_node_id_count == 1

    def test_stale_counter_is_per_instance(
        self,
        store: InMemoryGraphStore,
        persistence: SQLitePersistence,
        config: CogniGraphConfig,
    ) -> None:
        rl_a = ReinforcementLogger(store, persistence, config)
        rl_b = ReinforcementLogger(store, persistence, config)

        rl_a.log_and_reinforce(
            _interaction(route=RouteDecision.GRAPH_DIRECT, matched_node_id="ghost")
        )
        assert rl_a.stale_reinforcement_count == 1
        assert rl_b.stale_reinforcement_count == 0

    def test_missing_id_counter_logs_warning(
        self,
        logger: ReinforcementLogger,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level("WARNING", logger="cognigraph.reinforcement"):
            logger.log_and_reinforce(
                _interaction(
                    route=RouteDecision.GRAPH_DIRECT, matched_node_id=None
                )
            )
        assert any(
            "without matched_node_id" in r.message for r in caplog.records
        )


# --- Persistence integration ---


class TestPersistenceIntegration:
    def test_interaction_persisted_with_all_fields(
        self,
        store: InMemoryGraphStore,
        logger: ReinforcementLogger,
        persistence: SQLitePersistence,
    ) -> None:
        _put_node(store, "a")
        log = InteractionLog(
            timestamp=12345.0,
            input_text="raw input",
            normalized_text="raw input",
            route_decision=RouteDecision.GRAPH_DIRECT,
            matched_node_id="a",
            llm_response=None,
            response_text="answer",
            latency_ms=42.5,
        )
        logger.log_and_reinforce(log)

        [persisted] = persistence.get_interactions()
        assert persisted.input_text == "raw input"
        assert persisted.matched_node_id == "a"
        assert persisted.route_decision == RouteDecision.GRAPH_DIRECT
        assert persisted.latency_ms == 42.5
        assert persisted.timestamp == 12345.0

    def test_get_node_history_returns_node_specific_logs(
        self,
        store: InMemoryGraphStore,
        logger: ReinforcementLogger,
    ) -> None:
        _put_node(store, "a")
        _put_node(store, "b")
        for _ in range(3):
            logger.log_and_reinforce(
                _interaction(route=RouteDecision.GRAPH_DIRECT, matched_node_id="a")
            )
        for _ in range(2):
            logger.log_and_reinforce(
                _interaction(route=RouteDecision.GRAPH_DIRECT, matched_node_id="b")
            )

        hist_a = logger.get_node_history("a")
        hist_b = logger.get_node_history("b")
        assert len(hist_a) == 3
        assert len(hist_b) == 2
        assert all(log.matched_node_id == "a" for log in hist_a)
        assert all(log.matched_node_id == "b" for log in hist_b)

    def test_get_node_history_respects_limit(
        self,
        store: InMemoryGraphStore,
        logger: ReinforcementLogger,
    ) -> None:
        _put_node(store, "a")
        for i in range(10):
            logger.log_and_reinforce(
                _interaction(
                    route=RouteDecision.GRAPH_DIRECT,
                    matched_node_id="a",
                    timestamp=float(i),
                )
            )
        assert len(logger.get_node_history("a", limit=5)) == 5
        assert len(logger.get_node_history("a", limit=100)) == 10


# --- Configurability ---


class TestConfigurability:
    def test_custom_confidence_boost(
        self,
        store: InMemoryGraphStore,
        persistence: SQLitePersistence,
    ) -> None:
        cfg = CogniGraphConfig(confidence_boost=0.5)
        _put_node(store, "a", confidence=0.0)
        rl = ReinforcementLogger(store, persistence, cfg)
        rl.log_and_reinforce(
            _interaction(route=RouteDecision.GRAPH_DIRECT, matched_node_id="a")
        )
        assert store.get_node("a").confidence == pytest.approx(0.5)

    def test_custom_stability_thresholds(
        self,
        store: InMemoryGraphStore,
        persistence: SQLitePersistence,
    ) -> None:
        cfg = CogniGraphConfig(
            stability_medium_threshold=2, stability_high_threshold=4
        )
        _put_node(store, "a", reinforcement_count=0, stability=Stability.LOW)
        rl = ReinforcementLogger(store, persistence, cfg)
        # 1 → LOW, 2 → MEDIUM, 3 → MEDIUM, 4 → HIGH
        rl.log_and_reinforce(
            _interaction(route=RouteDecision.GRAPH_DIRECT, matched_node_id="a")
        )
        assert store.get_node("a").stability == Stability.LOW
        rl.log_and_reinforce(
            _interaction(route=RouteDecision.GRAPH_DIRECT, matched_node_id="a")
        )
        assert store.get_node("a").stability == Stability.MEDIUM
        rl.log_and_reinforce(
            _interaction(route=RouteDecision.GRAPH_DIRECT, matched_node_id="a")
        )
        assert store.get_node("a").stability == Stability.MEDIUM
        rl.log_and_reinforce(
            _interaction(route=RouteDecision.GRAPH_DIRECT, matched_node_id="a")
        )
        assert store.get_node("a").stability == Stability.HIGH

    def test_default_thresholds_match_spec(
        self, config: CogniGraphConfig
    ) -> None:
        # Sanity-check the spec values haven't drifted
        assert config.confidence_boost == 0.02
        assert config.stability_medium_threshold == 5
        assert config.stability_high_threshold == 20


# --- Failure path propagation (W6) ---


class _RaisingPersistence:
    """Minimal PersistenceProtocol that raises on log_interaction.

    Implements the rest as no-ops so isinstance(rl, ReinforcementLoggerProtocol)
    holds and the logger can construct successfully.
    """

    def __init__(self, error: Exception) -> None:
        self._error = error

    def log_interaction(self, log: InteractionLog) -> None:
        raise self._error

    def get_interactions(
        self, limit: int = 100, offset: int = 0
    ) -> list[InteractionLog]:
        return []

    def get_interactions_for_node(self, node_id: str) -> list[InteractionLog]:
        return []


class TestFailurePropagation:
    def test_persistence_error_propagates(
        self, store: InMemoryGraphStore, config: CogniGraphConfig
    ) -> None:
        """Logger contract: log_interaction errors must NOT be swallowed.

        The interaction log is the source of truth for the learner; if
        we silently dropped writes the learner would mine a corrupted
        history.
        """
        from cognigraph.exceptions import PersistenceError

        rl = ReinforcementLogger(
            store, _RaisingPersistence(PersistenceError("disk full")), config
        )
        _put_node(store, "a")
        with pytest.raises(PersistenceError, match="disk full"):
            rl.log_and_reinforce(
                _interaction(
                    route=RouteDecision.GRAPH_DIRECT, matched_node_id="a"
                )
            )
        # Reinforcement was NOT applied because the log raised first
        assert store.get_node("a").reinforcement_count == 0

    def test_put_node_error_propagates(
        self,
        persistence: SQLitePersistence,
        config: CogniGraphConfig,
    ) -> None:
        """If the graph store fails, the exception bubbles up.

        Per the docstring contract: the in-memory node may already be
        partially mutated by _reinforce, but the interaction log is
        authoritative.
        """

        class _FailingStore:
            def __init__(self, real_store: InMemoryGraphStore) -> None:
                self._real = real_store
                self._fail_on_put = False

            def get_node(self, node_id):
                return self._real.get_node(node_id)

            def put_node(self, node) -> None:
                if self._fail_on_put:
                    raise RuntimeError("graph store offline")
                self._real.put_node(node)

            def remove_node(self, node_id):
                return self._real.remove_node(node_id)

            def get_children(self, node_id):
                return self._real.get_children(node_id)

            def get_parents(self, node_id):
                return self._real.get_parents(node_id)

            def add_link(self, parent_id, child_link):
                return self._real.add_link(parent_id, child_link)

            def remove_link(self, parent_id, child_id, *, condition=...):
                return self._real.remove_link(parent_id, child_id, condition=condition)

            def all_nodes(self):
                return self._real.all_nodes()

            def node_count(self):
                return self._real.node_count()

        real = InMemoryGraphStore()
        _put_node(real, "a")
        store = _FailingStore(real)
        rl = ReinforcementLogger(store, persistence, config)

        store._fail_on_put = True
        with pytest.raises(RuntimeError, match="graph store offline"):
            rl.log_and_reinforce(
                _interaction(
                    route=RouteDecision.GRAPH_DIRECT, matched_node_id="a"
                )
            )

        # Per docstring: interaction was logged before put_node ran
        assert len(persistence.get_interactions()) == 1


# --- limit=None on get_node_history (W3) ---


class TestGetNodeHistoryLimits:
    def test_limit_none_returns_full_history(
        self,
        store: InMemoryGraphStore,
        logger: ReinforcementLogger,
    ) -> None:
        _put_node(store, "a")
        for i in range(7):
            logger.log_and_reinforce(
                _interaction(
                    route=RouteDecision.GRAPH_DIRECT,
                    matched_node_id="a",
                    timestamp=float(i),
                )
            )
        history = logger.get_node_history("a", limit=None)
        assert len(history) == 7

    def test_limit_zero_returns_empty_list(
        self,
        store: InMemoryGraphStore,
        logger: ReinforcementLogger,
    ) -> None:
        _put_node(store, "a")
        logger.log_and_reinforce(
            _interaction(route=RouteDecision.GRAPH_DIRECT, matched_node_id="a")
        )
        assert logger.get_node_history("a", limit=0) == []

    def test_negative_limit_raises(
        self,
        store: InMemoryGraphStore,
        logger: ReinforcementLogger,
    ) -> None:
        _put_node(store, "a")
        with pytest.raises(ValueError, match=">= 0 or None"):
            logger.get_node_history("a", limit=-1)


# --- LLM_FALLBACK matched_id round-trip (W6) ---


class TestLLMFallbackMatchedIdRoundTrip:
    def test_llm_fallback_preserves_matched_id_in_log(
        self,
        store: InMemoryGraphStore,
        logger: ReinforcementLogger,
        persistence: SQLitePersistence,
    ) -> None:
        """LLM_FALLBACK doesn't reinforce, but the log MUST preserve the
        matched_node_id so the learner can mine "near miss" interactions."""
        _put_node(store, "a", confidence=0.4)
        logger.log_and_reinforce(
            _interaction(
                route=RouteDecision.LLM_FALLBACK,
                matched_node_id="a",
                text="probe",
            )
        )
        [persisted] = persistence.get_interactions()
        assert persisted.matched_node_id == "a"
        assert persisted.route_decision == RouteDecision.LLM_FALLBACK


# --- Stability snap warning (W2) ---


class TestStabilitySnapWarning:
    def test_normal_promotion_does_not_warn(
        self,
        store: InMemoryGraphStore,
        logger: ReinforcementLogger,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A normal LOW→MEDIUM transition at the threshold count must NOT
        log a snap warning (count and stability agreed before the call)."""
        _put_node(store, "a", reinforcement_count=4, stability=Stability.LOW)
        with caplog.at_level("WARNING", logger="cognigraph.reinforcement"):
            logger.log_and_reinforce(
                _interaction(
                    route=RouteDecision.GRAPH_DIRECT, matched_node_id="a"
                )
            )
        assert store.get_node("a").stability == Stability.MEDIUM
        assert not any("snapped" in r.message for r in caplog.records)

    def test_stored_disagreement_warns(
        self,
        store: InMemoryGraphStore,
        logger: ReinforcementLogger,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A node loaded with HIGH stability but count=2 must surface a
        snap warning when reinforced (the stored stability disagreed
        with the count BEFORE this reinforcement)."""
        _put_node(store, "a", reinforcement_count=2, stability=Stability.HIGH)
        with caplog.at_level("WARNING", logger="cognigraph.reinforcement"):
            logger.log_and_reinforce(
                _interaction(
                    route=RouteDecision.GRAPH_DIRECT, matched_node_id="a"
                )
            )
        # After: count=3, recomputed stability = LOW (below medium threshold=5)
        assert store.get_node("a").stability == Stability.LOW
        assert any(
            "snapped" in r.message and "high" in r.message
            for r in caplog.records
        )
