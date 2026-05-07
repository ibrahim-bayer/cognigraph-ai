"""Tests for SafetyBoundary."""

from __future__ import annotations

import pytest

from cognigraph.config import CogniGraphConfig
from cognigraph.models import (
    HabitNode,
    MatchResult,
    RiskLevel,
    RouteDecision,
    SafetyDecision,
    Stability,
)
from cognigraph.protocols import SafetyBoundaryProtocol
from cognigraph.safety import SafetyBoundary


# --- Helpers ---


def _node(
    *,
    pattern_id: str = "n1",
    risk: RiskLevel = RiskLevel.LOW,
    volatile: bool = False,
    confidence: float = 0.9,
) -> HabitNode:
    return HabitNode(
        pattern_id=pattern_id,
        trigger_patterns=[pattern_id],
        embedding_vector=[0.1, 0.2, 0.3, 0.4],
        confidence=confidence,
        risk_level=risk,
        volatile=volatile,
        response=f"response-{pattern_id}",
    )


def _match(
    *,
    node: HabitNode | None,
    route: RouteDecision,
    candidates: list[tuple[str, float]] | None = None,
    score: float = 0.9,
    similarity: float = 0.9,
    ambiguous: bool = False,
) -> MatchResult:
    return MatchResult(
        node=node,
        score=score,
        similarity=similarity,
        route_decision=route,
        candidates=candidates if candidates is not None else (
            [(node.pattern_id, similarity)] if node else []
        ),
        ambiguous=ambiguous,
    )


# --- Fixtures ---


@pytest.fixture
def boundary() -> SafetyBoundary:
    return SafetyBoundary()


# --- Protocol conformance ---


class TestProtocolConformance:
    def test_implements_safety_boundary_protocol(
        self, boundary: SafetyBoundary
    ) -> None:
        assert isinstance(boundary, SafetyBoundaryProtocol)


# --- Risk gating ---


class TestRiskGating:
    def test_high_risk_node_escalates(self, boundary: SafetyBoundary) -> None:
        match = _match(
            node=_node(risk=RiskLevel.HIGH),
            route=RouteDecision.GRAPH_DIRECT,
        )
        decision = boundary.check(match, input_text="hello")
        assert decision.safe is False
        assert decision.reason == "high_risk_node"
        assert decision.override_route == RouteDecision.LLM_FALLBACK

    def test_medium_risk_node_passes(self, boundary: SafetyBoundary) -> None:
        match = _match(
            node=_node(risk=RiskLevel.MEDIUM),
            route=RouteDecision.GRAPH_DIRECT,
        )
        assert boundary.check(match, input_text="hello").safe is True

    def test_low_risk_node_passes(self, boundary: SafetyBoundary) -> None:
        match = _match(
            node=_node(risk=RiskLevel.LOW),
            route=RouteDecision.GRAPH_DIRECT,
        )
        assert boundary.check(match, input_text="hello").safe is True


# --- Volatility flag ---


class TestVolatilityFlag:
    def test_volatile_node_escalates(self, boundary: SafetyBoundary) -> None:
        match = _match(
            node=_node(volatile=True),
            route=RouteDecision.GRAPH_COMPOSED,
        )
        decision = boundary.check(match, input_text="what time is it")
        assert decision.safe is False
        assert decision.reason == "volatile_node"
        assert decision.override_route == RouteDecision.LLM_FALLBACK

    def test_non_volatile_node_passes(self, boundary: SafetyBoundary) -> None:
        match = _match(
            node=_node(volatile=False),
            route=RouteDecision.GRAPH_DIRECT,
        )
        assert boundary.check(match, input_text="what's my name").safe is True


# --- Ambiguity detection (delegates to matcher's `ambiguous` flag) ---


class TestAmbiguityDetection:
    def test_matcher_ambiguous_flag_escalates(
        self, boundary: SafetyBoundary
    ) -> None:
        """B2 fix: the safety boundary uses match_result.ambiguous,
        which is the matcher's own flag computed on combined score
        (sim × confidence). Single source of truth."""
        match = _match(
            node=_node(pattern_id="a"),
            route=RouteDecision.GRAPH_DIRECT,
            ambiguous=True,
        )
        decision = boundary.check(match, input_text="probe")
        assert decision.safe is False
        assert decision.reason == "ambiguous_match"
        assert decision.override_route == RouteDecision.LLM_FALLBACK

    def test_non_ambiguous_flag_passes(
        self, boundary: SafetyBoundary
    ) -> None:
        match = _match(
            node=_node(pattern_id="a"),
            route=RouteDecision.GRAPH_DIRECT,
            ambiguous=False,
        )
        assert boundary.check(match, input_text="probe").safe is True

    def test_ambiguous_flag_only_fires_on_graph_routes(
        self, boundary: SafetyBoundary
    ) -> None:
        """LLM routes are already deferring to the LLM — even an
        ambiguous match doesn't need a safety override."""
        match = _match(
            node=_node(pattern_id="a"),
            route=RouteDecision.LLM_FALLBACK,
            ambiguous=True,
        )
        assert boundary.check(match, input_text="probe").safe is True


# --- Pattern blocklist ---


class TestBlocklist:
    def test_blocklist_pattern_match_escalates_graph_route(self) -> None:
        cfg = CogniGraphConfig(blocklist_patterns=["password", "ssn"])
        boundary = SafetyBoundary(cfg)
        match = _match(node=_node(), route=RouteDecision.GRAPH_DIRECT)
        decision = boundary.check(match, input_text="what's my password")
        assert decision.safe is False
        assert decision.reason == "blocklist_match"
        assert decision.override_route == RouteDecision.LLM_FALLBACK

    def test_blocklist_match_with_no_node_overrides_to_llm_only(self) -> None:
        cfg = CogniGraphConfig(blocklist_patterns=["password"])
        boundary = SafetyBoundary(cfg)
        match = _match(node=None, route=RouteDecision.LLM_ONLY, candidates=[])
        decision = boundary.check(match, input_text="what's my password")
        assert decision.safe is False
        assert decision.override_route == RouteDecision.LLM_ONLY

    def test_blocklist_case_insensitive(self) -> None:
        cfg = CogniGraphConfig(blocklist_patterns=["PASSWORD"])
        boundary = SafetyBoundary(cfg)
        match = _match(node=_node(), route=RouteDecision.GRAPH_DIRECT)
        decision = boundary.check(match, input_text="What is my Password?")
        assert decision.safe is False
        assert decision.reason == "blocklist_match"

    def test_blocklist_substring_match(self) -> None:
        cfg = CogniGraphConfig(blocklist_patterns=["confidential"])
        boundary = SafetyBoundary(cfg)
        match = _match(node=_node(), route=RouteDecision.GRAPH_DIRECT)
        decision = boundary.check(
            match, input_text="show me the confidential report"
        )
        assert decision.safe is False

    def test_blocklist_no_match_passes(self) -> None:
        cfg = CogniGraphConfig(blocklist_patterns=["password", "ssn"])
        boundary = SafetyBoundary(cfg)
        match = _match(node=_node(), route=RouteDecision.GRAPH_DIRECT)
        assert boundary.check(match, input_text="what's the weather").safe is True

    def test_empty_blocklist_passes_anything(
        self, boundary: SafetyBoundary
    ) -> None:
        match = _match(node=_node(), route=RouteDecision.GRAPH_DIRECT)
        assert boundary.check(match, input_text="anything goes").safe is True

    def test_empty_input_text_no_match(self) -> None:
        cfg = CogniGraphConfig(blocklist_patterns=["password"])
        boundary = SafetyBoundary(cfg)
        match = _match(node=_node(), route=RouteDecision.GRAPH_DIRECT)
        assert boundary.check(match, input_text="").safe is True

    def test_add_to_blocklist(self, boundary: SafetyBoundary) -> None:
        boundary.add_to_blocklist("medical")
        match = _match(node=_node(), route=RouteDecision.GRAPH_DIRECT)
        assert (
            boundary.check(match, input_text="give me medical advice").safe
            is False
        )

    def test_remove_from_blocklist(self) -> None:
        cfg = CogniGraphConfig(blocklist_patterns=["medical", "password"])
        boundary = SafetyBoundary(cfg)
        boundary.remove_from_blocklist("medical")
        match = _match(node=_node(), route=RouteDecision.GRAPH_DIRECT)
        # "medical" no longer blocked
        assert boundary.check(match, input_text="medical question").safe is True
        # "password" still blocked
        assert boundary.check(match, input_text="my password").safe is False

    def test_remove_nonexistent_pattern_is_noop(
        self, boundary: SafetyBoundary
    ) -> None:
        boundary.remove_from_blocklist("never-was-there")
        # Should not raise

    def test_add_blocklist_normalizes_case_and_whitespace(
        self, boundary: SafetyBoundary
    ) -> None:
        boundary.add_to_blocklist("  PASSWORD  ")
        match = _match(node=_node(), route=RouteDecision.GRAPH_DIRECT)
        assert (
            boundary.check(match, input_text="my password").safe is False
        )

    def test_add_empty_pattern_is_noop(
        self, boundary: SafetyBoundary
    ) -> None:
        boundary.add_to_blocklist("")
        boundary.add_to_blocklist("   ")
        # An empty pattern would otherwise match every input — confirm
        # that didn't happen
        match = _match(node=_node(), route=RouteDecision.GRAPH_DIRECT)
        assert boundary.check(match, input_text="hello world").safe is True

    def test_blocklist_property_is_readonly_snapshot(
        self, boundary: SafetyBoundary
    ) -> None:
        boundary.add_to_blocklist("foo")
        snapshot = boundary.blocklist
        assert "foo" in snapshot
        # Mutating the snapshot doesn't affect the boundary
        with pytest.raises(AttributeError):
            snapshot.add("bar")  # type: ignore[attr-defined]


# --- Route-aware behavior ---


class TestRouteAware:
    def test_llm_only_route_skips_node_checks(
        self, boundary: SafetyBoundary
    ) -> None:
        """If matcher already routes LLM_ONLY, safety doesn't gate the
        (potentially unsafe) node — the LLM is going to handle it."""
        match = _match(node=None, route=RouteDecision.LLM_ONLY, candidates=[])
        assert boundary.check(match, input_text="").safe is True

    def test_llm_fallback_skips_node_checks_even_with_high_risk_node(
        self, boundary: SafetyBoundary
    ) -> None:
        """LLM_FALLBACK already goes to the LLM. Even if the matched
        node is high-risk, the safety boundary doesn't need to override —
        the LLM is the safer path anyway."""
        match = _match(
            node=_node(risk=RiskLevel.HIGH),
            route=RouteDecision.LLM_FALLBACK,
        )
        assert boundary.check(match, input_text="").safe is True

    def test_graph_direct_with_safe_node_passes(
        self, boundary: SafetyBoundary
    ) -> None:
        match = _match(
            node=_node(risk=RiskLevel.LOW, volatile=False),
            route=RouteDecision.GRAPH_DIRECT,
            candidates=[("n1", 0.95), ("other", 0.40)],
        )
        decision = boundary.check(match, input_text="harmless query")
        assert decision.safe is True
        assert decision.reason is None
        assert decision.override_route is None

    def test_graph_route_missing_node_overrides_llm_only(
        self, boundary: SafetyBoundary
    ) -> None:
        """If the matcher inconsistently returns GRAPH_* with node=None,
        the safety boundary catches it."""
        match = _match(
            node=None, route=RouteDecision.GRAPH_DIRECT, candidates=[]
        )
        decision = boundary.check(match, input_text="hi")
        assert decision.safe is False
        assert decision.reason == "graph_route_missing_node"
        assert decision.override_route == RouteDecision.LLM_ONLY


# --- Check ordering ---


class TestCheckOrdering:
    def test_blocklist_fires_before_other_checks(self) -> None:
        """A blocklist match preempts even high-risk-node detection.
        Blocklist is the only route-agnostic check; it fires first."""
        cfg = CogniGraphConfig(blocklist_patterns=["password"])
        boundary = SafetyBoundary(cfg)
        match = _match(
            node=_node(risk=RiskLevel.HIGH, volatile=True),
            route=RouteDecision.GRAPH_DIRECT,
            candidates=[("a", 0.92), ("b", 0.91)],  # also ambiguous
        )
        decision = boundary.check(match, input_text="my password is secret")
        assert decision.reason == "blocklist_match"

    def test_high_risk_fires_before_volatile_and_ambiguity(
        self, boundary: SafetyBoundary
    ) -> None:
        match = _match(
            node=_node(risk=RiskLevel.HIGH, volatile=True),
            route=RouteDecision.GRAPH_DIRECT,
            ambiguous=True,
        )
        decision = boundary.check(match, input_text="")
        assert decision.reason == "high_risk_node"

    def test_volatile_fires_before_ambiguity(
        self, boundary: SafetyBoundary
    ) -> None:
        match = _match(
            node=_node(risk=RiskLevel.LOW, volatile=True),
            route=RouteDecision.GRAPH_DIRECT,
            ambiguous=True,
        )
        decision = boundary.check(match, input_text="")
        assert decision.reason == "volatile_node"


# --- Additional architect tests (B1, W1, W4, N2) ---


class TestThreadSafety:
    """W1: blocklist mutation must not crash check() running concurrently."""

    def test_concurrent_add_and_check_no_crash(self) -> None:
        import threading

        cfg = CogniGraphConfig(blocklist_patterns=["base1", "base2"])
        boundary = SafetyBoundary(cfg)
        match = _match(node=_node(), route=RouteDecision.GRAPH_DIRECT)

        errors: list[Exception] = []
        stop = threading.Event()

        def reader() -> None:
            try:
                while not stop.is_set():
                    boundary.check(match, input_text="hello world")
            except Exception as e:  # pragma: no cover - failure path
                errors.append(e)

        def writer() -> None:
            try:
                for i in range(200):
                    boundary.add_to_blocklist(f"pattern{i}")
                    boundary.remove_from_blocklist(f"pattern{i}")
            except Exception as e:  # pragma: no cover
                errors.append(e)

        r = threading.Thread(target=reader)
        w = threading.Thread(target=writer)
        r.start()
        w.start()
        w.join(timeout=5)
        stop.set()
        r.join(timeout=5)

        assert errors == [], f"concurrent access raised: {errors}"


class TestObservability:
    """W4: unsafe blocks emit observability events."""

    def test_block_counts_track_reasons(self) -> None:
        cfg = CogniGraphConfig(blocklist_patterns=["password"])
        boundary = SafetyBoundary(cfg)

        # Two blocklist hits + one risk hit + one safe pass
        boundary.check(
            _match(node=_node(), route=RouteDecision.GRAPH_DIRECT),
            input_text="my password",
        )
        boundary.check(
            _match(node=_node(), route=RouteDecision.GRAPH_DIRECT),
            input_text="another password leak",
        )
        boundary.check(
            _match(
                node=_node(risk=RiskLevel.HIGH),
                route=RouteDecision.GRAPH_DIRECT,
            ),
            input_text="harmless",
        )
        boundary.check(
            _match(node=_node(), route=RouteDecision.GRAPH_DIRECT),
            input_text="harmless",
        )

        counts = boundary.block_counts
        assert counts.get("blocklist_match") == 2
        assert counts.get("high_risk_node") == 1
        # Safe pass doesn't increment any counter
        assert sum(counts.values()) == 3

    def test_block_counts_returns_copy(
        self, boundary: SafetyBoundary
    ) -> None:
        # Mutating the returned dict must not affect internal state
        snapshot = boundary.block_counts
        snapshot["fake"] = 999
        assert "fake" not in boundary.block_counts


class TestRegexSpecialChars:
    """N2: substring matching is literal, not regex. A pattern like 'a.b'
    must not match 'axb'."""

    def test_dot_pattern_is_literal(self) -> None:
        cfg = CogniGraphConfig(blocklist_patterns=["a.b"])
        boundary = SafetyBoundary(cfg)
        match = _match(node=_node(), route=RouteDecision.GRAPH_DIRECT)

        # Literal "a.b" does NOT match "axb" — substring is literal
        assert (
            boundary.check(match, input_text="say axb please").safe is True
        )
        # Literal "a.b" DOES match "ya.b!"
        assert (
            boundary.check(match, input_text="ya.b!").safe is False
        )

    def test_star_pattern_is_literal(self, boundary: SafetyBoundary) -> None:
        boundary.add_to_blocklist("a*b")
        match = _match(node=_node(), route=RouteDecision.GRAPH_DIRECT)
        # Literal "a*b" does NOT match "ab" or "aab"
        assert boundary.check(match, input_text="ab plain").safe is True
        # Literal "a*b" matches "...a*b..."
        assert boundary.check(match, input_text="hi a*b yo").safe is False


class TestVolatileSerialization:
    """B1: the volatile field must round-trip through HabitNode.to_dict /
    from_dict. Without this test, a future serialization refactor could
    silently drop the field and the volatility check becomes a no-op."""

    def test_volatile_true_round_trips(self) -> None:
        node = HabitNode(
            pattern_id="vol-1",
            trigger_patterns=["x"],
            embedding_vector=[0.1, 0.2, 0.3, 0.4],
            volatile=True,
            response="now",
        )
        restored = HabitNode.from_dict(node.to_dict())
        assert restored.volatile is True

    def test_volatile_false_round_trips(self) -> None:
        node = HabitNode(
            pattern_id="vol-2",
            trigger_patterns=["x"],
            embedding_vector=[0.1, 0.2, 0.3, 0.4],
            volatile=False,
            response="static",
        )
        restored = HabitNode.from_dict(node.to_dict())
        assert restored.volatile is False

    def test_volatile_default_when_missing_from_dict(self) -> None:
        """Forward-compat: an old DB row whose JSON doesn't have the
        `volatile` key must load with the default (False) — not crash."""
        legacy_data = {
            "pattern_id": "old-row",
            "trigger_patterns": ["x"],
            "embedding_vector": [0.1, 0.2, 0.3, 0.4],
            "response": "ok",
            # no "volatile" key
        }
        restored = HabitNode.from_dict(legacy_data)
        assert restored.volatile is False
