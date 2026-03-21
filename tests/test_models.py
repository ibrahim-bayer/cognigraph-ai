"""Tests for CogniGraph data models."""

from __future__ import annotations

import time

from cognigraph.models import (
    ChildLink,
    HabitNode,
    InteractionLog,
    NormalizedInput,
    ResponseForm,
    RiskLevel,
    RouteDecision,
    Stability,
)


class TestEnums:
    def test_stability_values(self) -> None:
        assert Stability.HIGH.value == "high"
        assert Stability.MEDIUM.value == "medium"
        assert Stability.LOW.value == "low"

    def test_risk_level_values(self) -> None:
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MEDIUM.value == "medium"
        assert RiskLevel.HIGH.value == "high"

    def test_response_form_values(self) -> None:
        assert ResponseForm.FIXED.value == "fixed"
        assert ResponseForm.TEMPLATE.value == "template"
        assert ResponseForm.PROCEDURAL.value == "procedural"

    def test_route_decision_values(self) -> None:
        assert RouteDecision.GRAPH_DIRECT.value == "graph_direct"
        assert RouteDecision.GRAPH_COMPOSED.value == "graph_composed"
        assert RouteDecision.LLM_FALLBACK.value == "llm_fallback"
        assert RouteDecision.LLM_ONLY.value == "llm_only"

    def test_enums_are_string_enums(self) -> None:
        assert isinstance(Stability.HIGH, str)
        assert isinstance(RiskLevel.LOW, str)
        assert isinstance(ResponseForm.FIXED, str)
        assert isinstance(RouteDecision.GRAPH_DIRECT, str)


class TestChildLink:
    def test_creates_with_defaults(self) -> None:
        link = ChildLink(habit_id="node-1")
        assert link.habit_id == "node-1"
        assert link.condition is None
        assert link.order == 0

    def test_creates_with_all_fields(self) -> None:
        link = ChildLink(habit_id="node-1", condition="if_staged", order=2)
        assert link.condition == "if_staged"
        assert link.order == 2

    def test_ordering(self) -> None:
        links = [
            ChildLink(habit_id="c", order=3),
            ChildLink(habit_id="a", order=1),
            ChildLink(habit_id="b", order=2),
        ]
        sorted_links = sorted(links, key=lambda l: l.order)
        assert [l.habit_id for l in sorted_links] == ["a", "b", "c"]

    def test_to_dict(self) -> None:
        link = ChildLink(habit_id="node-1", condition="ready", order=1)
        d = link.to_dict()
        assert d == {"habit_id": "node-1", "condition": "ready", "order": 1}

    def test_from_dict(self) -> None:
        link = ChildLink.from_dict({"habit_id": "node-1", "condition": "ready", "order": 1})
        assert link.habit_id == "node-1"
        assert link.condition == "ready"
        assert link.order == 1

    def test_from_dict_missing_optional_fields(self) -> None:
        link = ChildLink.from_dict({"habit_id": "node-1"})
        assert link.condition is None
        assert link.order == 0


class TestHabitNode:
    def test_creates_with_defaults(self) -> None:
        node = HabitNode()
        assert node.pattern_id  # UUID generated
        assert node.trigger_patterns == []
        assert node.embedding_vector == []
        assert node.confidence == 0.5
        assert node.reinforcement_count == 0
        assert node.last_used_at > 0
        assert node.decay_score == 0.0
        assert node.stability == Stability.LOW
        assert node.risk_level == RiskLevel.LOW
        assert node.response_form == ResponseForm.FIXED
        assert node.response == ""
        assert node.children == []
        assert node.parents == []
        assert node.is_composed is False
        assert node.sequence_position is None

    def test_creates_with_custom_fields(self) -> None:
        node = HabitNode(
            pattern_id="test-node",
            trigger_patterns=["hello", "hi"],
            confidence=0.9,
            stability=Stability.HIGH,
            risk_level=RiskLevel.MEDIUM,
            response_form=ResponseForm.TEMPLATE,
            response="Hello {name}",
            is_composed=True,
            sequence_position=1,
        )
        assert node.pattern_id == "test-node"
        assert node.trigger_patterns == ["hello", "hi"]
        assert node.confidence == 0.9
        assert node.stability == Stability.HIGH
        assert node.risk_level == RiskLevel.MEDIUM
        assert node.response == "Hello {name}"
        assert node.is_composed is True
        assert node.sequence_position == 1

    def test_unique_ids(self) -> None:
        node1 = HabitNode()
        node2 = HabitNode()
        assert node1.pattern_id != node2.pattern_id

    def test_to_dict(self) -> None:
        node = HabitNode(
            pattern_id="test-1",
            trigger_patterns=["hello"],
            confidence=0.8,
            stability=Stability.MEDIUM,
            risk_level=RiskLevel.LOW,
            response_form=ResponseForm.FIXED,
            response="Hi there",
            children=[ChildLink(habit_id="child-1", order=1)],
            parents=["parent-1"],
        )
        d = node.to_dict()
        assert d["pattern_id"] == "test-1"
        assert d["trigger_patterns"] == ["hello"]
        assert d["confidence"] == 0.8
        assert d["stability"] == "medium"
        assert d["risk_level"] == "low"
        assert d["response_form"] == "fixed"
        assert d["response"] == "Hi there"
        assert len(d["children"]) == 1
        assert d["children"][0]["habit_id"] == "child-1"
        assert d["parents"] == ["parent-1"]

    def test_serialization_round_trip(self) -> None:
        original = HabitNode(
            pattern_id="round-trip",
            trigger_patterns=["test input", "another input"],
            embedding_vector=[0.1, 0.2, 0.3],
            confidence=0.75,
            reinforcement_count=5,
            last_used_at=1000.0,
            decay_score=0.1,
            stability=Stability.MEDIUM,
            risk_level=RiskLevel.HIGH,
            response_form=ResponseForm.TEMPLATE,
            response="Answer: {value}",
            children=[
                ChildLink(habit_id="c1", condition="ready", order=1),
                ChildLink(habit_id="c2", order=2),
            ],
            parents=["p1", "p2"],
            is_composed=True,
            sequence_position=3,
        )
        d = original.to_dict()
        restored = HabitNode.from_dict(d)

        assert restored.pattern_id == original.pattern_id
        assert restored.trigger_patterns == original.trigger_patterns
        assert restored.embedding_vector == original.embedding_vector
        assert restored.confidence == original.confidence
        assert restored.reinforcement_count == original.reinforcement_count
        assert restored.last_used_at == original.last_used_at
        assert restored.decay_score == original.decay_score
        assert restored.stability == original.stability
        assert restored.risk_level == original.risk_level
        assert restored.response_form == original.response_form
        assert restored.response == original.response
        assert len(restored.children) == 2
        assert restored.children[0].habit_id == "c1"
        assert restored.children[0].condition == "ready"
        assert restored.children[1].habit_id == "c2"
        assert restored.parents == ["p1", "p2"]
        assert restored.is_composed is True
        assert restored.sequence_position == 3

    def test_from_dict_missing_optional_fields(self) -> None:
        """Handles schema evolution — missing fields get defaults."""
        minimal = {"pattern_id": "minimal-node"}
        node = HabitNode.from_dict(minimal)
        assert node.pattern_id == "minimal-node"
        assert node.trigger_patterns == []
        assert node.confidence == 0.5
        assert node.stability == Stability.LOW
        assert node.children == []
        assert node.is_composed is False


class TestNormalizedInput:
    def test_without_embedding(self) -> None:
        ni = NormalizedInput(original="Hello World", normalized="hello world")
        assert ni.original == "Hello World"
        assert ni.normalized == "hello world"
        assert ni.embedding is None

    def test_with_embedding(self) -> None:
        ni = NormalizedInput(
            original="Hello",
            normalized="hello",
            embedding=[0.1, 0.2, 0.3],
        )
        assert ni.embedding == [0.1, 0.2, 0.3]


class TestInteractionLog:
    def test_creates_with_defaults(self) -> None:
        log = InteractionLog()
        assert log.timestamp > 0
        assert log.input_text == ""
        assert log.normalized_text == ""
        assert log.route_decision == RouteDecision.LLM_ONLY
        assert log.matched_node_id is None
        assert log.llm_response is None
        assert log.response_text == ""
        assert log.latency_ms == 0.0

    def test_captures_all_fields(self) -> None:
        log = InteractionLog(
            timestamp=1000.0,
            input_text="What is your name?",
            normalized_text="what is your name?",
            route_decision=RouteDecision.GRAPH_DIRECT,
            matched_node_id="node-123",
            llm_response=None,
            response_text="My name is CogniGraph",
            latency_ms=0.5,
        )
        assert log.timestamp == 1000.0
        assert log.input_text == "What is your name?"
        assert log.route_decision == RouteDecision.GRAPH_DIRECT
        assert log.matched_node_id == "node-123"
        assert log.llm_response is None
        assert log.response_text == "My name is CogniGraph"
        assert log.latency_ms == 0.5

    def test_llm_fallback_fields(self) -> None:
        log = InteractionLog(
            route_decision=RouteDecision.LLM_FALLBACK,
            matched_node_id="node-456",
            llm_response="LLM generated this",
        )
        assert log.route_decision == RouteDecision.LLM_FALLBACK
        assert log.matched_node_id == "node-456"
        assert log.llm_response == "LLM generated this"
