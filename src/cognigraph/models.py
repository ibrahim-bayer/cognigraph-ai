"""Core data models for CogniGraph."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class Stability(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ResponseForm(str, Enum):
    FIXED = "fixed"
    TEMPLATE = "template"
    PROCEDURAL = "procedural"


class RouteDecision(str, Enum):
    GRAPH_DIRECT = "graph_direct"
    GRAPH_COMPOSED = "graph_composed"
    LLM_FALLBACK = "llm_fallback"
    LLM_ONLY = "llm_only"


@dataclass
class ChildLink:
    """A directed link from a parent node to a child node."""

    habit_id: str
    condition: str | None = None
    order: int = 0

    def to_dict(self) -> dict:
        return {
            "habit_id": self.habit_id,
            "condition": self.condition,
            "order": self.order,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ChildLink:
        return cls(
            habit_id=data["habit_id"],
            condition=data.get("condition"),
            order=data.get("order", 0),
        )


@dataclass
class HabitNode:
    """A learned habit node in the graph."""

    pattern_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trigger_patterns: list[str] = field(default_factory=list)
    embedding_vector: list[float] = field(default_factory=list)
    confidence: float = 0.5
    reinforcement_count: int = 0
    last_used_at: float = field(default_factory=time.time)
    decay_score: float = 0.0
    stability: Stability = Stability.LOW
    risk_level: RiskLevel = RiskLevel.LOW

    response_form: ResponseForm = ResponseForm.FIXED
    response: str = ""

    children: list[ChildLink] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)

    is_composed: bool = False
    sequence_position: int | None = None

    def to_dict(self) -> dict:
        return {
            "pattern_id": self.pattern_id,
            "trigger_patterns": self.trigger_patterns,
            "embedding_vector": self.embedding_vector,
            "confidence": self.confidence,
            "reinforcement_count": self.reinforcement_count,
            "last_used_at": self.last_used_at,
            "decay_score": self.decay_score,
            "stability": self.stability.value,
            "risk_level": self.risk_level.value,
            "response_form": self.response_form.value,
            "response": self.response,
            "children": [c.to_dict() for c in self.children],
            "parents": self.parents,
            "is_composed": self.is_composed,
            "sequence_position": self.sequence_position,
        }

    @classmethod
    def from_dict(cls, data: dict) -> HabitNode:
        return cls(
            pattern_id=data["pattern_id"],
            trigger_patterns=data.get("trigger_patterns", []),
            embedding_vector=data.get("embedding_vector", []),
            confidence=data.get("confidence", 0.5),
            reinforcement_count=data.get("reinforcement_count", 0),
            last_used_at=data.get("last_used_at", time.time()),
            decay_score=data.get("decay_score", 0.0),
            stability=Stability(data.get("stability", "low")),
            risk_level=RiskLevel(data.get("risk_level", "low")),
            response_form=ResponseForm(data.get("response_form", "fixed")),
            response=data.get("response", ""),
            children=[ChildLink.from_dict(c) for c in data.get("children", [])],
            parents=data.get("parents", []),
            is_composed=data.get("is_composed", False),
            sequence_position=data.get("sequence_position"),
        )


@dataclass
class NormalizedInput:
    """Result of input normalization."""

    original: str
    normalized: str
    embedding: list[float] | None = None


@dataclass
class InteractionLog:
    """Record of a single interaction through the pipeline."""

    timestamp: float = field(default_factory=time.time)
    input_text: str = ""
    normalized_text: str = ""
    route_decision: RouteDecision = RouteDecision.LLM_ONLY
    matched_node_id: str | None = None
    llm_response: str | None = None
    response_text: str = ""
    latency_ms: float = 0.0
