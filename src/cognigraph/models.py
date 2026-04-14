"""Core data models for CogniGraph."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

from cognigraph.exceptions import PersistenceError


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
        try:
            habit_id = data["habit_id"]
            if not isinstance(habit_id, str) or not habit_id:
                raise PersistenceError("ChildLink.habit_id must be a non-empty string")
            return cls(
                habit_id=habit_id,
                condition=data.get("condition"),
                order=int(data.get("order", 0)),
            )
        except KeyError:
            raise PersistenceError("ChildLink data missing required field: habit_id")
        except (TypeError, ValueError) as e:
            raise PersistenceError(f"ChildLink deserialization failed: {e}") from e


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
    def _safe_enum(cls, enum_cls: type[Enum], value: str, default: Enum) -> Enum:
        try:
            return enum_cls(value)
        except ValueError:
            return default

    @classmethod
    def from_dict(cls, data: dict) -> HabitNode:
        try:
            pattern_id = data["pattern_id"]
            if not isinstance(pattern_id, str) or not pattern_id:
                raise PersistenceError("HabitNode.pattern_id must be a non-empty string")
            return cls(
                pattern_id=pattern_id,
                trigger_patterns=list(data.get("trigger_patterns", [])),
                embedding_vector=list(data.get("embedding_vector", [])),
                confidence=float(data.get("confidence", 0.5)),
                reinforcement_count=int(data.get("reinforcement_count", 0)),
                last_used_at=float(data.get("last_used_at", time.time())),
                decay_score=float(data.get("decay_score", 0.0)),
                stability=cls._safe_enum(Stability, data.get("stability", "low"), Stability.LOW),
                risk_level=cls._safe_enum(RiskLevel, data.get("risk_level", "low"), RiskLevel.LOW),
                response_form=cls._safe_enum(ResponseForm, data.get("response_form", "fixed"), ResponseForm.FIXED),
                response=str(data.get("response", "")),
                children=[ChildLink.from_dict(c) for c in data.get("children", [])],
                parents=list(data.get("parents", [])),
                is_composed=bool(data.get("is_composed", False)),
                sequence_position=data.get("sequence_position"),
            )
        except KeyError:
            raise PersistenceError("HabitNode data missing required field: pattern_id")
        except PersistenceError:
            raise
        except (TypeError, ValueError) as e:
            raise PersistenceError(f"HabitNode deserialization failed: {e}") from e


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


@dataclass
class MatchResult:
    """Outcome of a NodeMatcher.match() call.

    - `node`: the winning HabitNode, or None when route is LLM_ONLY and
      no viable candidate exists.
    - `score`: combined score (similarity * confidence), clamped to [0, 1].
    - `similarity`: raw cosine similarity of the winning candidate (clamped
      to [0, 1]), or 0.0 if no winner. Useful for logging distinct from `score`.
    - `route_decision`: one of the four RouteDecision values.
    - `candidates`: raw FAISS hits (node_id, similarity), top-k DESC.
      Preserved even when `node` is None so the learner can see drift.
    - `ambiguous`: True when the top-1 and top-2 combined scores are within
      `config.ambiguity_gap`, signalling a close-call routing decision.
    """

    node: HabitNode | None
    score: float
    similarity: float
    route_decision: RouteDecision
    candidates: list = field(default_factory=list)
    ambiguous: bool = False
