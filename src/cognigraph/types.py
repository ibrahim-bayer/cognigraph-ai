"""Type aliases and shared data types for CogniGraph."""

from __future__ import annotations

from dataclasses import dataclass, field

NodeId = str
EmbeddingVector = list[float]
Timestamp = float


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    text: str = ""
    model: str = ""
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
