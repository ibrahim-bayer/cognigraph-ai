"""System-under-test wrappers for benchmarking.

Each wrapper implements the same `SystemUnderTest` protocol so the
runner can swap configurations and the reporter can compare numbers
apples-to-apples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class QueryResult:
    """One query's measurement."""

    response: str
    route: str  # "graph_hit" | "cache_hit" | "llm_call" | "error"
    input_tokens: int  # 0 if no LLM call happened
    output_tokens: int
    latency_ms: float  # end-to-end, including embedding/match/safety/LLM


@runtime_checkable
class SystemUnderTest(Protocol):
    """Contract every system being benchmarked must satisfy."""

    name: str

    def warmup(self, query: str) -> None:
        """Process a warm-up query. May be a no-op for stateless systems
        (raw LLM). Used for caches and learners that need to populate.
        Errors are logged by the runner; warmup never fails the benchmark."""

    def query(self, q: str) -> QueryResult:
        """Process one measurement query. Must record token counts and
        latency. Errors return a QueryResult with route="error"."""

    def close(self) -> None:
        """Release resources. Idempotent."""
