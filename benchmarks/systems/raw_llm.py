"""RawLLMSystem — baseline: call Claude on every query, no caching, no graph."""

from __future__ import annotations

import logging
import time

from cognigraph.config import CogniGraphConfig
from cognigraph.exceptions import LLMError
from cognigraph.llm_client import ClaudeLLMProvider

from benchmarks.systems import QueryResult, SystemUnderTest

logger = logging.getLogger(__name__)


class RawLLMSystem:
    """The stateless baseline. Every query is a fresh LLM call."""

    name = "raw_llm"

    def __init__(
        self,
        config: CogniGraphConfig | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._config = config or CogniGraphConfig()
        self._system_prompt = (
            system_prompt or self._config.pipeline_system_prompt
        )
        self._llm = ClaudeLLMProvider(config=self._config)

    def warmup(self, query: str) -> None:
        # Raw LLM is stateless; warmup is a no-op (would just waste tokens).
        pass

    def query(self, q: str) -> QueryResult:
        start = time.perf_counter()
        try:
            response = self._llm.generate(prompt=q, system=self._system_prompt)
        except LLMError as e:
            elapsed = (time.perf_counter() - start) * 1000.0
            logger.warning("LLM call failed: %s", e)
            return QueryResult(
                response=f"[error: {e}]",
                route="error",
                input_tokens=0,
                output_tokens=0,
                latency_ms=elapsed,
            )
        elapsed = (time.perf_counter() - start) * 1000.0
        return QueryResult(
            response=response.text,
            route="llm_call",
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=elapsed,
        )

    def close(self) -> None:
        self._llm.close()
