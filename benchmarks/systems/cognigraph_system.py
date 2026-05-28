"""CogniGraphSystem — wraps the full CogniGraph pipeline for benchmarking.

Wraps a real `ClaudeLLMProvider` in a token-recording proxy so the
benchmark can attribute LLM cost per query (the pipeline's
`PipelineResult` doesn't surface token counts today; this is the
cleanest way to capture them without changing production code).
"""

from __future__ import annotations

import dataclasses
import logging
import tempfile
import time
from pathlib import Path

from cognigraph.config import CogniGraphConfig
from cognigraph.lifecycle import ApplicationLifecycle
from cognigraph.llm_client import ClaudeLLMProvider
from cognigraph.models import RouteDecision
from cognigraph.types import LLMResponse

from benchmarks.systems import QueryResult, SystemUnderTest

logger = logging.getLogger(__name__)


class _TokenRecordingLLM:
    """Wraps a real LLMProvider and remembers the last call's token counts.

    Implements the LLMProvider protocol (generate, close) so the pipeline
    accepts it. After each pipeline.process() turn the benchmark reads
    `last_input_tokens` / `last_output_tokens` / `was_called` to attribute
    tokens to that query.
    """

    def __init__(self, inner: ClaudeLLMProvider) -> None:
        self._inner = inner
        self.last_input_tokens: int = 0
        self.last_output_tokens: int = 0
        self.was_called: bool = False

    def reset(self) -> None:
        self.last_input_tokens = 0
        self.last_output_tokens = 0
        self.was_called = False

    def generate(
        self,
        prompt: str,
        context: list[dict] | None = None,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        response = self._inner.generate(
            prompt=prompt, context=context, system=system, max_tokens=max_tokens
        )
        self.last_input_tokens = response.input_tokens
        self.last_output_tokens = response.output_tokens
        self.was_called = True
        return response

    def close(self) -> None:
        self._inner.close()


class CogniGraphSystem:
    """Full CogniGraph pipeline with real components, fresh state per run."""

    name = "cognigraph"

    def __init__(self, config: CogniGraphConfig | None = None) -> None:
        self._tmpdir = Path(tempfile.mkdtemp(prefix="cg-bench-"))
        base_config = config or CogniGraphConfig()
        # Replace persistence paths so each benchmark run starts clean
        # and doesn't collide with a real cognigraph.db on disk.
        self._config = dataclasses.replace(
            base_config,
            db_path=str(self._tmpdir / "bench.db"),
            faiss_index_path=str(self._tmpdir / "bench.faiss"),
        )

        # Build the LLM ourselves so we can wrap it for token tracking,
        # then inject into the lifecycle.
        self._llm_inner = ClaudeLLMProvider(config=self._config)
        self._llm = _TokenRecordingLLM(self._llm_inner)

        self._lifecycle = ApplicationLifecycle(
            config=self._config,
            llm=self._llm,
            install_signal_handlers=False,
        )
        self._pipeline = self._lifecycle.startup()

    def warmup(self, query: str) -> None:
        try:
            self._pipeline.process(query)
        except Exception as e:
            logger.warning("warmup query failed: %s", e)

    def query(self, q: str) -> QueryResult:
        self._llm.reset()
        start = time.perf_counter()
        try:
            result = self._pipeline.process(q)
        except Exception as e:
            elapsed = (time.perf_counter() - start) * 1000.0
            logger.warning("pipeline.process failed: %s", e)
            return QueryResult(
                response=f"[error: {e}]",
                route="error",
                input_tokens=0, output_tokens=0,
                latency_ms=elapsed,
            )
        elapsed = (time.perf_counter() - start) * 1000.0

        is_graph_route = result.route in (
            RouteDecision.GRAPH_DIRECT, RouteDecision.GRAPH_COMPOSED
        )
        route_label = "graph_hit" if is_graph_route else "llm_call"

        # If the pipeline routed through the LLM, the wrapper saw it.
        # If a graph route fired, the LLM was never called → zero tokens.
        return QueryResult(
            response=result.response,
            route=route_label,
            input_tokens=self._llm.last_input_tokens if self._llm.was_called else 0,
            output_tokens=self._llm.last_output_tokens if self._llm.was_called else 0,
            latency_ms=elapsed,
        )

    def close(self) -> None:
        try:
            self._lifecycle.shutdown()
        except Exception:
            logger.exception("CogniGraph lifecycle shutdown failed")

    def stats(self) -> dict:
        """Pipeline stats snapshot — useful for the report."""
        return self._pipeline.get_stats()
