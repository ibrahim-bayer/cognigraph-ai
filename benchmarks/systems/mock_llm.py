"""MockLLMSystem — a deterministic fake for harness validation.

NOT for measuring real latency or cost. Used only to verify the
benchmark wiring runs end-to-end before spending API tokens.

Returns a canonical response per intent based on input substring
matching. Simulates a short latency (5-15ms) so the harness emits
non-zero numbers we can sanity-check.
"""

from __future__ import annotations

import random
import time

from benchmarks.systems import QueryResult


class MockLLMSystem:
    """Fake LLM — deterministic, no API calls."""

    name = "mock_llm"

    # Substring → canonical response, mirroring the synthetic dataset.
    _RESPONSES = [
        ("password", "Click 'Forgot password' on the login screen."),
        ("refund", "Refunds process in 5-7 business days."),
        ("track", "Track your order at /orders/<id>/tracking."),
        ("cancel", "Cancel from /orders/<id> within 1 hour of placement."),
        ("address", "Edit shipping address from /account/addresses."),
        ("payment", "Try a different payment method."),
        ("support", "Reach support at /help or live chat."),
        ("delete", "Submit account-deletion request from /account/settings/delete."),
    ]
    _DEFAULT = "Can you provide more details about your question?"

    def __init__(self, seed: int = 0) -> None:
        self._rng = random.Random(seed)

    def warmup(self, query: str) -> None:
        # No-op: stateless mock
        pass

    def query(self, q: str) -> QueryResult:
        start = time.perf_counter()
        # Simulate 5-15ms network-ish latency so the harness has
        # non-trivial numbers to plot
        time.sleep(self._rng.uniform(0.005, 0.015))
        elapsed = (time.perf_counter() - start) * 1000.0

        lower = q.lower()
        response = next(
            (resp for needle, resp in self._RESPONSES if needle in lower),
            self._DEFAULT,
        )

        # Fake token counts: ~50 input + ~30 output
        return QueryResult(
            response=response,
            route="llm_call",
            input_tokens=self._rng.randint(40, 60),
            output_tokens=self._rng.randint(20, 40),
            latency_ms=elapsed,
        )

    def close(self) -> None:
        pass
