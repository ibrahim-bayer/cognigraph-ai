"""Safety boundary — gates graph execution against four risk axes.

The graph must never confidently serve a wrong answer. Before the
pipeline executes a graph route, the safety boundary inspects the
match and decides whether to allow it or escalate to the LLM:

  1. Risk gating       — HIGH-risk nodes always escalate.
  2. Volatility flag   — nodes whose answers change between calls
                         (e.g., "what time is it") always escalate.
  3. Ambiguity         — uses `match_result.ambiguous` (computed by
                         the matcher on combined score = sim × conf,
                         the same basis used to pick the winner).
                         The matcher is the single source of truth
                         for ambiguity; safety just consumes it.
  4. Pattern blocklist — input substrings that always go to the LLM
                         regardless of graph state (compliance-
                         sensitive topics, account questions, etc.).

The boundary is a pure decision function — it does not mutate state.
The pipeline is responsible for honoring `SafetyDecision.override_route`
when `safe=False`.

Spec note: issue #018 lists `check(match_result, node)` but `node` is
redundant with `match_result.node`. This impl drops the redundant
parameter and requires `input_text` so the blocklist gate has
something to match against. Future API extension that needs a
separate node (e.g., re-checking after a swap) can pass it via
match_result.
"""

from __future__ import annotations

import logging
import threading

from cognigraph.config import CogniGraphConfig
from cognigraph.models import (
    HabitNode,
    MatchResult,
    RiskLevel,
    RouteDecision,
    SafetyDecision,
)

logger = logging.getLogger(__name__)


class SafetyBoundary:
    """Inspect a MatchResult + input and decide whether the graph route is safe.

    Thread safety: `check()` is safe to call concurrently with
    `add_to_blocklist`/`remove_from_blocklist`. Internally the blocklist
    is guarded by a lock and `_check_blocklist` snapshots into a
    frozenset before iterating so set-mutation-during-iteration cannot
    raise.

    Blocklist persistence: `add_to_blocklist`/`remove_from_blocklist`
    mutate the in-memory working set ONLY. Patterns are NOT persisted
    across process restarts. The persistent source of truth is
    `CogniGraphConfig.blocklist_patterns`. Use the runtime API for
    hot-patches; persist via config for durable changes.
    """

    def __init__(self, config: CogniGraphConfig | None = None) -> None:
        self._config = config or CogniGraphConfig()
        # Mutable working set — initialized from config but extendable
        # via add_to_blocklist / remove_from_blocklist.
        self._blocklist: set[str] = {
            p.lower() for p in self._config.blocklist_patterns if p
        }
        # Guards mutations against concurrent reads from check().
        self._blocklist_lock = threading.Lock()
        # Observability counter — number of unsafe decisions emitted.
        self._block_counts: dict[str, int] = {}

    # --- Public API ---

    def check(
        self,
        match_result: MatchResult,
        input_text: str,
    ) -> SafetyDecision:
        """Decide whether `match_result.route_decision` is safe to execute.

        Args:
            match_result: the matcher's output for this turn. The
                boundary inspects `route_decision`, `node`,
                `ambiguous`, and `candidates`.
            input_text: the user's input (raw or normalized — the
                blocklist matches case-insensitively as a substring).
                **Required**: passing an empty string disables the
                blocklist for this call (there's nothing to match
                against), so callers should always pass the actual
                input. The required signature catches accidental
                omissions at type-check time.

        Order of checks:
          1. Blocklist — fires on any route. If matched, the override
             pulls graph routes down to LLM_FALLBACK and leaves LLM
             routes alone (they were already going to the LLM).
          2-4. Risk / volatility / ambiguity — fire only on graph
             routes; LLM routes don't trust the matched node.
        """
        # 1. Blocklist (route-agnostic)
        if self._check_blocklist(input_text):
            return self._unsafe(
                "blocklist_match", self._safe_route_for(match_result)
            )

        # If the matcher already deferred to the LLM, nothing to gate
        if not self._is_graph_route(match_result.route_decision):
            return SafetyDecision(safe=True)

        node = match_result.node
        # Defensive: the matcher should always populate node on graph
        # routes, but if it doesn't there's nothing to execute — defer
        # to LLM. Surface the matcher inconsistency so we can chase it.
        if node is None:
            logger.warning(
                "matcher returned %s with node=None; safety boundary "
                "overriding to LLM_ONLY",
                match_result.route_decision.value,
            )
            return self._unsafe(
                "graph_route_missing_node", RouteDecision.LLM_ONLY
            )

        # 2. Risk level
        if self._check_risk_level(node):
            return self._unsafe("high_risk_node", RouteDecision.LLM_FALLBACK)

        # 3. Volatile flag
        if node.volatile:
            return self._unsafe("volatile_node", RouteDecision.LLM_FALLBACK)

        # 4. Ambiguity — defer to the matcher's own flag (computed on
        # combined score = sim × confidence, the same basis used for
        # ranking). Single source of truth.
        if match_result.ambiguous:
            return self._unsafe(
                "ambiguous_match", RouteDecision.LLM_FALLBACK
            )

        return SafetyDecision(safe=True)

    def add_to_blocklist(self, pattern: str) -> None:
        """Add a pattern to the in-memory blocklist (case-insensitive).

        In-memory only — not persisted across restarts. See class docstring.
        """
        if pattern and pattern.strip():
            with self._blocklist_lock:
                self._blocklist.add(pattern.strip().lower())

    def remove_from_blocklist(self, pattern: str) -> None:
        """Remove a pattern. Silently no-ops if absent.

        In-memory only — not persisted across restarts.
        """
        if pattern:
            with self._blocklist_lock:
                self._blocklist.discard(pattern.strip().lower())

    @property
    def blocklist(self) -> frozenset[str]:
        """Read-only snapshot of the active blocklist."""
        with self._blocklist_lock:
            return frozenset(self._blocklist)

    @property
    def block_counts(self) -> dict[str, int]:
        """Counter of unsafe decisions by reason — for ops/audit."""
        return dict(self._block_counts)

    # --- Internals ---

    def _unsafe(
        self, reason: str, override: RouteDecision
    ) -> SafetyDecision:
        """Build an unsafe SafetyDecision and emit observability."""
        self._block_counts[reason] = self._block_counts.get(reason, 0) + 1
        logger.warning(
            "safety block: reason=%s override=%s (total %s blocks: %d)",
            reason,
            override.value,
            reason,
            self._block_counts[reason],
        )
        return SafetyDecision(
            safe=False, reason=reason, override_route=override
        )

    @staticmethod
    def _is_graph_route(route: RouteDecision) -> bool:
        return route in (RouteDecision.GRAPH_DIRECT, RouteDecision.GRAPH_COMPOSED)

    @staticmethod
    def _safe_route_for(match_result: MatchResult) -> RouteDecision:
        """Pick the override route when a check fails."""
        if match_result.node is None:
            return RouteDecision.LLM_ONLY
        return RouteDecision.LLM_FALLBACK

    @staticmethod
    def _check_risk_level(node: HabitNode) -> bool:
        """True if the node is gated by risk level (HIGH always escalates)."""
        return node.risk_level == RiskLevel.HIGH

    def _check_blocklist(self, input_text: str) -> bool:
        """True if any blocklist pattern is a case-insensitive substring.

        Snapshots the blocklist under the lock before iterating so
        concurrent add/remove cannot raise RuntimeError.
        """
        if not input_text:
            return False
        with self._blocklist_lock:
            patterns = tuple(self._blocklist)
        if not patterns:
            return False
        haystack = input_text.lower()
        return any(p in haystack for p in patterns)
