#!/usr/bin/env python3
"""Non-interactive smoke runner with a mocked anthropic backend.

Runs a scripted sequence of inputs through the full pipeline (no API key
needed), dumps every learned node, then re-runs a subset of the inputs
to demonstrate the cold→confident routing shift that the e2e tests
assert. Use this to eyeball the system without burning API tokens.

Usage:
    uv run python scratch/run_smoke.py
"""

from __future__ import annotations

import time
from pathlib import Path

from cognigraph.config import CogniGraphConfig
from cognigraph.embedding import EmbeddingService
from cognigraph.graph_store import InMemoryGraphStore
from cognigraph.llm_client import ClaudeLLMProvider
from cognigraph.matcher import NodeMatcher
from cognigraph.models import HabitNode, InteractionLog, RouteDecision
from cognigraph.normalizer import InputNormalizer
from cognigraph.persistence import SQLitePersistence
from cognigraph.vector_index import FAISSIndex


DB_PATH = "scratch/run_smoke.db"
FAISS_PATH = "scratch/run_smoke.faiss"
SYSTEM_PROMPT = "You are a concise fallback agent. Answer in one sentence."

# Response-divergence threshold for the create-vs-reinforce decision on
# LLM_FALLBACK turns (issue #22). When the LLM's new answer is below this
# cosine similarity to the matched node's stored response, treat the input
# as a novel intent and create a new node rather than reinforcing.
#
# Empirically (E5-Small + short English answers), identical-intent
# responses score ~1.0 while different-topic conversational replies
# cluster at 0.74-0.83. 0.85 cleanly separates them. The real
# FlatNodeLearner (#015) will make this configurable.
RESPONSE_DIVERGENCE_THRESHOLD = 0.85


# --- Fake anthropic backend (mirrors the test_e2e fake) ---


class _FakeUsage:
    def __init__(self, input_tokens: int = 12, output_tokens: int = 24) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, text: str, model: str = "claude-fake") -> None:
        self.content = [_FakeTextBlock(text)]
        self.model = model
        self.usage = _FakeUsage()


class _FakeMessages:
    """Lookup-table backend: matches incoming user prompt against known keys."""

    ANSWERS = {
        "name": "Ibrahim",
        "weather": "Sunny, 72°F.",
        "time": "It's 3:15 PM.",
        "date": "Today is 2026-04-14.",
        "joke": "Why don't scientists trust atoms? They make up everything.",
        "commit": "Run: git add -A && git commit -m \"<msg>\".",
        "deploy": "Push to main; CI handles the rest.",
        "two plus two": "Four.",
        "capital of france": "Paris.",
    }
    DEFAULT = "Let me think about that — I'm not sure."

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        last_user = next(
            (
                m["content"]
                for m in reversed(kwargs.get("messages", []))
                if m.get("role") == "user"
            ),
            "",
        ).lower()
        for key, ans in self.ANSWERS.items():
            if key in last_user:
                return _FakeResponse(ans)
        return _FakeResponse(self.DEFAULT)


class _FakeClient:
    def __init__(self) -> None:
        self.messages = _FakeMessages()


# --- Pipeline orchestrator ---


def _run_turn(
    raw: str,
    *,
    normalizer: InputNormalizer,
    embedder: EmbeddingService,
    matcher: NodeMatcher,
    store: InMemoryGraphStore,
    idx: FAISSIndex,
    llm: ClaudeLLMProvider,
    persist: SQLitePersistence,
    cfg: CogniGraphConfig,
) -> dict:
    """One full turn through the pipeline. Returns a summary dict."""
    turn_start = time.time()
    norm = normalizer.normalize(raw)
    qv = embedder.embed(norm.normalized)
    match = matcher.match(qv)

    summary = {
        "input": raw,
        "route": match.route_decision.value,
        "similarity": round(match.similarity, 3),
        "ambiguous": match.ambiguous,
        "matched_id": match.node.pattern_id if match.node else None,
        "matched_conf": (
            round(match.node.confidence, 3) if match.node else None
        ),
    }

    if match.route_decision in (
        RouteDecision.GRAPH_DIRECT,
        RouteDecision.GRAPH_COMPOSED,
    ):
        # Graph handles + reinforces
        match.node.reinforcement_count += 1
        match.node.confidence = min(
            1.0, match.node.confidence + cfg.confidence_boost
        )
        match.node.last_used_at = time.time()
        store.put_node(match.node)
        idx.add(match.node.pattern_id, match.node.embedding_vector)
        summary["response"] = match.node.response
        summary["llm_called"] = False

        persist.log_interaction(
            InteractionLog(
                timestamp=turn_start,
                input_text=raw,
                normalized_text=norm.normalized,
                route_decision=match.route_decision,
                matched_node_id=match.node.pattern_id,
                response_text=match.node.response,
                latency_ms=(time.time() - turn_start) * 1000,
            )
        )
    else:
        # LLM fallback
        resp = llm.generate(prompt=norm.normalized, system=SYSTEM_PROMPT)
        summary["response"] = resp.text
        summary["llm_called"] = True
        summary["llm_latency_ms"] = round(resp.latency_ms, 2)
        summary["llm_tokens"] = (resp.input_tokens, resp.output_tokens)

        if match.node is not None:
            # LLM_FALLBACK — reinforce OR branch based on response divergence.
            # Compare the new LLM answer against the matched node's stored
            # response. If they're semantically far apart, the input is a
            # novel intent that happens to embed near an existing node —
            # create a new node instead of overwriting the old response.
            existing_resp_vec = embedder.embed(match.node.response)
            new_resp_vec = embedder.embed(resp.text)
            response_sim = float(
                sum(a * b for a, b in zip(existing_resp_vec, new_resp_vec))
            )
            summary["response_sim"] = round(response_sim, 3)

            if response_sim >= RESPONSE_DIVERGENCE_THRESHOLD:
                # Same intent, refine the existing node. Do NOT overwrite
                # the stored response — the original answer is the anchor
                # for future divergence checks and overwriting it lets
                # h0 silently drift to whatever the latest LLM said.
                match.node.reinforcement_count += 1
                match.node.confidence = min(
                    1.0, match.node.confidence + cfg.confidence_boost
                )
                match.node.last_used_at = time.time()
                store.put_node(match.node)
                idx.add(match.node.pattern_id, match.node.embedding_vector)
                node_id = match.node.pattern_id
                summary["learning_action"] = "reinforced"
            else:
                # Divergent answer — create a new node for this intent
                pid = f"h{store.node_count()}"
                node = HabitNode(
                    pattern_id=pid,
                    trigger_patterns=[norm.normalized],
                    embedding_vector=qv,
                    confidence=cfg.learning_starting_confidence,
                    response=resp.text,
                    last_used_at=time.time(),
                )
                store.put_node(node)
                idx.add(pid, qv)
                node_id = pid
                summary["learning_action"] = "branched"
                summary["learned_as"] = pid
        else:
            # LLM_ONLY — create a fresh node
            pid = f"h{store.node_count()}"
            node = HabitNode(
                pattern_id=pid,
                trigger_patterns=[norm.normalized],
                embedding_vector=qv,
                confidence=cfg.learning_starting_confidence,
                response=resp.text,
                last_used_at=time.time(),
            )
            store.put_node(node)
            idx.add(pid, qv)
            node_id = pid
            summary["learned_as"] = pid
            summary["learning_action"] = "created"

        persist.log_interaction(
            InteractionLog(
                timestamp=turn_start,
                input_text=raw,
                normalized_text=norm.normalized,
                route_decision=match.route_decision,
                matched_node_id=node_id,
                llm_response=resp.text,
                response_text=resp.text,
                latency_ms=resp.latency_ms,
            )
        )

    persist.save_graph(store)
    idx.save(FAISS_PATH)
    return summary


def _print_turn(i: int, summary: dict) -> None:
    flag = "LLM " if summary["llm_called"] else "GRAPH"
    ambig = " [AMBIGUOUS]" if summary["ambiguous"] else ""
    matched = (
        f" matched={summary['matched_id']}({summary['matched_conf']})"
        if summary["matched_id"]
        else ""
    )
    response_sim = (
        f" rsim={summary['response_sim']}"
        if "response_sim" in summary
        else ""
    )
    action = (
        f" action={summary['learning_action']}"
        if "learning_action" in summary
        else ""
    )
    learned = (
        f" learned={summary['learned_as']}"
        if "learned_as" in summary
        else ""
    )
    print(
        f"  [{i:>2}] {flag} route={summary['route']:<14} "
        f"sim={summary['similarity']}{matched}{response_sim}{action}{learned}{ambig}"
    )
    print(f"        Q: {summary['input']}")
    print(f"        A: {summary['response']}")


def _dump_nodes(store: InMemoryGraphStore) -> None:
    print("\n" + "=" * 78)
    print(f"GRAPH DUMP — {store.node_count()} nodes")
    print("=" * 78)
    if not store.all_nodes():
        print("  (empty)")
        return
    nodes = sorted(
        store.all_nodes(),
        key=lambda n: (n.confidence, n.reinforcement_count),
        reverse=True,
    )
    for n in nodes:
        trigger = n.trigger_patterns[0] if n.trigger_patterns else "(none)"
        print(
            f"  {n.pattern_id:<6} conf={n.confidence:.3f}  "
            f"reinf={n.reinforcement_count:>2}  "
            f"stab={n.stability.value:<6}"
        )
        print(f"         trigger:  {trigger[:65]}")
        print(f"         response: {n.response[:65]}")


def _dump_logs(persist: SQLitePersistence) -> None:
    print("\n" + "=" * 78)
    logs = persist.get_interactions(limit=100)
    print(f"INTERACTION LOG — {len(logs)} entries (most recent first)")
    print("=" * 78)
    routes: dict[str, int] = {}
    for log in logs:
        r = log.route_decision.value
        routes[r] = routes.get(r, 0) + 1
    print(f"  Route distribution: {routes}")


def _validate_intent_separation(
    matcher: NodeMatcher,
    embedder: EmbeddingService,
    normalizer: InputNormalizer,
    expected_distinct_nodes: int,
    store: InMemoryGraphStore,
) -> list[tuple[bool, str]]:
    """Validate distinct-intent → distinct-node mapping.

    Run this BEFORE Phase 3 (heavy reinforcement of one node), otherwise
    `sim × conf` ranking will collapse every intent onto the dominant node.
    """
    checks: list[tuple[bool, str]] = []

    # Issue #22: distinct intents must produce distinct nodes when LLM
    # responses diverge.
    checks.append(
        (
            store.node_count() >= expected_distinct_nodes,
            f"#22: distinct intents produced distinct nodes "
            f"(have {store.node_count()}, expected >= {expected_distinct_nodes})",
        )
    )

    # Each Phase 1 intent has its own node and routes to that node.
    intent_to_canonical = {
        "name": "what is my name",
        "weather": "what's the weather",
        "time": "what time is it",
        "joke": "tell me a joke",
        "commit": "how do I commit my changes",
        "math": "what is two plus two",
    }
    seen_node_ids: set[str] = set()
    for intent, canonical in intent_to_canonical.items():
        m = matcher.match(
            embedder.embed(normalizer.normalize(canonical).normalized)
        )
        if m.node is None:
            checks.append((False, f"intent {intent!r} produced no match"))
            continue
        seen_node_ids.add(m.node.pattern_id)
    checks.append(
        (
            len(seen_node_ids) == len(intent_to_canonical),
            f"each Phase 1 intent maps to its own node "
            f"(saw {len(seen_node_ids)} distinct, expected "
            f"{len(intent_to_canonical)})",
        )
    )

    return checks


def _validate_post_phase3(
    matcher: NodeMatcher,
    store: InMemoryGraphStore,
    embedder: EmbeddingService,
    normalizer: InputNormalizer,
) -> list[tuple[bool, str]]:
    """Validations that depend on Phase 3 having run."""
    checks: list[tuple[bool, str]] = []

    # After enough reinforcement, route shifts to GRAPH_DIRECT.
    confident_nodes = [n for n in store.all_nodes() if n.confidence >= 0.7]
    checks.append(
        (
            len(confident_nodes) > 0,
            f"at least one node reached GRAPH_DIRECT confidence "
            f"(found {len(confident_nodes)})",
        )
    )

    # Stale-hit counter is zero (no FAISS/graph drift in this run).
    checks.append(
        (
            matcher.stale_hit_count == 0,
            f"no FAISS/graph drift (stale_hits={matcher.stale_hit_count})",
        )
    )

    # Out-of-distribution probe does NOT route GRAPH_DIRECT.
    odd_probe = "describe the political history of byzantium"
    odd_match = matcher.match(
        embedder.embed(normalizer.normalize(odd_probe).normalized)
    )
    checks.append(
        (
            odd_match.route_decision != RouteDecision.GRAPH_DIRECT,
            f"OOD probe {odd_probe!r} did NOT route GRAPH_DIRECT "
            f"(got {odd_match.route_decision.value})",
        )
    )

    return checks


def _print_validation(label: str, checks: list[tuple[bool, str]]) -> None:
    print("\n" + "=" * 78)
    print(label)
    print("=" * 78)
    for ok, line in checks:
        marker = "PASS" if ok else "FAIL"
        print(f"  [{marker}] {line}")
    fails = [c for c in checks if not c[0]]
    if fails:
        print(f"\n  {len(fails)}/{len(checks)} validation(s) failed.")
    else:
        print(f"\n  All {len(checks)} validations passed.")


def main() -> None:
    # Reset state from previous runs so output is reproducible
    for p in (DB_PATH, FAISS_PATH, FAISS_PATH + ".ids.json"):
        Path(p).unlink(missing_ok=True)

    print("Building pipeline...")
    cfg = CogniGraphConfig()
    normalizer = InputNormalizer(cfg)
    embedder = EmbeddingService(cfg)
    embedder.embed("warmup")  # eager-load the model

    persist = SQLitePersistence(DB_PATH)
    store = persist.load_graph()
    idx = FAISSIndex(dimension=cfg.embedding_dim)
    matcher = NodeMatcher(store, idx, cfg)

    fake_client = _FakeClient()
    llm = ClaudeLLMProvider(
        api_key="fake-key", model="claude-fake", config=cfg, client=fake_client
    )

    # --- Phase 1: cold start, every input is novel ---
    print("\n" + "=" * 78)
    print("PHASE 1: cold start (graph is empty)")
    print("=" * 78)
    novel_inputs = [
        "what's my name?",
        "what is the weather today?",
        "what time is it?",
        "tell me a joke",
        "how do I commit my changes?",
        "what is two plus two",
    ]
    for i, raw in enumerate(novel_inputs, 1):
        summary = _run_turn(
            raw,
            normalizer=normalizer,
            embedder=embedder,
            matcher=matcher,
            store=store,
            idx=idx,
            llm=llm,
            persist=persist,
            cfg=cfg,
        )
        _print_turn(i, summary)

    # --- Phase 2: reinforce — same intents, reworded ---
    print("\n" + "=" * 78)
    print("PHASE 2: reword the same intents — should match existing nodes")
    print("=" * 78)
    reworded = [
        "tell me my name",
        "how's the weather outside",
        "what's the current time",
        "make me laugh",
        "save my code edits",
        "compute 2 + 2",
    ]
    for i, raw in enumerate(reworded, 1):
        summary = _run_turn(
            raw,
            normalizer=normalizer,
            embedder=embedder,
            matcher=matcher,
            store=store,
            idx=idx,
            llm=llm,
            persist=persist,
            cfg=cfg,
        )
        _print_turn(i, summary)

    # --- Validate distinct-intent separation BEFORE heavy reinforcement ---
    # `sim × conf` ranking means a node hammered to conf=1.0 will dominate
    # cross-intent matches, so this check must run before Phase 3.
    intent_checks = _validate_intent_separation(
        matcher, embedder, normalizer, len(novel_inputs), store
    )
    _print_validation(
        "VALIDATION (post-Phase 2): intent separation", intent_checks
    )

    # --- Phase 3: pound on one habit until it crosses GRAPH_DIRECT ---
    print("\n" + "=" * 78)
    print("PHASE 3: hammer 'name' query 25× to drive it to GRAPH_DIRECT")
    print("=" * 78)
    last_summary = None
    transitions = []
    for i in range(1, 26):
        summary = _run_turn(
            "what is my name?",
            normalizer=normalizer,
            embedder=embedder,
            matcher=matcher,
            store=store,
            idx=idx,
            llm=llm,
            persist=persist,
            cfg=cfg,
        )
        if last_summary is None or last_summary["route"] != summary["route"]:
            transitions.append((i, summary["route"], summary["matched_conf"]))
        last_summary = summary
    print(f"  Route transitions during 25 repetitions:")
    for turn_n, route, conf in transitions:
        print(f"    turn {turn_n:>2}: {route:<14} (conf={conf})")
    print(f"  Final state: route={last_summary['route']}, "
          f"conf={last_summary['matched_conf']}, "
          f"llm_called={last_summary['llm_called']}")

    # --- Dumps + post-Phase 3 validation ---
    _dump_nodes(store)
    _dump_logs(persist)
    post_checks = _validate_post_phase3(matcher, store, embedder, normalizer)
    _print_validation("VALIDATION (post-Phase 3): confidence + safety", post_checks)

    # --- Durability check: close, reopen, and prove the graph survives ---
    print("\n" + "=" * 78)
    print("DURABILITY: closing all stores, reopening, replaying probes")
    print("=" * 78)
    persist.close()
    idx.close()
    llm.close()

    persist2 = SQLitePersistence(DB_PATH)
    store2 = persist2.load_graph()
    idx2 = FAISSIndex(dimension=cfg.embedding_dim)
    idx2.load(FAISS_PATH)
    matcher2 = NodeMatcher(store2, idx2, cfg)

    print(f"  Reloaded: {store2.node_count()} nodes, {idx2.count()} vectors")

    name_match = matcher2.match(
        embedder.embed(normalizer.normalize("what's my name").normalized)
    )
    print(
        f"  After reopen, 'what's my name' routes "
        f"{name_match.route_decision.value} (sim={name_match.similarity:.3f})"
    )
    if name_match.node:
        print(
            f"    matched node {name_match.node.pattern_id} "
            f"conf={name_match.node.confidence:.3f}"
        )

    persist2.close()
    idx2.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
