#!/usr/bin/env python3
"""Interactive smoke test — wires all completed components into a REPL.

Usage:
    ANTHROPIC_API_KEY=sk-... uv run python scratch/smoke.py

Type a question → it routes through:
  normalizer → embedder → FAISS matcher → graph store (if match) / LLM (if novel)
  → learns the response as a new node → saves to SQLite

Ask the same question again and watch it shift from LLM_ONLY → LLM_FALLBACK
→ GRAPH_DIRECT as confidence grows through reinforcement.

Type 'quit' or Ctrl-D to exit. Type 'stats' to see graph stats.
"""

from __future__ import annotations

import sys
import time

from cognigraph.config import CogniGraphConfig
from cognigraph.embedding import EmbeddingService
from cognigraph.graph_store import InMemoryGraphStore
from cognigraph.llm_client import ClaudeLLMProvider
from cognigraph.matcher import NodeMatcher
from cognigraph.models import HabitNode, InteractionLog, RouteDecision
from cognigraph.normalizer import InputNormalizer
from cognigraph.persistence import SQLitePersistence
from cognigraph.vector_index import FAISSIndex


DB_PATH = "scratch/smoke.db"
FAISS_PATH = "scratch/smoke.faiss"
SYSTEM_PROMPT = (
    "You are the cognitive fallback for a learned graph agent. "
    "Answer concisely in 1-2 sentences."
)


def main() -> None:
    cfg = CogniGraphConfig()
    normalizer = InputNormalizer(cfg)
    embedder = EmbeddingService(cfg)

    print("Loading model + state...")
    # Warm-load the embedding model
    embedder.embed("warmup")

    # Load persisted state (or start fresh)
    persist = SQLitePersistence(DB_PATH)
    store = persist.load_graph()

    # Rebuild FAISS from persisted graph (or load if saved previously)
    idx = FAISSIndex(dimension=cfg.embedding_dim)
    try:
        idx.load(FAISS_PATH)
        print(f"  Loaded FAISS index: {idx.count()} vectors")
    except Exception:
        # First run or corrupt index — rebuild from graph
        for node in store.all_nodes():
            if node.embedding_vector:
                idx.add(node.pattern_id, node.embedding_vector)
        print(f"  Rebuilt FAISS index from graph: {idx.count()} vectors")

    matcher = NodeMatcher(store, idx, cfg)

    # LLM — will raise if ANTHROPIC_API_KEY is not set
    try:
        llm = ClaudeLLMProvider(config=cfg)
    except Exception as e:
        print(f"\nWARNING: LLM not available ({e})")
        print("Fallback queries will show an error. Set ANTHROPIC_API_KEY to enable.\n")
        llm = None

    print(f"\nGraph: {store.node_count()} nodes | FAISS: {idx.count()} vectors")
    print("Type a question, 'stats', or 'quit'.\n")

    while True:
        try:
            raw = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not raw:
            continue
        if raw.lower() == "quit":
            break
        if raw.lower() == "stats":
            _print_stats(store, idx, matcher, persist)
            continue

        turn_start = time.time()
        norm = normalizer.normalize(raw)
        qv = embedder.embed(norm.normalized)
        match = matcher.match(qv)

        route = match.route_decision
        route_label = route.value
        sim_label = f"sim={match.similarity:.3f}"
        conf_label = (
            f"conf={match.node.confidence:.2f}" if match.node else "conf=n/a"
        )
        ambig_label = " AMBIGUOUS" if match.ambiguous else ""

        print(f"  [{route_label} {sim_label} {conf_label}{ambig_label}]")

        if route in (RouteDecision.GRAPH_DIRECT, RouteDecision.GRAPH_COMPOSED):
            # Graph handles it
            print(f"  >> {match.node.response}")
            # Reinforce
            match.node.reinforcement_count += 1
            match.node.confidence = min(
                1.0, match.node.confidence + cfg.confidence_boost
            )
            match.node.last_used_at = time.time()
            store.put_node(match.node)
            idx.add(match.node.pattern_id, match.node.embedding_vector)

            persist.log_interaction(
                InteractionLog(
                    timestamp=turn_start,
                    input_text=raw,
                    normalized_text=norm.normalized,
                    route_decision=route,
                    matched_node_id=match.node.pattern_id,
                    response_text=match.node.response,
                    latency_ms=(time.time() - turn_start) * 1000,
                )
            )

        elif route in (RouteDecision.LLM_FALLBACK, RouteDecision.LLM_ONLY):
            # LLM handles it
            if llm is None:
                print("  >> [LLM not available — set ANTHROPIC_API_KEY]")
                continue

            try:
                resp = llm.generate(
                    prompt=norm.normalized,
                    system=SYSTEM_PROMPT,
                )
                print(f"  >> {resp.text}  ({resp.latency_ms:.0f}ms)")
            except Exception as e:
                print(f"  >> LLM error: {e}")
                continue

            # Learn: create or reinforce a node
            if match.node is not None:
                # LLM_FALLBACK — node exists but wasn't confident enough
                match.node.reinforcement_count += 1
                match.node.confidence = min(
                    1.0, match.node.confidence + cfg.confidence_boost
                )
                match.node.last_used_at = time.time()
                # Update response if LLM gave a better one
                match.node.response = resp.text
                store.put_node(match.node)
                idx.add(match.node.pattern_id, match.node.embedding_vector)
                node_id = match.node.pattern_id
            else:
                # LLM_ONLY — genuinely novel, create a new node
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
                print(f"  [learned as {pid}]")

            persist.log_interaction(
                InteractionLog(
                    timestamp=turn_start,
                    input_text=raw,
                    normalized_text=norm.normalized,
                    route_decision=route,
                    matched_node_id=node_id,
                    llm_response=resp.text,
                    response_text=resp.text,
                    latency_ms=resp.latency_ms,
                )
            )

        # Save after every turn so state persists across restarts
        persist.save_graph(store)
        idx.save(FAISS_PATH)

    # Shutdown
    print("\nSaving state...")
    persist.save_graph(store)
    idx.save(FAISS_PATH)
    idx.close()
    if llm is not None:
        llm.close()
    persist.close()
    print(f"Done. {store.node_count()} nodes saved to {DB_PATH}")


def _print_stats(
    store: InMemoryGraphStore,
    idx: FAISSIndex,
    matcher: NodeMatcher,
    persist: SQLitePersistence,
) -> None:
    """Print a compact summary of the current state."""
    print(f"\n  Nodes in graph:  {store.node_count()}")
    print(f"  Vectors in FAISS: {idx.count()}")
    print(f"  Stale FAISS hits: {matcher.stale_hit_count}")
    logs = persist.get_interactions(limit=1000)
    print(f"  Interaction logs: {len(logs)}")
    if logs:
        routes = {}
        for log in logs:
            r = log.route_decision.value
            routes[r] = routes.get(r, 0) + 1
        for r, c in sorted(routes.items()):
            print(f"    {r}: {c}")
    # Show all nodes sorted by confidence
    nodes = sorted(store.all_nodes(), key=lambda n: n.confidence, reverse=True)
    if nodes:
        print(f"\n  Top nodes:")
        for n in nodes[:10]:
            triggers = n.trigger_patterns[0] if n.trigger_patterns else "(no trigger)"
            print(
                f"    [{n.confidence:.2f} x{n.reinforcement_count}] "
                f"{n.pattern_id}: {triggers[:50]}"
            )
    print()


if __name__ == "__main__":
    main()
