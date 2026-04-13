"""End-to-end integration tests.

Wires together every component that currently exists —
config, normalizer, embedder, graph store, persistence —
and exercises realistic multi-session user journeys against real
SQLite files and the real sentence-transformers embedding model.

These tests do NOT mock anything. They validate that the pieces built
so far actually compose into a working system, and that state survives
restart cycles the way the design requires.

The real embedding model is loaded once per module (session-scoped
fixture) to amortize the startup cost; individual tests then share it.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest

from cognigraph.config import CogniGraphConfig
from cognigraph.embedding import EmbeddingService
from cognigraph.exceptions import NodeNotFoundError
from cognigraph.graph_store import InMemoryGraphStore
from cognigraph.models import (
    ChildLink,
    HabitNode,
    InteractionLog,
    ResponseForm,
    RiskLevel,
    RouteDecision,
    Stability,
)
from cognigraph.normalizer import InputNormalizer
from cognigraph.persistence import SQLitePersistence


# --- Session-scoped heavy fixtures ---


@pytest.fixture(scope="module")
def config() -> CogniGraphConfig:
    return CogniGraphConfig()


@pytest.fixture(scope="module")
def embedder(config: CogniGraphConfig) -> EmbeddingService:
    """Real sentence-transformers model — loaded once per module."""
    svc = EmbeddingService(config)
    # Warm-load so the first actual test doesn't pay the download cost
    svc.embed("warmup")
    return svc


@pytest.fixture
def normalizer(config: CogniGraphConfig) -> InputNormalizer:
    return InputNormalizer(config)


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "e2e.db")


# --- Small helpers ---


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two L2-normalized vectors == dot product."""
    return float(np.dot(np.array(a), np.array(b)))


def _make_learned_node(
    pattern_id: str,
    trigger_text: str,
    response: str,
    embedder: EmbeddingService,
    confidence: float = 0.6,
    stability: Stability = Stability.LOW,
) -> HabitNode:
    """Build a node as the learning loop would: real embedding of a real trigger."""
    vec = embedder.embed(trigger_text)
    return HabitNode(
        pattern_id=pattern_id,
        trigger_patterns=[trigger_text],
        embedding_vector=vec,
        confidence=confidence,
        reinforcement_count=1,
        last_used_at=time.time(),
        stability=stability,
        risk_level=RiskLevel.LOW,
        response_form=ResponseForm.FIXED,
        response=response,
    )


# --- E2E Scenario 1: single-session learn + recall ---


class TestSingleSessionLearnAndRecall:
    """New user, empty graph: learn a habit within one session and recall it."""

    def test_empty_graph_to_first_habit(
        self,
        db_path: str,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
    ) -> None:
        # --- Start a cold session ---
        persistence = SQLitePersistence(db_path)
        store = persistence.load_graph()
        assert store.node_count() == 0  # baby mode

        # --- Turn 1: novel input, no match ---
        turn_start = time.time()
        raw = "What's my name?"
        norm = normalizer.normalize(raw)
        assert norm.normalized == "what's my name?"

        query_vec = embedder.embed(norm.normalized)
        assert len(query_vec) == 384

        # Nothing in graph yet → would fall through to LLM.
        # Simulate LLM producing a stable answer and create a habit node.
        simulated_llm_response = "Ibrahim"
        node = _make_learned_node(
            pattern_id="habit-name",
            trigger_text=norm.normalized,
            response=simulated_llm_response,
            embedder=embedder,
        )
        store.put_node(node)

        persistence.log_interaction(
            InteractionLog(
                timestamp=turn_start,
                input_text=raw,
                normalized_text=norm.normalized,
                route_decision=RouteDecision.LLM_ONLY,
                matched_node_id=None,
                llm_response=simulated_llm_response,
                response_text=simulated_llm_response,
                latency_ms=(time.time() - turn_start) * 1000,
            )
        )

        # --- Turn 2: same intent, different wording → graph should match ---
        raw2 = "what is my NAME??"
        norm2 = normalizer.normalize(raw2)
        q2 = embedder.embed(norm2.normalized)

        # Score against stored node by cosine similarity
        best: tuple[float, HabitNode] | None = None
        for n in store.all_nodes():
            score = _cosine(q2, n.embedding_vector)
            if best is None or score > best[0]:
                best = (score, n)

        assert best is not None
        score, matched = best
        # Same intent, same canonical form — should be very close to 1.0
        assert score > 0.95
        assert matched.pattern_id == "habit-name"
        assert matched.response == "Ibrahim"

        # Reinforce: bump count, confidence, and recency
        matched.reinforcement_count += 1
        matched.confidence = min(1.0, matched.confidence + 0.1)
        matched.last_used_at = time.time()
        store.put_node(matched)

        persistence.log_interaction(
            InteractionLog(
                timestamp=time.time(),
                input_text=raw2,
                normalized_text=norm2.normalized,
                route_decision=RouteDecision.GRAPH_DIRECT,
                matched_node_id=matched.pattern_id,
                response_text=matched.response,
                latency_ms=1.0,
            )
        )

        # --- Save session ---
        persistence.save_graph(store)
        persistence.close()

        # --- Reopen as a fresh process would ---
        p2 = SQLitePersistence(db_path)
        reloaded = p2.load_graph()
        assert reloaded.node_count() == 1
        restored = reloaded.get_node("habit-name")
        assert restored.response == "Ibrahim"
        assert restored.reinforcement_count == 2
        assert restored.confidence == pytest.approx(0.7, abs=1e-6)
        # Embedding survived JSON round-trip
        assert len(restored.embedding_vector) == 384

        logs = p2.get_interactions()
        assert len(logs) == 2
        # Stored in DESC timestamp order — second interaction is first
        assert logs[0].route_decision == RouteDecision.GRAPH_DIRECT
        assert logs[1].route_decision == RouteDecision.LLM_ONLY
        p2.close()


# --- E2E Scenario 2: multi-session durability ---


class TestMultiSessionDurability:
    """Simulate the user coming back after a 'restart' across many sessions."""

    def test_three_sessions_build_history(
        self,
        db_path: str,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
    ) -> None:
        inputs_session_1 = ["what time is it?", "what's the date?"]
        inputs_session_2 = ["time please", "set a timer"]
        inputs_session_3 = ["cancel the timer"]

        for session_idx, batch in enumerate(
            [inputs_session_1, inputs_session_2, inputs_session_3], start=1
        ):
            p = SQLitePersistence(db_path)
            store = p.load_graph()
            pre_count = store.node_count()

            for raw in batch:
                norm = normalizer.normalize(raw)
                vec = embedder.embed(norm.normalized)

                # Semantic de-dup: if a very-similar node already exists,
                # reinforce it; otherwise create a new one.
                matched: HabitNode | None = None
                best_score = 0.0
                for n in store.all_nodes():
                    score = _cosine(vec, n.embedding_vector)
                    if score > best_score:
                        best_score = score
                        matched = n

                if matched is not None and best_score > 0.95:
                    matched.reinforcement_count += 1
                    matched.confidence = min(1.0, matched.confidence + 0.05)
                    matched.last_used_at = time.time()
                    store.put_node(matched)
                    route = RouteDecision.GRAPH_DIRECT
                    node_id = matched.pattern_id
                    response = matched.response
                else:
                    new_id = f"s{session_idx}-{len(store.all_nodes())}"
                    node = _make_learned_node(
                        pattern_id=new_id,
                        trigger_text=norm.normalized,
                        response=f"(answer for {norm.normalized})",
                        embedder=embedder,
                    )
                    store.put_node(node)
                    route = RouteDecision.LLM_ONLY
                    node_id = new_id
                    response = node.response

                p.log_interaction(
                    InteractionLog(
                        timestamp=time.time(),
                        input_text=raw,
                        normalized_text=norm.normalized,
                        route_decision=route,
                        matched_node_id=node_id,
                        response_text=response,
                        latency_ms=1.0,
                    )
                )

            # At least one new node per session (the inputs are semantically distinct)
            assert store.node_count() > pre_count
            p.save_graph(store)
            p.close()

        # --- Final verification: reopen and ensure everything accumulated ---
        final = SQLitePersistence(db_path)
        final_store = final.load_graph()

        # 5 distinct inputs across 3 sessions
        assert final_store.node_count() == 5

        all_logs = final.get_interactions(limit=100)
        assert len(all_logs) == 5

        # Sanity: every logged node_id still resolves in the graph
        for log in all_logs:
            if log.matched_node_id is not None:
                final_store.get_node(log.matched_node_id)  # must not raise
        final.close()


# --- E2E Scenario 3: composed skill chain ---


class TestComposedSkillPersistence:
    """A realistic composed skill survives save/load with correct ordering."""

    def test_git_commit_workflow_persisted(
        self,
        db_path: str,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
    ) -> None:
        p = SQLitePersistence(db_path)
        store = p.load_graph()

        # Build a composed "git commit" chain: root → stage → write_msg → commit → verify
        root = _make_learned_node(
            "git-commit-root",
            "commit my changes",
            "running git commit workflow",
            embedder=embedder,
            confidence=0.9,
            stability=Stability.HIGH,
        )
        root.is_composed = True

        steps = [
            _make_learned_node("stage-files", "stage files", "git add .", embedder),
            _make_learned_node("write-msg", "write commit message", "draft commit msg", embedder),
            _make_learned_node("run-commit", "run git commit", "git commit -m …", embedder),
            _make_learned_node("verify", "verify repo state", "git status", embedder),
        ]
        for i, step in enumerate(steps):
            step.sequence_position = i

        store.put_node(root)
        for step in steps:
            store.put_node(step)
        for i, step in enumerate(steps):
            store.add_link(root.pattern_id, ChildLink(habit_id=step.pattern_id, order=i))

        # 'verify' is shared with a deploy workflow
        deploy_root = _make_learned_node(
            "deploy-root", "deploy my app", "running deploy", embedder
        )
        store.put_node(deploy_root)
        store.add_link("deploy-root", ChildLink(habit_id="verify", order=0))

        p.save_graph(store)
        p.close()

        # --- Reopen ---
        p2 = SQLitePersistence(db_path)
        loaded = p2.load_graph()

        assert loaded.node_count() == 6
        assert loaded.get_node("git-commit-root").is_composed is True
        assert loaded.get_node("git-commit-root").stability == Stability.HIGH

        # Sequence preserved in order
        children = loaded.get_children("git-commit-root")
        assert [c.habit_id for c in children] == [
            "stage-files",
            "write-msg",
            "run-commit",
            "verify",
        ]

        # Shared building block: verify is reachable from both roots
        assert loaded.get_parents("verify") == {"git-commit-root", "deploy-root"}

        # Running the loaded chain by semantic lookup still works
        query = normalizer.normalize("please commit these changes!")
        q_vec = embedder.embed(query.normalized)
        best = max(
            loaded.all_nodes(),
            key=lambda n: _cosine(q_vec, n.embedding_vector),
        )
        assert best.pattern_id == "git-commit-root"

        p2.close()


# --- E2E Scenario 4: semantic discrimination ---


class TestSemanticDiscrimination:
    """Real embeddings should keep distinct intents distinguishable end-to-end."""

    def test_unrelated_queries_land_on_different_nodes(
        self,
        db_path: str,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
    ) -> None:
        p = SQLitePersistence(db_path)
        store = p.load_graph()

        seeds = {
            "weather": "what is the weather today",
            "name": "what is my name",
            "time": "what time is it",
            "calc": "what is two plus two",
        }
        for pid, text in seeds.items():
            store.put_node(
                _make_learned_node(pid, text, f"response-{pid}", embedder)
            )
        p.save_graph(store)
        p.close()

        p2 = SQLitePersistence(db_path)
        loaded = p2.load_graph()

        probes = {
            "weather": "how's the weather outside",
            "name": "tell me my name",
            "time": "what's the current time",
            "calc": "compute 2 + 2",
        }
        for expected_id, probe in probes.items():
            norm = normalizer.normalize(probe)
            qv = embedder.embed(norm.normalized)
            ranked = sorted(
                loaded.all_nodes(),
                key=lambda n: _cosine(qv, n.embedding_vector),
                reverse=True,
            )
            assert ranked[0].pattern_id == expected_id, (
                f"probe '{probe}' matched {ranked[0].pattern_id!r} "
                f"instead of {expected_id!r}"
            )

        p2.close()


# --- E2E Scenario 5: input hygiene flows end-to-end ---


class TestInputHygieneEndToEnd:
    """Adversarial / messy inputs must still produce consistent stored state."""

    def test_unicode_and_control_chars_round_trip(
        self,
        db_path: str,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
    ) -> None:
        p = SQLitePersistence(db_path)
        store = p.load_graph()

        messy_inputs = [
            "  HELLO\t\tworld  ",              # whitespace + case
            "café\u0301",                      # combining accent (NFKC → café)
            "Hello\u200bWorld",                 # zero-width space
            "test\x00input\x01",                # control chars
            "こんにちは 🌍",                      # CJK + emoji
        ]

        for i, raw in enumerate(messy_inputs):
            norm = normalizer.normalize(raw)
            # Normalizer must strip control chars and zero-widths
            assert "\x00" not in norm.normalized
            assert "\u200b" not in norm.normalized

            vec = embedder.embed(norm.normalized)
            node = HabitNode(
                pattern_id=f"msg-{i}",
                trigger_patterns=[norm.normalized],
                embedding_vector=vec,
                response=norm.normalized,
            )
            store.put_node(node)

        p.save_graph(store)
        p.close()

        # Reopen & verify each survived JSON+SQLite
        p2 = SQLitePersistence(db_path)
        loaded = p2.load_graph()
        assert loaded.node_count() == len(messy_inputs)
        for i in range(len(messy_inputs)):
            n = loaded.get_node(f"msg-{i}")
            # Embedding is still a usable 384-d vector
            assert len(n.embedding_vector) == 384
            # Re-embedding the normalized trigger equals the stored vector
            fresh = embedder.embed(n.trigger_patterns[0])
            assert _cosine(fresh, n.embedding_vector) == pytest.approx(1.0, abs=1e-4)
        p2.close()


# --- E2E Scenario 6: remove cascade + durable delete ---


class TestRemoveCascadeDurability:
    """Removing a node propagates through persistence correctly."""

    def test_remove_node_persists_cascade(
        self,
        db_path: str,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
    ) -> None:
        p = SQLitePersistence(db_path)
        store = p.load_graph()

        store.put_node(_make_learned_node("parent", "start job", "starting", embedder))
        store.put_node(_make_learned_node("child1", "first step", "step 1", embedder))
        store.put_node(_make_learned_node("child2", "second step", "step 2", embedder))
        store.add_link("parent", ChildLink(habit_id="child1", order=0))
        store.add_link("parent", ChildLink(habit_id="child2", order=1))
        p.save_graph(store)
        p.close()

        # Remove the parent in a new session
        p2 = SQLitePersistence(db_path)
        s2 = p2.load_graph()
        s2.remove_node("parent")
        p2.save_graph(s2)
        p2.close()

        # A third session should see the cascade
        p3 = SQLitePersistence(db_path)
        s3 = p3.load_graph()
        assert s3.node_count() == 2
        with pytest.raises(NodeNotFoundError):
            s3.get_node("parent")
        # Children survived, and no longer have 'parent' as a parent
        assert s3.get_parents("child1") == set()
        assert s3.get_parents("child2") == set()
        p3.close()
