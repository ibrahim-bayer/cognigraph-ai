"""End-to-end integration tests.

Wires together every component that currently exists —
config, normalizer, embedder, graph store, SQLite persistence, FAISS
vector index — and exercises realistic multi-session user journeys
against real files and the real sentence-transformers embedding model.

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
from cognigraph.exceptions import LLMPermanentError, LLMRetriableError
from cognigraph.learner import FlatNodeLearner
from cognigraph.llm_client import ClaudeLLMProvider
from cognigraph.matcher import NodeMatcher
from cognigraph.normalizer import InputNormalizer
from cognigraph.persistence import SQLitePersistence
from cognigraph.pipeline import CogniGraphPipeline
from cognigraph.reinforcement import ReinforcementLogger
from cognigraph.safety import SafetyBoundary
from cognigraph.types import LLMResponse
from cognigraph.vector_index import FAISSIndex


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
    reinforcement_count: int = 0,
) -> HabitNode:
    """Build a node as the learning loop would: real embedding of a real trigger.

    `reinforcement_count` defaults to 0 — a brand-new node hasn't been
    used yet. Tests that need to seed a "previously-used" node should
    pass an explicit count.
    """
    vec = embedder.embed(trigger_text)
    return HabitNode(
        pattern_id=pattern_id,
        trigger_patterns=[trigger_text],
        embedding_vector=vec,
        confidence=confidence,
        reinforcement_count=reinforcement_count,
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
        # Helper seeds at count=0; we incremented once on Turn 2 → 1
        assert restored.reinforcement_count == 1
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


# =====================================================================
# FAISS-backed retrieval scenarios
# These exercise the full real pipeline: normalizer → embedder → FAISS
# vector index → graph store → SQLite persistence, across session
# boundaries and on-disk round-trips of both index files.
# =====================================================================


def _rebuild_faiss_from_graph(store: InMemoryGraphStore, dim: int) -> FAISSIndex:
    """Rebuild a FAISS index from whatever's in the graph store.

    Models the real "startup" flow where the authoritative graph lives
    in SQLite and the FAISS index is reconstructed from the loaded
    nodes' embeddings.
    """
    idx = FAISSIndex(dimension=dim)
    for node in store.all_nodes():
        if node.embedding_vector:
            idx.add(node.pattern_id, node.embedding_vector)
    return idx


class TestFAISSSemanticRetrievalE2E:
    """FAISS replaces linear cosine scan in the retrieval path."""

    def test_faiss_matches_brute_force_top_hit(
        self,
        db_path: str,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        """For every probe, FAISS's top-1 must agree with a brute-force
        cosine scan over the same node set. Any divergence would mean
        FAISS is drifting from the ground truth."""
        seeds = {
            "weather": "what is the weather today",
            "name": "what is my name",
            "time": "what time is it",
            "calc": "what is two plus two",
            "commit": "commit my changes to git",
            "deploy": "deploy the app to production",
            "timer": "set a timer for five minutes",
            "cancel": "cancel my timer",
        }

        store = InMemoryGraphStore()
        faiss_idx = FAISSIndex(dimension=config.embedding_dim)

        for pid, text in seeds.items():
            node = _make_learned_node(pid, text, f"response-{pid}", embedder)
            store.put_node(node)
            faiss_idx.add(pid, node.embedding_vector)

        # Probes — reworded versions of the seeds
        probes = {
            "weather": "how's the weather outside",
            "name": "tell me my name",
            "time": "what's the current time",
            "calc": "compute 2 + 2",
            "commit": "please commit these edits",
            "deploy": "push to prod",
            "timer": "start a 5-minute timer",
            "cancel": "stop the timer",
        }

        for expected_id, probe in probes.items():
            norm = normalizer.normalize(probe)
            qv = embedder.embed(norm.normalized)

            # Brute-force ground truth
            ranked_brute = sorted(
                store.all_nodes(),
                key=lambda n: _cosine(qv, n.embedding_vector),
                reverse=True,
            )
            brute_top = ranked_brute[0].pattern_id

            # FAISS top-1
            faiss_top = faiss_idx.search(qv, k=1)[0][0]

            assert faiss_top == brute_top, (
                f"probe {probe!r}: faiss={faiss_top}, brute={brute_top}"
            )
            # Confidence check: the correct seed is the top hit
            assert faiss_top == expected_id, (
                f"probe {probe!r} matched {faiss_top}, expected {expected_id}"
            )

        faiss_idx.close()

    def test_faiss_topk_ordering_matches_brute_force(
        self,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        """Top-k results must come back in the same order as brute force,
        with scores that match to float32 precision."""
        store = InMemoryGraphStore()
        faiss_idx = FAISSIndex(dimension=config.embedding_dim)

        texts = [
            "tell me a joke",
            "make me laugh",
            "say something funny",
            "what's the weather",
            "sunshine or rain",
            "commit my changes",
            "git commit now",
            "deploy to prod",
        ]
        for i, text in enumerate(texts):
            node = _make_learned_node(
                f"n{i}", text, f"r{i}", embedder
            )
            store.put_node(node)
            faiss_idx.add(f"n{i}", node.embedding_vector)

        probe = "something hilarious please"
        qv = embedder.embed(normalizer.normalize(probe).normalized)

        # Brute-force top-3
        brute = sorted(
            store.all_nodes(),
            key=lambda n: _cosine(qv, n.embedding_vector),
            reverse=True,
        )[:3]
        brute_ids = [n.pattern_id for n in brute]
        brute_scores = [_cosine(qv, n.embedding_vector) for n in brute]

        # FAISS top-3
        faiss_hits = faiss_idx.search(qv, k=3)
        faiss_ids = [h[0] for h in faiss_hits]
        faiss_scores = [h[1] for h in faiss_hits]

        assert faiss_ids == brute_ids
        for fs, bs in zip(faiss_scores, brute_scores):
            assert fs == pytest.approx(bs, abs=1e-5)

        faiss_idx.close()


class TestFullStackSessionRoundTrip:
    """Full startup→work→shutdown→startup cycle across all components."""

    def test_faiss_rebuilt_from_persisted_graph(
        self,
        tmp_path: Path,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        db = str(tmp_path / "fullstack.db")
        faiss_path = str(tmp_path / "fullstack.faiss")

        # --- Session 1: new user, build knowledge ---
        p1 = SQLitePersistence(db)
        store = p1.load_graph()  # empty
        assert store.node_count() == 0

        idx = _rebuild_faiss_from_graph(store, config.embedding_dim)
        assert idx.count() == 0

        inputs = [
            ("What's my name?", "Ibrahim"),
            ("What time is it?", "time is a social construct"),
            ("Commit my changes", "running git commit"),
            ("Deploy to prod", "deploying now"),
        ]
        for raw, answer in inputs:
            norm = normalizer.normalize(raw)
            qv = embedder.embed(norm.normalized)

            # Check FAISS for a match — empty so always miss first time
            hits = idx.search(qv, k=1)
            if hits and hits[0][1] > 0.95:
                route = RouteDecision.GRAPH_DIRECT
                matched_id = hits[0][0]
                response = store.get_node(matched_id).response
            else:
                route = RouteDecision.LLM_ONLY
                pid = f"h{len(store.all_nodes())}"
                node = _make_learned_node(pid, norm.normalized, answer, embedder)
                store.put_node(node)
                idx.add(pid, node.embedding_vector)
                matched_id = pid
                response = answer

            p1.log_interaction(
                InteractionLog(
                    timestamp=time.time(),
                    input_text=raw,
                    normalized_text=norm.normalized,
                    route_decision=route,
                    matched_node_id=matched_id,
                    response_text=response,
                    latency_ms=1.0,
                )
            )

        # Persist both stores
        p1.save_graph(store)
        idx.save(faiss_path)
        idx.close()
        p1.close()

        assert Path(db).exists()
        assert Path(faiss_path).exists()
        assert Path(faiss_path + ".ids.json").exists()

        # --- Session 2: restart, reload everything ---
        p2 = SQLitePersistence(db)
        store2 = p2.load_graph()
        assert store2.node_count() == 4

        # Two ways to rehydrate FAISS: load from disk, or rebuild from graph.
        # Both should produce equivalent indices. Test both paths.
        idx_loaded = FAISSIndex(dimension=config.embedding_dim)
        idx_loaded.load(faiss_path)

        idx_rebuilt = _rebuild_faiss_from_graph(store2, config.embedding_dim)

        assert idx_loaded.count() == 4
        assert idx_rebuilt.count() == 4

        # For each original input's rephrasing, both FAISS instances
        # should return the same top-1 node.
        rephrasings = [
            ("tell me my name", 0),
            ("what's the current time", 1),
            ("commit the code", 2),
            ("push to production", 3),
        ]
        for raw, _expected_idx in rephrasings:
            qv = embedder.embed(normalizer.normalize(raw).normalized)
            loaded_top = idx_loaded.search(qv, k=1)[0][0]
            rebuilt_top = idx_rebuilt.search(qv, k=1)[0][0]
            assert loaded_top == rebuilt_top, (
                f"probe {raw!r}: loaded={loaded_top}, rebuilt={rebuilt_top}"
            )
            # And both must resolve in the loaded graph store
            store2.get_node(loaded_top)

        # --- Session 3: reinforce + remove, persist again ---
        # User uses "commit" twice more → reinforce, then retires "deploy"
        commit_node = store2.get_node("h2")
        commit_node.reinforcement_count += 2
        commit_node.confidence = min(1.0, commit_node.confidence + 0.1)
        commit_node.stability = Stability.MEDIUM
        store2.put_node(commit_node)

        store2.remove_node("h3")  # retire deploy
        idx_loaded.remove("h3")

        p2.save_graph(store2)
        idx_loaded.save(faiss_path)
        idx_loaded.close()
        p2.close()

        # --- Session 4: verify retirement ---
        p3 = SQLitePersistence(db)
        store3 = p3.load_graph()
        assert store3.node_count() == 3

        idx3 = FAISSIndex(dimension=config.embedding_dim)
        idx3.load(faiss_path)
        assert idx3.count() == 3

        # "deploy" query no longer lands on h3 (which is gone)
        qv = embedder.embed(
            normalizer.normalize("ship to production").normalized
        )
        top = idx3.search(qv, k=3)
        top_ids = {h[0] for h in top}
        assert "h3" not in top_ids

        # Reinforcement survived (node was created with count=0, then +2)
        commit_reloaded = store3.get_node("h2")
        assert commit_reloaded.reinforcement_count == 2
        assert commit_reloaded.stability == Stability.MEDIUM

        idx3.close()
        p3.close()


class TestFAISSAndGraphStoreConsistency:
    """Mirror operations between graph store and FAISS must stay in sync."""

    def test_parallel_add_remove(
        self,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        store = InMemoryGraphStore()
        idx = FAISSIndex(dimension=config.embedding_dim)

        inputs = [f"query number {i} about {topic}" for i, topic in enumerate(
            ["cats", "dogs", "birds", "fish", "horses", "rabbits", "hamsters", "lizards"]
        )]
        for i, text in enumerate(inputs):
            n = _make_learned_node(f"n{i}", text, f"r{i}", embedder)
            store.put_node(n)
            idx.add(n.pattern_id, n.embedding_vector)

        assert store.node_count() == 8
        assert idx.count() == 8

        # Remove every other node in both stores
        for i in range(0, 8, 2):
            store.remove_node(f"n{i}")
            idx.remove(f"n{i}")

        assert store.node_count() == 4
        assert idx.count() == 4

        # Every graph-store node is findable in FAISS as its own top-1
        for node in store.all_nodes():
            hits = idx.search(node.embedding_vector, k=1)
            assert hits[0][0] == node.pattern_id
            assert hits[0][1] == pytest.approx(1.0, abs=1e-5)

        # Every FAISS search for a removed id returns something ELSE
        # (the remaining dogs/fish/rabbits/lizards)
        for removed_id in ["n0", "n2", "n4", "n6"]:
            with pytest.raises(NodeNotFoundError):
                store.get_node(removed_id)

        idx.close()

    def test_overwrite_stays_consistent_with_reinforcement(
        self,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        """When a node's trigger text is edited (re-embed + overwrite),
        FAISS must start returning matches for the new text, not the old."""
        store = InMemoryGraphStore()
        idx = FAISSIndex(dimension=config.embedding_dim)

        original = _make_learned_node(
            "mutable", "book a table for dinner", "reservation made", embedder
        )
        store.put_node(original)
        idx.add("mutable", original.embedding_vector)

        # User correction: same habit, different trigger
        updated_text = "schedule a meeting tomorrow"
        updated_vec = embedder.embed(updated_text)
        updated = store.get_node("mutable")
        updated.trigger_patterns = [updated_text]
        updated.embedding_vector = updated_vec
        store.put_node(updated)
        idx.add("mutable", updated_vec)  # overwrite path

        assert idx.count() == 1

        # FAISS now matches the new phrasing strongly
        q_new = embedder.embed("set up a meeting for tomorrow")
        hits = idx.search(q_new, k=1)
        assert hits[0][0] == "mutable"
        assert hits[0][1] > 0.85

        # ...and the old phrasing is now weakly matched (or at least
        # weaker than the new phrasing)
        q_old = embedder.embed("book a dinner reservation")
        score_old = idx.search(q_old, k=1)[0][1]
        score_new = hits[0][1]
        assert score_new > score_old

        idx.close()


class TestFAISSScaleWithRealEmbeddings:
    """Smoke-test scale with real E5 embeddings."""

    def test_fifty_real_nodes_retrievable(
        self,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        """50 distinct real sentences, each findable as its own top-1."""
        sentences = [
            "what is the capital of france",
            "who painted the mona lisa",
            "how do I boil an egg",
            "write a python function to reverse a string",
            "explain quantum entanglement",
            "what is the best programming language",
            "tell me about the roman empire",
            "how does photosynthesis work",
            "who invented the telephone",
            "what time zone is tokyo in",
            "recipe for chocolate chip cookies",
            "how tall is mount everest",
            "who wrote hamlet",
            "what is the speed of light",
            "how do vaccines work",
            "tell me a dad joke",
            "what is machine learning",
            "how to change a tire",
            "who is the president of the usa",
            "what is the meaning of life",
            "explain docker containers",
            "how to make sourdough bread",
            "what is the largest ocean",
            "who discovered america",
            "how to meditate",
            "tell me about black holes",
            "what is cryptocurrency",
            "how to learn spanish quickly",
            "what causes rain",
            "tell me the time in new york",
            "how to write a resume",
            "what is a neural network",
            "explain the theory of relativity",
            "how long do humans live on average",
            "what is the longest river",
            "who founded apple",
            "how does gps work",
            "what is blockchain",
            "how to start a garden",
            "tell me about ancient egypt",
            "what is the largest planet",
            "how do airplanes fly",
            "explain compound interest",
            "what is a black swan event",
            "how to improve sleep",
            "what is climate change",
            "who composed the four seasons",
            "how to fix a leaky faucet",
            "what is the pythagorean theorem",
            "tell me a bedtime story",
        ]
        assert len(sentences) == 50

        idx = FAISSIndex(dimension=config.embedding_dim)
        for i, s in enumerate(sentences):
            norm = normalizer.normalize(s)
            vec = embedder.embed(norm.normalized)
            idx.add(f"s{i}", vec)

        assert idx.count() == 50

        # Every sentence, when queried, returns its own id as top-1 with
        # score ≈ 1.0. This is the integration smoke test: normalize +
        # embed + FAISS all agree across 50 real samples.
        for i, s in enumerate(sentences):
            norm = normalizer.normalize(s)
            qv = embedder.embed(norm.normalized)
            hits = idx.search(qv, k=1)
            assert hits[0][0] == f"s{i}", (
                f"sentence {i} {s!r} mismatched: {hits[0][0]}"
            )
            assert hits[0][1] == pytest.approx(1.0, abs=1e-4)

        idx.close()


# =====================================================================
# NodeMatcher end-to-end scenarios
# Exercise the full routing layer: normalizer → embedder → matcher,
# across cold-start, learning, reinforcement, composed skills, and
# cross-session durability with real files and the real E5 model.
# =====================================================================


class TestMatcherColdStart:
    """An empty graph routes everything to LLM_ONLY."""

    def test_cold_start_routes_every_query_to_llm_only(
        self,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        store = InMemoryGraphStore()
        idx = FAISSIndex(dimension=config.embedding_dim)
        matcher = NodeMatcher(store, idx, config)

        for raw in ["hello", "what's my name", "make me a sandwich"]:
            qv = embedder.embed(normalizer.normalize(raw).normalized)
            result = matcher.match(qv)
            assert result.node is None
            assert result.route_decision == RouteDecision.LLM_ONLY
            assert result.candidates == []

        idx.close()


class TestMatcherLearningLoop:
    """Simulate a learning loop: LLM_ONLY → create → reinforce → GRAPH_DIRECT."""

    def test_cold_to_confident_recall(
        self,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        store = InMemoryGraphStore()
        idx = FAISSIndex(dimension=config.embedding_dim)
        matcher = NodeMatcher(store, idx, config)

        raw = "what is my name?"
        qv = embedder.embed(normalizer.normalize(raw).normalized)

        # Turn 1: cold, routes to LLM
        r1 = matcher.match(qv)
        assert r1.route_decision == RouteDecision.LLM_ONLY

        # Simulate LLM response → create a habit node at starting confidence.
        # learning_starting_confidence defaults to 0.5 which is below the 0.7
        # confidence_threshold, so immediately after creation it still
        # needs reinforcement.
        node = _make_learned_node(
            "habit-name",
            normalizer.normalize(raw).normalized,
            "Ibrahim",
            embedder,
            confidence=config.learning_starting_confidence,
        )
        store.put_node(node)
        idx.add(node.pattern_id, node.embedding_vector)

        # Turn 2: same question, graph matches strongly but confidence is
        # still below threshold → LLM_FALLBACK
        r2 = matcher.match(qv)
        assert r2.node is not None
        assert r2.node.pattern_id == "habit-name"
        assert r2.route_decision == RouteDecision.LLM_FALLBACK

        # Reinforce repeatedly until confidence crosses the threshold
        for _ in range(20):
            node.confidence = min(1.0, node.confidence + config.confidence_boost)
            node.reinforcement_count += 1
        store.put_node(node)
        idx.add(node.pattern_id, node.embedding_vector)

        # Turn N: now confident → GRAPH_DIRECT
        r3 = matcher.match(qv)
        assert r3.route_decision == RouteDecision.GRAPH_DIRECT
        assert r3.node.response == "Ibrahim"
        assert r3.score > config.confidence_threshold

        idx.close()


class TestMatcherRoutesOnRealRewording:
    """Real probes must hit GRAPH_DIRECT for in-distribution rewordings
    and LLM_ONLY for out-of-distribution queries."""

    def test_in_distribution_reworded_probe_routes_direct(
        self,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        store = InMemoryGraphStore()
        idx = FAISSIndex(dimension=config.embedding_dim)
        matcher = NodeMatcher(store, idx, config)

        seed = "what time is it"
        node = _make_learned_node(
            "time-habit", seed, "3:15 PM", embedder, confidence=0.9
        )
        store.put_node(node)
        idx.add(node.pattern_id, node.embedding_vector)

        # Reworded probe — should still match strongly
        reworded = "tell me the current time please"
        qv = embedder.embed(normalizer.normalize(reworded).normalized)
        result = matcher.match(qv)

        assert result.node is not None
        assert result.node.pattern_id == "time-habit"
        assert result.route_decision == RouteDecision.GRAPH_DIRECT

        idx.close()

    def test_out_of_distribution_probe_routes_llm_only(
        self,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        store = InMemoryGraphStore()
        idx = FAISSIndex(dimension=config.embedding_dim)
        matcher = NodeMatcher(store, idx, config)

        # Teach a bunch of habits about cooking
        cooking = [
            ("boil-eggs", "how do I boil an egg"),
            ("bake-bread", "how to bake sourdough bread"),
            ("grill-steak", "best way to grill a steak"),
        ]
        for pid, text in cooking:
            node = _make_learned_node(pid, text, "answer", embedder, confidence=0.95)
            store.put_node(node)
            idx.add(pid, node.embedding_vector)

        # Completely unrelated probe
        probe = "who was the 16th president of the united states"
        qv = embedder.embed(normalizer.normalize(probe).normalized)
        result = matcher.match(qv)

        # With real E5 the sim between unrelated sentences can still be in
        # the 0.6-0.75 range because all inputs are English questions, so
        # accept either LLM_ONLY or LLM_FALLBACK — the point is it must
        # NOT be GRAPH_DIRECT or GRAPH_COMPOSED.
        assert result.route_decision in (
            RouteDecision.LLM_ONLY,
            RouteDecision.LLM_FALLBACK,
        )

        idx.close()


class TestMatcherComposedSkillRouting:
    """A matched composed root routes GRAPH_COMPOSED, not GRAPH_DIRECT."""

    def test_composed_root_matched_via_fuzzy_query(
        self,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        store = InMemoryGraphStore()
        idx = FAISSIndex(dimension=config.embedding_dim)
        matcher = NodeMatcher(store, idx, config)

        # Build git commit chain
        root = _make_learned_node(
            "commit-root",
            "commit my changes",
            "running commit workflow",
            embedder,
            confidence=0.95,
            stability=Stability.HIGH,
        )
        store.put_node(root)
        idx.add(root.pattern_id, root.embedding_vector)

        for i, (pid, text) in enumerate([
            ("stage", "git stage files"),
            ("write", "draft commit message"),
            ("run", "run git commit"),
            ("verify", "check git status"),
        ]):
            n = _make_learned_node(pid, text, f"step-{pid}", embedder, confidence=0.9)
            store.put_node(n)
            idx.add(pid, n.embedding_vector)
            store.add_link("commit-root", ChildLink(habit_id=pid, order=i))

        # Fuzzy, reworded probe
        probe = "please save my edits to the repo"
        qv = embedder.embed(normalizer.normalize(probe).normalized)
        result = matcher.match(qv)

        assert result.node is not None
        assert result.node.pattern_id == "commit-root"
        assert result.route_decision == RouteDecision.GRAPH_COMPOSED

        idx.close()


class TestMatcherCrossSession:
    """Matcher behavior survives save/load of graph + FAISS index."""

    def test_matcher_on_reloaded_stack(
        self,
        tmp_path: Path,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        db = str(tmp_path / "matcher.db")
        faiss_path = str(tmp_path / "matcher.faiss")

        # --- S1: train ---
        p1 = SQLitePersistence(db)
        store = p1.load_graph()
        idx = FAISSIndex(dimension=config.embedding_dim)

        for pid, text in [
            ("weather", "what's the weather today"),
            ("joke", "tell me a joke"),
            ("commit", "commit my changes to git"),
        ]:
            n = _make_learned_node(pid, text, f"r-{pid}", embedder, confidence=0.95)
            store.put_node(n)
            idx.add(pid, n.embedding_vector)

        p1.save_graph(store)
        idx.save(faiss_path)
        idx.close()
        p1.close()

        # --- S2: reload + match ---
        p2 = SQLitePersistence(db)
        store2 = p2.load_graph()
        idx2 = FAISSIndex(dimension=config.embedding_dim)
        idx2.load(faiss_path)
        matcher = NodeMatcher(store2, idx2, config)

        probes = [
            ("how's the weather outside", "weather", RouteDecision.GRAPH_DIRECT),
            ("make me laugh", "joke", RouteDecision.GRAPH_DIRECT),
            ("push my code edits", "commit", RouteDecision.GRAPH_DIRECT),
        ]
        for raw, expected_id, expected_route in probes:
            qv = embedder.embed(normalizer.normalize(raw).normalized)
            result = matcher.match(qv)
            assert result.node is not None, f"probe {raw!r} unexpectedly missed"
            assert result.node.pattern_id == expected_id
            assert result.route_decision == expected_route

        idx2.close()
        p2.close()


class TestMatcherConflictDetection:
    """When two similar nodes are close in score, candidates expose both
    so the learning loop can flag conflict."""

    def test_candidates_reveal_ambiguity(
        self,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        store = InMemoryGraphStore()
        idx = FAISSIndex(dimension=config.embedding_dim)
        matcher = NodeMatcher(store, idx, config)

        # Two competing habits for almost the same intent
        a = _make_learned_node(
            "weather-today", "what's the weather today", "sunny", embedder, confidence=0.8
        )
        b = _make_learned_node(
            "weather-now", "what is the weather right now", "sunny", embedder, confidence=0.8
        )
        for n in (a, b):
            store.put_node(n)
            idx.add(n.pattern_id, n.embedding_vector)

        probe = "tell me today's weather"
        qv = embedder.embed(normalizer.normalize(probe).normalized)
        result = matcher.match(qv)

        assert result.node is not None
        # Both candidates present
        ids = {c[0] for c in result.candidates}
        assert "weather-today" in ids
        assert "weather-now" in ids
        # Gap between top-2 scores is small (conflict signal)
        sorted_scores = sorted((c[1] for c in result.candidates), reverse=True)
        gap = sorted_scores[0] - sorted_scores[1]
        assert gap < 0.2  # < 0.2 is "close call" territory for near-dupes

        idx.close()


# =====================================================================
# ClaudeLLMProvider end-to-end scenarios
# Exercise the full pipeline through the real provider code path using
# an injected fake anthropic client (no network). The provider's code —
# message building, error classification, latency tracking, token
# accounting, context validation — runs exactly as it will in prod.
# =====================================================================


class _FakeAnthropicUsage:
    def __init__(self, input_tokens: int = 10, output_tokens: int = 20) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeAnthropicResponse:
    def __init__(self, text: str, model: str = "claude-test") -> None:
        self.content = [_FakeTextBlock(text)]
        self.model = model
        self.usage = _FakeAnthropicUsage()


class _FakeAnthropicMessagesAPI:
    """Stateful fake: answers from a lookup table and records every call."""

    def __init__(self) -> None:
        self.answers: dict[str, str] = {}
        self.default_answer: str = "I'm not sure — let me think about it."
        self.calls: list[dict] = []
        self.error_to_raise: Exception | None = None

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error_to_raise is not None:
            err = self.error_to_raise
            self.error_to_raise = None  # one-shot so subsequent calls succeed
            raise err

        # Pick a response by matching the trailing user message against
        # the answers table (case-insensitive substring match).
        last_user = next(
            (
                m["content"]
                for m in reversed(kwargs.get("messages", []))
                if m.get("role") == "user"
            ),
            "",
        )
        needle = last_user.lower()
        text = self.default_answer
        for key, answer in self.answers.items():
            if key.lower() in needle:
                text = answer
                break
        return _FakeAnthropicResponse(text, model=kwargs.get("model", "claude-test"))


class _FakeAnthropicClient:
    def __init__(self) -> None:
        self.messages = _FakeAnthropicMessagesAPI()


class TestLLMProviderInPipeline:
    """Full graph-first pipeline with real ClaudeLLMProvider, fake backend."""

    def test_learning_loop_through_real_llm_provider(
        self,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        """Cold start → LLM fallback → node creation → reinforcement → direct recall.

        Unlike the earlier matcher learning-loop test which hand-codes
        LLM responses, this one routes every fallback through
        ClaudeLLMProvider.generate() so the provider's own code —
        latency tracking, token accounting, error handling — runs."""
        store = InMemoryGraphStore()
        idx = FAISSIndex(dimension=config.embedding_dim)
        matcher = NodeMatcher(store, idx, config)

        fake_client = _FakeAnthropicClient()
        fake_client.messages.answers = {
            "name": "Ibrahim",
            "weather": "sunny, 72°F",
            "time": "3:15 PM",
        }
        llm = ClaudeLLMProvider(
            api_key="test-key",
            model="claude-test",
            config=config,
            client=fake_client,
        )

        system_prompt = (
            "You are the cognitive fallback for a learned graph agent. "
            "Answer concisely."
        )

        # --- Turn 1: novel input → LLM_ONLY → call LLM → create node ---
        raw = "what's my name?"
        norm = normalizer.normalize(raw)
        qv = embedder.embed(norm.normalized)
        match = matcher.match(qv)
        assert match.route_decision == RouteDecision.LLM_ONLY

        llm_response = llm.generate(
            prompt=norm.normalized, system=system_prompt
        )
        assert llm_response.text == "Ibrahim"
        assert llm_response.latency_ms >= 0
        assert llm_response.input_tokens == 10
        assert llm_response.output_tokens == 20
        assert llm_response.model == "claude-test"

        # Confirm the provider passed through system + model + max_tokens
        last_call = fake_client.messages.calls[-1]
        assert last_call["system"] == system_prompt
        assert last_call["model"] == "claude-test"
        assert last_call["max_tokens"] == config.llm_max_tokens

        node = _make_learned_node(
            "habit-name",
            norm.normalized,
            llm_response.text,
            embedder,
            confidence=config.learning_starting_confidence,
        )
        store.put_node(node)
        idx.add(node.pattern_id, node.embedding_vector)

        # --- Turn 2: same intent, reworded — still fallback (low conf) ---
        qv2 = embedder.embed(normalizer.normalize("tell me my name").normalized)
        match2 = matcher.match(qv2)
        assert match2.node is not None
        assert match2.node.pattern_id == "habit-name"
        assert match2.route_decision == RouteDecision.LLM_FALLBACK

        # Fallback path still calls the LLM, but passes the graph hit as
        # context. Verify the provider correctly threads context through.
        followup = llm.generate(
            prompt="tell me my name",
            context=[
                {"role": "user", "content": norm.normalized},
                {"role": "assistant", "content": llm_response.text},
            ],
            system=system_prompt,
        )
        assert followup.text == "Ibrahim"
        last_call = fake_client.messages.calls[-1]
        assert [m["role"] for m in last_call["messages"]] == [
            "user",
            "assistant",
            "user",
        ]
        assert last_call["messages"][-1]["content"] == "tell me my name"

        # --- Reinforce to confident ---
        for _ in range(20):
            node.confidence = min(1.0, node.confidence + config.confidence_boost)
            node.reinforcement_count += 1
        store.put_node(node)
        idx.add(node.pattern_id, node.embedding_vector)

        # --- Turn N: now GRAPH_DIRECT — no LLM call needed ---
        call_count_before = len(fake_client.messages.calls)
        match3 = matcher.match(qv)
        assert match3.route_decision == RouteDecision.GRAPH_DIRECT
        # The pipeline would short-circuit the LLM here; prove nothing
        # new was sent to the fake client between turn 2 and now.
        assert len(fake_client.messages.calls) == call_count_before

        llm.close()
        idx.close()

    def test_llm_retriable_error_surfaces_through_pipeline(
        self,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        """A rate-limit-style transient error must reach the caller as
        LLMRetriableError (not a generic LLMError) so pipeline retry
        logic can distinguish it from permanent failures."""
        store = InMemoryGraphStore()
        idx = FAISSIndex(dimension=config.embedding_dim)
        fake_client = _FakeAnthropicClient()
        llm = ClaudeLLMProvider(
            api_key="k", model="claude-test", config=config, client=fake_client
        )

        # Inject a rate-limit error (classified by class name)
        rate_limit_cls = type("RateLimitError", (Exception,), {})
        fake_client.messages.error_to_raise = rate_limit_cls("please slow down")

        raw = "what's the weather"
        qv = embedder.embed(normalizer.normalize(raw).normalized)
        match = matcher_for(store, idx, config).match(qv)
        assert match.route_decision == RouteDecision.LLM_ONLY

        with pytest.raises(LLMRetriableError, match="RateLimitError"):
            llm.generate(normalizer.normalize(raw).normalized)

        # Subsequent call after the one-shot error should succeed,
        # proving the fake client state is clean and the provider is
        # re-usable after an error.
        fake_client.messages.answers = {"weather": "sunny"}
        result = llm.generate(normalizer.normalize(raw).normalized)
        assert result.text == "sunny"

        llm.close()
        idx.close()

    def test_llm_permanent_error_surfaces_through_pipeline(
        self,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        """An authentication error must reach the caller as
        LLMPermanentError so the pipeline does NOT retry it."""
        fake_client = _FakeAnthropicClient()
        llm = ClaudeLLMProvider(
            api_key="k", model="claude-test", config=config, client=fake_client
        )
        auth_cls = type("AuthenticationError", (Exception,), {})
        fake_client.messages.error_to_raise = auth_cls("bad api key")

        with pytest.raises(LLMPermanentError, match="AuthenticationError"):
            llm.generate("hi")

        llm.close()

    def test_llm_context_round_trip_across_multiple_turns(
        self,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        """A multi-turn conversation accumulates context and each turn
        sees the full history through the real provider."""
        fake_client = _FakeAnthropicClient()
        fake_client.messages.default_answer = "OK"
        llm = ClaudeLLMProvider(
            api_key="k", model="claude-test", config=config, client=fake_client
        )

        history: list[dict] = []
        turns = [
            "hi",
            "how are you",
            "tell me a joke",
            "make it shorter",
        ]
        for turn in turns:
            resp = llm.generate(
                prompt=normalizer.normalize(turn).normalized,
                context=history if history else None,
            )
            history.append({"role": "user", "content": normalizer.normalize(turn).normalized})
            history.append({"role": "assistant", "content": resp.text})

        # The final call should have seen 3 prior user/assistant pairs
        # plus the final user turn = 7 messages
        last_call = fake_client.messages.calls[-1]
        assert len(last_call["messages"]) == 7
        assert last_call["messages"][-1]["role"] == "user"
        assert last_call["messages"][-1]["content"] == "make it shorter"

        # Roles strictly alternate
        roles = [m["role"] for m in last_call["messages"]]
        assert roles == ["user", "assistant"] * 3 + ["user"]

        llm.close()

    def test_llm_invalid_context_rejected_before_sdk_call(
        self,
        config: CogniGraphConfig,
    ) -> None:
        """Bad context (ends in user) must raise LLMPermanentError
        before ever calling the SDK. Proves the validation lives in the
        provider, not at some downstream layer."""
        fake_client = _FakeAnthropicClient()
        llm = ClaudeLLMProvider(
            api_key="k", model="claude-test", config=config, client=fake_client
        )

        with pytest.raises(LLMPermanentError):
            llm.generate(
                "final",
                context=[
                    {"role": "user", "content": "a"},
                    {"role": "assistant", "content": "b"},
                    {"role": "user", "content": "c"},  # ends in user → reject
                ],
            )
        # Verify the SDK was NOT called
        assert fake_client.messages.calls == []

        llm.close()

    def test_llm_response_persisted_as_interaction_log(
        self,
        tmp_path: Path,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        """A full pipeline turn including LLM fallback is logged to SQLite
        with all fields populated from the real LLMResponse."""
        db = str(tmp_path / "llm_e2e.db")
        p = SQLitePersistence(db)
        store = p.load_graph()
        idx = FAISSIndex(dimension=config.embedding_dim)
        matcher = NodeMatcher(store, idx, config)

        fake_client = _FakeAnthropicClient()
        fake_client.messages.answers = {"time": "3:15 PM"}
        llm = ClaudeLLMProvider(
            api_key="k", model="claude-test", config=config, client=fake_client
        )

        raw = "what time is it"
        norm = normalizer.normalize(raw)
        qv = embedder.embed(norm.normalized)
        match = matcher.match(qv)
        assert match.route_decision == RouteDecision.LLM_ONLY

        turn_start = time.time()
        llm_response = llm.generate(norm.normalized)

        # Log the interaction with real LLM fields
        p.log_interaction(
            InteractionLog(
                timestamp=turn_start,
                input_text=raw,
                normalized_text=norm.normalized,
                route_decision=RouteDecision.LLM_ONLY,
                matched_node_id=None,
                llm_response=llm_response.text,
                response_text=llm_response.text,
                latency_ms=llm_response.latency_ms,
            )
        )
        p.close()

        # Reopen and verify
        p2 = SQLitePersistence(db)
        logs = p2.get_interactions()
        assert len(logs) == 1
        assert logs[0].llm_response == "3:15 PM"
        assert logs[0].response_text == "3:15 PM"
        assert logs[0].route_decision == RouteDecision.LLM_ONLY
        assert logs[0].latency_ms == pytest.approx(llm_response.latency_ms, abs=0.1)

        llm.close()
        idx.close()
        p2.close()


def matcher_for(
    store: InMemoryGraphStore,
    idx: FAISSIndex,
    config: CogniGraphConfig,
) -> NodeMatcher:
    """Small helper used by the retriable-error test."""
    return NodeMatcher(store, idx, config)


# =====================================================================
# ReinforcementLogger end-to-end scenarios
# Drive a node from cold creation through MEDIUM and HIGH stability
# tiers with the real logger writing to real SQLite, and verify the
# state survives close+reopen across the stability boundary.
# =====================================================================


class TestReinforcementLifecycleE2E:
    """Cold→MEDIUM→HIGH driven by ReinforcementLogger, real persistence."""

    def test_node_climbs_stability_tiers_via_logger(
        self,
        tmp_path: Path,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        db = str(tmp_path / "reinf_e2e.db")
        p = SQLitePersistence(db)
        store = p.load_graph()
        idx = FAISSIndex(dimension=config.embedding_dim)
        matcher = NodeMatcher(store, idx, config)
        rl = ReinforcementLogger(store, p, config)

        # Seed a confident node so the matcher routes GRAPH_DIRECT every turn
        node = _make_learned_node(
            "habit",
            "what is my name",
            "Ibrahim",
            embedder,
            confidence=0.95,
        )
        store.put_node(node)
        idx.add(node.pattern_id, node.embedding_vector)

        # Helper: run a turn and reinforce
        def _drive_one_turn() -> None:
            qv = embedder.embed(normalizer.normalize("what is my name").normalized)
            match = matcher.match(qv)
            assert match.route_decision == RouteDecision.GRAPH_DIRECT
            log = InteractionLog(
                timestamp=time.time(),
                input_text="what is my name",
                normalized_text="what is my name",
                route_decision=match.route_decision,
                matched_node_id=match.node.pattern_id,
                response_text=match.node.response,
                latency_ms=1.0,
            )
            assert rl.log_and_reinforce(log) is True

        # 4 reinforcements → still LOW
        for _ in range(4):
            _drive_one_turn()
        assert store.get_node("habit").reinforcement_count == 4
        assert store.get_node("habit").stability == Stability.LOW

        # 5th → MEDIUM
        _drive_one_turn()
        assert store.get_node("habit").reinforcement_count == 5
        assert store.get_node("habit").stability == Stability.MEDIUM

        # 14 more → still MEDIUM (count=19)
        for _ in range(14):
            _drive_one_turn()
        assert store.get_node("habit").reinforcement_count == 19
        assert store.get_node("habit").stability == Stability.MEDIUM

        # 20th → HIGH
        _drive_one_turn()
        assert store.get_node("habit").reinforcement_count == 20
        assert store.get_node("habit").stability == Stability.HIGH

        # Confidence saturated at 1.0 after the boost capacity
        assert store.get_node("habit").confidence == 1.0

        # Interaction log captured every turn
        history = rl.get_node_history("habit")
        assert len(history) == 20

        idx.close()
        p.close()

    def test_stability_persists_across_close_reopen(
        self,
        tmp_path: Path,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        db = str(tmp_path / "reinf_persist.db")

        # --- Session 1: drive to MEDIUM ---
        p1 = SQLitePersistence(db)
        store = p1.load_graph()
        idx1 = FAISSIndex(dimension=config.embedding_dim)
        matcher1 = NodeMatcher(store, idx1, config)
        rl1 = ReinforcementLogger(store, p1, config)

        node = _make_learned_node(
            "habit",
            "what's my name",
            "Ibrahim",
            embedder,
            confidence=0.9,
        )
        store.put_node(node)
        idx1.add(node.pattern_id, node.embedding_vector)

        for _ in range(5):
            qv = embedder.embed(normalizer.normalize("what's my name").normalized)
            match = matcher1.match(qv)
            rl1.log_and_reinforce(
                InteractionLog(
                    timestamp=time.time(),
                    input_text="what's my name",
                    normalized_text="what's my name",
                    route_decision=match.route_decision,
                    matched_node_id=match.node.pattern_id,
                    response_text=match.node.response,
                    latency_ms=1.0,
                )
            )
        assert store.get_node("habit").stability == Stability.MEDIUM
        assert store.get_node("habit").reinforcement_count == 5

        p1.save_graph(store)
        idx1.close()
        p1.close()

        # --- Session 2: reload and continue to HIGH ---
        p2 = SQLitePersistence(db)
        store2 = p2.load_graph()
        # Sanity: stability survived the round-trip
        assert store2.get_node("habit").stability == Stability.MEDIUM
        assert store2.get_node("habit").reinforcement_count == 5
        assert store2.node_count() == 1

        idx2 = FAISSIndex(dimension=config.embedding_dim)
        for n in store2.all_nodes():
            idx2.add(n.pattern_id, n.embedding_vector)
        matcher2 = NodeMatcher(store2, idx2, config)
        rl2 = ReinforcementLogger(store2, p2, config)

        # 15 more reinforcements → 20 total → HIGH
        for _ in range(15):
            qv = embedder.embed(normalizer.normalize("what's my name").normalized)
            match = matcher2.match(qv)
            rl2.log_and_reinforce(
                InteractionLog(
                    timestamp=time.time(),
                    input_text="what's my name",
                    normalized_text="what's my name",
                    route_decision=match.route_decision,
                    matched_node_id=match.node.pattern_id,
                    response_text=match.node.response,
                    latency_ms=1.0,
                )
            )
        assert store2.get_node("habit").reinforcement_count == 20
        assert store2.get_node("habit").stability == Stability.HIGH

        # Full history visible across sessions: 5 + 15 = 20 entries
        history = rl2.get_node_history("habit")
        assert len(history) == 20

        idx2.close()
        p2.close()

    def test_logger_skips_reinforcement_for_llm_routes_in_pipeline(
        self,
        tmp_path: Path,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        """When the matcher returns LLM_FALLBACK or LLM_ONLY, the logger
        records the interaction but does NOT touch the matched node."""
        db = str(tmp_path / "reinf_norein.db")
        p = SQLitePersistence(db)
        store = p.load_graph()
        idx = FAISSIndex(dimension=config.embedding_dim)
        matcher = NodeMatcher(store, idx, config)
        rl = ReinforcementLogger(store, p, config)

        # Low-confidence node → matcher returns LLM_FALLBACK
        node = _make_learned_node(
            "weak", "tell me a joke", "...", embedder, confidence=0.3
        )
        store.put_node(node)
        idx.add(node.pattern_id, node.embedding_vector)

        starting_count = store.get_node("weak").reinforcement_count
        starting_conf = store.get_node("weak").confidence

        qv = embedder.embed(normalizer.normalize("tell me a joke").normalized)
        match = matcher.match(qv)
        assert match.route_decision == RouteDecision.LLM_FALLBACK

        result = rl.log_and_reinforce(
            InteractionLog(
                timestamp=time.time(),
                input_text="tell me a joke",
                normalized_text="tell me a joke",
                route_decision=match.route_decision,
                matched_node_id=match.node.pattern_id,
                response_text="(LLM answer)",
                latency_ms=10.0,
            )
        )
        # Logger declined to reinforce
        assert result is False
        assert store.get_node("weak").reinforcement_count == starting_count
        assert store.get_node("weak").confidence == starting_conf

        # But the interaction was still logged
        assert len(p.get_interactions()) == 1

        idx.close()
        p.close()


# =====================================================================
# FlatNodeLearner end-to-end scenarios
# Drive the full pipeline (normalize → embed → matcher → LLM → log →
# learn) with the real E5 embedder, real FAISS, real SQLite, and the
# real FlatNodeLearner. Validates the spec acceptance criteria as well
# as the issue #22 fix on real data.
# =====================================================================


class TestLearnerLifecycleE2E:
    """Cold start → repeated LLM calls → learner promotes a habit to graph."""

    def test_three_repetitions_create_habit_and_route_graph_direct(
        self,
        tmp_path: Path,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        db = str(tmp_path / "learner_e2e.db")
        p = SQLitePersistence(db)
        store = p.load_graph()
        idx = FAISSIndex(dimension=config.embedding_dim)
        matcher = NodeMatcher(store, idx, config)
        rl = ReinforcementLogger(store, p, config)
        learner = FlatNodeLearner(store, idx, embedder, p, config)

        canonical_response = "The capital of France is Paris."
        rewordings = [
            "what is the capital of france",
            "tell me france's capital",
            "what's the capital city of france",
        ]

        outcome = None
        for i, raw in enumerate(rewordings):
            norm = normalizer.normalize(raw)
            qv = embedder.embed(norm.normalized)
            match = matcher.match(qv)
            log = InteractionLog(
                timestamp=time.time(),
                input_text=raw,
                normalized_text=norm.normalized,
                route_decision=match.route_decision,
                matched_node_id=(
                    match.node.pattern_id if match.node else None
                ),
                llm_response=canonical_response,
                response_text=canonical_response,
                latency_ms=10.0,
            )
            rl.log_and_reinforce(log)
            outcome = learner.evaluate_for_learning(log)

            if i < 2:
                assert outcome.created_node is None
            else:
                assert outcome.created_node is not None
                assert outcome.reason == "created"
                assert outcome.similar_count == 3

        assert outcome is not None
        learned = outcome.created_node
        assert learned is not None
        assert learned.response == canonical_response
        assert learned.confidence == config.learning_starting_confidence
        assert set(learned.trigger_patterns) == {
            normalizer.normalize(r).normalized for r in rewordings
        }

        # A new probe (semantically similar) finds the learned node
        probe = embedder.embed(
            normalizer.normalize("what's france's capital").normalized
        )
        probe_match = matcher.match(probe)
        assert probe_match.node is not None
        assert probe_match.node.pattern_id == learned.pattern_id

        idx.close()
        p.close()


class TestLearnerIssue22Fix:
    """Issue #22: distinct intents with similar embeddings but different
    LLM answers must produce distinct nodes, not collapse onto one."""

    def test_six_distinct_intents_create_six_nodes(
        self,
        tmp_path: Path,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        db = str(tmp_path / "learner_22.db")
        p = SQLitePersistence(db)
        store = p.load_graph()
        idx = FAISSIndex(dimension=config.embedding_dim)
        matcher = NodeMatcher(store, idx, config)
        rl = ReinforcementLogger(store, p, config)
        learner = FlatNodeLearner(store, idx, embedder, p, config)

        # Six distinct intents, each repeated 3 times with very minor
        # rewordings (punctuation, capitalization), each with a stable
        # but DIFFERENT response. With these tight rewordings every
        # within-intent pair clears the learning_stability_threshold
        # (0.9) while cross-intent pairs do not — the response-
        # divergence dedup then ensures distinct intents yield distinct
        # nodes (issue #22).
        intents = [
            (
                ["what is my name", "what is my name?", "what is my NAME"],
                "Ibrahim",
            ),
            (
                [
                    "what is the weather",
                    "what is the weather?",
                    "what is the WEATHER",
                ],
                "Sunny, 72°F.",
            ),
            (
                ["what time is it", "what time is it?", "what TIME is it"],
                "It's 3:15 PM.",
            ),
            (
                ["tell me a joke", "tell me a joke!", "tell me a JOKE"],
                "Why don't scientists trust atoms? They make up everything.",
            ),
            (
                [
                    "how do I commit changes",
                    "how do I commit changes?",
                    "how do I COMMIT changes",
                ],
                "Run: git add -A && git commit -m \"<msg>\".",
            ),
            (
                [
                    "what is two plus two",
                    "what is two plus two?",
                    "what IS two plus two",
                ],
                "Four.",
            ),
        ]

        for rewordings, response in intents:
            for raw in rewordings:
                norm = normalizer.normalize(raw)
                qv = embedder.embed(norm.normalized)
                match = matcher.match(qv)
                log = InteractionLog(
                    timestamp=time.time(),
                    input_text=raw,
                    normalized_text=norm.normalized,
                    route_decision=match.route_decision,
                    matched_node_id=(
                        match.node.pattern_id if match.node else None
                    ),
                    llm_response=response,
                    response_text=response,
                    latency_ms=10.0,
                )
                rl.log_and_reinforce(log)
                learner.evaluate_for_learning(log)

        # Issue #22 acceptance: distinct intents → distinct nodes
        assert store.node_count() == 6, (
            f"expected 6 distinct nodes, got {store.node_count()}: "
            f"{sorted(n.response[:30] for n in store.all_nodes())}"
        )

        # Each intent's canonical probe lands on its own node
        probes_to_response = [
            ("what is my name?", "Ibrahim"),
            ("what is the weather", "Sunny, 72°F."),
            ("what time is it?", "It's 3:15 PM."),
            (
                "tell me a joke",
                "Why don't scientists trust atoms? They make up everything.",
            ),
            (
                "how do I commit changes",
                "Run: git add -A && git commit -m \"<msg>\".",
            ),
            ("what is two plus two?", "Four."),
        ]
        seen_node_ids: set[str] = set()
        for probe, expected_response in probes_to_response:
            qv = embedder.embed(normalizer.normalize(probe).normalized)
            m = matcher.match(qv)
            assert m.node is not None, f"probe {probe!r} produced no match"
            assert m.node.response == expected_response, (
                f"probe {probe!r}: matched {m.node.response[:30]!r}, "
                f"expected {expected_response[:30]!r}"
            )
            seen_node_ids.add(m.node.pattern_id)
        assert len(seen_node_ids) == 6

        idx.close()
        p.close()


class TestLearnerSkipsAlreadyCovered:
    """When a node already covers a pattern (input + response), don't
    create a duplicate — even with 3+ similar interactions."""

    def test_existing_node_covering_pattern_blocks_duplicate(
        self,
        tmp_path: Path,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        db = str(tmp_path / "learner_dedup.db")
        p = SQLitePersistence(db)
        store = p.load_graph()
        idx = FAISSIndex(dimension=config.embedding_dim)
        rl = ReinforcementLogger(store, p, config)
        learner = FlatNodeLearner(store, idx, embedder, p, config)

        # Pre-seed a node that already covers the pattern
        existing = _make_learned_node(
            "existing-name",
            "what is my name",
            "Ibrahim",
            embedder,
            confidence=0.9,
        )
        store.put_node(existing)
        idx.add(existing.pattern_id, existing.embedding_vector)

        # Three more similar interactions with the SAME response
        outcome = None
        for raw in ("tell me my name", "say my name", "what's my name"):
            norm = normalizer.normalize(raw)
            log = InteractionLog(
                timestamp=time.time(),
                input_text=raw,
                normalized_text=norm.normalized,
                route_decision=RouteDecision.LLM_FALLBACK,
                matched_node_id=existing.pattern_id,
                llm_response="Ibrahim",
                response_text="Ibrahim",
                latency_ms=10.0,
            )
            rl.log_and_reinforce(log)
            outcome = learner.evaluate_for_learning(log)

        # No duplicate created
        assert store.node_count() == 1
        assert outcome is not None
        assert outcome.created_node is None
        assert outcome.reason == "already_covered_by_existing_node"

        idx.close()
        p.close()


class TestLearnerFailurePathsE2E:
    """Failure scenarios at the e2e level — the learner declines to
    crystallize, real components from end to end."""

    def test_insufficient_repetitions_no_node(
        self,
        tmp_path: Path,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        """Two LLM interactions for the same intent must not trigger
        learning — the count gate stops at min_repetitions=3."""
        db = str(tmp_path / "learner_insufficient.db")
        p = SQLitePersistence(db)
        store = p.load_graph()
        idx = FAISSIndex(dimension=config.embedding_dim)
        rl = ReinforcementLogger(store, p, config)
        learner = FlatNodeLearner(store, idx, embedder, p, config)

        canonical_response = "It is currently 3:15 PM."
        rewordings = [
            "what time is it",
            "what time is it?",
        ]

        outcome = None
        for raw in rewordings:
            norm = normalizer.normalize(raw)
            log = InteractionLog(
                timestamp=time.time(),
                input_text=raw,
                normalized_text=norm.normalized,
                route_decision=RouteDecision.LLM_ONLY,
                matched_node_id=None,
                llm_response=canonical_response,
                response_text=canonical_response,
                latency_ms=10.0,
            )
            rl.log_and_reinforce(log)
            outcome = learner.evaluate_for_learning(log)

        assert outcome is not None
        assert outcome.created_node is None
        assert outcome.reason == "insufficient_repetitions"
        assert outcome.similar_count == 2
        assert store.node_count() == 0
        assert idx.count() == 0

        # Both interactions DID land in persistence — they're available
        # to a future evaluation when the third arrives.
        assert len(p.get_interactions()) == 2

        idx.close()
        p.close()

    def test_unstable_responses_no_node(
        self,
        tmp_path: Path,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        """Three similar inputs with semantically-distinct LLM answers
        must NOT crystallize a node — the LLM was inconsistent and we
        can't pick a canonical response."""
        db = str(tmp_path / "learner_unstable.db")
        p = SQLitePersistence(db)
        store = p.load_graph()
        idx = FAISSIndex(dimension=config.embedding_dim)
        rl = ReinforcementLogger(store, p, config)
        learner = FlatNodeLearner(store, idx, embedder, p, config)

        # Three near-identical questions → input cluster passes
        rewordings = [
            "what is the weather",
            "what is the weather?",
            "what is the WEATHER",
        ]
        # Three semantically-distinct answers → response stability fails
        responses = [
            "Sunny, 72°F.",
            "Run: git add -A && git commit -m \"<msg>\".",
            "Why don't scientists trust atoms? They make up everything.",
        ]

        outcome = None
        for raw, resp in zip(rewordings, responses):
            norm = normalizer.normalize(raw)
            log = InteractionLog(
                timestamp=time.time(),
                input_text=raw,
                normalized_text=norm.normalized,
                route_decision=RouteDecision.LLM_ONLY,
                matched_node_id=None,
                llm_response=resp,
                response_text=resp,
                latency_ms=10.0,
            )
            rl.log_and_reinforce(log)
            outcome = learner.evaluate_for_learning(log)

        # Cluster reached 3, but stability check rejected it
        assert outcome is not None
        assert outcome.created_node is None
        assert outcome.reason == "responses_unstable"
        assert outcome.similar_count == 3
        assert store.node_count() == 0

        idx.close()
        p.close()


class TestLearnerCrossSession:
    """Learning history persists across close+reopen — partial
    repetitions in one session count toward the threshold in the next."""

    def test_two_in_session_1_then_third_creates_node_in_session_2(
        self,
        tmp_path: Path,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        db = str(tmp_path / "learner_xsession.db")
        canonical_response = "Paris is the capital of France."
        rewordings = [
            "what is the capital of france",
            "what is the capital of france?",
            "what is the CAPITAL of france",
        ]

        # --- Session 1: only 2 of 3 reps happen ---
        p1 = SQLitePersistence(db)
        store = p1.load_graph()
        idx = FAISSIndex(dimension=config.embedding_dim)
        rl = ReinforcementLogger(store, p1, config)
        learner = FlatNodeLearner(store, idx, embedder, p1, config)

        for raw in rewordings[:2]:
            norm = normalizer.normalize(raw)
            log = InteractionLog(
                timestamp=time.time(),
                input_text=raw,
                normalized_text=norm.normalized,
                route_decision=RouteDecision.LLM_ONLY,
                response_text=canonical_response,
                latency_ms=10.0,
            )
            rl.log_and_reinforce(log)
            outcome = learner.evaluate_for_learning(log)
            assert outcome.created_node is None  # short of threshold

        assert store.node_count() == 0
        p1.save_graph(store)
        idx.close()
        p1.close()

        # --- Session 2: third rep arrives, node should crystallize
        # using the persisted history from session 1 ---
        p2 = SQLitePersistence(db)
        store2 = p2.load_graph()
        idx2 = FAISSIndex(dimension=config.embedding_dim)
        for n in store2.all_nodes():
            idx2.add(n.pattern_id, n.embedding_vector)
        matcher2 = NodeMatcher(store2, idx2, config)
        rl2 = ReinforcementLogger(store2, p2, config)
        learner2 = FlatNodeLearner(store2, idx2, embedder, p2, config)

        # Confirm session-1 interactions are visible to the new learner
        assert len(p2.get_interactions()) == 2

        # Third rep in session 2
        raw = rewordings[2]
        norm = normalizer.normalize(raw)
        log = InteractionLog(
            timestamp=time.time(),
            input_text=raw,
            normalized_text=norm.normalized,
            route_decision=RouteDecision.LLM_ONLY,
            response_text=canonical_response,
            latency_ms=10.0,
        )
        rl2.log_and_reinforce(log)
        outcome = learner2.evaluate_for_learning(log)

        # Cross-session history made the cluster
        assert outcome.created_node is not None
        assert outcome.reason == "created"
        assert outcome.similar_count == 3
        assert store2.node_count() == 1
        assert idx2.count() == 1
        assert outcome.created_node.response == canonical_response

        # And the new node is immediately matchable
        probe_qv = embedder.embed(
            normalizer.normalize("what's the capital of france?").normalized
        )
        match = matcher2.match(probe_qv)
        assert match.node is not None
        assert match.node.pattern_id == outcome.created_node.pattern_id

        idx2.close()
        p2.close()


class TestLearnerColdToConfidentLifecycle:
    """Full cycle: learner creates a node at conf=0.5 (LLM_FALLBACK
    territory) → reinforcement boosts it through repeated graph hits →
    eventually GRAPH_DIRECT → no more LLM calls for that intent."""

    def test_learner_create_then_reinforce_to_graph_direct(
        self,
        tmp_path: Path,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        db = str(tmp_path / "learner_lifecycle.db")
        p = SQLitePersistence(db)
        store = p.load_graph()
        idx = FAISSIndex(dimension=config.embedding_dim)
        matcher = NodeMatcher(store, idx, config)
        rl = ReinforcementLogger(store, p, config)
        learner = FlatNodeLearner(store, idx, embedder, p, config)

        canonical_response = "Ibrahim"
        teaching_phrases = [
            "what is my name",
            "what is my name?",
            "what is my NAME",
        ]

        # Phase 1: 3 LLM calls teach the habit. Learner crystallizes
        # on the third with confidence=0.5.
        learned = None
        for raw in teaching_phrases:
            norm = normalizer.normalize(raw)
            qv = embedder.embed(norm.normalized)
            match = matcher.match(qv)
            log = InteractionLog(
                timestamp=time.time(),
                input_text=raw,
                normalized_text=norm.normalized,
                route_decision=match.route_decision,
                matched_node_id=(
                    match.node.pattern_id if match.node else None
                ),
                llm_response=canonical_response,
                response_text=canonical_response,
                latency_ms=10.0,
            )
            rl.log_and_reinforce(log)
            outcome = learner.evaluate_for_learning(log)
            if outcome.created_node is not None:
                learned = outcome.created_node

        assert learned is not None
        assert learned.confidence == config.learning_starting_confidence
        assert learned.stability == Stability.LOW

        # Phase 2: At conf=0.5 (below the 0.7 threshold) the matcher
        # routes LLM_FALLBACK even though the node exists. We're
        # simulating: the LLM agrees, the reinforcement logger fires
        # nothing (LLM routes don't reinforce), and we manually nudge
        # via repeated GRAPH_DIRECT-like reinforcement to model the
        # post-learner reinforcement loop.
        first_probe_qv = embedder.embed(
            normalizer.normalize("what is my name").normalized
        )
        first_match = matcher.match(first_probe_qv)
        assert first_match.node is not None
        assert first_match.node.pattern_id == learned.pattern_id
        assert first_match.route_decision == RouteDecision.LLM_FALLBACK

        # Phase 3: Drive the node above the confidence_threshold by
        # synthesizing GRAPH_DIRECT reinforcement events. This models
        # what the pipeline does once the node IS confident enough.
        # We force the route to GRAPH_DIRECT here because in reality
        # the matcher would do so once conf crosses 0.7 — but we want
        # to exercise the reinforcement path explicitly.
        for _ in range(15):
            rl.log_and_reinforce(
                InteractionLog(
                    timestamp=time.time(),
                    input_text="what is my name",
                    normalized_text="what is my name",
                    route_decision=RouteDecision.GRAPH_DIRECT,
                    matched_node_id=learned.pattern_id,
                    response_text=canonical_response,
                    latency_ms=1.0,
                )
            )

        # Confidence climbed through the boost loop to >= 0.7
        reloaded = store.get_node(learned.pattern_id)
        assert reloaded.confidence >= config.confidence_threshold
        assert reloaded.reinforcement_count == 15

        # Now matcher routes GRAPH_DIRECT
        final_qv = embedder.embed(
            normalizer.normalize("what is my name?").normalized
        )
        final_match = matcher.match(final_qv)
        assert final_match.node.pattern_id == learned.pattern_id
        assert final_match.route_decision == RouteDecision.GRAPH_DIRECT

        # Stability tier — 15 reinforcements crosses MEDIUM (5) but not HIGH (20)
        assert reloaded.stability == Stability.MEDIUM

        idx.close()
        p.close()


# =====================================================================
# SafetyBoundary end-to-end scenarios
# Wires the real matcher + real graph store + real FAISS through the
# safety boundary on real E5 embeddings to verify the four checks
# (risk, volatile, ambiguity, blocklist) override correctly.
# =====================================================================


class TestSafetyHighRiskNodeE2E:
    """A HIGH-risk node, even matched at GRAPH_DIRECT confidence,
    must be overridden to LLM_FALLBACK by the safety boundary."""

    def test_high_risk_node_matched_then_overridden(
        self,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        store = InMemoryGraphStore()
        idx = FAISSIndex(dimension=config.embedding_dim)
        matcher = NodeMatcher(store, idx, config)
        boundary = SafetyBoundary(config)

        risky = _make_learned_node(
            "delete-account",
            "delete my account",
            "Account deleted permanently.",
            embedder,
            confidence=0.95,
        )
        risky.risk_level = RiskLevel.HIGH
        store.put_node(risky)
        idx.add(risky.pattern_id, risky.embedding_vector)

        raw = "delete my account please"
        norm = normalizer.normalize(raw)
        match = matcher.match(embedder.embed(norm.normalized))
        # Matcher ranks it as a confident graph route...
        assert match.route_decision == RouteDecision.GRAPH_DIRECT
        assert match.node.pattern_id == "delete-account"

        # ...but safety vetoes
        decision = boundary.check(match, input_text=raw)
        assert decision.safe is False
        assert decision.reason == "high_risk_node"
        assert decision.override_route == RouteDecision.LLM_FALLBACK

        idx.close()


class TestSafetyVolatileNodeE2E:
    """A volatile node always escalates so the LLM produces a fresh answer."""

    def test_volatile_time_node_escalates_to_llm(
        self,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        store = InMemoryGraphStore()
        idx = FAISSIndex(dimension=config.embedding_dim)
        matcher = NodeMatcher(store, idx, config)
        boundary = SafetyBoundary(config)

        time_node = _make_learned_node(
            "tell-time",
            "what time is it",
            "Stale answer that shouldn't be served.",
            embedder,
            confidence=0.95,
        )
        time_node.volatile = True
        store.put_node(time_node)
        idx.add(time_node.pattern_id, time_node.embedding_vector)

        match = matcher.match(
            embedder.embed(normalizer.normalize("what time is it?").normalized)
        )
        assert match.route_decision == RouteDecision.GRAPH_DIRECT

        decision = boundary.check(match, input_text="what time is it?")
        assert decision.safe is False
        assert decision.reason == "volatile_node"
        assert decision.override_route == RouteDecision.LLM_FALLBACK

        idx.close()


class TestSafetyBlocklistE2E:
    """Blocklist patterns force the LLM route regardless of graph state."""

    def test_blocklist_match_overrides_graph_direct(
        self,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        cfg = CogniGraphConfig(
            embedding_dim=config.embedding_dim,
            blocklist_patterns=["password", "social security"],
        )
        store = InMemoryGraphStore()
        idx = FAISSIndex(dimension=cfg.embedding_dim)
        matcher = NodeMatcher(store, idx, cfg)
        boundary = SafetyBoundary(cfg)

        node = _make_learned_node(
            "pwd-reset",
            "how to reset my password",
            "Click 'Forgot password' on the login screen.",
            embedder,
            confidence=0.95,
        )
        store.put_node(node)
        idx.add(node.pattern_id, node.embedding_vector)

        raw = "what is my password"
        match = matcher.match(
            embedder.embed(normalizer.normalize(raw).normalized)
        )
        assert match.route_decision in (
            RouteDecision.GRAPH_DIRECT,
            RouteDecision.GRAPH_COMPOSED,
        )

        decision = boundary.check(match, input_text=raw)
        assert decision.safe is False
        assert decision.reason == "blocklist_match"
        assert decision.override_route == RouteDecision.LLM_FALLBACK
        assert boundary.block_counts["blocklist_match"] == 1

        idx.close()

    def test_blocklist_runtime_add_takes_effect_immediately(
        self,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        store = InMemoryGraphStore()
        idx = FAISSIndex(dimension=config.embedding_dim)
        matcher = NodeMatcher(store, idx, config)
        boundary = SafetyBoundary(config)

        node = _make_learned_node(
            "n", "hello world", "hi", embedder, confidence=0.95
        )
        store.put_node(node)
        idx.add(node.pattern_id, node.embedding_vector)

        match = matcher.match(
            embedder.embed(normalizer.normalize("hello world").normalized)
        )
        assert boundary.check(match, input_text="hello world").safe is True

        boundary.add_to_blocklist("hello")
        after = boundary.check(match, input_text="hello world")
        assert after.safe is False
        assert after.reason == "blocklist_match"

        idx.close()


class TestSafetyClearMatchPasses:
    """Sanity: a low-risk, non-volatile, unambiguous, non-blocklisted
    match still passes the boundary."""

    def test_safe_path_executes(
        self,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        store = InMemoryGraphStore()
        idx = FAISSIndex(dimension=config.embedding_dim)
        matcher = NodeMatcher(store, idx, config)
        boundary = SafetyBoundary(config)

        node = _make_learned_node(
            "name", "what is my name", "Ibrahim", embedder, confidence=0.95
        )
        store.put_node(node)
        idx.add(node.pattern_id, node.embedding_vector)

        unrelated = _make_learned_node(
            "math", "what is two plus two", "Four.", embedder, confidence=0.5
        )
        store.put_node(unrelated)
        idx.add(unrelated.pattern_id, unrelated.embedding_vector)

        raw = "tell me my name"
        match = matcher.match(
            embedder.embed(normalizer.normalize(raw).normalized)
        )
        assert match.node.pattern_id == "name"
        assert match.ambiguous is False

        decision = boundary.check(match, input_text=raw)
        assert decision.safe is True
        assert decision.reason is None
        assert decision.override_route is None
        assert sum(boundary.block_counts.values()) == 0

        idx.close()


class TestSafetyVolatileSurvivesPersistence:
    """B1 in real conditions: the volatile flag must survive the full
    SQLite save+reload cycle, otherwise the safety check silently
    becomes a no-op after process restart."""

    def test_volatile_node_volatile_after_reload(
        self,
        tmp_path: Path,
        normalizer: InputNormalizer,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        db = str(tmp_path / "safety_volatile.db")

        p1 = SQLitePersistence(db)
        store = p1.load_graph()
        node = _make_learned_node(
            "time", "what time is it", "stale", embedder, confidence=0.95
        )
        node.volatile = True
        store.put_node(node)
        p1.save_graph(store)
        p1.close()

        p2 = SQLitePersistence(db)
        store2 = p2.load_graph()
        idx = FAISSIndex(dimension=config.embedding_dim)
        for n in store2.all_nodes():
            idx.add(n.pattern_id, n.embedding_vector)
        matcher = NodeMatcher(store2, idx, config)
        boundary = SafetyBoundary(config)

        reloaded = store2.get_node("time")
        assert reloaded.volatile is True

        match = matcher.match(
            embedder.embed(normalizer.normalize("what time is it").normalized)
        )
        decision = boundary.check(match, input_text="what time is it")
        assert decision.safe is False
        assert decision.reason == "volatile_node"

        idx.close()
        p2.close()


# =====================================================================
# CogniGraphPipeline end-to-end scenarios
# Real E5 + real FAISS + real SQLite + the full pipeline orchestrator.
# LLM is stubbed via an injected fake-anthropic client (lookup-table)
# so tests stay deterministic without burning API tokens, but the rest
# of the system runs production code.
# =====================================================================


class _PipelineFakeUsage:
    def __init__(self) -> None:
        self.input_tokens = 10
        self.output_tokens = 20


class _PipelineFakeTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _PipelineFakeResponse:
    def __init__(self, text: str) -> None:
        self.content = [_PipelineFakeTextBlock(text)]
        self.model = "claude-fake"
        self.usage = _PipelineFakeUsage()


class _PipelineFakeMessages:
    """Lookup-table fake — answers from a substring map."""

    def __init__(self, table: dict[str, str], default: str) -> None:
        self._table = table
        self._default = default
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
        for needle, response in self._table.items():
            if needle.lower() in last_user:
                return _PipelineFakeResponse(response)
        return _PipelineFakeResponse(self._default)


class _PipelineFakeAnthropic:
    def __init__(self, table: dict[str, str], default: str = "(unknown)") -> None:
        self.messages = _PipelineFakeMessages(table, default)


def _build_pipeline(
    tmp_path: Path,
    embedder: EmbeddingService,
    config: CogniGraphConfig,
    *,
    answers: dict[str, str] | None = None,
    db_name: str = "pipeline_e2e.db",
    blocklist: list[str] | None = None,
) -> tuple[CogniGraphPipeline, SQLitePersistence, ClaudeLLMProvider]:
    """Construct a real pipeline with a fake-anthropic-backed LLM."""
    cfg = config
    if blocklist is not None:
        cfg = CogniGraphConfig(
            embedding_dim=config.embedding_dim,
            faiss_search_k=config.faiss_search_k,
            blocklist_patterns=blocklist,
        )
    fake_client = _PipelineFakeAnthropic(answers or {})
    llm = ClaudeLLMProvider(
        api_key="fake", model="claude-fake", config=cfg, client=fake_client
    )
    persistence = SQLitePersistence(str(tmp_path / db_name))
    pipeline = CogniGraphPipeline(
        config=cfg, embedder=embedder, persistence=persistence, llm=llm
    )
    return pipeline, persistence, llm


class TestPipelineColdToConfidentE2E:
    """Cold start, ask the same question 3 times, learner crystallizes
    a node, then the matcher routes around the LLM on the 4th call."""

    def test_three_repetitions_then_graph_route(
        self,
        tmp_path: Path,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        pipeline, persistence, llm = _build_pipeline(
            tmp_path, embedder, config,
            answers={"capital of france": "Paris is the capital of France."},
        )

        try:
            rewordings = [
                "what is the capital of france",
                "what is the capital of france?",
                "what is the CAPITAL of france",
            ]

            results = [pipeline.process(r) for r in rewordings]
            for r in results:
                assert r.route == RouteDecision.LLM_ONLY

            assert pipeline._graph_store.node_count() == 1

            # The next probe with similar text matches the new node.
            # learning_starting_confidence=0.5 < confidence_threshold=0.7,
            # so the matcher routes LLM_FALLBACK rather than GRAPH_DIRECT.
            probe = pipeline.process("what is the capital of france?!")
            assert probe.route == RouteDecision.LLM_FALLBACK
            assert probe.matched_node_id is not None

            stats = pipeline.get_stats()
            assert stats["total_requests"] == 4
            assert stats["llm_calls"] == 4
            assert stats["graph_hits"] == 0
            assert stats["node_count"] == 1
        finally:
            llm.close()
            persistence.close()


class TestPipelineCrossSession:
    """State persists across pipeline restart: graph nodes, FAISS
    vectors, and interaction history all reload."""

    def test_graph_persists_across_pipeline_restart(
        self,
        tmp_path: Path,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        pipeline1, p1, llm1 = _build_pipeline(
            tmp_path, embedder, config,
            answers={"name": "Ibrahim"},
            db_name="cross_session.db",
        )
        try:
            for r in (
                "what is my name",
                "what is my name?",
                "what is my NAME",
                "what is my name?!",
                "what is my Name",
                "what is my name?!?",
                "what's my name",
            ):
                pipeline1.process(r)
            # The pipeline doesn't auto-persist the graph (that's #020
            # startup/shutdown's job). Save explicitly here — this is
            # what the future shutdown hook will do.
            p1.save_graph(pipeline1._graph_store)
            pipeline1._faiss.save(str(tmp_path / "cross_session.faiss"))
            session1_node_count = pipeline1._graph_store.node_count()
            session1_log_count = len(p1.get_interactions())
            assert session1_node_count >= 1
        finally:
            llm1.close()
            p1.close()

        # Reopen — the pipeline default-builds an empty graph so we
        # explicitly inject the loaded ones (this is what #020 will own).
        p2 = SQLitePersistence(str(tmp_path / "cross_session.db"))
        store2 = p2.load_graph()
        idx2 = FAISSIndex(dimension=config.embedding_dim)
        idx2.load(str(tmp_path / "cross_session.faiss"))

        fake_client = _PipelineFakeAnthropic({"name": "Ibrahim"})
        llm2 = ClaudeLLMProvider(
            api_key="fake", model="claude-fake", config=config, client=fake_client
        )
        pipeline2 = CogniGraphPipeline(
            config=config,
            embedder=embedder,
            graph_store=store2,
            vector_index=idx2,
            persistence=p2,
            llm=llm2,
        )

        try:
            assert pipeline2._graph_store.node_count() == session1_node_count
            assert len(p2.get_interactions()) == session1_log_count

            result = pipeline2.process("what is my name?")
            assert result.matched_node_id is not None
            assert result.route in (
                RouteDecision.LLM_FALLBACK,
                RouteDecision.GRAPH_DIRECT,
            )
        finally:
            llm2.close()
            idx2.close()
            p2.close()


class TestPipelineSafetyIntegration:
    """Safety boundary intercepts a confident graph match through the
    full pipeline orchestration."""

    def test_high_risk_node_routes_around_graph(
        self,
        tmp_path: Path,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        pipeline, persistence, llm = _build_pipeline(
            tmp_path, embedder, config,
            answers={"delete": "Are you sure? This is irreversible."},
            db_name="safety_e2e.db",
        )
        try:
            risky = _make_learned_node(
                "delete-account",
                "delete my account",
                "Account deleted permanently.",
                embedder,
                confidence=0.95,
            )
            risky.risk_level = RiskLevel.HIGH
            pipeline._graph_store.put_node(risky)
            pipeline._faiss.add(risky.pattern_id, risky.embedding_vector)

            result = pipeline.process("delete my account")

            assert result.route == RouteDecision.LLM_FALLBACK
            assert result.matched_node_id == "delete-account"
            assert result.reason == "high_risk_node"
            assert "Are you sure" in result.response

            # The risky node was NOT reinforced
            after = pipeline._graph_store.get_node("delete-account")
            assert after.reinforcement_count == 0

            stats = pipeline.get_stats()
            assert stats["safety_overrides"] == 1
            assert stats["graph_hits"] == 0
            assert stats["llm_calls"] == 1
        finally:
            llm.close()
            persistence.close()

    def test_blocklist_intercepts_at_pipeline(
        self,
        tmp_path: Path,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        pipeline, persistence, llm = _build_pipeline(
            tmp_path, embedder, config,
            answers={"password": "Use the password reset link."},
            db_name="blocklist_e2e.db",
            blocklist=["password"],
        )
        try:
            result = pipeline.process("how do I reset my password")
            assert result.route in (
                RouteDecision.LLM_FALLBACK,
                RouteDecision.LLM_ONLY,
            )
            assert result.reason == "blocklist_match"

            stats = pipeline.get_stats()
            assert stats["safety_overrides"] == 1
        finally:
            llm.close()
            persistence.close()


class TestPipelineMixedTraffic:
    """Realistic session with seeded node + alternating known/novel
    queries — routing and stats track reality."""

    def test_alternating_known_and_novel(
        self,
        tmp_path: Path,
        embedder: EmbeddingService,
        config: CogniGraphConfig,
    ) -> None:
        pipeline, persistence, llm = _build_pipeline(
            tmp_path, embedder, config,
            answers={
                "novel-1": "(LLM-1)",
                "novel-2": "(LLM-2)",
            },
            db_name="mixed.db",
        )
        try:
            known = _make_learned_node(
                "name",
                "what is my name",
                "Ibrahim",
                embedder,
                confidence=0.95,
            )
            pipeline._graph_store.put_node(known)
            pipeline._faiss.add(known.pattern_id, known.embedding_vector)

            results = [
                pipeline.process("what is my name"),
                pipeline.process("tell me about novel-1 things"),
                pipeline.process("what is my name?"),
                pipeline.process("totally different novel-2 query"),
                pipeline.process("what is my NAME"),
            ]

            graph_routes = [
                r for r in results if r.route == RouteDecision.GRAPH_DIRECT
            ]
            llm_routes = [
                r for r in results if r.route in (
                    RouteDecision.LLM_ONLY, RouteDecision.LLM_FALLBACK
                )
            ]
            assert len(graph_routes) == 3
            assert len(llm_routes) == 2

            stats = pipeline.get_stats()
            assert stats["total_requests"] == 5
            assert stats["graph_hits"] == 3
            assert stats["llm_calls"] == 2
            assert stats["graph_hit_rate"] == pytest.approx(0.6)

            assert len(persistence.get_interactions()) == 5
        finally:
            llm.close()
            persistence.close()
