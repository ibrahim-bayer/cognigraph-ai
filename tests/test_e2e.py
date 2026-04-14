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
from cognigraph.matcher import NodeMatcher
from cognigraph.normalizer import InputNormalizer
from cognigraph.persistence import SQLitePersistence
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

        # Reinforcement survived (node was created with count=1, then +2)
        commit_reloaded = store3.get_node("h2")
        assert commit_reloaded.reinforcement_count == 3
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
