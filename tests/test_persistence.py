"""Tests for SQLitePersistence — real SQLite files via tmp_path."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from cognigraph.exceptions import PersistenceError
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
from cognigraph.persistence import CURRENT_SCHEMA_VERSION, SQLitePersistence


# --- Fixtures ---


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "test.db")


@pytest.fixture
def persistence(db_path: str) -> SQLitePersistence:
    p = SQLitePersistence(db_path)
    yield p
    p.close()


def _make_node(
    pattern_id: str,
    response: str = "",
    confidence: float = 0.5,
    trigger_patterns: list[str] | None = None,
    embedding_vector: list[float] | None = None,
    stability: Stability = Stability.LOW,
    risk_level: RiskLevel = RiskLevel.LOW,
    response_form: ResponseForm = ResponseForm.FIXED,
) -> HabitNode:
    return HabitNode(
        pattern_id=pattern_id,
        trigger_patterns=trigger_patterns or [f"trigger-{pattern_id}"],
        embedding_vector=embedding_vector or [0.1, 0.2, 0.3],
        confidence=confidence,
        reinforcement_count=5,
        last_used_at=time.time(),
        decay_score=0.1,
        stability=stability,
        risk_level=risk_level,
        response_form=response_form,
        response=response or f"response for {pattern_id}",
    )


# --- Initialization ---


class TestInitialization:
    def test_creates_db_file(self, db_path: str) -> None:
        assert not Path(db_path).exists()
        p = SQLitePersistence(db_path)
        assert Path(db_path).exists()
        p.close()

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c" / "test.db"
        p = SQLitePersistence(str(nested))
        assert nested.exists()
        p.close()

    def test_wal_mode_enabled(self, persistence: SQLitePersistence) -> None:
        assert persistence.journal_mode().lower() == "wal"

    def test_schema_auto_created(self, persistence: SQLitePersistence) -> None:
        tables = persistence._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        names = {r[0] for r in tables}
        assert "nodes" in names
        assert "links" in names
        assert "interaction_log" in names

    def test_reopen_existing_db(self, db_path: str) -> None:
        p1 = SQLitePersistence(db_path)
        store = InMemoryGraphStore()
        store.put_node(_make_node("persistent"))
        p1.save_graph(store)
        p1.close()

        p2 = SQLitePersistence(db_path)
        loaded = p2.load_graph()
        assert loaded.node_count() == 1
        assert loaded.get_node("persistent").pattern_id == "persistent"
        p2.close()

    def test_context_manager(self, db_path: str) -> None:
        with SQLitePersistence(db_path) as p:
            assert p.journal_mode().lower() == "wal"


# --- Graph save/load round-trip ---


class TestGraphRoundTrip:
    def test_save_empty_graph(self, persistence: SQLitePersistence) -> None:
        store = InMemoryGraphStore()
        persistence.save_graph(store)
        loaded = persistence.load_graph()
        assert loaded.node_count() == 0

    def test_save_single_node(self, persistence: SQLitePersistence) -> None:
        store = InMemoryGraphStore()
        node = _make_node("a", response="hello")
        store.put_node(node)
        persistence.save_graph(store)

        loaded = persistence.load_graph()
        assert loaded.node_count() == 1
        restored = loaded.get_node("a")
        assert restored.response == "hello"
        assert restored.pattern_id == "a"

    def test_save_preserves_all_node_fields(self, persistence: SQLitePersistence) -> None:
        store = InMemoryGraphStore()
        node = _make_node(
            "full",
            trigger_patterns=["hello", "hi", "hey"],
            embedding_vector=[0.1, 0.2, 0.3, 0.4, 0.5],
            confidence=0.87,
            stability=Stability.HIGH,
            risk_level=RiskLevel.MEDIUM,
            response_form=ResponseForm.TEMPLATE,
        )
        store.put_node(node)
        persistence.save_graph(store)

        loaded = persistence.load_graph()
        restored = loaded.get_node("full")
        assert restored.trigger_patterns == ["hello", "hi", "hey"]
        assert restored.embedding_vector == [0.1, 0.2, 0.3, 0.4, 0.5]
        assert restored.confidence == 0.87
        assert restored.reinforcement_count == 5
        assert restored.stability == Stability.HIGH
        assert restored.risk_level == RiskLevel.MEDIUM
        assert restored.response_form == ResponseForm.TEMPLATE

    def test_save_multiple_nodes(self, persistence: SQLitePersistence) -> None:
        store = InMemoryGraphStore()
        for i in range(10):
            store.put_node(_make_node(f"node-{i}", confidence=i / 10))
        persistence.save_graph(store)

        loaded = persistence.load_graph()
        assert loaded.node_count() == 10
        for i in range(10):
            node = loaded.get_node(f"node-{i}")
            assert node.confidence == pytest.approx(i / 10)

    def test_save_restores_links(self, persistence: SQLitePersistence) -> None:
        store = InMemoryGraphStore()
        store.put_node(_make_node("parent"))
        store.put_node(_make_node("child1"))
        store.put_node(_make_node("child2"))
        store.add_link("parent", ChildLink(habit_id="child1", order=1))
        store.add_link("parent", ChildLink(habit_id="child2", order=2))
        persistence.save_graph(store)

        loaded = persistence.load_graph()
        children = loaded.get_children("parent")
        assert [c.habit_id for c in children] == ["child1", "child2"]
        assert loaded.get_parents("child1") == {"parent"}
        assert loaded.get_parents("child2") == {"parent"}

    def test_save_restores_conditional_links(self, persistence: SQLitePersistence) -> None:
        store = InMemoryGraphStore()
        store.put_node(_make_node("p"))
        store.put_node(_make_node("c"))
        store.add_link("p", ChildLink(habit_id="c", condition="rain", order=1))
        store.add_link("p", ChildLink(habit_id="c", condition="sun", order=2))
        persistence.save_graph(store)

        loaded = persistence.load_graph()
        children = loaded.get_children("p")
        assert len(children) == 2
        conditions = {c.condition for c in children}
        assert conditions == {"rain", "sun"}

    def test_save_restores_shared_building_block(
        self, persistence: SQLitePersistence
    ) -> None:
        """A shared child with multiple parents — core graph feature."""
        store = InMemoryGraphStore()
        store.put_node(_make_node("p1"))
        store.put_node(_make_node("p2"))
        store.put_node(_make_node("p3"))
        store.put_node(_make_node("shared"))
        store.add_link("p1", ChildLink(habit_id="shared"))
        store.add_link("p2", ChildLink(habit_id="shared"))
        store.add_link("p3", ChildLink(habit_id="shared"))
        persistence.save_graph(store)

        loaded = persistence.load_graph()
        assert loaded.get_parents("shared") == {"p1", "p2", "p3"}

    def test_round_trip_twice(self, persistence: SQLitePersistence) -> None:
        """save → load → save → load should be stable."""
        store1 = InMemoryGraphStore()
        store1.put_node(_make_node("a"))
        store1.put_node(_make_node("b"))
        store1.add_link("a", ChildLink(habit_id="b", order=1))
        persistence.save_graph(store1)

        store2 = persistence.load_graph()
        persistence.save_graph(store2)
        store3 = persistence.load_graph()

        assert store3.node_count() == 2
        assert store3.get_children("a")[0].habit_id == "b"

    def test_save_overwrites_previous(self, persistence: SQLitePersistence) -> None:
        """Each save_graph is a full snapshot — old data is replaced."""
        store1 = InMemoryGraphStore()
        store1.put_node(_make_node("old"))
        persistence.save_graph(store1)

        store2 = InMemoryGraphStore()
        store2.put_node(_make_node("new"))
        persistence.save_graph(store2)

        loaded = persistence.load_graph()
        assert loaded.node_count() == 1
        with pytest.raises(Exception):
            loaded.get_node("old")
        assert loaded.get_node("new").pattern_id == "new"

    def test_durability_across_connections(self, db_path: str) -> None:
        """Data persists across process-style connection cycles."""
        p1 = SQLitePersistence(db_path)
        store = InMemoryGraphStore()
        store.put_node(_make_node("a", response="hi"))
        store.put_node(_make_node("b", response="bye"))
        store.add_link("a", ChildLink(habit_id="b"))
        p1.save_graph(store)
        p1.close()

        p2 = SQLitePersistence(db_path)
        loaded = p2.load_graph()
        assert loaded.node_count() == 2
        assert loaded.get_node("a").response == "hi"
        assert loaded.get_children("a")[0].habit_id == "b"
        p2.close()


# --- Interaction logging ---


class TestInteractionLog:
    def test_log_single_interaction(self, persistence: SQLitePersistence) -> None:
        log = InteractionLog(
            timestamp=1000.0,
            input_text="hello",
            normalized_text="hello",
            route_decision=RouteDecision.LLM_ONLY,
            matched_node_id=None,
            llm_response="hi there",
            response_text="hi there",
            latency_ms=42.5,
        )
        persistence.log_interaction(log)

        logs = persistence.get_interactions()
        assert len(logs) == 1
        assert logs[0].input_text == "hello"
        assert logs[0].latency_ms == 42.5
        assert logs[0].route_decision == RouteDecision.LLM_ONLY

    def test_log_preserves_all_fields(self, persistence: SQLitePersistence) -> None:
        log = InteractionLog(
            timestamp=12345.6,
            input_text="what is the weather",
            normalized_text="what is the weather",
            route_decision=RouteDecision.GRAPH_DIRECT,
            matched_node_id="weather-node",
            llm_response=None,
            response_text="sunny",
            latency_ms=1.2,
        )
        persistence.log_interaction(log)

        [restored] = persistence.get_interactions()
        assert restored.timestamp == 12345.6
        assert restored.input_text == "what is the weather"
        assert restored.normalized_text == "what is the weather"
        assert restored.route_decision == RouteDecision.GRAPH_DIRECT
        assert restored.matched_node_id == "weather-node"
        assert restored.llm_response is None
        assert restored.response_text == "sunny"
        assert restored.latency_ms == 1.2

    def test_get_interactions_pagination(self, persistence: SQLitePersistence) -> None:
        for i in range(25):
            persistence.log_interaction(
                InteractionLog(
                    timestamp=float(i),
                    input_text=f"q{i}",
                    normalized_text=f"q{i}",
                    route_decision=RouteDecision.LLM_ONLY,
                    response_text=f"a{i}",
                    latency_ms=float(i),
                )
            )

        page1 = persistence.get_interactions(limit=10, offset=0)
        page2 = persistence.get_interactions(limit=10, offset=10)
        page3 = persistence.get_interactions(limit=10, offset=20)
        assert len(page1) == 10
        assert len(page2) == 10
        assert len(page3) == 5

        # Ordered by timestamp DESC
        assert page1[0].input_text == "q24"
        assert page1[-1].input_text == "q15"

    def test_get_interactions_ordered_desc(
        self, persistence: SQLitePersistence
    ) -> None:
        persistence.log_interaction(
            InteractionLog(timestamp=100.0, input_text="old", response_text="", latency_ms=0)
        )
        persistence.log_interaction(
            InteractionLog(timestamp=200.0, input_text="new", response_text="", latency_ms=0)
        )
        logs = persistence.get_interactions()
        assert logs[0].input_text == "new"
        assert logs[1].input_text == "old"

    def test_get_interactions_for_node(self, persistence: SQLitePersistence) -> None:
        persistence.log_interaction(
            InteractionLog(
                timestamp=1,
                input_text="a",
                matched_node_id="node-1",
                response_text="",
                latency_ms=0,
            )
        )
        persistence.log_interaction(
            InteractionLog(
                timestamp=2,
                input_text="b",
                matched_node_id="node-2",
                response_text="",
                latency_ms=0,
            )
        )
        persistence.log_interaction(
            InteractionLog(
                timestamp=3,
                input_text="c",
                matched_node_id="node-1",
                response_text="",
                latency_ms=0,
            )
        )

        logs = persistence.get_interactions_for_node("node-1")
        assert len(logs) == 2
        assert {log.input_text for log in logs} == {"a", "c"}

    def test_empty_log_query(self, persistence: SQLitePersistence) -> None:
        assert persistence.get_interactions() == []
        assert persistence.get_interactions_for_node("anything") == []


# --- Real-data scenarios ---


class TestRealDataScenarios:
    def test_full_workflow_with_real_db_file(self, tmp_path: Path) -> None:
        """End-to-end: populate graph, save, log interactions, close,
        reopen, verify everything intact."""
        db = str(tmp_path / "real.db")

        # --- Session 1 ---
        p1 = SQLitePersistence(db)
        store = InMemoryGraphStore()

        # Build a realistic composed habit: "commit changes"
        store.put_node(_make_node(
            "commit-root",
            response="commit workflow",
            stability=Stability.HIGH,
            confidence=0.95,
        ))
        store.put_node(_make_node("stage-files", response="git add ."))
        store.put_node(_make_node("write-msg", response="draft commit message"))
        store.put_node(_make_node("run-commit", response="git commit"))
        store.put_node(_make_node("verify", response="git status"))

        store.add_link("commit-root", ChildLink(habit_id="stage-files", order=1))
        store.add_link("commit-root", ChildLink(habit_id="write-msg", order=2))
        store.add_link("commit-root", ChildLink(habit_id="run-commit", order=3))
        store.add_link("commit-root", ChildLink(habit_id="verify", order=4))

        # Shared building block: verify is also used by deploy
        store.put_node(_make_node("deploy-root", response="deploy workflow"))
        store.add_link("deploy-root", ChildLink(habit_id="verify", order=1))

        p1.save_graph(store)

        # Log some interactions
        p1.log_interaction(InteractionLog(
            timestamp=time.time(),
            input_text="commit my changes",
            normalized_text="commit my changes",
            route_decision=RouteDecision.GRAPH_COMPOSED,
            matched_node_id="commit-root",
            response_text="staging, committing, verifying",
            latency_ms=3.2,
        ))
        p1.log_interaction(InteractionLog(
            timestamp=time.time(),
            input_text="what time is it",
            normalized_text="what time is it",
            route_decision=RouteDecision.LLM_FALLBACK,
            matched_node_id=None,
            llm_response="2:30 PM",
            response_text="2:30 PM",
            latency_ms=250.0,
        ))
        p1.close()

        # --- Session 2: fresh connection, verify persistence ---
        p2 = SQLitePersistence(db)
        loaded = p2.load_graph()

        # Graph structure intact
        assert loaded.node_count() == 6
        commit_root = loaded.get_node("commit-root")
        assert commit_root.stability == Stability.HIGH
        assert commit_root.confidence == 0.95

        # Sequence preserved in order
        children = loaded.get_children("commit-root")
        assert [c.habit_id for c in children] == [
            "stage-files", "write-msg", "run-commit", "verify"
        ]

        # Shared building block still works
        assert loaded.get_parents("verify") == {"commit-root", "deploy-root"}

        # Interaction log intact
        logs = p2.get_interactions()
        assert len(logs) == 2
        commit_logs = p2.get_interactions_for_node("commit-root")
        assert len(commit_logs) == 1
        assert commit_logs[0].input_text == "commit my changes"

        p2.close()

        # Verify actual file on disk
        assert Path(db).exists()
        assert Path(db).stat().st_size > 0

    def test_large_graph_save_load(self, persistence: SQLitePersistence) -> None:
        """Save/load a graph with 500 nodes and 1000 links."""
        store = InMemoryGraphStore()
        for i in range(500):
            store.put_node(_make_node(f"n{i}", confidence=i / 500))

        # Create a chain: n0 -> n1 -> n2 -> ...
        for i in range(499):
            store.add_link(f"n{i}", ChildLink(habit_id=f"n{i+1}", order=0))

        # Add fan-out links from n0
        for i in range(100, 600):
            if i < 500:
                store.add_link("n0", ChildLink(habit_id=f"n{i}", order=i))

        persistence.save_graph(store)
        loaded = persistence.load_graph()

        assert loaded.node_count() == 500
        # Chain preserved
        for i in range(499):
            children = loaded.get_children(f"n{i}")
            assert any(c.habit_id == f"n{i+1}" for c in children)

        # Fan-out preserved
        n0_children = loaded.get_children("n0")
        assert len(n0_children) >= 400

    def test_sqlite_file_is_valid(self, db_path: str) -> None:
        """Verify we produce a valid SQLite file that other tools can read."""
        p = SQLitePersistence(db_path)
        store = InMemoryGraphStore()
        store.put_node(_make_node("test"))
        p.save_graph(store)
        p.close()

        # Open with raw sqlite3 — no helpers, prove it's a real DB
        raw = sqlite3.connect(db_path)
        try:
            row = raw.execute("SELECT COUNT(*) FROM nodes").fetchone()
            assert row[0] == 1
            row = raw.execute("SELECT pattern_id FROM nodes").fetchone()
            assert row[0] == "test"
        finally:
            raw.close()


# --- Error handling ---


class TestErrorHandling:
    def test_corrupt_node_json_raises(self, persistence: SQLitePersistence) -> None:
        with persistence._conn:
            persistence._conn.execute(
                "INSERT INTO nodes (pattern_id, data, created_at, updated_at) "
                "VALUES (?, ?, ?, ?)",
                ("broken", "not valid json {{{", time.time(), time.time()),
            )
        with pytest.raises(PersistenceError):
            persistence.load_graph()

    def test_invalid_route_decision_falls_back(
        self, persistence: SQLitePersistence
    ) -> None:
        """Stale route_decision values in the log should degrade gracefully."""
        with persistence._conn:
            persistence._conn.execute(
                "INSERT INTO interaction_log "
                "(timestamp, input_text, normalized_text, route_decision, "
                " response_text, latency_ms) VALUES (?, ?, ?, ?, ?, ?)",
                (1.0, "x", "x", "NONEXISTENT_ROUTE", "y", 0.0),
            )
        logs = persistence.get_interactions()
        assert len(logs) == 1
        assert logs[0].route_decision == RouteDecision.LLM_ONLY  # safe default

    def test_foreign_key_enforcement(self, persistence: SQLitePersistence) -> None:
        """Links pointing at nonexistent nodes must be rejected by FK constraints."""
        with pytest.raises(sqlite3.IntegrityError):
            with persistence._conn:
                persistence._conn.execute(
                    "INSERT INTO links (parent_id, child_id, condition, link_order) "
                    "VALUES (?, ?, ?, ?)",
                    ("ghost-parent", "ghost-child", None, 0),
                )


# --- Single-source-of-truth for links ---


class TestLinksSingleSourceOfTruth:
    def test_node_json_has_no_embedded_links(
        self, persistence: SQLitePersistence
    ) -> None:
        """Persisted node JSON must NOT carry its own children/parents list.

        The links table is the single source of truth; duplicating the data
        in the node blob risks drift on future refactors.
        """
        store = InMemoryGraphStore()
        store.put_node(HabitNode(pattern_id="p", response="parent"))
        store.put_node(HabitNode(pattern_id="c", response="child"))
        store.add_link("p", ChildLink(habit_id="c"))
        persistence.save_graph(store)

        row = persistence._conn.execute(
            "SELECT data FROM nodes WHERE pattern_id = ?", ("p",)
        ).fetchone()
        data = json.loads(row["data"])
        assert "children" not in data
        assert "parents" not in data

    def test_load_does_not_double_count_links(
        self, persistence: SQLitePersistence
    ) -> None:
        """Regression: links should appear exactly once after round-trip,
        even though HabitNode has legacy children/parents fields."""
        store = InMemoryGraphStore()
        store.put_node(HabitNode(pattern_id="p"))
        store.put_node(HabitNode(pattern_id="c"))
        store.add_link("p", ChildLink(habit_id="c", condition="rain", order=1))
        persistence.save_graph(store)

        loaded = persistence.load_graph()
        children = loaded.get_children("p")
        assert len(children) == 1
        assert children[0].condition == "rain"


# --- Schema versioning ---


class TestSchemaVersion:
    def test_fresh_db_gets_current_version(self, persistence: SQLitePersistence) -> None:
        assert persistence.schema_version() == CURRENT_SCHEMA_VERSION

    def test_rejects_newer_schema(self, db_path: str) -> None:
        p = SQLitePersistence(db_path)
        p._conn.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION + 1}")
        p._conn.commit()
        p.close()

        with pytest.raises(PersistenceError, match="newer than supported"):
            SQLitePersistence(db_path)

    def test_rejects_older_schema_without_migration(self, db_path: str) -> None:
        p = SQLitePersistence(db_path)
        # Only meaningful once CURRENT_SCHEMA_VERSION > 1; skip otherwise
        if CURRENT_SCHEMA_VERSION == 1:
            p.close()
            return
        p._conn.execute(f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION - 1}")
        p._conn.commit()
        p.close()

        with pytest.raises(PersistenceError, match="migration"):
            SQLitePersistence(db_path)


# --- Concurrency ---


class TestConcurrency:
    def test_wal_reader_sees_consistent_snapshot(self, db_path: str) -> None:
        """Under WAL, a reader on a separate connection should see the
        pre-save snapshot until save_graph commits."""
        writer = SQLitePersistence(db_path)
        store = InMemoryGraphStore()
        store.put_node(HabitNode(pattern_id="a"))
        writer.save_graph(store)

        # Reader opens its own connection
        reader = SQLitePersistence(db_path)
        loaded_before = reader.load_graph()
        assert loaded_before.node_count() == 1

        # Writer updates
        store.put_node(HabitNode(pattern_id="b"))
        store.put_node(HabitNode(pattern_id="c"))
        writer.save_graph(store)

        # Reader sees the new snapshot after writer commits
        loaded_after = reader.load_graph()
        assert loaded_after.node_count() == 3

        writer.close()
        reader.close()

    def test_thread_safe_shared_instance(self, persistence: SQLitePersistence) -> None:
        """A single instance shared between a reader thread and writer thread
        must not raise thread-safety errors."""
        store = InMemoryGraphStore()
        for i in range(20):
            store.put_node(HabitNode(pattern_id=f"n{i}"))
        persistence.save_graph(store)

        errors: list[Exception] = []
        stop = threading.Event()

        def reader() -> None:
            try:
                while not stop.is_set():
                    s = persistence.load_graph()
                    assert s.node_count() >= 20
            except Exception as e:
                errors.append(e)

        def writer() -> None:
            try:
                for i in range(20, 40):
                    store.put_node(HabitNode(pattern_id=f"n{i}"))
                    persistence.save_graph(store)
                    persistence.log_interaction(
                        InteractionLog(
                            timestamp=float(i),
                            input_text=f"q{i}",
                            normalized_text=f"q{i}",
                            route_decision=RouteDecision.LLM_ONLY,
                            response_text="",
                            latency_ms=0.0,
                        )
                    )
            except Exception as e:
                errors.append(e)

        r = threading.Thread(target=reader)
        w = threading.Thread(target=writer)
        r.start()
        w.start()
        w.join(timeout=10)
        stop.set()
        r.join(timeout=10)

        assert errors == [], f"threaded access raised: {errors}"
        final = persistence.load_graph()
        assert final.node_count() == 40
        assert len(persistence.get_interactions()) == 20


# --- Unicode / large payloads ---


class TestUnicodeAndLargePayloads:
    def test_unicode_round_trip(self, persistence: SQLitePersistence) -> None:
        """Emoji, CJK, and combining characters must round-trip through
        JSON-in-SQLite-TEXT without corruption."""
        store = InMemoryGraphStore()
        node = HabitNode(
            pattern_id="unicode-node",
            trigger_patterns=["héllo", "こんにちは", "🚀", "e\u0301"],
            response="مرحبا 🌍 café",
        )
        store.put_node(node)
        persistence.save_graph(store)

        loaded = persistence.load_graph()
        restored = loaded.get_node("unicode-node")
        assert restored.trigger_patterns == ["héllo", "こんにちは", "🚀", "e\u0301"]
        assert restored.response == "مرحبا 🌍 café"

    def test_unicode_interaction_log(self, persistence: SQLitePersistence) -> None:
        persistence.log_interaction(
            InteractionLog(
                timestamp=1.0,
                input_text="¿qué hora es? 🕒",
                normalized_text="¿qué hora es? 🕒",
                route_decision=RouteDecision.LLM_ONLY,
                response_text="son las 3 ⏰",
                latency_ms=1.0,
            )
        )
        [log] = persistence.get_interactions()
        assert log.input_text == "¿qué hora es? 🕒"
        assert log.response_text == "son las 3 ⏰"

    def test_large_embedding_vector(self, persistence: SQLitePersistence) -> None:
        """A 1536-dim embedding (production-scale) must round-trip cleanly."""
        store = InMemoryGraphStore()
        big_vec = [i / 1536.0 for i in range(1536)]
        store.put_node(
            HabitNode(pattern_id="big", embedding_vector=big_vec)
        )
        persistence.save_graph(store)
        loaded = persistence.load_graph()
        assert len(loaded.get_node("big").embedding_vector) == 1536
        assert loaded.get_node("big").embedding_vector[-1] == pytest.approx(
            1535 / 1536.0
        )
