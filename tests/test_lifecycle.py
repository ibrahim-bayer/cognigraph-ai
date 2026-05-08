"""Tests for ApplicationLifecycle — startup, shutdown, signals, persistence."""

from __future__ import annotations

import os
import signal
import time
from pathlib import Path

import pytest

from cognigraph.config import CogniGraphConfig
from cognigraph.lifecycle import ApplicationLifecycle
from cognigraph.models import (
    HabitNode,
    InteractionLog,
    RouteDecision,
)
from cognigraph.types import EmbeddingVector, LLMResponse


DIM = 4


# --- Fakes (deterministic, no E5 / no network) ---


class _FakeEmbedder:
    """Hash-based deterministic embedder; no model load."""

    def __init__(self) -> None:
        import random
        import math
        self._random = random
        self._math = math
        self._table: dict[str, list[float]] = {}

    def register(self, text: str, vector: list[float]) -> None:
        self._table[text] = list(vector)

    def embed(self, text: str) -> EmbeddingVector:
        if text in self._table:
            return list(self._table[text])
        rng = self._random.Random(hash(text) & 0xFFFFFFFF)
        v = [rng.gauss(0, 1) for _ in range(DIM)]
        norm = self._math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v]

    def embed_batch(self, texts: list[str]) -> list[EmbeddingVector]:
        return [self.embed(t) for t in texts]


class _FakeLLM:
    def __init__(self, default: str = "(LLM answer)") -> None:
        self._default = default
        self.calls: list[dict] = []

    def generate(
        self, prompt: str, context=None, system=None, max_tokens=None
    ) -> LLMResponse:
        self.calls.append({"prompt": prompt})
        return LLMResponse(
            text=self._default, model="fake", latency_ms=1.0,
            input_tokens=1, output_tokens=1,
        )

    def close(self) -> None:
        pass


# --- Fixtures ---


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "lifecycle.db")


@pytest.fixture
def faiss_path(tmp_path: Path) -> str:
    return str(tmp_path / "lifecycle.faiss")


@pytest.fixture
def config(db_path: str, faiss_path: str) -> CogniGraphConfig:
    return CogniGraphConfig(
        embedding_dim=DIM,
        db_path=db_path,
        faiss_index_path=faiss_path,
    )


@pytest.fixture
def llm() -> _FakeLLM:
    return _FakeLLM()


def _build_lifecycle(
    config: CogniGraphConfig,
    llm: _FakeLLM,
    *,
    install_handlers: bool = False,  # default off in tests; explicit opt-in
) -> ApplicationLifecycle:
    lifecycle = ApplicationLifecycle(
        config=config, llm=llm, install_signal_handlers=install_handlers
    )
    # Inject the fake embedder by monkeypatching the pipeline's default
    # construction. The cleanest way: build the lifecycle, run startup,
    # then swap the pipeline's _embed before any process() call. But we
    # need it BEFORE startup to avoid the real model loading. Instead,
    # build a custom startup path: caller patches into pipeline before
    # use.
    return lifecycle


# --- is_first_run ---


class TestIsFirstRun:
    def test_first_run_when_db_missing(
        self, config: CogniGraphConfig, llm: _FakeLLM
    ) -> None:
        # tmp_path is empty; DB file doesn't exist
        lifecycle = ApplicationLifecycle(
            config=config, llm=llm, install_signal_handlers=False
        )
        assert lifecycle.is_first_run() is True

    def test_not_first_run_when_db_exists(
        self, config: CogniGraphConfig, llm: _FakeLLM
    ) -> None:
        # Touch the DB file
        Path(config.db_path).touch()
        lifecycle = ApplicationLifecycle(
            config=config, llm=llm, install_signal_handlers=False
        )
        assert lifecycle.is_first_run() is False


# --- Startup ---


class TestStartup:
    def test_startup_returns_pipeline(
        self, config: CogniGraphConfig, llm: _FakeLLM
    ) -> None:
        lifecycle = ApplicationLifecycle(
            config=config, llm=llm, install_signal_handlers=False
        )
        pipeline = lifecycle.startup()
        from cognigraph.pipeline import CogniGraphPipeline
        assert isinstance(pipeline, CogniGraphPipeline)
        lifecycle.shutdown()

    def test_first_run_creates_empty_graph(
        self, config: CogniGraphConfig, llm: _FakeLLM
    ) -> None:
        lifecycle = ApplicationLifecycle(
            config=config, llm=llm, install_signal_handlers=False
        )
        pipeline = lifecycle.startup()
        try:
            assert pipeline.get_stats()["node_count"] == 0
            assert pipeline.get_stats()["vector_count"] == 0
        finally:
            lifecycle.shutdown()

    def test_startup_loads_persisted_graph(
        self, config: CogniGraphConfig, llm: _FakeLLM
    ) -> None:
        # Pre-seed: build a lifecycle, add nodes, save, shut down.
        lifecycle1 = ApplicationLifecycle(
            config=config, llm=llm, install_signal_handlers=False
        )
        pipeline1 = lifecycle1.startup()
        try:
            store = pipeline1._graph_store
            store.put_node(HabitNode(
                pattern_id="seed-1",
                trigger_patterns=["t1"],
                embedding_vector=[1.0, 0.0, 0.0, 0.0],
                response="r1",
            ))
            store.put_node(HabitNode(
                pattern_id="seed-2",
                trigger_patterns=["t2"],
                embedding_vector=[0.0, 1.0, 0.0, 0.0],
                response="r2",
            ))
            pipeline1._faiss.add("seed-1", [1.0, 0.0, 0.0, 0.0])
            pipeline1._faiss.add("seed-2", [0.0, 1.0, 0.0, 0.0])
        finally:
            lifecycle1.shutdown()

        # New lifecycle, new instance — must load from disk
        llm2 = _FakeLLM()
        lifecycle2 = ApplicationLifecycle(
            config=config, llm=llm2, install_signal_handlers=False
        )
        pipeline2 = lifecycle2.startup()
        try:
            assert pipeline2.get_stats()["node_count"] == 2
            assert pipeline2.get_stats()["vector_count"] == 2
            assert pipeline2._graph_store.get_node("seed-1").response == "r1"
            assert pipeline2._graph_store.get_node("seed-2").response == "r2"
        finally:
            lifecycle2.shutdown()

    def test_double_startup_raises(
        self, config: CogniGraphConfig, llm: _FakeLLM
    ) -> None:
        lifecycle = ApplicationLifecycle(
            config=config, llm=llm, install_signal_handlers=False
        )
        lifecycle.startup()
        try:
            with pytest.raises(RuntimeError, match="already called"):
                lifecycle.startup()
        finally:
            lifecycle.shutdown()

    def test_pipeline_property_unavailable_before_startup(
        self, config: CogniGraphConfig, llm: _FakeLLM
    ) -> None:
        lifecycle = ApplicationLifecycle(
            config=config, llm=llm, install_signal_handlers=False
        )
        with pytest.raises(RuntimeError, match="not available"):
            _ = lifecycle.pipeline


# --- Shutdown ---


class TestShutdown:
    def test_shutdown_saves_graph(
        self, config: CogniGraphConfig, llm: _FakeLLM
    ) -> None:
        lifecycle = ApplicationLifecycle(
            config=config, llm=llm, install_signal_handlers=False
        )
        pipeline = lifecycle.startup()
        pipeline._graph_store.put_node(HabitNode(
            pattern_id="x",
            trigger_patterns=["t"],
            embedding_vector=[1.0, 0.0, 0.0, 0.0],
            response="r",
        ))
        lifecycle.shutdown()

        # Check the DB file actually has the node
        from cognigraph.persistence import SQLitePersistence
        p = SQLitePersistence(config.db_path)
        try:
            store = p.load_graph()
            assert store.node_count() == 1
            assert store.get_node("x").response == "r"
        finally:
            p.close()

    def test_shutdown_saves_faiss(
        self, config: CogniGraphConfig, llm: _FakeLLM
    ) -> None:
        lifecycle = ApplicationLifecycle(
            config=config, llm=llm, install_signal_handlers=False
        )
        pipeline = lifecycle.startup()
        pipeline._faiss.add("v", [1.0, 0.0, 0.0, 0.0])
        lifecycle.shutdown()

        # FAISS file should exist
        assert Path(config.faiss_index_path).exists()

    def test_shutdown_is_idempotent(
        self, config: CogniGraphConfig, llm: _FakeLLM
    ) -> None:
        lifecycle = ApplicationLifecycle(
            config=config, llm=llm, install_signal_handlers=False
        )
        lifecycle.startup()
        lifecycle.shutdown()
        lifecycle.shutdown()  # must not raise

    def test_shutdown_does_not_save_empty_faiss(
        self, config: CogniGraphConfig, llm: _FakeLLM
    ) -> None:
        """No vectors → no FAISS file written. Avoids littering empty
        files in scratch directories on first runs."""
        lifecycle = ApplicationLifecycle(
            config=config, llm=llm, install_signal_handlers=False
        )
        lifecycle.startup()
        lifecycle.shutdown()
        assert not Path(config.faiss_index_path).exists()


# --- Round-trip ---


class TestRoundTrip:
    def test_startup_shutdown_startup_preserves_state(
        self, config: CogniGraphConfig, llm: _FakeLLM
    ) -> None:
        # Session 1: seed
        l1 = ApplicationLifecycle(
            config=config, llm=llm, install_signal_handlers=False
        )
        with l1 as pipeline:
            pipeline._graph_store.put_node(HabitNode(
                pattern_id="hab",
                trigger_patterns=["x"],
                embedding_vector=[1.0, 0.0, 0.0, 0.0],
                confidence=0.85,
                response="answer",
            ))
            pipeline._faiss.add("hab", [1.0, 0.0, 0.0, 0.0])

        # Session 2: re-open
        llm2 = _FakeLLM()
        l2 = ApplicationLifecycle(
            config=config, llm=llm2, install_signal_handlers=False
        )
        with l2 as pipeline2:
            assert pipeline2._graph_store.node_count() == 1
            node = pipeline2._graph_store.get_node("hab")
            assert node.response == "answer"
            assert node.confidence == pytest.approx(0.85)
            # FAISS rehydrated
            hits = pipeline2._faiss.search([1.0, 0.0, 0.0, 0.0], k=1)
            assert hits[0][0] == "hab"


# --- Context manager ---


class TestContextManager:
    def test_context_manager_calls_shutdown(
        self, config: CogniGraphConfig, llm: _FakeLLM
    ) -> None:
        lifecycle = ApplicationLifecycle(
            config=config, llm=llm, install_signal_handlers=False
        )
        with lifecycle as pipeline:
            pipeline._graph_store.put_node(HabitNode(
                pattern_id="ctx",
                trigger_patterns=["t"],
                embedding_vector=[1.0, 0.0, 0.0, 0.0],
                response="r",
            ))

        # After __exit__, the DB should have the node
        from cognigraph.persistence import SQLitePersistence
        p = SQLitePersistence(config.db_path)
        try:
            assert p.load_graph().node_count() == 1
        finally:
            p.close()

    def test_context_manager_saves_on_exception(
        self, config: CogniGraphConfig, llm: _FakeLLM
    ) -> None:
        lifecycle = ApplicationLifecycle(
            config=config, llm=llm, install_signal_handlers=False
        )

        with pytest.raises(ValueError, match="boom"):
            with lifecycle as pipeline:
                pipeline._graph_store.put_node(HabitNode(
                    pattern_id="exc",
                    trigger_patterns=["t"],
                    embedding_vector=[1.0, 0.0, 0.0, 0.0],
                    response="r",
                ))
                raise ValueError("boom")

        # State still saved despite the exception
        from cognigraph.persistence import SQLitePersistence
        p = SQLitePersistence(config.db_path)
        try:
            assert p.load_graph().node_count() == 1
        finally:
            p.close()


# --- FAISS rebuild from graph ---


class TestFAISSRebuild:
    def test_rebuilds_faiss_when_file_missing(
        self, config: CogniGraphConfig, llm: _FakeLLM
    ) -> None:
        # Session 1: seed graph, then DELETE the FAISS file before
        # session 2 opens. Lifecycle should rebuild from graph.
        l1 = ApplicationLifecycle(
            config=config, llm=llm, install_signal_handlers=False
        )
        with l1 as pipeline:
            pipeline._graph_store.put_node(HabitNode(
                pattern_id="r",
                trigger_patterns=["t"],
                embedding_vector=[1.0, 0.0, 0.0, 0.0],
                response="x",
            ))
            pipeline._faiss.add("r", [1.0, 0.0, 0.0, 0.0])

        # Delete the FAISS file (simulate it never having existed)
        Path(config.faiss_index_path).unlink(missing_ok=True)
        Path(str(config.faiss_index_path) + ".ids.json").unlink(missing_ok=True)

        # Session 2: should rebuild FAISS from the graph
        llm2 = _FakeLLM()
        l2 = ApplicationLifecycle(
            config=config, llm=llm2, install_signal_handlers=False
        )
        with l2 as pipeline2:
            assert pipeline2._faiss.count() == 1
            hits = pipeline2._faiss.search([1.0, 0.0, 0.0, 0.0], k=1)
            assert hits[0][0] == "r"

    def test_rebuilds_faiss_when_file_corrupt(
        self, config: CogniGraphConfig, llm: _FakeLLM, tmp_path: Path
    ) -> None:
        # Seed normally
        l1 = ApplicationLifecycle(
            config=config, llm=llm, install_signal_handlers=False
        )
        with l1 as pipeline:
            pipeline._graph_store.put_node(HabitNode(
                pattern_id="r",
                trigger_patterns=["t"],
                embedding_vector=[1.0, 0.0, 0.0, 0.0],
                response="x",
            ))
            pipeline._faiss.add("r", [1.0, 0.0, 0.0, 0.0])

        # Corrupt the FAISS file
        Path(config.faiss_index_path).write_bytes(b"\x00\x00garbage\x00\x00")

        # Session 2: load fails → rebuild from graph
        llm2 = _FakeLLM()
        l2 = ApplicationLifecycle(
            config=config, llm=llm2, install_signal_handlers=False
        )
        with l2 as pipeline2:
            assert pipeline2._faiss.count() == 1


# --- Signal handling ---


class TestSignalHandling:
    def test_signal_handler_registered_and_restored(
        self, config: CogniGraphConfig, llm: _FakeLLM
    ) -> None:
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)

        lifecycle = ApplicationLifecycle(
            config=config, llm=llm, install_signal_handlers=True
        )
        lifecycle.startup()

        # Handlers replaced
        assert signal.getsignal(signal.SIGINT) != original_sigint
        assert signal.getsignal(signal.SIGTERM) != original_sigterm

        lifecycle.shutdown()

        # Originals restored
        assert signal.getsignal(signal.SIGINT) == original_sigint
        assert signal.getsignal(signal.SIGTERM) == original_sigterm

    def test_sigterm_triggers_graceful_shutdown(
        self, config: CogniGraphConfig, llm: _FakeLLM
    ) -> None:
        lifecycle = ApplicationLifecycle(
            config=config, llm=llm, install_signal_handlers=True
        )
        with pytest.raises(KeyboardInterrupt, match="SIGTERM"):
            with lifecycle as pipeline:
                pipeline._graph_store.put_node(HabitNode(
                    pattern_id="sig",
                    trigger_patterns=["t"],
                    embedding_vector=[1.0, 0.0, 0.0, 0.0],
                    response="r",
                ))
                # Synchronously deliver SIGTERM (Python 3.8+).
                signal.raise_signal(signal.SIGTERM)
                # Unreachable — handler raises KeyboardInterrupt
                raise AssertionError("signal handler did not fire")

        # State was saved despite the interrupt
        from cognigraph.persistence import SQLitePersistence
        p = SQLitePersistence(config.db_path)
        try:
            assert p.load_graph().node_count() == 1
        finally:
            p.close()
        assert lifecycle.shutdown_requested is True

    def test_sigint_triggers_graceful_shutdown(
        self, config: CogniGraphConfig, llm: _FakeLLM
    ) -> None:
        lifecycle = ApplicationLifecycle(
            config=config, llm=llm, install_signal_handlers=True
        )
        with pytest.raises(KeyboardInterrupt, match="SIGINT"):
            with lifecycle as pipeline:
                pipeline._graph_store.put_node(HabitNode(
                    pattern_id="sigint",
                    trigger_patterns=["t"],
                    embedding_vector=[1.0, 0.0, 0.0, 0.0],
                    response="r",
                ))
                signal.raise_signal(signal.SIGINT)
                raise AssertionError("unreachable")

        from cognigraph.persistence import SQLitePersistence
        p = SQLitePersistence(config.db_path)
        try:
            assert p.load_graph().node_count() == 1
        finally:
            p.close()

    def test_install_signal_handlers_false_disables(
        self, config: CogniGraphConfig, llm: _FakeLLM
    ) -> None:
        original = signal.getsignal(signal.SIGTERM)
        lifecycle = ApplicationLifecycle(
            config=config, llm=llm, install_signal_handlers=False
        )
        lifecycle.startup()
        # Handler unchanged
        assert signal.getsignal(signal.SIGTERM) == original
        lifecycle.shutdown()


# --- Drift detection ---


class TestDriftDetection:
    def test_drift_warning_when_faiss_and_graph_disagree(
        self,
        config: CogniGraphConfig,
        llm: _FakeLLM,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # Session 1: seed 2 nodes, save graph, save FAISS
        l1 = ApplicationLifecycle(
            config=config, llm=llm, install_signal_handlers=False
        )
        with l1 as pipeline:
            for i in range(2):
                v = [0.0] * DIM
                v[i] = 1.0
                pipeline._graph_store.put_node(HabitNode(
                    pattern_id=f"n{i}",
                    trigger_patterns=[f"t{i}"],
                    embedding_vector=v,
                    response=f"r{i}",
                ))
                pipeline._faiss.add(f"n{i}", v)

        # Manually corrupt: load the DB, remove one node, save back —
        # leaving FAISS with 2 vectors and graph with 1.
        from cognigraph.persistence import SQLitePersistence
        p = SQLitePersistence(config.db_path)
        try:
            store = p.load_graph()
            store.remove_node("n1")
            p.save_graph(store)
        finally:
            p.close()

        # Session 2: lifecycle detects drift, logs warning
        llm2 = _FakeLLM()
        l2 = ApplicationLifecycle(
            config=config, llm=llm2, install_signal_handlers=False
        )
        with caplog.at_level("WARNING", logger="cognigraph.lifecycle"):
            with l2 as pipeline2:
                pass

        assert any(
            "FAISS / graph drift" in record.message
            for record in caplog.records
        )
