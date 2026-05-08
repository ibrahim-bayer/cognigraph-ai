"""Application lifecycle — load on startup, save on shutdown, signal-safe.

`ApplicationLifecycle` is the outer orchestrator. It owns the construction
sequence (load persistence → load FAISS → build pipeline with the loaded
stores), the shutdown sequence (save graph → save FAISS → close
components), and signal handlers for SIGINT/SIGTERM so a Ctrl-C or a
container shutdown writes state cleanly before the process exits.

Typical usage:

    with ApplicationLifecycle(config).startup() as pipeline:
        while True:
            user_input = input("> ")
            result = pipeline.process(user_input)
            print(result.response)

The context-manager form auto-saves on normal exit AND on
KeyboardInterrupt. SIGTERM (no KeyboardInterrupt by default) is
intercepted explicitly.

For programmatic use without a context manager:

    lifecycle = ApplicationLifecycle(config)
    pipeline = lifecycle.startup()
    try:
        ...
    finally:
        lifecycle.shutdown()
"""

from __future__ import annotations

import logging
import signal
from pathlib import Path
from typing import Any

from cognigraph.config import CogniGraphConfig
from cognigraph.graph_store import InMemoryGraphStore
from cognigraph.persistence import SQLitePersistence
from cognigraph.pipeline import CogniGraphPipeline
from cognigraph.protocols import LLMProvider
from cognigraph.vector_index import FAISSIndex

logger = logging.getLogger(__name__)


# Signals we handle. SIGINT normally raises KeyboardInterrupt — but on
# some platforms / under some runtimes it doesn't reach the main thread
# reliably, and we also want SIGTERM (containers, init systems) to
# trigger the same cleanup.
_HANDLED_SIGNALS: tuple[int, ...] = (signal.SIGINT, signal.SIGTERM)


class ApplicationLifecycle:
    """Owns the load-on-startup / save-on-shutdown lifecycle.

    Constructs the pipeline with persisted state, registers signal
    handlers, and provides a context-manager API for clean teardown.

    Thread safety: `startup()` and `shutdown()` are not concurrent-safe
    with each other or with themselves. Call from a single thread.
    Signal handlers ARE called from the main thread by Python's signal
    module — no extra synchronization needed for the typical case.
    """

    def __init__(
        self,
        config: CogniGraphConfig | None = None,
        *,
        llm: LLMProvider | None = None,
        install_signal_handlers: bool = True,
    ) -> None:
        """Build the lifecycle wrapper.

        Args:
            config: configuration (defaults to CogniGraphConfig()).
            llm: optional LLMProvider. If None, the pipeline will
                default-construct ClaudeLLMProvider (which requires
                ANTHROPIC_API_KEY in env). Pass an injected fake for
                offline tests.
            install_signal_handlers: when True (default), SIGINT and
                SIGTERM are handled by calling shutdown() then re-raising
                KeyboardInterrupt. Set False in tests that want to
                simulate signals manually, or in embedded contexts where
                the host process owns signal handling.
        """
        self._config = config or CogniGraphConfig()
        self._llm = llm
        self._install_signal_handlers = install_signal_handlers

        self._persistence: SQLitePersistence | None = None
        self._faiss: FAISSIndex | None = None
        self._graph_store: InMemoryGraphStore | None = None
        self._pipeline: CogniGraphPipeline | None = None

        self._started: bool = False
        self._shutdown_complete: bool = False
        self._shutdown_requested: bool = False
        self._original_handlers: dict[int, Any] = {}

    # --- Introspection ---

    def is_first_run(self) -> bool:
        """True iff the configured DB file doesn't yet exist on disk.

        Useful for first-run UI ("welcome, your graph will start empty")
        vs subsequent-run UI ("welcomed back, N habits loaded").
        """
        return not Path(self._config.db_path).exists()

    @property
    def shutdown_requested(self) -> bool:
        """True if a signal handler has fired but shutdown() may not have
        completed yet. Useful for REPL loops to break out of input()."""
        return self._shutdown_requested

    @property
    def pipeline(self) -> CogniGraphPipeline:
        """The active pipeline, only valid between startup() and shutdown()."""
        if self._pipeline is None:
            raise RuntimeError(
                "pipeline is not available — call startup() first "
                "(or you've already shut down)"
            )
        return self._pipeline

    # --- Lifecycle ---

    def startup(self) -> CogniGraphPipeline:
        """Load persisted state, build the pipeline, register signals.

        Returns the constructed pipeline. Idempotent within a single
        instance — calling startup() twice raises.
        """
        if self._started:
            raise RuntimeError("startup() already called on this instance")
        self._started = True
        first_run = self.is_first_run()

        try:
            self._persistence = SQLitePersistence(self._config.db_path)
            self._graph_store = self._persistence.load_graph()

            self._faiss = FAISSIndex(dimension=self._config.embedding_dim)
            faiss_path = self._config.faiss_index_path
            if Path(faiss_path).exists():
                try:
                    self._faiss.load(faiss_path)
                    self._verify_faiss_graph_consistency()
                except Exception:
                    logger.exception(
                        "FAISS load failed; rebuilding from graph store"
                    )
                    self._rebuild_faiss_from_graph()
            else:
                # First run, or FAISS file was deleted. Rebuild from
                # graph store contents (empty on first run).
                self._rebuild_faiss_from_graph()

            self._pipeline = CogniGraphPipeline(
                config=self._config,
                graph_store=self._graph_store,
                vector_index=self._faiss,
                persistence=self._persistence,
                llm=self._llm,
            )
        except BaseException:
            # If construction failed partway, clean up whatever we built
            self._cleanup_partial_construction()
            raise

        if self._install_signal_handlers:
            self._install_handlers()

        node_count = self._graph_store.node_count()
        logger.info(
            "lifecycle startup: first_run=%s nodes=%d vectors=%d",
            first_run, node_count, self._faiss.count(),
        )
        return self._pipeline

    def shutdown(self) -> None:
        """Save persisted state, close components, restore signal handlers.

        Idempotent — safe to call from a signal handler AND from a
        context manager exit on the same run. Subsequent calls are no-ops.
        """
        if self._shutdown_complete:
            return
        self._shutdown_complete = True

        # Snapshot what we need; the order of operations matters for
        # resource correctness.
        persistence = self._persistence
        faiss = self._faiss
        store = self._graph_store
        pipeline = self._pipeline

        node_count = store.node_count() if store is not None else 0
        vector_count = faiss.count() if faiss is not None else 0

        # 1. Save graph snapshot to SQLite (atomic — handled by persistence).
        if persistence is not None and store is not None:
            try:
                persistence.save_graph(store)
            except Exception:
                logger.exception("save_graph failed during shutdown")

        # 2. Save FAISS index to disk (also atomic).
        if faiss is not None and faiss.count() > 0:
            try:
                faiss.save(self._config.faiss_index_path)
            except Exception:
                logger.exception("FAISS save failed during shutdown")

        # 3. Close the pipeline (tears down its owned LLM if any). Also
        # tears down whatever components the pipeline default-built;
        # ours were injected so we close them ourselves below.
        if pipeline is not None:
            try:
                pipeline.close()
            except Exception:
                logger.exception("pipeline.close() failed during shutdown")

        # 4. Close the FAISS index and persistence we own.
        if faiss is not None:
            try:
                faiss.close()
            except Exception:
                logger.exception("faiss.close() failed")
        if persistence is not None:
            try:
                persistence.close()
            except Exception:
                logger.exception("persistence.close() failed")

        # 5. Restore original signal handlers.
        if self._install_signal_handlers:
            self._restore_handlers()

        logger.info(
            "lifecycle shutdown: saved %d nodes / %d vectors",
            node_count, vector_count,
        )

    # --- Context manager ---

    def __enter__(self) -> CogniGraphPipeline:
        return self.startup()

    def __exit__(self, *args: object) -> None:
        self.shutdown()

    # --- Signal handling ---

    def _install_handlers(self) -> None:
        """Register SIGINT/SIGTERM handlers, save originals for restore."""
        for sig in _HANDLED_SIGNALS:
            try:
                self._original_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._handle_signal)
            except (ValueError, OSError) as e:
                # Some runtimes (e.g., non-main-thread Python) don't allow
                # signal handler registration. Continue without it.
                logger.debug(
                    "could not register handler for signal %d: %s", sig, e
                )

    def _restore_handlers(self) -> None:
        """Restore signal handlers to whatever they were before startup()."""
        for sig, original in self._original_handlers.items():
            try:
                signal.signal(sig, original)
            except (ValueError, OSError):
                logger.debug("could not restore handler for signal %d", sig)
        self._original_handlers = {}

    def _handle_signal(self, sig: int, frame: Any) -> None:
        """Called when SIGINT or SIGTERM fires. Save state then re-raise.

        Re-raises as KeyboardInterrupt so the application's try/except
        or context-manager exit flows fire — the caller chooses whether
        to exit, log, or recover.
        """
        sig_name = signal.Signals(sig).name if sig in signal.Signals.__members__.values() else str(sig)
        logger.info("received signal %s; shutting down gracefully", sig_name)
        self._shutdown_requested = True
        try:
            self.shutdown()
        except Exception:
            logger.exception("shutdown failed inside signal handler")
        raise KeyboardInterrupt(f"signal {sig_name} received")

    # --- Internals ---

    def _verify_faiss_graph_consistency(self) -> None:
        """Warn if the loaded FAISS index and graph store disagree on count.

        Drift can happen if the process crashed between save_graph() and
        faiss.save(). Not fatal — the matcher's stale_hit_count will
        surface individual mismatches at request time.
        """
        store = self._graph_store
        faiss = self._faiss
        if store is None or faiss is None:
            return
        if faiss.count() != store.node_count():
            logger.warning(
                "FAISS / graph drift on startup: faiss=%d nodes=%d. "
                "Matcher will skip stale hits at request time.",
                faiss.count(), store.node_count(),
            )

    def _rebuild_faiss_from_graph(self) -> None:
        """Populate FAISS by replaying every node's stored embedding."""
        if self._graph_store is None or self._faiss is None:
            return
        for node in self._graph_store.all_nodes():
            if node.embedding_vector:
                self._faiss.add(node.pattern_id, node.embedding_vector)
        if self._graph_store.node_count() > 0:
            logger.info(
                "rebuilt FAISS from graph: %d vectors",
                self._faiss.count(),
            )

    def _cleanup_partial_construction(self) -> None:
        """Best-effort close of whatever was constructed before a startup() failure."""
        for resource_attr in ("_pipeline", "_faiss", "_persistence"):
            resource = getattr(self, resource_attr, None)
            if resource is None:
                continue
            close_fn = getattr(resource, "close", None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception:
                    logger.debug(
                        "close failed during partial-construction cleanup",
                        exc_info=True,
                    )
        self._pipeline = None
        self._faiss = None
        self._persistence = None
        self._graph_store = None
