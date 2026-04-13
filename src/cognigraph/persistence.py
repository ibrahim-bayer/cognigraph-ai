"""SQLite persistence for graph nodes, links, and interaction logs.

Background-only — never on the hot path. The in-memory graph store is the
source of truth during operation; SQLite provides durability across restarts.

Threading model:
    Each SQLitePersistence instance owns one connection. Connections are
    configured with check_same_thread=False and all mutating operations are
    guarded by an internal lock, so it is safe to share a single instance
    between a main thread (reads) and a background thread (writes). For
    multi-thread high-concurrency workloads, prefer one instance per thread.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from cognigraph.exceptions import PersistenceError
from cognigraph.graph_store import InMemoryGraphStore
from cognigraph.models import ChildLink, HabitNode, InteractionLog, RouteDecision


# Bump when the schema changes. Reject DBs newer than this; treat 0 as fresh.
# Spec (issue #7) specifies links PK (parent_id, child_id); we use a surrogate
# id + UNIQUE(parent, child, condition, order) so conditional / order variants
# coexist without NULL-in-PK gotchas.
CURRENT_SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    pattern_id TEXT PRIMARY KEY,
    data TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS links (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_id TEXT NOT NULL,
    child_id TEXT NOT NULL,
    condition TEXT,
    link_order INTEGER NOT NULL DEFAULT 0,
    UNIQUE (parent_id, child_id, condition, link_order),
    FOREIGN KEY (parent_id) REFERENCES nodes(pattern_id) ON DELETE CASCADE,
    FOREIGN KEY (child_id) REFERENCES nodes(pattern_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_links_parent ON links(parent_id);
CREATE INDEX IF NOT EXISTS idx_links_child ON links(child_id);

CREATE TABLE IF NOT EXISTS interaction_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    input_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    route_decision TEXT NOT NULL,
    matched_node_id TEXT,
    llm_response TEXT,
    response_text TEXT NOT NULL,
    latency_ms REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_log_timestamp ON interaction_log(timestamp);
CREATE INDEX IF NOT EXISTS idx_log_node ON interaction_log(matched_node_id);
"""


# Fields on HabitNode that are persisted separately in the `links` table.
# Strip from the node JSON blob so the edges have a single source of truth
# (the links table) and can't drift out of sync with the nodes table.
_LINK_FIELDS_STRIPPED_FROM_NODE_JSON = ("children", "parents")


class SQLitePersistence:
    """SQLite-backed persistence for graph state and interaction logs.

    WAL mode is enabled so background writes don't block concurrent reads.
    See module docstring for the threading model.
    """

    def __init__(self, db_path: str = "cognigraph.db") -> None:
        # TODO(W4): db_path is treated as trusted. If ever sourced from
        # untrusted input, resolve and reject paths outside an allowed base.
        self.db_path = db_path
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()

        self._conn.execute("PRAGMA foreign_keys = ON")
        # TODO(W2): capture and log if PRAGMA journal_mode doesn't actually
        # land on WAL (network FS, conflicting opens). Readers would block
        # writers in that case, silently.
        self._conn.execute("PRAGMA journal_mode = WAL")

        self._init_schema()

    def _init_schema(self) -> None:
        try:
            with self._lock, self._conn:
                version_row = self._conn.execute("PRAGMA user_version").fetchone()
                version = version_row[0] if version_row else 0

                if version == 0:
                    self._conn.executescript(_SCHEMA)
                    self._conn.execute(
                        f"PRAGMA user_version = {CURRENT_SCHEMA_VERSION}"
                    )
                elif version > CURRENT_SCHEMA_VERSION:
                    raise PersistenceError(
                        f"DB schema version {version} is newer than supported "
                        f"version {CURRENT_SCHEMA_VERSION}; upgrade cognigraph"
                    )
                elif version < CURRENT_SCHEMA_VERSION:
                    raise PersistenceError(
                        f"DB schema version {version} requires migration to "
                        f"version {CURRENT_SCHEMA_VERSION}; migrations are not "
                        f"yet implemented"
                    )
                # version == CURRENT_SCHEMA_VERSION: nothing to do
        except sqlite3.Error as e:
            raise PersistenceError(f"Failed to initialize schema: {e}") from e

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> SQLitePersistence:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # --- Graph save/load ---

    def save_graph(self, store: InMemoryGraphStore) -> None:
        """Persist all nodes and links from the graph store.

        Atomic: wipes existing nodes/links and writes the full snapshot in
        one transaction. The in-memory store is authoritative.
        """
        # TODO(W5): preserve original created_at instead of stamping now.
        # Requires HabitNode.created_at field (not yet in the model).
        now = time.time()
        try:
            with self._lock, self._conn:
                self._conn.execute("DELETE FROM links")
                self._conn.execute("DELETE FROM nodes")

                nodes = store.all_nodes()
                node_rows = [
                    (n.pattern_id, json.dumps(self._node_to_persist_dict(n)), now, now)
                    for n in nodes
                ]
                self._conn.executemany(
                    "INSERT INTO nodes (pattern_id, data, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?)",
                    node_rows,
                )

                link_rows = []
                for node in nodes:
                    for child in store.get_children(node.pattern_id):
                        link_rows.append(
                            (node.pattern_id, child.habit_id, child.condition, child.order)
                        )
                if link_rows:
                    self._conn.executemany(
                        "INSERT INTO links (parent_id, child_id, condition, link_order) "
                        "VALUES (?, ?, ?, ?)",
                        link_rows,
                    )
        except sqlite3.Error as e:
            raise PersistenceError(f"save_graph failed: {e}") from e

    def load_graph(self) -> InMemoryGraphStore:
        """Load nodes and links into a fresh graph store.

        Both SELECTs run inside one deferred transaction so nodes and links
        are read from a single consistent snapshot (matters under WAL when
        a background writer may commit mid-load).
        """
        store = InMemoryGraphStore()
        try:
            with self._lock:
                self._conn.execute("BEGIN")
                try:
                    node_rows = self._conn.execute(
                        "SELECT pattern_id, data FROM nodes"
                    ).fetchall()
                    link_rows = self._conn.execute(
                        "SELECT parent_id, child_id, condition, link_order FROM links"
                    ).fetchall()
                finally:
                    self._conn.execute("COMMIT")
        except sqlite3.Error as e:
            raise PersistenceError(f"load_graph failed: {e}") from e

        try:
            for row in node_rows:
                data = json.loads(row["data"])
                store.put_node(HabitNode.from_dict(data))

            for row in link_rows:
                link = ChildLink(
                    habit_id=row["child_id"],
                    condition=row["condition"],
                    order=row["link_order"],
                )
                store.add_link(row["parent_id"], link)
        except json.JSONDecodeError as e:
            raise PersistenceError(f"load_graph: corrupt node JSON: {e}") from e

        return store

    @staticmethod
    def _node_to_persist_dict(node: HabitNode) -> dict:
        """Serialize a node, stripping link fields that live in `links` table."""
        data = node.to_dict()
        for field in _LINK_FIELDS_STRIPPED_FROM_NODE_JSON:
            data.pop(field, None)
        return data

    # --- Interaction log ---

    def log_interaction(self, log: InteractionLog) -> None:
        # TODO(N1): each interaction is its own transaction / fsync. Batch
        # if QPS becomes a bottleneck.
        try:
            with self._lock, self._conn:
                self._conn.execute(
                    "INSERT INTO interaction_log "
                    "(timestamp, input_text, normalized_text, route_decision, "
                    " matched_node_id, llm_response, response_text, latency_ms) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        log.timestamp,
                        log.input_text,
                        log.normalized_text,
                        log.route_decision.value,
                        log.matched_node_id,
                        log.llm_response,
                        log.response_text,
                        log.latency_ms,
                    ),
                )
        except sqlite3.Error as e:
            raise PersistenceError(f"log_interaction failed: {e}") from e

    def get_interactions(
        self, limit: int = 100, offset: int = 0
    ) -> list[InteractionLog]:
        # TODO(N2): enumerate columns explicitly for forward-compat when
        # schema evolves.
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT * FROM interaction_log "
                    "ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
        except sqlite3.Error as e:
            raise PersistenceError(f"get_interactions failed: {e}") from e
        return [self._row_to_log(r) for r in rows]

    def get_interactions_for_node(self, node_id: str) -> list[InteractionLog]:
        try:
            with self._lock:
                rows = self._conn.execute(
                    "SELECT * FROM interaction_log WHERE matched_node_id = ? "
                    "ORDER BY timestamp DESC",
                    (node_id,),
                ).fetchall()
        except sqlite3.Error as e:
            raise PersistenceError(f"get_interactions_for_node failed: {e}") from e
        return [self._row_to_log(r) for r in rows]

    @staticmethod
    def _row_to_log(row: sqlite3.Row) -> InteractionLog:
        try:
            route = RouteDecision(row["route_decision"])
        except ValueError:
            route = RouteDecision.LLM_ONLY
        return InteractionLog(
            timestamp=row["timestamp"],
            input_text=row["input_text"],
            normalized_text=row["normalized_text"],
            route_decision=route,
            matched_node_id=row["matched_node_id"],
            llm_response=row["llm_response"],
            response_text=row["response_text"],
            latency_ms=row["latency_ms"],
        )

    # --- Introspection (for tests and diagnostics) ---

    def journal_mode(self) -> str:
        with self._lock:
            row = self._conn.execute("PRAGMA journal_mode").fetchone()
        return row[0] if row else ""

    def schema_version(self) -> int:
        with self._lock:
            row = self._conn.execute("PRAGMA user_version").fetchone()
        return row[0] if row else 0
