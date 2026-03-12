# ITD: Language and Stack Selection

**Date:** 2026-03-11
**Status:** Accepted
**Related:** [ITD_2026-03-11_graph-first-architecture](ITD_2026-03-11_graph-first-architecture.md)
**Decision:** Python 3.12+ with uv, FAISS, in-memory graph + SQLite persistence, sentence-transformers, Claude API

## Context

LLM-Habit is a human-like cognitive agent where a learned graph (nervous system) is the primary system (~95% of decisions) and an LLM (brain) is the fallback for novel situations (~5%). The graph starts empty, grows from experience, composes skills from linked nodes, and forgets what it doesn't use.

The stack must serve three workloads:

- **Graph traversal** (primary path, ~95%): Find matching node by embedding similarity → follow links through composed sequences → execute or escalate. This is the nervous system — reflexes do not go to disk. Must be sub-millisecond.
- **LLM fallback** (~5%): API call to Claude when the graph can't handle a request. Latency is the API's problem.
- **Learning loop** (background): Observe repeated patterns, detect sequences, create nodes, form links, decay unused nodes. Not latency-sensitive.

## Decision

**Python 3.12+** as the primary language, with the following stack:

| Component | Choice | Rationale |
|---|---|---|
| Language | Python 3.12+ | Best ML/embedding ecosystem, largest contributor pool, fastest to MVP |
| Package manager | uv | Fast, modern, replaces pip/poetry/venv |
| Embedding model | sentence-transformers (E5-Small) | Local inference, no API cost, 118M params, 384 dims |
| Vector search | FAISS | In-process, in-memory, fast for up to 500K+ nodes |
| Graph runtime | In-memory Python dicts + dataclasses | Sub-microsecond lookups, zero serialization, zero I/O on hot path |
| Graph persistence | SQLite (WAL mode) | Periodic snapshots + WAL for durability. Background queries (decay, eviction, pattern detection) |
| LLM provider | Claude API (anthropic SDK) | Best reasoning quality for the deliberative fallback |
| Async | asyncio | Learning loop, consolidation worker, sequence detection |
| API layer | FastAPI (when needed) | If HTTP interface is required later |
| Testing | pytest | Standard, rich ecosystem |

## Two-Layer Storage Architecture

The graph uses two layers. This is not a cache-in-front-of-DB pattern — the in-memory layer IS the primary system. SQLite is for persistence and background analytics only.

### Layer 1: In-Memory (Hot Path — every request)

All graph data lives in memory as Python objects:

```
nodes: dict[str, HabitNode]           # O(1) lookup by pattern_id
children: dict[str, list[ChildLink]]  # O(1) lookup of all children
parents: dict[str, list[str]]         # O(1) reverse lookup
```

**Performance:**
- Single node lookup: ~200 nanoseconds
- 7-hop composed skill traversal: ~1.5 microseconds
- Counter update (reinforcement): ~200 nanoseconds

**Memory footprint:**
- Per node: ~2KB (384-dim float32 vector + metadata)
- 1K nodes (early stage): ~2MB
- 100K nodes: ~200MB
- 500K nodes: ~1GB

This is the only architecture that achieves true sub-microsecond traversal. For comparison:

| Store | Single lookup | 7-hop traversal |
|---|---|---|
| **Python dict** | **~200ns** | **~1.5μs** |
| LMDB | ~4μs | ~28μs |
| SQLite (mmap) | ~8μs | ~56μs |
| SQLite (WAL) | ~10μs | ~70μs |
| Redis (local) | ~150μs | ~1ms |

In-memory is 50-500x faster than any persistent store.

### Layer 2: SQLite (Persistence + Background)

SQLite serves two purposes:

**1. Durability:** The in-memory graph is periodically snapshotted to SQLite. Mutations between snapshots are logged via a write-ahead approach. On startup, the full graph is loaded from SQLite into memory.

**2. Background queries:** Analytical operations that don't belong on the hot path:
- Decay scans: `SELECT * FROM nodes WHERE decay_score > threshold`
- Eviction candidates: `SELECT * FROM nodes ORDER BY habit_strength ASC LIMIT N`
- Pattern detection: `SELECT * FROM interaction_log WHERE ...`
- Link formation: sequence analysis across activation logs

SQLite is never touched during normal request handling. It is the "sleep/wake" mechanism — save state, restore state.

**Schema:**
- **nodes table**: pattern_id, trigger_patterns, response, confidence, reinforcement_count, decay_score, stability, risk_level
- **links table**: parent_id, child_id, condition, order, sequence_position
- **interaction_log**: prompt, route, latency, acceptance, timestamp

### Why Not Just SQLite?

SQLite on the hot path means every reflex goes to disk (or at best, through the OS page cache with mmap). At 10μs per lookup and 7 hops, that's 70μs per traversal — before any business logic. For a system handling 95% of all requests, that overhead compounds.

The nervous system analogy is literal: your reflexes operate from neurons in memory, not from a filing cabinet. The filing cabinet (SQLite) is for consolidation during sleep.

### Why Not Just In-Memory?

Process crash = total amnesia. SQLite provides crash recovery. The combination gives us sub-microsecond hot path AND durability.

## Vector Search: FAISS

The entry point to the graph is semantic matching — "does this input match any known node?" FAISS handles this:
- In-memory index (same process, no network hop)
- Embedding similarity search to find the best matching node
- Returns the entry point; in-memory dicts handle the rest (link traversal, child lookup)
- Scales to 500K+ vectors in-process

FAISS index is kept in sync with the in-memory graph. Both are persisted to disk and loaded on startup.

## Embedding Model: E5-Small

Every input gets embedded. Every node has an embedding. The model must be:
- Fast enough to embed on every request (<5ms)
- Small enough to stay in memory (~500MB)
- Accurate enough for habit matching (not general-purpose RAG)

E5-Small fits: 118M params, 384 dims, local inference, no API cost.

## LLM: Claude API

The LLM is the teacher, not the engine. It handles:
- Novel situations the graph hasn't learned
- Complex reasoning the graph can't compose
- Validation when a node's confidence is borderline

Used ~5% of the time at steady state. Cost decreases as the graph grows.

## Alternatives Considered

### Language Alternatives

**Rust**
- **Pros:** Maximum runtime performance, memory safety
- **Cons:** Smallest contributor pool, slowest to MVP, embedding ecosystem still maturing. The graph traversal is in-memory Python dicts — already sub-microsecond. Rust can't meaningfully improve on that for this workload.
- **Verdict:** Over-engineering. The hot path is already fast enough in Python.

**Go**
- **Pros:** Good concurrency model, decent performance
- **Cons:** Weakest ML/embedding ecosystem. Would require CGo bindings for FAISS and embedding inference.
- **Verdict:** Wrong tool for this job.

**TypeScript (Node.js)**
- **Pros:** Large contributor pool, first-class LLM SDK support
- **Cons:** Embedding and vector search ecosystem significantly weaker than Python.
- **Verdict:** Viable but inferior ecosystem for the ML-heavy components.

### Storage Alternatives

**Neo4j / Graph Databases**
- **Verdict: Rejected.** Heavy external dependency, network overhead on every traversal, operational complexity. Our graph structure is simple parent-child links — it doesn't need Cypher. In-memory dicts are faster by orders of magnitude and scale to 500K+ nodes.

**Redis**
- **Verdict: Rejected.** RedisGraph reached end-of-life January 2025. Network overhead makes it slower than SQLite for local use (~150μs per lookup vs ~10μs). Running a separate server for an embedded agent adds unnecessary complexity.

**LMDB**
- **Verdict: Runner-up.** Fastest persistent key-value store (~4μs reads via zero-copy mmap). Strong production track record (Pinterest, Caffe). But still 20x slower than in-memory dicts, requires serialization/deserialization, and loses SQL expressiveness for background queries.

**DuckDB**
- **Verdict: Rejected.** Columnar OLAP database optimized for aggregation scans, not point lookups. Architecturally wrong for our access pattern (single-row fetches by ID).

**RocksDB**
- **Verdict: Rejected.** Python bindings archived (June 2025). Slower reads than LMDB. Designed for write-heavy workloads exceeding RAM — neither applies here.

**Pure SQLite (no in-memory layer)**
- **Verdict: Rejected.** 10μs per lookup, 70μs per 7-hop traversal. Adequate for MVP but creates a performance ceiling that requires a storage migration to break through. We don't plan migrations — we pick the right architecture from the start.

## Why Python's "Slowness" Doesn't Matter Here

The graph traversal hot path:

1. **Embed input** — runs in C++ (ONNX Runtime / PyTorch backend)
2. **FAISS similarity search** — runs in C++ (find matching node)
3. **Dict lookups** — Python dict is a C hash table (~200ns per lookup)
4. **Confidence scoring** — trivial float arithmetic
5. **Response retrieval** — Python object attribute access

The only step that touches the Python interpreter meaningfully is the orchestration logic between these steps. Expected overhead: <1μs.

## What We're NOT Using (and Why)

| Skipped | Reason |
|---|---|
| Neo4j / graph DB | Network overhead, operational complexity, slower than in-memory dicts. Not revisiting. |
| Redis | RedisGraph is dead. Network overhead. Wrong tool for embedded agent. |
| LMDB | 20x slower than in-memory dicts. No SQL for background queries. |
| DuckDB | OLAP, wrong access pattern. |
| RocksDB | Python bindings abandoned. Write-optimized, we're read-heavy. |
| Qdrant / ChromaDB / Milvus | External vector DB unnecessary when FAISS runs in-process. |
| NetworkX | Adds abstraction overhead for simple parent-child traversal we can do with dicts. |
| Docker | No deployment target yet. |
| Django / Flask | FastAPI only if we need HTTP. |
| Poetry / pip | uv handles everything. |

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Process crash loses in-memory state | SQLite snapshots + mutation log. Recovery loads last snapshot and replays. |
| Startup time at large graph size | 500K nodes from SQLite: estimated 2-5 seconds. Acceptable for a system that runs continuously. |
| Memory pressure at 500K+ nodes | ~1GB at 500K. Monitor and alert. If exceeded, implement tiered loading (hot nodes in memory, cold nodes on demand). |
| Python GIL limits concurrency | asyncio for I/O; FAISS/ONNX release GIL during compute. Hot-path dict lookups are too fast for GIL to matter. |
| sentence-transformers model loading | Load once at startup, keep in memory. <2s cold start. |
| FAISS index out of sync with in-memory graph | Single code path for mutations: update dict + update FAISS + log to SQLite. Atomic at application level. |
| Graph traversal depth causes latency | Cap depth at 20. Most composed skills are 3-7 nodes. |
| Claude API availability | Abstract LLM provider interface. Swap to OpenAI, local model, or any provider. |

## Consequences

- All contributors need Python 3.12+ and uv
- Embedding model downloads ~500MB on first run (E5-Small weights)
- Graph lives in process memory — the process must stay running for the graph to be active
- SQLite file contains the full graph for persistence — easy to back up, version, inspect
- FAISS index file must be kept in sync with graph data
- No external services required (everything is local/in-process)
- No database migration between MVP and production — same architecture at every scale
- Graph traversal logic is pure Python dict operations, not a query language
