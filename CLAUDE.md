# Cognigraph: Runtime Cognitive Coordination Layer

## Positioning

**Cognigraph is a runtime cognitive coordination layer that transforms AI reasoning into adaptive enterprise execution.**

Unpacked:

- **Runtime** — real-time decisions on every request. Not offline training, not batch indexing.
- **Cognitive coordination** — orchestrates across four concerns the customer can't safely conflate: routing (graph vs LLM), safety (risk / volatility / ambiguity / blocklist), reinforcement (which habits strengthen), and learning (when novel patterns crystallize).
- **AI reasoning** — vendor-neutral input. Claude, GPT, Gemini, local models — anything implementing `LLMProvider`.
- **Adaptive** — learns from experience, reinforces what works, decays what doesn't, evolves with the workload over the lifetime of the deployment.
- **Enterprise execution** — operationally deployable: auditable to the row, safety-gated by design, local-first storage, multi-tenant-capable, protocol-driven for swap-out.

## Strategic Direction

**Cognigraph is not building an MVP product. It is building the reference implementation of an architectural pattern with the ambition of becoming foundational AI infrastructure — the coordination layer between AI reasoning and enterprise action.**

- **Pattern (the mechanism):** graph-first, LLM-fallback with crystallized habits. Head of the query distribution executes from a learned graph at sub-millisecond latency / zero token cost. Tail goes to the LLM. Both auditable.
- **Position (the wedge):** the runtime layer between an enterprise's LLM provider and the system that needs to act on the LLM's output. Drop it in front of any LLM-using product to make the head deterministic, cheap, and safe.
- **Distribution:** open-source GitHub repo, vendor-neutral via Protocol-driven swap-points. Long-horizon target is the named-pattern + reference-implementation outcome.
- **Horizon:** 18-36 months to credible standard-bearer. 1-2 months to a usable command-line system (#20 lifecycle + #21 REPL).
- **Pitch frame:** future-state / investment-grade direction, NOT a deploy-today product. When pitching to enterprise / non-engineering audiences, lead with the coordination-layer framing (turn AI reasoning into adaptive execution) and the market shift (LLM cost is permanent, audit pressure is rising, local inference is feasible). Lead with the pattern, not feature lists.

### What this means for code-level decisions

- **Build for an external developer audience**, not just internal use. The library is the demo of the pattern, not the product itself.
- **Code style: portable invariants over Python idioms.** Protocols are designed to be portable to TypeScript / Go / Rust later. Avoid Python-specific abstractions in the public contract surface.
- **Tests are also a contract document for re-implementers.** Behavior pinned by tests is part of the future spec.
- **ITDs are durable records.** Continue the ITD discipline — they become the spec when the pattern formalizes.
- **Architectural risk has been retired.** Eleven critical-priority components shipped, 527 tests. The remaining engineering work is adoption infrastructure (benchmarks, integrations, multi-tenant, observability), not core architecture.
- **Naming:** "Cognigraph" is the project name; **"runtime cognitive coordination layer"** is the positioning; **"graph-first, LLM-fallback with crystallized habits"** is the pattern name worth owning. The pattern outlives this specific implementation.
- **Priorities that gate adoption-as-standard** (treat as critical-priority once #20/#21 land):
  - Benchmarks vs alternatives (GPTCache, raw LLM, RAG) — every architect review has flagged the missing numbers
  - Adapter / integration work (LangChain, LlamaIndex, OpenAI Agents SDK)
  - Multi-tenant isolation (enterprise plausibility)
  - Observability / metric-sink hooks (#28 — ops credibility)
  - Eventual TypeScript port — design Protocols with that in mind now

### What this is NOT

- Not an agent framework (we sit between the framework and the LLM)
- Not RAG (we're complementary — RAG grounds the LLM in documents; we eliminate the LLM call on the head of the distribution)
- Not a fine-tune (no training loop; habits are inspectable text in a database, retrainable in seconds)
- Not a model (vendor-neutral via `LLMProvider` protocol)
- Not a cache (caches are passive; we coordinate — we route, gate, reinforce, and learn)

### Intellectual lineage (internal — not the public pitch)

The mechanism mirrors basal-ganglia / striatal procedural memory in neuroscience: routine decisions execute through a learned, gated, decay-pruned circuit; novel decisions escalate to cortex. System 1 / System 2 (Kahneman) is the same idea at a different level of description. These framings are useful for engineering reasoning; they are NOT the positioning we lead with externally — enterprise audiences relate to "coordination layer," not to "basal ganglia."

## Project Vision

A human-like agent built around a learned graph (nervous system) with an LLM (brain) as fallback. The graph is the primary system — it handles ~95% of decisions automatically. The LLM is the teacher — it handles novel situations and the graph learns from it so the LLM never has to handle that situation again.

This is not "LLM with a cache." This is a simulated human where the graph grows from experience, composes skills from building blocks, and forgets what it doesn't use.

## Core Principle

**The graph starts empty. The LLM handles everything. The graph watches, learns, and grows. Over time, the graph handles more, the LLM handles less. Just like a human.**

- Day 1: 100% LLM (baby — everything is novel)
- Week 1: ~70% LLM (common patterns learned)
- Month 1: ~30% LLM (most repeated tasks automated)
- Steady state: ~5-10% LLM (only genuinely novel situations)

## Architecture: Graph-First, LLM-Fallback

```
User Input
     ↓
Normalizer (sensory processing)
     ↓
Graph (nervous system — PRIMARY, ~95%)
  ├── known path, high confidence → execute automatically
  │     ├── single node → simple response (flat habit)
  │     └── linked nodes → composed sequence (skill chain)
  │
  └── no path / low confidence / novel → LLM (fallback, ~5%)
           ↓
       Learn from LLM result
           ↓
       Grow new nodes and paths in the graph
```

## The Graph

The graph is made of habit entries (nodes) that link to each other. Relationships between entries form the structure. There is no separate "tree" or "cache" — one unified structure.

### Node (Habit Entry)

```
pattern_id            unique identifier
trigger_patterns      input patterns that activate this node
embedding_vector      semantic representation
confidence            certainty (0.0 - 1.0)
reinforcement_count   successful activations
last_used_at          timestamp
decay_score           fading from disuse
stability             high | medium | low
risk_level            low | medium | high

# Response (if this node answers directly)
response_form         fixed | template | procedural
response              the answer or template

# Links (connections to other nodes)
children              [{ habit_id, condition, order }]
parents               [{ habit_id }]

# Composition
is_composed           formed by linking existing nodes?
sequence_position     position in parent sequence
```

### Graph Behaviors

- **Single node, no children** = simple recall ("what's your name?" → "Ibrahim")
- **Node with children** = composed skill ("commit changes" → stage → message → commit → verify)
- **Node with multiple parents** = shared building block ("check status" reused by commit, deploy, review)
- **Node with conditional children** = context branch ("pedestrian ahead" → crossing? → brake or proceed)
- **Any node** can escalate to LLM when confidence is low
- **Error at any node** snaps that node back to LLM-dependent

### How the Graph Learns (Bottom-Up)

1. **Flat nodes form first.** LLM answers a question. Same question, same answer, repeated. A node is created.
2. **Links form between nodes.** System notices nodes firing in sequence. Links are created.
3. **Composed nodes emerge.** A parent node wraps a sequence. The chain fires as one.
4. **Shared building blocks are discovered.** A node appears in multiple sequences. It becomes shared.

### How the Graph Forgets

- Unused nodes decay over time
- Weak nodes are evicted when capacity is reached
- Eviction uses habit strength scoring (not LRU)
- Composed nodes can partially decay: some children stay automatic, others fade to LLM-dependent

## Two Approaches to Graph Population

### Approach 1: Learn from Experience (MVP)
The graph starts empty. Every request goes to the LLM. The graph observes, learns, grows. Knowledge is earned through repetition and validation.

### Approach 2: Pre-loaded Procedures (Future, NOT MVP)
Pre-load the graph with known domain procedures. "You're a developer? Here's git workflows." Skips the baby phase for known domains. Requires careful confidence scoring since pre-loaded nodes haven't been validated through experience.

## Habit Strength Formula

```
habit_strength =
    (usage_frequency * a)
  + (recent_usage * b)
  + (answer_stability * c)
  + (user_acceptance * d)
  + (latency_savings * e)
  - (conflict_penalty * f)
  - (decay * g)
```

## Safety Boundary

The graph ONLY executes automatically when:
- High confidence
- Low ambiguity
- Low risk
- Low volatility

Everything else → LLM. The graph must never confidently serve wrong answers.

## Three Memory Layers

| Layer | Purpose | Lifetime | Example |
|---|---|---|---|
| Working Memory | Current conversation context | Session | Active topic, recent entities |
| Graph (Habit Nodes) | Learned response patterns and skill chains | Persistent + decay | Identity, preferences, workflows |
| Long-Term Store | Broad persistent knowledge | Persistent | User facts, history, definitions |

## Important Technical Decisions

All major technical decisions are recorded in `itds/`. Read the relevant ITD before revisiting a settled decision.

| ITD | Decision | Status |
|---|---|---|
| [ITD_2026-03-11_language-and-stack-selection](itds/ITD_2026-03-11_language-and-stack-selection.md) | Python 3.12+, uv, FAISS, in-memory graph + SQLite persistence, sentence-transformers, Claude API | Accepted |
| [ITD_2026-03-11_graph-first-architecture](itds/ITD_2026-03-11_graph-first-architecture.md) | Graph-first, LLM-fallback. Graph is primary system (~95%), LLM is teacher (~5%) | Accepted |

## Key Design Rules

- The graph is the primary system, the LLM is the fallback
- The graph starts empty and grows from experience (MVP)
- Nodes link to form composed skills — the structure is emergent, not predefined
- Never use raw chat history — use structured nodes with links
- Capacity-limited with cognitive eviction (habit strength scoring)
- Learning is selective — not everything deserves to become a node
- Decay is natural — unused nodes fade, active ones strengthen
- Error at any node escalates that specific node to LLM

## Reference: Embedding Models for Semantic Matching

- **E5-Small**: 118M params, 384 dims, 512 token context
- **EmbeddingGemma**: 308M params, <200MB RAM quantized, 100+ languages
- **LEAF (MongoDB)**: ≤30M params, 120 queries/sec, 87MB memory

## Reference: Related Projects

- **RouteLLM** (lm-sys): LLM routing decisions
- **GPTCache** (zilliztech): Semantic cache for LLM responses
- **OpenR** (openreasoner): Dual-process reasoning framework
- **NVIDIA LLM Router**: Multi-model routing blueprint

## Agent Auto-Activation Rules

Agents are activated automatically based on task context. Do not wait for explicit invocation.

### Developer Agent (`.claude/skills/developer.md`)
**Activate when:**
- Writing, modifying, or refactoring source code
- Implementing components (normalizer, graph store, traversal, responder, learner)
- Wiring components together
- Fixing bugs

### Research Agent (`.claude/skills/researcher.md`)
**Activate when:**
- Evaluating technology choices (embedding model, storage, LLM provider)
- Comparing libraries or approaches
- Investigating design questions
- Looking up papers, benchmarks, or prior art

### QA Automation Agent (`.claude/skills/tester.md`)
**Activate when:**
- Any new component or feature has been implemented (auto-trigger after developer work)
- Verifying acceptance criteria before merge (pre-merge QA gate)
- Changing graph traversal logic, thresholds, or confidence scoring
- Modifying node creation, reinforcement, decay, or linking logic
- A bug has been fixed (write regression test)
- Behavioral properties need verification (learning lifecycle, sequence composition)
- Adversarial robustness needs evaluation
- Cross-component integration needs validation
- Full test suite regression check requested

### Benchmark Agent (`.claude/skills/benchmarker.md`)
**Activate when:**
- A new embedding model or similarity search method is integrated
- Graph capacity or eviction logic changes
- Latency or cost concerns are raised
- Comparing graph-path vs LLM-fallback performance
- Evaluating whether the graph is actually reducing LLM usage over time

### Architect Agent (`.claude/skills/architect.md`)
**Activate when:**
- Any issue labeled `priority:critical` or `priority:high` is being implemented
- Security-related code is written or modified (safety boundary, risk gating, input validation, API key handling, SQLite queries)
- Architectural review is explicitly requested
- Pipeline or lifecycle wiring is changed (`pipeline.py`, `lifecycle.py`)
- Protocol ABCs or interfaces are changed (contract changes)
- Cross-component dependencies are introduced or modified
- Data model or configuration schema changes
- The graph-first/LLM-fallback boundary is touched (routing, thresholds, escalation)

**Do NOT activate for:**
- Cosmetic changes, test-only changes, documentation updates

### Auditor Agent (`.claude/skills/auditor.md`)
**Activate when:**
- Graph nodes are being created, modified, or migrated
- Reviewing safety boundary or risk classification
- Checking for stale, conflicting, or overgeneralized nodes
- Validating confidence calibration
- Any incident where a wrong graph response was served

### Multi-Agent Coordination

| Scenario | Agents to activate |
|---|---|
| New component implemented | Developer → Architect review → then Testing + Benchmark in parallel |
| Technology choice needed | Research → Architect review → then Developer |
| Graph traversal logic changed | Developer → then Testing + Auditor + Architect in parallel |
| Node/linking logic changed | Developer → then Testing + Auditor + Architect + Benchmark in parallel |
| Performance regression | Benchmark → then Developer to fix → Architect review |
| Wrong answer from graph | Auditor → then Developer to fix → then Testing |
| New embedding model evaluated | Research + Benchmark in parallel → Architect review |
| Pre-release validation | Testing + Benchmark + Auditor + Architect in parallel |
| Security-related change | Architect → then Developer → then Testing + Auditor in parallel |
| Pipeline/lifecycle wiring | Developer → Architect review → then Testing |
| Protocol/interface change | Architect review → then Developer → then Testing |
| Critical priority issue | Architect + Developer in parallel → then Testing + Auditor |

### Activation Priority
1. **Architect** — structural and security review first
2. **Auditor** — safety and data integrity
3. **QA Automation** — correctness and acceptance criteria before optimization
4. **Benchmark** — measure before declaring done
5. **Developer** — build what's decided
6. **Research** — investigate when uncertain

## Development Phases

### Phase 1: MVP (Learn from Experience)
- Input normalizer + embedding
- Graph store (nodes with links, SQLite + FAISS)
- Graph traversal (find node → follow links → respond or escalate)
- LLM integration (fallback + teacher)
- Reinforcement logger
- Learning loop: flat node creation from repeated stable LLM responses
- Learning loop: link detection from sequential node activations

### Phase 2: Maturity
- Node decay + cognitive eviction
- Conflict detection and resolution between nodes
- Capacity limits with habit strength scoring
- Composed node creation (automatic chunking)
- Shared building block discovery
- Pre-loaded procedure graphs (Approach 2)

### Phase 3: Advanced
- Context-sensitive branching (conditional children)
- Small local fast-responder model
- Offline graph optimization from logs
- Cross-session working memory
- Multi-user graph isolation
- Partial decay within composed sequences
