# CogniGraph

**Automate without coding. Build without knowing.**

AI is the gateway. CogniGraph is the cognitive runtime — where AI reasoning crystallizes into auditable habits, deployable without the AI that taught them.

This is not an MVP product. It is the reference implementation of a category: **post-software automation**, where systems learn what they should do from observed AI experience, and run without the AI when training is complete.

---

## Three eras

| Era | What you write | What automates |
|---|---|---|
| **Software** | code | deterministic execution |
| **AI** | prompts | non-deterministic generation, every call |
| **Cognitive** | nothing — describe to AI, train through use | auditable cognitive runtime, deployable without the AI |

---

## Invariants — don't break these

1. **Graph-first, LLM-fallback.** Graph is primary; LLM is teacher and fallback. Never the reverse.
2. **Protocols are the spec.** Every component swappable via `@runtime_checkable Protocol`. Designed for cross-language ports — write portable invariants, not Python idioms.
3. **Safety fires before graph routes execute.** Risk / volatility / ambiguity / blocklist. The graph cannot confidently serve a wrong answer by design.
4. **Tests are the contract.** Behavior pinned by tests is part of the future standard. Don't change a tested invariant without an ITD.
5. **Auditable to the row.** Every interaction lands in SQLite. The interaction log is the source of truth for the learner.
6. **ITDs in `itds/`** for architectural decisions. Not in commit messages.
7. **Train with AI, deploy without it.** Anything that makes the system require an LLM at runtime when the graph could handle it is a regression.

---

## State

11 critical-priority components shipped. 548 tests passing. Architectural risk is retired. See [README.md](README.md) for the public framing and [docs/architecture.md](docs/architecture.md) for the deep reference.

Remaining work is adoption infrastructure (benchmarks, integrations, multi-tenant, **deploy-mode without LLM**, observability hooks) and lifecycle/REPL polish — not core architecture.

The highest-leverage feature gap is **deploy-mode**: a `NoOpLLM` + LLM-less pipeline construction, plus a bulk-training CLI. That's what makes "train with AI, deploy without it" real instead of aspirational.

---

## Code-style guardrails

- **Build for an external developer audience.** The library is the demo of the pattern, not the product itself.
- **Tests double as cross-language spec.** Behavior should be reproducible in TypeScript / Go / Rust ports.
- **Architect-review every component.** BLOCK / WARN / NOTE discipline. BLOCKs and important WARNs fixed inline; lower-priority items filed as tracked issues.
- **No emojis in code.** No comments that re-state what the code does.
- **Don't sand off pre-MVP edges in writing.** Honesty about stage closes more design partners than polish.
- **Naming**: "CogniGraph" is the project; "graph-first, LLM-fallback with crystallized habits" is the pattern; "cognitive runtime" is the category we're defining.

---

## Agent activation

Six agents auto-activate by task context. Skills in `.claude/skills/`.

| Agent | Activate when |
|---|---|
| **Architect** | priority:critical or priority:high issues; security or safety code; pipeline/lifecycle wiring; protocol or interface changes; cross-component dependencies; routing/threshold changes |
| **Developer** | implementing or refactoring source code |
| **Tester** | after any new component (auto-trigger); before merge; logic / threshold / lifecycle changes; bug fixes (regression test) |
| **Auditor** | node creation or migration; safety / risk classification review; stale or conflicting nodes; wrong graph response served |
| **Benchmarker** | new embedding model or similarity method; capacity or eviction logic changes; latency concerns; graph-vs-LLM performance |
| **Researcher** | technology evaluation; comparing libraries; design questions; prior art lookup |

### Coordination

| Scenario | Order |
|---|---|
| New component | Developer → Architect → Tester + Benchmarker (parallel) |
| Tech choice | Researcher → Architect → Developer |
| Routing change | Developer → Tester + Auditor + Architect (parallel) |
| Safety change | Architect → Developer → Tester + Auditor (parallel) |
| Wrong answer served | Auditor → Developer → Tester |
| Pre-release | Tester + Benchmarker + Auditor + Architect (parallel) |

Priority: **Architect** (structural review) > **Auditor** (data integrity) > **Tester** (correctness) > **Benchmarker** (measurement) > **Developer** (build) > **Researcher** (investigate).
