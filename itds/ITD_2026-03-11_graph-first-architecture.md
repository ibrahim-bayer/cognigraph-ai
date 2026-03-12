# ITD: Graph-First Architecture

**Date:** 2026-03-11
**Status:** Accepted
**Supersedes:** Original flat habit cache design
**Decision:** The learned graph is the primary system (~95% of decisions). The LLM is a fallback for novel situations (~5%) and acts as the teacher that grows the graph.

## Context

The original architecture placed an LLM as the primary system with a habit cache as an optimization layer. This is backwards. Neuroscience research estimates ~95% of human decisions are subconscious — the conscious mind is the exception, not the rule.

If this project simulates a human-like agent, the graph (nervous system) must be the primary system and the LLM (conscious brain) must be the rare fallback.

## The Core Reframe

**Old:** LLM is primary, cache is optimization.
**New:** Graph is primary, LLM is fallback and teacher.

The LLM does not run the system. The LLM teaches the system. Over time, the graph handles more and the LLM is called less. This is how humans work — the more you practice, the less you think.

## Architecture

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

## Graph Structure

The graph is made of habit entries that link to each other. There is no separate "tree" or "cache" — the relationships between entries form the structure.

### Habit Entry (Node)

```
pattern_id            unique identifier
trigger_patterns      input patterns that activate this node
embedding_vector      semantic representation for similarity matching
confidence            certainty score (0.0 - 1.0)
reinforcement_count   successful activations
last_used_at          timestamp
decay_score           fading from disuse
stability             high | medium | low
risk_level            low | medium | high

# Response (if this node can answer directly)
response_form         fixed | template | procedural
response              the answer or template

# Links (connections to other nodes)
children              [{ habit_id, condition, order }]
parents               [{ habit_id }]

# Composition metadata
is_composed           was this formed by linking existing nodes?
sequence_position     position in a parent sequence
```

### How the Graph Works

- **Single node, no children** = simple recall ("what's your name?" → "Ibrahim")
- **Node with children** = composed skill ("commit changes" → stage → message → commit → verify)
- **Node with multiple parents** = shared building block ("check status" used by commit, deploy, review)
- **Node with conditional children** = context-dependent branching ("pedestrian ahead" → crossing? → brake or proceed)
- **Any node** can escalate to LLM when confidence is low
- **Error at any node** snaps that specific node back to LLM-dependent

### How It Learns (Bottom-Up)

1. **Individual nodes form first.** LLM answers a question. Same question, same answer, repeated. A flat habit node is created.
2. **Links form between nodes.** The system notices nodes firing in sequences. "Stage files" always fires before "write commit message" which fires before "run commit." Links are created.
3. **Composed nodes emerge.** A parent node "commit changes" is created, pointing to the sequence. The whole chain fires as one.
4. **Shared building blocks are discovered.** "Check git status" appears in multiple composed sequences. It becomes a shared node referenced by many parents.

### How It Forgets

- Unused nodes decay over time
- Weak nodes are evicted when capacity is reached
- Eviction uses habit strength scoring (frequency, recency, stability, acceptance — not LRU)
- Composed nodes can partially decay: some children stay automatic, others fade back to LLM-dependent

## Two Approaches to Graph Population

### Approach 1: Learn from Experience (MVP)

The graph starts empty. Every request goes to the LLM. The graph observes, learns, and grows from experience. Over time it handles more, the LLM handles less.

- Like a baby learning to walk
- Honest path — knowledge is earned
- Slow start, but the graph is high-quality because every node was validated through repetition
- **This is the MVP approach**

### Approach 2: Pre-loaded Procedures (Future)

Pre-load the graph with known procedures for a domain. "You're a developer? Here's git workflows, CI/CD patterns, deployment sequences."

- Like giving a student a textbook before class
- Skips the baby phase for known domains
- Risk: pre-loaded nodes haven't been validated through experience, need careful confidence scoring
- **This is Phase 2/3 — not in MVP scope**

## What Percentage Goes to LLM?

Day 1: 100% (empty graph, everything is novel)
Week 1: ~70% (common patterns learned)
Month 1: ~30% (most repeated tasks automated)
Steady state: ~5-10% (only genuinely novel situations)

The ratio shifts over time. The system gets cheaper, faster, and more autonomous with use.

## Example: Git Commit

### Day 1 (empty graph)
```
"commit these changes" → no graph path → LLM reasons through it
LLM: stage files, write message, run commit
Result logged.
```

### Day 30 (graph has learned)
```
"commit these changes"
     ↓
Graph traversal:
  → "check files changed" (node, automatic)
  → "stage files" (node, automatic)
  → "generate commit message" (node, uses learned convention)
  → "run commit" (node, automatic)
  → "verify success" (node, automatic)
     ↓
Done. No LLM called.
```

### Day 30, unexpected situation
```
"commit these changes"
     ↓
Graph traversal:
  → "check files changed" (automatic)
  → "stage files" (automatic)
  → "merge conflict detected" (no known path, low confidence)
  → ESCALATE TO LLM
     ↓
LLM handles the merge conflict
     ↓
Graph learns: new node "handle merge conflict" added
```

## Consequences

- The data model changes: habit entries gain children/parents/conditions (graph links)
- The router becomes graph traversal, not just cache lookup
- The learning loop must detect sequences, not just repetitions
- The system's value increases over time (unlike a static cache)
- The LLM cost decreases over time (fewer calls as graph grows)
- CLAUDE.md and all agent skills need updating to reflect graph-first architecture
