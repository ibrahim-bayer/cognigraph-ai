# LLM-Habit

**A human-like agent that learns from experience and stops thinking about things it already knows.**

LLM-Habit builds a learned graph (nervous system) around an LLM (brain). The graph starts empty — every request goes to the LLM. As patterns repeat, the graph grows nodes and links between them. Over time, the graph handles ~95% of decisions automatically. The LLM becomes a rare fallback for genuinely novel situations.

This isn't "LLM with a cache." This is a simulated human that learns, composes skills, and forgets what it doesn't use.

## Why

Every time you ask an LLM to do something it has done before, it reasons from scratch. It burns the same compute on "commit these changes" whether it's the first time or the thousandth.

Humans don't work this way. ~95% of human decisions are subconscious. A driver doesn't think about braking at a red light — they just do it. That skill was learned through repetition and became automatic.

LLM-Habit gives an LLM the same ability: **learn what's routine, and stop thinking about it.**

## How It Works

```
User Input
     ↓
Graph (nervous system — handles ~95%)
  ├── known path → execute automatically
  └── unknown → LLM (brain — handles ~5%)
                    ↓
                Graph learns from it
```

### The Graph Grows From Experience

**Day 1** — The graph is empty. Everything goes to the LLM. Like a baby.

**Week 1** — Common patterns are recognized. "What's your name?" gets a graph node. Simple questions stop hitting the LLM.

**Month 1** — Sequences are discovered. "Commit changes" becomes a chain: check files → stage → write message → commit → verify. The whole chain fires without the LLM.

**Steady state** — The graph handles almost everything. The LLM only gets called for genuinely novel situations. Like a skilled human who rarely needs to consciously think about routine tasks.

### How Habits Form

```
1. LLM answers a question
2. Same question appears again → LLM gives same answer
3. Repeats N times with stable, accepted answers
4. Graph creates a node → future answers skip the LLM
5. System notices nodes firing in sequence → links them
6. Composed skill emerges → entire workflows fire automatically
```

### How Habits Die

Unused habits decay. The graph has capacity limits. Weak habits get evicted to make room for stronger ones. This isn't a bug — it's what prevents the graph from filling with stale knowledge.

### How Errors Are Handled

If a graph node produces a wrong result, that specific node snaps back to LLM-dependent — like a driver who skids on ice suddenly paying full conscious attention to steering. The rest of the graph stays automatic.

## Architecture

### The Graph

The graph is made of **nodes** (learned habits) that **link to each other**. The structure emerges from experience — it isn't predefined.

```
"commit changes" (composed skill)
  ├── "check files changed" (node, automatic)
  ├── "stage files" (node, automatic)
  ├── "generate commit message" (node, uses learned convention)
  ├── "run commit" (node, automatic)
  └── "verify success" (node, automatic)

"deploy to staging" (composed skill)
  ├── "check files changed" (shared node — same one as above)
  ├── "run tests" (node, automatic)
  ├── "build" (node, automatic)
  └── "push to staging" (node, automatic)
```

Nodes can be:
- **Simple** — one question, one answer, no children
- **Composed** — a sequence of child nodes that fire in order
- **Shared** — the same node reused across multiple parent skills
- **Branching** — children with conditions (context-dependent paths)

Any node can escalate to the LLM when confidence is low.

### Node Structure

```
pattern_id            unique identifier
trigger_patterns      inputs that activate this node
embedding_vector      for semantic similarity matching
confidence            certainty score (0.0 - 1.0)
reinforcement_count   successful uses
decay_score           fading from disuse
risk_level            low | medium | high

response              the answer (if this node responds directly)
children              linked child nodes (if this is a composed skill)
parents               parent nodes that reference this one
```

### Components

| Component | Role | Human Analogy |
|---|---|---|
| **Input Normalizer** | Canonicalize text, generate embeddings | Sensory processing |
| **Graph Store** | Nodes, links, embeddings, similarity search | Nervous system |
| **Graph Traversal** | Find matching node, follow links, execute or escalate | Reflexes / muscle memory |
| **LLM (Slow Path)** | Handle novel situations, reason deliberately | Conscious brain |
| **Learning Loop** | Observe patterns, create nodes, discover links | Practice / repetition |
| **Decay / Eviction** | Weaken unused nodes, remove weakest | Forgetting |

### Human Behavior Mapping

| Human | System |
|---|---|
| Learning to drive | LLM handles everything, graph watches |
| Getting comfortable | Common actions become graph nodes |
| Muscle memory | Composed skills fire automatically |
| Forgetting unused skills | Decay removes unused nodes |
| Making an error | Node snaps back to LLM-dependent |
| Expert performance | Graph handles 95%, LLM handles 5% |

## Tech Stack

| Component | Choice |
|---|---|
| Language | Python 3.12+ |
| Package manager | uv |
| Embedding model | sentence-transformers (E5-Small) |
| Vector search | FAISS |
| Storage | SQLite |
| LLM | Claude API (anthropic SDK) |
| Testing | pytest |

See [ITD: Language and Stack Selection](itds/ITD_2026-03-11_language-and-stack-selection.md) for the full rationale.

## Roadmap

### Phase 1: MVP (Learn from Experience)
- [ ] Input normalizer + embedding generation
- [ ] Graph store (nodes with links, SQLite + FAISS)
- [ ] Graph traversal (find node → follow links → respond or escalate)
- [ ] LLM integration (fallback + teacher)
- [ ] Learning loop: node creation from repeated stable responses
- [ ] Learning loop: link detection from sequential activations
- [ ] Reinforcement and basic decay

### Phase 2: Maturity
- [ ] Cognitive eviction with habit strength scoring
- [ ] Conflict detection and resolution
- [ ] Automatic chunking (composed node creation)
- [ ] Shared building block discovery
- [ ] Pre-loaded procedure graphs (skip the baby phase for known domains)

### Phase 3: Advanced
- [ ] Context-sensitive branching (conditional children)
- [ ] Small local fast-responder model
- [ ] Offline graph optimization from logs
- [ ] Multi-user graph isolation
- [ ] Partial decay within composed sequences

## Project Status

**Pre-MVP — Architecture finalized, implementation starting.**

## Contributing

This project is early stage. Contributions are welcome.

### Where Help Is Needed

- **Graph traversal algorithms** — efficient path finding and execution in a learned habit graph
- **Embedding model benchmarking** — testing small models for node matching accuracy and speed
- **Eviction algorithm design** — habit strength formula needs empirical tuning
- **Adversarial testing** — finding failure modes in semantic matching and confidence scoring
- **Sequence detection** — algorithms to discover when nodes fire in consistent patterns
- **Safety boundary design** — what should never become a graph node

### How to Contribute

1. **Open an issue** to discuss ideas before writing code
2. **Fork and submit a PR** for implementation work
3. **Share research** — papers, benchmarks, prior art on learned agent behavior
4. **Report failure modes** — cases where the architecture would produce wrong results

### Design Principles

- The graph is the primary system, not the LLM
- The graph starts empty and earns its knowledge through experience
- When uncertain, always fall back to the LLM
- Nodes link to form composed skills — the structure is emergent
- Measure everything — graph hit rate, LLM call reduction, latency, accuracy
- Unused knowledge should decay — staleness is worse than ignorance

## Support the Project

If you find this project valuable, consider supporting its development:

- **GitHub Sponsors** — [Sponsor on GitHub](https://github.com/sponsors/ibrahim-bayer) *(0% fees — 100% goes to development)*
- **Ko-fi** — [Buy us a coffee](https://ko-fi.com/ibrahimbayer)

Every contribution helps keep this project alive and actively maintained.

## Contact

**Website:** [ibgroup.dev](https://ibgroup.dev)

## Inspiration

- **Daniel Kahneman** — *Thinking, Fast and Slow* (System 1 / System 2)
- **Neuroscience** — ~95% of human decisions are subconscious
- **Cognitive psychology** — habit formation, procedural memory, reinforcement, decay
- **Ebbinghaus forgetting curve** — memory decay models
- **Motor skill acquisition** — cognitive → associative → autonomous stages

## License

TBD — will be determined before first release. The intent is to be open source.
