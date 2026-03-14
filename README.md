# CogniGraph

**A cognitive agent that learns from experience — so it stops reasoning about things it already knows.**

CogniGraph builds a learned graph (nervous system) around an LLM (brain). The graph starts empty — every request goes to the LLM. As patterns repeat, the graph grows nodes and links between them. Over time, the graph handles ~95% of decisions automatically. The LLM becomes a rare fallback for genuinely novel situations.

This isn't "LLM with a cache." This is a cognitive agent that learns, composes skills from building blocks, and forgets what it doesn't use.

## Why This Exists

Every LLM call is stateless. Ask it to do something it has done a thousand times before — it reasons from scratch. It has no memory of what worked, no concept of routine, no ability to get faster at familiar tasks.

Humans don't work this way. A driver doesn't consciously think about braking at a red light. A developer doesn't reason through `git commit` from first principles every time. These skills were learned through repetition and became automatic. ~95% of human decisions are subconscious.

CogniGraph gives AI systems this same ability: **learn what's routine, and stop thinking about it.**

## What Makes This Different

### Not a Cache

Semantic caches (like GPTCache) store LLM responses and replay them for similar inputs. That's useful, but it's fundamentally a lookup table with fuzzy matching.

CogniGraph is a **learned nervous system**:

| | Semantic Cache | CogniGraph |
|---|---|---|
| **What it stores** | Input → output pairs | Nodes with confidence, links, decay scores |
| **Multi-step skills** | No | Yes — nodes link into composed skill chains |
| **Learning model** | Cache everything | Selective — only stable, repeated, validated patterns |
| **Forgetting** | TTL / LRU (time-based) | Cognitive decay (usage-based: unused fades, active strengthens) |
| **Safety** | None | Risk gating, ambiguity detection, blocklist, per-node confidence |
| **Skill reuse** | No | Shared building blocks across multiple skill chains |
| **Inspectability** | Cache entries | Full graph with nodes, links, parents, children, confidence scores |
| **Error recovery** | Serve stale or invalidate | Snap individual node back to LLM-dependent |

### Not a Cost Optimization

LLM costs are dropping. Prompt caching exists. If CogniGraph's only value were "cheaper API calls," it wouldn't be worth building.

The value is in capabilities that LLMs fundamentally cannot provide:

**1. Near-zero latency for learned patterns (~200ns vs 1-3 seconds)**
Network round-trips have a physical floor. The graph operates in-memory. For real-time applications — robotics, interactive agents, game AI — this is the difference between responsive and unusable. No amount of LLM cost reduction eliminates network latency.

**2. Emergent skill composition**
When the graph detects nodes firing in sequence (check files → stage → commit → verify), it links them into a composed skill. The next time, the entire chain fires as one unit. This is how humans build expertise — not by memorizing answers, but by composing building blocks into increasingly complex skills. LLMs don't do this. Caches don't do this.

**3. Offline operation**
A mature graph handles 90%+ of requests without any API call. Useful for edge deployment, air-gapped environments, mobile, or anywhere connectivity is unreliable.

**4. Cognitive lifecycle**
Nodes aren't static entries. They have confidence scores that increase with successful use, decay rates that weaken them when unused, stability tiers that control how aggressively they decay, and reinforcement counts that track their track record. This models how human skills actually work — practice strengthens, disuse weakens, errors trigger re-learning.

**5. Per-user personalization without fine-tuning**
Each user's graph learns their specific patterns. "Deploy the app" means different things to different teams. The graph learns each user's version through experience, without fine-tuning a model or managing per-user prompts.

**6. Full inspectability**
Every decision the system makes is traceable. You can see which node matched, what its confidence was, whether it escalated to the LLM, and why. For regulated industries or safety-critical applications, this auditability is essential and impossible to get from an LLM.

### Comparison Table

| Capability | LLM Only | LLM + Cache | CogniGraph |
|---|---|---|---|
| Handle any input | Yes | Yes | Yes (LLM fallback) |
| Learn from interactions | No | No | Yes |
| Sub-millisecond response | No | Yes (cache hit) | Yes (graph hit) |
| Multi-step skill chains | Via prompting | No | Emergent from experience |
| Work offline | No | Partial | Yes (for learned patterns) |
| Per-user learning | No | No | Yes |
| Inspectable decisions | No | Partial | Full graph visibility |
| Forget unused knowledge | N/A | TTL expiry | Cognitive decay |
| Error recovery | N/A | Invalidate entry | Snap node to LLM-dependent |
| Safety boundary | Prompt-level | None | Per-node risk gating |

## Where CogniGraph Fits

### Strong fit

- **High-volume repetitive interactions** — support bots, FAQ systems, internal tools where 95% of queries follow known patterns
- **Workflow automation** — developer tools, ops pipelines, where sequential patterns are the norm
- **Per-user agents** — personal assistants, productivity tools that should learn each user's habits
- **Offline / edge deployment** — embedded systems, mobile, air-gapped environments
- **Real-time applications** — anything where 1-3 second LLM latency is unacceptable
- **Regulated industries** — where every decision must be auditable and explainable

### Weak fit

- **Diverse, novel queries** — research, creative writing, where every input is unique
- **Fast-changing domains** — real-time news, live data, where cached answers go stale immediately
- **Single-turn interactions** — one-shot API calls with no patterns to learn

## How It Works

```
User Input
     |
Normalizer (sensory processing)
     |
Graph (nervous system — PRIMARY, ~95%)
  |-- known path, high confidence --> execute automatically
  |     |-- single node --> simple response
  |     +-- linked nodes --> composed skill chain
  |
  +-- no path / low confidence / novel --> LLM (fallback, ~5%)
           |
       Learn from LLM result
           |
       Grow new nodes and links in the graph
```

### The Graph Grows From Experience

**Day 1** — The graph is empty. Everything goes to the LLM. Like a baby.

**Week 1** — Common patterns are recognized. "What's your name?" gets a graph node. Simple questions stop hitting the LLM.

**Month 1** — Sequences are discovered. "Commit changes" becomes a chain: check files -> stage -> write message -> commit -> verify. The whole chain fires without the LLM.

**Steady state** — The graph handles almost everything. The LLM only gets called for genuinely novel situations. Like a skilled human who rarely needs to consciously think about routine tasks.

### How Habits Form

```
1. LLM answers a question
2. Same question appears again --> LLM gives same answer
3. Repeats N times with stable, accepted answers
4. Graph creates a node --> future answers skip the LLM
5. System notices nodes firing in sequence --> links them
6. Composed skill emerges --> entire workflows fire automatically
```

### How Habits Die

Unused habits decay. The graph has capacity limits. Weak habits get evicted to make room for stronger ones. This isn't a bug — it's how the system prevents filling up with stale knowledge.

### How Errors Are Handled

If a graph node produces a wrong result, that specific node snaps back to LLM-dependent — like a driver who skids on ice suddenly paying full conscious attention to steering. The rest of the graph stays automatic.

## Architecture

### The Graph

The graph is made of **nodes** (learned habits) that **link to each other**. The structure emerges from experience — it isn't predefined.

```
"commit changes" (composed skill)
  |-- "check files changed" (node, automatic)
  |-- "stage files" (node, automatic)
  |-- "generate commit message" (node, uses learned convention)
  |-- "run commit" (node, automatic)
  +-- "verify success" (node, automatic)

"deploy to staging" (composed skill)
  |-- "check files changed" (shared node -- same one as above)
  |-- "run tests" (node, automatic)
  |-- "build" (node, automatic)
  +-- "push to staging" (node, automatic)
```

Nodes can be:
- **Simple** — one question, one answer, no children
- **Composed** — a sequence of child nodes that fire in order
- **Shared** — the same node reused across multiple parent skills
- **Branching** — children with conditions (context-dependent paths)

Any node can escalate to the LLM when confidence is low.

### Components

| Component | Role | Human Analogy |
|---|---|---|
| **Input Normalizer** | Canonicalize text, generate embeddings | Sensory processing |
| **Graph Store** | Nodes, links, embeddings, similarity search | Nervous system |
| **Graph Traversal** | Find matching node, follow links, execute or escalate | Reflexes / muscle memory |
| **LLM (Slow Path)** | Handle novel situations, reason deliberately | Conscious brain |
| **Learning Loop** | Observe patterns, create nodes, discover links | Practice / repetition |
| **Decay / Eviction** | Weaken unused nodes, remove weakest | Forgetting |
| **Safety Boundary** | Risk gating, ambiguity detection, blocklist | Caution / self-doubt |

### Human Behavior Mapping

| Human | System |
|---|---|
| Learning to drive | LLM handles everything, graph watches |
| Getting comfortable | Common actions become graph nodes |
| Muscle memory | Composed skills fire automatically |
| Forgetting unused skills | Decay removes unused nodes |
| Making an error | Node snaps back to LLM-dependent |
| Expert performance | Graph handles 95%, LLM handles 5% |

## Competitive Landscape

| Project | What It Does | How CogniGraph Differs |
|---|---|---|
| **GPTCache** | Semantic LLM response cache | CogniGraph has composition, cognitive decay, safety, selective learning |
| **RouteLLM** | Routes between cheap/expensive models | Different problem — model selection, not learning from experience |
| **RAG** | Retrieves context before LLM call | Still calls LLM every time — no learning, no latency improvement |
| **Fine-tuning** | Bakes knowledge into model weights | Expensive, not incremental, not per-user, requires retraining |
| **Prompt Caching** | Reduces cost for repeated prompt prefixes | Server-side optimization — no learning, no composition, no offline |

## Tech Stack

| Component | Choice |
|---|---|
| Language | Python 3.12+ |
| Package manager | uv |
| Embedding model | sentence-transformers (E5-Small, 384 dims) |
| Vector search | FAISS (IndexFlatIP) |
| Storage | In-memory dicts (hot path) + SQLite (persistence) |
| LLM | Claude API (anthropic SDK) |
| Testing | pytest |

See [ITD: Language and Stack Selection](itds/ITD_2026-03-11_language-and-stack-selection.md) for the full rationale.

## Roadmap

### Phase 1: MVP (Learn from Experience)
- [ ] Input normalizer + embedding generation
- [ ] Graph store (nodes with links, SQLite + FAISS)
- [ ] Graph traversal (find node -> follow links -> respond or escalate)
- [ ] LLM integration (fallback + teacher)
- [ ] Learning loop: node creation from repeated stable responses
- [ ] Learning loop: link detection from sequential activations
- [ ] Reinforcement and basic decay
- [ ] Safety boundary (risk gating, ambiguity detection)

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

**Phase 1 MVP — In development.** Foundation scaffold complete, implementing core components.

## Contributing

Contributions are welcome. This project is early stage.

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
- **Motor skill acquisition** — cognitive -> associative -> autonomous stages

## License

TBD — will be determined before first release. The intent is to be open source.
