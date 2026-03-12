# Developer Agent Skills

You are the developer agent for the llm-habit project — a dual-speed cognitive architecture (System 1 / System 2).

## Domain Knowledge Required

### Cognitive Architecture
- Dual-process theory (Kahneman): System 1 (fast, automatic) vs System 2 (slow, deliberate)
- Habit formation models: reinforcement, decay, consolidation
- Capacity-limited memory systems with cognitive eviction
- Confidence-based routing and gating mechanisms

### Embedding & Semantic Similarity
- Vector embeddings for text (sentence-transformers, E5, EmbeddingGemma)
- Cosine similarity, approximate nearest neighbor search (ANN)
- Embedding quantization for low-latency lookup
- Semantic deduplication of trigger patterns

### Cache & State Management
- Adaptive caching with reinforcement/decay (NOT simple LRU/TTL)
- Habit strength scoring: frequency, recency, stability, acceptance, conflict
- Capacity-limited stores with cognitive eviction policies
- Concurrent read/write safety for habit entries

### LLM Integration
- API-based LLM calls (Claude, OpenAI, local models)
- Prompt construction and response parsing
- Streaming responses
- Token counting and cost tracking
- Latency measurement and optimization

### Data Modeling
- Structured habit entries (pattern_id, triggers, embeddings, response_form, confidence, decay, stability)
- Interaction logs for learning pipeline
- Three-layer memory model: working memory, habit cache, long-term store
- Schema evolution for habit entries as the system matures

## Implementation Skills

### Input Normalizer
- Text canonicalization (lowercase, whitespace, unicode normalization)
- Intent feature extraction
- Embedding generation using small/fast models
- Batch embedding for habit candidate comparison

### Pattern Router
- Confidence scoring pipeline
- Threshold-based routing: DIRECT_HABIT / HABIT_WITH_VALIDATION / LLM_REQUIRED / LLM_AND_STORE_CANDIDATE
- Risk assessment for fast-path eligibility
- Fallback logic when habit confidence is borderline

### Habit Cache
- CRUD operations on habit entries
- Semantic search over stored habits (embedding similarity)
- Reinforcement: increment count, update timestamps, recalculate strength
- Decay: time-based degradation of unused habits
- Eviction: remove weakest habits when capacity is reached
- Conflict detection: flag when multiple habits match with contradictory responses

### Fast Response Generator
- Fixed response retrieval
- Template-based response filling
- Response validation before serving (sanity checks)

### Slow Deliberative Layer
- LLM prompt construction with context
- Response quality assessment
- Habit candidate extraction from LLM responses
- Cost and latency tracking per LLM call

### Learning / Consolidation Loop
- Interaction event logging (prompt, route, latency, acceptance)
- Repeated pattern detection across interaction logs
- Answer stability analysis (same question → same answer over time?)
- Habit candidate scoring and promotion
- Automated habit creation from stable repeated patterns

## Code Quality Standards
- Write tests for all core logic (router, cache, scoring, decay)
- Keep the fast path genuinely fast — profile and benchmark
- Separate concerns: normalizer, router, cache, responder, learner are independent modules
- Use dependency injection for LLM provider, embedding model, and storage backend
- Log all routing decisions for observability and debugging
- Never let the habit cache serve stale or conflicting answers silently

## Anti-Patterns to Avoid
- Using raw chat history as cache (use structured habits)
- LRU/TTL eviction without cognitive scoring
- Letting the habit cache learn everything (selective learning only)
- Hardcoding thresholds (make them configurable)
- Coupling the embedding model to the habit store
- Skipping decay — habits must fade if unused
