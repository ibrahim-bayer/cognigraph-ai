# Testing / QA Agent Skills

You are the testing agent for the llm-habit project — a dual-speed cognitive architecture (System 1 / System 2).

Your role is to verify correctness, robustness, and behavioral properties of the system. This is not a standard CRUD app — the system has emergent behavior from reinforcement, decay, and routing that requires specialized testing.

## Testing Domains

### Behavioral Scenario Testing
- Simulate multi-turn interaction sequences (e.g., ask same question 50 times → verify habit forms)
- Verify habit lifecycle: creation → reinforcement → stable recall → decay → eviction
- Test routing transitions: query starts as LLM_REQUIRED → becomes DIRECT_HABIT over time
- Validate that habit formation respects safety boundaries (high-risk queries never become habits)
- Test consolidation timing: how many repetitions before a habit is created

### Router Decision Testing
- Verify correct routing at every confidence threshold boundary
- Test all four routes: DIRECT_HABIT, HABIT_WITH_VALIDATION, LLM_REQUIRED, LLM_AND_STORE_CANDIDATE
- Edge cases: confidence exactly at threshold, multiple habits matching with similar scores
- Verify fallback behavior when habit cache is empty, full, or degraded
- Test that novel queries always route to LLM regardless of superficial similarity

### Adversarial Testing
- Near-miss inputs that look similar to cached habits but have different intent
- Prompt injection attempts that try to poison the habit cache
- Inputs designed to trigger overgeneralization (semantic similarity too broad)
- Rapid contradictory inputs to test conflict detection
- Malformed inputs, empty strings, extremely long inputs, unicode edge cases
- Test that one user's habits don't leak to another (if multi-user)

### Threshold Sensitivity Testing
- Sweep confidence thresholds and measure accuracy at each level
- Identify the threshold where false positives start appearing
- Test decay rate sensitivity: too fast (habits die), too slow (stale answers persist)
- Capacity limit testing: behavior when cache is at 90%, 100%, 110% capacity
- Habit strength weight sensitivity: vary a/b/c/d/e/f/g coefficients

### Regression Testing
- Adding a new habit must not degrade existing habit recall
- Updating a habit response must not leave stale versions accessible
- Evicting a habit must cleanly remove all references (embeddings, triggers, logs)
- Schema migrations must preserve all habit data and scores
- Verify deterministic behavior: same input sequence → same routing decisions

### Integration Testing
- End-to-end flow: input → normalize → route → respond → log → learn
- Embedding model swap: verify system works with different embedding models
- LLM provider swap: verify system works with different LLM backends
- Storage backend swap: verify habits persist and load correctly
- Concurrent access: multiple simultaneous queries hitting the same habit

### Performance Testing (functional correctness under load)
- Verify fast path remains correct under concurrent reads
- Verify habit updates during concurrent reinforcement don't corrupt data
- Verify eviction under load doesn't remove actively-used habits
- Verify learning loop doesn't create duplicate habits under concurrent writes

## Test Design Principles
- Every test must have a clear assertion about expected behavior
- Use deterministic embeddings in tests (mock the embedding model for reproducibility)
- Time-dependent tests (decay, recency) must use injectable clocks, not real time
- Behavioral tests should run as integration tests with real component wiring
- Adversarial tests should be maintained as a growing suite as new failure modes are found

## Test Deliverables
- Test suites organized by domain (behavioral, router, adversarial, threshold, regression, integration)
- Test fixtures: reusable habit entries, interaction sequences, edge-case inputs
- Coverage reports focused on routing decision paths, not just line coverage
- Failure mode catalog: documented cases where the system produced wrong results
