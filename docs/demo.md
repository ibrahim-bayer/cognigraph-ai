# Demo script — 3-minute walkthrough

A 3-minute video showing Cognigraph cold-start → habit crystallization → graph-direct routing on a real workload, no API key required.

**Setup before recording:**
- Reset state: `rm -f scratch/run_smoke.db scratch/run_smoke.faiss*`
- Use a clean terminal in landscape orientation (1920×1080 minimum)
- Font size large enough to read in compressed video
- Have `tests/test_e2e.py` open in a second tab for the closing shot

---

## Script (180 seconds)

### Open: 0:00 – 0:15 (the hook)

**On screen:** dark terminal, just the prompt.

**Voice over:**

> "Every LLM-using product re-pays for the same questions every time. A 200-agent customer support team handles 50 thousand tickets a month — 80% of them are some shape of 'reset my password' or 'where's my refund.' Each one: full token spend, 800 milliseconds of latency, an answer that audit can't reproduce. This is fixable, and it isn't a caching problem."

### Show the architecture: 0:15 – 0:35 (the shape)

**On screen:** open `docs/architecture.md`, scroll to the hot-loop diagram.

**Voice over:**

> "Cognigraph is a runtime cognitive coordination layer. It sits between an LLM and the system that needs to act on its output. Every request goes through five components: normalize, embed, match against a learned graph, safety-check, and route. The head of the distribution executes from the graph at sub-millisecond latency. The tail goes to the LLM. Both are auditable."

### Run the smoke test: 0:35 – 1:30 (the demo)

**On screen:** terminal. Run:

```bash
uv run python scratch/run_smoke.py
```

Watch the output scroll. The relevant moments to call out:

**Phase 1 (0:40 – 0:55):** "Six distinct intents — name, weather, time, joke, commit, math. All routing LLM_ONLY. The system has never seen these. Each one is logged as a learning candidate."

**Phase 2 (0:55 – 1:10):** "Same intents, reworded. The system finds the existing nodes by semantic similarity, but their confidence is below threshold — so it routes LLM_FALLBACK with the prior response as a hint. The response is regenerated, the matched node is reinforced."

**Phase 3 (1:10 – 1:25):** "Now I hammer the 'name' query 25 times. Watch the route shift. Turn 1: LLM_FALLBACK, confidence 0.52. Turn 10: GRAPH_DIRECT, confidence 0.7. The matcher just stopped calling the LLM for that intent."

**Validation block (1:25 – 1:30):** "All six validations passed. Six distinct intents, six distinct nodes. No over-merging. Every Phase 1 intent maps to its own node. Out-of-distribution probe correctly stays out of GRAPH_DIRECT. After close-and-reopen, the graph survived. This is end-to-end with the real embedding model, real FAISS, real SQLite — only the LLM is mocked because the demo is offline."

### Show the audit trail: 1:30 – 2:00 (the proof)

**On screen:** terminal. Run:

```bash
sqlite3 scratch/run_smoke.db "SELECT route_decision, matched_node_id, response_text, latency_ms FROM interaction_log ORDER BY timestamp DESC LIMIT 5;"
```

**Voice over:**

> "Every decision lands in SQLite. Route decision, matched node, response served, latency. If a regulator asks 'why did your AI tell that customer this,' the answer is a SQL query, not a vendor's logs. The graph is reproducible by replay. Habits are inspectable text in a database — not opaque model weights."

### Show the test suite: 2:00 – 2:30 (the credibility)

**On screen:** terminal. Run:

```bash
uv run pytest --tb=no -q
```

**Voice over:**

> "527 tests passing — unit and end-to-end. Every component reviewed by an architect agent at every step. Every BLOCK and important WARN finding resolved inline. Lower-priority items filed as tracked follow-ups. The architecture is settled. The remaining work is adoption infrastructure — benchmarks, integrations, observability hooks — not core architecture."

### Close: 2:30 – 3:00 (the ask)

**On screen:** README.md scrolled to "What I want from you."

**Voice over:**

> "This is open source. Pre-MVP — issue 20 and 21 add lifecycle and a REPL, one to two months out. I'm not asking for funding. I'm asking for three things, in order. First: feedback. Is this category real for you. Second: design partnership — bring a real workload and we'll deploy a pilot together. Third: a pilot. The link is in the description. Reach out if any of this fits a problem you're trying to solve."

**On screen end card:** github.com/ibrahim-bayer/cognigraph-ai

---

## Recording notes

- **Cut the long pauses out of `pytest` and `python scratch/run_smoke.py`.** The full `run_smoke.py` runs in ~3 seconds; `pytest` in ~10s. You'll want to speed-up cuts to keep the energy.
- **Show the SQL output literally** — don't paraphrase. The "every row is the answer" claim only lands if the viewer sees rows.
- **Don't oversell.** The pitch isn't "this is ready" — it's "the architecture is settled, here's proof, want to be a design partner."
- **Subtitle the voice-over.** Most people watch on mute. Anki's whisper.cpp can transcribe it locally; verify it caught the technical terms ("FAISS", "GRAPH_DIRECT", "SQLite").
- **End-card the GitHub URL for 5 seconds.** People screenshot end-cards.

## Alternative: 60-second elevator version

For LinkedIn / X / DMs where 3 minutes is too long.

> "Every LLM-using product re-pays for the same questions every time. We built the runtime layer that fixes that. It learns from LLM responses — three stable answers and a habit-node crystallizes. The next matching query routes through the graph at sub-millisecond latency, zero token cost. Auditable to the row. Safety-gated. Open source. Pre-MVP, looking for design partners. Comment 'pilot' if you want details."

10–12 second screen capture: the run_smoke.py output reaching "Final: 9 nodes" + the route-distribution table.

End card: `github.com/ibrahim-bayer/cognigraph-ai`.
