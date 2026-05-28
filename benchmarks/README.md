# Benchmarks

Reproducible cost / latency / deflection comparison: **CogniGraph vs raw LLM** (and GPTCache once that wrapper lands), on a head-heavy support workload.

## What this measures

Per system, on the same query stream:

- **Deflection rate** — share of queries that did NOT hit the LLM
- **Latency** — p50, p95, p99 (end-to-end, wall-clock)
- **Cost** — derived from the Anthropic SDK's `usage` field × published pricing
- **Errors** — any query that failed

## How to run

Install benchmark deps:

```bash
uv sync --group benchmarks
```

**Smoke run** — validates the harness end-to-end with a deterministic mock LLM. Zero tokens, ~5 seconds.

```bash
uv run python -m benchmarks smoke --queries 50 --warmup 20
```

**Real run** — requires `ANTHROPIC_API_KEY` in env.

```bash
# Cheap: ~$0.50 on Haiku, takes ~3-5 minutes
uv run python -m benchmarks run --model haiku --queries 200 --warmup 100

# Publishable: ~$5-10 on Sonnet, takes ~10-15 minutes
uv run python -m benchmarks run --model sonnet --queries 1000 --warmup 500
```

Results land in `benchmarks/results/<timestamp>.{md,json}`. The markdown is the report; the JSON is per-query provenance so anyone can re-derive the numbers.

## What's in this directory

```
benchmarks/
├── README.md
├── __main__.py              # CLI entry point (smoke / run)
├── dataset.py               # Bitext loader + Zipfian workload generator
├── runner.py                # Orchestration + cost model + percentiles
├── report.py                # Markdown report renderer
├── systems/
│   ├── __init__.py          # SystemUnderTest protocol + QueryResult
│   ├── raw_llm.py           # Baseline: call Claude every time
│   ├── cognigraph_system.py # Full CogniGraph pipeline with token tracking
│   └── mock_llm.py          # Deterministic fake for smoke runs
└── results/                 # Generated reports (.md + .json per run)
```

## Methodology notes

- **Same query stream across systems.** Zipfian sample over the dataset's labeled intents, fixed seed. Both systems see identical inputs in identical order.
- **Warm-up phase** only feeds CogniGraph (raw LLM is stateless — warming it up would just spend tokens). Measurement phase runs both.
- **Latency is end-to-end** — embedding + match + safety + LLM (when applicable) + response. Not just LLM round-trip.
- **Cost derives from real tokens**, not estimates. The Anthropic SDK's response `usage` field is the source of truth. Pricing table is in `runner.py::PRICING`.
- **No human quality scoring** in this iteration. Per-query records are in the JSON — spot-check responses before publishing. Future iteration: cosine similarity vs canonical responses, or a judge-LLM rating.

## Caveats baked into every report

- Single workload, single dataset, single model — the pattern is more important than absolute numbers.
- `learning_starting_confidence` (0.5) is below `confidence_threshold` (0.7), so new nodes first route LLM_FALLBACK before graduating to GRAPH_DIRECT. Larger warm-up = more graduation.
- No GPTCache comparison in this run. Add when the wrapper lands.
