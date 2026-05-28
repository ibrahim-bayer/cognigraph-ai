"""Benchmark runner — orchestrates warm-up + measurement across systems."""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field

from benchmarks.dataset import Dataset, Sample
from benchmarks.systems import QueryResult, SystemUnderTest

logger = logging.getLogger(__name__)


# --- Pricing models (USD per 1M tokens) ---
# Update these when Anthropic changes pricing. Always re-quote in the
# published report so a reader can re-derive cost from raw tokens.
PRICING = {
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.00},
    "claude-haiku-4-5": {"input": 0.80, "output": 4.00},
}


def cost_usd(input_tokens: int, output_tokens: int, model: str) -> float:
    """Estimate USD cost for a given token count + model."""
    rates = PRICING.get(model)
    if rates is None:
        # Unknown model — use Sonnet rates as a conservative default
        rates = PRICING["claude-sonnet-4-5"]
    return (
        (input_tokens / 1_000_000.0) * rates["input"]
        + (output_tokens / 1_000_000.0) * rates["output"]
    )


@dataclass
class PerQueryRecord:
    """One row in the results JSON — full provenance of a measurement query."""

    system_name: str
    query: str
    expected_intent: str
    route: str
    response: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float


@dataclass
class SystemSummary:
    """Aggregate stats per system for the report."""

    name: str
    n_queries: int
    n_errors: int
    deflection_rate: float  # share of queries that did NOT hit the LLM
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    mean_latency_ms: float
    route_counts: dict[str, int]


@dataclass
class BenchmarkResult:
    """Everything needed to write a report and reproduce the run."""

    dataset_source: str
    n_intents: int
    warmup_size: int
    measurement_size: int
    head_intent_share: float  # share covered by top-5 intents
    model: str
    started_at: float
    finished_at: float
    summaries: list[SystemSummary] = field(default_factory=list)
    records: list[PerQueryRecord] = field(default_factory=list)


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def run_one_system(
    system: SystemUnderTest,
    warmup_stream: list[Sample],
    measurement_stream: list[Sample],
    model: str,
) -> tuple[SystemSummary, list[PerQueryRecord]]:
    """Run a single system through warm-up + measurement, return aggregates + per-row records."""
    logger.info("=== %s ===", system.name)
    logger.info("warm-up: %d queries", len(warmup_stream))
    for i, sample in enumerate(warmup_stream):
        system.warmup(sample.query)
        if (i + 1) % 100 == 0:
            logger.info("  warmup %d/%d", i + 1, len(warmup_stream))

    logger.info("measurement: %d queries", len(measurement_stream))
    records: list[PerQueryRecord] = []
    latencies: list[float] = []
    n_errors = 0
    n_deflected = 0  # not "llm_call"
    total_in = 0
    total_out = 0
    route_counts: dict[str, int] = {}
    total_cost = 0.0

    for i, sample in enumerate(measurement_stream):
        result = system.query(sample.query)
        c = cost_usd(result.input_tokens, result.output_tokens, model)
        records.append(
            PerQueryRecord(
                system_name=system.name,
                query=sample.query,
                expected_intent=sample.intent,
                route=result.route,
                response=result.response,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                latency_ms=result.latency_ms,
                cost_usd=c,
            )
        )
        latencies.append(result.latency_ms)
        if result.route == "error":
            n_errors += 1
        if result.route != "llm_call":
            n_deflected += 1
        total_in += result.input_tokens
        total_out += result.output_tokens
        total_cost += c
        route_counts[result.route] = route_counts.get(result.route, 0) + 1
        if (i + 1) % 100 == 0:
            logger.info("  measure %d/%d", i + 1, len(measurement_stream))

    summary = SystemSummary(
        name=system.name,
        n_queries=len(measurement_stream),
        n_errors=n_errors,
        deflection_rate=(
            n_deflected / len(measurement_stream)
            if measurement_stream else 0.0
        ),
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        total_cost_usd=total_cost,
        p50_latency_ms=percentile(latencies, 0.50),
        p95_latency_ms=percentile(latencies, 0.95),
        p99_latency_ms=percentile(latencies, 0.99),
        mean_latency_ms=(sum(latencies) / len(latencies)) if latencies else 0.0,
        route_counts=route_counts,
    )
    return summary, records


def run_benchmark(
    dataset: Dataset,
    systems: list[SystemUnderTest],
    warmup_stream: list[Sample],
    measurement_stream: list[Sample],
    *,
    model: str,
    head_intent_share: float,
) -> BenchmarkResult:
    """Run the full benchmark across all systems on the same streams."""
    started = time.time()
    result = BenchmarkResult(
        dataset_source=dataset.source,
        n_intents=dataset.intent_count(),
        warmup_size=len(warmup_stream),
        measurement_size=len(measurement_stream),
        head_intent_share=head_intent_share,
        model=model,
        started_at=started,
        finished_at=started,  # updated at end
    )
    for system in systems:
        try:
            summary, records = run_one_system(
                system, warmup_stream, measurement_stream, model=model
            )
            result.summaries.append(summary)
            result.records.extend(records)
        finally:
            try:
                system.close()
            except Exception:
                logger.exception("close failed for %s", system.name)
    result.finished_at = time.time()
    return result


def result_to_json(result: BenchmarkResult) -> dict:
    """Serializable dict — drop into a .json file."""
    return {
        "dataset_source": result.dataset_source,
        "n_intents": result.n_intents,
        "warmup_size": result.warmup_size,
        "measurement_size": result.measurement_size,
        "head_intent_share": result.head_intent_share,
        "model": result.model,
        "started_at": result.started_at,
        "finished_at": result.finished_at,
        "summaries": [asdict(s) for s in result.summaries],
        "records": [asdict(r) for r in result.records],
    }
