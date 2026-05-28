"""Report generator — turns BenchmarkResult into a markdown report."""

from __future__ import annotations

from datetime import datetime, timezone

from benchmarks.runner import BenchmarkResult, SystemSummary


def _fmt_money(usd: float) -> str:
    if usd < 0.01:
        return f"${usd*100:.2f}¢"  # show as cents below $0.01
    return f"${usd:,.2f}"


def _fmt_pct(x: float) -> str:
    return f"{x*100:.1f}%"


def _fmt_ms(ms: float) -> str:
    if ms < 1.0:
        return f"{ms*1000:.0f}μs"
    if ms < 1000:
        return f"{ms:.1f}ms"
    return f"{ms/1000:.2f}s"


def render_markdown(result: BenchmarkResult) -> str:
    """Render a publishable markdown report."""
    out: list[str] = []
    started_dt = datetime.fromtimestamp(result.started_at, tz=timezone.utc)
    duration_s = result.finished_at - result.started_at

    out.append(f"# CogniGraph benchmark — {started_dt.strftime('%Y-%m-%d')}")
    out.append("")
    out.append(
        f"**Model**: `{result.model}` · "
        f"**Dataset**: `{result.dataset_source}` · "
        f"**Wall-clock duration**: {duration_s/60:.1f} min"
    )
    out.append("")

    # --- Workload description ---
    out.append("## Workload")
    out.append("")
    out.append(
        f"- {result.measurement_size} measurement queries + "
        f"{result.warmup_size} warm-up queries"
    )
    out.append(f"- {result.n_intents} intents in the source dataset")
    out.append(
        f"- Top 5 intents cover {_fmt_pct(result.head_intent_share)} "
        f"of measurement queries (head-heavy Zipfian distribution)"
    )
    if result.dataset_source.startswith("synthetic"):
        out.append(
            "- ⚠️ This run used the **synthetic fallback dataset** — "
            "the numbers are illustrative, not from a real corpus."
        )
    out.append("")

    # --- Headline comparison ---
    out.append("## Headline numbers")
    out.append("")
    out.append(
        "| System | Deflection | Cost / 1k queries | p50 latency | "
        "p95 latency | p99 latency | Errors |"
    )
    out.append(
        "|---|---:|---:|---:|---:|---:|---:|"
    )
    for s in result.summaries:
        cost_per_1k = (
            (s.total_cost_usd / s.n_queries) * 1000.0 if s.n_queries else 0
        )
        out.append(
            f"| `{s.name}` | {_fmt_pct(s.deflection_rate)} | "
            f"{_fmt_money(cost_per_1k)} | "
            f"{_fmt_ms(s.p50_latency_ms)} | "
            f"{_fmt_ms(s.p95_latency_ms)} | "
            f"{_fmt_ms(s.p99_latency_ms)} | "
            f"{s.n_errors} |"
        )
    out.append("")

    # --- Cost delta vs raw_llm baseline ---
    raw = next((s for s in result.summaries if s.name == "raw_llm"), None)
    if raw is not None and raw.total_cost_usd > 0:
        out.append("## Savings vs `raw_llm` baseline")
        out.append("")
        out.append(
            "| System | Cost reduction | Latency reduction (p50) |"
        )
        out.append("|---|---:|---:|")
        for s in result.summaries:
            if s.name == "raw_llm":
                continue
            cost_delta = (
                1.0 - (s.total_cost_usd / raw.total_cost_usd)
                if raw.total_cost_usd else 0.0
            )
            lat_delta = (
                1.0 - (s.p50_latency_ms / raw.p50_latency_ms)
                if raw.p50_latency_ms else 0.0
            )
            out.append(
                f"| `{s.name}` | {_fmt_pct(cost_delta)} | "
                f"{_fmt_pct(lat_delta)} |"
            )
        out.append("")

    # --- Route breakdown ---
    out.append("## Route breakdown")
    out.append("")
    out.append(
        "| System | graph_hit | cache_hit | llm_call | error |"
    )
    out.append("|---|---:|---:|---:|---:|")
    for s in result.summaries:
        rc = s.route_counts
        out.append(
            f"| `{s.name}` | {rc.get('graph_hit', 0)} | "
            f"{rc.get('cache_hit', 0)} | {rc.get('llm_call', 0)} | "
            f"{rc.get('error', 0)} |"
        )
    out.append("")

    # --- Token + cost detail ---
    out.append("## Token + cost detail")
    out.append("")
    out.append(
        "| System | Total input tokens | Total output tokens | "
        "Total cost | Cost / query |"
    )
    out.append("|---|---:|---:|---:|---:|")
    for s in result.summaries:
        cost_per = s.total_cost_usd / s.n_queries if s.n_queries else 0
        out.append(
            f"| `{s.name}` | {s.total_input_tokens:,} | "
            f"{s.total_output_tokens:,} | "
            f"{_fmt_money(s.total_cost_usd)} | "
            f"{_fmt_money(cost_per)} |"
        )
    out.append("")

    # --- Methodology ---
    out.append("## Methodology")
    out.append("")
    out.append(
        "- Each system runs against the **same query stream** "
        "(same dataset, same Zipfian seed, same model). Cache /"
        " graph systems also see a warm-up phase; raw LLM does not "
        "(it's stateless, warm-up would only waste tokens)."
    )
    out.append(
        "- Latency is wall-clock from query in → response out, "
        "including all in-process work (normalize, embed, match, "
        "safety, LLM call when applicable)."
    )
    out.append(
        "- Token counts come from the Anthropic SDK's `usage` field. "
        "Cost is derived from published pricing at run time — see "
        "`benchmarks/runner.py::PRICING`."
    )
    out.append(
        "- `deflection` = share of queries that did NOT hit the LLM. "
        "For `raw_llm` this is structurally 0%."
    )
    out.append(
        "- Quality is not auto-scored in this run; spot-check the "
        "per-query records in the accompanying JSON file to verify "
        "responses are sensible. A future iteration could add cosine "
        "similarity vs canonical responses or a judge-LLM rating."
    )
    out.append("")

    # --- Caveats ---
    out.append("## Caveats")
    out.append("")
    out.append(
        "- Single workload, single dataset, single model. Real "
        "workloads vary. The pattern is more important than the "
        "absolute numbers."
    )
    out.append(
        "- CogniGraph's `learning_starting_confidence` (0.5) is below "
        "the `confidence_threshold` (0.7), so newly-crystallized nodes "
        "first route LLM_FALLBACK and only graduate to GRAPH_DIRECT "
        "after additional reinforcement. With a longer warm-up and / or "
        "tuned thresholds, deflection improves materially."
    )
    out.append(
        "- No GPTCache comparison in this run. The wrapper is on the "
        "roadmap; today the headline is CogniGraph vs raw LLM."
    )
    out.append("")

    return "\n".join(out)
