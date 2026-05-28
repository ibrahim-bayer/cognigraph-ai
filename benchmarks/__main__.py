"""CLI entry point.

Usage:
    uv run python -m benchmarks --help
    uv run python -m benchmarks smoke               # mock LLM only, no token
    uv run python -m benchmarks run --model haiku   # real LLM, ANTHROPIC_API_KEY required
    uv run python -m benchmarks run --model sonnet --queries 1000

The smoke subcommand validates the wiring end-to-end with the deterministic
MockLLM at zero token cost. It does NOT produce publishable latency numbers
(the mock sleeps a fixed range); it ONLY proves the harness, dataset loader,
and report renderer all run cleanly together.

The run subcommand executes the real benchmark and produces the publishable
artifacts under `benchmarks/results/<date>.{md,json}`.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.dataset import head_intent_share, load_dataset, zipfian_stream
from benchmarks.report import render_markdown
from benchmarks.runner import BenchmarkResult, result_to_json, run_benchmark
from benchmarks.systems import SystemUnderTest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("benchmarks")


_MODEL_ALIASES = {
    "haiku": "claude-haiku-4-5",
    "sonnet": "claude-sonnet-4-5",
}


def _resolve_model(alias_or_id: str) -> str:
    return _MODEL_ALIASES.get(alias_or_id, alias_or_id)


def _build_systems(mode: str, model: str) -> list[SystemUnderTest]:
    """Build the list of systems-under-test for the given mode."""
    from cognigraph.config import CogniGraphConfig

    if mode == "smoke":
        from benchmarks.systems.mock_llm import MockLLMSystem
        return [MockLLMSystem(seed=0)]

    # Real run — instantiate raw LLM + CogniGraph against the chosen model
    config = CogniGraphConfig(llm_model=model)

    from benchmarks.systems.cognigraph_system import CogniGraphSystem
    from benchmarks.systems.raw_llm import RawLLMSystem

    return [RawLLMSystem(config=config), CogniGraphSystem(config=config)]


def _write_results(result: BenchmarkResult, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    date_tag = datetime.fromtimestamp(
        result.started_at, tz=timezone.utc
    ).strftime("%Y-%m-%d_%H%M")

    json_path = out_dir / f"{date_tag}.json"
    md_path = out_dir / f"{date_tag}.md"

    json_path.write_text(json.dumps(result_to_json(result), indent=2))
    md_path.write_text(render_markdown(result))

    logger.info("results: %s", md_path)
    logger.info("raw    : %s", json_path)
    return md_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmarks")
    sub = parser.add_subparsers(dest="cmd", required=True)

    smoke = sub.add_parser(
        "smoke",
        help="Validate the harness end-to-end with the mock LLM (no token).",
    )
    smoke.add_argument("--queries", type=int, default=50)
    smoke.add_argument("--warmup", type=int, default=20)

    run = sub.add_parser(
        "run",
        help="Run the real benchmark (requires ANTHROPIC_API_KEY).",
    )
    run.add_argument(
        "--model",
        default="haiku",
        help=(
            "Model alias (haiku, sonnet) or full Claude model id. "
            "Defaults to haiku for cost. Use sonnet for the published run."
        ),
    )
    run.add_argument("--queries", type=int, default=200)
    run.add_argument("--warmup", type=int, default=100)
    run.add_argument(
        "--dataset",
        default="bitext",
        help="Dataset source (bitext or synthetic). Defaults to bitext.",
    )

    args = parser.parse_args(argv)
    mode = args.cmd
    model = "mock" if mode == "smoke" else _resolve_model(args.model)

    logger.info("mode=%s model=%s", mode, model)

    dataset = load_dataset(
        prefer_source=getattr(args, "dataset", "bitext") if mode == "run" else "synthetic"
    )
    logger.info(
        "dataset: source=%s samples=%d intents=%d",
        dataset.source, len(dataset.samples), dataset.intent_count(),
    )

    warmup_stream = zipfian_stream(dataset, args.warmup, seed=42)
    measurement_stream = zipfian_stream(dataset, args.queries, seed=7)
    share = head_intent_share(measurement_stream, top_n=5)
    logger.info("workload: top-5 intents = %.1f%% of queries", share * 100)

    systems = _build_systems(mode, model)
    result = run_benchmark(
        dataset, systems,
        warmup_stream, measurement_stream,
        model=model, head_intent_share=share,
    )

    out_dir = Path("benchmarks/results")
    md = _write_results(result, out_dir)
    print(md.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
