"""Dataset loader + Zipfian workload generator.

Default dataset: Bitext customer-support intent dataset
(bitext/Bitext-customer-support-llm-chatbot-training-dataset on HuggingFace).
~27 intents, ~26k labeled queries, English, openly licensed.

Fallback: a small synthetic dataset if Bitext can't be loaded (offline,
network-restricted, or license concerns). The synthetic fallback is
labeled clearly in any report.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Sample:
    """One labeled query from the dataset."""

    query: str
    intent: str  # the labeled intent (used for routing-correctness check)
    canonical_response: str  # known-good response for quality scoring


@dataclass(frozen=True)
class Dataset:
    """Loaded dataset with sample list and intent → samples index."""

    samples: list[Sample]
    by_intent: dict[str, list[Sample]]
    source: str  # "bitext" or "synthetic-fallback"

    def intent_count(self) -> int:
        return len(self.by_intent)


# --- Loaders ---


def load_dataset(prefer_source: str = "bitext") -> Dataset:
    """Try the preferred source first; fall back to synthetic on any failure."""
    if prefer_source == "bitext":
        try:
            return _load_bitext()
        except Exception as e:
            logger.warning(
                "Bitext load failed (%s); falling back to synthetic dataset",
                e,
            )
    return _load_synthetic()


def _load_bitext() -> Dataset:
    """Load Bitext via HuggingFace datasets.

    Requires `datasets` package (in benchmarks dependency group).
    """
    from datasets import load_dataset as hf_load_dataset

    raw = hf_load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        split="train",
    )

    samples: list[Sample] = []
    canonical_response_by_intent: dict[str, str] = {}

    # First pass: pick a canonical response per intent (the first row's response).
    # This becomes the "right answer" for that intent in quality scoring.
    for row in raw:
        intent = row.get("intent") or row.get("category") or "unknown"
        response = row.get("response", "")
        if intent and intent not in canonical_response_by_intent and response:
            canonical_response_by_intent[intent] = response

    # Second pass: build samples
    for row in raw:
        intent = row.get("intent") or row.get("category") or "unknown"
        query = row.get("instruction") or row.get("query") or ""
        if not query:
            continue
        samples.append(
            Sample(
                query=query.strip(),
                intent=intent,
                canonical_response=canonical_response_by_intent.get(
                    intent, ""
                ),
            )
        )

    by_intent: dict[str, list[Sample]] = {}
    for s in samples:
        by_intent.setdefault(s.intent, []).append(s)

    logger.info(
        "Loaded Bitext: %d samples across %d intents",
        len(samples), len(by_intent),
    )
    return Dataset(samples=samples, by_intent=by_intent, source="bitext")


def _load_synthetic() -> Dataset:
    """Tiny hand-crafted dataset for offline / smoke runs.

    8 intents × multiple variants. Marked as synthetic in any report so
    a reader knows the numbers are illustrative, not from a real corpus.
    """
    seed = {
        "password_reset": (
            [
                "how do I reset my password",
                "I forgot my password",
                "reset password please",
                "can't log in, need new password",
                "password recovery",
            ],
            "Click 'Forgot password' on the login screen and follow the email link.",
        ),
        "refund_status": (
            [
                "where's my refund",
                "when will I get my refund",
                "refund status please",
                "I'm waiting for a refund",
                "refund timing question",
            ],
            "Refunds are processed in 5-7 business days from approval.",
        ),
        "order_tracking": (
            [
                "track my order",
                "where is my package",
                "order shipping status",
                "when will my order arrive",
                "delivery tracking",
            ],
            "You can track your order at /orders/<id>/tracking using your order number.",
        ),
        "cancel_order": (
            [
                "cancel my order",
                "I want to cancel an order",
                "how do I stop an order",
                "order cancellation",
                "cancel pending order",
            ],
            "Orders can be cancelled from /orders/<id> within 1 hour of placement.",
        ),
        "change_address": (
            [
                "change shipping address",
                "update delivery address",
                "wrong address on order",
                "ship to a different address",
                "edit address",
            ],
            "Edit shipping address from /account/addresses or contact support if shipped.",
        ),
        "payment_issue": (
            [
                "my payment failed",
                "card declined",
                "couldn't pay",
                "payment error",
                "billing problem",
            ],
            "Try a different payment method; if persistent, contact your card issuer.",
        ),
        "contact_support": (
            [
                "I need to talk to someone",
                "how do I reach support",
                "human agent please",
                "speak to a representative",
                "live chat",
            ],
            "Reach support 9-5 ET at /help or live chat from the bottom-right widget.",
        ),
        "account_delete": (
            [
                "delete my account",
                "close account",
                "remove my profile",
                "deactivate account",
                "I want to leave",
            ],
            "Account deletion is permanent. Submit request from /account/settings/delete.",
        ),
    }

    samples: list[Sample] = []
    by_intent: dict[str, list[Sample]] = {}
    for intent, (variants, response) in seed.items():
        for v in variants:
            s = Sample(query=v, intent=intent, canonical_response=response)
            samples.append(s)
            by_intent.setdefault(intent, []).append(s)

    return Dataset(
        samples=samples, by_intent=by_intent, source="synthetic-fallback"
    )


# --- Zipfian workload generator ---


def zipfian_stream(
    dataset: Dataset,
    n_queries: int,
    *,
    seed: int = 42,
    skew: float = 1.2,
) -> list[Sample]:
    """Generate a head-heavy query stream by sampling intents Zipfian-style.

    Real support traffic follows a Zipfian distribution: a few intents
    dominate, a long tail of rare intents. `skew` controls how peaked
    the head is — 1.0 is mild, 2.0 is sharp. The default 1.2 mirrors
    typical support workloads.

    Within each chosen intent, samples are drawn uniformly.
    """
    rng = random.Random(seed)
    intents = list(dataset.by_intent.keys())
    if not intents:
        raise ValueError("dataset has no intents")

    # Zipfian weights: w_i = 1 / (i + 1) ** skew, normalized.
    raw_weights = [1.0 / ((i + 1) ** skew) for i in range(len(intents))]
    total = sum(raw_weights)
    weights = [w / total for w in raw_weights]

    # Shuffle intents once so the "popular" ones aren't always the same
    # alphabetical prefix across runs.
    rng.shuffle(intents)

    stream: list[Sample] = []
    for _ in range(n_queries):
        intent = rng.choices(intents, weights=weights, k=1)[0]
        sample = rng.choice(dataset.by_intent[intent])
        stream.append(sample)
    return stream


def head_intent_share(stream: list[Sample], top_n: int = 5) -> float:
    """Return the fraction of queries handled by the top-N intents.

    Lets reports describe the workload's shape ("top 5 intents = X% of
    queries"). For Zipfian skew=1.2 over 8 intents, this is ~60-70%.
    """
    if not stream:
        return 0.0
    from collections import Counter
    counts = Counter(s.intent for s in stream)
    top = sum(c for _, c in counts.most_common(top_n))
    return top / len(stream)
