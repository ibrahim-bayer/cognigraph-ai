"""Claude API client — the LLM brain fallback.

Wraps Anthropic's Messages API behind the LLMProvider protocol. Called
only when the graph cannot confidently handle a request (LLM_FALLBACK
or LLM_ONLY routes).

Thread safety: the underlying anthropic.Anthropic client is thread-safe
for concurrent `messages.create` calls (wraps httpx.Client). The provider
itself has no mutable state after construction, so sharing a single
ClaudeLLMProvider across threads is safe.

TODO(N5): add a concurrent-calls regression test that fires N threads
against a single provider simultaneously and asserts no exceptions and
well-formed responses. Defensive; the claim is already held up by code
review but we have no automated signal if it regresses.

TODO(N8): add streaming support. The LLMProvider protocol currently has
only `generate`; a `stream(prompt, ...) -> Iterator[str]` variant would
help reduce time-to-first-token for the fallback path once there is a
concrete latency complaint to justify the additional surface area.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from cognigraph.config import CogniGraphConfig
from cognigraph.exceptions import (
    LLMError,
    LLMPermanentError,
    LLMRetriableError,
)
from cognigraph.types import LLMResponse

logger = logging.getLogger(__name__)


class ClaudeLLMProvider:
    """LLMProvider implementation backed by the Anthropic Messages API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        config: CogniGraphConfig | None = None,
        client: Any | None = None,
    ) -> None:
        """Create a Claude-backed LLM provider.

        Args:
            api_key: Anthropic API key. If None, read from the
                ANTHROPIC_API_KEY environment variable. Raises LLMError
                if neither is available (and no client is injected).
            model: Claude model id. Defaults to ``config.llm_model``.
            config: CogniGraphConfig instance. Defaults to defaults.
            client: Optional pre-built anthropic client. Test-only escape
                hatch — when provided, bypasses SDK import and API-key
                resolution. Production code should never pass this.
        """
        self._config = config or CogniGraphConfig()
        self._model = model or self._config.llm_model
        self._max_tokens = self._config.llm_max_tokens

        if client is not None:
            self._client = client
            return

        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise LLMError(
                "No Anthropic API key provided. Pass api_key= or set "
                "ANTHROPIC_API_KEY in the environment."
            )

        try:
            import anthropic

            self._client = anthropic.Anthropic(
                api_key=resolved_key,
                timeout=self._config.llm_timeout_seconds,
                max_retries=self._config.llm_max_retries,
            )
        except ImportError as e:
            raise LLMError(f"anthropic package not installed: {e}") from e
        except Exception as e:
            raise LLMError(f"Failed to initialize Anthropic client: {e}") from e

    def __repr__(self) -> str:
        # Never let the API key surface through repr/logs. The SDK client's
        # own repr may or may not mask it; ours definitely will.
        return f"ClaudeLLMProvider(model={self._model!r})"

    # --- Lifecycle ---

    def close(self) -> None:
        """Release the underlying anthropic/httpx connection pool.

        Idempotent. Short-lived scripts that create and discard many
        providers should call this to avoid leaking pooled connections.
        """
        client = getattr(self, "_client", None)
        if client is None:
            return
        close_fn = getattr(client, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception:  # best-effort; never fail on teardown
                logger.debug("anthropic client close raised", exc_info=True)

    def __enter__(self) -> ClaudeLLMProvider:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    # --- Public API ---

    def generate(
        self,
        prompt: str,
        context: list[dict] | None = None,
        system: str | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Send a prompt (+ optional context / system) to Claude.

        Args:
            prompt: Non-empty user turn to complete.
            context: Optional prior conversation turns as
                ``[{"role": "user"|"assistant", "content": "..."}]``. Must
                alternate roles and not end in ``user`` (the prompt is
                appended as the trailing user turn).
            system: Optional system prompt. Passed via Anthropic's
                top-level ``system=`` parameter, not as a message role.
            max_tokens: Optional per-call override; otherwise uses
                ``config.llm_max_tokens``.

        Returns:
            LLMResponse with text, model, latency_ms, input/output tokens.

        Raises:
            LLMPermanentError: auth, bad request, empty prompt, malformed
                context — caller should not retry.
            LLMRetriableError: rate limit, timeout, transient network —
                caller may retry with backoff.
            LLMError: any other unclassified failure.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise LLMPermanentError("prompt must be a non-empty string")

        messages = self._build_messages(prompt, context)
        effective_max_tokens = (
            max_tokens if max_tokens is not None else self._max_tokens
        )
        if effective_max_tokens <= 0:
            raise LLMPermanentError(
                f"max_tokens must be positive, got {effective_max_tokens}"
            )

        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": effective_max_tokens,
            "messages": messages,
        }
        if system is not None:
            if not isinstance(system, str):
                raise LLMPermanentError(
                    f"system prompt must be a string, got {type(system).__name__}"
                )
            create_kwargs["system"] = system

        start = time.perf_counter()
        try:
            response = self._client.messages.create(**create_kwargs)
        except LLMError:
            raise
        except Exception as e:
            raise self._classify_api_error(e) from e

        latency_ms = (time.perf_counter() - start) * 1000.0

        try:
            text = self._extract_text(response)
            model = getattr(response, "model", self._model)
            usage = getattr(response, "usage", None)
            input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
            output_tokens = int(getattr(usage, "output_tokens", 0) or 0)
        except Exception as e:
            raise LLMPermanentError(
                f"Failed to parse Claude response: {e}"
            ) from e

        return LLMResponse(
            text=text,
            model=model,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    # --- Internals ---

    def _build_messages(
        self,
        prompt: str,
        context: list[dict] | None,
    ) -> list[dict]:
        """Format prompt + context into the Messages API shape.

        Context entries must each be ``{"role": "user"|"assistant",
        "content": str}``. Roles must alternate and the final context
        entry must be ``assistant`` (so the appended user prompt is a
        valid continuation). All violations raise LLMPermanentError —
        these are programming errors, not retriable.
        """
        messages: list[dict] = []

        if context:
            prev_role: str | None = None
            for i, entry in enumerate(context):
                if not isinstance(entry, dict):
                    raise LLMPermanentError(
                        f"context[{i}] must be a dict, got {type(entry).__name__}"
                    )
                role = entry.get("role")
                content = entry.get("content")
                if role not in ("user", "assistant"):
                    raise LLMPermanentError(
                        f"context[{i}] role must be 'user' or 'assistant', "
                        f"got {role!r}"
                    )
                if not isinstance(content, str):
                    raise LLMPermanentError(
                        f"context[{i}] content must be a string, got "
                        f"{type(content).__name__}"
                    )
                if prev_role is not None and role == prev_role:
                    raise LLMPermanentError(
                        f"context roles must alternate; context[{i}] role "
                        f"{role!r} follows another {role!r}"
                    )
                messages.append({"role": role, "content": content})
                prev_role = role

            if prev_role == "user":
                raise LLMPermanentError(
                    "context must not end in a 'user' turn — the prompt is "
                    "appended as the trailing user message"
                )

        messages.append({"role": "user", "content": prompt})
        return messages

    def _extract_text(self, response: Any) -> str:
        """Pull the text payload out of a Claude Messages API response.

        Walks ``response.content`` and concatenates blocks whose
        ``type == "text"``. Non-text blocks (tool_use, images) are
        skipped — the LLM-fallback path is text-only by design. Logs a
        warning when a non-empty content list produces zero text so
        "the LLM said nothing" bugs are diagnosable.
        """
        content = getattr(response, "content", None)
        if content is None:
            return ""
        if not content:
            return ""

        parts: list[str] = []
        dropped = 0
        for block in content:
            block_type = getattr(block, "type", None)
            if block_type != "text":
                dropped += 1
                continue
            text = getattr(block, "text", None)
            if text is None:
                dropped += 1
                continue
            if not isinstance(text, str):
                raise TypeError(
                    f"text block.text must be str, got {type(text).__name__}"
                )
            parts.append(text)

        if not parts and dropped:
            logger.warning(
                "Claude response had %d content blocks but no text blocks; "
                "returning empty string",
                dropped,
            )
        return "".join(parts)

    @staticmethod
    def _classify_api_error(exc: Exception) -> LLMError:
        """Classify an SDK exception as retriable, permanent, or generic.

        The anthropic SDK exposes typed exceptions (RateLimitError,
        APITimeoutError, APIConnectionError, AuthenticationError, etc.)
        but we avoid a hard import so tests without the SDK still work
        and SDK version drift doesn't break us. Classification uses the
        exception's class name.
        """
        cls_name = type(exc).__name__
        msg = f"Claude API call failed [{cls_name}]: {exc}"

        retriable_names = {
            "RateLimitError",
            "APITimeoutError",
            "APIConnectionError",
            "InternalServerError",
            "APIStatusError",  # generic 5xx fallback
        }
        permanent_names = {
            "AuthenticationError",
            "PermissionDeniedError",
            "BadRequestError",
            "NotFoundError",
            "UnprocessableEntityError",
        }

        if cls_name in retriable_names:
            return LLMRetriableError(msg)
        if cls_name in permanent_names:
            return LLMPermanentError(msg)
        return LLMError(msg)
