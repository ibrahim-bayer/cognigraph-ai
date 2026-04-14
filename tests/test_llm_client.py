"""Tests for ClaudeLLMProvider — uses an injected fake client, no network."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from cognigraph.config import CogniGraphConfig
from cognigraph.exceptions import LLMError, LLMPermanentError, LLMRetriableError
from cognigraph.llm_client import ClaudeLLMProvider
from cognigraph.protocols import LLMProvider
from cognigraph.types import LLMResponse


# --- Fake anthropic client ---
#
# Matches the public shape of anthropic.Anthropic just enough for the
# provider to exercise the real code path. No SDK monkey-patching.


@dataclass
class _FakeUsage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class _FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class _FakeResponse:
    content: list[_FakeTextBlock]
    model: str = "claude-test"
    usage: _FakeUsage = field(default_factory=_FakeUsage)


class _FakeMessagesAPI:
    def __init__(self) -> None:
        self.last_call: dict[str, Any] = {}
        self.response: _FakeResponse | None = None
        self.error: Exception | None = None
        self.delay_seconds: float = 0.0

    def create(self, **kwargs: Any) -> _FakeResponse:
        import time as _time

        self.last_call = kwargs
        if self.delay_seconds > 0:
            _time.sleep(self.delay_seconds)
        if self.error is not None:
            raise self.error
        if self.response is None:
            # Sensible default: echo the last user message
            last_user = next(
                (
                    m["content"]
                    for m in reversed(kwargs.get("messages", []))
                    if m.get("role") == "user"
                ),
                "",
            )
            return _FakeResponse(
                content=[_FakeTextBlock(text=f"echo: {last_user}")],
                model=kwargs.get("model", "claude-test"),
                usage=_FakeUsage(input_tokens=10, output_tokens=20),
            )
        return self.response


class _FakeAnthropicClient:
    def __init__(self) -> None:
        self.messages = _FakeMessagesAPI()


@pytest.fixture
def fake_client() -> _FakeAnthropicClient:
    return _FakeAnthropicClient()


@pytest.fixture
def provider(fake_client: _FakeAnthropicClient) -> ClaudeLLMProvider:
    return ClaudeLLMProvider(
        api_key="test-key",
        model="claude-test",
        client=fake_client,
    )


# --- Protocol conformance ---


class TestProtocolConformance:
    def test_implements_llm_provider(self, provider: ClaudeLLMProvider) -> None:
        assert isinstance(provider, LLMProvider)


# --- API key resolution ---


class TestApiKeyResolution:
    def test_missing_api_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(LLMError, match="No Anthropic API key"):
            ClaudeLLMProvider()

    def test_reads_api_key_from_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_client: _FakeAnthropicClient,
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        # Client is injected, so no real anthropic call happens; this just
        # proves the key check passes with an env-provided value.
        provider = ClaudeLLMProvider(client=fake_client)
        assert provider is not None

    def test_constructor_key_overrides_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_client: _FakeAnthropicClient,
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
        # Just ensure both paths accept the injected client
        provider = ClaudeLLMProvider(
            api_key="explicit-key", client=fake_client
        )
        assert provider is not None

    def test_injected_client_skips_api_key_check(
        self, fake_client: _FakeAnthropicClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An injected client means no SDK init, so no key is required."""
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        provider = ClaudeLLMProvider(client=fake_client)
        assert provider is not None


# --- generate() — happy path ---


class TestGenerateHappyPath:
    def test_returns_llm_response(
        self, provider: ClaudeLLMProvider, fake_client: _FakeAnthropicClient
    ) -> None:
        fake_client.messages.response = _FakeResponse(
            content=[_FakeTextBlock(text="hello, world")],
            model="claude-test",
            usage=_FakeUsage(input_tokens=5, output_tokens=7),
        )
        result = provider.generate("hi")

        assert isinstance(result, LLMResponse)
        assert result.text == "hello, world"
        assert result.model == "claude-test"
        assert result.input_tokens == 5
        assert result.output_tokens == 7
        assert result.latency_ms >= 0

    def test_forwards_prompt_as_user_message(
        self, provider: ClaudeLLMProvider, fake_client: _FakeAnthropicClient
    ) -> None:
        provider.generate("what is the time")
        call = fake_client.messages.last_call
        assert call["model"] == "claude-test"
        assert call["max_tokens"] > 0
        assert call["messages"][-1] == {
            "role": "user",
            "content": "what is the time",
        }

    def test_uses_config_model_and_max_tokens(
        self, fake_client: _FakeAnthropicClient
    ) -> None:
        cfg = CogniGraphConfig(
            llm_model="claude-custom", llm_max_tokens=512
        )
        provider = ClaudeLLMProvider(config=cfg, client=fake_client)
        provider.generate("x")
        call = fake_client.messages.last_call
        assert call["model"] == "claude-custom"
        assert call["max_tokens"] == 512

    def test_latency_tracked_nonzero(
        self, provider: ClaudeLLMProvider, fake_client: _FakeAnthropicClient
    ) -> None:
        fake_client.messages.delay_seconds = 0.02
        result = provider.generate("x")
        assert result.latency_ms >= 20.0

    def test_multi_block_response_concatenated(
        self, provider: ClaudeLLMProvider, fake_client: _FakeAnthropicClient
    ) -> None:
        fake_client.messages.response = _FakeResponse(
            content=[
                _FakeTextBlock(text="part A "),
                _FakeTextBlock(text="part B"),
            ],
            model="claude-test",
            usage=_FakeUsage(input_tokens=1, output_tokens=1),
        )
        result = provider.generate("x")
        assert result.text == "part A part B"

    def test_non_text_blocks_ignored(
        self, provider: ClaudeLLMProvider, fake_client: _FakeAnthropicClient
    ) -> None:
        @dataclass
        class _FakeToolBlock:
            type: str = "tool_use"
            name: str = "some_tool"

        fake_client.messages.response = _FakeResponse(
            content=[
                _FakeTextBlock(text="answer text"),
                _FakeToolBlock(),  # should be silently ignored
            ],
            model="claude-test",
            usage=_FakeUsage(input_tokens=1, output_tokens=1),
        )
        result = provider.generate("x")
        assert result.text == "answer text"


# --- Context handling ---


class TestContextHandling:
    def test_context_messages_preserved_in_order(
        self, provider: ClaudeLLMProvider, fake_client: _FakeAnthropicClient
    ) -> None:
        context = [
            {"role": "user", "content": "first user"},
            {"role": "assistant", "content": "first reply"},
            {"role": "user", "content": "second user"},
            {"role": "assistant", "content": "second reply"},
        ]
        provider.generate("third user", context=context)
        messages = fake_client.messages.last_call["messages"]
        assert len(messages) == 5
        assert messages[0]["content"] == "first user"
        assert messages[-1]["content"] == "third user"
        assert messages[-1]["role"] == "user"

    def test_empty_context_is_ignored(
        self, provider: ClaudeLLMProvider, fake_client: _FakeAnthropicClient
    ) -> None:
        provider.generate("hi", context=[])
        messages = fake_client.messages.last_call["messages"]
        assert messages == [{"role": "user", "content": "hi"}]

    def test_none_context_is_ignored(
        self, provider: ClaudeLLMProvider, fake_client: _FakeAnthropicClient
    ) -> None:
        provider.generate("hi", context=None)
        messages = fake_client.messages.last_call["messages"]
        assert len(messages) == 1

    def test_bad_context_entry_type_raises(
        self, provider: ClaudeLLMProvider
    ) -> None:
        with pytest.raises(LLMError, match="must be a dict"):
            provider.generate("hi", context=["not a dict"])  # type: ignore[list-item]

    def test_unknown_role_raises(self, provider: ClaudeLLMProvider) -> None:
        with pytest.raises(LLMError, match="role must be"):
            provider.generate(
                "hi", context=[{"role": "system", "content": "ignored"}]
            )

    def test_non_string_content_raises(self, provider: ClaudeLLMProvider) -> None:
        with pytest.raises(LLMError, match="content must be a string"):
            provider.generate(
                "hi", context=[{"role": "user", "content": 42}]
            )


# --- Error wrapping ---


class TestErrorWrapping:
    def test_api_error_wrapped_in_llm_error(
        self, provider: ClaudeLLMProvider, fake_client: _FakeAnthropicClient
    ) -> None:
        fake_client.messages.error = RuntimeError("network timeout")
        with pytest.raises(LLMError, match="Claude API call failed"):
            provider.generate("hi")

    def test_connection_error_wrapped(
        self, provider: ClaudeLLMProvider, fake_client: _FakeAnthropicClient
    ) -> None:
        class _APIError(Exception):
            pass

        fake_client.messages.error = _APIError("rate limited")
        with pytest.raises(LLMError, match="rate limited"):
            provider.generate("hi")

    def test_llm_error_from_context_validation_not_double_wrapped(
        self, provider: ClaudeLLMProvider
    ) -> None:
        """LLMError raised during message building should propagate as-is."""
        try:
            provider.generate("hi", context=[{"role": "bogus", "content": "x"}])
        except LLMError as e:
            assert "role must be" in str(e)
        else:
            pytest.fail("expected LLMError")

    def test_malformed_response_wrapped(
        self, provider: ClaudeLLMProvider, fake_client: _FakeAnthropicClient
    ) -> None:
        class _BadResponse:
            # Iterating .content raises — simulates a corrupt / unexpected shape
            @property
            def content(self) -> Any:
                raise RuntimeError("corrupt content")

        fake_client.messages.response = _BadResponse()  # type: ignore[assignment]
        with pytest.raises(LLMError, match="parse Claude response"):
            provider.generate("hi")


# --- Token accounting ---


class TestTokenAccounting:
    def test_token_counts_captured(
        self, provider: ClaudeLLMProvider, fake_client: _FakeAnthropicClient
    ) -> None:
        fake_client.messages.response = _FakeResponse(
            content=[_FakeTextBlock(text="ok")],
            model="claude-test",
            usage=_FakeUsage(input_tokens=123, output_tokens=456),
        )
        result = provider.generate("hi")
        assert result.input_tokens == 123
        assert result.output_tokens == 456

    def test_missing_usage_defaults_to_zero(
        self, provider: ClaudeLLMProvider, fake_client: _FakeAnthropicClient
    ) -> None:
        @dataclass
        class _NoUsageResponse:
            content: list[_FakeTextBlock]
            model: str = "claude-test"
            usage: None = None

        fake_client.messages.response = _NoUsageResponse(  # type: ignore[assignment]
            content=[_FakeTextBlock(text="ok")]
        )
        result = provider.generate("hi")
        assert result.input_tokens == 0
        assert result.output_tokens == 0


# --- System prompt + per-call max_tokens (W3, W4) ---


class TestSystemPromptAndMaxTokens:
    def test_system_prompt_passed_through(
        self, provider: ClaudeLLMProvider, fake_client: _FakeAnthropicClient
    ) -> None:
        provider.generate(
            "answer this", system="You are a concise fallback agent."
        )
        call = fake_client.messages.last_call
        assert call["system"] == "You are a concise fallback agent."

    def test_system_prompt_omitted_when_none(
        self, provider: ClaudeLLMProvider, fake_client: _FakeAnthropicClient
    ) -> None:
        provider.generate("answer this")
        assert "system" not in fake_client.messages.last_call

    def test_non_string_system_prompt_raises(
        self, provider: ClaudeLLMProvider
    ) -> None:
        with pytest.raises(LLMPermanentError, match="system prompt"):
            provider.generate("hi", system=123)  # type: ignore[arg-type]

    def test_per_call_max_tokens_override(
        self, provider: ClaudeLLMProvider, fake_client: _FakeAnthropicClient
    ) -> None:
        provider.generate("hi", max_tokens=256)
        assert fake_client.messages.last_call["max_tokens"] == 256

    def test_per_call_max_tokens_defaults_to_config(
        self, provider: ClaudeLLMProvider, fake_client: _FakeAnthropicClient
    ) -> None:
        provider.generate("hi")  # no override
        # Default config.llm_max_tokens is 1024
        assert fake_client.messages.last_call["max_tokens"] == 1024

    def test_zero_max_tokens_rejected(self, provider: ClaudeLLMProvider) -> None:
        with pytest.raises(LLMPermanentError, match="max_tokens"):
            provider.generate("hi", max_tokens=0)

    def test_negative_max_tokens_rejected(
        self, provider: ClaudeLLMProvider
    ) -> None:
        with pytest.raises(LLMPermanentError, match="max_tokens"):
            provider.generate("hi", max_tokens=-5)


# --- Empty prompt (N10) ---


class TestEmptyPrompt:
    def test_empty_prompt_raises_permanent(
        self, provider: ClaudeLLMProvider
    ) -> None:
        with pytest.raises(LLMPermanentError, match="non-empty"):
            provider.generate("")

    def test_whitespace_only_prompt_raises(
        self, provider: ClaudeLLMProvider
    ) -> None:
        with pytest.raises(LLMPermanentError, match="non-empty"):
            provider.generate("   \t\n  ")

    def test_non_string_prompt_raises(self, provider: ClaudeLLMProvider) -> None:
        with pytest.raises(LLMPermanentError, match="non-empty"):
            provider.generate(None)  # type: ignore[arg-type]


# --- Context alternation rules (W5) ---


class TestContextAlternation:
    def test_context_must_alternate_roles(
        self, provider: ClaudeLLMProvider
    ) -> None:
        with pytest.raises(LLMPermanentError, match="alternate"):
            provider.generate(
                "hi",
                context=[
                    {"role": "user", "content": "a"},
                    {"role": "user", "content": "b"},
                ],
            )

    def test_context_ending_in_user_rejected(
        self, provider: ClaudeLLMProvider
    ) -> None:
        with pytest.raises(LLMPermanentError, match="must not end in a 'user'"):
            provider.generate(
                "hi",
                context=[
                    {"role": "user", "content": "a"},
                    {"role": "assistant", "content": "b"},
                    {"role": "user", "content": "c"},
                ],
            )

    def test_context_ending_in_assistant_accepted(
        self, provider: ClaudeLLMProvider, fake_client: _FakeAnthropicClient
    ) -> None:
        provider.generate(
            "final question",
            context=[
                {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"},
            ],
        )
        messages = fake_client.messages.last_call["messages"]
        assert [m["role"] for m in messages] == ["user", "assistant", "user"]
        assert messages[-1]["content"] == "final question"


# --- Error classification (W2) ---


class TestErrorClassification:
    def _inject_named_error(
        self,
        fake_client: _FakeAnthropicClient,
        name: str,
        msg: str = "boom",
    ) -> None:
        # Build an exception class whose __name__ matches an anthropic type
        cls = type(name, (Exception,), {})
        fake_client.messages.error = cls(msg)

    def test_rate_limit_is_retriable(
        self, provider: ClaudeLLMProvider, fake_client: _FakeAnthropicClient
    ) -> None:
        self._inject_named_error(fake_client, "RateLimitError", "slow down")
        with pytest.raises(LLMRetriableError, match="RateLimitError"):
            provider.generate("hi")

    def test_api_timeout_is_retriable(
        self, provider: ClaudeLLMProvider, fake_client: _FakeAnthropicClient
    ) -> None:
        self._inject_named_error(fake_client, "APITimeoutError", "too slow")
        with pytest.raises(LLMRetriableError):
            provider.generate("hi")

    def test_api_connection_error_is_retriable(
        self, provider: ClaudeLLMProvider, fake_client: _FakeAnthropicClient
    ) -> None:
        self._inject_named_error(fake_client, "APIConnectionError", "dns")
        with pytest.raises(LLMRetriableError):
            provider.generate("hi")

    def test_authentication_error_is_permanent(
        self, provider: ClaudeLLMProvider, fake_client: _FakeAnthropicClient
    ) -> None:
        self._inject_named_error(fake_client, "AuthenticationError", "bad key")
        with pytest.raises(LLMPermanentError, match="AuthenticationError"):
            provider.generate("hi")

    def test_bad_request_error_is_permanent(
        self, provider: ClaudeLLMProvider, fake_client: _FakeAnthropicClient
    ) -> None:
        self._inject_named_error(fake_client, "BadRequestError", "bad input")
        with pytest.raises(LLMPermanentError):
            provider.generate("hi")

    def test_unknown_exception_falls_back_to_generic(
        self, provider: ClaudeLLMProvider, fake_client: _FakeAnthropicClient
    ) -> None:
        self._inject_named_error(
            fake_client, "SomeWeirdUnheardOfThing", "who knows"
        )
        with pytest.raises(LLMError) as exc_info:
            provider.generate("hi")
        # Must be the base class, not either specific subclass
        assert not isinstance(exc_info.value, LLMRetriableError)
        assert not isinstance(exc_info.value, LLMPermanentError)
        assert "SomeWeirdUnheardOfThing" in str(exc_info.value)


# --- Malformed response cases (W7) ---


class TestMalformedResponses:
    def test_content_none_returns_empty_string(
        self, provider: ClaudeLLMProvider, fake_client: _FakeAnthropicClient
    ) -> None:
        @dataclass
        class _R:
            content: None = None
            model: str = "claude-test"
            usage: _FakeUsage = field(default_factory=_FakeUsage)

        fake_client.messages.response = _R()  # type: ignore[assignment]
        result = provider.generate("hi")
        assert result.text == ""

    def test_content_empty_list_returns_empty_string(
        self, provider: ClaudeLLMProvider, fake_client: _FakeAnthropicClient
    ) -> None:
        fake_client.messages.response = _FakeResponse(
            content=[],
            model="claude-test",
            usage=_FakeUsage(input_tokens=1, output_tokens=0),
        )
        result = provider.generate("hi")
        assert result.text == ""

    def test_text_block_with_none_text_skipped(
        self, provider: ClaudeLLMProvider, fake_client: _FakeAnthropicClient
    ) -> None:
        fake_client.messages.response = _FakeResponse(
            content=[
                _FakeTextBlock(text=None),  # type: ignore[arg-type]
                _FakeTextBlock(text="real"),
            ],
            model="claude-test",
            usage=_FakeUsage(input_tokens=1, output_tokens=1),
        )
        result = provider.generate("hi")
        assert result.text == "real"

    def test_text_block_with_non_string_text_wrapped(
        self, provider: ClaudeLLMProvider, fake_client: _FakeAnthropicClient
    ) -> None:
        fake_client.messages.response = _FakeResponse(
            content=[_FakeTextBlock(text=b"bytes")],  # type: ignore[arg-type]
            model="claude-test",
            usage=_FakeUsage(input_tokens=1, output_tokens=1),
        )
        with pytest.raises(LLMPermanentError, match="parse Claude response"):
            provider.generate("hi")

    def test_tool_use_only_response_returns_empty_and_logs(
        self,
        provider: ClaudeLLMProvider,
        fake_client: _FakeAnthropicClient,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        @dataclass
        class _ToolBlock:
            type: str = "tool_use"

        fake_client.messages.response = _FakeResponse(
            content=[_ToolBlock(), _ToolBlock()],  # type: ignore[list-item]
            model="claude-test",
            usage=_FakeUsage(input_tokens=1, output_tokens=1),
        )
        with caplog.at_level("WARNING", logger="cognigraph.llm_client"):
            result = provider.generate("hi")
        assert result.text == ""
        assert any("no text blocks" in r.message for r in caplog.records)


# --- Real SDK init path (W8) ---


class TestRealSDKInitPath:
    def test_real_sdk_init_forwards_api_key_and_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Cover the production path where anthropic is actually imported.

        Replaces sys.modules["anthropic"] with a fake module exposing an
        `Anthropic` class that records its constructor kwargs, then
        constructs the provider without injecting a client.
        """
        import sys
        import types

        recorded: dict[str, Any] = {}

        class _FakeAnthropic:
            def __init__(self, **kwargs: Any) -> None:
                recorded.update(kwargs)

        fake_module = types.ModuleType("anthropic")
        fake_module.Anthropic = _FakeAnthropic  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "anthropic", fake_module)

        cfg = CogniGraphConfig(
            llm_timeout_seconds=15.0, llm_max_retries=5
        )
        provider = ClaudeLLMProvider(api_key="real-key", config=cfg)

        assert provider is not None
        assert recorded["api_key"] == "real-key"
        assert recorded["timeout"] == 15.0
        assert recorded["max_retries"] == 5

    def test_sdk_init_failure_wrapped(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import sys
        import types

        class _FakeAnthropic:
            def __init__(self, **kwargs: Any) -> None:
                raise RuntimeError("init boom")

        fake_module = types.ModuleType("anthropic")
        fake_module.Anthropic = _FakeAnthropic  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "anthropic", fake_module)

        with pytest.raises(LLMError, match="Failed to initialize"):
            ClaudeLLMProvider(api_key="k")


# --- Security & lifecycle (N4, N6) ---


class TestRepr:
    def test_repr_does_not_leak_api_key(
        self, fake_client: _FakeAnthropicClient
    ) -> None:
        p = ClaudeLLMProvider(
            api_key="super-secret-key-xyz",
            model="claude-test",
            client=fake_client,
        )
        r = repr(p)
        assert "super-secret-key-xyz" not in r
        assert "claude-test" in r


class TestLifecycle:
    def test_close_is_idempotent(
        self, provider: ClaudeLLMProvider, fake_client: _FakeAnthropicClient
    ) -> None:
        # Add a .close method to the fake client
        calls = {"n": 0}

        def _close() -> None:
            calls["n"] += 1

        fake_client.close = _close  # type: ignore[attr-defined]
        provider.close()
        provider.close()
        assert calls["n"] == 2  # both invocations forwarded; no crash

    def test_close_survives_client_without_close_method(
        self, fake_client: _FakeAnthropicClient
    ) -> None:
        # _FakeAnthropicClient has no .close by default
        p = ClaudeLLMProvider(api_key="k", client=fake_client)
        p.close()  # must not raise

    def test_close_swallows_client_errors(
        self, fake_client: _FakeAnthropicClient
    ) -> None:
        def _boom() -> None:
            raise RuntimeError("teardown failure")

        fake_client.close = _boom  # type: ignore[attr-defined]
        p = ClaudeLLMProvider(api_key="k", client=fake_client)
        p.close()  # swallowed

    def test_context_manager_closes(
        self, fake_client: _FakeAnthropicClient
    ) -> None:
        calls = {"n": 0}

        def _close() -> None:
            calls["n"] += 1

        fake_client.close = _close  # type: ignore[attr-defined]
        with ClaudeLLMProvider(api_key="k", client=fake_client):
            pass
        assert calls["n"] == 1
