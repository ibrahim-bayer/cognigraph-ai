"""Tests for InputNormalizer."""

from __future__ import annotations

from cognigraph.config import CogniGraphConfig
from cognigraph.models import NormalizedInput
from cognigraph.normalizer import InputNormalizer


class TestInputNormalizer:
    def setup_method(self) -> None:
        self.normalizer = InputNormalizer(CogniGraphConfig())

    def test_basic_normalization(self) -> None:
        result = self.normalizer.normalize("  Hello   World  ")
        assert result.normalized == "hello world"
        assert result.original == "  Hello   World  "

    def test_unicode_nfkc(self) -> None:
        result = self.normalizer.normalize("ﬁnance")
        assert result.normalized == "finance"

    def test_mixed_case(self) -> None:
        result = self.normalizer.normalize("HeLLo")
        assert result.normalized == "hello"

    def test_empty_string(self) -> None:
        result = self.normalizer.normalize("")
        assert result.normalized == ""
        assert result.original == ""

    def test_whitespace_only(self) -> None:
        result = self.normalizer.normalize("   \t\n  ")
        assert result.normalized == ""

    def test_tabs_and_newlines_collapsed(self) -> None:
        result = self.normalizer.normalize("hello\t\tworld\n\nfoo")
        assert result.normalized == "hello world foo"

    def test_preserves_meaningful_content(self) -> None:
        result = self.normalizer.normalize("What is your name?")
        assert result.normalized == "what is your name?"

    def test_returns_normalized_input(self) -> None:
        result = self.normalizer.normalize("Test")
        assert isinstance(result, NormalizedInput)

    def test_original_preserved(self) -> None:
        raw = "  Mixed   CASE  input  "
        result = self.normalizer.normalize(raw)
        assert result.original == raw

    def test_embedding_is_none(self) -> None:
        result = self.normalizer.normalize("hello")
        assert result.embedding is None

    def test_unicode_accents(self) -> None:
        result = self.normalizer.normalize("café résumé")
        assert result.normalized == "café résumé"

    def test_fullwidth_characters(self) -> None:
        # NFKC normalizes fullwidth Latin to ASCII
        result = self.normalizer.normalize("Ｈｅｌｌｏ")
        assert result.normalized == "hello"

    def test_multiple_spaces_between_words(self) -> None:
        result = self.normalizer.normalize("one    two     three")
        assert result.normalized == "one two three"

    def test_strips_control_characters(self) -> None:
        result = self.normalizer.normalize("hel\x00lo\x07 world\x1f")
        assert result.normalized == "hello world"

    def test_strips_zero_width_characters(self) -> None:
        result = self.normalizer.normalize("hel\u200blo\u200d world\ufeff")
        assert result.normalized == "hello world"

    def test_zero_width_chars_dont_affect_matching(self) -> None:
        r1 = self.normalizer.normalize("hello")
        r2 = self.normalizer.normalize("hel\u200blo")
        assert r1.normalized == r2.normalized

    def test_truncates_long_input(self) -> None:
        config = CogniGraphConfig(max_input_length=20)
        normalizer = InputNormalizer(config)
        long_input = "a" * 100
        result = normalizer.normalize(long_input)
        assert len(result.normalized) <= 20
        assert result.original == long_input

    def test_default_max_length(self) -> None:
        normalizer = InputNormalizer()
        result = normalizer.normalize("hello")
        assert result.normalized == "hello"

    def test_control_chars_at_boundaries_stripped_clean(self) -> None:
        result = self.normalizer.normalize("\x00 hello \x00")
        assert result.normalized == "hello"
