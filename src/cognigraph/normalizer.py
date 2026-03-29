"""Input normalization — the sensory processing layer."""

from __future__ import annotations

import re
import unicodedata

from cognigraph.config import CogniGraphConfig
from cognigraph.models import NormalizedInput

_WHITESPACE_RE = re.compile(r"\s+")

# Control characters (C0, DEL, C1) excluding common whitespace (\t, \n, \r)
_CONTROL_CHAR_RE = re.compile(
    r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]"
)

# Zero-width and invisible formatting characters
_ZERO_WIDTH_RE = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f\u2060\u2061\u2062\u2063\u2064\ufeff]"
)


class InputNormalizer:
    """Canonicalize raw user text for consistent graph matching."""

    def __init__(self, config: CogniGraphConfig | None = None) -> None:
        self._max_length = (config or CogniGraphConfig()).max_input_length

    def normalize(self, raw_text: str) -> NormalizedInput:
        text = raw_text[:self._max_length]
        text = text.strip()
        text = _CONTROL_CHAR_RE.sub("", text)
        text = _ZERO_WIDTH_RE.sub("", text)
        text = unicodedata.normalize("NFKC", text)
        text = text.lower()
        text = _WHITESPACE_RE.sub(" ", text)
        text = text.strip()
        return NormalizedInput(original=raw_text, normalized=text)
