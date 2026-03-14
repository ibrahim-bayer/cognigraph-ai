"""Shared test fixtures for CogniGraph."""

from __future__ import annotations

import pytest

from cognigraph.config import CogniGraphConfig


@pytest.fixture
def config() -> CogniGraphConfig:
    """Default config for tests."""
    return CogniGraphConfig()
