"""Shared pytest configuration for scDICE-metrics."""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: long-running or IO-heavy tests")
    config.addinivalue_line("markers", "smoke: optional end-to-end smoke tests (external data)")
