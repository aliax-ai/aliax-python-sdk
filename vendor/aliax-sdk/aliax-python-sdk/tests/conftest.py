"""Pytest config — shared fixtures for the Aliax SDK test suite."""
import os
import pytest


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    """Don't let a developer's real ALIAX_API_KEY leak into tests."""
    monkeypatch.delenv("ALIAX_API_KEY", raising=False)
    yield
