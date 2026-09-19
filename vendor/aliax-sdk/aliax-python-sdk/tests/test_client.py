"""Smoke tests for the Aliax client — packaging, env fallback, async ctx."""
import pytest

import aliax
from aliax import Aliax, AliaxError, AliaxInvalidKeyError, AliaxOutOfCreditsError


def test_version_exported():
    assert isinstance(aliax.__version__, str)
    assert aliax.__version__.count(".") >= 2


def test_public_surface():
    # Everything in __all__ must actually resolve.
    for name in aliax.__all__:
        assert hasattr(aliax, name), f"missing public symbol: {name}"


def test_exception_hierarchy():
    assert issubclass(AliaxInvalidKeyError, AliaxError)
    assert issubclass(AliaxOutOfCreditsError, AliaxError)


def test_missing_key_raises_when_anonymous_opted_out():
    with pytest.raises(ValueError, match="API key"):
        Aliax(api_key=None, allow_anonymous=False)


def test_missing_key_falls_back_to_anonymous_sandbox(tmp_path, monkeypatch):
    """Zero-signup path: no key -> machine-scoped free sandbox credential."""
    monkeypatch.setenv("ALIAX_CREDENTIALS_PATH", str(tmp_path / "credentials"))
    monkeypatch.delenv("ALIAX_API_KEY", raising=False)
    client = Aliax(api_key=None)
    assert client.anonymous is True
    assert client.api_key.startswith("sk_anon_")


def test_bad_key_prefix_raises():
    with pytest.raises(ValueError, match="sk_"):
        Aliax(api_key="not-a-real-key")


def test_env_var_fallback(monkeypatch):
    monkeypatch.setenv("ALIAX_API_KEY", "sk_test_env_fallback_xyz")
    client = Aliax(check_for_updates=False)
    assert client.api_key == "sk_test_env_fallback_xyz"


def test_dom_mapper_bundled():
    """The mapper is encrypted at rest and opened only after licensing."""
    client = Aliax(api_key="sk_test_bundle", check_for_updates=False)
    assert client.core_js is None
    from importlib.resources import files
    container = (files("aliax") / "dom-mapper.dat").read_bytes()
    assert container[:8] == b"ALIAXM1\0"
    assert len(container) > 60


def test_sdk_version_attached():
    client = Aliax(api_key="sk_test_ver", check_for_updates=False)
    assert client.sdk_version == aliax.__version__
