"""Async context manager + HTTP cleanup tests."""
import pytest

from aliax import Aliax


@pytest.mark.asyncio
async def test_async_context_manager_closes_http():
    async with Aliax(api_key="sk_test_ctx", check_for_updates=False) as client:
        # Force lazy http construction.
        http = await client._client()
        assert http is not None
        assert not http.is_closed
    # After __aexit__, the pool should be torn down.
    assert client._http is None


@pytest.mark.asyncio
async def test_aclose_is_idempotent():
    client = Aliax(api_key="sk_test_idem", check_for_updates=False)
    await client._client()
    await client.aclose()
    await client.aclose()  # second call must not raise
    assert client._http is None
