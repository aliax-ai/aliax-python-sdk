"""Zero-signup anonymous sandbox credentials.

Why this exists
---------------
The single biggest drop-off in a developer SDK is "go create an account
and paste an API key" before the very first line of code runs. Aliax
removes that step: ``Aliax()`` with no key bootstraps a machine-scoped
sandbox credential from the edge and caches it locally, so the quickstart
is literally::

    pip install aliax
    python my_agent.py

What a sandbox credential is
----------------------------
* An opaque ``sk_anon_...`` token bound to this machine + network.
* Metered exactly like a paid key — 1 credit per ``parse_ui``; ``execute``
  and ``capture_failure`` stay free — but the balance is a fixed grant
  that can never be topped up:
    - 500 parses from a residential / corporate ISP
    - 100 parses from datacenter / cloud egress (cheap to recycle, so a
      smaller bucket)
* Valid for 30 days, then it silently re-bootstraps.
* No dashboard, no crash-capture uploads, no invoices. Signing up
  (free, 1,000 credits) unlocks those.

Storage
-------
``~/.aliax/credentials`` (``%USERPROFILE%\\.aliax\\credentials`` on
Windows), JSON, chmod 0600 where the OS supports it. Deleting the file
is safe: the server recognises the same machine fingerprint and returns
the SAME remaining allowance rather than a fresh grant.

Everything here is best-effort. A read-only filesystem, an exotic
platform, or an offline machine degrades to "no cached credential" — it
never raises into the customer's agent loop.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import stat
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

log = logging.getLogger("aliax")

CREDENTIALS_ENV = "ALIAX_CREDENTIALS_PATH"
ANON_PREFIX = "sk_anon_"
SIGNUP_URL = "https://aliax.xyz/auth"


def credentials_path() -> Path:
    """Location of the cached sandbox credential."""
    override = os.getenv(CREDENTIALS_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".aliax" / "credentials"


def machine_fingerprint() -> str:
    """Stable, non-identifying machine id.

    Hashed locally before it ever leaves the process, and hashed AGAIN
    server-side with a server secret. We never transmit the hostname or
    MAC address themselves.
    """
    parts = [
        platform.node() or "",
        platform.system() or "",
        platform.machine() or "",
        str(uuid.getnode()),          # MAC-derived (falls back to random)
        str(Path.home()),
    ]
    return hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()


def load_cached(endpoint_base: str) -> Optional[Dict[str, Any]]:
    """Return the cached credential for this endpoint, or None."""
    path = credentials_path()
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text("utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    key = data.get("api_key")
    if not isinstance(key, str) or not key.startswith(ANON_PREFIX):
        return None
    # A credential minted against a different backend (self-hosted,
    # staging) must not be replayed at the public edge.
    if data.get("endpoint") and data.get("endpoint") != endpoint_base:
        return None
    return data


def save_cached(payload: Dict[str, Any]) -> None:
    """Persist the credential, 0600 where the OS supports it."""
    path = credentials_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        try:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        except Exception:
            pass  # Windows / exotic FS — content is not a long-lived secret.
    except Exception as err:  # read-only FS, sandboxed CI, etc.
        log.debug("Aliax: could not cache anonymous credential (%s)", err)


def clear_cached() -> None:
    try:
        credentials_path().unlink(missing_ok=True)
    except Exception:
        pass


class AnonymousQuotaExhausted(RuntimeError):
    """The machine/network has used up its free sandbox allowance."""


def bootstrap(
    endpoint_base: str,
    sdk_version: str,
    timeout: float = 8.0,
) -> Dict[str, Any]:
    """Request a fresh sandbox credential from ``POST /v1/auth/anonymous``.

    Raises ``AnonymousQuotaExhausted`` when the network is out of free
    allowance, and ``httpx`` transport errors when the edge is
    unreachable (the caller decides whether to fail over).
    """
    res = httpx.post(
        f"{endpoint_base}/auth/anonymous",
        json={"machine_id": machine_fingerprint(), "sdk_version": sdk_version},
        timeout=timeout,
        headers={"X-Aliax-SDK-Version": sdk_version},
    )
    try:
        data = res.json() if res.content else {}
    except Exception:
        data = {}
    if not isinstance(data, dict):
        data = {}

    if res.status_code == 429 or data.get("code") == "sandbox_exhausted":
        raise AnonymousQuotaExhausted(
            data.get("msg")
            or (
                "This network has used its free Aliax sandbox allowance. "
                f"Create a free account at {SIGNUP_URL} (1,000 credits) and "
                "set ALIAX_API_KEY."
            )
        )
    if res.status_code >= 400 or not isinstance(data.get("api_key"), str):
        raise RuntimeError(
            f"Aliax anonymous bootstrap failed (HTTP {res.status_code})."
        )

    payload = {
        "api_key": data["api_key"],
        "endpoint": endpoint_base,
        "mode": "anonymous",
        "tier": data.get("tier"),
        "granted": data.get("granted"),
        "remaining": data.get("remaining"),
        "expires_at": data.get("expires_at"),
    }
    save_cached(payload)
    return payload
