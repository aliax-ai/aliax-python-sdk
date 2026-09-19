"""Aliax SDK — the Real-Time Interceptor Middleware for Visual Web Agents.

Public surface (v1.0):

    aliax = Aliax(api_key="sk_live_...")

    # 1. Translate the live page → Set-of-Mark image + node map.
    ctx = await aliax.parse_ui(page)
    #   ctx.image_bytes  — JPEG screenshot with numbered Set-of-Mark
    #                       boxes drawn over every interactable element
    #                       (browser-native overlay, zero-reflow).
    #                       every interactable element (zero-reflow render).
    #   ctx.elements     — JSON map [{element_id, tag, bounds, text, ...}]
    #   ctx.viewport     — {width, height, dpr, scroll_x, scroll_y}

    # 2. Hand BOTH to the VLM. The LLM responds with structured action.
    #    e.g. {"action": "CLICK", "element_id": "el_41"}
    decision = await ask_llm(image=ctx.image_bytes, map=ctx.elements)

    # 3. Aliax does the messy execution — scroll-into-view, React onChange,
    #    iframe coord math, retina DPR, the works.
    await aliax.execute(page, decision)

    # 4. Safety net — when the AI loops on a popup, dump the whole state
    #    into the annotation queue.
    await aliax.capture_failure(
        page,
        goal="checkout cart",
        thoughts=agent.reasoning,
        last_attempted_action={"action": "CLICK", "element_id": "el_41"},
        failure_reason="modal_blocked",
    )

Architecture notes
------------------
- The DOM mapper (dom-mapper.min.js) ships bundled inside the wheel so
  customers can pin a deterministic version. We DO NOT CDN-fetch JS at
  runtime — that's an instant RCE flag for enterprise security review.
- Set-of-Mark boxes are rendered by Chromium itself: the bundled JS
  injects a position:fixed + pointer-events:none overlay container at
  the very top z-stack, Playwright snapshots the page natively
  (`type="jpeg", quality=…`), then the overlay is removed in a
  try/finally. This is the "zero-bloat" pivot — the SDK ships with no
  Pillow / OpenCV / Cairo deps, only `httpx + playwright`. Browsers are
  C++ rendering engines built to paint rectangles in microseconds; we
  let them do it instead of dragging the raw bytes into Python RAM.
- Image format / compression is controlled natively through Playwright's
  C++ screenshot path via the new ``render_config`` knob — power users
  who want VLM token costs ~85% lower can set
  ``render_config={"format": "jpeg", "quality": 40}`` and the only
  thing Python does to the bytes is hand them to httpx.
- Telemetry is fire-and-forget on the fast path (balance positive or
  unknown). When the balance enters overdraft (≤ 0), the next parse_ui
  ping is awaited so the caller gets an authoritative kill-switch signal
  and the 3-tick grace loop can decide whether to resume or surface
  AliaxOutOfCreditsError. execute / capture_failure are always async.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid as _uuid_module
import weakref
from contextlib import suppress
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Union
from urllib.parse import urlparse

import httpx

try:  # pragma: no cover - optional, ships with pip
    from packaging.version import Version as _PkgVersion  # type: ignore
except Exception:  # pragma: no cover
    _PkgVersion = None  # type: ignore[assignment]

from . import _anon
from ._crypto import open_container
from ._version import __version__
from .prompts import ALIAX_SYSTEM_INSTRUCTIONS

log = logging.getLogger("aliax")


def _load_bundled_text(filename: str) -> str:
    """Read a packaged data file as text.

    Uses ``importlib.resources`` so the SDK works when installed inside a
    zipped wheel / AWS Lambda layer / Google Cloud Run container where
    the on-disk ``__file__`` path does not physically exist. Falls back
    to the legacy ``read_text`` API on Python 3.8.
    """
    try:
        # Python 3.9+ — the modern, non-deprecated path.
        from importlib.resources import files  # type: ignore[attr-defined]

        return (files(__package__) / filename).read_text(encoding="utf-8")
    except (ImportError, AttributeError):
        # Python 3.8 fallback.
        from importlib.resources import read_text  # type: ignore[attr-defined]

        return read_text(__package__, filename, encoding="utf-8")


def _load_bundled_bytes(filename: str) -> bytes:
    from importlib.resources import files
    return (files(__package__) / filename).read_bytes()


def _format_dbg(stage: str, fields: Mapping[str, Any]) -> str:
    """Render a structured debug line. Truncates long string values so a
    10KB textarea paste never floods the console."""
    safe: dict = {}
    for k, v in fields.items():
        if isinstance(v, str) and len(v) > 120:
            safe[k] = v[:120] + f"…(+{len(v) - 120})"
        else:
            safe[k] = v
    return "[Aliax] " + stage + " " + " ".join(f"{k}={safe[k]!r}" for k in safe)



def _decision_text_value(decision: Mapping[str, Any]) -> str:
    """Extract text for TYPE-like actions from common agent schemas."""
    for key in ("value", "text_input", "input", "text"):
        if key in decision and decision.get(key) is not None:
            return str(decision.get(key))
    return ""


def _decision_key_value(decision: Mapping[str, Any], default: str = "Enter") -> str:
    """Extract key for PRESS-like actions from common agent schemas."""
    for key in ("key", "text_input", "value", "input"):
        if key in decision and decision.get(key) is not None:
            return str(decision.get(key))
    return default


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AliaxError(Exception):
    """Base class for all SDK-level errors."""


class AliaxOutOfCreditsError(AliaxError):
    """Raised by ``parse_ui()`` when the account is past the overdraft floor.

    The SDK is self-healing: a single HTTP 402 does NOT permanently
    brick the client. Instead the SDK first runs a 3-tick grace loop
    (poll the server every ~3s for ~9s total) to give a concurrent
    auto-top-up time to land. Only if all 3 grace pings still come back
    402 is this exception raised — and even then a subsequent
    ``parse_ui()`` call will re-enter the grace loop and recover
    automatically once a top-up clears.

    There is no manual ``.unlock()`` step required — top up and the next
    ``parse_ui()`` call self-heals. If you want to force an immediate
    server-side balance refresh without burning a real parse (e.g. on a
    long-lived daemon that's been idle since the top-up), call
    ``await aliax.refresh_billing_status()``; it issues a free ``execute``
    ping that clears the in-memory exhaustion flag the moment the Worker
    confirms credits are back above the overdraft floor.
    """

    def __init__(self, balance: Optional[int] = None):
        self.balance = balance
        bal_str = str(balance) if balance is not None else "unknown"
        super().__init__(
            f"Aliax account is past its overdraft floor (balance={bal_str}). "
            "Top up at https://aliax.xyz/dashboard — the SDK will resume "
            "automatically on the next call once the balance clears."
        )


class AliaxInvalidKeyError(AliaxError):
    """Raised by ``parse_ui()`` when the Worker rejected the API key
    (revoked, expired, or never existed)."""


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------



@dataclass
class AttemptedAction:
    """The "stupid move" payload — what the agent tried that didn't work.

    Used by ``capture_failure()`` so the annotation dashboard can pin a
    red-X at the exact pixel the agent clicked vs the right target.
    """

    action: str  # CLICK | TYPE | SCROLL | HOVER | NAVIGATE | EXTRACT | ...
    target_x: Optional[int] = None
    target_y: Optional[int] = None
    value: Optional[str] = None
    selector: Optional[str] = None
    element_id: Optional[str] = None  # if the agent referred to an Aliax id

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


AttemptedActionLike = Union[AttemptedAction, Mapping[str, Any]]


@dataclass
class ParseContext:
    """What :meth:`Aliax.parse_ui` returns. Pass to your LLM verbatim.

    Attributes:
        image_bytes: Native screenshot (JPEG by default; PNG opt-in via
            ``render_config``) with numbered Set-of-Mark boxes painted
            by the bundled JS overlay layer.
        image_mime: MIME type for ``image_bytes``.
        image_size: ``(width, height)`` of ``image_bytes`` in pixels.
        elements: lightweight JSON list — one entry per interactable
            element. Each has ``element_id`` (e.g. ``"el_41"``), ``tag``,
            ``role``, ``text``, ``bounds``, ``editable``, ``is_canvas``,
            ``attrs``, and optionally ``links_to`` (the route the
            control navigates to — clairvoyance for the VLM).
        viewport: ``{width, height, dpr, scroll_x, scroll_y}`` in CSS px.
        url: full page URL at the moment of capture (e.g.
            ``"https://app.viketa.xyz/settings/billing?tab=plans"``).
        path: URL pathname only (e.g. ``"/settings/billing"``). The
            cleanest, most token-efficient slice for "where am I?"
            prompts — handed to the VLM as Active Route.
        title: document.title at capture time. Doubles as a secondary
            anchor when two routes share an identical pathname (e.g.
            multi-tenant ``/app`` sub-rendering).
        truncated: True when the DOM mapper hit its element cap; the
            ``elements`` list is incomplete and the VLM should be told.
    """

    image_bytes: bytes
    image_mime: str
    image_size: tuple
    elements: List[dict] = field(default_factory=list)
    viewport: dict = field(default_factory=dict)
    url: str = ""
    path: str = ""
    title: str = ""
    truncated: bool = False

    @property
    def element_ids(self) -> List[str]:
        return [str(e.get("element_id")) for e in self.elements if e.get("element_id")]

    def route_context_block(self) -> str:
        """Compact "where am I standing?" header for the VLM prompt.

        Stitched in front of the element listing by orchestrators so the
        model can self-verify navigation ("I clicked Submit but the
        Active Route didn't change — the form probably has validation
        errors") and short-circuit redundant clicks ("goal is /billing
        and I'm already there — skip the menu hop").
        """
        lines = ["=== CURRENT BROWSER STATE ==="]
        if self.title:
            lines.append(f'- Page Title: "{self.title}"')
        if self.path:
            lines.append(f"- Active Route: {self.path}")
        if self.url:
            lines.append(f"- Full URL: {self.url}")
        # Page-level scroll capability — the mapper emits these flags
        # so the VLM can pick a global SCROLL_DOWN vs targeting a
        # specific scrollable container by element_id (Discord/Notion
        # case where <body> doesn't scroll). Without this hint the
        # model defaults to per-element scrolls even when a simple
        # page-scroll would do the job in one action.
        vp = self.viewport if isinstance(self.viewport, dict) else {}
        psy = bool(vp.get("page_scrollable_y"))
        psx = bool(vp.get("page_scrollable_x"))
        if psy and psx:
            lines.append("- Page Scroll: vertical + horizontal")
        elif psy:
            lines.append("- Page Scroll: vertical")
        elif psx:
            lines.append("- Page Scroll: horizontal")
        else:
            lines.append("- Page Scroll: none (target a scrollable container by element_id)")
        lines.append("=============================")
        return "\n".join(lines)

    def llm_text_block(self) -> str:
        """A compact text rendering of the node map — drop this straight
        into your LLM system prompt alongside ``image_bytes``.

        State annotations (``[DISABLED]``, ``[CHECKED]``, ``[EXPANDED]``,
        ``[REQUIRED]``, ``[INVALID]``, ``[BUSY]``, ``[READONLY]``,
        ``[SELECTED]``, ``[PRESSED]``) are intentionally surfaced so the
        VLM can reason about WHY a target is locked instead of panicking
        and claiming the element is missing.

        Navigation hints (``-> /route``) are appended for elements that
        declare a destination via ``<a href>``, ``data-href``,
        ``data-to``, ``data-route``, or a nested SPA link — letting the
        VLM plan a trajectory ("Settings -> /settings, so click it to
        reach /settings/billing") instead of guessing.
        """
        lines = []
        for e in self.elements:
            eid = e.get("element_id")
            tag = e.get("tag")
            text = (e.get("text") or "").strip()
            extra = []
            if e.get("editable"):
                extra.append("editable")
            if e.get("is_canvas"):
                extra.append("CANVAS")
            st = e.get("state") or {}
            # Order chosen so the most action-relevant flag (DISABLED) lands first.
            if st.get("disabled"):
                extra.append("DISABLED")
            if st.get("busy"):
                extra.append("BUSY")
            if st.get("invalid"):
                extra.append("INVALID")
            if st.get("required"):
                extra.append("REQUIRED")
            if st.get("readonly"):
                extra.append("READONLY")
            if "checked" in st:
                c = st["checked"]
                extra.append("CHECKED" if c is True else ("MIXED" if c == "mixed" else "UNCHECKED"))
            if st.get("selected"):
                extra.append("SELECTED")
            if "pressed" in st:
                extra.append("PRESSED" if st["pressed"] else "UNPRESSED")
            if "expanded" in st:
                extra.append("EXPANDED" if st["expanded"] else "COLLAPSED")
            # Scrollable hints — surfaced so the VLM picks targeted scrolls
            # over global ones on Discord/Notion-style fixed shells.
            if st.get("scrollable_y") and st.get("scrollable_x"):
                extra.append("SCROLLABLE")
            elif st.get("scrollable_y"):
                extra.append("SCROLLABLE_Y")
            elif st.get("scrollable_x"):
                extra.append("SCROLLABLE_X")
            tail = (" [" + ",".join(extra) + "]") if extra else ""
            base = f"{eid} {tag}{tail}: {text}" if text else f"{eid} {tag}{tail}"
            link = e.get("links_to")
            label = f"{base} -> {link}" if link else base
            lines.append(label)
        if self.truncated:
            lines.append(
                "[truncated] DOM mapper hit its element cap — additional "
                "interactable elements exist but are not listed."
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _coerce_action(a: Optional[AttemptedActionLike]) -> Optional[dict]:
    if a is None:
        return None
    if isinstance(a, AttemptedAction) or is_dataclass(a):
        return a.to_dict() if hasattr(a, "to_dict") else asdict(a)
    if isinstance(a, Mapping):
        return {k: v for k, v in a.items() if v is not None}
    raise TypeError(
        "last_attempted_action must be an AttemptedAction or a dict, got "
        f"{type(a).__name__}"
    )


def _is_older(current: str, latest: str) -> bool:
    """Robust semver compare. Uses `packaging` when available so
    pre-release tags (rc, beta) and unusual schemes don't trigger
    spurious upgrade nags."""
    if _PkgVersion is not None:
        try:
            return _PkgVersion(current) < _PkgVersion(latest)
        except Exception:
            return False

    def parts(v: str):
        out = []
        for chunk in v.split("."):
            try:
                out.append(int(chunk))
            except ValueError:
                num = ""
                for ch in chunk:
                    if ch.isdigit():
                        num += ch
                    else:
                        break
                out.append(int(num) if num else 0)
        return out
    return parts(current) < parts(latest)


def _is_context_destroyed(err: BaseException) -> bool:
    msg = str(err)
    return (
        "Execution context was destroyed" in msg
        or "context was destroyed" in msg
        or "Target page, context or browser has been closed" in msg
        or "Frame was detached" in msg
    )


def _is_intercepted(err: BaseException) -> bool:
    """Nightmare 4 — invisible-shield click hijack.

    Playwright's actionability check refuses to dispatch a real mouse
    event when the target's hit-test point is owned by another node
    (a transparent overlay, a sticky cookie banner, a z-index:9999
    "subscribe" trap). These messages all signal the same thing:
    the element IS there, the page just refuses to let us click it
    "as a human" — JS-level node.click() bypasses the obstruction.
    """
    msg = str(err).lower()
    return (
        "intercept" in msg                  # "subtree intercepts pointer events"
        or "obscured" in msg                # "Target is obscured by other element"
        or "outside of the viewport" in msg
        or "not stable" in msg              # ad shift mid-click
        or "element is not visible" in msg  # fades in then out
        or "element is hidden" in msg
    )


def _is_stamp_lost(err: BaseException) -> bool:
    """Nightmare 1 — SPA DOM-Wipe rebuilt the subtree, our stamp is gone."""
    msg = str(err).lower()
    return (
        "not attached" in msg
        or "no element found" in msg
        or "no elements found" in msg
        or "is not attached" in msg
        or "waiting for selector" in msg
        or "locator resolved to 0 elements" in msg
        or "timeout" in msg and "exceeded" in msg  # locator.wait_for(attached) timed out
    )


def _stamp_from_element_id(element_id: Optional[str]) -> Optional[str]:
    """Extract the numeric DNA suffix from an Aliax element_id.

    "el_41" → "41". Returns None if the input doesn't look like one of
    our ids — keeps caller code linear (no exception handling needed).
    """
    if not element_id:
        return None
    s = str(element_id).strip()
    if not s:
        return None
    return s[3:] if s.startswith("el_") else s

# ---------------------------------------------------------------------------
# Render config — the only knob power users get for VLM token economics
# ---------------------------------------------------------------------------


# Playwright's `page.screenshot()` natively understands only PNG and JPEG.
# WebP is NOT supported by the underlying CDP `Page.captureScreenshot`
# command, so we cannot pretend to offer it from a zero-dep SDK. Power
# users who specifically need WebP can pass type="png" and re-encode
# downstream — but the headline use case (cheap VLM tokens) is fully
# covered by JPEG which OpenAI / Anthropic / Google all bill at the
# "low-detail" tier when the image is small + lossy.
_NATIVE_SCREENSHOT_FORMATS = {"jpeg", "png"}

# Floor JPEG quality at 30 — below that, drawn Set-of-Mark IDs start
# turning into mush ("88" → "B8") and the VLM misclicks. Cap at 100.
# These bounds match the safety-rail logic discussed in the "Blurry 88
# Problem" design note; we silently clamp instead of raising so a
# power-user's `quality=999` typo never crashes their agent loop.
_MIN_QUALITY = 30
_MAX_QUALITY = 100


def _normalize_render_config(
    render_config: Optional[Mapping[str, Any]],
    *,
    legacy_format: Optional[str] = None,
    legacy_quality: Optional[int] = None,
) -> dict:
    """Resolve the final ``{format, quality}`` Playwright will use.

    Resolution order (highest precedence first):
      1. ``render_config={"format": ..., "quality": ...}`` — the new
         power-user knob.
      2. ``image_format=`` / ``image_quality=`` legacy kwargs — kept so
         pre-1.1 callers keep working without a code change.
      3. Defaults: ``format="jpeg"``, ``quality=80``.

    Unsupported formats (e.g. "webp") gracefully fall back to JPEG with
    a one-line warning; quality is clamped to ``[30, 100]``.
    """
    rc = dict(render_config or {})
    fmt = rc.get("format")
    q = rc.get("quality")
    if fmt is None and legacy_format is not None:
        fmt = legacy_format
    if q is None and legacy_quality is not None:
        q = legacy_quality

    fmt = str(fmt or "jpeg").strip().lower()
    if fmt in ("jpg", "jpeg"):
        fmt = "jpeg"
    if fmt not in _NATIVE_SCREENSHOT_FORMATS:
        log.warning(
            "Aliax: render_config format=%r is not supported by the native "
            "Playwright screenshot path (only jpeg / png). Falling back to "
            "jpeg — re-encode downstream if you specifically need %s.",
            fmt, fmt,
        )
        fmt = "jpeg"

    try:
        q_int = int(q) if q is not None else 80
    except (TypeError, ValueError):
        q_int = 80
    if q_int < _MIN_QUALITY:
        # log.warning (not debug) so a `quality=5` typo surfaces in
        # production logging — illegible Set-of-Mark labels are the
        # exact "Blurry 88 → misclick" hazard the floor exists to stop.
        log.warning(
            "Aliax: render_config quality=%s clamped up to %d (below that, "
            "drawn Set-of-Mark IDs become unreadable for VLMs).",
            q, _MIN_QUALITY,
        )
        q_int = _MIN_QUALITY
    elif q_int > _MAX_QUALITY:
        q_int = _MAX_QUALITY

    return {"format": fmt, "quality": q_int}


def _jpeg_dimensions(data: bytes) -> Optional[tuple]:
    """Return ``(width, height)`` for a JPEG byte string, or None.

    Tiny hand-rolled parser so the zero-dep SDK still reports an
    accurate ``image_size`` on the returned ParseContext without
    pulling Pillow back in. Walks the JPEG marker stream looking for
    the first SOFn frame header (FF C0–C3, C5–C7, C9–CB, CD–CF).
    """
    if not data or len(data) < 4 or data[0] != 0xFF or data[1] != 0xD8:
        return None
    i = 2
    n = len(data)
    SOF_MARKERS = {
        0xC0, 0xC1, 0xC2, 0xC3,
        0xC5, 0xC6, 0xC7,
        0xC9, 0xCA, 0xCB,
        0xCD, 0xCE, 0xCF,
    }
    try:
        # Loose outer guard — inner branches check their own payload bounds.
        # The previous `i + 9 < n` was off-by-one and missed valid SOFn
        # frames whose payload landed exactly at the stream boundary,
        # which on retina screenshots produced a wrong image_size and
        # nudged the dashboard's coordinate overlay by 1–2 px.
        while i + 1 < n:
            if data[i] != 0xFF:
                # Pad bytes / corrupt stream — bail.
                return None
            # Skip fill bytes (rare but legal — `FF FF ... FF Cx`).
            while i < n and data[i] == 0xFF:
                i += 1
            if i >= n:
                return None
            marker = data[i]
            i += 1
            if marker == 0xD8 or marker == 0xD9 or (0xD0 <= marker <= 0xD7):
                # SOI / EOI / RSTn — no length payload.
                continue
            if i + 1 >= n:
                return None
            seg_len = (data[i] << 8) | data[i + 1]
            if marker in SOF_MARKERS:
                # SOFn payload: [len(2)][precision(1)][height(2)][width(2)]…
                if i + 7 >= n:
                    return None
                height = (data[i + 3] << 8) | data[i + 4]
                width = (data[i + 5] << 8) | data[i + 6]
                return (width, height)
            i += seg_len
    except Exception:
        return None
    return None


def _png_dimensions(data: bytes) -> Optional[tuple]:
    """Return ``(width, height)`` for a PNG byte string, or None.

    PNG's IHDR chunk is always immediately after the 8-byte signature,
    making this a fixed offset read.
    """
    if not data or len(data) < 24:
        return None
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    try:
        width = int.from_bytes(data[16:20], "big")
        height = int.from_bytes(data[20:24], "big")
        return (width, height)
    except Exception:
        return None


def _image_dimensions(data: bytes, fmt: str) -> Optional[tuple]:
    fmt = (fmt or "").lower()
    if fmt == "jpeg":
        return _jpeg_dimensions(data)
    if fmt == "png":
        return _png_dimensions(data)
    return None




# ---------------------------------------------------------------------------
# Aliax client
# ---------------------------------------------------------------------------


class Aliax:
    """Aliax SDK client — the interceptor between your LLM and the page.

    Zero-setup keyless mode:
        Calling ``Aliax()`` with no key requires no signup. It automatically
        provisions and caches a free machine-scoped sandbox credential
        (500 free parses residential / 100 cloud egress) in ``~/.aliax/credentials``.

    Args:
        api_key: Optional API key (``sk_live_...``) from the dashboard or
            ``ALIAX_API_KEY`` env var. When omitted, automatically falls back
            to the free anonymous sandbox credential.
        allow_anonymous: When True (default), bootstrap a free machine-scoped
            sandbox credential if no API key is found. Pass False (or set
            ``ALIAX_DISABLE_ANONYMOUS=1``) to hard-fail in locked-down environments.
        endpoint: base URL for the Aliax API. Defaults to the public
            edge; point at the Cloudflare Worker URL until your DNS
            cutover is done.
        debug_mode: ``True`` writes ``debug_payload.json`` /
            ``debug_screenshot.jpg`` locally instead of POSTing to the
            cloud (handy for CI). Telemetry pings are skipped in debug.
        redact_selectors: extra CSS selectors to blur visually before
            screenshotting. Always merged with a built-in PII baseline
            (password fields, email inputs, card numbers, etc.).
        check_for_updates: when True (default), spawn a tiny daemon
            thread on init that pings ``/v1/version`` and ``logging.warning``s
            if a newer SDK is published. Pass ``False`` in airgapped envs.
        max_image_dim: DEPRECATED — no-op since v1.0 (the SDK ships
            without Pillow). Use ``render_config={"quality": N}`` on
            :meth:`parse_ui` for VLM token economy instead. The kwarg
            is kept so pre-1.1 callers keep importing without churn.
            Default is ``0`` so a default-constructed client does NOT
            emit the deprecation log on every parse_ui — only callers
            that explicitly opt in see the notice, and only on the
            first parse_ui call (latched via ``_max_image_dim_warned``).
        fallback_endpoint: secondary endpoint URL for automatic failover.
    """

    # Pre-packaged VLM system prompt — drop into your LLM call so the
    # action vocabulary and JSON schema match what execute() accepts.
    # Available as both `Aliax.SYSTEM_INSTRUCTIONS` and `aliax.SYSTEM_INSTRUCTIONS`.
    SYSTEM_INSTRUCTIONS = ALIAX_SYSTEM_INSTRUCTIONS


    # Workers.dev origin kept as a hard-coded automatic fallback in case
    # the apex (api.aliax.xyz) is in the middle of a DNS / cert reissue
    # or the customer's network blocks the apex but allows *.workers.dev.
    # The SDK swaps to this transparently on the first ConnectError /
    # DNS failure of the session and stays on it for subsequent calls.
    _DEFAULT_FALLBACK_ENDPOINT = (
        "https://aliax-cloudflare-worker.ogazievictorchi.workers.dev/v1"
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: str = "https://api.aliax.xyz/v1",
        debug_mode: bool = False,
        redact_selectors: Optional[list] = None,
        check_for_updates: bool = True,
        max_image_dim: int = 0,
        fallback_endpoint: Optional[str] = None,
        allow_anonymous: bool = True,
    ):
        self.sdk_version = __version__
        self._set_endpoint_base(endpoint)
        # Only wire a fallback when the caller is on the public default —
        # custom / self-hosted endpoints should NOT silently fail over to
        # the public worker. Caller can force one by passing it explicitly.
        if fallback_endpoint is not None:
            self.fallback_endpoint_base = (fallback_endpoint or "").rstrip("/") or None
        elif self.endpoint_base == "https://api.aliax.xyz/v1":
            self.fallback_endpoint_base = self._DEFAULT_FALLBACK_ENDPOINT
        else:
            self.fallback_endpoint_base = None
        self._fallback_active = False

        # ---- Key resolution ----------------------------------------
        # 1. explicit kwarg
        # 2. ALIAX_API_KEY env var (12-factor: Docker, Lambda, CI)
        # 3. cached anonymous sandbox credential (~/.aliax/credentials)
        # 4. fresh anonymous sandbox credential from the edge
        #
        # Step 3/4 is what makes `pip install aliax` → `Aliax()` work with
        # zero signup. It is opt-out via allow_anonymous=False or
        # ALIAX_DISABLE_ANONYMOUS=1 for locked-down enterprise installs.
        if api_key is None:
            api_key = os.getenv("ALIAX_API_KEY")

        self.anonymous = False
        self.anonymous_info: Optional[dict] = None
        self._anon_rebootstrapped = False

        if not api_key:
            env_optout = os.getenv("ALIAX_DISABLE_ANONYMOUS", "").strip().lower()
            anon_allowed = allow_anonymous and env_optout not in ("1", "true", "yes")
            if anon_allowed and not debug_mode:
                api_key = self._resolve_anonymous_key()
            if not api_key:
                raise ValueError(
                    "Aliax could not obtain an API key. Pass "
                    "Aliax(api_key='sk_...'), set the ALIAX_API_KEY "
                    "environment variable, or allow the free anonymous "
                    "sandbox (needs outbound HTTPS on first run). Create a "
                    "free key at https://aliax.xyz/auth."
                )
        elif not api_key.startswith("sk_"):
            raise ValueError(
                "Aliax(api_key=...) must be a key starting with 'sk_'. "
                "Pass it directly or set the ALIAX_API_KEY environment "
                "variable. Generate one from the dashboard /api-keys page."
            )

        self.api_key = api_key

        self.debug_mode = debug_mode
        self.redact_selectors = redact_selectors or []
        self.max_image_dim = int(max_image_dim) if max_image_dim else 0

        # Load the bundled, obfuscated DOM mapper artifact via
        # importlib.resources so the SDK works when installed inside a
        # zipped wheel / AWS Lambda layer / Google Cloud Run container
        # where the on-disk __file__ path does not physically exist.
        # The mapper is opened only after the authenticated session handshake.
        # Even debug mode requires the production session handshake.
        self.core_js: Optional[str] = None
        self._mapper_session_id: Optional[str] = None
        self._mapper_tickets: list[str] = []
        self._mapper_session_lock = asyncio.Lock()

        # Per-instance node cache so execute() can resolve element_ids
        # against the latest parse_ui without re-walking the DOM. Keyed
        # by the Page object via WeakKeyDictionary so closed pages are
        # evicted automatically (id() reuse would otherwise return a
        # stale map for a brand-new page).
        self._last_map_by_page: "weakref.WeakKeyDictionary[Any, list]" = (
            weakref.WeakKeyDictionary()
        )

        # Per-Page asyncio.Lock so two concurrent parse_ui() coroutines
        # on the SAME page can't sabotage each other's overlay. Without
        # this, coroutine B's drawOverlay() (which clears the existing
        # overlay on entry) tears down A's overlay mid-screenshot, and
        # both ParseContexts come back boxless silently. Locks are
        # WeakKeyDictionary-keyed so closed pages are evicted. asyncio
        # primitives are created lazily under a threading.Lock — see
        # _ensure_async_lock for the rationale on cross-loop usage.
        self._parse_lock_by_page: "weakref.WeakKeyDictionary[Any, asyncio.Lock]" = (
            weakref.WeakKeyDictionary()
        )
        self._parse_locks_init = threading.Lock()

        # Strong-reference set for fire-and-forget telemetry tasks.
        # Python's event loop only weak-refs scheduled tasks, so a GC
        # cycle between `create_task` and the first `await` inside
        # `_emit_telemetry` can silently cancel the task — meaning the
        # parse happened but the server never saw the debit, AND the
        # local `_billing_balance` cache stays stale forever. We pin
        # each task in this set and drop it via add_done_callback.
        self._bg_tasks: "set[asyncio.Task]" = set()

        # One-shot latch for the max_image_dim deprecation notice.
        self._max_image_dim_warned: bool = False

        # One shared async HTTP client for telemetry / capture posts —
        # avoids repeated TLS handshakes when the agent loops fast.
        self._http: Optional[httpx.AsyncClient] = None
        # asyncio.Lock cannot be safely lazy-initialised from a sync
        # context — two coroutines on different event loops could each
        # see `None` and create their own Lock, orphaning a TLS pool.
        # A threading.Lock guards the bootstrap so the asyncio.Lock is
        # created exactly once even under cross-thread races.
        self._http_lock_init = threading.Lock()
        self._http_lock: Optional[asyncio.Lock] = None

        # ---- Billing state (debt-ledger / self-healing model) ----
        #
        # The server allows a bounded overdraft (currently -500 credits)
        # so legitimate concurrency bursts never leak free parses past
        # zero. The SDK's job is just to (a) track the authoritative
        # balance the Worker echoes back on every telemetry response and
        # (b) throttle / grace-recover when the Worker says we are past
        # the overdraft floor.
        #
        # State:
        #   _billing_balance      last known balance, or None pre-first-ping.
        #                         Negative is normal under overdraft.
        #   _credits_exhausted    set True iff the most recent parse_ui
        #                         telemetry came back HTTP 402. Cleared
        #                         automatically by any 200 response —
        #                         this is the self-healing primitive.
        #   _invalid_key_reason   latched on HTTP 401 (revoked/expired/
        #                         unknown). Cannot self-heal because the
        #                         operator must mint a new key.
        self._billing_balance: Optional[int] = None
        self._credits_exhausted: bool = False
        self._invalid_key_reason: Optional[str] = None
        # Async lock that serialises the grace-recovery loop so two
        # concurrent parse_ui() coroutines do not stack 6 sync pings
        # on top of each other when an account first hits the floor.
        # Lazy-init under a threading.Lock so loops that construct the
        # client at import time (before any event loop exists) don't
        # crash. Same double-checked pattern as _http_lock.
        self._grace_lock_init = threading.Lock()
        self._grace_lock: Optional[asyncio.Lock] = None

        # ---- Failure-Detection Net (the Three-Pronged Airbag) ----
        # Per-page rolling histories used by report_issue()'s Gatekeeper
        # to verify the agent has actually proven it is stuck before any
        # /v1/capture payload leaves the box.
        #
        #   _state_history_by_page:  SHA256(url + spatial_map) per turn.
        #     3 identical hashes in a row => mathematically stagnant DOM
        #     (frozen screen, unkillable popup, dead zone).
        #
        #   _action_history_by_page: element_id sequence per execute().
        #     [A, B, A, B] over 4 turns => alternation loop (radios /
        #     checkboxes) the state-hash check is blind to because the
        #     DOM does technically mutate on each click.
        #
        # Both are bounded ring buffers (cap _MAX_HISTORY) so a long-
        # lived daemon never accumulates unbounded memory. Weak-keyed
        # so closed pages evict automatically.
        self._state_history_by_page: "weakref.WeakKeyDictionary[Any, list]" = (
            weakref.WeakKeyDictionary()
        )
        self._action_history_by_page: "weakref.WeakKeyDictionary[Any, list]" = (
            weakref.WeakKeyDictionary()
        )

        if check_for_updates and not debug_mode:
            threading.Thread(target=self._check_for_updates, daemon=True).start()

    # ------------------------------------------------------------------
    # Failure-Detection Net — Stagnation + Cycle helpers
    # ------------------------------------------------------------------

    _MIN_STAGNATION = 3   # 3 identical state hashes = "page is frozen".
    _MAX_HISTORY = 8

    def _record_state(self, page, url: str, elements: list) -> None:
        """Hash the (url, spatial_map) signature into the per-page ring.

        Serialises only identity-bearing fields (element_id + tag +
        text + bounds) so unrelated mapper metadata never fakes
        "progress" and starves the Gatekeeper of stagnation evidence.
        """
        import hashlib
        try:
            sig: list = []
            for e in elements or []:
                if not isinstance(e, Mapping):
                    continue
                b = e.get("bounds") or {}
                sig.append([
                    e.get("element_id"),
                    e.get("tag"),
                    (e.get("text") or "")[:64],
                    [b.get("x"), b.get("y"), b.get("width"), b.get("height")] if isinstance(b, Mapping) else None,
                ])
            blob = json.dumps([url or "", sig], separators=(",", ":"))
            digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        except Exception:
            return
        try:
            hist = self._state_history_by_page.get(page)
        except TypeError:
            return
        if hist is None:
            hist = []
            try:
                self._state_history_by_page[page] = hist
            except TypeError:
                return
        hist.append(digest)
        if len(hist) > self._MAX_HISTORY:
            del hist[: len(hist) - self._MAX_HISTORY]

    def _record_action(self, page, element_id: Optional[str]) -> None:
        """Append an executed element_id to the per-page action ring."""
        try:
            hist = self._action_history_by_page.get(page)
        except TypeError:
            return
        if hist is None:
            hist = []
            try:
                self._action_history_by_page[page] = hist
            except TypeError:
                return
        hist.append(element_id or None)
        if len(hist) > self._MAX_HISTORY:
            del hist[: len(hist) - self._MAX_HISTORY]

    def _failure_proof(self, page) -> tuple:
        """Return ``(eligible: bool, reason: str|None)``.

        Either: (1) 3 identical state hashes (frozen DOM) OR
        (2) a repeating action sub-cycle of period 2, 3, or 4 with
            distinct members in each period (catches radio-group cycles
            of 2, 3, or 4 mutually-exclusive options where the DOM is
            technically mutating each click but the agent is making no
            real progress).
        """
        try:
            sh = list(self._state_history_by_page.get(page) or [])
        except TypeError:
            sh = []
        if len(sh) >= self._MIN_STAGNATION:
            tail = sh[-self._MIN_STAGNATION :]
            if len(set(tail)) == 1:
                return True, "state_stagnation"

        try:
            ah = list(self._action_history_by_page.get(page) or [])
        except TypeError:
            ah = []
        # Generalised period-N cycle detection. The original check only
        # caught [A,B,A,B] (period 2) — but a 3-option radio group
        # (think country picker, time slot picker) can produce a
        # genuine [A,B,C,A,B,C] cycle that the period-2 check is
        # mathematically blind to. We scan periods 2..4 and require
        # each period to contain DISTINCT non-None ids (so legitimate
        # retries of the same action like a "Load More" loop —
        # [12,12,12,12] — are not flagged: period=1 is intentionally
        # excluded and [12,12] does not have len(set)==period for
        # period>=2). _MAX_HISTORY=8 is sufficient to hold a full
        # period-4 cycle twice.
        for period in (2, 3, 4):
            window = period * 2
            if len(ah) < window:
                continue
            chunk = ah[-window:]
            head = chunk[:period]
            tail = chunk[period:]
            if (all(x is not None for x in chunk)
                    and head == tail
                    and len(set(head)) == period):
                return True, "action_cycle"

        return False, None

    def _reset_failure_history(self, page) -> None:
        """Clear the rings after a successful escalation upload."""
        for ring in (self._state_history_by_page, self._action_history_by_page):
            try:
                if page in ring:
                    ring[page] = []
            except TypeError:
                pass


    # ------------------------------------------------------------------
    # Anonymous sandbox credentials (zero-signup onboarding)
    # ------------------------------------------------------------------

    def _resolve_anonymous_key(self, force_fresh: bool = False) -> Optional[str]:
        """Return a sandbox key from cache, or mint a fresh one.

        Best-effort by design: any failure returns None so the caller can
        raise a single, actionable error instead of a transport traceback.
        Tries the primary endpoint, then the workers.dev fallback, so a
        mid-DNS-flip apex never blocks a first-run developer.
        """
        if not force_fresh:
            cached = _anon.load_cached(self.endpoint_base)
            if cached:
                self.anonymous = True
                self.anonymous_info = cached
                return cached["api_key"]

        bases = [self.endpoint_base]
        if self.fallback_endpoint_base:
            bases.append(self.fallback_endpoint_base)

        for base in bases:
            try:
                info = _anon.bootstrap(base, self.sdk_version)
            except _anon.AnonymousQuotaExhausted as err:
                raise AliaxOutOfCreditsError(str(err)) from None
            except Exception as err:
                log.debug("Aliax: anonymous bootstrap via %s failed (%s)", base, err)
                continue
            self.anonymous = True
            self.anonymous_info = info
            granted = info.get("granted")
            remaining = info.get("remaining")
            log.warning(
                "Aliax: running on a free anonymous sandbox key (%s tier, "
                "%s of %s parses left, expires %s). Captures and the "
                "dashboard need a free account: https://aliax.xyz/auth",
                info.get("tier") or "unknown",
                remaining,
                granted,
                info.get("expires_at"),
            )
            return info["api_key"]
        return None

    def _rebootstrap_anonymous(self) -> bool:
        """Replace an expired/revoked sandbox key. Once per process."""
        if not self.anonymous or self._anon_rebootstrapped:
            return False
        self._anon_rebootstrapped = True
        _anon.clear_cached()
        try:
            key = self._resolve_anonymous_key(force_fresh=True)
        except AliaxOutOfCreditsError:
            return False
        except Exception:
            return False
        if not key:
            return False
        self.api_key = key
        self._invalid_key_reason = None
        self._http = None  # force a new client with the fresh auth header
        return True

    # ------------------------------------------------------------------
    # Internal HTTP plumbing
    # ------------------------------------------------------------------



    def _set_endpoint_base(self, base: str) -> None:
        """Set the endpoint base URL and recompute per-route URLs."""
        self.endpoint_base = base.rstrip("/")
        self.endpoint_capture = f"{self.endpoint_base}/capture"
        self.endpoint_telemetry = f"{self.endpoint_base}/telemetry"
        self.endpoint_session = f"{self.endpoint_base}/auth/session"
        self.endpoint_version = f"{self.endpoint_base}/version"

    def _maybe_swap_to_fallback(self) -> bool:
        """Swap to the workers.dev fallback after a primary DNS/connect failure.

        Returns True if a swap happened (caller should retry the request),
        False if there's no fallback configured or one was already used.
        """
        if self._fallback_active or not self.fallback_endpoint_base:
            return False
        log.warning(
            "Aliax primary endpoint unreachable; failing over to fallback %s",
            self.fallback_endpoint_base,
        )
        self._set_endpoint_base(self.fallback_endpoint_base)
        self._fallback_active = True
        return True

    def _auth_headers(self) -> dict:
        # API key travels in an Authorization header so it never lands
        # in request-body logs at proxies / CDNs / Worker logs.
        return {
            "Authorization": f"Bearer {self.api_key}",
            "X-Aliax-SDK-Version": self.sdk_version,
        }

    def _ensure_async_lock(self) -> asyncio.Lock:
        # Idempotent, thread-safe lazy init of the asyncio.Lock.
        if self._http_lock is None:
            with self._http_lock_init:
                if self._http_lock is None:
                    self._http_lock = asyncio.Lock()
        return self._http_lock

    async def _client(self) -> httpx.AsyncClient:
        lock = self._ensure_async_lock()
        async with lock:
            if self._http is None or self._http.is_closed:
                self._http = httpx.AsyncClient(
                    timeout=10.0,
                    headers=self._auth_headers(),
                )
            return self._http

    async def aclose(self) -> None:
        """Close the underlying HTTP client. Idempotent and race-safe."""
        lock = self._ensure_async_lock()
        async with lock:
            if self._http is not None and not self._http.is_closed:
                with suppress(Exception):
                    await self._http.aclose()
            self._http = None

    async def __aenter__(self) -> "Aliax":
        """Enter an ``async with`` block.

        Lets callers write ``async with Aliax() as aliax:`` and have the
        HTTP connection pool torn down deterministically on exit, even if
        the agent loop raises. Eliminates the socket-leak class of bugs
        you get when a long-running bot crashes without calling
        ``aclose()``.
        """
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.aclose()


    def _check_for_updates(self) -> None:
        try:
            res = httpx.get(self.endpoint_version, timeout=1.0)
            if res.status_code != 200:
                return
            latest = res.json().get("latest_sdk")
            if latest and _is_older(self.sdk_version, latest):
                log.warning(
                    "Aliax SDK Notice: you are running %s. Version %s is "
                    "available with improved iframe + shadow-DOM mapping. "
                    "Run 'pip install --upgrade aliax' to update.",
                    self.sdk_version,
                    latest,
                )
        except (httpx.ConnectError, httpx.ConnectTimeout):
            # The version ping is often the first network call the SDK
            # makes. If the apex is unreachable here, trip the failover
            # now so functional calls don't waste another DNS timeout.
            try:
                self._maybe_swap_to_fallback()
            except Exception:
                pass
        except Exception as exc:
            # Never crash the agent over a version ping.
            log.debug("aliax version check skipped: %s", exc)

    async def _emit_telemetry(
        self,
        event_type: str,
        units: int = 1,
        meta: Optional[dict] = None,
    ) -> None:
        """Send a usage ping. Never raises.

        Side-effects (via ``_absorb_billing_response``):
          - HTTP 200: cache the authoritative server balance and clear
            ``_credits_exhausted`` (self-heal — a successful 200 means
            the Worker accepted the debit, so by definition we're above
            the overdraft floor again).
          - HTTP 402: cache the balance and set ``_credits_exhausted``
            so the NEXT ``parse_ui()`` runs its 3-tick grace recovery
            loop before doing any client-side work. Only flips for
            ``parse_ui`` events — a 402 on a free event indicates a
            server billing misconfig and is logged but ignored.
          - HTTP 401: latch ``_invalid_key_reason``. This is the one
            terminal state — the operator must mint a new key.
        """
        if self.debug_mode:
            return
        try:
            client = await self._client()
            res = await client.post(
                self.endpoint_telemetry,
                json={
                    "event": event_type,
                    "sdk_version": self.sdk_version,
                    "meta": meta or {},
                    "session_id": self._mapper_session_id,
                },
                timeout=2.0,
            )
            self._absorb_billing_response(res, event_type=event_type)
        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.TimeoutException, httpx.RemoteProtocolError):
            # Trip the failover so the next request lands on the
            # workers.dev fallback instead of blocking on DNS / timeouts again.
            self._maybe_swap_to_fallback()
        except Exception:
            # Telemetry must never break the customer's pipeline.
            pass

    def _absorb_billing_response(
        self,
        res: "httpx.Response",
        event_type: Optional[str] = None,
    ) -> None:
        """Parse a telemetry HTTP response and update billing state.

        Tolerant of malformed bodies — never raises.
        """
        try:
            data = res.json() if res.content else {}
        except Exception:
            data = {}
        if not isinstance(data, dict):
            data = {}

        tickets = data.get("tickets")
        if isinstance(tickets, list):
            self._mapper_tickets.extend(str(t) for t in tickets if isinstance(t, str))
            if data.get("session_id"):
                self._mapper_session_id = str(data["session_id"])

        # Always update the cached balance when the server returns one,
        # regardless of HTTP status. This is the "distributed swarm
        # synchronisation" primitive — every response is a truth-from-DB
        # broadcast that lets independent SDK instances on the same key
        # converge on the real balance with zero extra round-trips.
        if "balance" in data:
            try:
                self._billing_balance = int(data.get("balance") or 0)
            except (TypeError, ValueError):
                pass

        if res.status_code == 402:
            # Free events that 402 mean server-side billing config drift,
            # not a customer credit signal. Log, do not enter grace mode.
            if event_type not in (None, "parse_ui", "execute"):
                log.error(
                    "Aliax: server returned 402 for non-billable event %s — "
                    "billing misconfiguration suspected. Ignoring.",
                    event_type,
                )
                return
            self._credits_exhausted = True
            if self.anonymous or data.get("code") == "sandbox_exhausted":
                # A sandbox grant can't be topped up — the only exit is a
                # free account, so say that instead of "top up".
                log.error(
                    "Aliax: free anonymous sandbox allowance is used up. "
                    "Create a free account (1,000 credits) at %s and set "
                    "ALIAX_API_KEY to keep going.",
                    _anon.SIGNUP_URL,
                )
                return
            log.error(
                "Aliax: past overdraft floor (balance=%s). Next parse_ui "
                "will pause for grace recovery; top up at "
                "https://aliax.xyz/dashboard to resume immediately.",
                self._billing_balance,
            )
            return

        if res.status_code == 401:
            code = data.get("code")
            # An expired 30-day sandbox credential is not an operator
            # error — silently mint a replacement instead of locking the
            # customer's agent loop.
            if data.get("rebootstrap") and self.anonymous:
                if self._rebootstrap_anonymous():
                    log.info("Aliax: sandbox credential renewed automatically.")
                    return
            if code in ("revoked_key", "revoked_api_key"):
                self._invalid_key_reason = "revoked_api_key"
            elif code in ("expired_key", "expired_api_key"):
                self._invalid_key_reason = "expired_api_key"
            else:
                self._invalid_key_reason = "invalid_api_key"
            log.error(
                "Aliax: API key rejected by server (%s). Calls are locked "
                "until a fresh key is provided.",
                self._invalid_key_reason,
            )
            return


        if 200 <= res.status_code < 300 and (
            data.get("ok") or data.get("status") == "ok"
        ):
            # Server accepted the request — by definition we are above
            # the overdraft floor. Clear the exhaustion flag so the next
            # parse_ui takes the fast async path. This is what makes the
            # SDK self-healing across top-ups: no manual unlock needed.
            if self._credits_exhausted:
                self._credits_exhausted = False
                log.info(
                    "Aliax: balance restored (balance=%s) — resuming.",
                    self._billing_balance,
                )

    def billing_status(self) -> dict:
        """Return the last billing state observed via telemetry.

        Cheap, in-memory snapshot — does not hit the network. Returns
        ``{locked, reason, balance, credits_exhausted}``.

        ``locked`` is True iff the key was rejected (401) — that state
        cannot self-heal. ``credits_exhausted`` is True iff the last
        billable response was 402 and grace recovery hasn't yet cleared
        it; it is NOT terminal — the next ``parse_ui`` will retry.

        ``balance`` is the user's remaining credit balance (``None``
        until the first telemetry response is processed). May be
        negative under the bounded overdraft model — that is normal.
        """
        return {
            "locked": self._invalid_key_reason is not None,
            "reason": self._invalid_key_reason,
            "balance": self._billing_balance,
            "credits_exhausted": self._credits_exhausted,
        }

    async def refresh_billing_status(self) -> dict:
        """Force a network round-trip to refresh billing state.

        Posts a zero-cost telemetry ping (``execute`` is free server-side)
        so ``billing_status()`` returns a fresh ``balance`` and clears
        any stale ``credits_exhausted`` flag the moment the server
        confirms the account is back above the overdraft floor.
        """
        await self._emit_telemetry("execute", units=1, meta={"reason": "refresh"})
        return self.billing_status()





    def _telemetry_in_background(
        self,
        event_type: str,
        units: int = 1,
        meta: Optional[dict] = None,
    ) -> None:
        """Schedule _emit_telemetry without awaiting it.

        CRITICAL: the returned Task MUST be retained in a strong
        reference, otherwise the event loop's weak-only bookkeeping can
        GC the task between scheduling and the first ``await`` inside
        ``_emit_telemetry`` — silently cancelling the billing ping. The
        symptom is the local cached balance staying stale forever while
        the server-side ledger never sees the debit.
        """
        try:
            loop = asyncio.get_running_loop()
            task = loop.create_task(self._emit_telemetry(event_type, units, meta))
            self._bg_tasks.add(task)
            task.add_done_callback(self._bg_tasks.discard)
        except RuntimeError:
            # No running loop — fall back to a synchronous best-effort POST.
            if self.debug_mode:
                return
            try:
                res = httpx.post(
                    self.endpoint_telemetry,
                    headers=self._auth_headers(),
                    json={
                        "event": event_type,
                        "sdk_version": self.sdk_version,
                        "meta": meta or {},
                    },
                    timeout=2.0,
                )
                self._absorb_billing_response(res, event_type=event_type)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Page helpers (robust against post-action navigation)
    # ------------------------------------------------------------------

    def _invalidate_page_cache(self, page) -> None:
        """Drop the cached node map for ``page``.

        Called whenever we know the cached element_id → pixel-coords
        mapping is stale: post-navigation re-injection, SCROLL, NAVIGATE,
        or any context-destroyed retry. Otherwise the next execute() may
        click at coordinates from the *previous* page.
        """
        try:
            self._last_map_by_page.pop(page, None)
        except (TypeError, KeyError):
            pass

    async def _safe_inject(self, page) -> None:
        """Inject the core mapper JS. Retries once after a context-
        destroyed error caused by a still-settling navigation.

        Invariant: ALWAYS drop the cached node map first. Re-injection
        reseeds ``__AliaxCore._idCache`` (a fresh WeakMap on a fresh
        ``window.__AliaxCore``), so any cached Python-side records for
        this page now hold ids the new mapper instance has never seen.
        Letting them survive would cause ``_coords_for`` to hit on a
        stale record and click pre-injection coordinates that no
        longer map to any live element.
        """
        self._invalidate_page_cache(page)
        await self._ensure_licensed()
        for attempt in range(2):
            try:
                # Evaluate the generated IIFE as global script source. Passing
                # it directly makes Playwright parse the obfuscated payload
                # as a function-body expression on some Chromium versions.
                await page.evaluate("(source) => (0, eval)(source)", self.core_js)
                return
            except Exception as exc:
                if attempt == 0 and _is_context_destroyed(exc):
                    with suppress(Exception):
                        await page.wait_for_load_state(
                            "domcontentloaded", timeout=10_000
                        )
                    continue
                raise

    async def _ensure_licensed(self) -> None:
        if self.core_js and self._mapper_tickets:
            return
        async with self._mapper_session_lock:
            if self.core_js and not self._mapper_tickets:
                await self._emit_telemetry("version_check", units=1, meta={"reason": "mapper_ticket_refresh"})
                if self._mapper_tickets:
                    return
            if self.core_js is None:
                client = await self._client()
                response = await client.post(
                    self.endpoint_session,
                    headers=self._auth_headers(),
                    json={"sdk_version": self.sdk_version},
                    timeout=5.0,
                )
                if response.status_code != 200:
                    self._absorb_billing_response(response, event_type="version_check")
                    raise AliaxError(f"Aliax mapper session rejected ({response.status_code}).")
                payload = response.json()
                key_hex = payload.get("asset_key")
                if not isinstance(key_hex, str) or len(key_hex) != 64:
                    raise AliaxError("Aliax mapper session returned no valid asset key.")
                try:
                    container = _load_bundled_bytes("dom-mapper.dat")
                    self.core_js = open_container(container, bytes.fromhex(key_hex))
                except Exception as exc:
                    raise AliaxError(f"Aliax mapper artifact could not be opened: {exc}") from exc
                self._mapper_session_id = str(payload.get("session_id") or "")
                self._mapper_tickets = [str(t) for t in payload.get("tickets", []) if isinstance(t, str)]
            if not self._mapper_tickets:
                raise AliaxError("Aliax mapper session has no execution tickets.")

    def _next_mapper_ticket(self) -> str:
        if not self._mapper_tickets:
            raise AliaxError("Aliax execution ticket exhausted; wait for telemetry renewal.")
        return self._mapper_tickets.pop(0)

    async def _safe_eval(self, page, expression, arg=None):
        """page.evaluate() with the same one-shot retry against navigation."""
        for attempt in range(2):
            try:
                if arg is None:
                    return await page.evaluate(expression)
                return await page.evaluate(expression, arg)
            except Exception as exc:
                if attempt == 0 and _is_context_destroyed(exc):
                    self._invalidate_page_cache(page)
                    with suppress(Exception):
                        await page.wait_for_load_state(
                            "domcontentloaded", timeout=10_000
                        )
                    await self._safe_inject(page)
                    continue
                raise

    def _dbg(self, stage: str, **fields) -> None:
        """Structured debug logger for every action stage in execute().

        No-op unless ``self.debug_mode`` is True — production runs stay
        silent, but flipping debug_mode=True floods stdout AND the
        ``aliax`` logger with one ``[Aliax]`` line per action sub-step
        (begin / probe / fallback / verify / done) so you never have
        to guess what the agent's last action actually did to the DOM.
        """
        if not getattr(self, "debug_mode", False):
            return
        try:
            msg = _format_dbg(stage, fields)
            log.info(msg)
            print(msg, flush=True)
        except Exception:
            # Logging must never break the action loop.
            pass



    # ------------------------------------------------------------------
    # parse_ui — the headline feature
    # ------------------------------------------------------------------

    # Grace-recovery tuning. 3 ticks × 3s = ~9s of patient retry before
    # we surface AliaxOutOfCreditsError. Wide enough to cover a normal
    # Paystack webhook + Supabase commit + Realtime fan-out, narrow
    # enough that a genuinely-broke agent doesn't hang for a minute.
    _GRACE_TICKS = 3
    _GRACE_SLEEP_S = 3.0

    def _ensure_grace_lock(self) -> asyncio.Lock:
        """Lazy init for the grace-recovery serialisation lock.

        Built lazily because ``__init__`` may run before any event loop
        exists. Uses the same threading.Lock + double-checked init as
        ``_http_lock`` so two coroutines on different event loops can
        never each create their own asyncio.Lock — that would orphan
        one of them and raise "Future attached to a different loop"
        the first time the late-arriving caller hit ``async with``.
        """
        if self._grace_lock is None:
            with self._grace_lock_init:
                if self._grace_lock is None:
                    self._grace_lock = asyncio.Lock()
        return self._grace_lock

    def _ensure_parse_lock(self, page) -> asyncio.Lock:
        """Per-Page asyncio.Lock for parse_ui.

        Two coroutines that call ``parse_ui`` on the same Page must NOT
        interleave their blur → map → draw → screenshot → clear sequence:
        the JS ``drawOverlay`` clears any existing overlay on entry, so
        coroutine B will silently strip coroutine A's overlay mid-shot
        and both ParseContexts come back without Set-of-Mark boxes.

        Locks are keyed by Page via WeakKeyDictionary so closed pages are
        evicted automatically. The dict mutation itself is guarded by a
        threading.Lock so cross-thread page handoffs don't race on
        `__setitem__`.
        """
        try:
            lock = self._parse_lock_by_page.get(page)
        except TypeError:
            # Page not weak-refable (rare — some test mocks). Fall back
            # to a process-wide singleton so behaviour is still correct,
            # just less granular.
            lock = self._parse_lock_by_page.get(self)  # type: ignore[arg-type]
            if lock is None:
                with self._parse_locks_init:
                    lock = self._parse_lock_by_page.get(self)  # type: ignore[arg-type]
                    if lock is None:
                        lock = asyncio.Lock()
                        try:
                            self._parse_lock_by_page[self] = lock  # type: ignore[index]
                        except TypeError:
                            pass
            return lock
        if lock is None:
            with self._parse_locks_init:
                lock = self._parse_lock_by_page.get(page)
                if lock is None:
                    lock = asyncio.Lock()
                    self._parse_lock_by_page[page] = lock
        return lock

    async def _grace_recover(self) -> bool:
        """3-tick synchronous grace loop.

        Called from ``parse_ui()`` only when ``_credits_exhausted`` is
        True at entry — i.e. a previous billable ping came back 402.
        Sleeps GRACE_SLEEP_S, then fires a free ``execute`` ping to
        refresh balance from the server. As soon as any response clears
        the exhaustion flag (server returned 200, meaning the account
        is back above the overdraft floor), returns True. After all
        ticks expire, returns False and the caller raises.

        Serialised across coroutines so a burst of 100 concurrent
        parse_ui()s at the moment of exhaustion does not fire 300
        sync pings in parallel.
        """
        lock = self._ensure_grace_lock()
        async with lock:
            # Another coroutine may have recovered while we waited.
            if not self._credits_exhausted:
                return True
            for tick in range(self._GRACE_TICKS):
                await asyncio.sleep(self._GRACE_SLEEP_S)
                # Free ping — `execute` is server-side cost=0 so we can
                # poll without burning credits ourselves.
                await self._emit_telemetry(
                    "execute", units=1, meta={"reason": "grace_check", "tick": tick + 1}
                )
                # 401 latches separately — bail out immediately, the
                # outer loop will raise AliaxInvalidKeyError.
                if self._invalid_key_reason:
                    return False
                if not self._credits_exhausted:
                    log.info(
                        "Aliax: grace recovery succeeded on tick %d (balance=%s).",
                        tick + 1, self._billing_balance,
                    )
                    return True
            return False

    async def parse_ui(
        self,
        page,
        *,
        render_config: Optional[Mapping[str, Any]] = None,
        draw_overlay: bool = True,
        min_size: int = 12,
        max_elements: int = 200,
        # Legacy kwargs — kept for backward compatibility with pre-1.1
        # callers. ``image_format`` / ``image_quality`` are folded into
        # ``render_config`` if the new knob is not provided. ``max_image_dim``
        # is now a NO-OP (the SDK no longer ships Pillow, so it cannot
        # resize image bytes in Python) and emits a one-line debug log
        # the first time it's seen.
        max_image_dim: Optional[int] = None,
        image_format: Optional[str] = None,
        image_quality: Optional[int] = None,
        _internal_retry: bool = False,
    ) -> ParseContext:
        """Translate the live page into a VLM-friendly Set-of-Mark image.

        Steps (all native — zero Python image-processing deps):

          1. Inject the bundled DOM mapper into the page.
          2. Visually blur PII (passwords, emails, card numbers, plus any
             ``redact_selectors`` you configured).
          3. Walk the DOM once — pull the interactable node list +
             viewport telemetry in a single ``page.evaluate`` round-trip.
          4. Ask the bundled JS to inject a pointer-events:none overlay
             of numbered, colored boxes at the top of the z-stack. The
             host page's layout is untouched (``contain: strict`` +
             ``position: fixed``).
          5. Take ONE Playwright screenshot at the requested
             ``render_config["format"]`` / ``["quality"]``. Chromium
             handles the JPEG encoding natively in C++; the SDK never
             touches the pixel buffer.
          6. Tear down the overlay and the PII blur in a try/finally so
             the host page is left exactly as we found it.
          7. Cache the node list keyed by the page object so a later
             ``execute(page, {"element_id": "el_41"})`` can resolve.
          8. Send a ``/v1/telemetry`` ping. The ping mode is adaptive:
             when the cached balance is positive we fire-and-forget
             (sub-millisecond overhead). When the cached balance has
             gone non-positive (we're in overdraft), we await the ping
             synchronously so the kill-switch is authoritative before
             the next call.

        ``render_config`` — the only knob the SDK exposes for VLM token
        economics::

            ctx = await aliax.parse_ui(
                page,
                render_config={"format": "jpeg", "quality": 60},
            )

        Format is ``"jpeg"`` (default — what every VLM bills at the
        cheap "low-detail" tier) or ``"png"`` (lossless). Quality is
        clamped to ``[30, 100]`` so a power-user's ``quality=5`` typo
        cannot turn the drawn Set-of-Mark IDs into illegible pixel mush
        and crash their agent.

        Returns a :class:`ParseContext`. Hand its ``image_bytes`` and
        ``elements`` (or :meth:`ParseContext.llm_text_block`) to your VLM.

        ``_internal_retry`` is private — set when this call is the
        recursive re-run triggered by a mid-parse SPA navigation. It
        suppresses the duplicate telemetry ping so a single user-facing
        ``parse_ui`` call still debits exactly one credit.
        """
        # 0a. Invalid-key latch is terminal — no self-heal possible.
        if self._invalid_key_reason:
            raise AliaxInvalidKeyError(
                f"Aliax API key was rejected by the server "
                f"({self._invalid_key_reason}). Mint a new one in the dashboard."
            )

        # 0b. Self-healing brake. If a previous billable ping returned
        #     402 we are past the overdraft floor. Run the 3-tick grace
        #     loop BEFORE doing any client-side work — that way a
        #     pending Paystack top-up has ~9s to land and unstick the
        #     agent without a manual unlock(). If recovery fails we
        #     raise BEFORE the screenshot so we don't burn local
        #     compute on a request we know will be rejected.
        if self._credits_exhausted:
            recovered = await self._grace_recover()
            if self._invalid_key_reason:
                raise AliaxInvalidKeyError(
                    f"Aliax API key was rejected by the server "
                    f"({self._invalid_key_reason}). Mint a new one in the dashboard."
                )
            if not recovered:
                raise AliaxOutOfCreditsError(balance=self._billing_balance)

        # Resolve the screenshot config once — legacy kwargs get folded in.
        cfg = _normalize_render_config(
            render_config,
            legacy_format=image_format,
            legacy_quality=image_quality,
        )
        fmt = cfg["format"]
        quality = cfg["quality"]
        # max_image_dim is no longer honoured — Playwright cannot
        # arbitrarily downscale a screenshot, and Pillow has been
        # dropped from the wheel for zero-bloat parity. Surface a
        # one-line warning the FIRST time a non-recursive call passes
        # it; latched on the instance so a hot agent loop doesn't spam
        # logs every parse. Default __init__ value is 0 so default-
        # constructed clients never trigger the notice at all.
        if (max_image_dim or self.max_image_dim) and not _internal_retry:
            if not self._max_image_dim_warned:
                self._max_image_dim_warned = True
                log.warning(
                    "Aliax: max_image_dim=%s is a no-op since v1.0 (the SDK "
                    "ships without Pillow). Pass render_config={'quality': N} "
                    "to parse_ui() for VLM token economy. This notice fires "
                    "only once per client instance.",
                    max_image_dim or self.max_image_dim,
                )

        # Per-Page lock — see _ensure_parse_lock for the rationale.
        # Two concurrent parse_ui calls on the same Page would otherwise
        # mutually clear each other's overlay mid-screenshot. The lock
        # is released before any recursive retry so SPA-nav recovery
        # doesn't deadlock on itself.
        parse_lock = self._ensure_parse_lock(page)

        # Pre-compute so the post-lock retry decision is reachable.
        nav_detected_via_epoch = False
        nav_detected_via_url = False
        url_now = ""
        elements: list = []
        viewport: dict = {}
        truncated: bool = False
        image_bytes: bytes = b""
        dom_result: Any = None

        async with parse_lock:
            # 1. Inject the obfuscated mapper (nav-safe).
            await self._safe_inject(page)

            # Capture page identity BEFORE any work:
            #   - URL: catches the simple A→B case.
            #   - Nav epoch: a counter we stamp onto `window`. Since SPA
            #     navigation wipes window state, comparing the post-shot
            #     epoch to our captured value detects A→B→A round-trips
            #     that URL-equality would silently miss (Next.js prefetch,
            #     React Router loader retries) — the "torn ParseContext"
            #     hazard from BUG-5.
            url_before = ""
            try:
                u = page.url
                url_before = u if isinstance(u, str) else ""
            except Exception:
                url_before = ""
            nav_epoch_before: Optional[int] = None
            try:
                ep = await self._safe_eval(
                    page,
                    "() => { window.__aliaxNavEpoch = (window.__aliaxNavEpoch || 0) + 1;"
                    " return window.__aliaxNavEpoch; }",
                )
                if isinstance(ep, (int, float)):
                    nav_epoch_before = int(ep)
            except Exception:
                nav_epoch_before = None

            # 2. Visual PII blur + DOM map + overlay paint + native screenshot —
            #    wrapped in try/finally so a crash in any step still tears
            #    down the overlay layer and unblurs the PII before returning
            #    control to the caller. Order matters: we map FIRST (so the
            #    overlay sees the same coordinate snapshot the VLM gets),
            #    then paint, then snap, then tear down in reverse.
            selectors_json = json.dumps(self.redact_selectors)
            # OPTIMISTIC blur flag (BUG-8): blurPII iterates element-by-
            # element in JS; if the JS context is destroyed mid-loop
            # SOME elements get the `aliax-redacted` class even though
            # the Python await raised. Setting blurred=True before the
            # call guarantees the finally always attempts unblurPII, and
            # unblurPII on a clean page is a harmless no-op.
            blurred = True
            overlay_drawn = False
            try:
                await self._safe_eval(
                    page,
                    f"() => window.__AliaxCore.blurPII({selectors_json})",
                )

                # 3. Map DOM + viewport in one round-trip. Returned bounds
                #    are CSS px, viewport-relative — drop-in for the JS
                #    overlay (which uses position:fixed at top-left).
                dom_result = await self._safe_eval(
                    page,
                    "(opts) => window.__AliaxCore.mapDOM(opts)",
                    {"min_size": min_size, "max_elements": max_elements, "__t": self._next_mapper_ticket()},
                )

                # Pull the element list early so the overlay JS can re-use it.
                if isinstance(dom_result, dict) and "elements" in dom_result:
                    overlay_elements = list(dom_result.get("elements") or [])
                else:
                    overlay_elements = list(dom_result or [])

                # 4. Browser-native overlay paint. Chromium renders the
                #    rectangles + numeric labels in <2ms for 200 boxes;
                #    no Python image processing involved.
                if draw_overlay and overlay_elements:
                    with suppress(Exception):
                        await self._safe_eval(
                            page,
                            "(els) => window.__AliaxCore.drawOverlay(els)",
                            overlay_elements,
                        )
                        overlay_drawn = True

                # 5. Native screenshot. Playwright's CDP path lets us pick
                #    JPEG quality directly — zero Pillow, zero re-encode.
                #    PNG is supported but ignores `quality`.
                shot_kwargs: dict = {"type": fmt}
                if fmt == "jpeg":
                    shot_kwargs["quality"] = quality
                image_bytes = await page.screenshot(**shot_kwargs)
            finally:
                # 6a. Clear the overlay before unblur so a navigation race
                #     can't leave a giant fixed-position layer pinned to a
                #     fresh document.
                if overlay_drawn:
                    with suppress(Exception):
                        await self._safe_eval(
                            page,
                            "() => { if (window.__AliaxCore) window.__AliaxCore.clearOverlay(); }",
                        )
                # 6b. Unblur with ONE explicit retry on non-nav errors
                #     (BUG-6). _safe_eval already handles context-destroyed
                #     internally, but a transient CDP timeout would
                #     otherwise leave the live page permanently candy-
                #     striped with the .aliax-redacted blur class.
                if blurred:
                    unblur_js = (
                        "() => { if (window.__AliaxCore) "
                        "window.__AliaxCore.unblurPII(); }"
                    )
                    try:
                        await self._safe_eval(page, unblur_js)
                    except Exception as exc1:
                        try:
                            await asyncio.sleep(0.3)
                            await self._safe_eval(page, unblur_js)
                        except Exception as exc2:
                            log.warning(
                                "Aliax: unblurPII failed twice (%s; %s); "
                                "page may remain visually blurred until "
                                "next navigation",
                                exc1, exc2,
                            )

            # Detect a mid-call navigation. URL check catches A→B; the
            # nav-epoch check catches A→B→A round-trips where URL is
            # identical but window state was reset by navigation.
            try:
                url_now = page.url if isinstance(page.url, str) else ""
            except Exception:
                url_now = ""
            nav_detected_via_url = bool(
                url_before and url_now and url_before != url_now
            )
            if nav_epoch_before is not None:
                try:
                    ep_after = await self._safe_eval(
                        page, "() => window.__aliaxNavEpoch",
                    )
                except Exception:
                    ep_after = None
                if not isinstance(ep_after, (int, float)) or int(ep_after) != nav_epoch_before:
                    nav_detected_via_epoch = True

        # Lock released. Recursive retry must happen OUTSIDE the lock or
        # _ensure_parse_lock would re-enter and deadlock (asyncio.Lock
        # is NOT reentrant).
        if nav_detected_via_url or nav_detected_via_epoch:
            log.info(
                "Aliax: page navigated mid-parse_ui (url %s → %s, epoch_drift=%s);"
                " re-running once",
                url_before if 'url_before' in locals() else "?",
                url_now, nav_detected_via_epoch,
            )
            self._invalidate_page_cache(page)
            return await self.parse_ui(
                page,
                render_config=render_config,
                draw_overlay=draw_overlay,
                min_size=min_size,
                max_elements=max_elements,
                max_image_dim=max_image_dim,
                image_format=image_format,
                image_quality=image_quality,
                # (audit #3) The outer call already booked telemetry
                # responsibility for this user-visible invocation; the
                # recursive retry must NOT debit a second credit.
                _internal_retry=True,
            )

        if isinstance(dom_result, dict) and "elements" in dom_result:
            elements = list(dom_result.get("elements") or [])
            viewport = dict(dom_result.get("viewport") or {})
            truncated = bool(dom_result.get("truncated"))
        else:
            elements = list(dom_result or [])
            viewport = {}
            truncated = False
        if truncated:
            log.warning(
                "Aliax: DOM mapper hit its element cap (%d); some elements "
                "are missing from ctx.elements", max_elements,
            )

        # Best-effort URL — Playwright's async API exposes page.url as a
        # plain str property (never a coroutine).
        try:
            url = page.url if isinstance(page.url, str) else ""
        except Exception:
            url = ""

        # Clairvoyance: derive the clean route slice + page title so the
        # VLM gets a "where am I standing?" anchor for free. Both are
        # best-effort — title() can throw on a closed page mid-call,
        # urlparse never raises but may return an empty path for opaque
        # schemes (about:blank, data:). Failures degrade to "".
        path = ""
        if url:
            try:
                path = urlparse(url).path or ""
            except Exception:
                path = ""
        title = ""
        try:
            t = await page.title()
            if isinstance(t, str):
                title = t.strip()[:200]
        except Exception:
            title = ""

        # Image dimensions — parsed from the bytes directly (tiny
        # hand-rolled JPEG/PNG header reader; no Pillow needed). Fall
        # back to viewport*dpr if the parser bails on a weird codec.
        mime = f"image/{fmt}"
        size = _image_dimensions(image_bytes, fmt)
        if size is None:
            try:
                dpr_fb = float((viewport or {}).get("dpr") or 1.0) or 1.0
            except (TypeError, ValueError):
                dpr_fb = 1.0
            vw = int((viewport or {}).get("width") or 0)
            vh = int((viewport or {}).get("height") or 0)
            if vw and vh:
                size = (int(vw * dpr_fb), int(vh * dpr_fb))
            else:
                size = (0, 0)

        # 7. Cache for execute() — WeakKeyDictionary auto-evicts closed pages.
        try:
            self._last_map_by_page[page] = elements
        except TypeError:
            # Some Playwright mocks may not be weak-refable; degrade silently.
            pass

        # 7b. Failure-Detection Net — hash the (url, spatial_map) signature.
        # Three consecutive identical hashes is the Gatekeeper's first
        # admissible proof that the agent is stuck.
        #
        # CRITICAL: do NOT record state for an internal SPA-recovery
        # retry. A mid-parse navigation triggers a recursive parse_ui
        # call with _internal_retry=True; if we pushed the first
        # (possibly stale / partially-painted) hash AND the second
        # (clean) hash, a single user-visible turn would advance the
        # ring by two. Combined with a pre-existing hash in the ring,
        # that could complete the 3-in-a-row stagnation condition on
        # turn 2 instead of turn 3 — opening the Gatekeeper on false
        # evidence. Only the outer, user-visible parse contributes.
        if not _internal_retry:
            self._record_state(page, url or "", elements)




        # 8. Adaptive telemetry. Two modes:
        #
        #    Fast (cached balance > 0 OR unknown):
        #      Fire-and-forget. Zero added latency on the agent's hot
        #      path. The async ping updates the cached balance for the
        #      NEXT call's mode decision. We may briefly overshoot into
        #      overdraft under burst concurrency — that is fine, the
        #      server's overdraft floor (-500) bounds the leak and
        #      every parse is debited exactly once.
        #
        #    Strict (cached balance <= 0):
        #      Await the ping. The Worker either returns 200 (we are
        #      still inside the overdraft window — continue) or 402
        #      (we just crossed the floor — set _credits_exhausted so
        #      the NEXT parse_ui enters the grace loop). We do NOT
        #      raise here on 402: the user has already received a
        #      ParseContext they paid for; failing the next call is
        #      the cleaner UX. We DO raise on 401 — a rejected key
        #      means the just-emitted ping was unauthenticated, so
        #      the ParseContext shouldn't be acted on.
        #
        # Recursive retries (mid-parse SPA navigation) skip telemetry
        # entirely so a single user-visible parse_ui still debits
        # exactly one credit.
        if not _internal_retry:
            telemetry_meta = {
                "element_count": len(elements),
                "url": url[:200] if url else "",
                "path": (path or "")[:120],
                "title": (title or "")[:120],
                "image_format": fmt,
                "image_quality": quality if fmt == "jpeg" else None,
                "image_bytes": len(image_bytes),
            }
            cached = self._billing_balance
            if cached is None or cached > 0:
                # Fast path — no await, no latency. The response's
                # `balance` field will update the cache for next time.
                self._telemetry_in_background("parse_ui", units=1, meta=telemetry_meta)
            else:
                # Strict path — already in overdraft; confirm with the
                # server before handing the caller a context built on a
                # potentially-rejected debit.
                await self._emit_telemetry("parse_ui", units=1, meta=telemetry_meta)
                if self._invalid_key_reason:
                    raise AliaxInvalidKeyError(
                        f"Aliax API key was rejected by the server "
                        f"({self._invalid_key_reason}). Mint a new one in the dashboard."
                    )

        return ParseContext(
            image_bytes=image_bytes,
            image_mime=mime,
            image_size=size,
            elements=elements,
            viewport=viewport,
            url=url or "",
            path=path or "",
            title=title or "",
            truncated=truncated,
        )

    # ------------------------------------------------------------------
    # execute — runs the LLM's chosen action against the page
    # ------------------------------------------------------------------

    async def execute(
        self,
        page,
        decision: Mapping[str, Any],
        *,
        type_delay_ms: int = 50,
    ) -> dict:
        """Carry out an action the LLM picked, against the live page.

        Accepted ``decision`` shape::

            {
              "action": "CLICK" | "TYPE" | "SCROLL" | "HOVER"
                        | "NAVIGATE" | "WAIT" | "PRESS" | "DONE",
              "element_id": "el_41",     # preferred — resolves via the
                                          # cached node map from parse_ui
              "x": 905, "y": 150,        # OR raw CSS-pixel coords
              "value": "Nike Shoes",     # for TYPE
              "url":   "https://...",    # for NAVIGATE
              "key":   "Enter",          # for PRESS
              "ms":    1500              # for WAIT
            }

        Returns a small dict like ``{"ok": True, "action": "CLICK",
        "coords": [905, 150], "element_id": "el_41"}``. On invalid input
        returns ``{"ok": False, "error": "..."}`` — never raises so the
        agent loop can keep going.
        """
        raw_action = str(decision.get("action") or "").upper()
        # ---- Action aliases ----
        # SCROLL_DOWN / SCROLL_UP / SCROLL_LEFT / SCROLL_RIGHT are
        # first-class verbs in the ReAct vocabulary we recommend in
        # docs ("if you don't see the target, output SCROLL_DOWN").
        # They normalize to SCROLL with an axis hint; the actual delta
        # is computed inside the SCROLL branch from the live container
        # geometry (80% viewport overlap → 20% safety) so a half-clipped
        # button never jumps off-screen past the camera.
        scroll_dir = {
            "SCROLL_DOWN":  ("y", +1),
            "SCROLL_UP":    ("y", -1),
            "SCROLL_RIGHT": ("x", +1),
            "SCROLL_LEFT":  ("x", -1),
        }.get(raw_action)
        if scroll_dir is not None:
            action = "SCROLL"
            decision = dict(decision)  # copy — never mutate caller dict
            decision.setdefault("_axis", scroll_dir[0])
            decision.setdefault("_sign", scroll_dir[1])
        # TYPE_AND_ENTER — Google-search / autocomplete macro. Splitting
        # this into TYPE then PRESS risks state drift when the suggest
        # dropdown remounts the input between turns.
        elif raw_action == "TYPE_AND_ENTER":
            action = "TYPE_AND_ENTER"
        else:
            action = raw_action
        result: dict = {"ok": False, "action": action}

        # One structured line per execute() entry so a debug-mode run
        # can be diff'd action-by-action against the dump artifacts.
        self._dbg(
            "execute.begin",
            action=action,
            element_id=decision.get("element_id") or decision.get("id"),
            x=decision.get("x"),
            y=decision.get("y"),
            url=decision.get("url"),
            key=decision.get("key"),
            ms=decision.get("ms"),
            has_value=_decision_text_value(decision) != "" if action in ("TYPE", "TYPE_AND_ENTER") else None,
        )

        try:
            if action in ("DONE", "FINISH", "NOOP"):
                self._dbg("execute.terminal", action=action)
                result["ok"] = True

            elif action == "REPORT_ISSUE":
                # The AI's escalation tool. Routed through the same
                # Gatekeeper as the developer-facing report_issue() —
                # see that method for the Stamina-Gate logic. Never
                # raises; on rejection we hand a structured message
                # back to the LLM so it has to try again instead of
                # spamming /v1/capture on the first hint of confusion.
                rctx_raw = decision.get("context")
                expected = None
                actual = None
                if isinstance(rctx_raw, Mapping):
                    expected = rctx_raw.get("expected_outcome")
                    actual = rctx_raw.get("actual_outcome")
                rep = await self.report_issue(
                    page,
                    reason=str(decision.get("reason") or "Agent escalation via REPORT_ISSUE"),
                    expected_outcome=expected,
                    actual_outcome=actual,
                    goal=str(decision.get("goal") or "Agent escalation via REPORT_ISSUE"),
                    thoughts=decision.get("thoughts"),
                )
                # Mirror the report payload into the execute result so
                # the orchestrator can branch on result["status"].
                result.update(rep)
                result["ok"] = (rep.get("status") == "success")
                # Short-circuit telemetry for REPORT_ISSUE — report_issue
                # already emits its own capture_failure telemetry on
                # success, and a rejection isn't a billable execute.
                self._dbg("execute.report_issue.end", status=rep.get("status"))
                return result





            elif action == "WAIT":
                ms = int(decision.get("ms") or 1000)
                self._dbg("wait.begin", ms=ms)
                await asyncio.sleep(ms / 1000.0)
                self._dbg("wait.done", ms=ms)
                result["ok"] = True

            elif action == "NAVIGATE":
                url = decision.get("url")
                if not url:
                    self._dbg("navigate.error", reason="missing-url")
                    result["error"] = "NAVIGATE requires `url`"
                else:
                    # SSRF / open-redirect guard: only http(s) is acceptable.
                    # `file://`, `javascript:`, `data:`, `chrome://` would
                    # let an adversarial LLM (or prompt injection in the
                    # page) reach cloud metadata endpoints, exfil local
                    # files, or run arbitrary JS in the browser context.
                    safe_url = str(url).strip()
                    lowered = safe_url.lower()
                    if not (lowered.startswith("http://") or lowered.startswith("https://")):
                        self._dbg("navigate.blocked", scheme=lowered.split(":", 1)[0], url=safe_url)
                        result["error"] = (
                            "NAVIGATE blocked: only http:// and https:// URLs "
                            "are allowed. Got scheme " + lowered.split(":", 1)[0]
                        )
                    else:
                        # Stale element_id cache is invalid once we navigate.
                        self._invalidate_page_cache(page)
                        # Failure-Detection Net: clear the per-page action
                        # ring too. element_ids are only window-unique —
                        # `el_100` on page A and `el_100` on page B refer
                        # to completely different DOM nodes (the JS counter
                        # resets when the navigation creates a new window).
                        # Without this reset, a cross-page sequence like
                        # [el_100 on A, el_200 on A, el_100 on B, el_200 on B]
                        # would fake an [A,B,A,B] cycle and trigger a false
                        # Gatekeeper escalation. We also clear the state
                        # ring — the new page legitimately has a different
                        # DOM hash, but leaving the old hashes in place can
                        # produce a [old, old, new, new, ...] tail that
                        # delays legitimate stagnation detection on the new
                        # page by up to 2 turns.
                        self._reset_failure_history(page)
                        self._dbg("navigate.begin", url=safe_url)
                        await page.goto(safe_url, wait_until="domcontentloaded")
                        try:
                            final_url = page.url
                        except Exception:
                            final_url = safe_url
                        self._dbg("navigate.done", requested=safe_url, final=final_url)
                        result["ok"] = True
                        result["url"] = safe_url


            elif action == "COMBO":
                # ---- Dual-Action Combo (max 2 steps) ----
                # The VLM chains TWO trivially-sequential actions in
                # ONE turn (e.g. TYPE name + CLICK Continue, or CLICK
                # accept-cookies + SCROLL_DOWN). Strict ceiling of 2 —
                # longer chains drift and compound errors. Halt on
                # first failure and report partial_success so the AI
                # can re-scan the (possibly half-mutated) DOM.
                #
                # Each sub-action is dispatched back through execute()
                # so it inherits the full DNA-stamp / locator-first /
                # coord-fallback pipeline, debug logging, telemetry
                # accounting, and capture-failure plumbing. Nested
                # COMBO is rejected (would explode the 2-step ceiling
                # into a tree). BATCH_TYPE IS allowed as a sub-action —
                # the canonical "fill the whole form + click Submit"
                # combo collapses an N-field signup into a single turn.
                actions_list = decision.get("actions")
                if not isinstance(actions_list, list) or not actions_list:
                    result["error"] = "COMBO requires `actions`: a list of 1-2 sub-action dicts."
                    self._dbg("combo.error", reason="missing-actions")
                elif len(actions_list) > 2:
                    result["error"] = "COMBO violation: maximum of 2 sequential actions per turn."
                    self._dbg("combo.error", reason="too-many", count=len(actions_list))
                else:
                    self._dbg("combo.begin", count=len(actions_list))
                    completed: list = []
                    combo_ok = True
                    fail_reason: Optional[str] = None
                    fail_step: Optional[int] = None
                    for idx, sub in enumerate(actions_list):
                        if not isinstance(sub, Mapping):
                            combo_ok = False
                            fail_step = idx + 1
                            fail_reason = f"step {idx+1} is not an action object"
                            break
                        sub_raw = str(sub.get("action") or "").upper()
                        if sub_raw == "COMBO":
                            combo_ok = False
                            fail_step = idx + 1
                            fail_reason = f"step {idx+1}: nested COMBO is not allowed"
                            break
                        self._dbg("combo.step.begin", step=idx + 1, sub_action=sub_raw,
                                  element_id=sub.get("element_id") or sub.get("id"))
                        try:
                            step_result = await self.execute(page, sub, type_delay_ms=type_delay_ms)
                        except Exception as exc:
                            step_result = {"ok": False, "error": str(exc), "action": sub_raw}
                        completed.append({
                            "step": idx + 1,
                            "action": step_result.get("action") or sub_raw,
                            "ok": bool(step_result.get("ok")),
                            "element_id": step_result.get("element_id"),
                            "execution_tier": step_result.get("execution_tier"),
                            "error": step_result.get("error"),
                        })
                        self._dbg("combo.step.result", step=idx + 1,
                                  ok=bool(step_result.get("ok")),
                                  tier=step_result.get("execution_tier"),
                                  error=step_result.get("error"))
                        if not step_result.get("ok"):
                            combo_ok = False
                            fail_step = idx + 1
                            fail_reason = step_result.get("error") or f"step {idx+1} ({sub_raw}) failed"
                            break
                        # Brief DOM-stabilization beat between steps —
                        # long enough for an inline-validation paint
                        # or a dismiss-animation, short enough that
                        # we still beat one-action-per-turn latency.
                        if idx < len(actions_list) - 1:
                            try:
                                await page.wait_for_timeout(200)
                            except Exception:
                                await asyncio.sleep(0.2)
                    result["completed_steps"] = completed
                    if combo_ok:
                        result["ok"] = True
                        result["status"] = "success"
                        self._dbg("combo.done", count=len(completed))
                    else:
                        result["ok"] = False
                        result["status"] = "partial_success"
                        result["failed_at_step"] = fail_step
                        result["error"] = fail_reason
                        self._dbg("combo.aborted", completed=len(completed),
                                  failed_at_step=fail_step, reason=fail_reason)

            elif action == "BATCH_TYPE":
                # ---- The Form-Filler Protocol (safe batching) ----
                # The VLM saw N empty inputs on one screenshot and is
                # filling them in a single turn. We strictly enforce:
                #   1. inputs is a list of {element_id, value}
                #   2. NO CLICK / SCROLL / nav verbs allowed inside
                #   3. each target must resolve via DNA stamp + be
                #      visible + editable at fill-time. First failure
                #      aborts the batch and hands control back to the
                #      LLM so it can re-scan the (possibly mutated) DOM.
                inputs = decision.get("inputs")
                if not isinstance(inputs, list) or not inputs:
                    result["error"] = "BATCH_TYPE requires `inputs`: a non-empty list of {element_id, value} entries."
                    self._dbg("batch_type.error", reason="missing-inputs")
                else:
                    self._dbg("batch_type.begin", count=len(inputs))
                    filled: list = []
                    batch_ok = True
                    fail_reason: Optional[str] = None
                    fail_id: Optional[str] = None
                    for idx, task in enumerate(inputs):
                        if not isinstance(task, Mapping):
                            batch_ok = False
                            fail_reason = f"input[{idx}] is not an object"
                            break
                        el_id = task.get("element_id") or task.get("id")
                        val = _decision_text_value(task)
                        stamp = _stamp_from_element_id(el_id) if el_id else None
                        if not stamp:
                            batch_ok = False
                            fail_id = el_id
                            fail_reason = f"input[{idx}] missing element_id"
                            break
                        self._dbg("batch_type.item.begin", index=idx, element_id=el_id, stamp=stamp, value=val)
                        cached = self._cached_record(page, el_id)
                        sub_decision = {"action": "TYPE", "element_id": el_id, "value": val}
                        try:
                            ok_item, extra_item = await self._act_via_locator(
                                page, stamp, "TYPE", sub_decision, cached, type_delay_ms,
                            )
                        except Exception as exc:
                            ok_item, extra_item = False, {"error": str(exc)}
                        self._dbg(
                            "batch_type.item.result",
                            index=idx, element_id=el_id,
                            ok=ok_item, tier=extra_item.get("execution_tier") if ok_item else None,
                        )
                        if not ok_item:
                            batch_ok = False
                            fail_id = el_id
                            fail_reason = (
                                extra_item.get("error")
                                if isinstance(extra_item, Mapping) and extra_item.get("error")
                                else f"failed to type into {el_id} (locator path rejected; element may be hidden or non-editable)"
                            )
                            break
                        filled.append({
                            "element_id": el_id,
                            "value": val,
                            "execution_tier": extra_item.get("execution_tier"),
                        })
                    result["filled"] = filled
                    if batch_ok:
                        result["ok"] = True
                        result["status"] = "success"
                        self._dbg("batch_type.done", count=len(filled))
                    else:
                        # Partial success — the AI gets to re-scan and finish.
                        result["ok"] = False
                        result["status"] = "partial_success"
                        result["error"] = fail_reason
                        if fail_id:
                            result["element_id"] = fail_id
                        self._dbg("batch_type.aborted", filled=len(filled), failed_element=fail_id, reason=fail_reason)

            elif action in ("CLICK", "HOVER", "TYPE", "PRESS", "TYPE_AND_ENTER"):

                # ---- DNA-stamp execution path (Absolute Targeting) ----
                # Resolve element_id → Playwright Locator on the DNA
                # stamp. Falls back through three tiers (intercepted JS
                # click bypass → SPA-wipe semantic rebind → cached
                # coordinate click) before giving up. CANVAS elements
                # never use the locator path: they have no sub-DOM, so
                # absolute (x, y) is the only meaningful target inside
                # them (e.g. Google Maps, trading charts).
                element_id = decision.get("element_id") or decision.get("id")
                stamp = _stamp_from_element_id(element_id)
                cached_record = self._cached_record(page, element_id) if element_id else None
                is_canvas = bool(cached_record and cached_record.get("is_canvas"))

                # ---- State-aware safety catch ----
                # If the cached record (or, when missing, a live probe)
                # says the target is disabled, DO NOT click — clicking a
                # disabled <button> is a no-op in Chromium but clicking
                # an `aria-disabled` <div role="button"> often DOES fire
                # the handler and triggers form-validation toasts that
                # the agent then loops on. We return a structured error
                # the LLM can read and self-correct ("fix form first").
                # HOVER is allowed — hovering a disabled control is fine
                # and sometimes reveals a tooltip explaining WHY.
                if action in ("CLICK", "TYPE", "PRESS", "TYPE_AND_ENTER") and element_id and not is_canvas:
                    disabled_hint = await self._probe_disabled(
                        page, stamp, cached_record,
                    )
                    self._dbg(
                        "stamp.disabled-probe",
                        element_id=element_id, stamp=stamp,
                        disabled=bool(disabled_hint), hint=disabled_hint,
                    )
                    if disabled_hint:
                        result["error"] = (
                            f"Action blocked: element {element_id} is currently "
                            f"disabled ({disabled_hint}). Inspect the surrounding "
                            f"form (unchecked required boxes, empty required "
                            f"fields, invalid inputs) and fix those first."
                        )
                        result["element_id"] = element_id
                        result["execution_tier"] = "blocked_disabled"
                        result["blocked"] = "disabled"
                        self._dbg("execute.blocked", element_id=element_id, reason="disabled")
                        # Telemetry still fires below; intentionally not raising.
                        self._telemetry_in_background(
                            "execute",
                            units=1,
                            meta={
                                "action": action,
                                "ok": False,
                                "tier": "blocked_disabled",
                            },
                        )
                        return result

                stamp_ok = False
                if stamp and not is_canvas and decision.get("x") is None and decision.get("y") is None:
                    self._dbg("stamp.try", element_id=element_id, stamp=stamp, action=action)
                    stamp_ok, stamp_extra = await self._act_via_locator(
                        page, stamp, action, decision, cached_record, type_delay_ms,
                    )
                    self._dbg(
                        "stamp.result",
                        element_id=element_id, stamp=stamp,
                        ok=stamp_ok, tier=stamp_extra.get("execution_tier") if stamp_ok else None,
                        coords=stamp_extra.get("coords") if stamp_ok else None,
                    )
                    if stamp_ok:
                        result["ok"] = True
                        result["element_id"] = element_id

                        result.update(stamp_extra)
                        # Click / PRESS can trigger SPA navigation. The
                        # locator path already did its work; drop the
                        # cached node map so the NEXT execute() against
                        # this page resolves against fresh geometry.
                        if action in ("CLICK", "PRESS", "TYPE_AND_ENTER"):
                            with suppress(Exception):
                                await page.wait_for_load_state(
                                    "domcontentloaded", timeout=5_000
                                )
                            self._invalidate_page_cache(page)

                if not stamp_ok:
                    # ---- Coordinate fallback (legacy + canvas) ----
                    # Reached when: no element_id at all, raw (x,y)
                    # provided, target IS a canvas, or all three
                    # locator tiers failed. The mouse-pixel path is
                    # CLS-fragile by definition (Holy Grail caveat) but
                    # it's the last resort that lets capture_failure()
                    # still see "we tried, here's what happened".
                    coords = await self._coords_for(page, decision)
                    self._dbg(
                        "coords.resolved",
                        action=action,
                        element_id=element_id,
                        coords=([coords["x"], coords["y"]] if coords else None),
                        source=(coords.get("source") if isinstance(coords, dict) else None),
                    )
                    if coords is None:
                        result["error"] = (
                            "Could not resolve element. Provide either "
                            "`element_id` (from the most recent parse_ui) or "
                            "explicit `x`/`y` CSS-pixel coordinates."
                        )
                    else:
                        x, y = coords["x"], coords["y"]
                        if action == "HOVER":
                            self._dbg("coords.HOVER.begin", x=x, y=y)
                            await page.mouse.move(x, y)
                            self._dbg("coords.HOVER.done", x=x, y=y)

                        elif action == "CLICK":
                            # Best-effort scroll-into-view via JS so a
                            # post-action navigation race doesn't
                            # surface as a context-destroyed crash.
                            self._dbg("coords.CLICK.begin", x=x, y=y, element_id=element_id)
                            try:
                                await self._safe_eval(
                                    page,
                                    "([x,y]) => window.scrollBy({"
                                    "  left: Math.max(0, x - window.innerWidth/2 - window.scrollX),"
                                    "  top:  Math.max(0, y - window.innerHeight/2 - window.scrollY),"
                                    "  behavior: 'instant'"
                                    "})",
                                    [x, y],
                                )
                            except Exception as exc:
                                self._dbg("coords.CLICK.scroll-into-view-failed", error=str(exc))
                            await page.mouse.click(x, y)
                            with suppress(Exception):
                                await page.wait_for_load_state(
                                    "domcontentloaded", timeout=5_000
                                )
                            self._invalidate_page_cache(page)
                            self._dbg("coords.CLICK.done", x=x, y=y)



                        elif action == "TYPE":
                            value = _decision_text_value(decision)
                            stamp_for_js = coords.get("element_id") or decision.get("element_id")
                            stamp_for_js = _stamp_from_element_id(stamp_for_js)
                            self._dbg("coords.TYPE.begin", stamp=stamp_for_js, x=x, y=y, value=value)
                            react_done = await self._set_plain_editable_value(page, value, stamp_for_js)
                            if not (react_done and react_done.get("ok")):
                                self._dbg("coords.TYPE.retry-after-click", stamp=stamp_for_js, reason=react_done.get("reason") if isinstance(react_done, dict) else None)
                                await page.mouse.click(x, y)
                                react_done = await self._set_plain_editable_value(page, value, stamp_for_js)
                            if not (react_done and react_done.get("ok")):
                                self._dbg("coords.TYPE.keyboard-fallback", stamp=stamp_for_js, reason=react_done.get("reason") if isinstance(react_done, dict) else None)
                                with suppress(Exception):
                                    await page.keyboard.press("ControlOrMeta+A")
                                    await page.keyboard.press("Backspace")
                                await page.keyboard.type(value, delay=type_delay_ms)
                                final_val = await self._verify_typed_value(page, stamp_for_js)
                                self._dbg("coords.TYPE.keyboard-verify", stamp=stamp_for_js, expected=value, got=final_val, ok=(final_val == value))
                            else:
                                self._dbg("coords.TYPE.done", stamp=stamp_for_js, method=react_done.get("method"), valueAfter=react_done.get("valueAfter"))
                            result["value"] = value

                        elif action == "TYPE_AND_ENTER":
                            value = _decision_text_value(decision)
                            stamp_for_js = coords.get("element_id") or decision.get("element_id")
                            stamp_for_js = _stamp_from_element_id(stamp_for_js)
                            self._dbg("coords.TYPE_AND_ENTER.begin", stamp=stamp_for_js, x=x, y=y, value=value)
                            react_done = await self._set_plain_editable_value(page, value, stamp_for_js)
                            if not (react_done and react_done.get("ok")):
                                self._dbg("coords.TYPE_AND_ENTER.retry-after-click", stamp=stamp_for_js, reason=react_done.get("reason") if isinstance(react_done, dict) else None)
                                await page.mouse.click(x, y)
                                react_done = await self._set_plain_editable_value(page, value, stamp_for_js)
                            if not (react_done and react_done.get("ok")):
                                self._dbg("coords.TYPE_AND_ENTER.keyboard-fallback", stamp=stamp_for_js)
                                with suppress(Exception):
                                    await page.keyboard.press("ControlOrMeta+A")
                                    await page.keyboard.press("Backspace")
                                await page.keyboard.type(value, delay=type_delay_ms)
                                final_val = await self._verify_typed_value(page, stamp_for_js)
                                self._dbg("coords.TYPE_AND_ENTER.keyboard-verify", stamp=stamp_for_js, expected=value, got=final_val, ok=(final_val == value))
                            else:
                                self._dbg("coords.TYPE_AND_ENTER.done", stamp=stamp_for_js, method=react_done.get("method"), valueAfter=react_done.get("valueAfter"))
                            await page.keyboard.press("Enter")
                            with suppress(Exception):
                                await page.wait_for_load_state(
                                    "domcontentloaded", timeout=5_000
                                )
                            self._invalidate_page_cache(page)
                            result["value"] = value
                            result["key"] = "Enter"

                        elif action == "PRESS":
                            key = _decision_key_value(decision)
                            self._dbg("coords.PRESS.begin", x=x, y=y, key=key)
                            await page.mouse.click(x, y)
                            await page.keyboard.press(key)
                            with suppress(Exception):
                                await page.wait_for_load_state(
                                    "domcontentloaded", timeout=5_000
                                )
                            self._invalidate_page_cache(page)
                            self._dbg("coords.PRESS.done", x=x, y=y, key=key)
                            result["key"] = key


                        result["ok"] = True
                        result["coords"] = [int(x), int(y)]
                        result["execution_tier"] = "coords"
                        if coords.get("element_id"):
                            result["element_id"] = coords["element_id"]

            elif action == "SCROLL":
                # ---- Targeted, safety-overlapped scrolling ----
                # Three modes:
                #   (a) element_id present → scroll THAT container by
                #       ~80% of its own clientHeight/clientWidth. Cures
                #       the Discord/Notion case where <body> doesn't
                #       scroll at all.
                #   (b) explicit dx/dy → scroll the window by those exact
                #       pixel deltas (escape hatch).
                #   (c) bare directional (SCROLL_DOWN/UP/LEFT/RIGHT
                #       without element_id, no dx/dy) → 80% of viewport
                #       on the requested axis. The 20% overlap keeps a
                #       half-clipped button from jumping past the camera.
                element_id = decision.get("element_id") or decision.get("id")
                axis = decision.get("_axis")  # set by alias normalization
                sign = decision.get("_sign")
                raw_dx = decision.get("dx", decision.get("delta_x"))
                raw_dy = decision.get("dy", decision.get("delta_y"))
                stamp = _stamp_from_element_id(element_id) if element_id else None

                self._dbg(
                    "scroll.begin",
                    element_id=element_id, stamp=stamp,
                    axis=axis, sign=sign, dx=raw_dx, dy=raw_dy,
                )

                scrolled = False
                # (a) Targeted container scroll via the DNA stamp.
                if stamp and raw_dx is None and raw_dy is None:
                    safe_stamp = str(stamp).replace('"', '\\"')
                    js = (
                        "({s, axis, sign}) => {"
                        "  const el = document.querySelector('[data-aliax-id=\"' + s + '\"]');"
                        "  if (!el) return {ok:false, reason:'no-node'};"
                        "  const before = {x: el.scrollLeft, y: el.scrollTop};"
                        "  const dy = (axis === 'y' || !axis) ? (sign||1) * Math.max(80, el.clientHeight * 0.8) : 0;"
                        "  const dx = (axis === 'x')          ? (sign||1) * Math.max(80, el.clientWidth  * 0.8) : 0;"
                        "  el.scrollBy({left: dx, top: dy, behavior: 'instant'});"
                        "  const moved = (el.scrollLeft !== before.x) || (el.scrollTop !== before.y);"
                        "  return {ok:true, moved, dx, dy, before, after:{x:el.scrollLeft, y:el.scrollTop}};"
                        "}"
                    )
                    try:
                        probe = await self._safe_eval(page, js, {
                            "s": safe_stamp,
                            "axis": axis or "y",
                            "sign": int(sign) if sign is not None else 1,
                        })
                        self._dbg("scroll.container.probe", probe=probe)
                        if isinstance(probe, dict) and probe.get("ok"):
                            scrolled = True
                            result["target"] = element_id
                            result["scroll"] = [int(probe.get("dx") or 0), int(probe.get("dy") or 0)]
                            # The container hit its scroll limit — fall
                            # back to a window scroll so the agent still
                            # makes progress (e.g. inner panel done,
                            # need to scroll the outer shell now).
                            if not probe.get("moved"):
                                self._dbg("scroll.container.exhausted", element_id=element_id)
                                scrolled = False
                    except Exception as exc:
                        self._dbg("scroll.container.error", error=str(exc))
                        log.debug("Aliax: targeted scroll fell back (%s)", exc)

                # (b) Explicit pixel delta on the window.
                if not scrolled and (raw_dx is not None or raw_dy is not None):
                    dx = int(raw_dx) if raw_dx is not None else 0
                    dy = int(raw_dy) if raw_dy is not None else 0
                    self._dbg("scroll.window.explicit", dx=dx, dy=dy)
                    await self._safe_eval(
                        page,
                        "({dx, dy}) => window.scrollBy({left: dx, top: dy, behavior: 'instant'})",
                        {"dx": dx, "dy": dy},
                    )
                    result["scroll"] = [dx, dy]
                    scrolled = True

                # (c) Directional fallback / panic scroll on the window.
                if not scrolled:
                    ax = axis or "y"
                    sg = int(sign) if sign is not None else 1
                    self._dbg("scroll.window.directional", axis=ax, sign=sg)
                    await self._safe_eval(
                        page,
                        "({axis, sign}) => {"
                        "  const dy = axis === 'y' ? sign * Math.floor(window.innerHeight * 0.8) : 0;"
                        "  const dx = axis === 'x' ? sign * Math.floor(window.innerWidth  * 0.8) : 0;"
                        "  window.scrollBy({left: dx, top: dy, behavior: 'instant'});"
                        "}",
                        {"axis": ax, "sign": sg},
                    )
                    result["scroll"] = (
                        [0, sg * 800] if ax == "y" else [sg * 600, 0]
                    )

                # The viewport (or an inner container) moved — cached element coords are stale.
                self._invalidate_page_cache(page)
                result["ok"] = True
                self._dbg("scroll.done", scroll=result.get("scroll"), target=result.get("target"))




            else:
                self._dbg("execute.unknown-action", action=action)
                result["error"] = f"Unknown action: {action!r}"
        except Exception as e:
            result["ok"] = False
            result["error"] = f"{type(e).__name__}: {e}"
            self._dbg("execute.exception", action=action, error=result["error"])

        # One closing line summarizing what actually happened — pair
        # this with the matching ``execute.begin`` to reconstruct the
        # full action lifecycle without re-reading the dump artifact.
        self._dbg(
            "execute.end",
            action=action,
            ok=bool(result.get("ok")),
            tier=result.get("execution_tier"),
            element_id=result.get("element_id"),
            coords=result.get("coords"),
            error=result.get("error"),
        )

        # Failure-Detection Net — log the targeted element_id for the
        # Cycle-Detection arm of the Gatekeeper. Only targeting verbs
        # (CLICK / TYPE / PRESS / HOVER / TYPE_AND_ENTER) contribute a
        # meaningful id; everything else advances the ring with None so
        # global SCROLLs / WAITs don't fake an [A, B, A, B] cycle.
        #
        # COMBO is intentionally NOT handled here: each COMBO sub-action
        # is dispatched through self.execute(...) recursively (see the
        # COMBO branch), so the sub-call's own _record_action runs at
        # the bottom of that recursive frame. Double-recording (once
        # inside the recursion AND once here from result["completed_steps"])
        # would corrupt the ring — a 2-step COMBO of CLICK el_1 + CLICK el_2
        # would push [el_1, el_2, el_1, el_2] in a single turn and instantly
        # satisfy the period-2 cycle predicate, producing a false escalation.
        #
        # BATCH_TYPE is different: its sub-items go through _act_via_locator
        # directly (NOT execute), so they have no recursive _record_action
        # frame. The outer BATCH_TYPE branch below is the sole recorder.
        if action in ("CLICK", "TYPE", "PRESS", "HOVER", "TYPE_AND_ENTER"):
            self._record_action(
                page, str(decision.get("element_id") or decision.get("id") or "") or None
            )
        elif action == "BATCH_TYPE":
            for filled_info in (result.get("filled") or []):
                if isinstance(filled_info, Mapping):
                    fid = filled_info.get("element_id")
                    self._record_action(page, str(fid) if fid else None)
        elif action == "COMBO":
            # No-op: sub-actions already recorded by their recursive
            # execute() frames. Do NOT push anything here, and do NOT
            # fall through to the catch-all `None` push (which would
            # break legit cycles ending in a COMBO turn).
            pass
        else:
            self._record_action(page, None)

        # Telemetry — 1 unit per execute call (success or not).

        self._telemetry_in_background(
            "execute",
            units=1,
            meta={
                "action": action,
                "ok": bool(result.get("ok")),
                "tier": result.get("execution_tier"),
            },
        )
        return result



    # ------------------------------------------------------------------
    # get_element_coords — escape hatch
    # ------------------------------------------------------------------

    async def get_element_coords(self, page, element_id: str) -> Optional[dict]:
        """Return ``{x, y, width, height, tag, editable, is_canvas}`` for
        an Aliax element id, in CSS pixels (viewport-relative).

        Useful when you have your own Playwright runner and just want the
        raw geometry. ``parse_ui()`` must have been called on this page
        at least once — we re-resolve through the bundled mapper so a
        DOM mutation since the last parse still gets fresh coords.
        """
        # Inject the mapper only if it isn't already present, and wrap
        # in the nav-safe helper so a mid-flight navigation doesn't crash.
        try:
            already = await self._safe_eval(
                page, "() => typeof window.__AliaxCore !== 'undefined'"
            )
        except Exception:
            already = False
        if not already:
            await self._safe_inject(page)
        return await self._safe_eval(
            page,
            "(id) => window.__AliaxCore.resolveElement(id)",
            element_id,
        )

    # ------------------------------------------------------------------
    # report_issue — the unified Gatekeeper escalation entrypoint
    # ------------------------------------------------------------------

    # User-facing rejection message returned to BOTH the AI (when it
    # emits {"action": "REPORT_ISSUE"} too eagerly) and the developer
    # (when they call aliax.report_issue() before the agent has actually
    # been stuck for 3 turns). Phrased as instructions the LLM can act
    # on — "try alternative actions" maps directly to the prompt's
    # Stamina Rule. The message is the AI's behavioural feedback signal.
    _REPORT_REJECT_MSG = (
        "REPORT_ISSUE rejected: insufficient stagnation evidence. The SDK "
        "has not yet observed (a) three consecutive parse_ui rounds with "
        "an identical page state, or (b) a repeating click sub-cycle "
        "across the same 2-4 elements (e.g. [A,B,A,B] or [A,B,C,A,B,C]). "
        "You must attempt at least 2 DISTINCT alternative actions on this "
        "screen before escalation is allowed — and 'distinct' means a "
        "different VERB (e.g. SCROLL or WAIT or HOVER), not the same verb "
        "on a different element_id (which is exactly the cycling pattern "
        "the Gatekeeper detects). Pull yourself together and try a "
        "different KIND of action first."
    )

    async def report_issue(
        self,
        page,
        *,
        reason: str,
        goal: Optional[str] = None,
        expected_outcome: Optional[str] = None,
        actual_outcome: Optional[str] = None,
        step: Optional[int] = None,
        thoughts: Optional[str] = None,
        last_attempted_action: Optional[AttemptedActionLike] = None,
        force: bool = False,
    ) -> dict:
        """The single unified escalation entrypoint.

        Routes through a strict Gatekeeper that REJECTS the call unless
        the SDK's per-page failure history proves the agent has actually
        been stuck (3-tick state stagnation, or a repeating action
        sub-cycle of period 2/3/4 — e.g. radio-group loops).
        A rejected call does NOT contact the Aliax API, does NOT debit
        credits, and does NOT bloat the annotation queue with "slow-
        network panic" false positives. Instead it returns a structured
        rejection payload the caller (orchestrator or LLM) can read and
        retry from.

        Same function is exposed in three places:

          1. The Python developer's manual ``try/except`` catch.
          2. The AI's ``{"action": "REPORT_ISSUE"}`` JSON tool.
          3. The SDK's own internal escalators (future use).

        Pass ``force=True`` ONLY from a developer's hard assertion catch
        (Playwright timeout, business-logic invariant violation) where
        the developer knows for certain something is wrong and the
        Gatekeeper's heuristic-based evidence is irrelevant. The AI
        ``REPORT_ISSUE`` action can never set this flag.

        .. warning::
           Stagnation evidence is accumulated by :meth:`parse_ui` (which
           hashes the current ``url + spatial_map`` into the per-page
           state ring) and by :meth:`execute` (which pushes the targeted
           ``element_id`` into the per-page action ring). If you drive
           the SDK with a custom loop that only calls ``execute()`` —
           never refreshing the page model with a fresh ``parse_ui()``
           on each turn — the state ring will never advance, the
           Gatekeeper will never see stagnation proof, and every
           non-``force`` ``report_issue()`` call will be rejected. The
           reference orchestrator always parses at the top of each
           ReAct iteration; mirror that contract in any consumer.
        """
        eligible, gate_reason = self._failure_proof(page)
        if not eligible and not force:
            log.debug(
                "Aliax: report_issue rejected (no stagnation evidence) — "
                "state_hashes=%d, actions=%d",
                len(self._state_history_by_page.get(page, []) or []),
                len(self._action_history_by_page.get(page, []) or []),
            )
            return {
                "status": "rejected",
                "ok": False,
                "capture_id": None,
                "msg": self._REPORT_REJECT_MSG,
                "message": self._REPORT_REJECT_MSG,
                "gatekeeper": "rejected",
                "evidence": {
                    "state_history_len": len(self._state_history_by_page.get(page, []) or []),
                    "action_history_len": len(self._action_history_by_page.get(page, []) or []),
                },
            }

        # Build the structured context the dashboard renders alongside
        # the screenshot — gives annotators the AI's expectation-vs-
        # reality so they can label the corrective action without
        # having to reverse-engineer what the agent was thinking.
        ctx_payload: dict = {
            "trigger": "developer_assert_force" if force else f"gatekeeper:{gate_reason or 'forced'}",
            "reason": reason,
        }
        if expected_outcome:
            ctx_payload["expected_outcome"] = str(expected_outcome)[:500]
        if actual_outcome:
            ctx_payload["actual_outcome"] = str(actual_outcome)[:500]

        upload_goal = (goal or reason or "Agent escalation").strip() or "Agent escalation"

        result = await self.capture_failure(
            page,
            goal=upload_goal,
            thoughts=thoughts,
            last_attempted_action=last_attempted_action,
            failure_reason=(reason if reason else (gate_reason or "agent_stuck")),
            step=step,
            context=ctx_payload,
        )

        # A successful upload "consumes" the stagnation proof — wipe the
        # rings so the next legitimate failure on the same page must
        # rebuild evidence before it can escalate again. (Without this,
        # one frozen popup that lingers across 5 turns would dump 5
        # near-identical /v1/capture rows into the annotation queue.)
        if isinstance(result, Mapping) and result.get("status") == "success":
            self._reset_failure_history(page)

        if isinstance(result, dict):
            result.setdefault("gatekeeper", gate_reason or ("forced" if force else "allowed"))
            result["ok"] = (result.get("status") == "success")
        return result

    # ------------------------------------------------------------------
    # capture_failure — the annotation safety net
    # ------------------------------------------------------------------

    async def capture_failure(

        self,
        page,
        *,
        goal: str,
        thoughts: Optional[str] = None,
        last_attempted_action: Optional[AttemptedActionLike] = None,
        failure_reason: Optional[str] = None,
        step: Optional[int] = None,
        context: Optional[Union[str, Mapping[str, Any]]] = None,
    ) -> dict:
        """When the agent loops or hallucinates past the point of recovery,
        snapshot the full state and ship it to the Aliax annotation queue.

        This is the *escape hatch* — the interceptor (`parse_ui` +
        `execute`) handles the 95% happy path. Failures land here and get
        labeled offline by the VelocityEarn workforce.

        Behaviour matches the v0.2 ``capture()`` API exactly so existing
        integrations keep working. The endpoint hit is ``POST /v1/capture``.
        """
        # 1. Inject mapper (nav-safe). The previous action may still be
        #    settling — _safe_inject waits + retries on context destroyed.
        with suppress(Exception):
            await page.wait_for_load_state("domcontentloaded", timeout=10_000)
        await self._safe_inject(page)

        selectors_json = json.dumps(self.redact_selectors)
        blurred = False
        # The annotation pipeline accepts WebP / PNG / JPEG (worker
        # magic-byte validator covers all three). With Pillow dropped
        # from the SDK we go straight to JPEG at q=85 — visually
        # indistinguishable from lossless to a human annotator and
        # ~5-10× smaller than PNG for UI screenshots dominated by flat
        # colors. Encoding happens inside Chromium's native C++ path.
        shot_bytes = b""
        spatial_map: list = []
        vp_meta: dict = {}
        try:
            await self._safe_eval(
                page,
                f"() => window.__AliaxCore.blurPII({selectors_json})",
            )
            blurred = True

            # 2. Native JPEG screenshot (no Pillow re-encode).
            shot_bytes = await page.screenshot(type="jpeg", quality=85)

            # 3. Spatial map + viewport. capture_failure is the ONLY
            #    place we pass redact_text_for_pii=true — the visual
            #    blur above hides pixels in the screenshot, but the
            #    spatial_map JSON gets persisted forever in Postgres
            #    and is what human annotators read. Without this flag,
            #    the email/CC/password value the agent typed would
            #    leak in cleartext into the annotation database even
            #    though the screenshot looks redacted. parse_ui leaves
            #    this OFF (the live agent loop needs the real text to
            #    reason about "is the field already filled?").
            map_opts = {
                "redact_text_for_pii": True,
                "pii_selectors": list(self.redact_selectors or []),
                "__t": self._next_mapper_ticket(),
            }
            dom_result = await self._safe_eval(
                page, "(opts) => window.__AliaxCore.mapDOM(opts)", arg=map_opts
            )

            if isinstance(dom_result, dict) and "elements" in dom_result:
                spatial_map = dom_result.get("elements") or []
                vp_meta = dom_result.get("viewport") or {}
            else:
                spatial_map = dom_result or []
                vp_meta = {}

        finally:
            # 4. Unblur — guaranteed, even on screenshot/mapDOM error.
            #    Use _safe_eval so we don't crash on context destroyed.
            if blurred:
                try:
                    await self._safe_eval(
                        page,
                        "() => { if (window.__AliaxCore) window.__AliaxCore.unblurPII(); }",
                    )
                except Exception as exc:
                    log.warning("Aliax: unblurPII failed (%s)", exc)

        try:
            dpr = float(vp_meta.get("dpr") or 1)
            if not (dpr > 0):
                dpr = 1.0
        except (TypeError, ValueError):
            dpr = 1.0

        # Prefer the actual screenshot dimensions (parsed from the JPEG
        # header — tiny hand-rolled scanner so we don't pull Pillow back
        # in) divided by dpr — that's the authoritative CSS-pixel
        # viewport for overlay coordinate math. Fall back to
        # page.viewport_size, then to the documented 1280×800 default.
        actual_image_size = _image_dimensions(shot_bytes, "jpeg")
        viewport = page.viewport_size
        if actual_image_size and dpr > 0:
            width = int(actual_image_size[0] / dpr)
            height = int(actual_image_size[1] / dpr)
        elif viewport:
            width = viewport["width"]
            height = viewport["height"]
        else:
            width = 1280
            height = 800


        if isinstance(context, Mapping):
            context_str = json.dumps(context)
        elif context is None:
            context_str = ""
        else:
            context_str = str(context)

        # Route Context for the dashboard — lets the failure list be
        # grouped by `path` ("Your agent has a 99% success rate on
        # /dashboard, 45% on /checkout/pay"). Title is a secondary
        # anchor for multi-tenant routes where the same pathname renders
        # different UIs per workspace. All three are best-effort; a
        # closed-mid-call page just leaves them blank.
        cap_url = ""
        try:
            cap_url = page.url if isinstance(page.url, str) else ""
        except Exception:
            cap_url = ""
        cap_path = ""
        if cap_url:
            try:
                cap_path = urlparse(cap_url).path or ""
            except Exception:
                cap_path = ""
        cap_title = ""
        try:
            t = await page.title()
            if isinstance(t, str):
                cap_title = t.strip()[:200]
        except Exception:
            cap_title = ""

        # NB: api_key intentionally NOT in the body — it's sent via the
        # Authorization header on the shared httpx client.
        payload_data: dict = {
            "goal": goal,
            "context": context_str,
            "viewport": json.dumps({"width": width, "height": height, "dpr": dpr}),
            "spatial_map": json.dumps(spatial_map),
            "sdk_version": self.sdk_version,
        }
        if cap_url:
            payload_data["page_url"] = cap_url[:500]
        if cap_path:
            payload_data["page_path"] = cap_path[:200]
        if cap_title:
            payload_data["page_title"] = cap_title
        if thoughts:
            payload_data["ai_thoughts"] = str(thoughts)
        action_dict = _coerce_action(last_attempted_action)
        if action_dict:
            payload_data["last_attempted_action"] = json.dumps(action_dict)
        if failure_reason:
            payload_data["failure_reason"] = str(failure_reason)
        if step is not None:
            payload_data["step"] = str(int(step))

        if self.debug_mode:
            # Persist debug output to tempdir so this works in containers
            # with read-only CWD / on Windows with restricted paths.
            import tempfile, uuid as _uuid
            debug_dir = Path(tempfile.gettempdir())
            shot_path = debug_dir / "aliax_debug_screenshot.jpg"
            payload_path = debug_dir / "aliax_debug_payload.json"
            try:
                shot_path.write_bytes(shot_bytes)
            except Exception as e:
                log.warning("Aliax: failed writing debug screenshot to %s: %s", shot_path, e)
            debug_payload = {
                **payload_data,
                "spatial_map": spatial_map,
                "viewport": {"width": width, "height": height, "dpr": dpr},
                "last_attempted_action": action_dict,
                "screenshot_url": str(shot_path),
            }
            try:
                payload_path.write_text(
                    json.dumps(debug_payload, indent=2), encoding="utf-8"
                )
            except Exception as e:
                log.warning("Aliax: failed writing debug payload: %s", e)
            return {
                "status": "success",
                "capture_id": f"debug_capture_{_uuid.uuid4().hex[:8]}",
                "screenshot_url": str(shot_path),
                "debug_payload_path": str(payload_path),
                "msg": f"Debug mode: wrote {payload_path.name} + {shot_path.name} in {debug_dir}",
            }

        # Generate a client-side idempotency UUID up-front so a retry after
        # a transient 502/504 doesn't create duplicate capture rows. The
        # Worker stores this as captures.id (PK) and short-circuits on
        # conflict after verifying user ownership.
        idempotency_key = str(_uuid_module.uuid4())

        client = await self._client()
        files = {"screenshot": ("screenshot.jpg", shot_bytes, "image/jpeg")}
        last_status: Optional[int] = None
        last_err: Optional[str] = None
        for attempt in range(3):
            try:
                # Authorization header carries the bearer; the form body
                # MUST NOT echo the key — reverse-proxies / CDN access
                # logs / Worker request-body logs would otherwise capture
                # plaintext credentials on every upload.
                response = await client.post(
                    self.endpoint_capture,
                    data=payload_data,
                    files=files,
                    headers={"X-Idempotency-Key": idempotency_key},
                    timeout=15.0,
                )
                last_status = response.status_code
                if response.status_code in (200, 201):
                    res_data = response.json()
                    is_replay = bool(res_data.get("idempotent_replay"))
                    # Only emit a telemetry event on the FIRST acceptance.
                    # Idempotent replays must not double-count usage.
                    if not is_replay:
                        self._telemetry_in_background(
                            "capture_failure",
                            units=1,
                            meta={
                                "goal": goal[:120],
                                "failure_reason": failure_reason or "",
                                "path": cap_path[:120],
                                "title": cap_title[:120],
                            },
                        )
                    return {
                        "status": "success",
                        "capture_id": res_data.get("capture_id"),
                        "screenshot_url": res_data.get("screenshot_url"),
                        "idempotent_replay": is_replay,
                        "msg": "Failure snapshot queued for annotation.",
                    }
                # Retry only on transient server / rate-limit errors.
                if response.status_code in (429, 502, 503, 504) and attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                # (audit SDK #10) Surface 401/402 through the billing
                # state machine so the kill-switch and billing_status()
                # stay consistent — capture_failure should never silently
                # swallow a credit / key error.
                if response.status_code in (401, 402):
                    self._absorb_billing_response(
                        response, event_type="capture_failure"
                    )
                return {
                    "status": "error",
                    "capture_id": None,
                    "msg": f"Failed with status code: {response.status_code}",
                }
            except (httpx.ConnectError, httpx.TimeoutException, httpx.RemoteProtocolError) as e:
                last_err = type(e).__name__
                # First DNS / connect failure on the primary endpoint? Fail
                # over to the workers.dev origin and try again immediately
                # without burning a retry attempt.
                if isinstance(e, httpx.ConnectError) and self._maybe_swap_to_fallback():
                    continue
                if attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                # Sanitized — never echo the raw exception (it may embed
                # the request URL / headers including the bearer token).
                log.debug("capture_failure transport error: %s", e)
                return {
                    "status": "error",
                    "capture_id": None,
                    "msg": f"Network error contacting Aliax API ({last_err}).",
                }
            except Exception as e:
                log.debug("capture_failure unexpected error: %s", e)
                return {
                    "status": "error",
                    "capture_id": None,
                    "msg": "Unexpected error contacting Aliax API.",
                }
        return {
            "status": "error",
            "capture_id": None,
            "msg": f"Failed after retries (status={last_status}, err={last_err}).",
        }


    # ------------------------------------------------------------------
    # Back-compat alias
    # ------------------------------------------------------------------

    async def capture(self, page, goal: str, **kwargs) -> dict:
        """Deprecated v0.2 alias for :meth:`capture_failure`.

        The interceptor model (``parse_ui`` + ``execute``) is the new
        default; ``capture`` now means "treat this as a failure event".
        Existing integrations keep working unchanged.
        """
        # Emit at WARNING level + raise a real DeprecationWarning so the
        # signal is visible in production log aggregators AND surfaces
        # in `python -W error::DeprecationWarning` CI runs. The previous
        # log.debug call was silent unless the consumer had explicitly
        # opted into DEBUG-level logging, which nobody does in prod —
        # meaning a deprecated call could ship for months unnoticed.
        import warnings
        warnings.warn(
            "aliax.capture() is a v0.2 alias for capture_failure() and will "
            "be removed in a future major release. Migrate to "
            "aliax.parse_ui() + aliax.execute() for the live interceptor flow, "
            "or call aliax.capture_failure() / aliax.report_issue() directly.",
            DeprecationWarning,
            stacklevel=2,
        )
        log.warning(
            "aliax.capture() is a v0.2 alias and now routes to capture_failure() — "
            "consider migrating to parse_ui()/execute() for the live interceptor flow."
        )
        return await self.capture_failure(page, goal=goal, **kwargs)

    # ------------------------------------------------------------------
    # Internal: DNA-stamp execution (Absolute Targeting Engine)
    # ------------------------------------------------------------------

    def _cached_record(self, page, element_id: Optional[str]) -> Optional[dict]:
        """Return the full cached node record for ``element_id`` (or None).

        Used by the locator execution path to (a) detect canvases (which
        skip the locator tier entirely) and (b) feed semantic hints
        ``{tag, text, bounds}`` into the SPA-wipe rebind fallback so we
        can re-stamp the new node React just remounted.
        """
        if not element_id:
            return None
        try:
            cached = self._last_map_by_page.get(page)
        except TypeError:
            cached = None
        if not cached:
            return None
        for el in cached:
            if el.get("element_id") == element_id:
                return el
        return None

    async def _probe_disabled(
        self,
        page,
        stamp: Optional[str],
        cached_record: Optional[dict],
    ) -> Optional[str]:
        """Return a short reason string if the target is disabled, else None.

        Order:
          1. Trust the cached node map first — zero round-trips, zero
             latency. parse_ui() ran moments ago so the state is fresh
             enough for the very next execute().
          2. If the cache says "not disabled" (or has no record), do a
             tiny live probe through the DNA stamp. SPA frameworks often
             toggle disabled mid-think (validation re-runs on every
             keystroke), so a stale "enabled" cache could lure us into
             clicking a freshly-locked Submit.

        We never raise here — a probe failure (stamp wiped, navigation
        race) just returns None so the locator path takes over and its
        own three-tier fallback handles the situation.
        """
        # 1. Cache check — cheap and conservative.
        if cached_record:
            st = cached_record.get("state") or {}
            if st.get("disabled"):
                hints = []
                if st.get("required"):
                    hints.append("required field empty")
                if st.get("invalid"):
                    hints.append("invalid input flagged")
                if st.get("busy"):
                    hints.append("element is busy")
                why = ", ".join(hints) if hints else "aria-disabled / disabled attribute"
                return why

        # 2. Live probe — only when we have a stamp to target. Tight
        #    timeout so a wedged page can't stall the agent loop here.
        if not stamp:
            return None
        try:
            safe_stamp = str(stamp).replace('"', '\\"')
            probe = await self._safe_eval(
                page,
                "(s) => {"
                "  const el = document.querySelector('[data-aliax-id=\"' + s + '\"]');"
                "  if (!el) return null;"
                "  const ad = el.getAttribute && el.getAttribute('aria-disabled') === 'true';"
                "  const fs = el.closest && !!el.closest('fieldset[disabled]');"
                "  const cls = el.classList && (el.classList.contains('disabled') || el.classList.contains('is-disabled'));"
                "  const native = el.disabled === true;"
                "  if (native) return 'native disabled attribute';"
                "  if (ad) return 'aria-disabled=true';"
                "  if (fs) return 'inside <fieldset disabled>';"
                "  if (cls) return 'has .disabled class';"
                "  return null;"
                "}",
                safe_stamp,
            )
            if isinstance(probe, str) and probe:
                return probe
        except Exception as exc:
            log.debug("Aliax: disabled probe skipped (%s)", exc)
        return None

    async def _set_plain_editable_value(
        self,
        page,
        value: str,
        stamp: Optional[str] = None,
    ) -> dict:
        """Reliably set a focused/stamped <input>/<textarea> and sync React state.

        Returns a structured diagnostic dict so callers (and the
        console) can see EXACTLY what happened: which candidate node
        was resolved, what its value was before/after, which write
        strategy succeeded, and whether the post-write read-back
        matched. The dict always contains at minimum::

            {"ok": bool, "plain": bool, "reason": str | None}

        ``ok=False`` means the caller should fall through to the
        keyboard fallback. Never raises — defensive by design.
        """
        try:
            probe = await self._safe_eval(
                page,
                """
                async ({s, v}) => {
                  const diag = { plain: false, ok: false, reason: null, stamp: s, requested: String(v) };
                  const target = s ? document.querySelector('[data-aliax-id="' + String(s) + '"]') : null;
                  diag.stampResolved = !!target;
                  if (target) {
                    diag.stampTag = target.tagName;
                    diag.stampDisabled = !!target.disabled;
                    diag.stampReadOnly = !!target.readOnly;
                  }
                  const badInputTypes = new Set(['button','submit','reset','image','checkbox','radio','file','hidden']);
                  const isPlain = (el) => {
                    if (!el || !el.tagName || el.isContentEditable) return false;
                    const tag = el.tagName.toUpperCase();
                    if (tag !== 'INPUT' && tag !== 'TEXTAREA') return false;
                    if (el.disabled || el.readOnly) return false;
                    if (tag === 'INPUT' && badInputTypes.has(String(el.type || '').toLowerCase())) return false;
                    return true;
                  };
                  const deepActive = () => {
                    let a = document.activeElement;
                    while (a && a.shadowRoot && a.shadowRoot.activeElement) a = a.shadowRoot.activeElement;
                    return a;
                  };
                  const active = deepActive();
                  diag.activeTag = active && active.tagName;
                  diag.activeIsBody = !!(active && active.tagName === 'BODY');
                  const candidates = [];
                  const sources = [];
                  const add = (el, src) => { if (isPlain(el) && !candidates.includes(el)) { candidates.push(el); sources.push(src); } };
                  add(target, 'stamp');
                  try { if (target && target.control) add(target.control, 'label.control'); } catch (_) {}
                  try {
                    if (target && target.getAttribute) {
                      const id = target.getAttribute('for') || target.getAttribute('aria-controls');
                      if (id) add(document.getElementById(id), 'for/aria-controls');
                    }
                  } catch (_) {}
                  try { if (target && target.querySelector) add(target.querySelector('input:not([type="hidden"]), textarea'), 'descendant'); } catch (_) {}
                  try {
                    if (isPlain(active) && (!target || active === target || target.contains(active) || (target.control && active === target.control))) add(active, 'activeElement');
                  } catch (_) { if (!target) add(active, 'activeElement'); }
                  const el = candidates[0];
                  diag.candidateSource = sources[0] || null;
                  diag.candidatesTried = sources;
                  if (!el) { diag.reason = 'no editable candidate'; return diag; }

                  diag.plain = true;
                  diag.elTag = el.tagName;
                  diag.elType = el.type || null;
                  diag.valueBefore = String(el.value);

                  const fireInput = (type, init) => {
                    try { el.dispatchEvent(new InputEvent(type, Object.assign({ bubbles: true, composed: true }, init || {}))); }
                    catch (_) { el.dispatchEvent(new Event(type, { bubbles: true, composed: true })); }
                  };
                  const fireKey = (type) => {
                    try { el.dispatchEvent(new KeyboardEvent(type, { bubbles: true, composed: true, key: 'Unidentified' })); } catch (_) {}
                  };
                  const nativeSet = (val) => {
                    const proto = Object.getPrototypeOf(el);
                    const protoSetter = proto && Object.getOwnPropertyDescriptor(proto, 'value') && Object.getOwnPropertyDescriptor(proto, 'value').set;
                    const ownSetter = Object.getOwnPropertyDescriptor(el, 'value') && Object.getOwnPropertyDescriptor(el, 'value').set;
                    if (protoSetter && protoSetter !== ownSetter) protoSetter.call(el, val);
                    else if (ownSetter) ownSetter.call(el, val);
                    else el.value = val;
                  };
                  const settle = () => new Promise((resolve) => {
                    const done = () => setTimeout(resolve, 35);
                    if (typeof requestAnimationFrame === 'function') requestAnimationFrame(() => requestAnimationFrame(done));
                    else done();
                  });

                  try { el.focus({ preventScroll: true }); } catch (_) { try { el.focus(); } catch (_) {} }
                  diag.focusedAfterFocus = (deepActive() === el);
                  try { if (typeof el.select === 'function') el.select(); } catch (_) {}

                  let method = 'native-setter';
                  try {
                    nativeSet('');
                    fireInput('input', { inputType: 'deleteContentBackward', data: null });
                    try { if (typeof el.select === 'function') el.select(); } catch (_) {}
                    if (document.execCommand) {
                      const usedExec = document.execCommand('insertText', false, String(v));
                      if (usedExec || String(el.value) === String(v)) method = 'execCommand';
                    }
                  } catch (_) {}

                  if (String(el.value) !== String(v)) {
                    fireKey('keydown');
                    fireKey('keypress');
                    fireInput('beforeinput', { cancelable: true, inputType: 'insertReplacementText', data: String(v) });
                    nativeSet(String(v));
                    fireInput('input', { inputType: 'insertReplacementText', data: String(v) });
                    fireKey('keyup');
                  }
                  el.dispatchEvent(new Event('change', { bubbles: true, composed: true }));
                  try { el.focus({ preventScroll: true }); } catch (_) { try { el.focus(); } catch (_) {} }
                  await settle();
                  diag.valueAfter = String(el.value);
                  diag.method = method;
                  diag.ok = String(el.value) === String(v);
                  if (!diag.ok) diag.reason = 'value mismatch after write';
                  return diag;
                }
                """,
                {"s": str(stamp) if stamp else None, "v": value},
            )
            if not isinstance(probe, dict):
                self._dbg("set-editable.bad-probe", stamp=stamp, probe=str(probe)[:200])
                return {"ok": False, "plain": False, "reason": "probe returned non-dict"}
            self._dbg(
                "set-editable",
                stamp=stamp,
                stampResolved=probe.get("stampResolved"),
                candidateSource=probe.get("candidateSource"),
                elTag=probe.get("elTag"),
                elType=probe.get("elType"),
                activeTag=probe.get("activeTag"),
                focusedAfterFocus=probe.get("focusedAfterFocus"),
                valueBefore=probe.get("valueBefore"),
                valueAfter=probe.get("valueAfter"),
                method=probe.get("method"),
                ok=probe.get("ok"),
                reason=probe.get("reason"),
            )
            return probe
        except Exception as exc:
            self._dbg("set-editable.exception", stamp=stamp, error=str(exc))
            return {"ok": False, "plain": False, "reason": f"exception: {exc}"}

    async def _verify_typed_value(self, page, stamp: Optional[str]) -> Optional[str]:
        """Read back the live value of the stamped/active editable
        node after a keyboard-fallback ``type()`` so the orchestrator
        can confirm the keystrokes actually landed. Returns the
        string value or ``None`` if we cannot read it."""
        try:
            val = await self._safe_eval(
                page,
                """
                ({s}) => {
                  const tgt = s ? document.querySelector('[data-aliax-id="' + String(s) + '"]') : null;
                  const pick = (el) => {
                    if (!el) return null;
                    const t = (el.tagName || '').toUpperCase();
                    if (t === 'INPUT' || t === 'TEXTAREA') return String(el.value);
                    if (el.isContentEditable) return String(el.textContent || '');
                    return null;
                  };
                  let v = pick(tgt);
                  if (v !== null) return v;
                  if (tgt) {
                    const inner = tgt.querySelector && tgt.querySelector('input:not([type="hidden"]), textarea');
                    v = pick(inner);
                    if (v !== null) return v;
                  }
                  let a = document.activeElement;
                  while (a && a.shadowRoot && a.shadowRoot.activeElement) a = a.shadowRoot.activeElement;
                  return pick(a);
                }
                """,
                {"s": str(stamp) if stamp else None},
            )
            return val if isinstance(val, str) else None
        except Exception as exc:
            self._dbg("verify.exception", stamp=stamp, error=str(exc))
            return None



    async def _act_via_locator(
        self,
        page,
        stamp: str,
        action: str,
        decision: Mapping[str, Any],
        cached_record: Optional[dict],
        type_delay_ms: int,
    ) -> tuple:
        """Drive a CLICK/HOVER/TYPE/PRESS through a Playwright Locator
        bound to the DNA stamp. Returns ``(ok, extra)``.

        ``extra`` includes ``execution_tier`` (``"locator"`` |
        ``"jsclick"`` | ``"rebind"``) and ``coords`` from the live
        bounding box at action time — populated for the dashboard
        replay's bullseye animation even though the click itself was
        node-driven. We deliberately swallow recoverable failures and
        return ``(False, {})`` so the main ``execute()`` falls through
        to the coordinate fallback rather than crashing the agent loop.
        """
        from contextlib import suppress as _suppress

        # CSS attribute selectors don't accept arbitrary strings safely
        # — escape any double quotes in the (numeric) stamp just in case
        # a custom mapper override fed in something exotic.
        safe_stamp = str(stamp).replace('"', '\\"')
        sel = f'[data-aliax-id="{safe_stamp}"]'
        # .first because some SPAs render duplicate nodes briefly during
        # transition animations — we always want the first hit.
        locator = page.locator(sel).first

        async def _post_act_coords() -> Optional[list]:
            """Pull the LIVE bounding box from the locator post-action.
            Used purely for dashboard-replay telemetry — execution
            itself was driven by the stamp, not these coords."""
            try:
                box = await locator.bounding_box(timeout=500)
                if box:
                    cx = int(box["x"] + box["width"] / 2)
                    cy = int(box["y"] + box["height"] / 2)
                    return [cx, cy]
            except Exception:
                pass
            if cached_record:
                b = cached_record.get("bounds") or {}
                return [
                    int(float(b.get("x", 0)) + float(b.get("width", 0)) / 2),
                    int(float(b.get("y", 0)) + float(b.get("height", 0)) / 2),
                ]
            return None

        async def _do_action(tier: str) -> tuple:
            extra: dict = {"execution_tier": tier}
            if action == "HOVER":
                self._dbg("locator.HOVER.begin", tier=tier, stamp=stamp)
                await locator.hover(timeout=4_000)
                self._dbg("locator.HOVER.done", tier=tier, stamp=stamp)
            elif action == "CLICK":
                self._dbg("locator.CLICK.begin", tier=tier, stamp=stamp)
                if tier == "jsclick":
                    # Nightmare 4 bypass — fire .click() at the node
                    # level, ignoring Playwright's actionability check
                    # so an invisible-shield overlay can't hijack us.
                    await locator.evaluate("n => n.click()")
                else:
                    # Auto-waits for attached + visible + stable + enabled
                    # + hit-test-receives-events. ~10s default budget.
                    await locator.click(timeout=8_000)
                self._dbg("locator.CLICK.done", tier=tier, stamp=stamp)

            elif action == "TYPE":
                value = _decision_text_value(decision)
                self._dbg("locator.TYPE.begin", tier=tier, stamp=stamp, value=value)
                with _suppress(Exception):
                    await locator.scroll_into_view_if_needed(timeout=2_000)
                react_done = await self._set_plain_editable_value(page, value, stamp)
                if not (react_done and react_done.get("ok")):
                    self._dbg("locator.TYPE.retry-after-focus", tier=tier, stamp=stamp, reason=react_done.get("reason") if isinstance(react_done, dict) else None)
                    if tier == "jsclick":
                        await locator.evaluate("n => n.focus && n.focus()")
                    else:
                        await locator.click(timeout=8_000)
                    react_done = await self._set_plain_editable_value(page, value, stamp)
                if not (react_done and react_done.get("ok")):
                    self._dbg("locator.TYPE.keyboard-fallback", tier=tier, stamp=stamp)
                    with _suppress(Exception):
                        await page.keyboard.press("ControlOrMeta+A")
                        await page.keyboard.press("Backspace")
                    await page.keyboard.type(value, delay=type_delay_ms)
                    final_val = await self._verify_typed_value(page, stamp)
                    self._dbg("locator.TYPE.keyboard-verify", stamp=stamp, expected=value, got=final_val, ok=(final_val == value))
                else:
                    self._dbg("locator.TYPE.done", tier=tier, stamp=stamp, method=react_done.get("method"), valueAfter=react_done.get("valueAfter"))
                extra["value"] = value
            elif action == "TYPE_AND_ENTER":
                value = _decision_text_value(decision)
                self._dbg("locator.TYPE_AND_ENTER.begin", tier=tier, stamp=stamp, value=value)
                with _suppress(Exception):
                    await locator.scroll_into_view_if_needed(timeout=2_000)
                react_done = await self._set_plain_editable_value(page, value, stamp)
                if not (react_done and react_done.get("ok")):
                    self._dbg("locator.TYPE_AND_ENTER.retry-after-focus", tier=tier, stamp=stamp, reason=react_done.get("reason") if isinstance(react_done, dict) else None)
                    if tier == "jsclick":
                        await locator.evaluate("n => n.focus && n.focus()")
                    else:
                        await locator.click(timeout=8_000)
                    react_done = await self._set_plain_editable_value(page, value, stamp)
                if not (react_done and react_done.get("ok")):
                    self._dbg("locator.TYPE_AND_ENTER.keyboard-fallback", tier=tier, stamp=stamp)
                    with _suppress(Exception):
                        await page.keyboard.press("ControlOrMeta+A")
                        await page.keyboard.press("Backspace")
                    await page.keyboard.type(value, delay=type_delay_ms)
                    final_val = await self._verify_typed_value(page, stamp)
                    self._dbg("locator.TYPE_AND_ENTER.keyboard-verify", stamp=stamp, expected=value, got=final_val, ok=(final_val == value))
                else:
                    self._dbg("locator.TYPE_AND_ENTER.done", tier=tier, stamp=stamp, method=react_done.get("method"), valueAfter=react_done.get("valueAfter"))
                await page.keyboard.press("Enter")
                extra["value"] = value
                extra["key"] = "Enter"
            elif action == "PRESS":
                key = _decision_key_value(decision)
                self._dbg("locator.PRESS.begin", tier=tier, stamp=stamp, key=key)
                if tier == "jsclick":
                    await locator.evaluate("n => n.focus && n.focus()")
                else:
                    await locator.click(timeout=8_000)
                await page.keyboard.press(key)
                self._dbg("locator.PRESS.done", tier=tier, stamp=stamp, key=key)
                extra["key"] = key

            coords = await _post_act_coords()
            if coords is not None:
                extra["coords"] = coords
            return True, extra

        # ---- Tier 1: native Playwright actionability ----
        try:
            await locator.wait_for(state="attached", timeout=1_500)
            return await _do_action("locator")
        except Exception as e1:
            # Tier 2: invisible-shield bypass (force-click via JS).
            if _is_intercepted(e1):
                try:
                    return await _do_action("jsclick")
                except Exception as e2:
                    log.debug("Aliax: jsclick bypass also failed: %s", e2)
                    # fall through to rebind / coord fallback

            # Tier 3: SPA DOM-wipe → semantic rebind.
            if _is_stamp_lost(e1) and cached_record is not None:
                hint = {
                    "element_id": cached_record.get("element_id"),
                    "tag": cached_record.get("tag"),
                    "text": cached_record.get("text"),
                    "bounds": cached_record.get("bounds") or {},
                }
                rebound = None
                try:
                    rebound = await self._safe_eval(
                        page,
                        "(h) => window.__AliaxCore && window.__AliaxCore.rebindStamp(h)",
                        hint,
                    )
                except Exception as e_rebind:
                    log.debug("Aliax: rebindStamp threw: %s", e_rebind)
                if rebound:
                    # Stamp has been physically re-applied to the new
                    # node — the same locator now resolves. Retry once.
                    try:
                        await locator.wait_for(state="attached", timeout=1_500)
                        ok, extra = await _do_action("rebind")
                        extra["rebound_from"] = rebound.get("rebound_from")
                        return ok, extra
                    except Exception as e3:
                        log.debug("Aliax: rebind retry failed: %s", e3)

            log.debug("Aliax: locator tier failed (%s): %s", type(e1).__name__, e1)
            return False, {}

    # ------------------------------------------------------------------
    # Internal: resolve coords for execute()
    # ------------------------------------------------------------------



    async def _coords_for(
        self,
        page,
        decision: Mapping[str, Any],
    ) -> Optional[dict]:
        """Resolve the decision into ``{x, y, element_id?}`` in CSS px."""
        # Explicit coords win — useful for raw-fallback agents.
        if decision.get("x") is not None and decision.get("y") is not None:
            return {"x": float(decision["x"]), "y": float(decision["y"])}

        element_id = decision.get("element_id") or decision.get("id")
        if not element_id:
            return None

        # Fast path: cached node map from the most recent parse_ui.
        cached: Optional[list] = None
        try:
            cached = self._last_map_by_page.get(page)
        except TypeError:
            cached = None
        if cached:
            for el in cached:
                if el.get("element_id") == element_id:
                    b = el.get("bounds") or {}
                    return {
                        "x": float(b.get("x", 0)) + float(b.get("width", 0)) / 2,
                        "y": float(b.get("y", 0)) + float(b.get("height", 0)) / 2,
                        "element_id": element_id,
                    }

        # Fallback: re-walk the live DOM via the mapper.
        resolved = await self.get_element_coords(page, element_id)
        if not resolved:
            return None
        return {
            "x": float(resolved["x"]),
            "y": float(resolved["y"]),
            "element_id": element_id,
        }
