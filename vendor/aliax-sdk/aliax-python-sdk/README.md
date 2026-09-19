### ⚡ See it work before you install — [run the live browser benchmark →](https://aliax.xyz/benchmark)

> Paste any URL. Watch Aliax shrink the page into a tiny, numbered map your model can actually read — side-by-side with the raw HTML, in seconds. No signup, no install.

---

# Aliax — Python SDK

[![PyPI version](https://img.shields.io/pypi/v/aliax.svg?style=flat-square&color=10B981)](https://pypi.org/project/aliax/)
[![Python versions](https://img.shields.io/pypi/pyversions/aliax.svg?style=flat-square&color=3B82F6)](https://pypi.org/project/aliax/)
[![CI](https://img.shields.io/github/actions/workflow/status/aliax/aliax-sdk/test.yml?branch=main&style=flat-square&label=tests)](https://github.com/aliax/aliax-sdk/actions/workflows/test.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-10B981.svg?style=flat-square)](LICENSE)

> 🚀 **No API Key Required to Start:** `Aliax()` works out of the box with zero signup. It automatically provisions a free machine-scoped anonymous sandbox credential (**500 free parses** on home/office networks, **100 free parses** in cloud environments) cached locally in `~/.aliax/credentials`. You can install and run immediately.
>
> 🧠 **Stop Writing Custom Prompts:** Do NOT write or manually craft VLM system prompts from scratch. Aliax already exports `SYSTEM_INSTRUCTIONS` (or `ALIAX_SYSTEM_INSTRUCTIONS`) — a battle-tested instruction prompt tuned precisely for Set-of-Mark visual navigation, deterministic action output, and the exact `execute()` schema (`CLICK`, `TYPE`, `HOVER`, `SCROLL`, `DONE`). Import it directly: `from aliax import SYSTEM_INSTRUCTIONS`.

**The production-grade reliability layer for VLM web agents.**

Stop your AI agents from guessing browser coordinates and failing on
responsive layouts. Aliax sits between your VLM and the live page,
translates the DOM into a numbered Set-of-Mark image, executes the
action the model picks deterministically, and ships every real-world
failure into a labelled dataset you can fine-tune on.

It is the layer that turns *"works in my notebook"* into *"holds up at
3am for paying corporate clients."*

Python: 3.8 – 3.12 · [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md)

## Install

```bash
pip install aliax
playwright install chromium
```

`aliax` is the install name AND the import — no aliases, no version
pinning required.

## Zero-Config Quickstart (No Key Needed)

You do **not** need an API key, account, or credit card to run your first agent loop:

```python
from aliax import Aliax, SYSTEM_INSTRUCTIONS
from playwright.async_api import async_playwright

# Zero setup: bootstraps free anonymous sandbox automatically
async with Aliax() as aliax:
    async with async_playwright() as p:
        page = await (await p.chromium.launch()).new_page()
        await page.goto("https://shop.example.com/cart")

        # 1. Translate the live page into a Set-of-Mark image + node map.
        ctx = await aliax.parse_ui(page)

        # 2. Hand both to your VLM. Drop SYSTEM_INSTRUCTIONS into your prompt.
        decision = await ask_llm(
            system=SYSTEM_INSTRUCTIONS,
            image=ctx.image_bytes,
            map=ctx.llm_text_block(),
        )

        # 3. Aliax executes it natively — scroll, click, type, iframe coords.
        await aliax.execute(page, decision)
```

## 🧠 Pre-Packaged System Instructions (Stop Writing Prompts Manually)

The single hardest part of an AI web agent is prompt engineering: get one word wrong and your VLM will hallucinate element IDs, output invalid schemas, click disabled buttons, or get stuck in repetitive loops.

**Aliax ships a battle-tested instruction string directly in the SDK — you do not need to invent your own:**

```python
from aliax import SYSTEM_INSTRUCTIONS  # or ALIAX_SYSTEM_INSTRUCTIONS
```

### Why you should import `SYSTEM_INSTRUCTIONS`:
1. **100% Schema Alignment:** Teaches the model the exact JSON action catalog supported by `aliax.execute()` (`CLICK`, `TYPE`, `TYPE_AND_ENTER`, `HOVER`, `SCROLL_DOWN/UP/LEFT/RIGHT`, `REPORT_ISSUE`, `DONE`).
2. **Zero-Hallucination Set-of-Mark Protocol:** Instructs Claude, GPT-4o, and Gemini on how to read numbered bounding boxes (`el_41`), inspect state flags (`DISABLED`, `REQUIRED`, `INVALID`, `BUSY`, `READONLY`), and calculate navigation paths using `links_to`.
3. **Built-in Gatekeeper Protocol:** Details the structured Expectation-vs-Reality JSON format required by `REPORT_ISSUE` and `capture_failure` when an agent is genuinely stuck.
4. **Invisible Version Upgrades:** When Aliax adds new actions or capabilities, simply updating the SDK package updates the prompt automatically — zero prompt-engineering churn on your end.

### Drop into your VLM call:

```python
# OpenAI (GPT-4o)
response = await openai_client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Goal: {user_goal}\n\nDOM Map:\n{ctx.llm_text_block()}"},
                {"type": "image_url", "image_url": {"url": f"data:{ctx.image_mime};base64,{ctx.image_base64}"}}
            ],
        },
    ],
    response_format={"type": "json_object"},
)
decision = json.loads(response.choices[0].message.content)
await aliax.execute(page, decision)
```

## Authentication Modes

Aliax supports three clear ways to authenticate, from instant zero-signup experimentation to locked-down enterprise deployments:

| Mode | Configuration | Allowance & Capabilities | Best For |
| :--- | :--- | :--- | :--- |
| **1. Free Anonymous Sandbox** *(Default)* | `Aliax()` (no key passed, no env var) | **500 parses** (home/office) or **100 parses** (cloud egress). 30-day life, cached in `~/.aliax/credentials`. Full `parse_ui` and `execute`. | Quickstarts, local development, evaluation without signup. |
| **2. Personal API Key** | `ALIAX_API_KEY="sk_live_..."` or `Aliax(api_key="sk_live_...")` | **1,000 free starter credits** + pay-as-you-go top-ups. Unlocks the [web dashboard](https://aliax.xyz/dashboard) and visual crash-capture uploads ([Flight Recorder](https://aliax.xyz/captures)). | Production agents, team collaboration, crash debugging. |
| **3. Strict Enterprise Opt-Out** | `Aliax(allow_anonymous=False)` or `ALIAX_DISABLE_ANONYMOUS=1` | **Fail-closed.** Throws immediately if no valid API key is supplied; never contacts the edge unauthenticated. | SOC2 environments, strict CI/CD runners, air-gapped systems. |

### Upgrading from Anonymous to a Personal API Key
When you're ready for the web dashboard, visual crash captures (`capture_failure`), or more credits:
1. Create a free account at [aliax.xyz/auth](https://aliax.xyz/auth) (includes 1,000 free credits).
2. Grab your key from [API Keys](https://aliax.xyz/api-keys).
3. Set `export ALIAX_API_KEY="sk_live_..."` in your environment or secret manager. Your code doesn't need to change.

That's the entire happy path. No XPath wrangling, no Playwright
locator boilerplate, no pixel guessing.

---

## Pillar 1 — The Steady Hand

Deterministic VLM-to-DOM translation. Your model never sees raw
coordinates; it sees `el_41` and Aliax handles the browser physics.

`parse_ui(page)` returns a `ParseContext`:

| Attribute     | Type         | Description |
|---------------|--------------|-------------|
| `image_bytes` | `bytes`      | Native Chromium screenshot (JPEG by default) with numbered Set-of-Mark boxes painted by the bundled JS overlay. Zero Python image processing — no Pillow, no OpenCV. |
| `image_mime`  | `str`        | `"image/jpeg"` by default, `"image/png"` if you opt in. |
| `image_size`  | `(int, int)` | `(width, height)` in physical pixels, parsed from the image header. |
| `elements`    | `list[dict]` | Each element: `element_id`, `tag`, `role`, `text`, `bounds`, `editable`, `is_canvas`, `state`, `attrs`, `links_to`. Only **truly interactable** nodes ≥12 CSS px. |
| `viewport`    | `dict`       | `{width, height, dpr, scroll_x, scroll_y, page_scrollable_x, page_scrollable_y}` in CSS px. |
| `url`         | `str`        | Page URL at capture time. |
| `path`        | `str`        | URL pathname only — e.g. `"/settings/billing"`. |
| `title`       | `str`        | `document.title` at capture time. |
| `truncated`   | `bool`       | `True` when the DOM mapper hit its element cap; `llm_text_block()` appends a notice so the VLM knows the map is partial. |

Convenience helpers: `ctx.llm_text_block()` (drop-in for your prompt)
and `ctx.route_context_block()` (URL + title summary).

### Execute — the full verb catalog

Every action `execute()` recognises:

```python
await aliax.execute(page, {"action": "CLICK",          "element_id": "el_41"})
await aliax.execute(page, {"action": "HOVER",          "element_id": "el_14"})
await aliax.execute(page, {"action": "TYPE",           "element_id": "el_7", "value": "Nike Shoes"})
await aliax.execute(page, {"action": "TYPE_AND_ENTER", "element_id": "el_7", "value": "Nike Shoes"})
await aliax.execute(page, {"action": "PRESS",          "element_id": "el_7", "key": "Enter"})
await aliax.execute(page, {"action": "SCROLL_DOWN",    "dy": 600})       # also: dx / delta_x / delta_y
await aliax.execute(page, {"action": "SCROLL_UP",      "dy": 600})
await aliax.execute(page, {"action": "SCROLL_LEFT",    "dx": 400})
await aliax.execute(page, {"action": "SCROLL_RIGHT",   "dx": 400})
await aliax.execute(page, {"action": "NAVIGATE",       "url": "https://..."})
await aliax.execute(page, {"action": "WAIT",           "ms": 1500})
await aliax.execute(page, {"action": "COMBO",          "actions": [...]})  # compound action pipeline
await aliax.execute(page, {"action": "BATCH_TYPE",     "fields": [...]})   # multi-field fill in one turn
await aliax.execute(page, {"action": "REPORT_ISSUE",   "reason": "stuck"}) # routes through the Gatekeeper
await aliax.execute(page, {"action": "NOOP"})                              # terminal alias
await aliax.execute(page, {"action": "DONE"})                              # `FINISH` is accepted as an alias
```

Raw coordinates are accepted as an escape hatch:

```python
await aliax.execute(page, {"action": "CLICK", "x": 905, "y": 150})
```

Under the hood, `execute()` scrolls the target into view, fires
React-friendly events (focus → keystrokes with a 50ms cadence),
refuses to click disabled controls, and **never raises** — your
agent loop stays alive.

### Token-aware rendering

VLM providers bill on file weight + dimensions. Drop to `quality=40`
for roughly **10× cheaper** VLM calls without sacrificing Set-of-Mark
ID legibility (the IDs are crisp DOM text rendered *before* the JPEG
encoder runs).

```python
ctx = await aliax.parse_ui(
    page,
    render_config={"format": "jpeg", "quality": 40},
)
```

Quality is clamped to `[30, 100]` so a typo like `quality=5` can't
turn the numbered boxes into illegible mush and crash your loop.

---

## Pillar 2 — The Flight Recorder

Every agent eventually loops on a popup or hallucinates past
recovery. Aliax ships two escalation entry points that double as
your agentic telemetry stream:

### `report_issue()` — the gated escalation

```python
await aliax.report_issue(
    page,
    reason="submit_button_dead_zone",
    expected_outcome="navigate to /dashboard",
    actual_outcome="still on /login after 3 tries",
)
```

A strict Gatekeeper rejects the call unless the SDK's per-page
failure history *proves* the agent is stuck:

- **3 identical state hashes** of `(url, spatial_map)` → frozen DOM, and
- **period-2 / 3 / 4 action cycle detection** → radio-group / checkbox
  alternation loops the state hash is blind to.

Rejected calls never hit the API, never debit credits, and never
flood the annotation queue with slow-network panic. Pass `force=True`
only from a developer-side assertion (Playwright timeout, business
invariant violation).

### `capture_failure()` — the airbag

```python
await aliax.capture_failure(
    page,
    goal="Close newsletter popup",
    thoughts=agent.current_reasoning,
    last_attempted_action={"action": "CLICK", "element_id": "el_41"},
    failure_reason="modal_blocked",
    step=agent.loop_step,
)
```

`POST /v1/capture` with idempotency UUID, 3 attempts and exponential
backoff on 429/502/503/504, automatic transport-error fallback. The
full state — screenshot, DOM map, viewport, the agent's own
reasoning trail — lands in the **Aliax Inbox** for human triage.

---

## Pillar 3 — The Closed Loop

Every captured failure becomes a `(negative, positive)` DPO pair:
the agent's wrong move plus the annotator's corrected tap. Export
the dataset and fine-tune; your next deployment fails less often on
the exact failure modes that bit you in production.

That's the loop:

```
parse_ui → VLM → execute → (95% happy path)
                            ↓ stuck
                       report_issue → Inbox → DPO dataset → fine-tune
```

---

## Architecture

1. The bundled encrypted DOM mapper (`dom-mapper.dat`, shipped inside
  the wheel) is opened only after the authenticated licensing session,
  then walks the live page including shadow DOMs and same-origin
   iframes, returning only interactable elements ≥12 CSS px.
2. The same JS module paints a `position:fixed; pointer-events:none;
   contain:strict` overlay of numbered boxes. The host page's
   layout / hover / IntersectionObserver state is untouched.
3. Playwright snaps **one** native screenshot via its C++ CDP path
   at the format / quality you asked for. No Python pixel processing.
4. The overlay is torn down in a `try/finally` so a mid-capture
   crash leaves the live page exactly as we found it.
5. Coordinates are cached so `execute(page, {"element_id": "el_41"})`
   resolves instantly against the most recent `parse_ui()`.
6. Every call fires a non-blocking telemetry ping for billing.

### Hardening for long-running agents

- **`async with Aliax() as aliax:`** — full async context manager
  support so the `httpx` connection pool is torn down deterministically
  even if the loop crashes. No socket leaks in 24/7 bots.
- **Zip-safe asset loading** — `dom-mapper.dat` is loaded via
  `importlib.resources`, so the SDK runs inside AWS Lambda layers
  and Cloud Run images where `__file__` is virtual.
- **PEP 561 typed** — ships `py.typed`; `mypy`, Pyright, VS Code and
  PyCharm consume the inline type hints out of the box.
- **Automatic `*.workers.dev` fallback** — if the apex DNS / TLS
  flakes, the SDK transparently retries against the worker origin.
  Opt out with `fallback_endpoint=""`.
- **Self-healing billing latches** — HTTP 402 sets
  `credits_exhausted` and clears on any successful 200. HTTP 401
  latches `invalid_key_reason` (terminal — mint a new key).

## Public surface

```python
from aliax import (
    Aliax,
    ParseContext,
    AttemptedAction,
    AliaxError,
    AliaxOutOfCreditsError,
    AliaxInvalidKeyError,
    ALIAX_SYSTEM_INSTRUCTIONS,  # primary — drop into your VLM system prompt
    SYSTEM_INSTRUCTIONS,        # short alias for the same string
)
```

`Aliax` constructor (all keyword-only, all optional):

| Kwarg               | Default                      | Notes |
|---------------------|------------------------------|-------|
| `api_key`           | `os.getenv("ALIAX_API_KEY")`, else automatic free anonymous sandbox key | Optional. Omit to run keyless with zero signup. Personal keys start with `sk_live_`. |
| `allow_anonymous`   | `True` | With no key, bootstrap a free machine-scoped sandbox credential (cached in `~/.aliax/credentials`): 500 parses on a home/office network, 100 from cloud egress, 30-day life, no crash captures. Set `False` (or `ALIAX_DISABLE_ANONYMOUS=1`) to hard-fail instead. |
| `endpoint`          | `https://api.aliax.xyz/v1`   | Override for self-hosted. |
| `fallback_endpoint` | workers.dev origin (auto)    | Pass `""` to disable. |
| `debug_mode`        | `False`                      | Writes payloads locally instead of POSTing. |
| `redact_selectors`  | `[]`                         | Merged with built-in PII baseline (passwords, emails, card numbers). |
| `check_for_updates` | `True`                       | Pings `/v1/version` once at init; pass `False` in airgapped envs. |
| `max_image_dim`     | `0`                          | **Deprecated — no-op since v1.0.** Use `render_config={"quality": N}` on `parse_ui` for VLM token economy. Kept in the signature so pre-1.0 callers keep importing; a one-shot deprecation log fires only when explicitly set. |

Methods: `parse_ui`, `execute`, `report_issue`, `capture_failure`,
`get_element_coords`, `billing_status`, `refresh_billing_status`,
`aclose`.

---

[Start Keyless (Free Sandbox)](https://aliax.xyz) · [Get a Personal API Key](https://aliax.xyz/auth) · [Documentation](https://aliax.xyz/docs) · [Capture Inbox](https://aliax.xyz/captures)
