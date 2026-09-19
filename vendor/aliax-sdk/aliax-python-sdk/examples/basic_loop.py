"""Minimal Aliax agent loop — copy, paste, run.

Prerequisites:
    pip install aliax
    playwright install chromium
    export ALIAX_API_KEY=sk_live_...   # mint at https://aliax.xyz/api-keys

Run:
    python basic_loop.py
"""
import asyncio
import os

from aliax import Aliax
from playwright.async_api import async_playwright


async def fake_vlm(image_bytes: bytes, elements: list) -> dict:
    """Stand-in for your real VLM call.

    Real code would POST image_bytes + a JSON-stringified elements map to
    GPT-4o / Claude / Gemini and parse the returned action. Here we just
    click the first link-like element so the example is self-contained.
    """
    for el in elements:
        if el.get("tag") == "A":
            return {"action": "CLICK", "element_id": el["element_id"]}
    return {"action": "DONE"}


async def main() -> None:
    if not os.getenv("ALIAX_API_KEY"):
        raise SystemExit("Set ALIAX_API_KEY first — see header.")

    # `async with` ensures the httpx pool is torn down even if the agent
    # loop raises. `api_key` defaults to the ALIAX_API_KEY env var.
    async with Aliax() as aliax, async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        await page.goto("https://news.ycombinator.com")

        for step in range(3):
            # 1. Translate the live DOM into a VLM-friendly Set-of-Mark.
            ctx = await aliax.parse_ui(page)

            # 2. Ask your VLM.
            decision = await fake_vlm(ctx.image_bytes, ctx.elements)
            print(f"step {step}: {decision}")

            if decision["action"] == "DONE":
                break

            # 3. Aliax executes it natively — scroll, React onChange,
            #    iframe coords, retina DPR, all handled.
            await aliax.execute(page, decision)

        await browser.close()


if __name__ == "__main__":
    asyncio.run(main())
