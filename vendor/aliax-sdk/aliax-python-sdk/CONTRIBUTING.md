# Contributing to the Aliax Python SDK

Thanks for hacking on Aliax. This SDK ships to production agent
deployments — keep changes small, typed, and covered.

## Local setup

```bash
cd aliax-python-sdk
python -m venv .venv && source .venv/bin/activate
pip install -e ".[test]"
python -m pytest -v
```

The bundled `aliax/dom-mapper.min.js` is regenerated from
`../aliax-core-js/` — see that package's README for the JS build steps.

## Tests

- Unit tests live in `tests/` and run on every PR across Python 3.8–3.12.
- Network calls must be mocked (e.g. with `respx`). Tests must never hit
  the live Cloudflare Worker or require a real API key.
- New public surface needs at least one test in `tests/`.

## Releasing

See [`../RELEASING.md`](../RELEASING.md) for the cut-a-version checklist.
At minimum: bump `aliax/_version.py`, update `CHANGELOG.md`, tag, then
`python -m build && twine upload dist/*`.

## Code style

- Type hints on every public function.
- Keep dependencies minimal — the SDK ships with only `httpx` and
  `playwright`. Don't add Pillow / OpenCV / Cairo.
- Docstrings on public APIs; explain *why* not *what*.
