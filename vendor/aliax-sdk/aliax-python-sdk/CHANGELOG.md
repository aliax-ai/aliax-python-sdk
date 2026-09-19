# Changelog

All notable changes to the Aliax Python SDK are documented here. This
project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.3] — 2026-09-13

Ships the encrypted `dom-mapper.dat` artifact, authenticated mapper sessions,
single-use execution tickets, and telemetry-backed ticket renewal.

## [1.0.2] — 2026-06-22

Version-parity release with the Node SDK (`aliax@1.0.2`). No behavioural
changes from `1.0.0`; bumped so `pip install aliax` and `npm i aliax` land
on matching majors/minors/patches, and so the `X-Aliax-SDK-Version` header
the Worker logs is identical across both runtimes.

### Changed
- `aliax.__version__` → `"1.0.2"` (stamped on every `/v1/capture`,
  `/v1/telemetry`, and version-ping request).

---

## [1.0.0] — 2026-06-20

First official PyPI release. The SDK has been hardened for enterprise
deployment (zipped wheels, serverless layers, long-running agents).

### Added
- **`async with Aliax()` support.** `__aenter__` / `__aexit__` close the
  `httpx` connection pool deterministically — no more socket leaks when an
  agent loop crashes without calling `aclose()`.
- **`ALIAX_API_KEY` environment variable fallback.** Constructor accepts
  `api_key=None` and reads from the env, so 12-factor deployments don't
  need to thread the key through code.
- **`py.typed` marker.** PEP 561 type-checker support — `mypy`, Pyright,
  VS Code and PyCharm now consume the inline type hints.
- **`aliax.__version__` exported** and included in `__all__`.
- Structured `tests/` directory with `pytest` and a CI matrix
  (Python 3.8 → 3.12) plus an sdist/wheel build check via `twine check`.

### Changed
- **PEP 621 packaging.** Migrated from `setup.py` to `pyproject.toml`
  with `setuptools.build_meta`. Version is now read dynamically from
  `aliax/_version.py` via `tool.setuptools.dynamic`.
- **Distribution renamed** from `aliax-sdk` to `aliax` so the install
  name matches the import name (`pip install aliax` → `import aliax`).
- **Zip-safe bundled-asset loading.** `dom-mapper.min.js` is now read
  via `importlib.resources` instead of `os.path.join(__file__, …)`, so
  the SDK works inside zipped wheels, AWS Lambda layers, and Google
  Cloud Run container images.

### Added (legal)
- MIT `LICENSE` file in the package root and SPDX classifier on the
  distribution metadata.
