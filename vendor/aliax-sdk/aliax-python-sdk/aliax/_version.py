"""Single source of truth for the SDK version.

Bumped manually on each PyPI release. Read by:
  - aliax.client.Aliax (sent as `sdk_version` form field on every capture
    and in the /v1/telemetry pings emitted from parse_ui / execute)
  - pyproject.toml (via [tool.setuptools.dynamic] — so pip metadata matches)
  - the background version-ping that warns users on stale installs

Bump the Cloudflare Worker's LATEST_SDK_VERSION wrangler var in the same
release so existing installs see the upgrade nag.
"""

__version__ = "1.0.5"
