"""Shared query niche dimensions, expressed as cell counts in the manifest."""

import json
from pathlib import Path

MANIFEST_PATH = Path(__file__).with_name("manifests") / "query_niche_dimensions.json"
with MANIFEST_PATH.open(encoding="utf-8") as handle:
    _manifest = json.load(handle)

COMPOSITION_COMPLEXITY = _manifest.get("composition_complexity", {})
QUERY_NICHE_DIMENSIONS = _manifest["niche_size"]


def parse_niche_size(value: str) -> int:
    """Accept a manifest dimension name or an explicit positive cell count."""
    if value in QUERY_NICHE_DIMENSIONS:
        return QUERY_NICHE_DIMENSIONS[value]
    try:
        size = int(value)
    except ValueError:
        raise ValueError(
            f"Expected one of {', '.join(QUERY_NICHE_DIMENSIONS)} or a positive cell count"
        ) from None
    if size < 1:
        raise ValueError("Niche size must be a positive cell count")
    return size
