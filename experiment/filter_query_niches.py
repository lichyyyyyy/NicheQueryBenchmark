"""Filter exported niches by size and composition complexity from the manifest.

Run from the repository root::

    .venv/bin/python experiment/filter_query_niches.py \
        --source-slice C57BL6J-638850.28 --dimension large simple

Reads ``query_niche_metrics/large/<source-slice>.csv`` and writes a one-column
CSV of matching center_cell_name values under
``filtered_query_niches/large/simple/<source-slice>.csv``. All paths default to
directories beside this script. Override with --metrics-dir and --output-file.
Cell IDs remain strings. Input files are read-only and processed row by row.

All conditions in the selected manifest rule must hold. "simple" currently
requires 1–2 represented parcellations and a dominant cell fraction from 0.85
through 1.0; all bounds are inclusive.
Zero entries in the composition vector do not count as represented types.
Only exported niches meeting the selected size are eligible; no neighborhoods
are generated from H5AD files here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from collections.abc import Sequence

if __package__:
    from .query_niche_dimensions import COMPOSITION_COMPLEXITY, QUERY_NICHE_DIMENSIONS
else:
    from query_niche_dimensions import COMPOSITION_COMPLEXITY, QUERY_NICHE_DIMENSIONS

DEFAULT_METRICS_DIR = Path(__file__).resolve().parent / "query_niche_metrics"
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "query_niche_metrics/filtered_query_niches"
)


def matches_complexity(fractions_json: str, rule: dict) -> bool:
    """Apply the manifest rule to a fraction array or an ID-to-fraction map."""
    composition = json.loads(fractions_json)
    if isinstance(composition, dict):
        composition = list(composition.values())
    if not isinstance(composition, list) or not composition:
        raise ValueError("parcellation_fractions must be a nonempty JSON array or map")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for value in composition
    ):
        raise ValueError("Composition fractions must be finite, nonnegative numbers")
    if not math.isclose(sum(composition), 1.0, rel_tol=1e-6, abs_tol=1e-8):
        raise ValueError("Composition fractions must sum to 1")
    represented = sum(value > 0 for value in composition)
    dominant = max(composition)
    if "parcellation_count_at_least" in rule and not represented >= rule["parcellation_count_at_least"]:
        return False
    if "parcellation_count_at_most" in rule and not represented <= rule["parcellation_count_at_most"]:
        return False
    if "dominant_parcellation_fraction_at_least" in rule and not dominant >= rule["dominant_parcellation_fraction_at_least"]:
        return False
    if "dominant_parcellation_fraction_at_most" in rule and not dominant <= rule["dominant_parcellation_fraction_at_most"]:
        return False
    return True


def filter_niche_centers(
    source_slice: str,
    dimension: Sequence[str],
    *,
    metrics_dir: str | Path = DEFAULT_METRICS_DIR,
    output_file: str | Path | None = None,
) -> Path:
    """Write matching centers for a [size, complexity] pair, preserving row order."""
    if isinstance(dimension, str) or len(dimension) != 2:
        raise ValueError(
            "dimension must contain [size, complexity], e.g. ['large', 'simple']"
        )
    size_name, complexity_name = dimension
    if size_name not in QUERY_NICHE_DIMENSIONS:
        raise ValueError(
            f"Unknown size {size_name!r}; choose from {list(QUERY_NICHE_DIMENSIONS)}"
        )
    if complexity_name not in COMPOSITION_COMPLEXITY:
        raise ValueError(
            f"Unknown complexity {complexity_name!r}; choose from {list(COMPOSITION_COMPLEXITY)}"
        )
    if (
        not source_slice
        or Path(source_slice).name != source_slice
        or source_slice in {".", ".."}
    ):
        raise ValueError("source_slice must be a slice name without directories")
    rule = COMPOSITION_COMPLEXITY[complexity_name]
    for key in ("parcellation_count_at_least", "parcellation_count_at_most"):
        limit = rule.get(key)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError(f"{key} must be a positive integer")
    if rule["parcellation_count_at_least"] > rule["parcellation_count_at_most"]:
        raise ValueError("parcellation count lower bound cannot exceed upper bound")
    for key in (
        "dominant_parcellation_fraction_at_least",
        "dominant_parcellation_fraction_at_most",
    ):
        limit = rule.get(key)
        if (
            isinstance(limit, bool)
            or not isinstance(limit, (int, float))
            or not 0 <= limit <= 1
        ):
            raise ValueError(f"{key} must be between 0 and 1")
    if (
        rule["dominant_parcellation_fraction_at_least"]
        > rule["dominant_parcellation_fraction_at_most"]
    ):
        raise ValueError("dominant fraction lower bound cannot exceed upper bound")
    target_size = QUERY_NICHE_DIMENSIONS[size_name]
    source_path = Path(metrics_dir) / size_name / f"{source_slice}.csv"
    output_path = (
        Path(output_file)
        if output_file is not None
        else DEFAULT_OUTPUT_DIR / size_name / complexity_name / f"{source_slice}.csv"
    )
    if source_path.resolve() == output_path.resolve():
        raise ValueError("output_file must not overwrite the source metrics CSV")

    # The unused niche_cell_names field can exceed the default CSV field limit.
    previous_limit = csv.field_size_limit()
    csv.field_size_limit(100_000_000)
    temporary_path = None
    try:
        with source_path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            required = {
                "center_cell_name",
                "niche_cell_count",
                "parcellation_fractions",
            }
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{source_path}: missing columns {sorted(missing)}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                mode="w",
                newline="",
                encoding="utf-8",
                dir=output_path.parent,
                prefix=f".{output_path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as output:
                temporary_path = Path(output.name)
                writer = csv.writer(output)
                writer.writerow(["center_cell_name"])
                seen = set()
                for record in reader:
                    try:
                        if (
                            "target_niche_size" in record
                            and int(record["target_niche_size"]) != target_size
                        ):
                            continue
                        if (
                            "target_size_reached" in record
                            and record["target_size_reached"].strip().lower() != "true"
                        ):
                            continue
                        if int(record["niche_cell_count"]) < target_size:
                            continue
                        center = record["center_cell_name"]
                        if not center or center in seen:
                            raise ValueError(
                                f"Empty or duplicate eligible center ID: {center!r}"
                            )
                        seen.add(center)
                        if matches_complexity(record["parcellation_fractions"], rule):
                            writer.writerow([center])
                    except (ValueError, TypeError) as error:
                        raise ValueError(
                            f"{source_path}, CSV line {reader.line_num}: {error}"
                        ) from error
            os.replace(temporary_path, output_path)
    finally:
        csv.field_size_limit(previous_limit)
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source-slice", required=True, help="Slice name without the .csv extension"
    )
    parser.add_argument(
        "--dimension",
        nargs=2,
        required=True,
        metavar=("SIZE", "COMPLEXITY"),
        help="Manifest size and composition complexity, e.g. large simple",
    )
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--output-file", type=Path)
    args = parser.parse_args()
    try:
        path = filter_niche_centers(
            args.source_slice,
            args.dimension,
            metrics_dir=args.metrics_dir,
            output_file=args.output_file,
        )
    except (ValueError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
    print(path)
