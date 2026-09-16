r"""Keep candidates with >SOURCE_MIN_MATCHED_NICHES source matches and >=MIN_MATCHED_NICHES matches in >=REQUIRED_MATCHED_SLICE other slices.

Example (run from the repository root)::

    .venv/bin/python experiment/filter_query_niche_on_prevalence.py \
        --input-file experiment/query_niche_metrics/filtered_query_niches/large/simple/C57BL6J-638850.28.csv \
        --source-slice C57BL6J-638850.28

Input is an upstream candidate CSV containing center_cell_name (extra columns
are preserved). Output defaults to <input-stem>_prevalence.csv beside the input.
Each retained center has target_slices (a JSON array of matching target slice
IDs) and target_similar_niche_counts (a JSON map of slice ID to match count).
These columns are recomputed if already present in the input. Every listed
target has >=MIN_MATCHED_NICHES similar niches; only the source slice is excluded
from target comparisons. Candidates must qualify in at least REQUIRED_MATCHED_SLICE
target slices. There is still one output row per center.
The niche size is inferred from the input's <size>/<complexity>/,
<size>_<complexity>/, or <size>/ directory; no dimension argument is needed.
Comparison populations are ALL eligible exported niches in metrics-dir/dimension,
not just the input candidates. Every other CSV in that directory is checked;
an empty slice contributes no qualifying matches, and missing CSVs are not
compared. Fewer than REQUIRED_MATCHED_SLICE available target slices yields a header-only output.

Matches visualize_query_niche_v2.ipynb: every parcellation fraction must differ
strictly by <0.05 in the source slice and <0.25 in other slices. IDs are aligned,
missing IDs mean zero, and equality within 1e-12 is excluded. Source counts
include the query itself and overlapping niches. No H5AD files are needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np

if __package__:
    from .query_niche_dimensions import COMPOSITION_COMPLEXITY, QUERY_NICHE_DIMENSIONS
else:
    from query_niche_dimensions import COMPOSITION_COMPLEXITY, QUERY_NICHE_DIMENSIONS

DEFAULT_METRICS_DIR = Path(__file__).resolve().parent / "query_niche_metrics"
SOURCE_THRESHOLD = 0.05
TARGET_THRESHOLD = 0.25
REQUIRED_MATCHED_SLICE = 6
MIN_MATCHED_NICHES = 10
SOURCE_MIN_MATCHED_NICHES = 100


def validated_ids(values):
    if (
        not isinstance(values, list)
        or not values
        or any(isinstance(x, bool) or not isinstance(x, int) for x in values)
        or len(values) != len(set(values))
    ):
        raise ValueError("Parcellation IDs must be a nonempty list of unique integers")
    return values


def read_composition(record, shared_ids):
    ids = (
        validated_ids(json.loads(record["parcellation_ids"]))
        if "parcellation_ids" in record
        else shared_ids
    )
    if ids is None:
        raise ValueError("Missing parcellation_ids.json and per-row parcellation_ids")
    fractions = np.asarray(json.loads(record["parcellation_fractions"]), dtype=float)
    if (
        fractions.ndim != 1
        or len(fractions) != len(ids)
        or not np.isfinite(fractions).all()
        or (fractions < 0).any()
        or not np.isclose(fractions.sum(), 1.0, rtol=1e-6, atol=1e-8)
    ):
        raise ValueError(
            f"Invalid composition for center {record['center_cell_name']!r}"
        )
    fractions = fractions / fractions.sum()
    return tuple(
        sorted((key, float(value)) for key, value in zip(ids, fractions) if value > 0)
    )


def load_population(path, target_size, shared_ids, requested=()):
    """Stream large membership CSVs; retain only unique compositions and counts."""
    population = Counter()
    selected = {}
    seen = set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"center_cell_name", "niche_cell_count", "parcellation_fractions"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            try:
                if (
                    "target_niche_size" in row
                    and int(row["target_niche_size"]) != target_size
                ):
                    continue
                if (
                    "target_size_reached" in row
                    and row["target_size_reached"].strip().lower() != "true"
                ):
                    continue
                if int(row["niche_cell_count"]) < target_size:
                    continue
                center = row["center_cell_name"]
                if not center or center in seen:
                    raise ValueError(f"Empty or duplicate eligible center: {center!r}")
                seen.add(center)
                composition = read_composition(row, shared_ids)
                population[composition] += 1
                if center in requested:
                    selected[center] = composition
            except (ValueError, TypeError) as error:
                raise ValueError(
                    f"{path}, CSV line {reader.line_num}: {error}"
                ) from error
    return population, selected


def similar_counts(queries, population, threshold):
    """Count matches once per distinct query, without a query-by-population tensor."""
    ids = sorted(
        {key for composition in population for key, _ in composition}
        | {key for composition in queries for key, _ in composition}
    )
    columns = {key: index for index, key in enumerate(ids)}
    matrix = np.zeros((len(population), len(ids)), dtype=float)
    weights = np.fromiter(population.values(), dtype=np.int64, count=len(population))
    for index, composition in enumerate(population):
        for key, value in composition:
            matrix[index, columns[key]] = value
    counts = {}
    for composition in queries:
        query = dict(composition)
        matches = np.arange(len(population))
        # Large query fractions tend to eliminate mismatches early.
        for key in sorted(ids, key=lambda key: query.get(key, 0.0), reverse=True):
            differences = np.abs(matrix[matches, columns[key]] - query.get(key, 0.0))
            matches = matches[
                (differences < threshold)
                & ~np.isclose(differences, threshold, rtol=0.0, atol=1e-12)
            ]
            if not len(matches):
                break
        counts[composition] = int(weights[matches].sum())
    return counts


def filter_niche_centers(
    input_file,
    source_slice,
    *,
    metrics_dir=DEFAULT_METRICS_DIR,
    output_file=None,
    exclude_slices=(),
):
    """Filter a single source/dimension candidate CSV, preserving columns and order."""
    if (
        not source_slice
        or Path(source_slice).name != source_slice
        or source_slice in {".", ".."}
    ):
        raise ValueError("source_slice must be a slice name without directories")
    excluded = set(exclude_slices or ())
    invalid_excluded = [
        name
        for name in excluded
        if not name or Path(name).name != name or name in {".", ".."}
    ]
    if invalid_excluded:
        raise ValueError(
            "exclude_slices must contain slice names without directories: "
            f"{sorted(invalid_excluded)}"
        )
    input_path = Path(input_file)
    parent = input_path.absolute().parent
    combined_dimensions = {
        f"{size}_{complexity}": size
        for size in QUERY_NICHE_DIMENSIONS
        for complexity in COMPOSITION_COMPLEXITY
    }
    # Check size/complexity first: "median" can name either of them.
    if (
        parent.name in COMPOSITION_COMPLEXITY
        and parent.parent.name in QUERY_NICHE_DIMENSIONS
    ):
        dimension = parent.parent.name
    else:
        dimension = combined_dimensions.get(parent.name, parent.name)
    if dimension not in QUERY_NICHE_DIMENSIONS:
        raise ValueError(
            "Cannot infer niche size from input path; place the candidate CSV "
            "in a <size>/<complexity>/, <size>_<complexity>/, or <size>/ directory "
            f"(sizes: {', '.join(QUERY_NICHE_DIMENSIONS)})"
        )
    output_path = (
        Path(output_file)
        if output_file is not None
        else input_path.with_name(f"{input_path.stem}_prevalence.csv")
    )
    directory = Path(metrics_dir) / dimension
    source_path = directory / f"{source_slice}.csv"
    paths = sorted(directory.glob("*.csv"))
    if not source_path.is_file():
        raise FileNotFoundError(f"Source metrics CSV not found: {source_path}")
    if output_path.resolve() in {
        input_path.resolve(),
        *(path.resolve() for path in paths),
    }:
        raise ValueError("Output must not overwrite candidates or metrics")
    if output_path.resolve().parent == directory.resolve():
        raise ValueError("Output must be outside the slice metrics directory")
    previous_limit = csv.field_size_limit()
    csv.field_size_limit(100_000_000)
    temporary_path = None
    try:
        with input_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            if not fields or "center_cell_name" not in fields:
                raise ValueError(f"{input_path}: missing center_cell_name column")
            rows = list(reader)
        centers = [row["center_cell_name"] for row in rows]
        if any(not center for center in centers) or len(set(centers)) != len(centers):
            raise ValueError(
                "Input contains empty or duplicate center_cell_name values"
            )
        ids_path = directory / "parcellation_ids.json"
        shared_ids = (
            validated_ids(json.loads(ids_path.read_text(encoding="utf-8")))
            if ids_path.is_file()
            else None
        )
        target_size = QUERY_NICHE_DIMENSIONS[dimension]
        population, selected = load_population(
            source_path, target_size, shared_ids, set(centers)
        )
        missing = set(centers) - selected.keys()
        if missing:
            raise ValueError(
                f"Candidates absent from eligible source niches: {sorted(missing)[:5]}"
            )
        counts = similar_counts(set(selected.values()), population, SOURCE_THRESHOLD)
        kept = [
            center
            for center in centers
            if counts[selected[center]] > SOURCE_MIN_MATCHED_NICHES
        ]
        target_counts = {
            composition: {} for composition in {selected[center] for center in kept}
        }
        print(
            f"{source_slice}: {len(kept)}/{len(centers)} candidates have >{SOURCE_MIN_MATCHED_NICHES} source matches",
            flush=True,
        )
        del population
        target_paths = [
            path for path in paths if path.resolve() != source_path.resolve()
        ]
        if excluded:
            excluded_available = sorted(
                path.stem
                for path in paths
                if path.resolve() != source_path.resolve() and path.stem in excluded
            )
            message = (
                "--exclude-slice is ignored during prevalence comparisons; "
                f"including {len(excluded_available)} matching target slice CSVs"
            )
            if excluded_available:
                message += ": " + ", ".join(excluded_available)
            print(message, flush=True)
        if len(target_paths) < REQUIRED_MATCHED_SLICE:
            print(
                f"Only {len(target_paths)} other slice CSVs found; at least {REQUIRED_MATCHED_SLICE} are required.",
                flush=True,
            )
        for path in target_paths:
            if not kept:
                break
            population, _ = load_population(path, target_size, shared_ids)
            counts = similar_counts(
                {selected[center] for center in kept}, population, TARGET_THRESHOLD
            )
            for composition in {selected[center] for center in kept}:
                if counts[composition] >= MIN_MATCHED_NICHES:
                    target_counts[composition][path.stem] = counts[composition]
            qualifying_count = sum(
                counts[selected[center]] >= MIN_MATCHED_NICHES for center in kept
            )
            print(
                f"{path.stem}: {qualifying_count} candidates have >={MIN_MATCHED_NICHES} matches",
                flush=True,
            )
            del population
        retained = {
            center
            for center in kept
            if len(target_counts[selected[center]]) >= REQUIRED_MATCHED_SLICE
        }
        print(
            f"Retained {len(retained)}/{len(centers)} candidates with >{SOURCE_MIN_MATCHED_NICHES} source matches "
            f"and >={MIN_MATCHED_NICHES} matches in at least {REQUIRED_MATCHED_SLICE} other slices",
            flush=True,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=output_path.parent,
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            output_fields = fields + [
                name
                for name in ("target_slices", "target_similar_niche_counts")
                if name not in fields
            ]
            writer = csv.DictWriter(handle, fieldnames=output_fields)
            writer.writeheader()
            for row in rows:
                center = row["center_cell_name"]
                if center not in retained:
                    continue
                matches = target_counts[selected[center]]
                writer.writerow(
                    {
                        **row,
                        "target_slices": json.dumps(list(matches)),
                        "target_similar_niche_counts": json.dumps(matches),
                    }
                )
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
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--source-slice", required=True)
    parser.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    parser.add_argument("--output-file", type=Path)
    parser.add_argument(
        "--exclude-slice",
        dest="exclude_slices",
        action="append",
        default=[],
        help=(
            "Accepted for compatibility, but ignored: prevalence comparisons include "
            "every non-source target slice."
        ),
    )
    args = parser.parse_args()
    try:
        result = filter_niche_centers(**vars(args))
    except (ValueError, OSError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
    print(result)
