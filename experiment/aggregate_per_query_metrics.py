"""Aggregate per-query evaluation metrics by a requested dimension.

This script reads the per-query metrics produced by
``process_niche_query_raw_results.py`` and writes one macro-averaged row per
the requested dimension(s). Pearson and Spearman are averaged in Fisher-z space; ranking
metrics use an arithmetic mean, so every query has equal weight.

Examples::

    # Generate embedding_type-only and embedding_type/transferability CSVs.
    .venv/bin/python experiment/aggregate_per_query_metrics.py

    # Generate only the requested dimension combination.
    .venv/bin/python experiment/aggregate_per_query_metrics.py \
        --dimension embedding_type,transferability \
        --output experiment/evaluation_results/evaluation_metrics_by_embedding_and_transferability.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import tempfile
from fnmatch import fnmatch
from pathlib import Path

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_QUERY_MANIFEST = EXPERIMENT_DIR / "manifests" / "query_manifest.csv"
DEFAULT_INPUT = (
    EXPERIMENT_DIR / "evaluation_results" / "evaluation_metrics_per_query.csv"
)
DEFAULT_DIMENSION = "embedding_type"
SLICE_METADATA = {
    "Zhuang-ABCA-1.*": {"sample": "Zhuang-ABCA-1", "lab": "Zhuang"},
    "Zhuang-ABCA-3.*": {"sample": "Zhuang-ABCA-3", "lab": "Zhuang"},
    "C57BL6J-638850.*": {"sample": "C57BL6J-638850", "lab": "C57BL6J"},
}
TRANSFERABILITY_TIERS = (
    "same_sample_same_lab",
    "cross_sample_same_lab",
    "cross_sample_cross_lab",
)
DEFAULT_OUTPUT = (
    EXPERIMENT_DIR
    / "evaluation_results"
    / f"evaluation_metrics_by_{DEFAULT_DIMENSION}.csv"
)
DEFAULT_COMBINED_OUTPUT = (
    EXPERIMENT_DIR
    / "evaluation_results"
    / "evaluation_metrics_by_embedding_type_and_transferability.csv"
)
OUTPUT_COLUMNS = (
    "query_id",
    "pearson",
    "spearman",
    "ndcg_at_0_5_pct",
    "ndcg_at_1pct",
    "ndcg_at_5pct",
    "ndcg_at_top_200",
    "enrichment_at_0_5pct",
    "enrichment_at_1pct",
    "enrichment_at_5pct",
    "enrichment_at_top_200",
    "recall_at_0_5pct",
    "recall_at_1pct",
    "recall_at_5pct",
    "recall_at_top_200",
)
PAIRING_COLUMNS = (
    "query_niche_id",
    "source_slice",
    "target_slice",
    "niche_query_k",
)
logger = logging.getLogger(__name__)


def load_metric_rows(path: Path) -> list[dict[str, int | float]]:
    """Load and validate the complete per-query metric table."""
    if not path.is_file():
        raise FileNotFoundError(f"Per-query metrics not found: {path}")

    rows: list[dict[str, int | float]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = set(OUTPUT_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        seen: set[int] = set()
        for line_number, raw in enumerate(reader, start=2):
            try:
                query_id = int((raw.get("query_id") or "").strip())
                if query_id <= 0:
                    raise ValueError("query_id must be positive")
                row: dict[str, int | float] = {"query_id": query_id}
                for column in OUTPUT_COLUMNS[1:]:
                    value = float(raw[column])
                    if not math.isfinite(value) and not math.isnan(value):
                        raise ValueError(f"{column} must be finite or NaN")
                    row[column] = value
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid metric row: {exc}"
                ) from exc
            if query_id in seen:
                raise ValueError(f"Duplicate query_id={query_id} in {path}")
            seen.add(query_id)
            rows.append(row)

    if not rows:
        raise ValueError(f"Per-query metrics have no data rows: {path}")
    return rows


def _fisher_mean(values: list[float]) -> float:
    """Average correlations in Fisher-z space and transform back."""
    correlations = np.asarray(values, dtype=np.float64)
    correlations = correlations[np.isfinite(correlations)]
    if correlations.size == 0:
        return float("nan")
    if np.any((correlations < -1.0) | (correlations > 1.0)):
        raise ValueError("Correlation values must lie in [-1, 1]")
    correlations = np.clip(
        correlations,
        np.nextafter(-1.0, 0.0),
        np.nextafter(1.0, 0.0),
    )
    return float(np.tanh(np.mean(np.arctanh(correlations))))


def aggregate_metrics_by_embedding_type(
    rows: list[dict[str, int | float]], query_manifest_path: Path
) -> list[dict[str, str | float]]:
    """Aggregate paired per-query metrics with equal query weighting."""
    required_columns = {"query_id", "embedding_type", *PAIRING_COLUMNS}
    metadata: dict[int, tuple[str, tuple[str, ...]]] = {}
    manifest_query_ids: set[int] = set()
    excluded_query_ids: set[int] = set()
    embedding_order: list[str] = []
    paired_keys: dict[str, set[tuple[str, ...]]] = {}
    with query_manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{query_manifest_path} is missing columns: {sorted(missing)}"
            )
        for line_number, raw in enumerate(reader, start=2):
            try:
                query_id = int(raw["query_id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{query_manifest_path}:{line_number}: invalid query_id"
                ) from exc
            embedding_type = (raw["embedding_type"] or "").strip()
            if not embedding_type:
                raise ValueError(
                    f"{query_manifest_path}:{line_number}: empty embedding_type"
                )
            if query_id in manifest_query_ids:
                raise ValueError(
                    f"{query_manifest_path}:{line_number}: duplicate query_id {query_id}"
                )
            manifest_query_ids.add(query_id)
            pair_key = tuple((raw[column] or "").strip() for column in PAIRING_COLUMNS)
            if (raw["source_slice"] or "").strip() == (
                raw["target_slice"] or ""
            ).strip():
                excluded_query_ids.add(query_id)
                continue
            if embedding_type not in paired_keys:
                embedding_order.append(embedding_type)
                paired_keys[embedding_type] = set()
            if pair_key in paired_keys[embedding_type]:
                raise ValueError(
                    f"{query_manifest_path}:{line_number}: duplicate paired query "
                    f"configuration for embedding_type={embedding_type!r}"
                )
            paired_keys[embedding_type].add(pair_key)
            metadata[query_id] = (embedding_type, pair_key)

    if not embedding_order:
        raise ValueError(f"Query manifest has no data rows: {query_manifest_path}")
    reference_pairs = paired_keys[embedding_order[0]]
    for embedding_type in embedding_order[1:]:
        if paired_keys[embedding_type] != reference_pairs:
            raise ValueError(
                "Embedding types do not cover identical paired query configurations: "
                f"{embedding_order[0]!r} has {len(reference_pairs)}, "
                f"{embedding_type!r} has {len(paired_keys[embedding_type])}"
            )

    rows_by_embedding: dict[str, list[dict[str, int | float]]] = {
        embedding_type: [] for embedding_type in embedding_order
    }
    seen_query_ids: set[int] = set()
    for row in rows:
        query_id = int(row["query_id"])
        if query_id not in manifest_query_ids:
            raise KeyError(
                f"query_id={query_id} does not exist in {query_manifest_path}"
            )
        if query_id in seen_query_ids:
            raise ValueError(f"Duplicate evaluation row for query_id={query_id}")
        seen_query_ids.add(query_id)
        if query_id not in excluded_query_ids:
            rows_by_embedding[metadata[query_id][0]].append(row)

    missing_query_ids = set(metadata) - seen_query_ids
    if missing_query_ids:
        logger.warning(
            "Ignoring %d manifest query(ies) missing evaluation results: %s",
            len(missing_query_ids),
            sorted(missing_query_ids),
        )

    aggregated: list[dict[str, str | float]] = []
    for embedding_type in embedding_order:
        group = rows_by_embedding[embedding_type]
        result: dict[str, str | float] = {
            "embedding_type": embedding_type,
            "pearson": _fisher_mean([float(row["pearson"]) for row in group]),
            "spearman": _fisher_mean([float(row["spearman"]) for row in group]),
        }
        for column in OUTPUT_COLUMNS[3:]:
            result[column] = float(np.nanmean([float(row[column]) for row in group]))
        aggregated.append(result)
    return aggregated


def _slice_metadata(slice_name: str) -> dict[str, str]:
    for pattern, metadata in SLICE_METADATA.items():
        if fnmatch(slice_name, pattern):
            return metadata
    raise ValueError(f"Unrecognized slice: {slice_name!r}")


def _transferability(source_slice: str, target_slice: str) -> str:
    source = _slice_metadata(source_slice)
    target = _slice_metadata(target_slice)
    if source["sample"] == target["sample"]:
        return "same_sample_same_lab"
    if source["lab"] == target["lab"]:
        return "cross_sample_same_lab"
    return "cross_sample_cross_lab"


def aggregate_metrics_by_transferability(
    rows: list[dict[str, int | float]], query_manifest_path: Path
) -> list[dict[str, str | float]]:
    """Aggregate paired per-query metrics by transferability tier."""
    required_columns = {"query_id", *PAIRING_COLUMNS}
    metadata: dict[int, str] = {}
    tier_order: list[str] = list(TRANSFERABILITY_TIERS)
    manifest_query_ids: set[int] = set()
    with query_manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required_columns - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{query_manifest_path} is missing columns: {sorted(missing)}"
            )
        for line_number, raw in enumerate(reader, start=2):
            try:
                query_id = int(raw["query_id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{query_manifest_path}:{line_number}: invalid query_id"
                ) from exc
            if query_id in manifest_query_ids:
                raise ValueError(
                    f"{query_manifest_path}:{line_number}: duplicate query_id {query_id}"
                )
            manifest_query_ids.add(query_id)
            source_slice = (raw["source_slice"] or "").strip()
            target_slice = (raw["target_slice"] or "").strip()
            if not source_slice or not target_slice:
                raise ValueError(
                    f"{query_manifest_path}:{line_number}: source_slice and target_slice are required"
                )
            if source_slice == target_slice:
                continue
            metadata[query_id] = _transferability(source_slice, target_slice)

    rows_by_tier: dict[str, list[dict[str, int | float]]] = {
        tier: [] for tier in tier_order
    }
    seen_query_ids: set[int] = set()
    for row in rows:
        query_id = int(row["query_id"])
        if query_id not in manifest_query_ids:
            raise KeyError(
                f"query_id={query_id} does not exist in {query_manifest_path}"
            )
        if query_id in seen_query_ids:
            raise ValueError(f"Duplicate evaluation row for query_id={query_id}")
        seen_query_ids.add(query_id)
        if query_id in metadata:
            rows_by_tier[metadata[query_id]].append(row)

    aggregated: list[dict[str, str | float]] = []
    for tier in tier_order:
        group = rows_by_tier[tier]
        result: dict[str, str | float] = {
            "transferability": tier,
            "pearson": _fisher_mean([float(row["pearson"]) for row in group]),
            "spearman": _fisher_mean([float(row["spearman"]) for row in group]),
        }
        for column in OUTPUT_COLUMNS[3:]:
            result[column] = float(np.nanmean([float(row[column]) for row in group]))
        aggregated.append(result)
    return aggregated


def aggregate_metrics_by_dimensions(
    rows: list[dict[str, int | float]],
    query_manifest_path: Path,
    dimensions: tuple[str, ...],
) -> list[dict[str, str | float]]:
    """Aggregate paired per-query metrics by a combination of dimensions."""
    metadata: dict[int, tuple[str, ...]] = {}
    manifest_query_ids: set[int] = set()
    group_order: list[tuple[str, ...]] = []
    with query_manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"query_id", "embedding_type", *PAIRING_COLUMNS}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{query_manifest_path} is missing columns: {sorted(missing)}"
            )
        for line_number, raw in enumerate(reader, start=2):
            try:
                query_id = int(raw["query_id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{query_manifest_path}:{line_number}: invalid query_id"
                ) from exc
            if query_id in manifest_query_ids:
                raise ValueError(
                    f"{query_manifest_path}:{line_number}: duplicate query_id {query_id}"
                )
            manifest_query_ids.add(query_id)
            source_slice = (raw["source_slice"] or "").strip()
            target_slice = (raw["target_slice"] or "").strip()
            if source_slice == target_slice:
                continue
            values = {
                "embedding_type": (raw["embedding_type"] or "").strip(),
                "transferability": _transferability(source_slice, target_slice),
            }
            group = tuple(values[dimension] for dimension in dimensions)
            metadata[query_id] = group
            if group not in group_order:
                group_order.append(group)

    rows_by_group: dict[tuple[str, ...], list[dict[str, int | float]]] = {
        group: [] for group in group_order
    }
    seen_query_ids: set[int] = set()
    for row in rows:
        query_id = int(row["query_id"])
        if query_id not in manifest_query_ids:
            raise KeyError(
                f"query_id={query_id} does not exist in {query_manifest_path}"
            )
        if query_id in seen_query_ids:
            raise ValueError(f"Duplicate evaluation row for query_id={query_id}")
        seen_query_ids.add(query_id)
        if query_id in metadata:
            rows_by_group[metadata[query_id]].append(row)

    aggregated: list[dict[str, str | float]] = []
    for group_key in group_order:
        group_rows = rows_by_group[group_key]
        result: dict[str, str | float] = dict(zip(dimensions, group_key))
        result.update(
            {
                "pearson": _fisher_mean([float(row["pearson"]) for row in group_rows]),
                "spearman": _fisher_mean(
                    [float(row["spearman"]) for row in group_rows]
                ),
            }
        )
        for column in OUTPUT_COLUMNS[3:]:
            result[column] = float(
                np.nanmean([float(row[column]) for row in group_rows])
            )
        aggregated.append(result)
    return aggregated


def aggregate_metrics(
    rows: list[dict[str, int | float]],
    query_manifest_path: Path,
    dimension: str | tuple[str, ...],
) -> list[dict[str, str | float]]:
    """Aggregate metrics for the requested dimension."""
    if isinstance(dimension, tuple):
        return aggregate_metrics_by_dimensions(rows, query_manifest_path, dimension)
    if dimension == "embedding_type":
        return aggregate_metrics_by_embedding_type(rows, query_manifest_path)
    if dimension == "transferability":
        return aggregate_metrics_by_transferability(rows, query_manifest_path)
    raise ValueError(f"Unsupported aggregation dimension: {dimension!r}")


def write_metrics(
    rows: list[dict[str, object]],
    path: Path,
    dimensions: tuple[str, ...] = (DEFAULT_DIMENSION,),
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(
                handle,
                fieldnames=(*dimensions, *OUTPUT_COLUMNS[1:]),
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dimension",
        default=None,
        help="Dimension or comma-separated dimensions to aggregate.",
    )
    parser.add_argument("--query-manifest", type=Path, default=DEFAULT_QUERY_MANIFEST)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--combined-output",
        type=Path,
        default=DEFAULT_COMBINED_OUTPUT,
        help="Output CSV for embedding_type and transferability aggregates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    rows = load_metric_rows(args.input.resolve())
    manifest = args.query_manifest.resolve()
    output = args.output.resolve()
    if args.dimension is None:
        embedding_aggregated = aggregate_metrics(rows, manifest, "embedding_type")
        combined_dimensions = ("embedding_type", "transferability")
        combined_aggregated = aggregate_metrics(rows, manifest, combined_dimensions)
        combined_output = args.combined_output.resolve()
        write_metrics(embedding_aggregated, output)
        write_metrics(combined_aggregated, combined_output, combined_dimensions)
        logger.info(
            "Wrote %d embedding-type aggregate rows to %s",
            len(embedding_aggregated),
            output,
        )
        logger.info(
            "Wrote %d embedding-type/transferability aggregate rows to %s",
            len(combined_aggregated),
            combined_output,
        )
        return

    dimensions = tuple(
        dimension.strip()
        for dimension in args.dimension.split(",")
        if dimension.strip()
    )
    supported_dimensions = {"embedding_type", "transferability"}
    if (
        not dimensions
        or len(set(dimensions)) != len(dimensions)
        or any(dimension not in supported_dimensions for dimension in dimensions)
    ):
        raise ValueError(
            "--dimension must contain unique values from embedding_type and "
            "transferability, separated by commas"
        )
    dimension_arg: str | tuple[str, ...] = (
        dimensions[0] if len(dimensions) == 1 else dimensions
    )
    aggregated = aggregate_metrics(rows, manifest, dimension_arg)
    write_metrics(aggregated, output, dimensions)
    logger.info("Wrote %d %s aggregate rows to %s", len(aggregated), args.dimension, output)


if __name__ == "__main__":
    main()
