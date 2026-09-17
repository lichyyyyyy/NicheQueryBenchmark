"""Aggregate per-query evaluation metrics by embedding type.

This script reads the per-query metrics produced by
``process_niche_query_raw_results.py`` and writes one macro-averaged row per
embedding type. Pearson and Spearman are averaged in Fisher-z space; ranking
metrics use an arithmetic mean, so every query has equal weight.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import tempfile
from pathlib import Path

import numpy as np


EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_QUERY_MANIFEST = EXPERIMENT_DIR / "manifests" / "query_manifest.csv"
DEFAULT_INPUT = (
    EXPERIMENT_DIR / "evaluation_results" / "evaluation_metrics_per_query.csv"
)
DEFAULT_OUTPUT = (
    EXPERIMENT_DIR / "evaluation_results" / "evaluation_metrics_by_embedding_type.csv"
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
AGGREGATE_OUTPUT_COLUMNS = ("embedding_type", *OUTPUT_COLUMNS[1:])
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
                raise ValueError(f"{path}:{line_number}: invalid metric row: {exc}") from exc
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
            raise ValueError(f"{query_manifest_path} is missing columns: {sorted(missing)}")
        for line_number, raw in enumerate(reader, start=2):
            try:
                query_id = int(raw["query_id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{query_manifest_path}:{line_number}: invalid query_id"
                ) from exc
            embedding_type = (raw["embedding_type"] or "").strip()
            if not embedding_type:
                raise ValueError(f"{query_manifest_path}:{line_number}: empty embedding_type")
            if query_id in manifest_query_ids:
                raise ValueError(f"{query_manifest_path}:{line_number}: duplicate query_id {query_id}")
            manifest_query_ids.add(query_id)
            pair_key = tuple((raw[column] or "").strip() for column in PAIRING_COLUMNS)
            if (raw["source_slice"] or "").strip() == (raw["target_slice"] or "").strip():
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
            raise KeyError(f"query_id={query_id} does not exist in {query_manifest_path}")
        if query_id in seen_query_ids:
            raise ValueError(f"Duplicate evaluation row for query_id={query_id}")
        seen_query_ids.add(query_id)
        if query_id not in excluded_query_ids:
            rows_by_embedding[metadata[query_id][0]].append(row)

    missing_query_ids = set(metadata) - seen_query_ids
    if missing_query_ids:
        raise ValueError(f"Evaluation results are missing {len(missing_query_ids)} manifest queries")

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


def write_metrics(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=AGGREGATE_OUTPUT_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-manifest", type=Path, default=DEFAULT_QUERY_MANIFEST)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    rows = load_metric_rows(args.input.resolve())
    aggregated = aggregate_metrics_by_embedding_type(rows, args.query_manifest.resolve())
    output = args.output.resolve()
    write_metrics(aggregated, output)
    logger.info("Wrote %d embedding-type aggregate rows to %s", len(aggregated), output)


if __name__ == "__main__":
    main()
