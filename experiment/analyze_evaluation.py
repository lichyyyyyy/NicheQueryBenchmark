"""Compute per-query agreement between niche-query and RM-Ideal scores.

The query list comes from ``query_manifest.csv``.  For every manifest row, the
corresponding ``raw_results/query_<query_id>.csv`` file is loaded and Pearson
and Spearman correlation coefficients are calculated between
``niche_query_score`` and ``rm_ideal_score``.  NDCG at the top 0.5%, top 1%,
top 5%, and top 200 retrieved cells is also calculated using RM-Ideal as graded
relevance.
Normalized enrichment at the top 0.5%, 1%, 5%, and 200 cells compares the
above-baseline mean RM-Ideal relevance of the true and predicted top-K sets.
Recall at the same cutoffs measures the fraction of the RM-Ideal top-K set
retrieved by the predicted top-K set.
Per-query metrics are also aggregated by embedding type. Pearson and Spearman
use a Fisher-z mean; ranking metrics use an arithmetic mean.

Example
-------
Run the analysis with the default paths::

    python experiment/analyze_evaluation.py

Use custom input/output locations::

    python experiment/analyze_evaluation.py \
        --query-manifest path/to/query_manifest.csv \
        --raw-results-dir path/to/raw_results \
        --output path/to/evaluation_metrics_per_query.csv
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
from scipy.stats import pearsonr, spearmanr

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_QUERY_MANIFEST = EXPERIMENT_DIR / "manifests" / "query_manifest.csv"
DEFAULT_RAW_RESULTS_DIR = EXPERIMENT_DIR / "raw_results"
DEFAULT_OUTPUT = (
    EXPERIMENT_DIR / "evaluation_results" / "evaluation_metrics_per_query.csv"
)
DEFAULT_AGGREGATE_OUTPUT = (
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
SCORE_COLUMNS = ("query_id", "niche_query_score", "rm_ideal_score")

logger = logging.getLogger(__name__)


def load_query_ids(path: Path) -> list[int]:
    """Load unique positive query IDs in manifest order."""
    if not path.is_file():
        raise FileNotFoundError(f"Query manifest not found: {path}")

    query_ids: list[int] = []
    seen: set[int] = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if "query_id" not in (reader.fieldnames or []):
            raise ValueError(f"{path} is missing required column 'query_id'")
        for line_number, row in enumerate(reader, start=2):
            raw_query_id = (row.get("query_id") or "").strip()
            try:
                query_id = int(raw_query_id)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid query_id {raw_query_id!r}"
                ) from exc
            if query_id <= 0:
                raise ValueError(
                    f"{path}:{line_number}: query_id must be positive; got {query_id}"
                )
            if query_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate query_id {query_id}")
            seen.add(query_id)
            query_ids.append(query_id)

    if not query_ids:
        raise ValueError(f"Query manifest has no data rows: {path}")
    return query_ids


def load_score_pairs(
    path: Path, expected_query_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """Load finite paired niche-query and RM-Ideal scores from one result CSV."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing raw result for query_id={expected_query_id}: {path}"
        )

    niche_scores: list[float] = []
    rm_ideal_scores: list[float] = []
    dropped = 0
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = set(SCORE_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        for line_number, row in enumerate(reader, start=2):
            raw_query_id = (row.get("query_id") or "").strip()
            try:
                row_query_id = int(raw_query_id)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid query_id {raw_query_id!r}"
                ) from exc
            if row_query_id != expected_query_id:
                raise ValueError(
                    f"{path}:{line_number}: expected query_id={expected_query_id}, "
                    f"got {row_query_id}"
                )

            try:
                niche_score = float(row["niche_query_score"])
                rm_ideal_score = float(row["rm_ideal_score"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}:{line_number}: score values must be numeric"
                ) from exc
            if not (math.isfinite(niche_score) and math.isfinite(rm_ideal_score)):
                dropped += 1
                continue
            niche_scores.append(niche_score)
            rm_ideal_scores.append(rm_ideal_score)

    if dropped:
        logger.warning(
            "query_id=%d: dropped %d row(s) containing non-finite scores",
            expected_query_id,
            dropped,
        )
    if len(niche_scores) < 2:
        raise ValueError(
            f"query_id={expected_query_id} has fewer than two finite score pairs in {path}"
        )
    return np.asarray(niche_scores), np.asarray(rm_ideal_scores)


def correlations(
    niche_scores: np.ndarray, rm_ideal_scores: np.ndarray
) -> tuple[float, float]:
    """Return Pearson r and tie-aware Spearman rho, or NaN if either input is constant."""
    if np.ptp(niche_scores) == 0.0 or np.ptp(rm_ideal_scores) == 0.0:
        return float("nan"), float("nan")
    pearson = float(pearsonr(niche_scores, rm_ideal_scores).statistic)
    spearman = float(spearmanr(niche_scores, rm_ideal_scores).statistic)
    return pearson, spearman


def _normalized_relevance(relevance: np.ndarray) -> np.ndarray:
    """Min-max normalize graded relevance to [0, 1]."""
    low = float(np.min(relevance))
    span = float(np.max(relevance)) - low
    if span < 1e-12:
        return np.zeros_like(relevance, dtype=np.float64)
    return (relevance - low) / span


def _dcg(relevance: np.ndarray) -> float:
    """Compute discounted cumulative gain for relevance in ranked order."""
    ranks = np.arange(relevance.size, dtype=np.float64)
    gains = np.power(2.0, relevance) - 1.0
    return float(np.sum(gains / np.log2(ranks + 2.0)))


def ndcg_metrics(
    niche_scores: np.ndarray, rm_ideal_scores: np.ndarray
) -> dict[str, float]:
    """Compute NDCG at top 0.5%, 1%, 5%, and 200 predictions."""
    relevance = _normalized_relevance(rm_ideal_scores)
    predicted_order = np.argsort(-niche_scores, kind="mergesort")
    predicted_relevance = relevance[predicted_order]
    ideal_relevance = np.sort(relevance)[::-1]
    n_cells = niche_scores.size

    cutoffs = {
        "ndcg_at_0_5_pct": max(1, math.ceil(0.005 * n_cells)),
        "ndcg_at_1pct": max(1, math.ceil(0.01 * n_cells)),
        "ndcg_at_5pct": max(1, math.ceil(0.05 * n_cells)),
        "ndcg_at_top_200": min(200, n_cells),
    }
    metrics: dict[str, float] = {}
    for name, cutoff in cutoffs.items():
        dcg = _dcg(predicted_relevance[:cutoff])
        ideal_dcg = _dcg(ideal_relevance[:cutoff])
        metrics[name] = 1.0 if ideal_dcg <= 1e-12 and dcg <= 1e-12 else dcg / ideal_dcg
    return metrics


def enrichment_metrics(
    niche_scores: np.ndarray, rm_ideal_scores: np.ndarray
) -> dict[str, float]:
    """Compute normalized enrichment for each top-K prediction set.

    The metric is ``(mean(predicted top-K) - mean(all)) / (mean(true top-K) -
    mean(all))``. True top-K is ranked by RM-Ideal itself, while predicted top-K
    is ranked by the niche-query score.
    """
    baseline = float(np.mean(rm_ideal_scores))
    n_cells = niche_scores.size
    cutoffs = {
        "enrichment_at_0_5pct": max(1, math.ceil(0.005 * n_cells)),
        "enrichment_at_1pct": max(1, math.ceil(0.01 * n_cells)),
        "enrichment_at_5pct": max(1, math.ceil(0.05 * n_cells)),
        "enrichment_at_top_200": min(200, n_cells),
    }
    predicted_order = np.argsort(-niche_scores, kind="mergesort")
    true_order = np.argsort(-rm_ideal_scores, kind="mergesort")
    predicted_relevance = rm_ideal_scores[predicted_order]
    true_relevance = rm_ideal_scores[true_order]

    metrics: dict[str, float] = {}
    for name, cutoff in cutoffs.items():
        predicted_enrichment = float(np.mean(predicted_relevance[:cutoff]) - baseline)
        true_enrichment = float(np.mean(true_relevance[:cutoff]) - baseline)
        if abs(true_enrichment) <= 1e-12:
            logger.warning("%s is undefined because its denominator is zero", name)
            metrics[name] = float("nan")
        else:
            metrics[name] = predicted_enrichment / true_enrichment
    return metrics


def recall_metrics(
    niche_scores: np.ndarray, rm_ideal_scores: np.ndarray
) -> dict[str, float]:
    """Compute ``|TopK_prediction intersect TopK_RM-Ideal| / K``."""
    n_cells = niche_scores.size
    cutoffs = {
        "recall_at_0_5pct": max(1, math.ceil(0.005 * n_cells)),
        "recall_at_1pct": max(1, math.ceil(0.01 * n_cells)),
        "recall_at_5pct": max(1, math.ceil(0.05 * n_cells)),
        "recall_at_top_200": min(200, n_cells),
    }
    predicted_order = np.argsort(-niche_scores, kind="mergesort")
    true_order = np.argsort(-rm_ideal_scores, kind="mergesort")
    return {
        name: float(
            np.intersect1d(
                predicted_order[:cutoff], true_order[:cutoff], assume_unique=True
            ).size
            / cutoff
        )
        for name, cutoff in cutoffs.items()
    }


def _fisher_mean(values: list[float]) -> float:
    """Average correlations in Fisher-z space and transform back."""
    correlations = np.asarray(values, dtype=np.float64)
    if correlations.size == 0:
        return float("nan")
    if np.any((correlations < -1.0) | (correlations > 1.0)):
        raise ValueError("Correlation values must lie in [-1, 1]")
    # Exact +/-1 maps to infinity. Clipping to the nearest interior floats keeps
    # the Fisher transform finite while preserving the limiting behavior.
    correlations = np.clip(
        correlations,
        np.nextafter(-1.0, 0.0),
        np.nextafter(1.0, 0.0),
    )
    return float(np.tanh(np.mean(np.arctanh(correlations))))


def aggregate_metrics_by_embedding_type(
    rows: list[dict[str, int | float]], query_manifest_path: Path
) -> list[dict[str, str | float]]:
    """Aggregate paired per-query metrics by manifest ``embedding_type``.

    Pearson and Spearman use a Fisher-z mean. NDCG, recall, and normalized
    enrichment metrics use an arithmetic mean. The function also verifies that
    every embedding type covers the same paired query configurations. Queries
    whose source and target slices are identical are excluded from aggregation.
    """
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
        for line_number, row in enumerate(reader, start=2):
            try:
                query_id = int(row["query_id"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{query_manifest_path}:{line_number}: invalid query_id"
                ) from exc
            embedding_type = (row["embedding_type"] or "").strip()
            if not embedding_type:
                raise ValueError(
                    f"{query_manifest_path}:{line_number}: empty embedding_type"
                )
            pair_key = tuple((row[column] or "").strip() for column in PAIRING_COLUMNS)
            if query_id in manifest_query_ids:
                raise ValueError(
                    f"{query_manifest_path}:{line_number}: duplicate query_id {query_id}"
                )
            manifest_query_ids.add(query_id)
            source_slice = (row["source_slice"] or "").strip()
            target_slice = (row["target_slice"] or "").strip()
            if source_slice == target_slice:
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
    reference_embedding = embedding_order[0]
    reference_pairs = paired_keys[reference_embedding]
    for embedding_type in embedding_order[1:]:
        if paired_keys[embedding_type] != reference_pairs:
            raise ValueError(
                "Embedding types do not cover identical paired query configurations: "
                f"{reference_embedding!r} has {len(reference_pairs)}, "
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
        if query_id in excluded_query_ids:
            continue
        embedding_type = metadata[query_id][0]
        rows_by_embedding[embedding_type].append(row)

    missing_query_ids = set(metadata) - seen_query_ids
    if missing_query_ids:
        raise ValueError(
            f"Evaluation results are missing {len(missing_query_ids)} manifest queries"
        )

    aggregated: list[dict[str, str | float]] = []
    arithmetic_columns = OUTPUT_COLUMNS[3:]
    for embedding_type in embedding_order:
        group = rows_by_embedding[embedding_type]
        result: dict[str, str | float] = {
            "embedding_type": embedding_type,
            "pearson": _fisher_mean([float(row["pearson"]) for row in group]),
            "spearman": _fisher_mean([float(row["spearman"]) for row in group]),
        }
        for column in arithmetic_columns:
            result[column] = float(np.mean([float(row[column]) for row in group]))
        aggregated.append(result)
    return aggregated


def write_metrics(
    rows: list[dict[str, object]], output_path: Path, columns: tuple[str, ...]
) -> None:
    """Atomically write the metric table so failed runs do not leave partial output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--query-manifest",
        type=Path,
        default=DEFAULT_QUERY_MANIFEST,
        help=f"Query manifest (default: {DEFAULT_QUERY_MANIFEST})",
    )
    parser.add_argument(
        "--raw-results-dir",
        type=Path,
        default=DEFAULT_RAW_RESULTS_DIR,
        help=f"Directory containing query_<id>.csv files (default: {DEFAULT_RAW_RESULTS_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output metric CSV (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--aggregate-output",
        type=Path,
        default=DEFAULT_AGGREGATE_OUTPUT,
        help=f"Embedding-type aggregate CSV (default: {DEFAULT_AGGREGATE_OUTPUT})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    query_ids = load_query_ids(args.query_manifest.resolve())
    raw_results_dir = args.raw_results_dir.resolve()
    rows: list[dict[str, int | float]] = []
    for position, query_id in enumerate(query_ids, start=1):
        result_path = raw_results_dir / f"query_{query_id}.csv"
        niche_scores, rm_ideal_scores = load_score_pairs(result_path, query_id)
        pearson, spearman = correlations(niche_scores, rm_ideal_scores)
        ndcg = ndcg_metrics(niche_scores, rm_ideal_scores)
        enrichment = enrichment_metrics(niche_scores, rm_ideal_scores)
        recall = recall_metrics(niche_scores, rm_ideal_scores)
        rows.append(
            {
                "query_id": query_id,
                "pearson": pearson,
                "spearman": spearman,
                **ndcg,
                **enrichment,
                **recall,
            }
        )
        logger.info(
            "[%d/%d] query_id=%d Pearson=%.6f Spearman=%.6f "
            "NDCG@1%%=%.6f NDCG@5%%=%.6f NDCG@top200=%.6f",
            position,
            len(query_ids),
            query_id,
            pearson,
            spearman,
            ndcg["ndcg_at_1pct"],
            ndcg["ndcg_at_5pct"],
            ndcg["ndcg_at_top_200"],
        )

    output_path = args.output.resolve()
    write_metrics(rows, output_path, OUTPUT_COLUMNS)
    logger.info("Wrote %d query metric rows to %s", len(rows), output_path)

    aggregate_rows = aggregate_metrics_by_embedding_type(
        rows, args.query_manifest.resolve()
    )
    aggregate_output_path = args.aggregate_output.resolve()
    write_metrics(
        aggregate_rows,
        aggregate_output_path,
        AGGREGATE_OUTPUT_COLUMNS,
    )
    logger.info(
        "Wrote %d embedding-type aggregate rows to %s",
        len(aggregate_rows),
        aggregate_output_path,
    )


if __name__ == "__main__":
    main()
