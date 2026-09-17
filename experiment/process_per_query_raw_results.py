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
Aggregation is handled separately by
``aggregate_niche_query_metrics.py``.

Example
-------
Run the analysis with the default paths::

    python experiment/process_niche_query_raw_results.py

Use custom input/output locations::

    python experiment/process_niche_query_raw_results.py \
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


def load_existing_metrics(path: Path) -> dict[int, dict[str, int | float]]:
    """Load reusable per-query metric rows from an existing output file.

    Rows with the current schema and numeric metric values are returned.  An
    old or incomplete output file is treated as having no reusable rows so the
    missing metrics can be recomputed normally.
    """
    if not path.is_file():
        return {}

    reusable: dict[int, dict[str, int | float]] = {}
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = set(OUTPUT_COLUMNS) - set(reader.fieldnames or [])
            if missing:
                logger.warning(
                    "Existing metrics file %s has an old/incomplete schema; "
                    "recomputing affected queries",
                    path,
                )
                return {}

            for line_number, raw in enumerate(reader, start=2):
                try:
                    query_id = int((raw.get("query_id") or "").strip())
                    if query_id <= 0:
                        raise ValueError("query_id must be positive")
                    parsed: dict[str, int | float] = {"query_id": query_id}
                    for column in OUTPUT_COLUMNS[1:]:
                        parsed[column] = float(raw[column])
                except (TypeError, ValueError) as exc:
                    logger.warning(
                        "Ignoring invalid existing metric row %s:%d: %s",
                        path,
                        line_number,
                        exc,
                    )
                    continue
                if query_id in reusable:
                    raise ValueError(
                        f"Duplicate query_id={query_id} in existing metrics {path}"
                    )
                reusable[query_id] = parsed
    except csv.Error as exc:
        logger.warning(
            "Could not read existing metrics file %s; recomputing queries: %s",
            path,
            exc,
        )
        return {}

    return reusable


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    query_ids = load_query_ids(args.query_manifest.resolve())
    output_path = args.output.resolve()
    existing_metrics = load_existing_metrics(output_path)
    raw_results_dir = args.raw_results_dir.resolve()
    rows: list[dict[str, int | float]] = []
    for position, query_id in enumerate(query_ids, start=1):
        if query_id in existing_metrics:
            rows.append(existing_metrics[query_id])
            logger.info(
                "[%d/%d] query_id=%d reused metrics from %s",
                position,
                len(query_ids),
                query_id,
                output_path,
            )
            continue

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

    write_metrics(rows, output_path, OUTPUT_COLUMNS)
    logger.info("Wrote %d query metric rows to %s", len(rows), output_path)


if __name__ == "__main__":
    main()
