"""Run niche queries described by ``query_manifest.csv``.

For every selected query row, this script:

1. Looks up ``query_niche_id`` in ``query_niche_manifest.csv``.
2. Loads the source and target slice AnnData files.
3. Reuses a binary source-niche mask when available, or constructs the niche
   from the manifest's center cell, k-hop, cell limit, and parcellations.
4. Persists a newly constructed mask in
   ``source_adata.obs[query_niche_id]`` (and in ``obsm[niche_name]``).
5. Runs the requested embedding-based niche query and RM-Ideal scoring, then
   writes one result CSV per query under ``experiment/raw_results``. Target
   AnnData files are not changed.

Examples
--------
Run all manifest rows::

    PYTHONPATH=. python experiment/run_manifest_niche_queries.py

Run a small subset::

    PYTHONPATH=. python experiment/run_manifest_niche_queries.py \
        --query-ids 1 2 3

Validate manifests and input paths without loading AnnData::

    python experiment/run_manifest_niche_queries.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "20260601_225717"
DEFAULT_QUERY_MANIFEST = Path(__file__).with_name("query_manifest.csv")
DEFAULT_NICHE_MANIFEST = Path(__file__).with_name("query_niche_manifest.csv")
DEFAULT_RAW_RESULTS_DIR = Path(__file__).with_name("raw_results")
RM_IDEAL_MEMORY_CACHE_SIZE = 8
RESULT_COLUMNS = (
    "query_id",
    "cell_id",
    "niche_query_score",
    "niche_query_score_rank",
    "rm_ideal_score",
    "rm_ideal_rank",
)

# Database.construct_from_sample_adatas uses ``gene_expression`` for adata.X;
# every other value is interpreted as an adata.obsm key.
DEFAULT_EMBEDDING_FEATURES = {
    "gene_expr_quest": "X_gene_expr_quest",
    "scgpt": "X_scgpt",
    "gene_expr": "gene_expression",
}

QUERY_REQUIRED_COLUMNS = {
    "query_id",
    "embedding_type",
    "query_niche_id",
    "source_slice",
    "target_slice",
    "niche_query_k",
}
NICHE_REQUIRED_COLUMNS = {
    "query_niche_id",
    "source_slice",
    "niche_name",
    "center_cell",
    "k_hop",
    "cells",
    "parcellations_in_niche",
}

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QueryRow:
    query_id: int
    embedding_type: str
    query_niche_id: str
    source_slice: str
    target_slice: str
    niche_query_k: int


@dataclass(frozen=True)
class NicheRow:
    query_niche_id: str
    source_slice: str
    niche_name: str
    center_cell: str
    k_hop: int
    cell_limit: int
    parcellations: tuple[int, ...]


RmIdealKey = tuple[str, str, int]
RmIdealCache = OrderedDict[RmIdealKey, Any]


def rm_ideal_key(row: QueryRow) -> RmIdealKey:
    """Identify inputs that fully determine an RM-Ideal score vector."""
    return (row.query_niche_id, row.target_slice, row.niche_query_k)


def _read_csv(path: Path, required_columns: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        actual = set(reader.fieldnames or [])
        missing = required_columns - actual
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Manifest has no data rows: {path}")
    return rows


def _required_text(row: dict[str, str], key: str, path: Path) -> str:
    value = (row.get(key) or "").strip()
    if not value:
        raise ValueError(f"Empty {key!r} value in {path}")
    return value


def _positive_int(row: dict[str, str], key: str, path: Path) -> int:
    raw = _required_text(row, key, path)
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid integer {key}={raw!r} in {path}") from exc
    if value <= 0:
        raise ValueError(f"{key} must be positive in {path}; got {value}")
    return value


def load_niche_manifest(path: Path) -> dict[str, NicheRow]:
    # Index niche definitions by ID so every query row can perform an O(1)
    # lookup instead of repeatedly scanning query_niche_manifest.csv.
    result: dict[str, NicheRow] = {}
    for raw in _read_csv(path, NICHE_REQUIRED_COLUMNS):
        query_niche_id = _required_text(raw, "query_niche_id", path)
        if query_niche_id in result:
            raise ValueError(f"Duplicate query_niche_id {query_niche_id!r} in {path}")
        parcellations_raw = _required_text(raw, "parcellations_in_niche", path)
        # Multiple parcellation IDs are encoded as a semicolon-delimited field,
        # for example "207;123".
        try:
            parcellations = tuple(
                int(value.strip())
                for value in parcellations_raw.split(";")
                if value.strip()
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid parcellations_in_niche={parcellations_raw!r} in {path}"
            ) from exc
        if not parcellations:
            raise ValueError(
                f"No parcellations supplied for query_niche_id={query_niche_id!r}"
            )
        result[query_niche_id] = NicheRow(
            query_niche_id=query_niche_id,
            source_slice=_required_text(raw, "source_slice", path),
            niche_name=_required_text(raw, "niche_name", path),
            center_cell=_required_text(raw, "center_cell", path),
            k_hop=_positive_int(raw, "k_hop", path),
            cell_limit=_positive_int(raw, "cells", path),
            parcellations=parcellations,
        )
    return result


def load_query_manifest(
    path: Path,
    niches: dict[str, NicheRow],
    selected_query_ids: set[int] | None,
) -> list[QueryRow]:
    result: list[QueryRow] = []
    seen_ids: set[int] = set()
    for raw in _read_csv(path, QUERY_REQUIRED_COLUMNS):
        query_id = _positive_int(raw, "query_id", path)
        if query_id in seen_ids:
            raise ValueError(f"Duplicate query_id={query_id} in {path}")
        seen_ids.add(query_id)
        # Filter early so a small --query-ids run does not retain all 288 rows.
        if selected_query_ids is not None and query_id not in selected_query_ids:
            continue

        query_niche_id = _required_text(raw, "query_niche_id", path)
        if query_niche_id not in niches:
            raise KeyError(
                f"query_id={query_id}: query_niche_id={query_niche_id!r} "
                f"does not exist in the niche manifest"
            )
        source_slice = _required_text(raw, "source_slice", path)
        # The niche manifest is authoritative for where a query niche lives.
        # Reject mismatches rather than accidentally constructing it on another slice.
        expected_source = niches[query_niche_id].source_slice
        if source_slice != expected_source:
            raise ValueError(
                f"query_id={query_id}: source_slice={source_slice!r} does not match "
                f"the niche manifest value {expected_source!r}"
            )
        niche_query_k = _positive_int(raw, "niche_query_k", path)
        if not 1 <= niche_query_k <= 10:
            raise ValueError(
                f"query_id={query_id}: niche_query_k must be between 1 and 10; "
                f"got {niche_query_k}"
            )
        result.append(
            QueryRow(
                query_id=query_id,
                embedding_type=_required_text(raw, "embedding_type", path),
                query_niche_id=query_niche_id,
                source_slice=source_slice,
                target_slice=_required_text(raw, "target_slice", path),
                niche_query_k=niche_query_k,
            )
        )

    if selected_query_ids is not None:
        missing = selected_query_ids - {row.query_id for row in result}
        if missing:
            raise KeyError(
                f"Requested query_id values are absent from {path}: {sorted(missing)}"
            )
    return result


def parse_embedding_overrides(values: Iterable[str]) -> dict[str, str]:
    # Start with repository conventions and let callers override keys when their
    # AnnData files use a different obsm naming scheme.
    mapping = dict(DEFAULT_EMBEDDING_FEATURES)
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Invalid --embedding-feature {value!r}; expected EMBEDDING_TYPE=FEATURE"
            )
        embedding_type, feature = (part.strip() for part in value.split("=", 1))
        if not embedding_type or not feature:
            raise ValueError(
                f"Invalid --embedding-feature {value!r}; both sides are required"
            )
        mapping[embedding_type] = feature
    return mapping


def validate_inputs(
    rows: list[QueryRow], data_dir: Path, embedding_features: dict[str, str]
) -> None:
    missing_types = sorted(
        {row.embedding_type for row in rows} - embedding_features.keys()
    )
    if missing_types:
        raise KeyError(
            f"No feature mapping for embedding type(s) {missing_types}; use "
            "--embedding-feature EMBEDDING_TYPE=FEATURE"
        )
    # Validate every unique slice once; source and target sets can overlap.
    slice_ids = {row.source_slice for row in rows} | {row.target_slice for row in rows}
    missing_files = [
        str(data_dir / f"{slice_id}.h5ad")
        for slice_id in sorted(slice_ids)
        if not (data_dir / f"{slice_id}.h5ad").is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Missing slice file(s):\n  " + "\n  ".join(missing_files)
        )


def _binary_mask(values: Any, n_obs: int) -> np.ndarray | None:
    import numpy as np

    # A query-niche membership column must be finite, aligned to observations,
    # binary, and contain at least one member. Continuous RM-Ideal columns can
    # share similar names, so accepting arbitrary numeric values would be unsafe.
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.shape[0] != n_obs or not np.isfinite(array).all():
        return None
    if not np.isin(array, (0.0, 1.0)).all() or not np.any(array == 1.0):
        return None
    return array.astype(bool)


def _niche_from_mask(db: Any, sample_id: str, mask: np.ndarray) -> Any:
    from src.NicheQueryPrototype.niche import Niche

    sample = db.get_sample(sample_id)
    if sample is None or sample.adata is None:
        raise ValueError(f"Source sample {sample_id!r} is missing from the database")
    # Database preserves AnnData observation order when it constructs Cells, so
    # the mask and sample.cells can be zipped directly.
    cells = [cell for cell, keep in zip(sample.cells, mask) if keep]
    if not cells:
        raise ValueError(f"Niche mask for source sample {sample_id!r} is empty")
    niche = Niche()
    niche.construct(cells, sample_id)
    # Niche.construct only assigns cells; the mean feature must be computed
    # explicitly before NicheQuery can compare it with target neighborhoods.
    niche.compute_niche_feature()
    return niche


def get_or_construct_niche(db: Any, metadata: NicheRow) -> tuple[Any, bool]:
    """Return the source niche and whether a new obs mask must be persisted."""
    import numpy as np

    from src.NicheQueryPrototype.niche import Niche

    sample = db.get_sample(metadata.source_slice)
    if sample is None or sample.adata is None:
        raise ValueError(f"Source sample {metadata.source_slice!r} is unavailable")
    adata = sample.adata

    # Prefer the full query_niche_id in obs, as requested by the experiment
    # manifest. Only a true binary membership vector is safe to reuse.
    obs_exists = metadata.query_niche_id in adata.obs.columns
    if obs_exists:
        mask = _binary_mask(adata.obs[metadata.query_niche_id], adata.n_obs)
        if mask is not None:
            logger.info(
                "Reusing obs[%r] as the source niche mask", metadata.query_niche_id
            )
            return _niche_from_mask(db, metadata.source_slice, mask), False
        logger.warning(
            "obs[%r] exists but is not a non-empty binary mask; it may be an "
            "RM-Ideal score. It will not be interpreted as niche membership.",
            metadata.query_niche_id,
        )

    # Older repository code stores masks in obsm under a shorter niche_name.
    # Reuse that representation and mirror it into obs when obs is missing.
    if metadata.niche_name in adata.obsm:
        mask = _binary_mask(adata.obsm[metadata.niche_name], adata.n_obs)
        if mask is not None:
            logger.info(
                "Reusing obsm[%r] as the source niche mask", metadata.niche_name
            )
            if not obs_exists:
                adata.obs[metadata.query_niche_id] = mask.astype(np.int8)
                return _niche_from_mask(db, metadata.source_slice, mask), True
            return _niche_from_mask(db, metadata.source_slice, mask), False

    logger.info(
        "Constructing %s: center=%s, k=%d, cell_limit=%d, parcellations=%s",
        metadata.query_niche_id,
        metadata.center_cell,
        metadata.k_hop,
        metadata.cell_limit,
        list(metadata.parcellations),
    )
    # Neither supported storage location contained a usable mask. Reconstruct
    # the niche using the exact experimental parameters from the niche manifest.
    niche = Niche()
    niche.construct_by_k_hop(
        db,
        center_cell_id=metadata.center_cell,
        sample_id=metadata.source_slice,
        k=metadata.k_hop,
        cell_limit=metadata.cell_limit,
        parcellation_index=list(metadata.parcellations),
        niche_name=metadata.niche_name if not obs_exists else None,
    )
    if not niche.cells:
        raise ValueError(f"Constructed niche {metadata.query_niche_id!r} is empty")

    # Do not overwrite an existing continuous column: it may contain RM-Ideal
    # ground-truth scores. A missing column receives a binary membership mask.
    if not obs_exists:
        niche_cell_ids = {str(cell.id) for cell in niche.cells}
        adata.obs[metadata.query_niche_id] = np.fromiter(
            (1 if str(cell_id) in niche_cell_ids else 0 for cell_id in adata.obs_names),
            dtype=np.int8,
            count=adata.n_obs,
        )
        return niche, True
    return niche, False


def atomic_write_h5ad(adata: Any, destination: Path) -> None:
    """Replace an h5ad only after a complete temporary write succeeds."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".h5ad", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        # Writing beside the destination keeps os.replace on the same filesystem,
        # where replacement is atomic on normal local filesystems.
        adata.write_h5ad(temporary)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_query_results(
    query_id: int,
    cell_ids: Iterable[Any],
    niche_query_scores: Iterable[float],
    ranks: Iterable[int],
    rm_ideal_scores: Iterable[float],
    rm_ideal_ranks: Iterable[int],
    destination: Path,
) -> None:
    """Write a query result CSV without exposing a partially written file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".csv", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        # Stream rows instead of building a large in-memory table: each target
        # slice can contain tens of thousands of cells.
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(RESULT_COLUMNS)
            writer.writerows(
                (
                    query_id,
                    str(cell_id),
                    float(niche_query_score),
                    int(rank),
                    float(rm_ideal_score),
                    int(rm_ideal_rank),
                )
                for (
                    cell_id,
                    niche_query_score,
                    rank,
                    rm_ideal_score,
                    rm_ideal_rank,
                ) in zip(
                    cell_ids,
                    niche_query_scores,
                    ranks,
                    rm_ideal_scores,
                    rm_ideal_ranks,
                )
            )
        os.replace(temporary, destination)
    except BaseException:
        # os.fdopen owns the descriptor after it succeeds. If it failed before
        # taking ownership, close the descriptor explicitly.
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def result_csv_has_current_schema(path: Path) -> bool:
    """Return whether an existing result can be safely treated as complete."""
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            header = next(csv.reader(handle), None)
    except OSError:
        return False
    return header == list(RESULT_COLUMNS)


def find_rm_ideal_sources(
    rows: Iterable[QueryRow], raw_results_dir: Path
) -> dict[RmIdealKey, Path]:
    """Index completed results that can supply embedding-independent scores."""
    sources: dict[RmIdealKey, Path] = {}
    for row in rows:
        result_path = raw_results_dir / f"query_{row.query_id}.csv"
        if result_csv_has_current_schema(result_path):
            sources.setdefault(rm_ideal_key(row), result_path)
    return sources


def read_rm_ideal_scores(path: Path) -> Any:
    """Read only the RM-Ideal column from a current-schema result CSV."""
    import numpy as np

    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(RESULT_COLUMNS):
            raise ValueError(
                f"Cannot reuse RM-Ideal scores from {path}: schema changed"
            )
        return np.fromiter(
            (float(result["rm_ideal_score"]) for result in reader), dtype=float
        )


def remember_rm_ideal_scores(cache: RmIdealCache, key: RmIdealKey, scores: Any) -> None:
    """Store an RM-Ideal vector in a small LRU cache to bound memory usage."""
    cache[key] = scores
    cache.move_to_end(key)
    while len(cache) > RM_IDEAL_MEMORY_CACHE_SIZE:
        cache.popitem(last=False)


def run_one_query(
    row: QueryRow,
    metadata: NicheRow,
    data_dir: Path,
    raw_results_dir: Path,
    feature_name: str,
    overwrite_results: bool,
    knn_backend: str,
    rm_ideal_sources: dict[RmIdealKey, Path],
    rm_ideal_cache: RmIdealCache,
) -> bool:
    """Run one row; return False when an existing result caused a skip."""
    import anndata as ad
    import numpy as np

    from src.NicheQueryPrototype.database import Database
    from src.NicheQueryPrototype.query import NicheQuery

    source_path = data_dir / f"{row.source_slice}.h5ad"
    target_path = data_dir / f"{row.target_slice}.h5ad"
    result_path = raw_results_dir / f"query_{row.query_id}.csv"

    # A result with the current schema acts as a checkpoint for resumable runs.
    # Files created by the earlier cosine-only format are regenerated so they
    # receive the new rm_ideal_score column.
    if result_path.is_file() and not overwrite_results:
        if result_csv_has_current_schema(result_path):
            logger.info(
                "[SKIP] query_id=%d: %s already exists", row.query_id, result_path
            )
            return False
        logger.info(
            "[RECOMPUTE] query_id=%d: %s uses an outdated result schema",
            row.query_id,
            result_path,
        )

    # Only the source and target needed by this manifest row are loaded. This
    # avoids holding all twelve large slices in memory at once.
    source_adata = ad.read_h5ad(source_path)
    target_adata = (
        source_adata if source_path == target_path else ad.read_h5ad(target_path)
    )

    # When source == target, pass one AnnData object under one sample ID.
    sample_adatas = {row.source_slice: source_adata}
    if row.target_slice != row.source_slice:
        sample_adatas[row.target_slice] = target_adata

    # Build a row-local database using either adata.X (gene_expr) or the mapped
    # embedding matrix in adata.obsm (QueST/scGPT).
    db = Database(target_sample_ids=list(sample_adatas))
    db.construct_from_sample_adatas(
        sample_adatas=sample_adatas,
        feature_name=feature_name,
        parcellation_obs_key="parcellation_index",
        parcellation_write_obs_key="parcellation_index",
        spatial_obsm_key="spatial",
        preserve_input_adata=True,
    )

    niche, source_dirty = get_or_construct_niche(db, metadata)
    db_target = db.get_sample(row.target_slice)
    if db_target is None or db_target.adata is None:
        raise ValueError(f"Target sample {row.target_slice!r} is unavailable")

    # Generate target-cell neighborhood features, then compare each one with the
    # mean query-niche feature using cosine similarity.
    niche_query = NicheQuery(db=db, niche=niche, k=row.niche_query_k)
    niche_query.generate_niche_features_and_parcellation_mask(
        [db_target], knn_backend=knn_backend
    )
    niche_query_scores = np.asarray(
        niche_query.niche_query_within_a_sample(db_target), dtype=float
    ).reshape(-1)
    if niche_query_scores.shape[0] != target_adata.n_obs:
        raise ValueError(
            f"query_id={row.query_id}: niche-query score count "
            f"{niche_query_scores.shape[0]} does not "
            f"match target n_obs {target_adata.n_obs}"
        )

    # Rank predicted cosine scores from highest to lowest. Stable sorting makes
    # ties deterministic by preserving the target AnnData observation order.
    descending_order = np.argsort(-niche_query_scores, kind="stable")
    ranks = np.empty(niche_query_scores.shape[0], dtype=np.int64)
    ranks[descending_order] = np.arange(1, niche_query_scores.shape[0] + 1)

    # RM-Ideal depends on niche geometry, target spatial labels, and k, but not on
    # the embedding. Reuse it across the three embedding variants and across
    # resumed runs instead of solving the same transport problems repeatedly.
    score_key = rm_ideal_key(row)
    rm_ideal_scores = rm_ideal_cache.get(score_key)
    if rm_ideal_scores is not None:
        rm_ideal_cache.move_to_end(score_key)
        logger.info("[CACHE] Reusing in-memory RM-Ideal scores")
    elif score_key in rm_ideal_sources:
        source_result = rm_ideal_sources[score_key]
        rm_ideal_scores = read_rm_ideal_scores(source_result)
        remember_rm_ideal_scores(rm_ideal_cache, score_key, rm_ideal_scores)
        logger.info("[CACHE] Reusing RM-Ideal scores from %s", source_result)
    else:
        # Passing no output key keeps scores off target_adata.obs. The sigmoid
        # transform matches the repository's existing RM-Ideal workflow.
        niche_query.compute_rm_ideal_score(
            samples=[db_target],
            rm_ideal_output_key=None,
            rm_ideal_post_transform="sigmoid",
            overwrite=True,
        )
        rm_ideal_scores = np.asarray(db_target.rm_ideal_score, dtype=float).reshape(-1)
        remember_rm_ideal_scores(rm_ideal_cache, score_key, rm_ideal_scores)

    rm_ideal_scores = np.asarray(rm_ideal_scores, dtype=float).reshape(-1)
    if rm_ideal_scores.shape[0] != target_adata.n_obs:
        raise ValueError(
            f"query_id={row.query_id}: RM-Ideal score count "
            f"{rm_ideal_scores.shape[0]} does not match target n_obs "
            f"{target_adata.n_obs}"
        )

    # Rank RM-Ideal scores independently from the predicted cosine scores.
    # Rank 1 is the highest RM-Ideal score; target observation order breaks ties.
    rm_ideal_descending_order = np.argsort(-rm_ideal_scores, kind="stable")
    rm_ideal_ranks = np.empty(rm_ideal_scores.shape[0], dtype=np.int64)
    rm_ideal_ranks[rm_ideal_descending_order] = np.arange(
        1, rm_ideal_scores.shape[0] + 1
    )
    # For gene expression, Database may create gene-aligned copies. Copy only the
    # newly constructed masks back to the original AnnData before writing it.
    if source_dirty:
        db_source_adata = db.get_sample(row.source_slice).adata
        source_adata.obs[metadata.query_niche_id] = np.asarray(
            db_source_adata.obs[metadata.query_niche_id], dtype=np.int8
        )
        source_adata.obsm[metadata.niche_name] = np.asarray(
            db_source_adata.obsm[metadata.niche_name], dtype=float
        )

    # Only a newly created source-niche mask is written to H5AD. Query scores are
    # kept out of target_adata.obs and persisted exclusively in the result CSV.
    if source_dirty:
        atomic_write_h5ad(source_adata, source_path)
        logger.info("Persisted new source niche to %s", source_path)

    # query_id is unique across embedding types, niches, and slice pairs, so one
    # file per ID is both unambiguous and easy to resume or process in parallel.
    atomic_write_query_results(
        query_id=row.query_id,
        cell_ids=target_adata.obs_names,
        niche_query_scores=niche_query_scores,
        ranks=ranks,
        rm_ideal_scores=rm_ideal_scores,
        rm_ideal_ranks=rm_ideal_ranks,
        destination=result_path,
    )
    rm_ideal_sources.setdefault(score_key, result_path)

    logger.info(
        "[DONE] query_id=%d embedding=%s niche=%s target=%s niche_query_k=%d result=%s",
        row.query_id,
        row.embedding_type,
        row.query_niche_id,
        row.target_slice,
        row.niche_query_k,
        result_path,
    )
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-manifest", type=Path, default=DEFAULT_QUERY_MANIFEST)
    parser.add_argument("--niche-manifest", type=Path, default=DEFAULT_NICHE_MANIFEST)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--raw-results-dir",
        type=Path,
        default=DEFAULT_RAW_RESULTS_DIR,
        help="Directory for per-query CSV files (default: experiment/raw_results).",
    )
    parser.add_argument(
        "--query-ids",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Optional query_id subset, for example --query-ids 1 2 3. "
            "If omitted or passed without IDs, every manifest row is run."
        ),
    )
    parser.add_argument(
        "--embedding-feature",
        action="append",
        default=[],
        metavar="EMBEDDING_TYPE=FEATURE",
        help=(
            "Override/add a feature mapping. FEATURE is an obsm key, except "
            "gene_expression selects adata.X. May be repeated."
        ),
    )
    parser.add_argument(
        "--overwrite-results",
        action="store_true",
        help="Recompute query CSV files that already exist.",
    )
    parser.add_argument(
        "--knn-backend",
        choices=("auto", "exact", "hnsw"),
        default="auto",
        help=(
            "Neighbor-search backend. auto uses bounded-memory FAISS HNSW for "
            "high-dimensional CPU embeddings and exact search otherwise."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate manifests, mappings, and slice paths without running queries.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    # Make imports work even when this file is launched by absolute path.
    sys.path.insert(0, str(REPO_ROOT))

    niches = load_niche_manifest(args.niche_manifest.resolve())
    # Treat both an omitted parameter (None) and an explicitly empty parameter
    # ([]) as "run all rows in query_manifest.csv".
    selected_ids = set(args.query_ids) if args.query_ids else None
    query_manifest = args.query_manifest.resolve()
    rows = load_query_manifest(query_manifest, niches, selected_ids)
    embedding_features = parse_embedding_overrides(args.embedding_feature)
    data_dir = args.data_dir.resolve()
    raw_results_dir = args.raw_results_dir.resolve()
    validate_inputs(rows, data_dir, embedding_features)
    logger.info(
        "Validated %d query row(s) and %d niche definition(s)", len(rows), len(niches)
    )

    if args.dry_run:
        logger.info("Dry run complete; no AnnData files were loaded or changed")
        return

    completed = 0
    skipped = 0
    # An explicitly overwritten run ignores results from older runs, but still
    # shares newly computed RM-Ideal vectors among rows completed in this run.
    rm_ideal_sources = (
        {}
        if args.overwrite_results
        else find_rm_ideal_sources(
            load_query_manifest(query_manifest, niches, None), raw_results_dir
        )
    )
    rm_ideal_cache: RmIdealCache = OrderedDict()
    logger.info("Found %d reusable RM-Ideal result group(s)", len(rm_ideal_sources))
    # Process in manifest order so log positions and query_id values are easy to
    # cross-reference with query_manifest.csv.
    for position, row in enumerate(rows, start=1):
        print(f"Query manifest row: {row}", flush=True)
        logger.info(
            "[START] row %d/%d query_id=%d embedding=%s source=%s target=%s niche_query_k=%d",
            position,
            len(rows),
            row.query_id,
            row.embedding_type,
            row.source_slice,
            row.target_slice,
            row.niche_query_k,
        )
        if run_one_query(
            row=row,
            metadata=niches[row.query_niche_id],
            data_dir=data_dir,
            raw_results_dir=raw_results_dir,
            feature_name=embedding_features[row.embedding_type],
            overwrite_results=args.overwrite_results,
            knn_backend=args.knn_backend,
            rm_ideal_sources=rm_ideal_sources,
            rm_ideal_cache=rm_ideal_cache,
        ):
            completed += 1
        else:
            skipped += 1

    logger.info("Finished: %d completed, %d skipped", completed, skipped)


if __name__ == "__main__":
    main()
