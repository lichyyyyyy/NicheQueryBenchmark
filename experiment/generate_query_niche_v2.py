"""Compute parcellation composition and diversity for candidate query niches.

Run one slice from the repository root::

    .venv/bin/python experiment/generate_query_niche_v2.py \
        --h5ad-file data/20260601_225717/C57BL6J-638850.28.h5ad \
        --output-dir experiment/niche_metrics --k-hop 2

This writes ``experiment/niche_metrics/C57BL6J-638850.28.csv`` with one row per
center cell. Omit ``--h5ad-file`` to process all H5AD files in the default data
directory, or supply ``--data-dir`` to select another directory. Input H5AD
files are read-only; existing output CSVs are replaced.

CSV columns:
    center_cell_name: Center cell ID from ``adata.obs_names`` (the cell_id index).
    niche_cell_count: Number of cells in the neighborhood (N).
    represented_parcellation_count: Number of represented parcellations (K).
    dominant_parcellation_fraction: Largest parcellation fraction (D).
    parcellation_entropy: Shannon entropy using natural logarithms (H).
    normalized_parcellation_entropy: H / log(K), defined as 0 for K=1.
    parcellation_ids: JSON array of represented parcellation IDs.
    parcellation_cell_counts: JSON array of counts aligned with parcellation_ids.
    parcellation_fractions: JSON array of fractions aligned with parcellation_ids.
    niche_cell_names: JSON array of cell IDs belonging to the neighborhood.

Compute metrics directly for selected positional cell indices::

    metrics = compute_candidate_niche_metrics(
        coordinates=adata.obsm["spatial"],
        labels=adata.obs["parcellation_index"].to_numpy(),
        candidate_centers=[0, 10, 20],
        k_hop=2,
    )

Each result contains the neighborhood indices, represented parcellation IDs,
counts, composition vector ``p``, size ``N``, richness ``K``, dominant fraction
``D``, entropy ``H``, and normalized entropy ``H_norm``. Parcellation IDs, counts,
and fractions have matching positions. No composition filtering is applied.
"""

from __future__ import annotations

from collections.abc import Sequence
import csv
import json
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "20260601_225717"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "query_niche_metrics"


def compute_candidate_niche_metrics(
    *,
    coordinates: Any,
    labels: Any,
    candidate_centers: Sequence[int] | None = None,
    seed_regions: Sequence[Sequence[int]] | None = None,
    k_hop: int = 2,
) -> list[dict[str, Any]]:
    """Expand each center or seed region and compute its niche statistics.

    ``coordinates`` is an (n_cells, >=2) array and ``labels`` is an aligned
    one-dimensional array of integer parcellation IDs. Only the first two
    coordinate columns are used, matching ``generate_query_niche.py``. Every
    label is counted, including 0 if present; missing labels are rejected.

    Supply either positional ``candidate_centers`` or ``seed_regions`` (one
    sequence of positional cell indices per region). If neither is supplied,
    every cell is a candidate center. Seeds themselves belong to the niche;
    overlapping neighborhoods are evaluated independently, with each cell
    counted once within a neighborhood.

    Expansion follows outgoing nearest-neighbor links for ``k_hop`` rounds.
    The graph connects each cell to ``k_hop`` nearest cells, as in the existing
    generator. All cells reached within these hops are included. Small datasets
    use all available neighbors. Zero hops evaluates the seed cells alone.

    Entropy uses natural logarithms. For K=1, H_norm is defined as 0; otherwise
    H_norm = H / log(K). Empty seed regions are rejected. Results preserve
    candidate order and sort neighborhood indices and represented label IDs.
    """
    if isinstance(k_hop, bool) or not isinstance(k_hop, Integral) or k_hop < 0:
        raise ValueError("k_hop must be a non-negative integer")
    points = np.asarray(coordinates, dtype=float)
    parcellations = np.asarray(labels)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError("coordinates must have shape (n_cells, >=2)")
    if not np.isfinite(points[:, :2]).all():
        raise ValueError("Spatial coordinates must be finite")
    if parcellations.ndim != 1 or len(parcellations) != len(points):
        raise ValueError("labels must be one-dimensional and aligned with coordinates")
    if not all(
        isinstance(label, Integral) and not isinstance(label, (bool, np.bool_))
        for label in parcellations
    ):
        raise ValueError(
            "labels must contain integer parcellation IDs without missing values"
        )
    if candidate_centers is not None and seed_regions is not None:
        raise ValueError("Supply candidate_centers or seed_regions, not both")

    n_cells = len(points)
    regions = (
        seed_regions
        if seed_regions is not None
        else (
            [index]
            for index in (
                range(n_cells) if candidate_centers is None else candidate_centers
            )
        )
    )
    seeds = []
    for region in regions:
        indices = list(region)
        if not indices:
            raise ValueError("Seed regions must contain at least one cell")
        if any(
            isinstance(index, (bool, np.bool_))
            or not isinstance(index, Integral)
            or not 0 <= index < n_cells
            for index in indices
        ):
            raise ValueError("Candidate indices must be integers in [0, n_cells)")
        seeds.append(tuple(sorted(set(int(index) for index in indices))))
    if not seeds:
        return []

    degree = min(k_hop, n_cells - 1)
    neighbors = np.empty((n_cells, degree), dtype=np.intp)
    if k_hop > 0 and degree > 0:
        tree = cKDTree(points[:, :2])
        _, raw_neighbors = tree.query(points[:, :2], k=degree + 1, workers=-1)
        for index, row in enumerate(raw_neighbors):
            # Explicit removal also handles coincident cells, where the query
            # point need not be the first result (or appear among tied results).
            neighbors[index] = row[row != index][:degree]

    results = []
    for seed in seeds:
        visited = set(seed)
        frontier = set(seed)
        for _ in range(k_hop):
            frontier = {
                int(neighbor)
                for index in frontier
                for neighbor in neighbors[index]
                if int(neighbor) not in visited
            }
            visited.update(frontier)
            if not frontier:
                break
        niche_indices = sorted(visited)
        ids, counts = np.unique(parcellations[niche_indices], return_counts=True)
        size = len(niche_indices)
        richness = len(ids)
        fractions = counts / size
        entropy = float(-np.sum(fractions * np.log(fractions))) if richness > 1 else 0.0
        results.append(
            {
                "seed_indices": seed,
                "niche_indices": tuple(niche_indices),
                "parcellations": tuple(int(value) for value in ids),
                "counts": tuple(int(value) for value in counts),
                "p": tuple(float(value) for value in fractions),
                "N": size,
                "K": richness,
                "D": float(fractions.max()),
                "H": entropy,
                "H_norm": float(entropy / np.log(richness)) if richness > 1 else 0.0,
            }
        )
    return results


def export_slice_niche_metrics(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    h5ad_file: str | Path | None = None,
    k_hop: int = 2,
    spatial_key: str = "spatial",
    parcellation_key: str = "parcellation_index",
) -> list[Path]:
    """Write one CSV per H5AD slice, evaluating every cell as a niche center.

    Supply ``h5ad_file`` to process only that file instead of ``data_dir``.

    Files are named ``<slice_name>.csv`` using the H5AD filename stem. The first
    column, ``center_cell_name``, is the unique center cell's ``adata.obs_names``
    value. Scalar columns describe niche cell count, represented parcellation
    count, dominant parcellation fraction, entropy, and normalized entropy.
    ``parcellation_ids``, ``parcellation_cell_counts``, and
    ``parcellation_fractions`` are aligned JSON arrays. ``niche_cell_names`` is
    a JSON array of neighborhood members. All cells and parcellations are
    included, with no filtering.

    Read one slice at a time in backed mode to avoid loading expression data.
    Existing output CSVs are replaced atomically only after that slice has
    been computed and written successfully. H5AD inputs are opened read-only.
    Return output paths in sorted input filename order.

    Example::

        paths = export_slice_niche_metrics(k_hop=2)
    """
    import os
    import tempfile

    import anndata as ad

    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    if h5ad_file is not None:
        source_path = Path(h5ad_file)
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if source_path.suffix.lower() != ".h5ad":
            raise ValueError(f"Expected an H5AD file: {source_path}")
        slice_paths = [source_path]
    else:
        if not data_dir.is_dir():
            raise NotADirectoryError(data_dir)
        slice_paths = sorted(data_dir.glob("*.h5ad"))
        if not slice_paths:
            raise FileNotFoundError(f"No H5AD files found in {data_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    scalar_columns = {
        "niche_cell_count": "N",
        "represented_parcellation_count": "K",
        "dominant_parcellation_fraction": "D",
        "parcellation_entropy": "H",
        "normalized_parcellation_entropy": "H_norm",
    }
    array_columns = {
        "parcellation_ids": "parcellations",
        "parcellation_cell_counts": "counts",
        "parcellation_fractions": "p",
    }
    columns = ("center_cell_name", *scalar_columns, *array_columns, "niche_cell_names")
    output_paths = []
    for source_path in slice_paths:
        adata = ad.read_h5ad(source_path, backed="r")
        try:
            if not adata.obs_names.is_unique:
                raise ValueError(f"{source_path}: cell names must be unique")
            if spatial_key not in adata.obsm:
                raise KeyError(f"{source_path}: missing obsm[{spatial_key!r}]")
            if parcellation_key not in adata.obs:
                raise KeyError(f"{source_path}: missing obs[{parcellation_key!r}]")
            cell_names = adata.obs_names.astype(str).tolist()
            metrics = compute_candidate_niche_metrics(
                coordinates=adata.obsm[spatial_key],
                labels=adata.obs[parcellation_key].to_numpy(),
                k_hop=k_hop,
            )
        finally:
            adata.file.close()

        output_path = output_dir / f"{source_path.stem}.csv"
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="",
                dir=output_dir,
                prefix=f".{source_path.stem}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                writer = csv.DictWriter(handle, fieldnames=columns)
                writer.writeheader()
                for cell_name, metric in zip(cell_names, metrics):
                    row = {
                        column: metric[key] for column, key in scalar_columns.items()
                    }
                    row["center_cell_name"] = cell_name
                    for column, key in array_columns.items():
                        row[column] = json.dumps(metric[key])
                    row["niche_cell_names"] = json.dumps(
                        [cell_names[index] for index in metric["niche_indices"]]
                    )
                    writer.writerow(row)
            os.replace(temporary_path, output_path)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
        output_paths.append(output_path)
        # Release the previous slice's results before computing the next one.
        del metrics
    return output_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export per-cell niche metrics for H5AD slices."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    source.add_argument("--h5ad-file", type=Path, help="Process only this H5AD file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k-hop", type=int, default=2)
    args = parser.parse_args()
    for path in export_slice_niche_metrics(
        args.data_dir,
        args.output_dir,
        h5ad_file=args.h5ad_file,
        k_hop=args.k_hop,
    ):
        print(path)
