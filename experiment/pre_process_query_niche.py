"""Compute parcellation composition and diversity for candidate query niches.

Run one slice from the repository root::

    .venv/bin/python experiment/pre_process_query_niche.py \
        --h5ad-file data/20260601_225717/C57BL6J-638850.28.h5ad \
        --output-dir experiment/query_niche_metrics/preprocessed/large --k-hop 10 --niche-size large

This writes ``experiment/query_niche_metrics/C57BL6J-638850.28.csv`` with one row per
center cell whose niche reaches the target size. Omit ``--h5ad-file`` to process all H5AD files in the default data
directory, or supply ``--data-dir`` to select another directory. Input H5AD
files are read-only; existing output CSVs are replaced.
``--niche-size`` selects a name from ``manifests/query_niche_dimensions.json``
(currently ``large`` = 300, ``median`` = 100, ``small`` = 20 cells).

For each candidate, try k=2 through the maximum ``k_hop`` and keep the first
neighborhood containing at least ``niche_size`` cells. Keep the full neighborhood
even if it exceeds the target. If the target is never reached, omit the niche
from the output CSV.

CSV columns:
    center_cell_name: Center cell ID from ``adata.obs_names`` (the cell_id index).
    center_spatial_x: Center cell's first spatial coordinate, in original units.
    center_spatial_y: Center cell's second spatial coordinate, in original units.
    niche_cell_count: Number of cells in the neighborhood (N).
    selected_k_hop: First k reaching the target.
    represented_parcellation_count: Number of represented parcellations (K).
    parcellation_entropy: Shannon entropy using natural logarithms (H).
    normalized_parcellation_entropy: H / log(K), defined as 0 for K=1.
    parcellation_fractions: JSON array of fractions aligned with parcellation_ids.
    niche_cell_names: JSON array of cell IDs belonging to the neighborhood.

Compute metrics directly for selected positional cell indices::

    metrics = compute_candidate_niche_metrics(
        coordinates=adata.obsm["spatial"],
        labels=adata.obs["parcellation_index"].to_numpy(),
        candidate_centers=[0, 10, 20],
        k_hop=10,
        niche_size=50,
    )

Each result contains the neighborhood indices, shared ordered parcellation IDs,
counts, composition vector ``p``, size ``N``, richness ``K``, dominant fraction
``D``, entropy ``H``, and normalized entropy ``H_norm``. Parcellation IDs, counts,
and fractions have matching positions, including zeros for absent types.
``parcellation_fractions`` is the composition vector: entry i is the fraction
of cells with ID ``parcellation_ids[i]``. Export uses the sorted union of IDs
across the input directory (including sibling slices for ``--h5ad-file``).
The shared IDs are saved once as a JSON array in
``<output-dir>/parcellation_ids.json``, rather than repeated in each CSV row.
Use ``--parcellation-order`` to fix the axis across different datasets or runs
whose input files change. Pass ``--composition-complexity`` to retain only
niches matching a manifest rule; by default no composition filtering is applied.
"""

from __future__ import annotations

from collections.abc import Sequence
import csv
import json
from numbers import Integral
from pathlib import Path
from typing import Any

if __package__:
    from .query_niche_dimensions import COMPOSITION_COMPLEXITY, QUERY_NICHE_DIMENSIONS
else:
    from query_niche_dimensions import COMPOSITION_COMPLEXITY, QUERY_NICHE_DIMENSIONS

import numpy as np
from scipy.spatial import cKDTree

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "20260601_225717"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "query_niche_metrics"


def matches_composition_complexity(
    metric: dict[str, Any], rule: dict[str, Any]
) -> bool:
    """Return whether a computed niche satisfies a manifest complexity rule."""
    count = int(metric["K"])
    dominant = float(metric["D"])
    if (
        "parcellation_count_less_than" in rule
        and not count < rule["parcellation_count_less_than"]
    ):
        return False
    if (
        "parcellation_count_at_least" in rule
        and not count >= rule["parcellation_count_at_least"]
    ):
        return False
    if (
        "parcellation_count_at_most" in rule
        and not count <= rule["parcellation_count_at_most"]
    ):
        return False
    if (
        "dominant_parcellation_fraction_at_least" in rule
        and not dominant >= rule["dominant_parcellation_fraction_at_least"]
    ):
        return False
    if (
        "dominant_parcellation_fraction_at_most" in rule
        and not dominant <= rule["dominant_parcellation_fraction_at_most"]
    ):
        return False
    return True


def _integer_parcellations(values: Any) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim != 1 or not all(
        isinstance(label, Integral) and not isinstance(label, (bool, np.bool_))
        for label in labels
    ):
        raise ValueError(
            "labels must contain one-dimensional integer parcellation IDs without missing values"
        )
    return labels


def _composition_order(labels: Any, order: Sequence[int] | None) -> tuple[int, ...]:
    present = set(int(value) for value in _integer_parcellations(labels))
    if order is None:
        return tuple(sorted(present))
    axis = tuple(int(value) for value in _integer_parcellations(order))
    if len(axis) != len(set(axis)):
        raise ValueError("parcellation_order must not contain duplicate IDs")
    missing = present - set(axis)
    if missing:
        raise ValueError(
            f"parcellation_order is missing observed IDs: {sorted(missing)}"
        )
    return axis


def compute_candidate_niche_metrics(
    *,
    coordinates: Any,
    labels: Any,
    niche_size: int,
    candidate_centers: Sequence[int] | None = None,
    seed_regions: Sequence[Sequence[int]] | None = None,
    k_hop: int = 2,
    parcellation_order: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Expand each center or seed region and compute its niche statistics.

    ``coordinates`` is an (n_cells, >=2) array and ``labels`` is an aligned
    one-dimensional array of integer parcellation IDs. Only the first two
    coordinate columns are used, matching ``generate_query_niche.py``. Every
    label is counted, including 0 if present; missing labels are rejected.
    ``parcellation_order`` defines the shared vector axis and must include all
    observed labels. By default it is the sorted union of input labels. Pass
    the same order to separate calls to compare their vectors directly.
    Absent types have zero counts and fractions; K counts only present types.

    Supply either positional ``candidate_centers`` or ``seed_regions`` (one
    sequence of positional cell indices per region). If neither is supplied,
    every cell is a candidate center. Seeds themselves belong to the niche;
    overlapping neighborhoods are evaluated independently, with each cell
    counted once within a neighborhood.

    Try k=2 through ``k_hop`` inclusive, stopping independently for each seed
    when its neighborhood contains at least ``niche_size`` cells. At each k,
    follow outgoing links to k nearest cells for k rounds, matching the existing
    generator's convention. All reached cells are included without truncation.
    Small datasets use all available neighbors. If the target is unreachable,
    return the maximum-k neighborhood with ``target_size_reached=False``.

    Entropy uses natural logarithms. For K=1, H_norm is defined as 0; otherwise
    H_norm = H / log(K). Empty seed regions are rejected. Results preserve
    candidate order and sort neighborhood indices.
    """
    if isinstance(k_hop, bool) or not isinstance(k_hop, Integral) or k_hop < 2:
        raise ValueError("k_hop must be an integer at least 2 (the maximum k)")
    if (
        isinstance(niche_size, bool)
        or not isinstance(niche_size, Integral)
        or niche_size < 1
    ):
        raise ValueError("niche_size must be a positive integer")
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
    composition_ids = _composition_order(parcellations, parcellation_order)
    composition_index = {label: index for index, label in enumerate(composition_ids)}

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
        for selected_k in range(2, k_hop + 1):
            visited = set(seed)
            frontier = set(seed)
            for _ in range(selected_k):
                frontier = {
                    int(neighbor)
                    for index in frontier
                    for neighbor in neighbors[index, :selected_k]
                    if int(neighbor) not in visited
                }
                visited.update(frontier)
                if not frontier:
                    break
            if len(visited) >= niche_size:
                break
        niche_indices = sorted(visited)
        ids, counts = np.unique(parcellations[niche_indices], return_counts=True)
        size = len(niche_indices)
        richness = len(ids)
        fractions = counts / size
        entropy = float(-np.sum(fractions * np.log(fractions))) if richness > 1 else 0.0
        aligned_counts = np.zeros(len(composition_ids), dtype=np.int64)
        for label, count in zip(ids, counts):
            aligned_counts[composition_index[int(label)]] = count
        aligned_fractions = aligned_counts / size
        results.append(
            {
                "seed_indices": seed,
                "niche_indices": tuple(niche_indices),
                "parcellations": composition_ids,
                "counts": tuple(int(value) for value in aligned_counts),
                "p": tuple(float(value) for value in aligned_fractions),
                "N": size,
                "target_niche_size": int(niche_size),
                "selected_k_hop": selected_k,
                "max_k_hop": int(k_hop),
                "target_size_reached": size >= niche_size,
                "K": richness,
                "D": float(fractions.max()),
                "H": entropy,
                "H_norm": float(entropy / np.log(richness)) if richness > 1 else 0.0,
            }
        )
    return results


def export_slice_niche_metrics(
    data_dir: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path | None = None,
    *,
    niche_size: str,
    h5ad_file: str | Path | None = None,
    k_hop: int = 2,
    spatial_key: str = "spatial",
    parcellation_key: str = "parcellation_index",
    parcellation_order: Sequence[int] | None = None,
    composition_complexity: str | None = None,
) -> list[Path]:
    """Write one CSV per H5AD slice, evaluating every cell as a niche center.

    Supply ``h5ad_file`` to process only that file instead of ``data_dir``.
    ``k_hop`` is the maximum k; ``niche_size`` is a dimension name from
    ``manifests/query_niche_dimensions.json``, resolved to a minimum cell count.
    CSV rows include the selected k.
    Niches with ``target_size_reached=False`` are omitted from the CSV.
    By default, outputs go under ``query_niche_metrics/<dimension>`` using the
    dimensions manifest.

    Files are named ``<slice_name>.csv`` using the H5AD filename stem. The first
    column, ``center_cell_name``, is the unique center cell's ``adata.obs_names``
    value. ``center_spatial_x`` and ``center_spatial_y`` contain its coordinates
    from ``adata.obsm[spatial_key]`` in original units. Scalar columns describe niche cell
    count, represented parcellation count,
    entropy, and normalized entropy.
    ``parcellation_fractions`` is a JSON array aligned with the shared
    ``parcellation_ids.json`` in the output directory.
    ``niche_cell_names`` is
    a JSON array of neighborhood members. All cells and parcellations within
    retained niches are included; by default no composition filtering is applied.
    Each retained niche is also printed with its center cell name, first two
    spatial coordinates, and nonzero parcellation fractions.
    All rows share one composition axis, with zeros for absent types. By
    default, scan labels across all slices in ``data_dir``, or all H5AD siblings
    of ``h5ad_file`` for single-file exports. ``parcellation_order`` overrides
    this scan with an explicit order that must include every exported label.
    ``composition_complexity`` optionally filters retained niches using a
    manifest rule; when omitted, all niches reaching the target are retained.
    An existing ``parcellation_ids.json`` must match the requested order to
    prevent changing the interpretation of previously exported vectors.

    Read one slice at a time in backed mode to avoid loading expression data.
    Existing output CSVs are replaced atomically only after that slice has
    been computed and written successfully. H5AD inputs are opened read-only.
    Return CSV paths in sorted input filename order.

    Example::

        paths = export_slice_niche_metrics(k_hop=10, niche_size="large")
    """
    import os
    import tempfile

    import anndata as ad

    if not isinstance(niche_size, str) or niche_size not in QUERY_NICHE_DIMENSIONS:
        raise ValueError(
            f"niche_size must be one of: {', '.join(QUERY_NICHE_DIMENSIONS)}"
        )
    target_cell_count = QUERY_NICHE_DIMENSIONS[niche_size]
    if (
        composition_complexity is not None
        and composition_complexity not in COMPOSITION_COMPLEXITY
    ):
        raise ValueError(
            f"composition_complexity must be one of: {', '.join(COMPOSITION_COMPLEXITY)}"
        )
    complexity_rule = (
        COMPOSITION_COMPLEXITY[composition_complexity]
        if composition_complexity is not None
        else None
    )
    data_dir = Path(data_dir)
    if output_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR / niche_size
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
    if parcellation_order is None:
        axis_paths = (
            sorted(set(slice_paths) | set(slice_paths[0].parent.glob("*.h5ad")))
            if h5ad_file is not None
            else slice_paths
        )
        all_ids = set()
        for axis_path in axis_paths:
            axis_data = ad.read_h5ad(axis_path, backed="r")
            try:
                if parcellation_key not in axis_data.obs:
                    raise KeyError(f"{axis_path}: missing obs[{parcellation_key!r}]")
                all_ids.update(_integer_parcellations(axis_data.obs[parcellation_key]))
            finally:
                axis_data.file.close()
        parcellation_order = tuple(sorted(int(value) for value in all_ids))
    else:
        parcellation_order = _composition_order([], parcellation_order)
    output_dir.mkdir(parents=True, exist_ok=True)
    ids_path = output_dir / "parcellation_ids.json"
    if ids_path.exists():
        with ids_path.open(encoding="utf-8") as handle:
            existing_order = _composition_order([], json.load(handle))
        if existing_order != parcellation_order:
            raise ValueError(
                f"{ids_path}: existing parcellation order differs; use a new "
                "output directory to regenerate vectors with a different order"
            )
    else:
        temporary_ids_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output_dir,
                prefix=".parcellation_ids.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_ids_path = Path(handle.name)
                json.dump(parcellation_order, handle, indent=2)
                handle.write("\n")
            os.replace(temporary_ids_path, ids_path)
        finally:
            if temporary_ids_path is not None:
                temporary_ids_path.unlink(missing_ok=True)
    scalar_columns = {
        "niche_cell_count": "N",
        "selected_k_hop": "selected_k_hop",
        "represented_parcellation_count": "K",
        "parcellation_entropy": "H",
        "normalized_parcellation_entropy": "H_norm",
    }
    array_columns = {
        "parcellation_fractions": "p",
    }
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
            coordinates = np.asarray(adata.obsm[spatial_key], dtype=float).copy()
            metrics = compute_candidate_niche_metrics(
                coordinates=coordinates,
                labels=adata.obs[parcellation_key].to_numpy(),
                k_hop=k_hop,
                niche_size=target_cell_count,
                parcellation_order=parcellation_order,
            )
        finally:
            adata.file.close()

        coordinate_columns = ["center_spatial_x", "center_spatial_y"]
        columns = (
            "center_cell_name",
            *coordinate_columns,
            *scalar_columns,
            *array_columns,
            "niche_cell_names",
        )
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
                for cell_name, center_coordinates, metric in zip(
                    cell_names, coordinates, metrics
                ):
                    if not metric["target_size_reached"]:
                        continue
                    if (
                        complexity_rule is not None
                        and not matches_composition_complexity(metric, complexity_rule)
                    ):
                        continue
                    parcellation_fractions = [
                        f"{parcellation_id}: {fraction:.6f}"
                        for parcellation_id, fraction in zip(
                            metric["parcellations"], metric["p"]
                        )
                        if fraction > 0.0
                    ]
                    print(
                        f"Cell {cell_name}: "
                        f"coordinate=({center_coordinates[0]:.6f}, "
                        f"{center_coordinates[1]:.6f}); "
                        f"parcellations={{{', '.join(parcellation_fractions)}}}"
                    )
                    row = {
                        column: metric[key] for column, key in scalar_columns.items()
                    }
                    row["center_cell_name"] = cell_name
                    row.update(zip(coordinate_columns, center_coordinates))
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
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: query_niche_metrics/<dimension>)",
    )
    parser.add_argument(
        "--k-hop", type=int, default=10, help="Maximum k to try (at least 2)."
    )
    parser.add_argument(
        "--niche-size",
        type=str,
        choices=list(QUERY_NICHE_DIMENSIONS),
        required=True,
        help="Target dimension from manifests/query_niche_dimensions.json.",
    )
    parser.add_argument(
        "--parcellation-order",
        type=int,
        nargs="+",
        help="Shared composition-vector IDs in order (default: sorted IDs across sibling slices).",
    )
    parser.add_argument(
        "--composition-complexity",
        choices=list(COMPOSITION_COMPLEXITY),
        help="Optional composition rule from the manifest (default: keep all complexities).",
    )
    args = parser.parse_args()
    for path in export_slice_niche_metrics(
        args.data_dir,
        args.output_dir,
        h5ad_file=args.h5ad_file,
        k_hop=args.k_hop,
        niche_size=args.niche_size,
        parcellation_order=args.parcellation_order,
        composition_complexity=args.composition_complexity,
    ):
        print(path)
