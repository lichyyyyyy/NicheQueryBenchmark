"""Find and persist a query niche with a requested parcellation composition.

Every selected parcellation must occur in all 12 slice H5AD files.  Every cell in
the source slice is randomly ordered as a candidate center, except cells already
recorded in the manifest's ``center_cell`` column.  For each candidate, the script
constructs the same filtered k-hop neighborhood used by
``Niche.construct_by_k_hop`` and stops at the first neighborhood for which every
parcellation fraction is within ``delta`` of its requested proportion.

Examples
--------
Request an approximately even two-parcellation niche::

    PYTHONPATH=. python experiment/generate_query_niche.py \
        --source-slice Zhuang-ABCA-1.096 \
        --parcellations 207 123 \
        --proportions 0.5 0.5 \
        --delta 0.01

Omit ``--parcellations`` to randomly search cells.  A cell is tested as a center
only when its parcellation has at least 100 cells in every one of the 12 slices.
For multiple requested proportions, every ordered selection of remaining
parcellations from the same globally valid set is tested for each center::

    PYTHONPATH=. python experiment/generate_query_niche.py \
        --source-slice Zhuang-ABCA-1.096 \
        --proportions 0.5 0.5 \
        --delta 0.01

Percent values are also accepted (``--proportions 50 50``).  On success the
script writes a binary membership mask to both
``adata.obs[query_niche_id]`` and ``adata.obsm[niche_name]``, atomically replaces
the source H5AD, and appends one row to ``query_niche_manifest.csv``.
Use ``--exclude-parcellations 997 8`` to prevent specific IDs from being used.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "20260601_225717"
DEFAULT_MANIFEST = Path(__file__).with_name("manifests") / "query_niche_manifest.csv"
DEFAULT_PARCELLATION_STRUCTURE = (
    REPO_ROOT / "data" / "ccf_parcellation" / "structure.json"
)
MANIFEST_COLUMNS = (
    "query_niche_id",
    "source_slice",
    "niche_name",
    "center_cell",
    "k_hop",
    "cells",
    "parcellations_in_niche",
    "parcellation_composition",
    "parcellation_names",
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Match:
    center_index: int
    parcellations: tuple[int, ...]
    niche_indices: tuple[int, ...]
    counts: tuple[int, ...]
    fractions: tuple[float, ...]
    max_error: float


def normalize_proportions(values: Sequence[float]) -> tuple[float, ...]:
    """Return fractions after accepting values that sum to either 1 or 100."""
    if not values or any(value < 0 for value in values):
        raise ValueError("Proportions must be a non-empty list of non-negative values")
    total = float(sum(values))
    if abs(total - 1.0) <= 1e-9:
        normalized = tuple(float(value) for value in values)
    elif abs(total - 100.0) <= 1e-7:
        normalized = tuple(float(value) / 100.0 for value in values)
    else:
        raise ValueError(
            f"Proportions must sum to 1 or 100; received a sum of {total:g}"
        )
    if any(value > 1 for value in normalized):
        raise ValueError("Each normalized proportion must be between 0 and 1")
    return normalized


def validate_request(
    parcellations: Sequence[int], proportions: Sequence[float], delta: float
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    if not parcellations:
        raise ValueError("At least one parcellation is required")
    parsed_parcellations = tuple(int(value) for value in parcellations)
    if len(set(parsed_parcellations)) != len(parsed_parcellations):
        raise ValueError("Parcellation IDs must be unique")
    if any(value <= 0 for value in parsed_parcellations):
        raise ValueError("Parcellation IDs must be positive (0 means unassigned)")
    normalized = normalize_proportions(proportions)
    if len(parsed_parcellations) != len(normalized):
        raise ValueError(
            "--parcellations and --proportions must contain the same number of values"
        )
    if not 0 <= delta <= 1:
        raise ValueError("--delta must be between 0 and 1")
    return parsed_parcellations, normalized


def parcellation_coverage_across_slices(
    *,
    data_dir: Path,
    source_path: Path,
    source_labels: Any,
    expected_slice_count: int,
    min_parcellation_cells: int,
) -> tuple[set[int], set[int]]:
    """Return IDs present in all slices and IDs meeting the count in all."""
    import anndata as ad
    import numpy as np

    slice_paths = sorted(data_dir.glob("*.h5ad"))
    if len(slice_paths) != expected_slice_count:
        raise ValueError(
            f"Expected {expected_slice_count} slice H5AD files in {data_dir}, "
            f"but found {len(slice_paths)}"
        )
    source_resolved = source_path.resolve()
    common_present: set[int] | None = None
    common_populated: set[int] | None = None
    for path in slice_paths:
        if path.resolve() == source_resolved:
            labels = source_labels
        else:
            backed = ad.read_h5ad(path, backed="r")
            try:
                if "parcellation_index" not in backed.obs:
                    raise KeyError(f"{path} has no obs['parcellation_index']")
                labels = np.asarray(
                    backed.obs["parcellation_index"].astype(int), dtype=np.int64
                )
            finally:
                backed.file.close()
        parcellation_ids, counts = np.unique(labels, return_counts=True)
        present = {int(value) for value in parcellation_ids if int(value) > 0}
        populated = {
            int(value)
            for value, count in zip(parcellation_ids, counts)
            if int(value) > 0 and int(count) >= min_parcellation_cells
        }
        common_present = present if common_present is None else common_present & present
        common_populated = (
            populated if common_populated is None else common_populated & populated
        )

    assert common_present is not None and common_populated is not None
    logger.info(
        "%d parcellations occur in all %d slices; %d have at least %d cells "
        "in every slice",
        len(common_present),
        len(slice_paths),
        len(common_populated),
        min_parcellation_cells,
    )
    return common_present, common_populated


def composition(
    niche_indices: Sequence[int], labels: Any, parcellations: Sequence[int]
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    counts = tuple(
        sum(1 for index in niche_indices if int(labels[index]) == parcellation)
        for parcellation in parcellations
    )
    total = len(niche_indices)
    fractions = tuple(count / total if total else 0.0 for count in counts)
    return counts, fractions


def find_matching_niche(
    *,
    coordinates: Any,
    labels: Any,
    parcellations: Sequence[int] | None,
    valid_center_parcellations: set[int],
    proportions: Sequence[float],
    delta: float,
    k_hop: int,
    cell_limit: int,
    progress_every: int,
    random_seed: int | None,
    cell_ids: Sequence[Any],
    excluded_center_ids: set[str],
) -> tuple[Match | None, Match | None]:
    """Return the first valid match and the closest attempted neighborhood."""
    from itertools import permutations

    import numpy as np
    from scipy.spatial import cKDTree

    # Search every source cell exactly once, in randomized order.  Supplying a
    # seed makes the first-valid result reproducible; omitting it uses fresh
    # operating-system entropy for each run.
    eligible_indices = np.asarray(
        [
            index
            for index, cell_id in enumerate(cell_ids)
            if str(cell_id) not in excluded_center_ids
        ],
        dtype=np.int64,
    )
    if eligible_indices.size == 0:
        raise ValueError(
            "Every cell in the source slice is already listed as a center_cell "
            "in the query niche manifest"
        )
    rng = np.random.default_rng(random_seed)
    candidate_indices = rng.permutation(eligible_indices).tolist()
    logger.info(
        "Randomly searching %d eligible center cells (%d manifest centers excluded)",
        len(candidate_indices),
        len(labels) - len(candidate_indices),
    )

    logger.info(
        "Building the spatial graph once for %d cells (k=%d)",
        len(labels),
        k_hop,
    )
    if len(labels) <= k_hop:
        raise ValueError(
            f"--k-hop ({k_hop}) must be smaller than the number of cells "
            f"({len(labels)})"
        )
    # ``Niche.construct_by_k_hop`` uses the same value for the number of nearest
    # neighbors and the number of graph hops.  Query k+1 points so that removing
    # the point itself leaves k neighbors.  Keeping this adjacency in a dense
    # integer array is considerably lighter than a PyG edge tensor and does not
    # require the optional pyg-lib/torch-cluster packages.
    tree = cKDTree(np.asarray(coordinates[:, :2], dtype=float))
    _, raw_neighbors = tree.query(coordinates[:, :2], k=k_hop + 1, workers=-1)
    neighbors = np.empty((len(labels), k_hop), dtype=np.int64)
    for index, row in enumerate(np.atleast_2d(raw_neighbors)):
        without_self = [int(value) for value in row if int(value) != index]
        if len(without_self) < k_hop:
            raise RuntimeError(f"Could not find {k_hop} neighbors for cell {index}")
        neighbors[index] = without_self[:k_hop]
    best: Match | None = None
    tested_arrangements = 0

    for attempt, center_index in enumerate(candidate_indices, start=1):
        if parcellations is None:
            center_parcellation = int(labels[center_index])
            if center_parcellation not in valid_center_parcellations:
                if attempt % progress_every == 0:
                    logger.info(
                        "[%d/%d] center_index=%d skipped: parcellation %d is "
                        "not globally valid",
                        attempt,
                        len(candidate_indices),
                        center_index,
                        center_parcellation,
                    )
                continue
            remaining_count = len(proportions) - 1
            remaining_pool = sorted(valid_center_parcellations - {center_parcellation})
            if len(remaining_pool) < remaining_count:
                raise ValueError(
                    f"Need {len(proportions)} globally valid parcellations, but "
                    f"only {len(valid_center_parcellations)} are available"
                )
            # Order matters because each parcellation is compared with the
            # proportion at the same position.  Therefore, try permutations
            # rather than only unordered combinations.
            trial_options = (
                (center_parcellation, *remaining)
                for remaining in permutations(remaining_pool, remaining_count)
            )
        else:
            trial_options = iter((tuple(parcellations),))

        visited = {center_index}
        frontier = {center_index}
        for _ in range(k_hop):
            frontier = {
                int(neighbor)
                for index in frontier
                for neighbor in neighbors[index]
                if int(neighbor) not in visited
            }
            if not frontier:
                break
            visited.update(frontier)
        arrangements_for_center = 0
        for trial_parcellations in trial_options:
            tested_arrangements += 1
            arrangements_for_center += 1
            allowed = set(trial_parcellations)
            # This order and truncation match Niche.construct_by_k_hop.
            indices = [
                index for index in sorted(visited) if int(labels[index]) in allowed
            ][:cell_limit]
            counts, fractions = composition(indices, labels, trial_parcellations)
            errors = tuple(
                abs(actual - requested)
                for actual, requested in zip(fractions, proportions)
            )
            candidate = Match(
                center_index=center_index,
                parcellations=trial_parcellations,
                niche_indices=tuple(indices),
                counts=counts,
                fractions=fractions,
                max_error=max(errors),
            )
            if best is None or candidate.max_error < best.max_error:
                best = candidate

            # ``cell_limit`` is the requested niche size, not merely an upper
            # bound for benchmark generation.  Do not persist undersized
            # neighborhoods even when their proportions happen to match.
            matched = len(indices) == cell_limit and all(
                error <= delta + 1e-12 for error in errors
            )
            if matched:
                details = ", ".join(
                    f"{parcellation}={count} ({fraction:.3%})"
                    for parcellation, count, fraction in zip(
                        trial_parcellations, counts, fractions
                    )
                )
                logger.info(
                    "[%d/%d] center_index=%d n=%d %s",
                    attempt,
                    len(candidate_indices),
                    center_index,
                    len(indices),
                    details,
                )
                return candidate, best

        if attempt % progress_every == 0:
            logger.info(
                "[%d/%d] center_index=%d tried %d parcellation arrangements (%d total)",
                attempt,
                len(candidate_indices),
                center_index,
                arrangements_for_center,
                tested_arrangements,
            )

    return None, best


def load_parcellation_names(path: Path) -> dict[int, str]:
    """Load CCF names, preferring structure IDs over graph-order fallbacks."""
    if not path.is_file():
        raise FileNotFoundError(f"CCF parcellation structure not found: {path}")
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)

    names_by_id: dict[int, str] = {}
    names_by_graph_order: dict[int, str] = {}
    stack = list(data.get("msg") or [])
    while stack:
        node = stack.pop()
        name = str(node.get("name") or "").strip()
        if name:
            if "id" in node:
                names_by_id[int(node["id"])] = name
            if "graph_order" in node:
                names_by_graph_order[int(node["graph_order"])] = name
        stack.extend(node.get("children") or [])
    # Some benchmark H5ADs contain graph_order values in parcellation_index for
    # structures whose IDs are absent from this JSON.  Start with graph-order
    # names, then overwrite collisions with true structure IDs.
    names = dict(names_by_graph_order)
    names.update(names_by_id)
    if not names:
        raise ValueError(f"No parcellation names found in {path}")
    return names


def names_for_parcellations(
    path: Path, parcellations: Sequence[int]
) -> tuple[str, ...]:
    """Resolve every requested ID, failing rather than writing incorrect names."""
    name_by_id = load_parcellation_names(path)
    missing = [value for value in parcellations if value not in name_by_id]
    if missing:
        raise KeyError(f"Parcellation IDs missing from {path}: {missing}")
    return tuple(name_by_id[value] for value in parcellations)


def read_manifest_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise ValueError(
                f"Unexpected manifest columns in {path}: {reader.fieldnames!r}"
            )
        return list(reader)


def append_manifest_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        if needs_header:
            # Existing manifests use an unquoted header and fully quoted rows.
            csv.writer(handle, lineterminator="\n").writerow(MANIFEST_COLUMNS)
        writer = csv.DictWriter(
            handle,
            fieldnames=MANIFEST_COLUMNS,
            quoting=csv.QUOTE_ALL,
            lineterminator="\n",
        )
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def atomic_write_h5ad(adata: Any, destination: Path) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".h5ad", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        adata.write_h5ad(temporary)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-slice", required=True)
    parser.add_argument(
        "--parcellations",
        type=int,
        nargs="+",
        help=(
            "Parcellation IDs; when omitted, randomly inspect cells and use each "
            "eligible cell's parcellation as the first niche parcellation"
        ),
    )
    parser.add_argument(
        "--exclude-parcellations",
        type=int,
        nargs="+",
        default=(),
        metavar="ID",
        help=(
            "Parcellation IDs that cannot be used as center or remaining niche "
            "parcellations"
        ),
    )
    parser.add_argument(
        "--proportions",
        "--percentages",
        dest="proportions",
        type=float,
        nargs="+",
        required=True,
        help="Desired composition, summing to 1 (0.5 0.5) or 100 (50 50)",
    )
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--k-hop", type=int, default=10)
    parser.add_argument("--cell-limit", type=int, default=100)
    parser.add_argument(
        "--min-parcellation-cells",
        type=int,
        default=100,
        help=(
            "In automatic mode, a center parcellation must have at least this "
            "many cells in every slice (default: 100)"
        ),
    )
    parser.add_argument(
        "--expected-slice-count",
        type=int,
        default=12,
        help="Required number of top-level slice H5AD files (default: 12)",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional seed for reproducible center-cell ordering",
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--parcellation-structure",
        type=Path,
        default=DEFAULT_PARCELLATION_STRUCTURE,
        help="CCF structure JSON used to populate parcellation_names",
    )
    parser.add_argument(
        "--niche-name",
        help="Default: query_niche_<parcellation IDs joined by underscores>",
    )
    parser.add_argument("--query-niche-id", help="Default: <source_slice>_<niche_name>")
    parser.add_argument(
        "--parcellation-names",
        nargs="+",
        help="Optional names in the same order as --parcellations",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (
        args.k_hop <= 0
        or args.cell_limit <= 0
        or args.progress_every <= 0
        or args.min_parcellation_cells < 0
        or args.expected_slice_count <= 0
    ):
        raise ValueError(
            "--k-hop, --cell-limit, and --progress-every must be positive; "
            "--expected-slice-count must be positive; "
            "--min-parcellation-cells must be non-negative"
        )
    proportions = normalize_proportions(args.proportions)
    excluded_parcellations = set(args.exclude_parcellations)
    if any(value <= 0 for value in excluded_parcellations):
        raise ValueError(
            "--exclude-parcellations IDs must be positive (0 is already unassigned)"
        )
    if not 0 <= args.delta <= 1:
        raise ValueError("--delta must be between 0 and 1")
    if args.parcellations is None and args.parcellation_names is not None:
        raise ValueError(
            "--parcellation-names can only be used when --parcellations is given"
        )

    manifest_rows = read_manifest_rows(args.manifest)
    excluded_center_ids = {
        (row.get("center_cell") or "").strip()
        for row in manifest_rows
        if (row.get("center_cell") or "").strip()
    }

    source_path = args.data_dir / f"{args.source_slice}.h5ad"
    if not source_path.is_file():
        raise FileNotFoundError(f"Source slice not found: {source_path}")

    import anndata as ad
    import numpy as np

    logger.info("Loading %s", source_path)
    adata = ad.read_h5ad(source_path)
    if "parcellation_index" not in adata.obs:
        raise KeyError(f"{source_path} has no obs['parcellation_index']")
    if "spatial" not in adata.obsm:
        raise KeyError(f"{source_path} has no obsm['spatial']")
    labels = np.asarray(adata.obs["parcellation_index"].astype(int), dtype=np.int64)
    coordinates = np.asarray(adata.obsm["spatial"], dtype=float)
    shared_parcellations, valid_center_parcellations = (
        parcellation_coverage_across_slices(
            data_dir=args.data_dir,
            source_path=source_path,
            source_labels=labels,
            expected_slice_count=args.expected_slice_count,
            min_parcellation_cells=args.min_parcellation_cells,
        )
    )
    shared_parcellations -= excluded_parcellations
    valid_center_parcellations -= excluded_parcellations
    structure_names: dict[int, str] | None = None
    if args.parcellation_names is None:
        # Resolve names before the expensive center search so automatic mode
        # cannot find a niche and only then fail during manifest creation.
        structure_names = load_parcellation_names(args.parcellation_structure)
        unresolved_valid = valid_center_parcellations - structure_names.keys()
        if unresolved_valid:
            logger.warning(
                "Ignoring globally valid parcellations with no CCF name: %s",
                sorted(unresolved_valid),
            )
            valid_center_parcellations -= unresolved_valid
    if excluded_parcellations:
        logger.info(
            "Excluded parcellations from niche generation: %s",
            sorted(excluded_parcellations),
        )

    if args.parcellations is None:
        parcellations = None
        if len(valid_center_parcellations) < len(proportions):
            raise ValueError(
                f"Need {len(proportions)} parcellations with at least "
                f"{args.min_parcellation_cells} cells in every slice, but only "
                f"{len(valid_center_parcellations)} qualify"
            )
    else:
        parcellations, proportions = validate_request(
            args.parcellations, proportions, args.delta
        )
        explicitly_excluded = set(parcellations) & excluded_parcellations
        if explicitly_excluded:
            raise ValueError(
                "Requested --parcellations also listed in "
                f"--exclude-parcellations: {sorted(explicitly_excluded)}"
            )
        missing_from_some_slice = set(parcellations) - shared_parcellations
        if missing_from_some_slice:
            raise ValueError(
                "These requested parcellations do not occur in every slice: "
                f"{sorted(missing_from_some_slice)}"
            )
        if structure_names is not None:
            missing_names = set(parcellations) - structure_names.keys()
            if missing_names:
                raise KeyError(
                    f"Parcellation IDs missing from {args.parcellation_structure}: "
                    f"{sorted(missing_names)}"
                )
    if (
        args.parcellation_names is not None
        and parcellations is not None
        and len(args.parcellation_names) != len(parcellations)
    ):
        raise ValueError("--parcellation-names must contain one value per parcellation")

    existing_niche_ids = {
        (row.get("query_niche_id") or "").strip() for row in manifest_rows
    }
    if parcellations is not None:
        niche_name = args.niche_name or "query_niche_" + "_".join(
            str(value) for value in parcellations
        )
        query_niche_id = args.query_niche_id or f"{args.source_slice}_{niche_name}"
        if query_niche_id in existing_niche_ids:
            raise ValueError(
                f"query_niche_id {query_niche_id!r} already exists in {args.manifest}"
            )

    match, nearest = find_matching_niche(
        coordinates=coordinates,
        labels=labels,
        parcellations=parcellations,
        valid_center_parcellations=valid_center_parcellations,
        proportions=proportions,
        delta=args.delta,
        k_hop=args.k_hop,
        cell_limit=args.cell_limit,
        progress_every=args.progress_every,
        random_seed=args.seed,
        cell_ids=adata.obs_names,
        excluded_center_ids=excluded_center_ids,
    )
    if match is None:
        if nearest is None:
            logger.error(
                "No unused cell had a globally valid center parcellation. "
                "No files were changed."
            )
            return 1
        details = ", ".join(
            f"{parcellation}={fraction:.3%} (wanted {wanted:.3%})"
            for parcellation, fraction, wanted in zip(
                nearest.parcellations, nearest.fractions, proportions
            )
        )
        center_id = str(adata.obs_names[nearest.center_index])
        logger.error(
            "No center met delta %.3g. Closest center=%s, n=%d: %s. "
            "No files were changed.",
            args.delta,
            center_id,
            len(nearest.niche_indices),
            details,
        )
        return 1

    parcellations = match.parcellations
    niche_name = args.niche_name or "query_niche_" + "_".join(
        str(value) for value in parcellations
    )
    query_niche_id = args.query_niche_id or f"{args.source_slice}_{niche_name}"
    if query_niche_id in existing_niche_ids:
        raise ValueError(
            f"query_niche_id {query_niche_id!r} already exists in {args.manifest}"
        )

    center_id = str(adata.obs_names[match.center_index])
    mask = np.zeros(adata.n_obs, dtype=np.int8)
    mask[list(match.niche_indices)] = 1
    if query_niche_id in adata.obs:
        raise ValueError(f"obs[{query_niche_id!r}] already exists in {source_path}")
    if niche_name in adata.obsm:
        logger.warning(
            "Replacing unmanifested obsm[%r] in %s with the newly matched niche",
            niche_name,
            source_path,
        )
    adata.obs[query_niche_id] = mask
    adata.obsm[niche_name] = mask.astype(float).reshape(-1, 1)

    names = (
        tuple(args.parcellation_names)
        if args.parcellation_names is not None
        else tuple(structure_names[parcellation] for parcellation in parcellations)
    )
    row = {
        "query_niche_id": query_niche_id,
        "source_slice": args.source_slice,
        "niche_name": niche_name,
        "center_cell": center_id,
        "k_hop": args.k_hop,
        # Store the realized cell count so manifest reconstruction is exact.
        "cells": len(match.niche_indices),
        "parcellations_in_niche": ";".join(map(str, parcellations)),
        "parcellation_composition": ";".join(map(str, match.counts)),
        "parcellation_names": ";".join(names),
    }

    # Persist the mask first.  A manifest row is never created for an H5AD write
    # that failed; both writes occur only after a qualifying match was found.
    atomic_write_h5ad(adata, source_path)
    append_manifest_row(args.manifest, row)
    details = ", ".join(
        f"{parcellation}={count} ({fraction:.3%})"
        for parcellation, count, fraction in zip(
            parcellations, match.counts, match.fractions
        )
    )
    logger.info(
        "Created %s: center=%s, n=%d, %s",
        query_niche_id,
        center_id,
        len(match.niche_indices),
        details,
    )
    logger.info("Updated %s and appended %s", source_path, args.manifest)
    return 0


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    raise SystemExit(main())
