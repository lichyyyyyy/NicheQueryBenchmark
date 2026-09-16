#!/usr/bin/env python3
"""Generate query manifests from stored AnnData obs niche masks.

The generator scans source slice H5AD files for obs columns named like::

    <niche-size>_<composition-complexity>_niche_<center_cell_id>

Each matching obs column becomes one row in ``query_niche_manifest.csv`` with
the obs column name as ``query_niche_id``. Niche size, k-hop, parcellation IDs,
and parcellation composition are resolved from the corresponding preprocessed
CSV row in ``query_niche_metrics/preprocessed/<niche-size>/<source_slice>.csv``.
The script also writes ``query_manifest.csv`` with one row for each query niche,
embedding type, and non-source target slice.

Example::

    .venv/bin/python experiment/generate_query_manifests.py \
        --niche-size large --composition-complexity simple
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import anndata as ad
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "20260601_225717"
DEFAULT_PREPROCESSED_ROOT = EXPERIMENT_DIR / "query_niche_metrics" / "preprocessed"
DEFAULT_NICHE_MANIFEST = EXPERIMENT_DIR / "manifests" / "query_niche_manifest.csv"
DEFAULT_QUERY_MANIFEST = EXPERIMENT_DIR / "manifests" / "query_manifest.csv"
DEFAULT_PARCELLATION_MEMBERSHIP = (
    REPO_ROOT
    / "data"
    / "ccf_parcellation"
    / "parcellation_to_parcellation_term_membership.csv"
)
DEFAULT_EMBEDDING_TYPES = ("quest_gene_expr", "quest_scgpt", "scgpt", "gene_expr")
NICHE_MANIFEST_COLUMNS = [
    "query_niche_id",
    "source_slice",
    "niche_name",
    "center_cell",
    "k_hop",
    "cells",
    "parcellations_in_niche",
    "parcellation_composition",
    "parcellation_names",
]
QUERY_MANIFEST_COLUMNS = [
    "query_id",
    "embedding_type",
    "query_niche_id",
    "source_slice",
    "target_slice",
    "niche_query_k",
]


def validate_dimension_part(value: str, *, label: str) -> str:
    name = str(value).strip()
    if (
        not name
        or Path(name).name != name
        or name in {".", ".."}
        or any(character.isspace() for character in name)
    ):
        raise ValueError(f"{label} must be a single path-safe name, got {value!r}")
    return name


def load_parcellation_ids(preprocessed_dir: Path) -> list[int]:
    path = preprocessed_dir / "parcellation_ids.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing parcellation ID axis: {path}")
    values = json.loads(path.read_text())
    ids = [int(value) for value in values]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate parcellation IDs in {path}")
    return ids


def load_parcellation_names(path: Path) -> dict[int, str]:
    if not path.is_file():
        return {}
    frame = pd.read_csv(
        path,
        usecols=[
            "parcellation_index",
            "parcellation_term_set_name",
            "parcellation_term_name",
        ],
    )
    structure_rows = frame[frame["parcellation_term_set_name"] == "structure"]
    if structure_rows.empty:
        structure_rows = frame
    result: dict[int, str] = {}
    for row in structure_rows.itertuples(index=False):
        index = int(row.parcellation_index)
        result.setdefault(index, str(row.parcellation_term_name))
    return result


def load_preprocessed_slice(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing preprocessed slice CSV: {path}")
    required = [
        "center_cell_name",
        "niche_cell_count",
        "selected_k_hop",
        "parcellation_fractions",
    ]
    try:
        frame = pd.read_csv(path, usecols=required, dtype=str)
    except ValueError as exc:
        raise ValueError(f"{path}: missing required preprocessed column") from exc
    frame["center_cell_name"] = frame["center_cell_name"].str.strip()
    duplicates = frame["center_cell_name"][
        frame["center_cell_name"].duplicated()
    ].unique()
    if len(duplicates):
        raise ValueError(f"{path}: duplicate center(s): {', '.join(duplicates)}")
    return {
        str(row.center_cell_name): {
            "niche_cell_count": str(row.niche_cell_count),
            "selected_k_hop": str(row.selected_k_hop),
            "parcellation_fractions": str(row.parcellation_fractions),
        }
        for row in frame.itertuples(index=False)
    }


def composition_from_fractions(
    fractions_json: str, *, parcellation_ids: list[int], cell_count: int
) -> tuple[list[int], list[int]]:
    try:
        fractions = json.loads(fractions_json)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON in parcellation_fractions") from exc
    if not isinstance(fractions, list) or len(fractions) != len(parcellation_ids):
        raise ValueError("parcellation_fractions does not match parcellation_ids.json")

    positive = [
        (parcellation_id, float(fraction))
        for parcellation_id, fraction in zip(parcellation_ids, fractions)
        if float(fraction) > 0
    ]
    ids = [parcellation_id for parcellation_id, _ in positive]
    counts = [int(round(fraction * cell_count)) for _, fraction in positive]
    difference = cell_count - sum(counts)
    if counts and difference:
        largest = max(range(len(positive)), key=lambda index: positive[index][1])
        counts[largest] += difference
    if any(count <= 0 for count in counts):
        ids_counts = [
            (parcellation_id, count)
            for parcellation_id, count in zip(ids, counts)
            if count > 0
        ]
        ids = [parcellation_id for parcellation_id, _ in ids_counts]
        counts = [count for _, count in ids_counts]
    return ids, counts


def matching_obs_columns(h5ad_path: Path, prefix: str) -> list[str]:
    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        return sorted(
            column for column in adata.obs.columns if column.startswith(prefix)
        )
    finally:
        adata.file.close()


def generate_niche_rows(
    *,
    data_dir: Path,
    preprocessed_dir: Path,
    niche_size: str,
    composition_complexity: str,
    parcellation_names: dict[int, str],
) -> list[dict[str, str]]:
    parcellation_ids = load_parcellation_ids(preprocessed_dir)
    prefix = f"{niche_size}_{composition_complexity}_niche_"
    rows: list[dict[str, str]] = []

    for h5ad_path in sorted(data_dir.glob("*.h5ad")):
        source_slice = h5ad_path.stem
        obs_columns = matching_obs_columns(h5ad_path, prefix)
        if not obs_columns:
            continue
        preprocessed = load_preprocessed_slice(preprocessed_dir / f"{source_slice}.csv")

        for query_niche_id in obs_columns:
            center_cell = query_niche_id.removeprefix(prefix)
            if center_cell not in preprocessed:
                raise ValueError(
                    f"{source_slice}: center {center_cell!r} from obs column "
                    f"{query_niche_id!r} is missing from preprocessed CSV"
                )
            record = preprocessed[center_cell]
            cell_count = int(record["niche_cell_count"])
            ids, counts = composition_from_fractions(
                record["parcellation_fractions"],
                parcellation_ids=parcellation_ids,
                cell_count=cell_count,
            )
            names = [parcellation_names.get(pid, str(pid)) for pid in ids]
            rows.append(
                {
                    "query_niche_id": query_niche_id,
                    "source_slice": source_slice,
                    "niche_name": query_niche_id,
                    "center_cell": center_cell,
                    "k_hop": str(int(record["selected_k_hop"])),
                    "cells": str(cell_count),
                    "parcellations_in_niche": ";".join(map(str, ids)),
                    "parcellation_composition": ";".join(map(str, counts)),
                    "parcellation_names": ";".join(names),
                }
            )

    return rows


def slice_ids(data_dir: Path) -> list[str]:
    return sorted(path.stem for path in data_dir.glob("*.h5ad"))


def generate_query_rows(
    niche_rows: list[dict[str, str]],
    *,
    all_slice_ids: list[str],
    embedding_types: tuple[str, ...],
    niche_query_k: int,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    query_id = 1
    for niche_row in niche_rows:
        source_slice = niche_row["source_slice"]
        target_slices = [
            target_slice
            for target_slice in all_slice_ids
            if target_slice != source_slice
        ]
        for embedding_type in embedding_types:
            for target_slice in target_slices:
                rows.append(
                    {
                        "query_id": str(query_id),
                        "embedding_type": embedding_type,
                        "query_niche_id": niche_row["query_niche_id"],
                        "source_slice": source_slice,
                        "target_slice": target_slice,
                        "niche_query_k": str(niche_query_k),
                    }
                )
                query_id += 1
    return rows


def read_existing_manifest(path: Path, columns: list[str]) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(columns) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        return [{column: row.get(column, "") for column in columns} for row in reader]


def merge_niche_rows(
    existing_rows: list[dict[str, str]], new_rows: list[dict[str, str]]
) -> tuple[list[dict[str, str]], int]:
    seen = {row["query_niche_id"] for row in existing_rows}
    merged = list(existing_rows)
    skipped = 0
    for row in new_rows:
        query_niche_id = row["query_niche_id"]
        if query_niche_id in seen:
            skipped += 1
            continue
        seen.add(query_niche_id)
        merged.append(row)
    return merged, skipped


def query_row_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (
        row["embedding_type"],
        row["query_niche_id"],
        row["source_slice"],
        row["target_slice"],
        row["niche_query_k"],
    )


def merge_query_rows(
    existing_rows: list[dict[str, str]], new_rows: list[dict[str, str]]
) -> tuple[list[dict[str, str]], int]:
    seen = {query_row_key(row) for row in existing_rows}
    merged = list(existing_rows)
    skipped = 0
    next_query_id = 1
    for row in existing_rows:
        raw_query_id = row["query_id"]
        if raw_query_id:
            try:
                next_query_id = max(next_query_id, int(raw_query_id) + 1)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid query_id in existing manifest: {raw_query_id}"
                ) from exc

    for row in new_rows:
        key = query_row_key(row)
        if key in seen:
            skipped += 1
            continue
        seen.add(key)
        new_row = dict(row)
        new_row["query_id"] = str(next_query_id)
        next_query_id += 1
        merged.append(new_row)
    return merged, skipped


def write_manifest(
    path: Path, rows: list[dict[str, str]], columns: list[str], *, quote_all: bool
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        quoting = csv.QUOTE_ALL if quote_all else csv.QUOTE_MINIMAL
        writer = csv.DictWriter(handle, fieldnames=columns, quoting=quoting)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--niche-size", default="large")
    parser.add_argument("--composition-complexity", default="simple")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--preprocessed-dir", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_NICHE_MANIFEST)
    parser.add_argument("--query-output", type=Path, default=DEFAULT_QUERY_MANIFEST)
    parser.add_argument(
        "--embedding-type",
        dest="embedding_types",
        action="append",
        help=(
            "Embedding type to include in query_manifest.csv. Repeat to override "
            "the default order: quest_gene_expr, quest_scgpt, scgpt, gene_expr."
        ),
    )
    parser.add_argument("--niche-query-k", type=int, default=3)
    parser.add_argument(
        "--parcellation-membership",
        type=Path,
        default=DEFAULT_PARCELLATION_MEMBERSHIP,
        help="CSV used to map parcellation_index to parcellation_names.",
    )
    args = parser.parse_args()

    niche_size = validate_dimension_part(args.niche_size, label="--niche-size")
    composition_complexity = validate_dimension_part(
        args.composition_complexity, label="--composition-complexity"
    )
    preprocessed_dir = args.preprocessed_dir or DEFAULT_PREPROCESSED_ROOT / niche_size
    if args.niche_query_k <= 0:
        parser.error("--niche-query-k must be positive")
    embedding_types = tuple(args.embedding_types or DEFAULT_EMBEDDING_TYPES)

    generated_niche_rows = generate_niche_rows(
        data_dir=args.data_dir,
        preprocessed_dir=preprocessed_dir,
        niche_size=niche_size,
        composition_complexity=composition_complexity,
        parcellation_names=load_parcellation_names(args.parcellation_membership),
    )
    generated_query_rows = generate_query_rows(
        generated_niche_rows,
        all_slice_ids=slice_ids(args.data_dir),
        embedding_types=embedding_types,
        niche_query_k=args.niche_query_k,
    )

    existing_niche_rows = read_existing_manifest(args.output, NICHE_MANIFEST_COLUMNS)
    existing_query_rows = read_existing_manifest(
        args.query_output, QUERY_MANIFEST_COLUMNS
    )
    niche_rows, skipped_niche_rows = merge_niche_rows(
        existing_niche_rows, generated_niche_rows
    )
    query_rows, skipped_query_rows = merge_query_rows(
        existing_query_rows, generated_query_rows
    )

    write_manifest(
        args.output,
        niche_rows,
        NICHE_MANIFEST_COLUMNS,
        quote_all=True,
    )
    write_manifest(
        args.query_output,
        query_rows,
        QUERY_MANIFEST_COLUMNS,
        quote_all=False,
    )
    print(f"Wrote {len(niche_rows)} query niche manifest row(s) to {args.output}")
    print(f"Skipped {skipped_niche_rows} duplicate query niche row(s)")
    print(f"Wrote {len(query_rows)} query manifest row(s) to {args.query_output}")
    print(f"Skipped {skipped_query_rows} duplicate query row(s)")


if __name__ == "__main__":
    main()
