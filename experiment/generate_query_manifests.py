#!/usr/bin/env python3
"""Generate query niche manifests from stored AnnData obs niche masks.

The generator scans source slice H5AD files for obs columns named like::

    <niche-size>_<composition-complexity>_niche_<center_cell_id>

Each matching obs column becomes one row in ``query_niche_manifest.csv`` with
the obs column name as ``query_niche_id``. Niche size, k-hop, parcellation IDs,
and parcellation composition are resolved from the corresponding preprocessed
CSV row in ``query_niche_metrics/preprocessed/<niche-size>/<source_slice>.csv``.

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
DEFAULT_MANIFEST = EXPERIMENT_DIR / "manifests" / "query_niche_manifest.csv"
DEFAULT_PARCELLATION_MEMBERSHIP = (
    REPO_ROOT
    / "data"
    / "ccf_parcellation"
    / "parcellation_to_parcellation_term_membership.csv"
)
MANIFEST_COLUMNS = [
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
        usecols=["parcellation_index", "parcellation_term_set_name", "parcellation_term_name"],
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
    duplicates = frame["center_cell_name"][frame["center_cell_name"].duplicated()].unique()
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
        return sorted(column for column in adata.obs.columns if column.startswith(prefix))
    finally:
        adata.file.close()


def generate_rows(
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


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--niche-size", default="large")
    parser.add_argument("--composition-complexity", default="simple")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--preprocessed-dir", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
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

    rows = generate_rows(
        data_dir=args.data_dir,
        preprocessed_dir=preprocessed_dir,
        niche_size=niche_size,
        composition_complexity=composition_complexity,
        parcellation_names=load_parcellation_names(args.parcellation_membership),
    )
    write_manifest(args.output, rows)
    print(f"Wrote {len(rows)} query niche manifest row(s) to {args.output}")


if __name__ == "__main__":
    main()
