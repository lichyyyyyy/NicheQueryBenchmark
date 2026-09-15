#!/usr/bin/env python3
"""Store selected niche membership masks in source-slice AnnData.

Reads selected centers from::

    experiment/query_niche_metrics/niche_visualizations/agent_proposed/<niche-size>/<composition-complexity>

and resolves each center against the corresponding preprocessed slice CSV in::

    experiment/query_niche_metrics/preprocessed/<niche-size>

For every selected center, this writes a binary membership mask to the source
slice H5AD as ``adata.obs["<niche-size>_<composition-complexity>_niche_{center_cell_id}"]``. Cells in the
selected niche are assigned 1, and every other cell in the source slice is
assigned 0.

Example::

    .venv/bin/python experiment/store_selected_niches.py \
        --niche-size large --composition-complexity simple

Use ``--dry-run`` to validate inputs and report planned columns without writing
the H5AD files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

csv.field_size_limit(100_000_000)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SELECTION_DIR = (
    Path(__file__).resolve().parent
    / "query_niche_metrics"
    / "niche_visualizations"
    / "agent_proposed"
)
DEFAULT_PREPROCESSED_DIR = (
    Path(__file__).resolve().parent / "query_niche_metrics" / "preprocessed"
)
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "20260601_225717"


@dataclass(frozen=True)
class SelectedNiche:
    source_slice: str
    center_cell_id: str
    members: frozenset[str]


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


def obs_key_for(
    center_cell_id: str, *, niche_size: str, composition_complexity: str
) -> str:
    return f"{niche_size}_{composition_complexity}_niche_{center_cell_id}"


def atomic_write_h5ad(adata: Any, destination: Path) -> None:
    """Replace an H5AD file only after a complete temporary write succeeds."""
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


def validate_slice_name(value: str, *, source: Path) -> str:
    name = str(value).strip()
    if not name or Path(name).name != name or name in {".", ".."}:
        raise ValueError(f"{source}: invalid source_slice value {value!r}")
    return name


def parse_member_list(value: str, *, center: str, source: Path) -> frozenset[str]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{source}: invalid JSON niche_cell_names for center {center!r}"
        ) from exc
    if not isinstance(parsed, list) or not all(
        isinstance(item, str) for item in parsed
    ):
        raise ValueError(
            f"{source}: niche_cell_names must be a JSON string list for {center!r}"
        )
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"{source}: duplicate niche cell IDs for center {center!r}")
    return frozenset(parsed)


def read_selected_centers(selection_csv: Path) -> list[str]:
    centers: list[str] = []
    with selection_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"source_slice", "center_cell_name"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{selection_csv}: missing column(s): {sorted(missing)}")
        expected_slice = selection_csv.name.removesuffix("_selected_centers.csv")
        for row in reader:
            source_slice = validate_slice_name(
                row["source_slice"], source=selection_csv
            )
            if source_slice != expected_slice:
                raise ValueError(
                    f"{selection_csv}: source_slice={source_slice!r} does not match "
                    f"file slice {expected_slice!r}"
                )
            center = str(row["center_cell_name"]).strip()
            if not center:
                raise ValueError(f"{selection_csv}: empty center_cell_name")
            if center in centers:
                raise ValueError(
                    f"{selection_csv}: duplicate selected center {center!r}"
                )
            centers.append(center)
    return centers


def read_preprocessed_niches(
    preprocessed_csv: Path, selected_centers: list[str]
) -> list[SelectedNiche]:
    wanted = set(selected_centers)
    try:
        frame = pd.read_csv(
            preprocessed_csv,
            usecols=["center_cell_name", "niche_cell_names"],
            dtype=str,
        )
    except ValueError as exc:
        raise ValueError(
            f"{preprocessed_csv}: missing required preprocessed column"
        ) from exc

    frame["center_cell_name"] = frame["center_cell_name"].str.strip()
    selected = frame[frame["center_cell_name"].isin(wanted)]
    duplicates = selected["center_cell_name"][
        selected["center_cell_name"].duplicated()
    ].unique()
    if len(duplicates):
        raise ValueError(
            f"{preprocessed_csv}: duplicate center(s): {', '.join(duplicates)}"
        )
    rows = {
        str(row.center_cell_name): parse_member_list(
            str(row.niche_cell_names),
            center=str(row.center_cell_name),
            source=preprocessed_csv,
        )
        for row in selected.itertuples(index=False)
    }

    missing_centers = [center for center in selected_centers if center not in rows]
    if missing_centers:
        raise ValueError(
            f"{preprocessed_csv}: selected center(s) missing from preprocessed CSV: "
            + ", ".join(missing_centers)
        )

    source_slice = preprocessed_csv.stem
    return [
        SelectedNiche(
            source_slice=source_slice,
            center_cell_id=center,
            members=rows[center],
        )
        for center in selected_centers
    ]


def iter_selected_niches(selection_dir: Path, preprocessed_dir: Path):
    selection_paths = sorted(selection_dir.glob("*_selected_centers.csv"))
    if not selection_paths:
        raise FileNotFoundError(
            f"No *_selected_centers.csv files found in {selection_dir}"
        )

    for selection_csv in selection_paths:
        source_slice = selection_csv.name.removesuffix("_selected_centers.csv")
        preprocessed_csv = preprocessed_dir / f"{source_slice}.csv"
        if not preprocessed_csv.is_file():
            raise FileNotFoundError(
                f"Missing preprocessed CSV for {source_slice}: {preprocessed_csv}"
            )
        centers = read_selected_centers(selection_csv)
        yield source_slice, read_preprocessed_niches(preprocessed_csv, centers)


def store_niches_for_slice(
    h5ad_path: Path,
    niches: list[SelectedNiche],
    *,
    niche_size: str,
    composition_complexity: str,
    overwrite: bool,
    dry_run: bool,
) -> list[str]:
    if not h5ad_path.is_file():
        raise FileNotFoundError(f"Missing source slice H5AD: {h5ad_path}")

    adata = ad.read_h5ad(h5ad_path, backed="r") if dry_run else ad.read_h5ad(h5ad_path)
    try:
        cell_names = np.asarray(adata.obs_names.astype(str))
        available = set(cell_names)
        written_keys: list[str] = []

        for niche in niches:
            obs_key = obs_key_for(
                niche.center_cell_id,
                niche_size=niche_size,
                composition_complexity=composition_complexity,
            )
            missing_members = sorted(niche.members - available)
            if missing_members:
                examples = ", ".join(missing_members[:5])
                raise ValueError(
                    f"{h5ad_path}: {obs_key} has {len(missing_members)} "
                    f"member cell(s) absent from the H5AD; examples: {examples}"
                )
            if obs_key in adata.obs and not overwrite:
                raise ValueError(
                    f"{h5ad_path}: obs column {obs_key!r} already exists; "
                    "pass --overwrite to replace it"
                )
            if not dry_run:
                adata.obs[obs_key] = np.fromiter(
                    (1 if cell_id in niche.members else 0 for cell_id in cell_names),
                    dtype=np.int8,
                    count=adata.n_obs,
                )
            written_keys.append(obs_key)

        if not dry_run:
            atomic_write_h5ad(adata, h5ad_path)
        return written_keys
    finally:
        adata.file.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--niche-size", default="large")
    parser.add_argument("--composition-complexity", default="simple")
    parser.add_argument(
        "--selection-dir",
        type=Path,
        help=(
            "Directory containing *_selected_centers.csv files. Default: "
            "query_niche_metrics/niche_visualizations/agent_proposed/"
            "<niche-size>/<composition-complexity>."
        ),
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        help=(
            "Directory containing preprocessed <slice>.csv files. Default: "
            "query_niche_metrics/preprocessed/<niche-size>."
        ),
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing selected-niche adata.obs columns.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report planned writes without modifying H5AD files.",
    )
    args = parser.parse_args()
    niche_size = validate_dimension_part(args.niche_size, label="--niche-size")
    composition_complexity = validate_dimension_part(
        args.composition_complexity, label="--composition-complexity"
    )
    selection_dir = (
        args.selection_dir
        if args.selection_dir is not None
        else DEFAULT_SELECTION_DIR / niche_size / composition_complexity
    )
    preprocessed_dir = (
        args.preprocessed_dir
        if args.preprocessed_dir is not None
        else DEFAULT_PREPROCESSED_DIR / niche_size
    )

    total_columns = 0
    for source_slice, niches in iter_selected_niches(
        selection_dir, preprocessed_dir
    ):
        h5ad_path = args.data_dir / f"{source_slice}.h5ad"
        keys = store_niches_for_slice(
            h5ad_path,
            niches,
            niche_size=niche_size,
            composition_complexity=composition_complexity,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        total_columns += len(keys)
        action = "would write" if args.dry_run else "wrote"
        print(f"{source_slice}: {action} {len(keys)} obs column(s)")
        for key in keys:
            print(f"  {key}")

    action = "Validated" if args.dry_run else "Stored"
    print(
        f"{action} {total_columns} selected "
        f"{niche_size}/{composition_complexity} niche obs column(s)."
    )


if __name__ == "__main__":
    main()
