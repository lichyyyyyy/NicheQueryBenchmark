#!/usr/bin/env python3
"""Run the find-query-niches workflow without invoking the Codex skill.

Default behavior matches the find-query-niches skill:

    .venv/bin/python experiment/run_find_query_niches.py

It filters dimension-qualified niches, filters by prevalence, selects up to k
mutually compatible niches, and writes both selection artifacts. Per-niche
HTML/PNG reports are generated only when --reports is passed.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_SOURCE_SLICE = ""
DEFAULT_DIMENSIONS = ("large", "simple")
DEFAULT_K = 5
EXPECTED_REPORT_ARTIFACTS = (
    "metrics.csv",
    "composition_summary.csv",
    "center_coordinates.csv",
    "members.csv",
    "composition.csv",
    "source_composition_similarity.csv",
    "target_slice_composition.csv",
    "niche.png",
    "report.html",
)
SELECTED_CENTERS_COLUMNS = (
    "source_slice",
    "center_cell_name",
    "niche_dimension",
    "composition_complexity",
    "niche_cell_count",
    "parcellation_ids",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_python(root: Path) -> Path:
    return root / ".venv/bin/python"


def run_command(command: list[str], root: Path) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=root, check=True)


def require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")


def require_dir(path: Path, description: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"missing {description}: {path}")


def read_selection(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    payload.setdefault("center_cell_names", [])
    payload.setdefault("parcellation_ids", {})
    return payload


def write_selected_centers_csv(
    *,
    selection: dict,
    source_slice: str,
    dim1: str,
    dim2: str,
    preprocessed_csv: Path,
    output_csv: Path,
) -> None:
    centers = [str(center) for center in selection["center_cell_names"]]
    selected = set(centers)
    records: dict[str, dict[str, str]] = {}

    with preprocessed_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"center_cell_name", "niche_cell_count"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{preprocessed_csv}: missing columns {sorted(missing)}")
        for row in reader:
            center = str(row["center_cell_name"])
            if center in selected:
                records[center] = row

    missing_centers = [center for center in centers if center not in records]
    if missing_centers:
        raise ValueError(
            "selected centers missing from preprocessed metrics: "
            + ", ".join(missing_centers)
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SELECTED_CENTERS_COLUMNS)
        writer.writeheader()
        for center in centers:
            parcellation_ids = selection["parcellation_ids"].get(center, [])
            writer.writerow(
                {
                    "source_slice": source_slice,
                    "center_cell_name": center,
                    "niche_dimension": dim1,
                    "composition_complexity": dim2,
                    "niche_cell_count": records[center]["niche_cell_count"],
                    "parcellation_ids": json.dumps(
                        sorted(int(value) for value in parcellation_ids)
                    ),
                }
            )


def generate_reports(
    *,
    python: Path,
    root: Path,
    source_slice: str,
    dim1: str,
    metrics_root: Path,
    report_root: Path,
    centers: list[str],
    data_dir: Path | None,
) -> list[Path]:
    report_dirs: list[Path] = []
    for center in centers:
        output_dir = report_root / f"{source_slice}_{center}"
        command = [
            str(python),
            "experiment/visualize_query_niche.py",
            "--source-slice",
            source_slice,
            "--cell-id",
            center,
            "--all-niche-candidates-dir",
            str(metrics_root / dim1),
            "--output-dir",
            str(output_dir),
        ]
        if data_dir is not None:
            command.extend(["--data-dir", str(data_dir)])
        run_command(command, root)

        missing = [
            artifact
            for artifact in EXPECTED_REPORT_ARTIFACTS
            if not (output_dir / artifact).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"{output_dir}: missing report artifacts {missing}")
        report_dirs.append(output_dir)
    return report_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-slice",
        default=DEFAULT_SOURCE_SLICE,
        help=(
            "Source slice stem, or a comma-separated list of stems. Pass an empty "
            "string to run all slices under metrics-root/preprocessed/<dimension[0]>."
        ),
    )
    parser.add_argument("--dimension", nargs=2, default=DEFAULT_DIMENSIONS)
    parser.add_argument("-k", type=int, default=DEFAULT_K)
    parser.add_argument(
        "--max-cell-overlap-percent",
        "--max_cell_overlap_percent",
        type=float,
        default=0.0,
        help=(
            "Maximum pairwise member-cell overlap as a percent of the smaller "
            "niche when selecting non-overlapping niches. Default: 0."
        ),
    )
    parser.add_argument(
        "--metrics-root",
        type=Path,
        default=Path("experiment/query_niche_metrics"),
    )
    parser.add_argument("--python", type=Path, help="Python executable to use")
    parser.add_argument(
        "--reports",
        action="store_true",
        help="Also generate and verify per-niche report artifacts.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Optional data directory for visualize_query_niche.py.",
    )
    parser.add_argument(
        "--exclude-slice",
        dest="exclude_slices",
        action="append",
        default=[],
        help="Target slice CSV stem to exclude from prevalence comparisons. Repeatable.",
    )
    return parser.parse_args()


def resolve_source_slices(
    source_slice: str,
    *,
    root: Path,
    metrics_root: Path,
    dim1: str,
) -> list[str]:
    if source_slice:
        source_slices: list[str] = []
        seen: set[str] = set()
        invalid: list[str] = []
        for raw_name in source_slice.split(","):
            name = raw_name.strip()
            if not name:
                continue
            if Path(name).name != name or name in {".", ".."}:
                invalid.append(name)
                continue
            if name not in seen:
                source_slices.append(name)
                seen.add(name)
        if invalid:
            raise ValueError(
                "--source-slice values must be slice names without directories: "
                f"{sorted(invalid)}"
            )
        if not source_slices:
            raise ValueError(
                "--source-slice must include at least one slice name when non-empty"
            )
        return source_slices

    preprocessed_dir = root / metrics_root / "preprocessed" / dim1
    require_dir(preprocessed_dir, "preprocessed source-slice directory")
    slices = sorted(path.stem for path in preprocessed_dir.glob("*.csv"))
    if not slices:
        raise FileNotFoundError(
            f"no preprocessed slice CSVs found under {preprocessed_dir}"
        )
    return slices


def normalize_exclude_slices(exclude_slices: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in exclude_slices:
        for name in value.split(","):
            name = name.strip()
            if name and name not in seen:
                normalized.append(name)
                seen.add(name)
    return normalized


def resolve_input_metrics_root(root: Path, metrics_root: Path, dim1: str) -> Path:
    """Return the directory whose <dim1>/ contains per-slice metrics CSVs."""
    direct_dir = root / metrics_root / dim1
    if direct_dir.is_dir():
        return metrics_root

    preprocessed_root = metrics_root / "preprocessed"
    preprocessed_dir = root / preprocessed_root / dim1
    if preprocessed_dir.is_dir():
        return preprocessed_root

    raise FileNotFoundError(
        "missing size metrics directory: "
        f"{direct_dir} (also checked {preprocessed_dir})"
    )


def run_for_source_slice(
    *,
    source_slice: str,
    args: argparse.Namespace,
    root: Path,
    python: Path,
    metrics_root: Path,
    input_metrics_root: Path,
    dim1: str,
    dim2: str,
) -> dict:
    dimension_candidates = (
        metrics_root
        / "filtered_query_niches_on_dimensions"
        / dim1
        / dim2
        / f"{source_slice}.csv"
    )
    final_candidates = (
        metrics_root
        / "final_query_niche_candidates"
        / dim1
        / dim2
        / f"{source_slice}.csv"
    )
    preprocessed_csv = metrics_root / "preprocessed" / dim1 / f"{source_slice}.csv"
    report_root = metrics_root / "niche_visualizations/agent_proposed" / dim1 / dim2
    selected_centers_csv = report_root / f"{source_slice}_selected_centers.csv"
    selection_checks_json = report_root / f"{source_slice}_selection_checks.json"

    require_file(
        root / input_metrics_root / dim1 / f"{source_slice}.csv", "source metrics CSV"
    )
    require_file(root / preprocessed_csv, "preprocessed metrics CSV")

    print()
    print(f"=== {source_slice} ===", flush=True)

    run_command(
        [
            str(python),
            "experiment/filter_query_niches_on_dimensions.py",
            "--source-slice",
            source_slice,
            "--dimension",
            dim1,
            dim2,
            "--metrics-dir",
            str(input_metrics_root),
            "--output-file",
            str(dimension_candidates),
        ],
        root,
    )
    prevalence_command = [
        str(python),
        "experiment/filter_query_niche_on_prevalence.py",
        "--input-file",
        str(dimension_candidates),
        "--source-slice",
        source_slice,
        "--metrics-dir",
        str(input_metrics_root),
        "--output-file",
        str(final_candidates),
    ]
    for excluded_slice in args.exclude_slices:
        prevalence_command.extend(["--exclude-slice", excluded_slice])
    run_command(prevalence_command, root)
    run_command(
        [
            str(python),
            "experiment/select_non_overlapping_niches.py",
            "-k",
            str(args.k),
            "--candidates",
            str(final_candidates),
            "--preprocessed",
            str(preprocessed_csv),
            "--max-cell-overlap-percent",
            str(args.max_cell_overlap_percent),
            "--output",
            str(selection_checks_json),
        ],
        root,
    )

    selection = read_selection(root / selection_checks_json)
    write_selected_centers_csv(
        selection=selection,
        source_slice=source_slice,
        dim1=dim1,
        dim2=dim2,
        preprocessed_csv=root / preprocessed_csv,
        output_csv=root / selected_centers_csv,
    )

    report_dirs: list[Path] = []
    if args.reports:
        report_dirs = generate_reports(
            python=python,
            root=root,
            source_slice=source_slice,
            dim1=dim1,
            metrics_root=input_metrics_root,
            report_root=report_root,
            centers=[str(center) for center in selection["center_cell_names"]],
            data_dir=args.data_dir,
        )

    print()
    print(f"source_slice: {source_slice}")
    print(f"requested_k: {selection.get('requested_k', args.k)}")
    print(f"achieved_k: {selection.get('achieved_k', 0)}")
    print(f"shortfall: {selection.get('shortfall', 0)}")
    print("selected centers:")
    for center in selection["center_cell_names"]:
        parcellations = selection["parcellation_ids"].get(str(center), [])
        print(f"  {center}: {parcellations}")
    print(f"selected_centers_csv: {selected_centers_csv}")
    print(f"selection_checks_json: {selection_checks_json}")
    if args.exclude_slices:
        print(f"excluded target slices: {sorted(args.exclude_slices)}")
    if report_dirs:
        print("report directories:")
        for report_dir in report_dirs:
            print(f"  {report_dir}")
    elif not args.reports:
        print(
            "report generation skipped; pass --reports to generate HTML/PNG artifacts"
        )

    return {
        "source_slice": source_slice,
        "requested_k": selection.get("requested_k", args.k),
        "achieved_k": selection.get("achieved_k", 0),
        "shortfall": selection.get("shortfall", 0),
        "selected_centers_csv": str(selected_centers_csv),
        "selection_checks_json": str(selection_checks_json),
    }


def main() -> int:
    args = parse_args()
    if args.k < 1:
        raise ValueError("-k must be positive")
    if not 0 <= args.max_cell_overlap_percent <= 100:
        raise ValueError("--max-cell-overlap-percent must be between 0 and 100")
    args.exclude_slices = normalize_exclude_slices(args.exclude_slices)
    invalid_excluded = [
        name
        for name in args.exclude_slices
        if not name or Path(name).name != name or name in {".", ".."}
    ]
    if invalid_excluded:
        raise ValueError(
            "--exclude-slice values must be slice names without directories: "
            f"{sorted(invalid_excluded)}"
        )

    root = repo_root()
    python = args.python or default_python(root)
    metrics_root = args.metrics_root
    dim1, dim2 = args.dimension
    input_metrics_root = resolve_input_metrics_root(root, metrics_root, dim1)

    require_file(python, "Python executable")
    require_file(
        root / "experiment/manifests/query_niche_dimensions.json", "dimension manifest"
    )
    require_file(
        root / "experiment/filter_query_niches_on_dimensions.py",
        "dimension filter script",
    )
    require_file(
        root / "experiment/filter_query_niche_on_prevalence.py",
        "prevalence filter script",
    )
    require_file(
        root / "experiment/select_non_overlapping_niches.py", "selection script"
    )
    if args.reports:
        require_file(
            root / "experiment/visualize_query_niche.py", "visualization script"
        )
        if args.data_dir is not None:
            require_dir(root / args.data_dir, "visualization data directory")

    source_slices = resolve_source_slices(
        args.source_slice,
        root=root,
        metrics_root=metrics_root,
        dim1=dim1,
    )
    print(
        f"processing {len(source_slices)} source slice(s): "
        + ", ".join(source_slices),
        flush=True,
    )
    if args.exclude_slices:
        excluded = set(args.exclude_slices)
        skipped_source_slices = [
            source_slice for source_slice in source_slices if source_slice in excluded
        ]
        source_slices = [
            source_slice
            for source_slice in source_slices
            if source_slice not in excluded
        ]
        if skipped_source_slices:
            print(
                f"skipping excluded source slices: {sorted(skipped_source_slices)}",
                flush=True,
            )
        if not source_slices:
            print("no source slices to run after applying exclusions", flush=True)
            return 0
    if not args.source_slice:
        print(
            f"--source-slice is empty; running {len(source_slices)} slices from "
            f"{metrics_root / 'preprocessed' / dim1}",
            flush=True,
        )

    summaries = [
        run_for_source_slice(
            source_slice=source_slice,
            args=args,
            root=root,
            python=python,
            metrics_root=metrics_root,
            input_metrics_root=input_metrics_root,
            dim1=dim1,
            dim2=dim2,
        )
        for source_slice in source_slices
    ]

    if len(summaries) > 1:
        print()
        print("=== summary ===")
        for summary in summaries:
            print(
                f"{summary['source_slice']}: "
                f"{summary['achieved_k']}/{summary['requested_k']} selected, "
                f"shortfall {summary['shortfall']}"
            )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        print(
            f"command failed with exit code {error.returncode}: {error.cmd}",
            file=sys.stderr,
        )
        raise SystemExit(error.returncode)
