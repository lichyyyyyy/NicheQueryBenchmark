"""Draw selected agent-proposed niches for each source slice.

The script reads ``*_selected_centers.csv`` files produced by the agent
selection step, resolves each selected center cell in the matching preprocessed
niche CSV, and writes one spatial PNG per source slice.

Example
-------
Run from the repository root::

    python experiment/draw_agent_selected_niches.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import scanpy as sc
except ImportError:  # pragma: no cover - exercised only in minimal envs.
    sc = None


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parent
DEFAULT_SELECTION_DIR = (
    EXPERIMENT_DIR
    / "query_niche_metrics/niche_visualizations/agent_proposed/large/simple"
)
DEFAULT_PREPROCESSED_DIR = EXPERIMENT_DIR / "query_niche_metrics/preprocessed/large"
DEFAULT_DATA_DIR = REPO_ROOT / "data/20260601_225717"
DEFAULT_OUTPUT_DIR = (
    EXPERIMENT_DIR
    / "query_niche_metrics/niche_visualizations/agent_proposed/large/simple/visualizations"
)
BACKGROUND_COLOR = "#d3d3d3"
NICHE_COLOR = "#d62728"
CENTER_COLOR = "#ffd700"


@dataclass(frozen=True)
class SelectedNiche:
    slice_name: str
    center_cell_name: str
    niche_cell_names: tuple[str, ...]
    niche_cell_count: int
    selected_k_hop: int | None


def resolve_path(value: Path | str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def read_selection_file(path: Path) -> tuple[str, list[str]]:
    if not path.name.endswith("_selected_centers.csv"):
        raise ValueError(f"Unexpected selection filename: {path.name}")
    slice_name = path.name[: -len("_selected_centers.csv")]
    centers: list[str] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"source_slice", "center_cell_name"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        for row in reader:
            source_slice = str(row["source_slice"]).strip()
            if source_slice != slice_name:
                raise ValueError(
                    f"{path}: source_slice={source_slice!r} does not match "
                    f"file slice {slice_name!r}"
                )
            center = str(row["center_cell_name"]).strip()
            if not center:
                raise ValueError(f"{path}: empty center_cell_name")
            centers.append(center)
    if len(set(centers)) != len(centers):
        raise ValueError(f"{path}: duplicate center_cell_name values")
    return slice_name, centers


def load_selected_niches(
    slice_name: str, center_cell_names: list[str], preprocessed_dir: Path
) -> list[SelectedNiche]:
    csv_path = preprocessed_dir / f"{slice_name}.csv"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing preprocessed niche CSV: {csv_path}")

    selected = set(center_cell_names)
    found: dict[str, SelectedNiche] = {}
    previous_limit = csv.field_size_limit()
    csv.field_size_limit(100_000_000)
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"center_cell_name", "niche_cell_names", "niche_cell_count"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{csv_path}: missing columns {sorted(missing)}")
            for row in reader:
                center = row["center_cell_name"]
                if center not in selected:
                    continue
                member_names = json.loads(row["niche_cell_names"])
                if not isinstance(member_names, list) or not all(
                    isinstance(x, str) for x in member_names
                ):
                    raise ValueError(
                        f"{csv_path}: invalid niche_cell_names for center {center}"
                    )
                if center not in member_names:
                    raise ValueError(
                        f"{csv_path}: niche for center {center} omits the center cell"
                    )
                found[center] = SelectedNiche(
                    slice_name=slice_name,
                    center_cell_name=center,
                    niche_cell_names=tuple(member_names),
                    niche_cell_count=int(row["niche_cell_count"]),
                    selected_k_hop=(
                        int(row["selected_k_hop"])
                        if row.get("selected_k_hop", "").strip()
                        else None
                    ),
                )
    finally:
        csv.field_size_limit(previous_limit)

    missing_centers = [center for center in center_cell_names if center not in found]
    if missing_centers:
        raise ValueError(
            f"{csv_path}: missing selected centers: {', '.join(missing_centers[:5])}"
        )
    return [found[center] for center in center_cell_names]


def load_slice_coordinates(
    data_dir: Path, slice_name: str, spatial_key: str
) -> tuple[pd.Index, np.ndarray]:
    h5ad_path = data_dir / f"{slice_name}.h5ad"
    if not h5ad_path.is_file():
        raise FileNotFoundError(f"Missing source slice H5AD: {h5ad_path}")

    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        if spatial_key not in adata.obsm:
            raise KeyError(
                f"{h5ad_path}: missing obsm[{spatial_key!r}]; "
                f"available keys: {list(adata.obsm)}"
            )
        cell_names = pd.Index(adata.obs_names.astype(str))
        coordinates = np.asarray(adata.obsm[spatial_key], dtype=float).copy()
    finally:
        adata.file.close()

    if not cell_names.is_unique:
        raise ValueError(f"{h5ad_path}: cell IDs are not unique")
    if (
        coordinates.ndim != 2
        or coordinates.shape[0] != len(cell_names)
        or coordinates.shape[1] < 2
    ):
        raise ValueError(
            f"{h5ad_path}: obsm[{spatial_key!r}] must have shape (n_obs, >=2)"
        )
    if not np.isfinite(coordinates).all():
        raise ValueError(f"{h5ad_path}: coordinates contain non-finite values")
    return cell_names, coordinates[:, :2]


def make_plot_adata(cell_names: pd.Index, coordinates: np.ndarray, labels: np.ndarray):
    plot_adata = ad.AnnData(obs=pd.DataFrame(index=cell_names))
    plot_adata.obs["selection"] = pd.Categorical(
        labels,
        categories=["Other slice cells", "Selected niche cells"],
        ordered=True,
    )
    plot_adata.obsm["X_selected_niche_spatial"] = coordinates
    return plot_adata


def plot_background_with_scanpy(ax: Any, cell_names: pd.Index, xy: np.ndarray, labels):
    if sc is None:
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=2,
            c=BACKGROUND_COLOR,
            alpha=0.45,
            linewidths=0,
            rasterized=True,
            label="Other slice cells",
        )
        return

    plot_adata = make_plot_adata(cell_names, xy, labels)
    sc.pl.embedding(
        plot_adata,
        basis="selected_niche_spatial",
        color="selection",
        palette={
            "Other slice cells": BACKGROUND_COLOR,
            "Selected niche cells": NICHE_COLOR,
        },
        size=8,
        alpha=0.55,
        ax=ax,
        show=False,
        frameon=True,
        legend_loc=None,
        title="",
    )


def padded_limits(
    points: np.ndarray, all_points: np.ndarray
) -> tuple[tuple[float, float], tuple[float, float]]:
    low = points.min(axis=0)
    high = points.max(axis=0)
    padding = np.maximum(
        (high - low) * 0.18, np.maximum(np.ptp(all_points, axis=0) * 0.005, 1e-6)
    )
    return (low[0] - padding[0], high[0] + padding[0]), (
        low[1] - padding[1],
        high[1] + padding[1],
    )


def draw_slice(
    slice_name: str,
    niches: list[SelectedNiche],
    *,
    data_dir: Path,
    output_dir: Path,
    spatial_key: str,
    invert_y_axis: bool,
) -> Path:
    cell_names, xy = load_slice_coordinates(data_dir, slice_name, spatial_key)
    cell_to_index = pd.Series(np.arange(len(cell_names)), index=cell_names)

    niche_indices_by_center: dict[str, np.ndarray] = {}
    selected_mask = np.zeros(len(cell_names), dtype=bool)
    center_indices = []
    for niche in niches:
        member_indices = cell_to_index.reindex(niche.niche_cell_names).to_numpy()
        if pd.isna(member_indices).any():
            missing = [
                name
                for name, idx in zip(niche.niche_cell_names, member_indices)
                if pd.isna(idx)
            ]
            raise ValueError(
                f"{slice_name}: {len(missing)} niche members missing from H5AD; "
                f"examples: {missing[:5]}"
            )
        member_indices = member_indices.astype(int)
        center_index = cell_names.get_loc(niche.center_cell_name)
        selected_mask[member_indices] = True
        center_indices.append(center_index)
        niche_indices_by_center[niche.center_cell_name] = member_indices

    labels = np.where(selected_mask, "Selected niche cells", "Other slice cells")
    center_indices_array = np.asarray(center_indices, dtype=int)
    n_niches = len(niches)
    n_cols = min(3, max(1, n_niches))
    n_rows = math.ceil(n_niches / n_cols)

    fig = plt.figure(
        figsize=(7.2 * (n_cols + 1), 5.8 * n_rows), constrained_layout=True
    )
    grid = fig.add_gridspec(n_rows, n_cols + 1)
    ax_full = fig.add_subplot(grid[:, 0])
    plot_background_with_scanpy(ax_full, cell_names, xy, labels)
    ax_full.scatter(
        xy[selected_mask, 0],
        xy[selected_mask, 1],
        s=13,
        c=NICHE_COLOR,
        linewidths=0,
        zorder=3,
        label=f"Selected niche cells ({int(selected_mask.sum())})",
    )
    ax_full.scatter(
        xy[center_indices_array, 0],
        xy[center_indices_array, 1],
        s=170,
        marker="*",
        c=CENTER_COLOR,
        edgecolors="black",
        linewidths=0.9,
        zorder=4,
        label=f"Center cells ({n_niches})",
    )
    ax_full.set_title(f"Full slice: {slice_name}")
    ax_full.set_xlabel(f"{spatial_key} coordinate 0")
    ax_full.set_ylabel(f"{spatial_key} coordinate 1")
    ax_full.set_aspect("equal", adjustable="box")
    ax_full.legend(loc="best")

    for plot_index, niche in enumerate(niches):
        row = plot_index // n_cols
        col = plot_index % n_cols + 1
        ax = fig.add_subplot(grid[row, col])
        member_indices = niche_indices_by_center[niche.center_cell_name]
        niche_xy = xy[member_indices]
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            s=2,
            c=BACKGROUND_COLOR,
            alpha=0.35,
            linewidths=0,
            rasterized=True,
            label="Other slice cells",
        )
        ax.scatter(
            niche_xy[:, 0],
            niche_xy[:, 1],
            s=16,
            c=NICHE_COLOR,
            linewidths=0,
            zorder=3,
            label=f"Niche ({len(member_indices)} cells)",
        )
        center_xy = xy[cell_names.get_loc(niche.center_cell_name)]
        ax.scatter(
            center_xy[0],
            center_xy[1],
            s=170,
            marker="*",
            c=CENTER_COLOR,
            edgecolors="black",
            linewidths=0.9,
            zorder=4,
            label="Center cell",
        )
        xlim, ylim = padded_limits(niche_xy, xy)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_title(
            f"Center cell: {niche.center_cell_name}\n"
            f"Niche close-up | {niche.niche_cell_count} cells"
        )
        ax.set_xlabel(f"{spatial_key} coordinate 0")
        ax.set_ylabel(f"{spatial_key} coordinate 1")
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="best")

    for empty_index in range(n_niches, n_rows * n_cols):
        row = empty_index // n_cols
        col = empty_index % n_cols + 1
        fig.add_subplot(grid[row, col]).axis("off")

    if invert_y_axis:
        for ax in fig.axes:
            if ax.has_data():
                ax.invert_yaxis()

    fig.suptitle(f"Agent-selected niches: {slice_name}", fontsize=16)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{slice_name}_selected_niches.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_gallery_html(output_paths: list[Path], output_dir: Path) -> Path:
    gallery_path = output_dir / "index.html"
    cards = []
    for image_path in sorted(output_paths):
        rel_path = image_path.relative_to(output_dir)
        title = image_path.name.removesuffix("_selected_niches.png")
        cards.append(
            "<article>"
            f"<h2>{escape(title)}</h2>"
            f'<a href="{escape(rel_path.as_posix(), quote=True)}">'
            f'<img src="{escape(rel_path.as_posix(), quote=True)}" '
            f'alt="{escape(title, quote=True)} selected niche graph">'
            "</a>"
            "</article>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Agent-selected niche graphs</title>
<style>
body {{
    margin: 0;
    font: 16px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    color: #20242c;
    background: #f4f6f8;
}}
main {{
    max-width: 1500px;
    margin: 0 auto;
    padding: 28px;
}}
header {{
    margin-bottom: 22px;
}}
h1 {{
    margin: 0 0 6px;
    font-size: 1.8rem;
}}
p {{
    margin: 0;
    color: #5b6472;
}}
.gallery {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
    gap: 18px;
}}
article {{
    background: white;
    border: 1px solid #d9dfe7;
    border-radius: 8px;
    padding: 14px;
}}
h2 {{
    margin: 0 0 10px;
    font-size: 1rem;
}}
img {{
    display: block;
    width: 100%;
    height: auto;
    border: 1px solid #e5e9ef;
}}
@media (max-width: 520px) {{
    main {{ padding: 16px; }}
    .gallery {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<main>
<header>
<h1>Agent-selected niche graphs</h1>
<p>{len(output_paths)} source-slice figures generated from selected-center CSVs.</p>
</header>
<section class="gallery">
{"".join(cards)}
</section>
</main>
</body>
</html>
"""
    gallery_path.write_text(html, encoding="utf-8")
    return gallery_path


def draw_selected_niches(
    *,
    selection_dir: Path = DEFAULT_SELECTION_DIR,
    preprocessed_dir: Path = DEFAULT_PREPROCESSED_DIR,
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    spatial_key: str = "spatial",
    invert_y_axis: bool = False,
) -> list[Path]:
    selection_dir = resolve_path(selection_dir)
    preprocessed_dir = resolve_path(preprocessed_dir)
    data_dir = resolve_path(data_dir)
    output_dir = resolve_path(output_dir)

    selection_paths = sorted(selection_dir.glob("*_selected_centers.csv"))
    if not selection_paths:
        raise FileNotFoundError(f"No *_selected_centers.csv files in {selection_dir}")

    outputs = []
    for selection_path in selection_paths:
        slice_name, centers = read_selection_file(selection_path)
        niches = load_selected_niches(slice_name, centers, preprocessed_dir)
        output_path = draw_slice(
            slice_name,
            niches,
            data_dir=data_dir,
            output_dir=output_dir,
            spatial_key=spatial_key,
            invert_y_axis=invert_y_axis,
        )
        print(f"{slice_name}: wrote {output_path}")
        outputs.append(output_path)
    gallery_path = write_gallery_html(outputs, output_dir)
    print(f"HTML gallery: {gallery_path}")
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--selection-dir", type=Path, default=DEFAULT_SELECTION_DIR)
    parser.add_argument(
        "--preprocessed-dir", type=Path, default=DEFAULT_PREPROCESSED_DIR
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--spatial-key", default="spatial")
    parser.add_argument("--invert-y-axis", action="store_true")
    args = parser.parse_args()
    try:
        outputs = draw_selected_niches(**vars(args))
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"Wrote {len(outputs)} figure(s).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
