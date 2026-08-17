"""Draw a representative niche-query result as coordinated 2-D panels.

Given one query ID, the script finds the other manifest rows with the same
niche, source slice, target slice, and neighborhood size.  Those rows supply
one prediction panel per embedding type.  The resulting figure contains:

1. source-slice query-niche membership;
2. source-slice parcellations represented by the niche;
3. the same parcellations on the target slice;
4. the top-N target cells by RM-Ideal score; and
5. the top-N target cells by prediction score for every embedding type; and
6. the top 1% of target cells by RM-Ideal and each embedding prediction score; and
7. RM-Ideal-versus-prediction hexbin/scatter comparisons for every embedding; and
8. top-k overlap/Recall@k curves comparing each prediction ranking with the
   RM-Ideal ranking across candidate-retrieval fractions.

The benchmark h5ad files currently store their 2-D slice coordinates in
``obsm["spatial"]``.  A true UMAP can be plotted instead with
``--coordinate-key X_umap`` when that key is present.

Example
-------
Run from the repository root::

    python experiment/draw_representative_query_results.py --query-id 1
"""

from __future__ import annotations

import argparse
import csv
import math
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parent
DEFAULT_QUERY_MANIFEST = EXPERIMENT_DIR / "manifests" / "query_manifest.csv"
DEFAULT_NICHE_MANIFEST = EXPERIMENT_DIR / "manifests" / "query_niche_manifest.csv"
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "20260601_225717"
DEFAULT_RAW_RESULTS_DIR = EXPERIMENT_DIR / "raw_results"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "evaluation_results" / "drawings"

PAIRING_COLUMNS = (
    "query_niche_id",
    "source_slice",
    "target_slice",
    "niche_query_k",
)
QUERY_COLUMNS = {"query_id", "embedding_type", *PAIRING_COLUMNS}
NICHE_COLUMNS = {
    "query_niche_id",
    "source_slice",
    "niche_name",
    "parcellations_in_niche",
    "parcellation_names",
}
RESULT_COLUMNS = {
    "query_id",
    "cell_id",
    "niche_query_score",
    "rm_ideal_score",
}

BACKGROUND_COLOR = "#d3d3d3"
NICHE_COLOR = "#d62728"
PARCELLATION_COLORS = (
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#a65628",
    "#f781bf",
    "#17becf",
)
EMBEDDING_LABELS = {
    "gene_expr_quest": "Gene expression + QUEST",
    "scgpt": "scGPT",
    "gene_expr": "Gene expression",
}


@dataclass(frozen=True)
class QueryRow:
    query_id: int
    embedding_type: str
    query_niche_id: str
    source_slice: str
    target_slice: str
    niche_query_k: int

    @property
    def pairing_key(self) -> tuple[str, str, str, int]:
        return (
            self.query_niche_id,
            self.source_slice,
            self.target_slice,
            self.niche_query_k,
        )


@dataclass(frozen=True)
class NicheRow:
    query_niche_id: str
    source_slice: str
    niche_name: str
    parcellation_ids: tuple[int, ...]
    parcellation_names: tuple[str, ...]


def _read_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV file has no data rows: {path}")
    return rows


def _required(row: dict[str, str], column: str, path: Path) -> str:
    value = (row.get(column) or "").strip()
    if not value:
        raise ValueError(f"Empty {column!r} value in {path}")
    return value


def load_query_rows(path: Path) -> list[QueryRow]:
    rows: list[QueryRow] = []
    seen: set[int] = set()
    for raw in _read_rows(path, QUERY_COLUMNS):
        try:
            query_id = int(_required(raw, "query_id", path))
            niche_query_k = int(_required(raw, "niche_query_k", path))
        except ValueError as exc:
            raise ValueError(
                f"query_id and niche_query_k must be integers in {path}"
            ) from exc
        if query_id <= 0 or niche_query_k <= 0:
            raise ValueError(f"query_id and niche_query_k must be positive in {path}")
        if query_id in seen:
            raise ValueError(f"Duplicate query_id={query_id} in {path}")
        seen.add(query_id)
        rows.append(
            QueryRow(
                query_id=query_id,
                embedding_type=_required(raw, "embedding_type", path),
                query_niche_id=_required(raw, "query_niche_id", path),
                source_slice=_required(raw, "source_slice", path),
                target_slice=_required(raw, "target_slice", path),
                niche_query_k=niche_query_k,
            )
        )
    return rows


def select_query_group(
    rows: Iterable[QueryRow], query_id: int
) -> tuple[QueryRow, list[QueryRow]]:
    rows = list(rows)
    selected = next((row for row in rows if row.query_id == query_id), None)
    if selected is None:
        raise KeyError(f"query_id={query_id} is absent from the query manifest")
    group = [row for row in rows if row.pairing_key == selected.pairing_key]
    seen_embeddings: set[str] = set()
    for row in group:
        if row.embedding_type in seen_embeddings:
            raise ValueError(
                "More than one matching row for embedding_type="
                f"{row.embedding_type!r} and query_id={query_id}"
            )
        seen_embeddings.add(row.embedding_type)
    return selected, group


def load_niche_row(path: Path, query_niche_id: str) -> NicheRow:
    matches = [
        row
        for row in _read_rows(path, NICHE_COLUMNS)
        if _required(row, "query_niche_id", path) == query_niche_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one niche-manifest row for {query_niche_id!r}; found {len(matches)}"
        )
    raw = matches[0]
    try:
        parcellation_ids = tuple(
            int(value.strip())
            for value in _required(raw, "parcellations_in_niche", path).split(";")
            if value.strip()
        )
    except ValueError as exc:
        raise ValueError(
            f"Invalid parcellations for {query_niche_id!r} in {path}"
        ) from exc
    names = tuple(
        value.strip() for value in _required(raw, "parcellation_names", path).split(";")
    )
    if not parcellation_ids or len(names) != len(parcellation_ids):
        raise ValueError(
            f"Parcellation IDs and names do not align for {query_niche_id!r}"
        )
    return NicheRow(
        query_niche_id=query_niche_id,
        source_slice=_required(raw, "source_slice", path),
        niche_name=_required(raw, "niche_name", path),
        parcellation_ids=parcellation_ids,
        parcellation_names=names,
    )


def load_result_scores(
    path: Path, query_id: int, target_cell_ids: list[str]
) -> tuple[Any, Any]:
    """Return prediction and RM-Ideal arrays aligned to target AnnData order."""
    import numpy as np

    by_cell: dict[str, tuple[float, float]] = {}
    for line_number, raw in enumerate(_read_rows(path, RESULT_COLUMNS), start=2):
        try:
            row_query_id = int(_required(raw, "query_id", path))
            prediction = float(_required(raw, "niche_query_score", path))
            rm_ideal = float(_required(raw, "rm_ideal_score", path))
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: invalid numeric value") from exc
        if row_query_id != query_id:
            raise ValueError(
                f"{path}:{line_number}: expected query_id={query_id}, got {row_query_id}"
            )
        if not (math.isfinite(prediction) and math.isfinite(rm_ideal)):
            raise ValueError(f"{path}:{line_number}: scores must be finite")
        cell_id = _required(raw, "cell_id", path)
        if cell_id in by_cell:
            raise ValueError(f"{path}:{line_number}: duplicate cell_id={cell_id!r}")
        by_cell[cell_id] = (prediction, rm_ideal)

    target_set = set(target_cell_ids)
    missing = [cell_id for cell_id in target_cell_ids if cell_id not in by_cell]
    extra = set(by_cell) - target_set
    if missing or extra:
        raise ValueError(
            f"{path} does not align with target AnnData: "
            f"{len(missing)} missing and {len(extra)} extra cell IDs"
        )
    prediction = np.asarray([by_cell[cell_id][0] for cell_id in target_cell_ids])
    rm_ideal = np.asarray([by_cell[cell_id][1] for cell_id in target_cell_ids])
    return prediction, rm_ideal


def _coordinates(adata: Any, key: str, sample_id: str) -> Any:
    import numpy as np

    if key not in adata.obsm:
        raise KeyError(
            f"{sample_id!r} is missing obsm[{key!r}]; available keys: "
            f"{sorted(adata.obsm.keys())}"
        )
    coordinates = np.asarray(adata.obsm[key], dtype=float)
    if (
        coordinates.ndim != 2
        or coordinates.shape[0] != adata.n_obs
        or coordinates.shape[1] < 2
    ):
        raise ValueError(
            f"{sample_id!r} obsm[{key!r}] must have shape (n_obs, >=2); "
            f"got {coordinates.shape}"
        )
    coordinates = coordinates[:, :2]
    if not np.isfinite(coordinates).all():
        raise ValueError(f"{sample_id!r} obsm[{key!r}] contains non-finite values")
    return coordinates


def _binary_niche_mask(adata: Any, niche: NicheRow) -> Any:
    import numpy as np

    candidates: list[tuple[str, Any]] = []
    if niche.query_niche_id in adata.obs:
        candidates.append(
            (f"obs[{niche.query_niche_id!r}]", adata.obs[niche.query_niche_id])
        )
    if niche.niche_name in adata.obsm:
        candidates.append((f"obsm[{niche.niche_name!r}]", adata.obsm[niche.niche_name]))
    for _label, values in candidates:
        array = np.asarray(values, dtype=float).reshape(-1)
        if (
            array.shape[0] == adata.n_obs
            and np.isfinite(array).all()
            and np.isin(array, (0.0, 1.0)).all()
            and np.any(array == 1.0)
        ):
            return array.astype(bool)
    locations = ", ".join(label for label, _ in candidates) or "neither location exists"
    raise ValueError(
        f"No non-empty binary niche mask for {niche.query_niche_id!r}; checked {locations}"
    )


def _parcellation_values(adata: Any, sample_id: str, obs_key: str) -> Any:
    import numpy as np

    if obs_key not in adata.obs:
        raise KeyError(f"{sample_id!r} is missing obs[{obs_key!r}]")
    values: list[int] = []
    for value in adata.obs[obs_key]:
        try:
            values.append(int(float(value)))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{sample_id!r} obs[{obs_key!r}] contains a non-integer value {value!r}"
            ) from exc
    return np.asarray(values, dtype=int)


def _point_sizes(n_cells: int, scale: float) -> tuple[float, float]:
    base = min(8.0, max(0.35, 12_000.0 / max(n_cells, 1))) * scale
    return base * 0.55, base * 1.6


def _format_axis(ax: Any, invert_y: bool) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    if invert_y:
        ax.invert_yaxis()


def _plot_binary(
    ax: Any, coords: Any, mask: Any, title: str, point_scale: float, invert_y: bool
) -> None:
    from matplotlib.lines import Line2D

    background_size, highlight_size = _point_sizes(len(coords), point_scale)
    ax.scatter(
        coords[~mask, 0],
        coords[~mask, 1],
        c=BACKGROUND_COLOR,
        s=background_size,
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        coords[mask, 0],
        coords[mask, 1],
        c=NICHE_COLOR,
        s=highlight_size,
        linewidths=0,
        rasterized=True,
    )
    ax.set_title(f"{title}\nquery niche (n={int(mask.sum()):,})", fontsize=10)
    ax.legend(
        handles=[
            Line2D(
                [], [], marker="o", linestyle="", color=NICHE_COLOR, label="Query niche"
            )
        ],
        loc="best",
        frameon=False,
        fontsize=8,
    )
    _format_axis(ax, invert_y)


def _plot_parcellations(
    ax: Any,
    coords: Any,
    values: Any,
    niche: NicheRow,
    title: str,
    point_scale: float,
    invert_y: bool,
) -> None:
    import numpy as np
    from matplotlib.lines import Line2D

    background_size, highlight_size = _point_sizes(len(coords), point_scale)
    selected = np.isin(values, niche.parcellation_ids)
    ax.scatter(
        coords[~selected, 0],
        coords[~selected, 1],
        c=BACKGROUND_COLOR,
        s=background_size,
        linewidths=0,
        rasterized=True,
    )
    handles = []
    for index, (parcellation_id, name) in enumerate(
        zip(niche.parcellation_ids, niche.parcellation_names)
    ):
        mask = values == parcellation_id
        color = PARCELLATION_COLORS[index % len(PARCELLATION_COLORS)]
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=color,
            s=highlight_size,
            linewidths=0,
            rasterized=True,
        )
        handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color=color,
                label=f"{parcellation_id}: {name}",
            )
        )
    ax.set_title(f"{title}\nniche parcellations", fontsize=10)
    ax.legend(handles=handles, loc="best", frameon=False, fontsize=7)
    _format_axis(ax, invert_y)


def _top_n_mask(scores: Any, top_n: int) -> Any:
    import numpy as np

    count = min(top_n, scores.size)
    mask = np.zeros(scores.size, dtype=bool)
    if count:
        order = np.argsort(-scores, kind="stable")
        mask[order[:count]] = True
    return mask


def _top_percent_count(n_cells: int, top_percent: float) -> int:
    """Return the ceiling of a percentage of cells, bounded to [1, n_cells]."""
    if n_cells <= 0:
        return 0
    return max(1, min(n_cells, math.ceil(n_cells * top_percent / 100.0)))


def _plot_scores(
    ax: Any,
    coords: Any,
    scores: Any,
    title: str,
    top_n: int,
    point_scale: float,
    cmap: str,
    invert_y: bool,
    selection_label: str | None = None,
) -> Any:
    import numpy as np
    from matplotlib import colors

    mask = _top_n_mask(scores, top_n)
    background_size, highlight_size = _point_sizes(len(coords), point_scale)
    ax.scatter(
        coords[~mask, 0],
        coords[~mask, 1],
        c=BACKGROUND_COLOR,
        s=background_size,
        linewidths=0,
        rasterized=True,
    )
    selected_scores = scores[mask]
    low, high = float(selected_scores.min()), float(selected_scores.max())
    if math.isclose(low, high):
        pad = max(abs(low) * 1e-6, 1e-12)
        low, high = low - pad, high + pad
    norm = colors.Normalize(vmin=low, vmax=high)
    # Draw from lower to higher scores so the strongest cells remain visible.
    selected_indices = np.flatnonzero(mask)
    selected_indices = selected_indices[
        np.argsort(scores[selected_indices], kind="stable")
    ]
    artist = ax.scatter(
        coords[selected_indices, 0],
        coords[selected_indices, 1],
        c=scores[selected_indices],
        cmap=cmap,
        norm=norm,
        s=highlight_size,
        linewidths=0,
        rasterized=True,
    )
    selection_label = selection_label or f"top {int(mask.sum()):,}"
    ax.set_title(f"{title}\n{selection_label}", fontsize=10)
    _format_axis(ax, invert_y)
    return artist


def _plot_score_comparison(
    ax: Any,
    rm_ideal: Any,
    prediction: Any,
    title: str,
    cmap: str,
) -> Any:
    """Plot prediction against RM-Ideal using hexbin density and scatter points."""
    import numpy as np

    combined = np.concatenate((rm_ideal, prediction))
    low, high = float(combined.min()), float(combined.max())
    if math.isclose(low, high):
        pad = max(abs(low) * 1e-6, 1e-12)
        low, high = low - pad, high + pad

    artist = ax.hexbin(
        rm_ideal,
        prediction,
        gridsize=65,
        mincnt=1,
        bins="log",
        cmap=cmap,
        linewidths=0,
        rasterized=True,
    )
    ax.scatter(
        rm_ideal,
        prediction,
        s=0.45,
        c="black",
        alpha=0.06,
        linewidths=0,
        rasterized=True,
    )
    ax.plot((low, high), (low, high), color="white", linewidth=2.4, alpha=0.9)
    ax.plot(
        (low, high),
        (low, high),
        color="#333333",
        linewidth=0.8,
        linestyle="--",
    )
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("RM-Ideal score")
    ax.set_ylabel("Prediction score")
    ax.set_title(f"{title}\nRM-Ideal vs. prediction", fontsize=10)
    return artist


def _plot_top_k_overlap(
    ax: Any,
    rm_ideal: Any,
    predictions: list[tuple[QueryRow, Any]],
) -> None:
    """Plot |prediction top-k intersect RM-Ideal top-k| / k."""
    import numpy as np
    from matplotlib.ticker import PercentFormatter

    n_cells = rm_ideal.size
    ranks = np.arange(1, n_cells + 1)
    candidate_fractions = ranks / n_cells
    oracle_order = np.argsort(-rm_ideal, kind="stable")
    oracle_ranks = np.empty(n_cells, dtype=int)
    oracle_ranks[oracle_order] = np.arange(n_cells)
    for row, scores in predictions:
        prediction_order = np.argsort(-scores, kind="stable")
        prediction_ranks = np.empty(n_cells, dtype=int)
        prediction_ranks[prediction_order] = np.arange(n_cells)

        # A cell enters both top-k sets when k exceeds its worse (larger) rank.
        entry_k = np.maximum(oracle_ranks, prediction_ranks) + 1
        entry_counts = np.bincount(entry_k, minlength=n_cells + 1)
        intersection_sizes = np.cumsum(entry_counts[1:])
        recall_at_k = intersection_sizes / ranks
        label = EMBEDDING_LABELS.get(row.embedding_type, row.embedding_type)
        ax.plot(
            candidate_fractions,
            recall_at_k,
            linewidth=1.8,
            label=f"{label} (query {row.query_id})",
        )

    ax.plot(
        candidate_fractions,
        np.ones(n_cells),
        color="black",
        linewidth=1.5,
        linestyle="--",
        label="RM-Ideal ranking (oracle)",
    )
    ax.plot(
        candidate_fractions,
        candidate_fractions,
        color="#888888",
        linewidth=1.2,
        linestyle=":",
        label="Random-ranking expectation",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_xlabel("Fraction of candidates retrieved (k/N)")
    ax.set_ylabel("Top-k overlap / Recall@k")
    ax.set_title(
        "Top-k overlap / Recall@k: |prediction top-k ∩ RM-Ideal top-k| / k",
        fontsize=11,
    )
    ax.grid(alpha=0.2, linewidth=0.6)
    ax.legend(loc="lower right", frameon=False, fontsize=8)


def draw_representative_query(
    *,
    query_id: int,
    query_manifest: Path,
    niche_manifest: Path,
    data_dir: Path,
    raw_results_dir: Path,
    output: Path,
    coordinate_key: str = "spatial",
    parcellation_obs_key: str = "parcellation_index",
    top_n: int = 200,
    top_percent: float = 1.0,
    point_scale: float = 1.0,
    cmap: str = "plasma",
    dpi: int = 220,
    invert_y: bool | None = None,
) -> Path:
    import anndata as ad
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if not math.isfinite(top_percent) or not 0 < top_percent <= 100:
        raise ValueError("top_percent must be finite and in the interval (0, 100]")
    if point_scale <= 0:
        raise ValueError("point_scale must be positive")

    selected, group = select_query_group(load_query_rows(query_manifest), query_id)
    niche = load_niche_row(niche_manifest, selected.query_niche_id)
    if niche.source_slice != selected.source_slice:
        raise ValueError("Source slice differs between query and niche manifests")

    source_path = data_dir / f"{selected.source_slice}.h5ad"
    target_path = data_dir / f"{selected.target_slice}.h5ad"
    for path in (source_path, target_path):
        if not path.is_file():
            raise FileNotFoundError(f"AnnData file not found: {path}")

    source = ad.read_h5ad(source_path)
    target = source if source_path == target_path else ad.read_h5ad(target_path)
    try:
        source_coords = _coordinates(source, coordinate_key, selected.source_slice)
        target_coords = _coordinates(target, coordinate_key, selected.target_slice)
        niche_mask = _binary_niche_mask(source, niche)
        source_parcellations = _parcellation_values(
            source, selected.source_slice, parcellation_obs_key
        )
        target_parcellations = _parcellation_values(
            target, selected.target_slice, parcellation_obs_key
        )
        target_cell_ids = [str(cell_id) for cell_id in target.obs_names]

        predictions: list[tuple[QueryRow, Any]] = []
        group_rm_ideal: list[tuple[QueryRow, Any]] = []
        rm_ideal = None
        for row in group:
            result_path = raw_results_dir / f"query_{row.query_id}.csv"
            prediction, row_rm_ideal = load_result_scores(
                result_path, row.query_id, target_cell_ids
            )
            predictions.append((row, prediction))
            group_rm_ideal.append((row, row_rm_ideal))
            if row.query_id == selected.query_id:
                rm_ideal = row_rm_ideal
        if rm_ideal is None:
            raise RuntimeError(f"No RM-Ideal scores loaded for query_id={query_id}")

        # RM-Ideal is embedding-independent; flag inconsistent result files,
        # which usually means one member of the group is stale.
        for row, comparison_rm in group_rm_ideal:
            if row.query_id == selected.query_id:
                continue
            if not np.allclose(rm_ideal, comparison_rm, rtol=1e-7, atol=1e-10):
                warnings.warn(
                    f"RM-Ideal scores differ between query_{query_id}.csv and "
                    f"query_{row.query_id}.csv; plotting RM-Ideal from the "
                    "requested query ID",
                    stacklevel=2,
                )

        ncols = max(3, len(predictions) + 1)
        fig = plt.figure(
            figsize=(4.2 * ncols, 20.0),
            constrained_layout=True,
        )
        grid = fig.add_gridspec(5, ncols)
        axes = np.empty((4, ncols), dtype=object)
        for row_index in range(4):
            for column in range(ncols):
                axes[row_index, column] = fig.add_subplot(grid[row_index, column])
        top_k_overlap_ax = fig.add_subplot(grid[4, :])
        should_invert = coordinate_key == "spatial" if invert_y is None else invert_y

        _plot_binary(
            axes[0, 0],
            source_coords,
            niche_mask,
            f"Source: {selected.source_slice}",
            point_scale,
            should_invert,
        )
        _plot_parcellations(
            axes[0, 1],
            source_coords,
            source_parcellations,
            niche,
            f"Source: {selected.source_slice}",
            point_scale,
            should_invert,
        )
        _plot_parcellations(
            axes[0, 2],
            target_coords,
            target_parcellations,
            niche,
            f"Target: {selected.target_slice}",
            point_scale,
            should_invert,
        )
        for column in range(3, ncols):
            axes[0, column].set_axis_off()

        rm_artist = _plot_scores(
            axes[1, 0],
            target_coords,
            rm_ideal,
            "RM-Ideal score",
            top_n,
            point_scale,
            cmap,
            should_invert,
        )
        fig.colorbar(
            rm_artist, ax=axes[1, 0], shrink=0.72, pad=0.02, label="RM-Ideal score"
        )
        for column, (row, scores) in enumerate(predictions, start=1):
            label = EMBEDDING_LABELS.get(row.embedding_type, row.embedding_type)
            artist = _plot_scores(
                axes[1, column],
                target_coords,
                scores,
                f"Prediction: {label} (query {row.query_id})",
                top_n,
                point_scale,
                cmap,
                should_invert,
            )
            fig.colorbar(
                artist,
                ax=axes[1, column],
                shrink=0.72,
                pad=0.02,
                label="Prediction score",
            )
        for column in range(len(predictions) + 1, ncols):
            axes[1, column].set_axis_off()

        top_percent_n = _top_percent_count(target.n_obs, top_percent)
        percent_label = f"top {top_percent:g}% (n={top_percent_n:,})"
        rm_percent_artist = _plot_scores(
            axes[2, 0],
            target_coords,
            rm_ideal,
            "RM-Ideal score",
            top_percent_n,
            point_scale,
            cmap,
            should_invert,
            selection_label=percent_label,
        )
        fig.colorbar(
            rm_percent_artist,
            ax=axes[2, 0],
            shrink=0.72,
            pad=0.02,
            label="RM-Ideal score",
        )
        for column, (row, scores) in enumerate(predictions, start=1):
            label = EMBEDDING_LABELS.get(row.embedding_type, row.embedding_type)
            artist = _plot_scores(
                axes[2, column],
                target_coords,
                scores,
                f"Prediction: {label} (query {row.query_id})",
                top_percent_n,
                point_scale,
                cmap,
                should_invert,
                selection_label=percent_label,
            )
            fig.colorbar(
                artist,
                ax=axes[2, column],
                shrink=0.72,
                pad=0.02,
                label="Prediction score",
            )
        for column in range(len(predictions) + 1, ncols):
            axes[2, column].set_axis_off()

        axes[3, 0].set_axis_off()
        for column, (row, scores) in enumerate(predictions, start=1):
            label = EMBEDDING_LABELS.get(row.embedding_type, row.embedding_type)
            artist = _plot_score_comparison(
                axes[3, column],
                rm_ideal,
                scores,
                f"{label} (query {row.query_id})",
                cmap,
            )
            fig.colorbar(
                artist,
                ax=axes[3, column],
                shrink=0.72,
                pad=0.02,
                label="Cells per hexbin (log scale)",
            )
        for column in range(len(predictions) + 1, ncols):
            axes[3, column].set_axis_off()

        _plot_top_k_overlap(top_k_overlap_ax, rm_ideal, predictions)

        fig.suptitle(
            f"Representative niche query {query_id}: {selected.query_niche_id}\n"
            f"source={selected.source_slice}, target={selected.target_slice}, "
            f"k={selected.niche_query_k}, coordinates={coordinate_key}",
            fontsize=13,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
    finally:
        if target is not source and getattr(target, "isbacked", False):
            target.file.close()
        if getattr(source, "isbacked", False):
            source.file.close()
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-id", type=int, required=True)
    parser.add_argument("--query-manifest", type=Path, default=DEFAULT_QUERY_MANIFEST)
    parser.add_argument("--niche-manifest", type=Path, default=DEFAULT_NICHE_MANIFEST)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--raw-results-dir", type=Path, default=DEFAULT_RAW_RESULTS_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output PNG path (default: experiment/evaluation_results/drawings/"
            "representative_query_<query-id>.png)."
        ),
    )
    parser.add_argument(
        "--coordinate-key",
        default="spatial",
        help="AnnData obsm key containing 2-D coordinates (default: spatial).",
    )
    parser.add_argument("--parcellation-obs-key", default="parcellation_index")
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument(
        "--top-percent",
        type=float,
        default=1.0,
        help="Percentage of top-ranked cells shown in the additional score row (default: 1).",
    )
    parser.add_argument("--point-scale", type=float, default=1.0)
    parser.add_argument("--cmap", default="plasma")
    parser.add_argument("--dpi", type=int, default=220)
    orientation = parser.add_mutually_exclusive_group()
    orientation.add_argument("--invert-y", dest="invert_y", action="store_true")
    orientation.add_argument("--no-invert-y", dest="invert_y", action="store_false")
    parser.set_defaults(invert_y=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = (
        args.output or DEFAULT_OUTPUT_DIR / f"representative_query_{args.query_id}.png"
    )
    result = draw_representative_query(
        query_id=args.query_id,
        query_manifest=args.query_manifest.resolve(),
        niche_manifest=args.niche_manifest.resolve(),
        data_dir=args.data_dir.resolve(),
        raw_results_dir=args.raw_results_dir.resolve(),
        output=output.resolve(),
        coordinate_key=args.coordinate_key,
        parcellation_obs_key=args.parcellation_obs_key,
        top_n=args.top_n,
        top_percent=args.top_percent,
        point_scale=args.point_scale,
        cmap=args.cmap,
        dpi=args.dpi,
        invert_y=args.invert_y,
    )
    print(f"Wrote {result}")


if __name__ == "__main__":
    main()
