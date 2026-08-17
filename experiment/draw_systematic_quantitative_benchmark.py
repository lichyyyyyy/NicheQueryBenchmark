"""Draw task-level metric distributions for the systematic benchmark.

The per-query evaluation table does not contain embedding labels, so this
script joins it to ``manifests/query_manifest.csv`` by ``query_id`` and draws
a multi-panel figure containing every metric. Each violin overlays one point
per query. All evaluation rows are included, including queries whose source
slice is the same as their target slice.

Example
-------
Run with the repository defaults::

    python experiment/draw_systematic_quantitative_benchmark.py
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_METRICS = (
    EXPERIMENT_DIR / "evaluation_results" / "evaluation_metrics_per_query.csv"
)
DEFAULT_AGGREGATED_METRICS = (
    EXPERIMENT_DIR / "evaluation_results" / "evaluation_metrics_by_embedding_type.csv"
)
DEFAULT_MANIFEST = EXPERIMENT_DIR / "manifests" / "query_manifest.csv"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "evaluation_results" / "drawings"
DEFAULT_OUTPUT_NAME = "all_metrics_violin_by_embedding.png"
DEFAULT_HEATMAP_OUTPUT_NAME = "all_metrics_by_niche_and_embedding_heatmaps.png"
DEFAULT_DOT_PLOT_OUTPUT_NAME = "aggregated_metrics_cleveland_dot_plot.png"
DEFAULT_CONFIDENCE_LEVEL = 95.0
BOOTSTRAP_RESAMPLES = 10_000

METRIC_LABELS = {
    "pearson": "Pearson correlation",
    "spearman": "Spearman correlation",
    "ndcg_at_0_5_pct": "NDCG @ 0.5%",
    "ndcg_at_1pct": "NDCG @ 1%",
    "ndcg_at_5pct": "NDCG @ 5%",
    "ndcg_at_top_200": "NDCG @ top 200",
    "enrichment_at_0_5pct": "Normalized enrichment @ 0.5%",
    "enrichment_at_1pct": "Normalized enrichment @ 1%",
    "enrichment_at_5pct": "Normalized enrichment @ 5%",
    "enrichment_at_top_200": "Normalized enrichment @ top 200",
    "recall_at_0_5pct": "Recall @ 0.5%",
    "recall_at_1pct": "Recall @ 1%",
    "recall_at_5pct": "Recall @ 5%",
    "recall_at_top_200": "Recall @ top 200",
}

# Keep the benchmark's manifest order while using presentation-friendly labels.
EMBEDDING_LABELS = {
    "gene_expr_quest": "Gene expression\n+ QUEST",
    "scgpt": "scGPT",
    "gene_expr": "Gene expression",
}
COLORS = ("#4C78A8", "#F58518", "#54A24B")
MARKERS = ("o", "s", "D")

DOT_PLOT_METRIC_LABELS = {
    "pearson": "Pearson",
    "spearman": "Spearman",
    "ndcg_at_0_5_pct": "NDCG @ 0.5%",
    "ndcg_at_1pct": "NDCG @ 1%",
    "ndcg_at_5pct": "NDCG @ 5%",
    "ndcg_at_top_200": "NDCG @ top 200",
    "enrichment_at_0_5pct": "Enrichment @ 0.5%",
    "enrichment_at_1pct": "Enrichment @ 1%",
    "enrichment_at_5pct": "Enrichment @ 5%",
    "enrichment_at_top_200": "Enrichment @ top 200",
    "recall_at_0_5pct": "Recall @ 0.5%",
    "recall_at_1pct": "Recall @ 1%",
    "recall_at_5pct": "Recall @ 5%",
    "recall_at_top_200": "Recall @ top 200",
}


def load_embedding_by_query(path: Path) -> tuple[dict[str, str], list[str]]:
    """Return query-to-embedding mapping and embedding order from the manifest."""
    if not path.is_file():
        raise FileNotFoundError(f"Query manifest not found: {path}")

    embedding_by_query: dict[str, str] = {}
    embedding_order: list[str] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"query_id", "embedding_type"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        for line_number, row in enumerate(reader, start=2):
            query_id = (row.get("query_id") or "").strip()
            embedding = (row.get("embedding_type") or "").strip()
            if not query_id or not embedding:
                raise ValueError(
                    f"{path}:{line_number}: query_id and embedding_type are required"
                )
            if query_id in embedding_by_query:
                raise ValueError(f"{path}:{line_number}: duplicate query_id {query_id}")
            embedding_by_query[query_id] = embedding
            if embedding not in embedding_order:
                embedding_order.append(embedding)

    if not embedding_by_query:
        raise ValueError(f"Query manifest has no data rows: {path}")
    return embedding_by_query, embedding_order


def load_metrics_by_embedding(
    path: Path, embedding_by_query: dict[str, str]
) -> dict[str, dict[str, list[float]]]:
    """Group all queries by metric and embedding without filtering slice pairs."""
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation metrics not found: {path}")

    values: dict[str, dict[str, list[float]]] = {
        metric: defaultdict(list) for metric in METRIC_LABELS
    }
    seen_query_ids: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"query_id", *METRIC_LABELS}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        for line_number, row in enumerate(reader, start=2):
            # Intentionally retain every evaluation query, including manifest
            # entries whose source_slice equals target_slice.
            query_id = (row.get("query_id") or "").strip()
            if query_id in seen_query_ids:
                raise ValueError(f"{path}:{line_number}: duplicate query_id {query_id}")
            seen_query_ids.add(query_id)
            if query_id not in embedding_by_query:
                raise ValueError(
                    f"{path}:{line_number}: query_id {query_id!r} is absent from the manifest"
                )
            embedding = embedding_by_query[query_id]
            for metric in METRIC_LABELS:
                try:
                    metric_value = float(row[metric])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{path}:{line_number}: {metric} must be numeric"
                    ) from exc
                if not math.isfinite(metric_value):
                    raise ValueError(f"{path}:{line_number}: {metric} must be finite")
                values[metric][embedding].append(metric_value)

    if not values:
        raise ValueError(f"Evaluation metrics have no data rows: {path}")
    return {
        metric: dict(values_by_embedding)
        for metric, values_by_embedding in values.items()
    }


def load_aggregated_metrics(
    path: Path,
) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Load one aggregated value per metric and embedding type."""
    if not path.is_file():
        raise FileNotFoundError(f"Aggregated evaluation metrics not found: {path}")

    values: dict[str, dict[str, float]] = {metric: {} for metric in METRIC_LABELS}
    embedding_order: list[str] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"embedding_type", *METRIC_LABELS}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        for line_number, row in enumerate(reader, start=2):
            embedding = (row.get("embedding_type") or "").strip()
            if not embedding:
                raise ValueError(f"{path}:{line_number}: embedding_type is required")
            if embedding in embedding_order:
                raise ValueError(
                    f"{path}:{line_number}: duplicate embedding_type {embedding!r}"
                )
            embedding_order.append(embedding)

            for metric in METRIC_LABELS:
                try:
                    value = float(row[metric])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{path}:{line_number}: {metric} must be numeric"
                    ) from exc
                if not math.isfinite(value):
                    raise ValueError(f"{path}:{line_number}: {metric} must be finite")
                values[metric][embedding] = value

    if len(embedding_order) != 3:
        raise ValueError(
            "Expected exactly 3 embedding types in the aggregated metrics; "
            f"found {len(embedding_order)}: {embedding_order}"
        )
    return values, embedding_order


def load_niche_embedding_metrics(
    metrics_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, np.ndarray], list[str], list[str], list[int]]:
    """Return mean metrics and task counts for each niche and embedding.

    Rows and columns follow their first-appearance order in the query manifest.
    Task counts are totals per niche across all embedding types. Same-source/
    target-slice queries are intentionally included.
    """
    query_metadata: dict[str, tuple[str, str]] = {}
    niche_order: list[str] = []
    embedding_order: list[str] = []
    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"query_id", "query_niche_id", "embedding_type"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{manifest_path} is missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            # No source/target-slice exclusion: every evaluation row contributes
            # to its niche-by-embedding mean and niche task count.
            query_id = (row.get("query_id") or "").strip()
            niche = (row.get("query_niche_id") or "").strip()
            embedding = (row.get("embedding_type") or "").strip()
            if not query_id or not niche or not embedding:
                raise ValueError(
                    f"{manifest_path}:{line_number}: query_id, query_niche_id, "
                    "and embedding_type are required"
                )
            if query_id in query_metadata:
                raise ValueError(
                    f"{manifest_path}:{line_number}: duplicate query_id {query_id}"
                )
            query_metadata[query_id] = (niche, embedding)
            if niche not in niche_order:
                niche_order.append(niche)
            if embedding not in embedding_order:
                embedding_order.append(embedding)

    if len(niche_order) != 8 or len(embedding_order) != 3:
        raise ValueError(
            "Expected an 8 niche x 3 embedding benchmark; found "
            f"{len(niche_order)} niches x {len(embedding_order)} embeddings"
        )

    grouped_values: dict[str, dict[tuple[str, str], list[float]]] = {
        metric: defaultdict(list) for metric in METRIC_LABELS
    }
    task_counts: dict[str, int] = defaultdict(int)
    seen_query_ids: set[str] = set()
    with metrics_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"query_id", *METRIC_LABELS}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{metrics_path} is missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            query_id = (row.get("query_id") or "").strip()
            if query_id in seen_query_ids:
                raise ValueError(
                    f"{metrics_path}:{line_number}: duplicate query_id {query_id}"
                )
            seen_query_ids.add(query_id)
            if query_id not in query_metadata:
                raise ValueError(
                    f"{metrics_path}:{line_number}: query_id {query_id!r} is absent "
                    "from the manifest"
                )
            niche, embedding = query_metadata[query_id]
            for metric in METRIC_LABELS:
                try:
                    metric_value = float(row[metric])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{metrics_path}:{line_number}: {metric} must be numeric"
                    ) from exc
                if not math.isfinite(metric_value):
                    raise ValueError(
                        f"{metrics_path}:{line_number}: {metric} must be finite"
                    )
                grouped_values[metric][(niche, embedding)].append(metric_value)
            task_counts[niche] += 1

    matrices: dict[str, np.ndarray] = {}
    for metric in METRIC_LABELS:
        matrix = np.empty((len(niche_order), len(embedding_order)), dtype=np.float64)
        for row_index, niche in enumerate(niche_order):
            for column_index, embedding in enumerate(embedding_order):
                values = grouped_values[metric][(niche, embedding)]
                if not values:
                    raise ValueError(
                        f"No {metric} results for niche={niche!r}, "
                        f"embedding={embedding!r}"
                    )
                matrix[row_index, column_index] = np.mean(values)
        matrices[metric] = matrix

    return (
        matrices,
        niche_order,
        embedding_order,
        [task_counts[niche] for niche in niche_order],
    )


def _validate_embeddings(
    values_by_embedding: dict[str, list[float]], embedding_order: list[str]
) -> list[str]:
    """Return the three embeddings represented in the metric data."""
    present_embeddings = [
        embedding for embedding in embedding_order if embedding in values_by_embedding
    ]
    if len(present_embeddings) != 3:
        raise ValueError(
            "Expected exactly 3 embedding types in the evaluation results; "
            f"found {len(present_embeddings)}: {present_embeddings}"
        )
    return present_embeddings


def _draw_violin_panel(
    ax: plt.Axes,
    values_by_embedding: dict[str, list[float]],
    present_embeddings: list[str],
    title: str,
    point_size: float,
    seed: int,
    confidence_level: float,
) -> None:
    """Draw violins, observations, and a bootstrap CI for the mean."""
    distributions = [values_by_embedding[name] for name in present_embeddings]
    positions = np.arange(1, len(present_embeddings) + 1)

    violins = ax.violinplot(
        distributions,
        positions=positions,
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method="scott",
    )
    for body, color in zip(violins["bodies"], COLORS):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.28)
        body.set_linewidth(1.2)
    rng = np.random.default_rng(seed)
    for position, distribution, color in zip(positions, distributions, COLORS):
        jitter = rng.uniform(-0.20, 0.20, size=len(distribution))
        ax.scatter(
            position + jitter,
            distribution,
            s=point_size,
            color=color,
            alpha=0.72,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )

        samples = np.asarray(distribution, dtype=np.float64)
        bootstrap_means = rng.choice(
            samples,
            size=(BOOTSTRAP_RESAMPLES, samples.size),
            replace=True,
        ).mean(axis=1)
        tail_probability = (100.0 - confidence_level) / 2.0
        lower, upper = np.percentile(
            bootstrap_means,
            [tail_probability, 100.0 - tail_probability],
        )
        mean = float(np.mean(samples))
        ax.errorbar(
            position,
            mean,
            yerr=[[mean - lower], [upper - mean]],
            fmt="o",
            color="#222222",
            markerfacecolor="white",
            markeredgewidth=1.3,
            markersize=5,
            capsize=5,
            elinewidth=1.8,
            capthick=1.8,
            zorder=4,
        )

    labels = [
        EMBEDDING_LABELS.get(name, name.replace("_", " "))
        for name in present_embeddings
    ]
    ax.set_xticks(positions, labels)
    ax.set_title(title, fontsize=12, pad=8, weight="bold")
    if min(min(distribution) for distribution in distributions) < 0:
        ax.axhline(0.0, color="#777777", linewidth=0.8, linestyle="--", zorder=0)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=8.5)


def draw_all_metrics_violin(
    metrics: dict[str, dict[str, list[float]]],
    embedding_order: list[str],
    output_path: Path,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
) -> None:
    """Draw every task-level metric in one large multi-panel figure."""
    if not 0.0 < confidence_level < 100.0:
        raise ValueError("confidence_level must be between 0 and 100")

    n_columns = 4
    n_rows = math.ceil(len(METRIC_LABELS) / n_columns)
    fig, axes = plt.subplots(
        n_rows,
        n_columns,
        figsize=(20, 18),
    )
    # Correlations share the centered first row. Each four-metric family gets
    # its own row so related cutoffs can be compared horizontally.
    panel_locations = {
        "pearson": (0, 1),
        "spearman": (0, 2),
        "ndcg_at_0_5_pct": (1, 0),
        "ndcg_at_1pct": (1, 1),
        "ndcg_at_5pct": (1, 2),
        "ndcg_at_top_200": (1, 3),
        "enrichment_at_0_5pct": (2, 0),
        "enrichment_at_1pct": (2, 1),
        "enrichment_at_5pct": (2, 2),
        "enrichment_at_top_200": (2, 3),
        "recall_at_0_5pct": (3, 0),
        "recall_at_1pct": (3, 1),
        "recall_at_5pct": (3, 2),
        "recall_at_top_200": (3, 3),
    }
    used_locations = set(panel_locations.values())

    for index, (metric, title) in enumerate(METRIC_LABELS.items()):
        values_by_embedding = metrics[metric]
        present_embeddings = _validate_embeddings(values_by_embedding, embedding_order)
        _draw_violin_panel(
            axes[panel_locations[metric]],
            values_by_embedding,
            present_embeddings,
            title=title,
            point_size=10,
            seed=2026 + index,
            confidence_level=confidence_level,
        )

    for row in range(n_rows):
        for column in range(n_columns):
            if (row, column) not in used_locations:
                axes[row, column].set_visible(False)

    fig.suptitle(
        "Systematic Quantitative Benchmark: Task-level Metrics\n"
        f"Mean with {confidence_level:g}% bootstrap confidence interval",
        fontsize=22,
        weight="bold",
        y=0.975,
    )
    fig.supxlabel("Embedding", fontsize=15, y=0.018)
    fig.supylabel("Task-level metric", fontsize=15, x=0.018)
    fig.subplots_adjust(
        left=0.06,
        right=0.985,
        bottom=0.065,
        top=0.92,
        hspace=0.38,
        wspace=0.28,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_aggregated_metrics_dot_plot(
    metrics: dict[str, dict[str, float]],
    embedding_order: list[str],
    output_path: Path,
) -> None:
    """Draw a grouped Cleveland dot plot of all aggregated metrics."""
    missing_metrics = set(METRIC_LABELS) - set(metrics)
    if missing_metrics:
        raise ValueError(f"Missing aggregated metrics: {sorted(missing_metrics)}")
    if len(embedding_order) != 3 or len(set(embedding_order)) != 3:
        raise ValueError("Dot plot requires exactly 3 unique embedding types")

    for metric in METRIC_LABELS:
        missing_embeddings = set(embedding_order) - set(metrics[metric])
        if missing_embeddings:
            raise ValueError(
                f"{metric} is missing embeddings: {sorted(missing_embeddings)}"
            )

    metric_names = list(METRIC_LABELS)
    y_positions = np.arange(len(metric_names))
    all_values = np.asarray(
        [
            metrics[metric][embedding]
            for metric in metric_names
            for embedding in embedding_order
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(all_values)):
        raise ValueError("All aggregated metric values must be finite")

    fig, ax = plt.subplots(figsize=(11, 9.5))

    # The connector makes both the spread and the ordering of embeddings easy
    # to scan, while the points retain each embedding's exact position.
    for y_position, metric in zip(y_positions, metric_names):
        row_values = [metrics[metric][name] for name in embedding_order]
        ax.hlines(
            y_position,
            min(row_values),
            max(row_values),
            color="#B8BDC5",
            linewidth=2.0,
            zorder=1,
        )

    for embedding, color, marker in zip(embedding_order, COLORS, MARKERS):
        values = [metrics[metric][embedding] for metric in metric_names]
        legend_label = EMBEDDING_LABELS.get(
            embedding, embedding.replace("_", " ")
        ).replace("\n", " ")
        ax.scatter(
            values,
            y_positions,
            s=72,
            color=color,
            marker=marker,
            edgecolors="white",
            linewidths=0.9,
            label=legend_label,
            zorder=3,
        )

    ax.set_yticks(
        y_positions,
        [DOT_PLOT_METRIC_LABELS[metric] for metric in metric_names],
    )
    ax.invert_yaxis()
    ax.set_xlabel("Aggregated metric value", fontsize=12)
    fig.suptitle(
        "Aggregated Metrics by Embedding",
        fontsize=17,
        weight="bold",
        y=0.98,
    )
    ax.grid(axis="x", color="#D9DDE2", linewidth=0.8)
    ax.set_axisbelow(True)
    if float(np.min(all_values)) < 0.0:
        ax.axvline(0.0, color="#777777", linewidth=0.9, linestyle="--", zorder=0)
    else:
        ax.set_xlim(left=0.0)
    ax.margins(x=0.06, y=0.025)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, labelsize=10)
    ax.tick_params(axis="x", labelsize=9)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
        ncols=3,
        frameon=False,
        fontsize=10,
        handletextpad=0.5,
        columnspacing=1.8,
    )

    fig.subplots_adjust(left=0.24, right=0.98, bottom=0.08, top=0.86)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_all_metrics_heatmaps(
    mean_metrics: dict[str, np.ndarray],
    niche_order: list[str],
    embedding_order: list[str],
    task_counts: list[int],
    output_path: Path,
) -> None:
    """Draw all metric means as annotated niche-by-embedding heatmaps."""
    expected_shape = (len(niche_order), len(embedding_order))
    if len(niche_order) != 8 or len(embedding_order) != 3:
        raise ValueError("Heatmaps require exactly 8 niches and 3 embeddings")
    if len(task_counts) != len(niche_order):
        raise ValueError("task_counts must contain one value per niche")
    missing_metrics = set(METRIC_LABELS) - set(mean_metrics)
    if missing_metrics:
        raise ValueError(f"Missing heatmap metrics: {sorted(missing_metrics)}")
    for metric, matrix in mean_metrics.items():
        if metric in METRIC_LABELS and matrix.shape != expected_shape:
            raise ValueError(
                f"{metric} has shape {matrix.shape}; expected {expected_shape}"
            )

    panel_locations = {
        "pearson": (0, 1),
        "spearman": (0, 2),
        "ndcg_at_0_5_pct": (1, 0),
        "ndcg_at_1pct": (1, 1),
        "ndcg_at_5pct": (1, 2),
        "ndcg_at_top_200": (1, 3),
        "enrichment_at_0_5pct": (2, 0),
        "enrichment_at_1pct": (2, 1),
        "enrichment_at_5pct": (2, 2),
        "enrichment_at_top_200": (2, 3),
        "recall_at_0_5pct": (3, 0),
        "recall_at_1pct": (3, 1),
        "recall_at_5pct": (3, 2),
        "recall_at_top_200": (3, 3),
    }
    families = {
        "Correlation": ("pearson", "spearman"),
        "NDCG": (
            "ndcg_at_0_5_pct",
            "ndcg_at_1pct",
            "ndcg_at_5pct",
            "ndcg_at_top_200",
        ),
        "Normalized enrichment": (
            "enrichment_at_0_5pct",
            "enrichment_at_1pct",
            "enrichment_at_5pct",
            "enrichment_at_top_200",
        ),
        "Recall": (
            "recall_at_0_5pct",
            "recall_at_1pct",
            "recall_at_5pct",
            "recall_at_top_200",
        ),
    }

    family_styles: dict[str, tuple[str, float, float]] = {}
    for family, family_metrics in families.items():
        family_values = np.concatenate(
            [mean_metrics[metric].ravel() for metric in family_metrics]
        )
        if family in {"Correlation", "Normalized enrichment"}:
            limit = max(float(np.max(np.abs(family_values))), 1e-12)
            family_styles[family] = ("RdBu_r", -limit, limit)
        else:
            maximum = max(float(np.max(family_values)), 1e-12)
            family_styles[family] = ("YlGnBu", 0.0, maximum)

    fig, axes = plt.subplots(4, 4, figsize=(28, 24))

    embedding_labels = [
        EMBEDDING_LABELS.get(name, name.replace("_", " ")) for name in embedding_order
    ]
    row_label_axes = {0: (0, 1), 1: (1, 0), 2: (2, 0), 3: (3, 0)}
    family_images: dict[str, plt.AxesImage] = {}

    for family, family_metrics in families.items():
        cmap, vmin, vmax = family_styles[family]
        for metric in family_metrics:
            row, column = panel_locations[metric]
            ax = axes[row, column]
            matrix = mean_metrics[metric]
            image = ax.imshow(
                matrix,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                aspect="auto",
            )
            family_images[family] = image
            ax.set_title(METRIC_LABELS[metric], fontsize=12, weight="bold", pad=8)
            ax.set_xticks(np.arange(len(embedding_order)), embedding_labels)
            if (row, column) == row_label_axes[row]:
                labels = [
                    f"{niche}  [n={count}]"
                    for niche, count in zip(niche_order, task_counts)
                ]
                ax.set_yticks(np.arange(len(niche_order)), labels)
                ax.tick_params(axis="y", labelsize=7.5)
            else:
                ax.set_yticks(np.arange(len(niche_order)), [""] * len(niche_order))

            for row_index in range(matrix.shape[0]):
                for column_index in range(matrix.shape[1]):
                    value = matrix[row_index, column_index]
                    normalized = image.norm(value)
                    if cmap == "RdBu_r":
                        text_color = (
                            "white" if abs(normalized - 0.5) > 0.30 else "#222222"
                        )
                    else:
                        text_color = "white" if normalized > 0.58 else "#222222"
                    ax.text(
                        column_index,
                        row_index,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=7.5,
                        weight="bold",
                    )

            ax.set_xticks(np.arange(-0.5, len(embedding_order), 1), minor=True)
            ax.set_yticks(np.arange(-0.5, len(niche_order), 1), minor=True)
            ax.grid(which="minor", color="white", linewidth=1.4)
            ax.tick_params(which="minor", bottom=False, left=False)
            ax.tick_params(axis="x", labelsize=7.5)
            for spine in ax.spines.values():
                spine.set_visible(False)

    used_locations = set(panel_locations.values())
    for row in range(4):
        for column in range(4):
            if (row, column) not in used_locations:
                axes[row, column].set_visible(False)

    fig.suptitle(
        "Mean Task-level Metrics by Query Niche and Embedding",
        fontsize=24,
        weight="bold",
        y=0.975,
    )
    fig.supxlabel("Embedding", fontsize=16, y=0.018)
    fig.supylabel(
        "query_niche_id [total query tasks]",
        fontsize=16,
        x=0.012,
    )
    fig.subplots_adjust(
        left=0.25,
        right=0.94,
        bottom=0.06,
        top=0.92,
        hspace=0.34,
        wspace=0.22,
    )

    # One shared scale per metric family keeps the panels comparable without
    # crowding the figure with fourteen separate colorbars.
    for family, family_metrics in families.items():
        family_axes = [axes[panel_locations[metric]] for metric in family_metrics]
        boxes = [ax.get_position() for ax in family_axes]
        bottom = min(box.y0 for box in boxes)
        top = max(box.y1 for box in boxes)
        colorbar_ax = fig.add_axes([0.955, bottom, 0.009, top - bottom])
        colorbar = fig.colorbar(family_images[family], cax=colorbar_ax)
        colorbar.set_label(f"Mean {family.lower()}", fontsize=10, labelpad=10)
        colorbar.ax.tick_params(labelsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--query-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--aggregated-metrics",
        type=Path,
        default=DEFAULT_AGGREGATED_METRICS,
        help="Aggregated metrics CSV for the Cleveland dot plot (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / DEFAULT_OUTPUT_NAME,
        help="Output image path (default: %(default)s)",
    )
    parser.add_argument(
        "--dot-output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / DEFAULT_DOT_PLOT_OUTPUT_NAME,
        help="Cleveland dot plot output path (default: %(default)s)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embedding_by_query, embedding_order = load_embedding_by_query(
        args.query_manifest.resolve()
    )
    metrics = load_metrics_by_embedding(args.metrics.resolve(), embedding_by_query)
    output_path = args.output.resolve()
    draw_all_metrics_violin(
        metrics,
        embedding_order,
        output_path,
        confidence_level=DEFAULT_CONFIDENCE_LEVEL,
    )
    print(
        f"Wrote {output_path} ({len(METRIC_LABELS)} metric panels, "
        f"{DEFAULT_CONFIDENCE_LEVEL:g}% confidence intervals)"
    )

    mean_heatmap_metrics, niche_order, heatmap_embeddings, task_counts = (
        load_niche_embedding_metrics(
            args.metrics.resolve(),
            args.query_manifest.resolve(),
        )
    )
    heatmap_output_path = (DEFAULT_OUTPUT_DIR / DEFAULT_HEATMAP_OUTPUT_NAME).resolve()
    draw_all_metrics_heatmaps(
        mean_heatmap_metrics,
        niche_order,
        heatmap_embeddings,
        task_counts,
        heatmap_output_path,
    )
    print(
        f"Wrote {heatmap_output_path} "
        f"({len(METRIC_LABELS)} metrics, 8 niches x 3 embeddings)"
    )

    aggregated_metrics, aggregated_embedding_order = load_aggregated_metrics(
        args.aggregated_metrics.resolve()
    )
    dot_output_path = args.dot_output.resolve()
    draw_aggregated_metrics_dot_plot(
        aggregated_metrics,
        aggregated_embedding_order,
        dot_output_path,
    )
    print(
        f"Wrote {dot_output_path} "
        f"({len(METRIC_LABELS)} metrics x {len(aggregated_embedding_order)} embeddings)"
    )


if __name__ == "__main__":
    main()
