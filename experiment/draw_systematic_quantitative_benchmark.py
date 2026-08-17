"""Draw task-level metric distributions for the systematic benchmark.

The per-query evaluation table does not contain embedding labels, so this
script joins it to ``manifests/query_manifest.csv`` by ``query_id`` and draws
a multi-panel figure containing every metric. Each violin overlays one point
per query.

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
DEFAULT_MANIFEST = EXPERIMENT_DIR / "manifests" / "query_manifest.csv"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "evaluation_results" / "drawings"
DEFAULT_OUTPUT_NAME = "all_metrics_violin_by_embedding.png"
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
    """Load all per-query metrics and group them by metric and embedding."""
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
                    raise ValueError(
                        f"{path}:{line_number}: {metric} must be finite"
                    )
                values[metric][embedding].append(metric_value)

    if not values:
        raise ValueError(f"Evaluation metrics have no data rows: {path}")
    return {
        metric: dict(values_by_embedding)
        for metric, values_by_embedding in values.items()
    }


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
        present_embeddings = _validate_embeddings(
            values_by_embedding, embedding_order
        )
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--query-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / DEFAULT_OUTPUT_NAME,
        help="Output image path (default: %(default)s)",
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


if __name__ == "__main__":
    main()
