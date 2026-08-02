"""Run niche query on benchmark h5ad files using QueST embeddings."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np

from src.NicheQueryPrototype.database import Database
from src.NicheQueryPrototype.niche import Niche
from src.NicheQueryPrototype.query import NicheQuery
from add_embeddings import (
    DEFAULT_EMB_FOLDER,
    DEFAULT_H5AD_FOLDER,
    DEFAULT_OBSM_KEY,
    add_quest_embeddings,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

NICHE_OBSM_KEYS = ("query_niche_9_15",)
QUERY_NICHE_SAMPLE_ID = "Zhuang-ABCA-1.098"
EMBEDDING_OBSM_KEY = DEFAULT_OBSM_KEY
NICHE_QUERY_K = 3

SAMPLE_DATA_DIR = DEFAULT_H5AD_FOLDER
PARCELLATION_PATH = Path("notebook/ccf/data/parcellation/structure.json")
SPATIAL_OBSM_KEY = "spatial"
CCF_COORDINATES_PATHS = [
    "notebook/ccf/data/zhuang-abca-1/ccf_coordinates.csv",
    "notebook/ccf/data/zhuang-abca-2/ccf_coordinates.csv",
    "notebook/ccf/data/zhuang-abca-3/ccf_coordinates.csv",
    "notebook/ccf/data/zeng-mwb/ccf_coordinates.csv",
]

TARGET_SAMPLE_IDS = [
    "Zhuang-ABCA-1.098",
    "Zhuang-ABCA-1.099",
    "Zhuang-ABCA-1.097",
    "Zhuang-ABCA-1.096",
    "Zhuang-ABCA-3.001",
    "Zhuang-ABCA-3.002",
    "Zhuang-ABCA-3.004",
    "Zhuang-ABCA-3.005",
    "C57BL6J-638850.28",
    "C57BL6J-638850.29",
    "C57BL6J-638850.30",
    "C57BL6J-638850.31",
]

K_HOP = 10
CELL_LIMIT = 100
PCT_LO = 0.40
PCT_HI = 0.60

_NICHE_NAME_RE = re.compile(r"^query_niche_(?P<p1>\d+)_(?P<p2>\d+)$")


def parse_niche_parcellations(niche_obsm_key: str) -> tuple[int, int]:
    match = _NICHE_NAME_RE.match(niche_obsm_key)
    if match is None:
        raise ValueError(
            f"Cannot parse parcellation ids from niche key {niche_obsm_key!r}. "
            "Expected format: query_niche_<p1>_<p2>"
        )
    return int(match.group("p1")), int(match.group("p2"))


def load_sample_adatas(
    sample_dir: Path, sample_ids: list[str]
) -> dict[str, ad.AnnData]:
    adatas: dict[str, ad.AnnData] = {}
    for sid in sample_ids:
        path = sample_dir / f"{sid}.h5ad"
        if not path.exists():
            raise FileNotFoundError(f"Missing sample file: {path}")
        adatas[sid] = ad.read_h5ad(str(path))
    return adatas


def load_niche_from_obsm(db: Database, sample_id: str, niche_obsm_key: str) -> Niche:
    sample = db.get_sample(sample_id)
    if sample is None:
        raise ValueError(f"Sample not in db: {sample_id!r}")
    adata = sample.adata
    if adata is None:
        raise ValueError(f"Sample {sample_id!r} has no AnnData.")
    if niche_obsm_key not in adata.obsm:
        raise KeyError(f"obsm[{niche_obsm_key!r}] not found for {sample_id!r}.")

    mask = np.asarray(adata.obsm[niche_obsm_key], dtype=float).reshape(-1)
    if mask.shape[0] != adata.n_obs:
        raise ValueError(
            f"obsm[{niche_obsm_key!r}] length {mask.shape[0]} != n_obs {adata.n_obs} "
            f"for {sample_id!r}"
        )

    id_to_cell = {str(c.id): c for c in sample.cells}
    cells = []
    for oid, flag in zip(adata.obs_names.astype(str), mask):
        if flag > 0.5:
            cell = id_to_cell.get(oid)
            if cell is not None:
                cells.append(cell)

    if not cells:
        raise ValueError(
            f"No niche cells loaded from obsm[{niche_obsm_key!r}] for {sample_id!r}"
        )

    niche = Niche()
    niche.construct(cells, sample_id)
    niche.compute_niche_feature()
    logger.info(
        "Loaded %r from %r: %d cells, parcellations=%s",
        niche_obsm_key,
        sample_id,
        len(cells),
        niche.unique_parcellation_indices,
    )
    return niche


def construct_query_niche_if_missing(
    db: Database,
    sample_id: str,
    niche_obsm_key: str,
) -> Niche:
    sample = db.get_sample(sample_id)
    if sample is None or sample.adata is None:
        raise ValueError(f"Sample not ready: {sample_id!r}")
    if niche_obsm_key in sample.adata.obsm:
        return load_niche_from_obsm(db, sample_id, niche_obsm_key)

    parc_primary, parc_secondary = parse_niche_parcellations(niche_obsm_key)
    allowed_parcellations = [parc_primary, parc_secondary]

    cells_primary = [c for c in sample.cells if c.parcellation_index == parc_primary]
    if not cells_primary:
        raise ValueError(
            f"Sample {sample_id!r} has no cells with parcellation_index={parc_primary}"
        )

    logger.info(
        "Constructing %r on %r from %d candidate center cells (parc=%d)",
        niche_obsm_key,
        sample_id,
        len(cells_primary),
        parc_primary,
    )

    niche_to_query: Niche | None = None
    matched_center_id: str | None = None
    best_near: tuple[float, Any, float, int, int, int] | None = None

    for i, center in enumerate(cells_primary):
        trial = Niche()
        trial.construct_by_k_hop(
            db,
            center_cell_id=str(center.id),
            sample_id=sample_id,
            k=K_HOP,
            cell_limit=CELL_LIMIT,
            parcellation_index=allowed_parcellations,
        )
        n = len(trial.cells)
        n_primary = sum(1 for c in trial.cells if c.parcellation_index == parc_primary)
        n_secondary = sum(
            1 for c in trial.cells if c.parcellation_index == parc_secondary
        )
        frac_secondary = n_secondary / n if n else 0.0

        if PCT_LO <= frac_secondary <= PCT_HI:
            dist = 0.0
        else:
            dist = min(abs(frac_secondary - PCT_LO), abs(frac_secondary - PCT_HI))

        if best_near is None or dist < best_near[0]:
            best_near = (dist, center, frac_secondary, n, n_primary, n_secondary)

        if (i + 1) % 25 == 0 or PCT_LO <= frac_secondary <= PCT_HI:
            logger.info(
                "  [%d/%d] center=%s n=%d %d=%d %d=%d frac_%d=%.3f",
                i + 1,
                len(cells_primary),
                center.id,
                n,
                parc_primary,
                n_primary,
                parc_secondary,
                n_secondary,
                parc_secondary,
                frac_secondary,
            )

        if PCT_LO <= frac_secondary <= PCT_HI:
            matched_center_id = str(center.id)
            niche_to_query = Niche()
            niche_to_query.construct_by_k_hop(
                db,
                center_cell_id=matched_center_id,
                sample_id=sample_id,
                k=K_HOP,
                cell_limit=CELL_LIMIT,
                parcellation_index=allowed_parcellations,
                niche_name=niche_obsm_key,
            )
            logger.info(
                "Constructed %r on %r: center=%s n=%d frac_%d=%.3f",
                niche_obsm_key,
                sample_id,
                matched_center_id,
                len(niche_to_query.cells),
                parc_secondary,
                frac_secondary,
            )
            break

    if niche_to_query is None:
        if best_near is None:
            raise RuntimeError(
                f"Failed to construct {niche_obsm_key!r} on {sample_id!r}"
            )
        _, center, frac_secondary, n, n_primary, n_secondary = best_near
        matched_center_id = str(center.id)
        niche_to_query = Niche()
        niche_to_query.construct_by_k_hop(
            db,
            center_cell_id=matched_center_id,
            sample_id=sample_id,
            k=K_HOP,
            cell_limit=CELL_LIMIT,
            parcellation_index=allowed_parcellations,
            niche_name=niche_obsm_key,
        )
        logger.warning(
            "No center met frac_%d in [%.0f%%, %.0f%%]; using nearest match "
            "center=%s n=%d frac_%d=%.3f",
            parc_secondary,
            PCT_LO * 100,
            PCT_HI * 100,
            matched_center_id,
            n,
            parc_secondary,
            frac_secondary,
        )

    return niche_to_query


def export_samples(db: Database, sample_dir: Path, sample_ids: list[str]) -> None:
    for sample_id in sample_ids:
        sample = db.get_sample(sample_id)
        if sample is None or sample.adata is None:
            raise ValueError(f"Sample not ready for export: {sample_id!r}")
        out_path = sample_dir / f"{sample_id}.h5ad"
        sample.adata.write_h5ad(out_path)
        logger.info(
            "Wrote %s (n_obs=%d, obsm=%s)",
            out_path.name,
            sample.adata.n_obs,
            list(sample.adata.obsm.keys()),
        )


def save_open_figures(output_dir: Path, prefix: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for i, fig_num in enumerate(plt.get_fignums(), start=1):
        fig = plt.figure(fig_num)
        out_path = output_dir / f"{prefix}_viz_{i}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        saved.append(out_path)
        logger.info("Saved figure %s", out_path)
    plt.close("all")
    return saved


def run_niche_query(
    *,
    sample_data_dir: Path,
    emb_folder: Path,
    embedding_obsm_key: str,
    query_niche_sample_id: str,
    niche_obsm_keys: tuple[str, ...],
    target_sample_ids: list[str],
    niche_query_k: int,
    ensure_embeddings: bool,
    output_dir: Path,
    export_dir: Path | None,
    show_target_niches: bool,
) -> None:
    logger.info("Using embedding obsm key: %r", embedding_obsm_key)

    if ensure_embeddings:
        add_quest_embeddings(
            h5ad_folder=sample_data_dir,
            emb_folder=emb_folder,
            obsm_key=embedding_obsm_key,
            sample_ids=target_sample_ids,
            overwrite=False,
            write=True,
        )

    sample_adatas = load_sample_adatas(sample_data_dir, target_sample_ids)
    for sid, adata in sample_adatas.items():
        if embedding_obsm_key not in adata.obsm:
            raise KeyError(
                f"Sample {sid!r} is missing obsm[{embedding_obsm_key!r}]. "
                "Run with --ensure-embeddings or add embeddings first."
            )

    db = Database(target_sample_ids=target_sample_ids)
    db.construct_from_sample_adatas(
        sample_adatas=sample_adatas,
        feature_name=embedding_obsm_key,
        parcellation_path=str(PARCELLATION_PATH),
        parcellation_obs_key=None,
        ccf_coordinates_path=CCF_COORDINATES_PATHS,
        parcellation_write_obs_key="parcellation_index",
        spatial_obsm_key=SPATIAL_OBSM_KEY,
        preserve_input_adata=True,
    )

    query_samples = [s for s in db.samples.values() if s.id in target_sample_ids]

    output_dir.mkdir(parents=True, exist_ok=True)

    for niche_obsm_key in niche_obsm_keys:
        niche_to_query = construct_query_niche_if_missing(
            db,
            sample_id=query_niche_sample_id,
            niche_obsm_key=niche_obsm_key,
        )
        run_prefix = f"{query_niche_sample_id}_{niche_obsm_key}"

        niche_query = NicheQuery(db=db, niche=niche_to_query, k=niche_query_k)
        niche_query.niche_query_visualization(
            query_samples,
            show_target_niches=show_target_niches,
            rm_target_top_pct=[1, 5, 10],
            rm_target_top_k=200,
        )
        save_open_figures(output_dir, prefix=run_prefix)

        metrics_path = output_dir / f"{run_prefix}_quantitive_metrics.txt"
        niche_query.niche_query_quantitive_metrics(
            search_samples=query_samples,
            result_txt_path=str(metrics_path),
        )
        logger.info("Wrote quantitative metrics to %s", metrics_path)

        out_dir = export_dir if export_dir is not None else sample_data_dir
        export_samples(db=db, sample_dir=out_dir, sample_ids=target_sample_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Run niche query on benchmark samples using QueST embeddings."),
    )
    parser.add_argument(
        "--sample-data-dir",
        type=Path,
        default=SAMPLE_DATA_DIR,
        help="Folder with per-sample .h5ad files.",
    )
    parser.add_argument(
        "--emb-folder",
        type=Path,
        default=DEFAULT_EMB_FOLDER,
        help="Folder with QueST .pt embeddings.",
    )
    parser.add_argument(
        "--embedding-obsm-key",
        default=EMBEDDING_OBSM_KEY,
        help="obsm key for cell embeddings used as niche-query features.",
    )
    parser.add_argument(
        "--query-niche-sample-id",
        default=QUERY_NICHE_SAMPLE_ID,
        help="Sample that defines the query niche.",
    )
    parser.add_argument(
        "--niche-obsm-keys",
        nargs="+",
        default=list(NICHE_OBSM_KEYS),
        help="Query niche masks in obsm on the query sample.",
    )
    parser.add_argument(
        "--niche-query-k",
        type=int,
        default=NICHE_QUERY_K,
        help="k for niche query graph smoothing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SAMPLE_DATA_DIR / "quest_niche_query_output",
        help="Folder for visualization PNGs and quantitative metrics text.",
    )
    parser.add_argument(
        "--export-dir",
        type=Path,
        default=None,
        help="Optional folder for updated .h5ad files; defaults to --sample-data-dir.",
    )
    parser.add_argument(
        "--skip-embeddings",
        action="store_true",
        help="Do not add QueST embeddings before running.",
    )
    return parser.parse_args()


"""
PYTHONPATH=. .venv/bin/python run_niche_query.py \
  --sample-data-dir "notebook/ccf/data/benchmark_samples/20260601_225717" \
  --embedding-obsm-key X_gene_expr_quest \
  --query-niche-sample-id Zhuang-ABCA-1.098 \
  --niche-obsm-keys query_niche_9_15 \
  --output-dir "notebook/ccf/data/benchmark_samples/20260601_225717/quest_niche_query_output"
"""


def main() -> None:
    args = parse_args()
    run_niche_query(
        sample_data_dir=args.sample_data_dir,
        emb_folder=args.emb_folder,
        embedding_obsm_key=args.embedding_obsm_key,
        query_niche_sample_id=args.query_niche_sample_id,
        niche_obsm_keys=tuple(args.niche_obsm_keys),
        target_sample_ids=TARGET_SAMPLE_IDS,
        niche_query_k=args.niche_query_k,
        ensure_embeddings=not args.skip_embeddings,
        output_dir=args.output_dir,
        export_dir=args.export_dir,
        show_target_niches=False,
    )


if __name__ == "__main__":
    main()
