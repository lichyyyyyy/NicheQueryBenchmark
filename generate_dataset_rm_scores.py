from src.NicheQueryPrototype.database import Database
from src.NicheQueryPrototype.niche import Niche
from src.NicheQueryPrototype.query import NicheQuery

import logging
import os
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

NICHE_NAMES = ["query_niche_9_15"]
QUERY_NICHE_SAMPLE_IDS = [
    # "Zhuang-ABCA-1.098",
    # "Zhuang-ABCA-1.099",
    # "Zhuang-ABCA-1.097",
    # "Zhuang-ABCA-1.096",
    "C57BL6J-638850.28",
    "C57BL6J-638850.29",
    "C57BL6J-638850.30",
    "C57BL6J-638850.31",
]
NICHE_QUERY_K = 3  # wl_iters for RmIdeal (same as niche_query_multi.ipynb)

# Load per-sample AnnData from this folder (one .h5ad per sample).
SAMPLE_DATA_DIR = Path("notebook/ccf/data/benchmark_samples/20260601_225717")
PARCELLATION_PATH = Path("notebook/ccf/data/parcellation/structure.json")
SPATIAL_OBSM_KEY = "spatial"

# Source CCF coordinate CSVs (used to load integer parcellation ids per cell_label).
zhuang_abca_1_folder = "notebook/ccf/data/zhuang-abca-1/"
zhuang_abca_2_folder = "notebook/ccf/data/zhuang-abca-2/"
zhuang_abca_3_folder = "notebook/ccf/data/zhuang-abca-3/"
zeng_mwb_folder = "notebook/ccf/data/zeng-mwb/"
CCF_COORDINATES_PATHS = [
    zhuang_abca_1_folder + "ccf_coordinates.csv",
    zhuang_abca_2_folder + "ccf_coordinates.csv",
    zhuang_abca_3_folder + "ccf_coordinates.csv",
    zeng_mwb_folder + "ccf_coordinates.csv",
]

# By default, score and export these samples (must exist as <sample_id>.h5ad in SAMPLE_DATA_DIR).
target_sample_ids = [
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

# Write progress exports here (files are overwritten as new scores are added).
export_dir = str(SAMPLE_DATA_DIR / "checkpoints")
os.makedirs(export_dir, exist_ok=True)


def export_all_samples(db, export_dir: str, sample_ids: list[str]) -> None:
    for sample_id in sample_ids:
        sample = db.get_sample(sample_id)
        if sample is None or sample.adata is None:
            raise ValueError(f"Sample not ready for export: {sample_id!r}")
        out_path = os.path.join(export_dir, f"{sample_id}.h5ad")
        sample.adata.write_h5ad(out_path)
        print(f"Exported {len(sample_ids)} sample(s) to {export_dir}\n")
        print(f"  n_obs={sample.adata.n_obs}, n_vars={sample.adata.n_vars}")
        print(
            f"  obs ({len(sample.adata.obs.columns)}): {list(sample.adata.obs.columns)}"
        )
        print(f"  obsm ({len(sample.adata.obsm)}): {list(sample.adata.obsm.keys())}")


def load_niche_from_obsm(db, sample_id: str, niche_obsm_key: str) -> Niche:
    """Load query niche cells from ``sample.adata.obsm[niche_obsm_key]``."""
    sample = db.get_sample(sample_id)
    if sample is None:
        raise ValueError(f"Sample not in db: {sample_id!r}")
    adata = sample.adata
    if adata is None:
        raise ValueError(f"Sample {sample_id!r} has no AnnData.")
    if niche_obsm_key not in adata.obsm:
        raise KeyError(
            f"obsm[{niche_obsm_key!r}] not found for {sample_id!r}. "
            "Run the query-niche construction cells above first."
        )

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
            c = id_to_cell.get(oid)
            if c is not None:
                cells.append(c)

    if not cells:
        raise ValueError(
            f"No niche cells loaded from obsm[{niche_obsm_key!r}] for {sample_id!r}"
        )

    niche = Niche()
    niche.construct(cells, sample_id)
    niche.compute_niche_feature()
    print(
        f"Loaded {niche_obsm_key!r} from {sample_id!r}: "
        f"{len(cells)} cells, parcellations={niche.unique_parcellation_indices}"
    )
    return niche


def load_sample_adatas(sample_dir: Path, sample_ids: list[str]) -> dict[str, Any]:
    adatas: dict[str, Any] = {}
    for sid in sample_ids:
        path = sample_dir / f"{sid}.h5ad"
        if not path.exists():
            raise FileNotFoundError(f"Missing sample file: {path}")
        adatas[sid] = ad.read_h5ad(str(path))
    return adatas


sample_adatas = load_sample_adatas(SAMPLE_DATA_DIR, target_sample_ids)

db = Database(target_sample_ids=target_sample_ids)
db.construct_from_sample_adatas(
    sample_adatas=sample_adatas,
    feature_name="gene_expression",
    parcellation_path=str(PARCELLATION_PATH),
    parcellation_obs_key=None,  # auto-detect if present; otherwise defaults to 0
    ccf_coordinates_path=CCF_COORDINATES_PATHS,  # load cell_label -> parcellation_index from CCF
    parcellation_write_obs_key="parcellation_index",
    spatial_obsm_key=SPATIAL_OBSM_KEY,
    rm_ideal_output_key="rm_ideal_score",
    preserve_input_adata=True,  # keep niche masks in obsm
)

query_samples = [s for s in db.samples.values() if s.id in target_sample_ids]

for niche_name in NICHE_NAMES:
    for query_niche_sample_id in QUERY_NICHE_SAMPLE_IDS:
        niche_to_query = load_niche_from_obsm(db, query_niche_sample_id, niche_name)
        rm_ideal_output_key = f"{query_niche_sample_id}_{niche_name}"

        niche_query = NicheQuery(db=db, niche=niche_to_query, k=NICHE_QUERY_K)
        niche_query.compute_rm_ideal_score(
            samples=query_samples,
            rm_ideal_output_key=rm_ideal_output_key,
            overwrite=False,
            rm_ideal_post_transform="sigmoid",
        )
        print(
            f"Wrote RM-Ideal scores to obs[{rm_ideal_output_key!r}] on {len(query_samples)} sample(s)"
        )

        # Export updated AnnData after each RM-Ideal run.
        export_all_samples(db=db, export_dir=export_dir, sample_ids=target_sample_ids)

print(f"RM-Ideal progress exports are in: {export_dir}")
