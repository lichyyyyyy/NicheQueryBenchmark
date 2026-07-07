"""Add QueST embeddings from .pt files to AnnData .h5ad files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_EMB_FOLDER = Path(
    "/Users/sheryli/Documents/life/biology/Ji Lab/ST_FM_benchmark/"
    "exploration/QueST/output/embeddings/quest_20260601_225717_gene_expr_quest"
)
DEFAULT_H5AD_FOLDER = Path(
    "/Users/sheryli/Documents/life/biology/Ji Lab/ST_FM_benchmark/"
    "BenchmarkNicheQuery/notebook/ccf/data/benchmark_samples/20260601_225717"
)
DEFAULT_OBSM_KEY = "X_quest_emb"


def load_quest_embedding(emb_path: Path) -> np.ndarray:
    emb_tensor = torch.load(emb_path, map_location="cpu", weights_only=False)
    if isinstance(emb_tensor, torch.Tensor):
        emb = emb_tensor.detach().cpu().numpy()
    else:
        emb = np.asarray(emb_tensor)

    if emb.ndim != 2:
        raise ValueError(f"{emb_path} embedding must be 2D, got {emb.shape}")
    return emb


def add_quest_embeddings_to_h5ad(
    h5ad_path: Path,
    emb_path: Path,
    obsm_key: str = DEFAULT_OBSM_KEY,
    overwrite: bool = False,
    write: bool = True,
) -> ad.AnnData:
    adata = ad.read_h5ad(h5ad_path)

    if obsm_key in adata.obsm and not overwrite:
        logger.info("[SKIP] %s already has obsm[%r]", h5ad_path.name, obsm_key)
        return adata

    if not emb_path.exists():
        raise FileNotFoundError(f"Embedding not found: {emb_path}")

    emb = load_quest_embedding(emb_path)
    if emb.shape[0] != adata.n_obs:
        raise ValueError(
            f"Cell number mismatch for {h5ad_path.stem}: "
            f"embedding {emb.shape[0]} vs adata {adata.n_obs}"
        )

    adata.obsm[obsm_key] = emb
    logger.info(
        "[DONE] %s: obsm[%r] shape=%s",
        h5ad_path.stem,
        obsm_key,
        emb.shape,
    )

    if write:
        adata.write_h5ad(h5ad_path)
    return adata


def add_quest_embeddings(
    h5ad_folder: Path,
    emb_folder: Path,
    obsm_key: str = DEFAULT_OBSM_KEY,
    sample_ids: list[str] | None = None,
    overwrite: bool = False,
    write: bool = True,
) -> list[ad.AnnData]:
    h5ad_folder = Path(h5ad_folder)
    emb_folder = Path(emb_folder)

    h5ad_paths = sorted(h5ad_folder.glob("*.h5ad"))
    if sample_ids is not None:
        sample_id_set = set(sample_ids)
        h5ad_paths = [p for p in h5ad_paths if p.stem in sample_id_set]

    if not h5ad_paths:
        raise FileNotFoundError(f"No .h5ad files found in {h5ad_folder}")

    updated: list[ad.AnnData] = []
    for h5ad_path in h5ad_paths:
        emb_path = emb_folder / f"{h5ad_path.stem}.pt"
        if not emb_path.exists():
            logger.warning("[SKIP] No embedding for %s", h5ad_path.name)
            continue

        logger.info("[LOAD] %s", emb_path)
        adata = add_quest_embeddings_to_h5ad(
            h5ad_path=h5ad_path,
            emb_path=emb_path,
            obsm_key=obsm_key,
            overwrite=overwrite,
            write=write,
        )
        updated.append(adata)

    return updated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add QueST .pt embeddings to matching .h5ad files as an obsm key.",
    )
    parser.add_argument(
        "--emb-folder",
        type=Path,
        default=DEFAULT_EMB_FOLDER,
        help="Folder containing QueST embedding .pt files.",
    )
    parser.add_argument(
        "--h5ad-folder",
        type=Path,
        default=DEFAULT_H5AD_FOLDER,
        help="Folder containing per-sample .h5ad files.",
    )
    parser.add_argument(
        "--obsm-key",
        default=DEFAULT_OBSM_KEY,
        help="obsm key used to store embeddings.",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="*",
        default=None,
        help="Optional subset of sample ids to process.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing obsm key instead of skipping.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate embeddings without writing .h5ad files.",
    )
    return parser.parse_args()


"""
# # Validate without writing
python src/add_embeddings.py --dry-run

# Write embeddings into the h5ad files
python src/add_embeddings.py

# Single sample
python src/add_embeddings.py --sample-ids Zhuang-ABCA-1.098

# Custom paths
python src/add_embeddings.py \
  --emb-folder "/path/to/quest_20260601_225717_gene_expr_quest" \
  --h5ad-folder "/path/to/h5ad/folder" \
  --obsm-key X_quest_emb
"""


def main() -> None:
    args = parse_args()
    add_quest_embeddings(
        h5ad_folder=args.h5ad_folder,
        emb_folder=args.emb_folder,
        obsm_key=args.obsm_key,
        sample_ids=args.sample_ids,
        overwrite=args.overwrite,
        write=not args.dry_run,
    )


if __name__ == "__main__":
    main()
