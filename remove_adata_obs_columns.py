"""
Clean selected ``obs``, ``obsm``, and ``layers`` entries in .h5ad files.


# python3 remove_adata_obs_columns.py --inspect --dry-run
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import anndata as ad

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_H5AD_FOLDER = Path("data/20260601_225717")

DEFAULT_KEEP_OBS_COLUMNS = ["sample_id", "parcellation_index"]

DEFAULT_OBSM_KEYS = [
    "X_quest_emb",
    "X_scgpt_quest",
    "query_niche_207_123",
    "query_niche_9_15",
    "query_niche_9_841",
    "Zhuang-ABCA-3.005_query_niche_9_841",
    "Zhuang-ABCA-3.005_query_niche_207_123",
    "Zhuang-ABCA-3.005_query_niche_9_15",
]

DEFAULT_OBSM_RENAMES = {
    "X_gene_expr_quest": "X_quest_gene_expr",
}


def _keys(value: Any) -> list[str]:
    if hasattr(value, "keys"):
        return [str(key) for key in value.keys()]
    return []


def print_adata_attributes(adata: ad.AnnData, h5ad_path: Path) -> None:
    print(f"\n{h5ad_path}")
    print(f"  shape: {adata.shape}")
    print(f"  n_obs: {adata.n_obs}")
    print(f"  n_vars: {adata.n_vars}")
    print(f"  obs columns ({len(adata.obs.columns)}): {list(adata.obs.columns)}")
    print(f"  var columns ({len(adata.var.columns)}): {list(adata.var.columns)}")
    print(f"  obsm keys ({len(adata.obsm)}): {_keys(adata.obsm)}")
    print(f"  varm keys ({len(adata.varm)}): {_keys(adata.varm)}")
    print(f"  obsp keys ({len(adata.obsp)}): {_keys(adata.obsp)}")
    print(f"  varp keys ({len(adata.varp)}): {_keys(adata.varp)}")
    print(f"  layers keys ({len(adata.layers)}): {_keys(adata.layers)}")
    print(f"  uns keys ({len(adata.uns)}): {_keys(adata.uns)}")
    print(f"  raw: {'present' if adata.raw is not None else 'None'}")


def clean_h5ad(
    h5ad_path: Path,
    keep_obs_columns: list[str],
    obsm_keys: list[str],
    obsm_renames: dict[str, str],
    remove_layers: bool = True,
    write: bool = True,
    inspect: bool = False,
) -> dict[str, list[str]]:
    adata = ad.read_h5ad(h5ad_path)
    if inspect:
        print_adata_attributes(adata, h5ad_path)

    keep_obs_columns_set = set(keep_obs_columns)
    obs_columns_to_remove = [
        column for column in adata.obs.columns if column not in keep_obs_columns_set
    ]
    obsm_keys_to_remove = [key for key in obsm_keys if key in adata.obsm]
    layer_keys_to_remove = list(adata.layers.keys()) if remove_layers else []
    renamed_obsm_keys: list[str] = []

    for old_key, new_key in obsm_renames.items():
        if old_key not in adata.obsm:
            continue
        if new_key in adata.obsm:
            logger.warning(
                "[SKIP] %s: obsm[%r] already exists; did not rename obsm[%r]",
                h5ad_path,
                new_key,
                old_key,
            )
            continue
        adata.obsm[new_key] = adata.obsm[old_key]
        del adata.obsm[old_key]
        renamed_obsm_keys.append(f"{old_key}->{new_key}")

    if obs_columns_to_remove:
        adata.obs.drop(columns=obs_columns_to_remove, inplace=True)

    for key in obsm_keys_to_remove:
        del adata.obsm[key]

    for key in layer_keys_to_remove:
        del adata.layers[key]

    if not (
        obs_columns_to_remove
        or obsm_keys_to_remove
        or layer_keys_to_remove
        or renamed_obsm_keys
    ):
        logger.info(
            "[SKIP] %s: none of the requested AnnData entries were present", h5ad_path
        )
    else:
        status = "[DONE]" if write else "[DRY-RUN]"
        logger.info(
            "%s %s: obs_removed=%s obsm_removed=%s obsm_renamed=%s layers_removed=%s",
            status,
            h5ad_path,
            obs_columns_to_remove,
            obsm_keys_to_remove,
            renamed_obsm_keys,
            layer_keys_to_remove,
        )

    if write:
        adata.write_h5ad(h5ad_path)

    return {
        "obs_removed": obs_columns_to_remove,
        "obsm_removed": obsm_keys_to_remove,
        "obsm_renamed": renamed_obsm_keys,
        "layers_removed": layer_keys_to_remove,
    }


def clean_h5ad_folder(
    h5ad_folder: Path,
    keep_obs_columns: list[str],
    obsm_keys: list[str],
    obsm_renames: dict[str, str],
    remove_layers: bool = True,
    sample_ids: list[str] | None = None,
    write: bool = True,
    inspect: bool = False,
) -> dict[Path, dict[str, list[str]]]:
    h5ad_folder = Path(h5ad_folder)
    h5ad_paths = sorted(h5ad_folder.glob("*.h5ad"))

    if sample_ids is not None:
        sample_id_set = set(sample_ids)
        h5ad_paths = [path for path in h5ad_paths if path.stem in sample_id_set]

    if not h5ad_paths:
        raise FileNotFoundError(f"No matching .h5ad files found in {h5ad_folder}")

    cleaned_by_path: dict[Path, dict[str, list[str]]] = {}
    for h5ad_path in h5ad_paths:
        cleaned_by_path[h5ad_path] = clean_h5ad(
            h5ad_path=h5ad_path,
            keep_obs_columns=keep_obs_columns,
            obsm_keys=obsm_keys,
            obsm_renames=obsm_renames,
            remove_layers=remove_layers,
            write=write,
            inspect=inspect,
        )

    return cleaned_by_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean obs, obsm, and layers in one or more .h5ad files.",
    )
    parser.add_argument(
        "--h5ad-folder",
        type=Path,
        default=DEFAULT_H5AD_FOLDER,
        help="Folder containing .h5ad files to update.",
    )
    parser.add_argument(
        "--sample-ids",
        nargs="*",
        default=None,
        help="Optional subset of sample ids to process; each should match an .h5ad stem.",
    )
    parser.add_argument(
        "--keep-obs-columns",
        nargs="+",
        default=DEFAULT_KEEP_OBS_COLUMNS,
        help="Only these obs columns will be kept. Defaults to sample_id.",
    )
    parser.add_argument(
        "--obsm-keys",
        nargs="+",
        default=DEFAULT_OBSM_KEYS,
        help="obsm keys to remove.",
    )
    parser.add_argument(
        "--keep-layers",
        action="store_true",
        help="Keep layers instead of removing all layer keys.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report columns that would be removed without writing .h5ad files.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print AnnData attributes and container keys before removing obs columns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_h5ad_folder(
        h5ad_folder=args.h5ad_folder,
        keep_obs_columns=args.keep_obs_columns,
        obsm_keys=args.obsm_keys,
        obsm_renames=DEFAULT_OBSM_RENAMES,
        remove_layers=not args.keep_layers,
        sample_ids=args.sample_ids,
        write=not args.dry_run,
        inspect=args.inspect,
    )


if __name__ == "__main__":
    main()
