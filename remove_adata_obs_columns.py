"""
Keep only selected ``obsm`` and ``uns`` entries in .h5ad files.


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

DEFAULT_KEEP_OBSM_KEYS = [
    "X_quest_gene_expr",
    "X_quest_scgpt",
    "X_scgpt",
    "spatial",
]
DEFAULT_KEEP_UNS_KEYS = ["library_id"]


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
    keep_obsm_keys: list[str],
    keep_uns_keys: list[str],
    write: bool = True,
    inspect: bool = False,
) -> dict[str, list[str]]:
    adata = ad.read_h5ad(h5ad_path)
    if inspect:
        print_adata_attributes(adata, h5ad_path)

    keep_obsm_keys_set = set(keep_obsm_keys)
    keep_uns_keys_set = set(keep_uns_keys)
    obsm_keys_to_remove = [
        key for key in adata.obsm.keys() if key not in keep_obsm_keys_set
    ]
    uns_keys_to_remove = [
        key for key in adata.uns.keys() if key not in keep_uns_keys_set
    ]

    for key in obsm_keys_to_remove:
        del adata.obsm[key]

    for key in uns_keys_to_remove:
        del adata.uns[key]

    if not (obsm_keys_to_remove or uns_keys_to_remove):
        logger.info(
            "[SKIP] %s: obsm and uns already match the requested allowlists", h5ad_path
        )
    else:
        status = "[DONE]" if write else "[DRY-RUN]"
        logger.info(
            "%s %s: obsm_removed=%s uns_removed=%s",
            status,
            h5ad_path,
            obsm_keys_to_remove,
            uns_keys_to_remove,
        )

    if write and (obsm_keys_to_remove or uns_keys_to_remove):
        adata.write_h5ad(h5ad_path)

    return {
        "obsm_removed": obsm_keys_to_remove,
        "uns_removed": uns_keys_to_remove,
    }


def clean_h5ad_folder(
    h5ad_folder: Path,
    keep_obsm_keys: list[str],
    keep_uns_keys: list[str],
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
            keep_obsm_keys=keep_obsm_keys,
            keep_uns_keys=keep_uns_keys,
            write=write,
            inspect=inspect,
        )

    return cleaned_by_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Keep only selected obsm and uns keys in one or more .h5ad files.",
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
        "--keep-obsm-keys",
        nargs="+",
        default=DEFAULT_KEEP_OBSM_KEYS,
        help="Only these obsm keys will be kept.",
    )
    parser.add_argument(
        "--keep-uns-keys",
        nargs="+",
        default=DEFAULT_KEEP_UNS_KEYS,
        help="Only these uns keys will be kept.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report keys that would be removed without writing .h5ad files.",
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Print AnnData attributes and container keys before filtering obsm and uns.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clean_h5ad_folder(
        h5ad_folder=args.h5ad_folder,
        keep_obsm_keys=args.keep_obsm_keys,
        keep_uns_keys=args.keep_uns_keys,
        sample_ids=args.sample_ids,
        write=not args.dry_run,
        inspect=args.inspect,
    )


if __name__ == "__main__":
    main()
