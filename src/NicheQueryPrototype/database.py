import datetime
import json
import logging
import os
from typing import Any, Dict, Union
from typing import List, Optional

# Import torch before anndata. anndata pulls ``anndata.experimental.pytorch``, which
# imports torch; initializing torch from inside that chain can trigger duplicate
# ``TORCH_LIBRARY`` registration in Jupyter (e.g. after autoreload or a failed import).
try:
    import torch  # noqa: F401
except ImportError:
    pass

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from matplotlib.axes import Axes

logger = logging.getLogger(__name__)

# Chunk size for filtering large CSVs when ``target_sample_ids`` is set.
_METADATA_CSV_CHUNKSIZE = 500_000


class Cell:
    """
    Represents a single cell with spatial coordinates and metadata.
    """

    def __init__(
        self,
        x: float,
        y: float,
        z: float,
        cell_id: str,
        parcellation_index: int,
        parcellation_info: Dict[str, Union[str, int]],
        sample_id: str,
    ):
        # Spatial coordinates
        self.x = x
        self.y = y
        self.z = z

        # Unique identifier for the cell
        self.id = cell_id

        # Parcellation information
        self.parcellation_index = parcellation_index
        self.parcellation_info = parcellation_info

        # ID of the sample this cell belongs to
        self.sample_id = sample_id
        # Gene expression vector (same genes as source AnnData .X); independent of query feature_name.
        self.X: Optional[np.ndarray] = None
        self.feature: Optional[np.ndarray] = None

        # The cosine similarity between this cell and the query cell.
        self.similarity: float = 0.0


def cell_matches_parcellation_or_ancestor(
    cell: Cell, target_parcellation_index: int
) -> bool:
    """
    True if the cell's parcellation is ``target_parcellation_index``, or that
    target appears on the cell's path in the CCF tree (i.e. in
    ``parcellation_info['parent_ids']``, which includes the node's own id).
    """

    if cell.parcellation_index == target_parcellation_index:
        return True
    info = cell.parcellation_info or {}
    parent_ids = info.get("parent_ids")
    if not parent_ids:
        return False
    return target_parcellation_index in parent_ids


class Sample:
    """
    Represents a biological sample containing a list of cells.
    """

    def __init__(self, sample_id: str):
        # Unique identifier for the sample
        self.id = sample_id

        # List of Cell objects
        self.cells: List[Cell] = []

        # List of niche features
        self.niche_features: List[np.ndarray] = []

        # If ture, the cell i belongs to the target parcellation section.
        self.target_parcellation_mask: Optional[np.ndarray] = None

        # RM-Ideal score for each cell in this sample (aligned with ``self.cells`` order).
        self.rm_ideal_score: Optional[np.ndarray] = None

        self.adata: Optional[AnnData] = None

    """
    Construct an adata from the source adata list.
    """

    def construct_adata(
        self, var, require_features=False, rm_ideal_output_key="rm_ideal_score"
    ):
        cell_ids = [c.id for c in self.cells]
        if len(cell_ids) == 0:
            self.adata = None
            return

        obs = pd.DataFrame(index=pd.Index(cell_ids, name="cell_id"))
        coords = np.asarray([[c.x, c.y] for c in self.cells], dtype=float)

        if require_features:
            self.adata = ad.AnnData(obs=obs, var=var)
        else:
            n_cells = len(self.cells)
            rows = [c.X for c in self.cells]
            if any(r is None for r in rows):
                bad = sum(1 for r in rows if r is None)
                raise ValueError(
                    f"Sample {self.id!r}: {bad}/{n_cells} cells have no expression vector "
                    "(cell.X is None); cannot build dense X."
                )
            # Pre-allocate instead of np.stack: lower peak RAM and faster for large n_cells.
            first = np.asarray(rows[0]).reshape(-1)
            X = np.empty((n_cells, first.shape[0]), dtype=first.dtype)
            X[0] = first
            for i in range(1, n_cells):
                X[i] = np.asarray(rows[i]).reshape(-1)
            self.adata = ad.AnnData(X=X, obs=obs, var=var)

        if require_features:
            # Multi-dimensional embeddings belong in obsm, not obs (pandas columns are 1D).
            feats = [c.feature for c in self.cells]
            if any(f is None for f in feats):
                bad = sum(1 for f in feats if f is None)
                raise ValueError(
                    f"Sample {self.id!r}: {bad}/{len(feats)} cells have no feature vector "
                    "(cell.feature is None); cannot build obsm['X_feature']."
                )
            first_f = np.asarray(feats[0]).reshape(-1)
            F = np.empty((len(feats), first_f.shape[0]), dtype=first_f.dtype)
            F[0] = first_f
            for i in range(1, len(feats)):
                F[i] = np.asarray(feats[i]).reshape(-1)
            self.adata.obsm["X_feature"] = F
        self.adata.obsm["spatial"] = coords
        self.adata.obs["sample_id"] = self.id
        if self.rm_ideal_score is not None:
            rm_scores = np.asarray(self.rm_ideal_score, dtype=float).reshape(-1)
            if rm_scores.shape[0] != len(self.cells):
                raise ValueError(
                    f"Sample {self.id!r}: rm_ideal_score length ({rm_scores.shape[0]}) "
                    f"does not match number of cells ({len(self.cells)})."
                )
            self.adata.obs[rm_ideal_output_key] = rm_scores
        self.adata.uns["library_id"] = self.id

    def visualize_parcellation_cells(
        self,
        target_parcellation_index: int,
        obs_key: str = "target_parcellation",
        spot_size: float = 0.02,
        show: Optional[bool] = None,
        print_metadata: bool = False,
        metadata_txt_path: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        On this sample's ``AnnData``, highlight cells whose parcellation equals
        ``target_parcellation_index`` or lies under that structure in the tree
        (same rule as ``parent_ids`` in :meth:`Database.parse_parcellation_structure`).

        Parameters
        ----------
        target_parcellation_index
            CCF structure id to query.
        obs_key
            Column written to ``adata.obs`` for plotting.
        print_metadata
            If True, log a summary with :func:`logging.info`, print each highlighted
            cell's ``cell_id``, ``x``, ``y``, and ``parcellation_index``, and if
            ``metadata_txt_path`` is set, write the same text to that file.
        metadata_txt_path
            When ``print_metadata`` is True, optional path to a UTF-8 text file
            receiving the same lines as the info log and stdout (header plus
            one line per cell). Ignored when ``print_metadata`` is False.
            The parent directory is created if it does not exist.
        spot_size, show, **kwargs
            Passed to :func:`scanpy.pl.spatial` (``show`` defaults to
            :obj:`True` when omitted).
        """
        if self.adata is None:
            raise ValueError(
                f"Sample {self.id!r} has no AnnData; construct the database first."
            )
        matched_cells = [
            c
            for c in self.cells
            if cell_matches_parcellation_or_ancestor(c, target_parcellation_index)
        ]
        matched_ids = {c.id for c in matched_cells}
        if print_metadata:
            metadata = (
                f"Highlighted cells (n={len(matched_cells)}) for parcellation "
                f"{target_parcellation_index} in sample {self.id!r}:"
            )
            logger.info(metadata)
            lines = [metadata]
            for c in matched_cells:
                line = (
                    f"  cell_id={c.id!r}  x={c.x}  y={c.y}  "
                    f"parcellation_index={c.parcellation_index}"
                )
                logger.info(line)
                lines.append(line)
            if metadata_txt_path is not None:
                _out = os.path.abspath(metadata_txt_path)
                _parent = os.path.dirname(_out)
                if _parent:
                    os.makedirs(_parent, exist_ok=True)
                with open(metadata_txt_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(lines))
        adata = self.adata
        adata.obs[obs_key] = adata.obs_names.isin(matched_ids)
        if show is None:
            show = True
        sc.pl.spatial(
            adata,
            color=obs_key,
            palette=["lightgrey", "red"],
            spot_size=spot_size,
            title=f"Parcellation {target_parcellation_index} (self or subtree) | {self.id}",
            show=show,
            **kwargs,
        )


class Database:
    """
    Represents a database containing multiple samples.
    """

    def __init__(self, target_sample_ids: Optional[List[str]] = None):
        self.cells: List[Cell] = []
        self.samples: Dict[str, Sample] = {}
        self.merged_cell_metadata: pd.DataFrame = pd.DataFrame()
        self.parcellation_tree = {}
        self.target_sample_ids: Optional[List[str]] = target_sample_ids

    def parse_parcellation_structure(
        self, json_path: str
    ) -> Dict[int, Dict[str, Union[str, int, set[int]]]]:
        """
        Parse parcellation structure and return a dictionary:

        key: structure id
        value: {
            "parent_ids": set[int],   # includes self
            "name": str,
            "st_level": int
        }
        """

        with open(json_path, "r") as f:
            data = json.load(f)

        result = {}

        def dfs(node, ancestors: set[int]) -> None:
            node_id = int(node.get("id"))
            st_level_raw = node.get("st_level")
            st_level = int(st_level_raw) if st_level_raw is not None else None
            name = node.get("name")

            current_ancestors = set(ancestors)
            current_ancestors.add(node_id)  # parent includes self

            entry = {
                "parent_ids": current_ancestors,
                "name": name,
                "st_level": st_level,
            }

            result[node_id] = entry

            for child in node.get("children", []):
                dfs(child, current_ancestors)

        # Root is inside data["msg"]
        for root_node in data["msg"]:
            dfs(root_node, set())

        return result

    @staticmethod
    def _row_to_1d_numpy(x: Any) -> np.ndarray:
        """
        AnnData row slices may be dense ndarray, ``np.matrix``, scipy sparse,
        or sparse *views* without ``.squeeze()`` — normalize to shape ``(n_features,)``.
        """
        if hasattr(x, "toarray") and callable(getattr(x, "toarray", None)):
            x = x.toarray()
        arr = np.asarray(x)
        return np.reshape(arr, (-1,))

    def get_cell_expression(
        self, adata_list: List[AnnData], cell_id: str
    ) -> Optional[np.ndarray]:
        for adata in adata_list:
            if cell_id in adata.obs.index:
                return self._row_to_1d_numpy(adata[cell_id, :].X)
        return None

    def get_cell_embedding(
        self, adata_list: List[AnnData], cell_id: str, embedding_key: str
    ) -> Optional[np.ndarray]:
        for adata in adata_list:
            if embedding_key in adata.obsm.keys() and cell_id in adata.obs.index:
                return self._row_to_1d_numpy(adata[cell_id, :].obsm[embedding_key])
        return None

    """
    Loads and construct the database.
    
    `feature_name`: 'gene_expression' or the name of embeddings in adata.obsm.
    """

    def construct(
        self,
        adata_path: List[str],
        cell_metadata_path: List[str],
        ccf_coordinates_path: List[str],
        feature_name: str,
        rm_ideal_output_key: str = "rm_ideal_score",
        parcellation_path: Optional[str] = None,
    ):
        target_set = (
            set(self.target_sample_ids) if self.target_sample_ids is not None else None
        )

        def _read_cell_metadata_csv(path: str) -> pd.DataFrame:
            if target_set is None:
                return pd.read_csv(path)
            parts: List[pd.DataFrame] = []
            for chunk in pd.read_csv(path, chunksize=_METADATA_CSV_CHUNKSIZE):
                if "brain_section_label" not in chunk.columns:
                    raise KeyError(
                        f"{path!r} is missing column 'brain_section_label', "
                        "required when ``target_sample_ids`` is set."
                    )
                sub = chunk[chunk["brain_section_label"].isin(target_set)]
                if not sub.empty:
                    parts.append(sub)
            if not parts:
                return pd.DataFrame()
            return pd.concat(parts, axis=0, ignore_index=True)

        def _read_ccf_csv(path: str, allowed_cell_labels: set) -> pd.DataFrame:
            if target_set is None:
                return pd.read_csv(path)
            parts: List[pd.DataFrame] = []
            for chunk in pd.read_csv(path, chunksize=_METADATA_CSV_CHUNKSIZE):
                sub = chunk[chunk["cell_label"].isin(allowed_cell_labels)]
                if not sub.empty:
                    parts.append(sub)
            if not parts:
                return pd.DataFrame()
            return pd.concat(parts, axis=0, ignore_index=True)

        # step 1: load tabular data first, then AnnData (often one file with all cells).
        logger.info("Loading cell metadata...")
        cell_metadata_list: List[pd.DataFrame] = []
        for path in cell_metadata_path:
            cell_metadata_list.append(_read_cell_metadata_csv(path))
        cell_metadata = pd.concat(cell_metadata_list, axis=0, ignore_index=True)
        cell_metadata.drop_duplicates(subset=["cell_label"], inplace=True)

        if target_set is not None and cell_metadata.empty:
            raise ValueError(
                "No rows left after filtering cell metadata by ``target_sample_ids``. "
                "Check that ``brain_section_label`` values match the ids you passed."
            )

        allowed_cell_labels = (
            set(cell_metadata["cell_label"]) if target_set is not None else None
        )
        logger.info("Loading CCF coordinates...")
        ccf_coordinates_list: List[pd.DataFrame] = []
        for path in ccf_coordinates_path:
            if allowed_cell_labels is not None:
                ccf_coordinates_list.append(_read_ccf_csv(path, allowed_cell_labels))
            else:
                ccf_coordinates_list.append(pd.read_csv(path))
        ccf_coordinates = pd.concat(ccf_coordinates_list, axis=0, ignore_index=True)
        ccf_coordinates.drop_duplicates(subset=["cell_label"], inplace=True)
        self.merged_cell_metadata = pd.merge(
            cell_metadata, ccf_coordinates, on="cell_label", suffixes=("", "_ccf")
        )
        logger.info("Constructing parcellation tree...")
        self.parcellation_tree = self.parse_parcellation_structure(parcellation_path)

        logger.info("Loading AnnData...")
        adata_list: List[AnnData] = []
        for path in adata_path:
            ad = sc.read(path)
            if allowed_cell_labels is not None:
                keep = ad.obs_names.isin(allowed_cell_labels)
                n_before = int(ad.n_obs)
                n_keep = int(keep.sum())
                if n_keep == 0:
                    logger.error(
                        f"No overlap between AnnData obs index and filtered cell_metadata "
                        f"``cell_label`` for {path!r} (n_obs={n_before}). "
                        f"Ensure ``adata.obs_names`` match ``cell_label`` (same dtype/format)."
                    )
                    continue
                ad = ad[keep]
                logger.info(
                    "Subset %r to %d / %d cells (from ``target_sample_ids`` metadata).",
                    os.path.basename(path),
                    n_keep,
                    n_before,
                )
            adata_list.append(ad)

        # Optional: recover precomputed RM-Ideal scores from raw AnnData obs.
        rm_ideal_score_by_cell: Dict[str, float] = {}
        n_adatas_with_rm = 0
        for ad in adata_list:
            if "rm_ideal_score" not in ad.obs.columns:
                continue
            n_adatas_with_rm += 1
            rm_col = pd.to_numeric(ad.obs["rm_ideal_score"], errors="coerce")
            for cid, score in zip(ad.obs_names.astype(str), rm_col.to_numpy()):
                if pd.isna(score):
                    continue
                rm_ideal_score_by_cell[cid] = float(score)
        if n_adatas_with_rm > 0:
            logger.info(
                "Detected rm_ideal_score in %d AnnData file(s); loaded %d cell-level scores.",
                n_adatas_with_rm,
                len(rm_ideal_score_by_cell),
            )

        if feature_name == "gene_expression" and adata_list:
            name_sets = [set(ad.var_names.astype(str)) for ad in adata_list]
            common_genes = sorted(set.intersection(*name_sets))
            if not common_genes:
                raise ValueError(
                    "No overlapping gene names across AnnData objects. "
                    "Cannot align ``gene_expression`` features."
                )
            logger.info(
                "Gene expression: using %d common genes across %d AnnData file(s).",
                len(common_genes),
                len(adata_list),
            )
            adata_list = [ad[:, common_genes].copy() for ad in adata_list]

        # step 2: construct cells and samples
        logger.info("Constructing database for niche query...")
        for row in self.merged_cell_metadata.itertuples(index=False):
            parcellation_info = self.parcellation_tree.get(row.parcellation_index, {})
            cell_id = row.cell_label
            sample_id = row.brain_section_label
            cell = Cell(
                x=row.x,
                y=row.y,
                z=row.z,
                cell_id=cell_id,
                parcellation_index=row.parcellation_index,
                parcellation_info=parcellation_info,
                sample_id=sample_id,
            )
            cell.X = self.get_cell_expression(adata_list, cell_id)
            if feature_name == "gene_expression":
                cell.feature = cell.X
            else:
                cell.feature = self.get_cell_embedding(
                    adata_list, cell_id, feature_name
                )
            self.cells.append(cell)
            if sample_id not in self.samples.keys():
                self.samples[sample_id] = Sample(sample_id=sample_id)
            self.samples[sample_id].cells.append(cell)
            if len(self.cells) % 100000 == 0:
                logger.info(f"\t{len(self.cells)} cells loaded...")
        logger.info(
            f"Processed {len(self.cells)} cells and {len(self.samples.keys())} samples."
        )

        if rm_ideal_score_by_cell:
            n_samples_with_rm = 0
            for sample in self.samples.values():
                rm_scores = np.array(
                    [
                        rm_ideal_score_by_cell.get(str(c.id), np.nan)
                        for c in sample.cells
                    ],
                    dtype=float,
                )
                if np.isnan(rm_scores).all():
                    continue
                sample.rm_ideal_score = rm_scores
                n_samples_with_rm += 1
            logger.info(
                "Loaded rm_ideal_score into %d/%d samples.",
                n_samples_with_rm,
                len(self.samples),
            )

        if feature_name != "gene_expression":
            feat_dims: set = set()
            for c in self.cells:
                if c.feature is not None:
                    feat_dims.add(int(np.asarray(c.feature).reshape(-1).shape[0]))
            if len(feat_dims) > 1:
                raise ValueError(
                    "Inconsistent embedding dimensions across cells for "
                    f"feature_name={feature_name!r}: found sizes {sorted(feat_dims)}. "
                    "All cells must share the same embedding width."
                )

        # Different source .h5ad files can carry different numbers of genes.
        # Build a lookup so each sample gets a `var` aligned to its own `X` width.
        var_by_n_vars: Dict[int, pd.DataFrame] = {}
        for adata in adata_list:
            if adata.n_vars not in var_by_n_vars:
                var_by_n_vars[int(adata.n_vars)] = adata.var.copy(deep=False)

        n_samples = len(self.samples)
        logger.info(
            "Constructing adata for each sample (%d total; this can take a while if "
            "samples are large — memory is proportional to n_cells × n_features per sample)...",
            n_samples,
        )
        for i, sample in enumerate(self.samples.values()):
            n_cells_s = len(sample.cells)
            # Avoid a long silent stretch on the first huge sample(s).
            _stride = 25 if n_samples > 60 else (10 if n_samples > 20 else 1)
            _verbose = (
                n_samples <= 30 or i < 3 or i == n_samples - 1 or (i + 1) % _stride == 0
            )
            if _verbose:
                logger.info(
                    "  [%d/%d] sample %r: %d cells — building AnnData...",
                    i + 1,
                    n_samples,
                    sample.id,
                    n_cells_s,
                )

            sample_n_vars: Optional[int] = None
            for c in sample.cells:
                if c.X is not None:
                    sample_n_vars = int(np.asarray(c.X).reshape(-1).shape[0])
                    break

            sample_var = None
            if sample_n_vars is not None:
                sample_var = var_by_n_vars.get(sample_n_vars)
                if sample_var is None:
                    logger.warning(
                        "No source `var` found with n_vars=%d for sample %r; using placeholder feature ids.",
                        sample_n_vars,
                        sample.id,
                    )
                    sample_var = pd.DataFrame(
                        index=pd.Index(
                            [f"feature_{j}" for j in range(sample_n_vars)],
                            name="feature_id",
                        )
                    )
            elif var_by_n_vars:
                # Fallback for feature-only workflows without gene expression vectors.
                sample_var = next(iter(var_by_n_vars.values()))
            else:
                sample_var = pd.DataFrame()

            sample.construct_adata(
                var=sample_var,
                require_features=(feature_name != "gene_expression"),
                rm_ideal_output_key="rm_ideal_score",
            )
            if _verbose:
                logger.info(
                    "  [%d/%d] sample %r: done.",
                    i + 1,
                    n_samples,
                    sample.id,
                )

    def get_sample(self, sample_id: str) -> Optional[Sample]:
        return self.samples.get(sample_id, None)

    def get_cell(self, cell_id: str) -> Optional[Cell]:
        for cell in self.cells:
            if cell.id == cell_id:
                return cell
        return None

    @staticmethod
    def cell_matches_parcellation_or_ancestor(
        cell: Cell, target_parcellation_index: int
    ) -> bool:
        """
        True if the cell's parcellation is `target_parcellation_index`, or that
        target appears on the cell's path in the CCF tree (i.e. in
        ``parcellation_info['parent_ids']``, which includes the node's own id).
        """
        return cell_matches_parcellation_or_ancestor(cell, target_parcellation_index)

    """
    Export adata for `target_sample_ids` if not null; otherwise, export all samples. One sample per file.
    """

    def export_sample_adata(self, export_dir: str):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        dir = os.path.join(export_dir, timestamp)
        os.makedirs(dir, exist_ok=True)
        for sample in self.samples.values():
            filename = f"{dir}/{sample.id}.h5ad"
            sample.adata.write(filename)
            logging.info(f"Exported {filename}")
