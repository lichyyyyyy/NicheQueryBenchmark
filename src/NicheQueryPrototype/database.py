import datetime
import json
import logging
import os
from typing import Any, Dict, Union
from typing import List, Optional

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from matplotlib.axes import Axes

logger = logging.getLogger(__name__)


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

        self.adata: Optional[AnnData] = None

    """
    Construct an adata from the source adata list.
    """

    def construct_adata(self, var, require_features=False):
        cell_ids = [c.id for c in self.cells]
        if len(cell_ids) == 0:
            self.adata = None
            return

        obs = pd.DataFrame(index=pd.Index(cell_ids, name="cell_id"))
        coords = np.asarray([[c.x, c.y] for c in self.cells], dtype=float)

        if require_features:
            self.adata = ad.AnnData(obs=obs, var=var)
        else:
            X = np.stack(
                [c.X for c in self.cells], axis=0
            )  # shape: (n_cells, n_features)
            self.adata = ad.AnnData(X=X, obs=obs, var=var)

        if require_features:
            # Multi-dimensional embeddings belong in obsm, not obs (pandas columns are 1D).
            self.adata.obsm["X_feature"] = np.stack(
                [c.feature for c in self.cells], axis=0
            )
        self.adata.obsm["spatial"] = coords
        self.adata.obs["sample_id"] = self.id
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

    def get_cell_expression(
        self, adata_list: List[AnnData], cell_id: str
    ) -> Optional[np.ndarray]:
        for adata in adata_list:
            if cell_id in adata.obs.index:
                return adata[cell_id, :].X.squeeze()
        return None

    def get_cell_embedding(
        self, adata_list: List[AnnData], cell_id: str, embedding_key: str
    ) -> Optional[np.ndarray]:
        for adata in adata_list:
            if embedding_key in adata.obsm.keys() and cell_id in adata.obs.index:
                return adata[cell_id, :].obsm[embedding_key].squeeze()
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
        parcellation_path: Optional[str],
    ):
        # step 1: load data from source
        logger.info("Loading adata...")
        cell_metadata_list, ccf_coordinates_list, adata_list = [], [], []
        for path in adata_path:
            adata_list.append(sc.read(path))

        logger.info("Loading cell metadata...")
        for path in cell_metadata_path:
            cell_metadata_list.append(pd.read_csv(path))
        for path in ccf_coordinates_path:
            ccf_coordinates_list.append(pd.read_csv(path))
        cell_metadata = pd.concat(cell_metadata_list, axis=0, ignore_index=True)
        ccf_coordinates = pd.concat(ccf_coordinates_list, axis=0, ignore_index=True)
        cell_metadata.drop_duplicates(subset=["cell_label"], inplace=True)
        ccf_coordinates.drop_duplicates(subset=["cell_label"], inplace=True)
        self.merged_cell_metadata = pd.merge(
            cell_metadata, ccf_coordinates, on="cell_label", suffixes=("", "_ccf")
        )
        logger.info("Constructing parcellation tree...")
        self.parcellation_tree = self.parse_parcellation_structure(parcellation_path)

        # step 2: construct cells and samples
        logger.info("Constructing database for niche query...")
        for row in self.merged_cell_metadata.itertuples(index=False):
            parcellation_info = self.parcellation_tree.get(row.parcellation_index, {})
            cell_id = row.cell_label
            sample_id = row.brain_section_label
            if (
                self.target_sample_ids is not None
                and sample_id not in self.target_sample_ids
            ):
                continue
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

        logger.info("Constructing adata for each sample...")
        for i, sample in enumerate(self.samples.values()):
            sample.construct_adata(
                var=adata_list[0].var,
                require_features=(feature_name != "gene_expression"),
            )
            if i > 0 and i % 20 == 0:
                logger.info(f"\t{i} samples constructed...")

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
