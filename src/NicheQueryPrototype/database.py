import json
import logging
from typing import Dict, Union
from typing import List, Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

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
            parcellation_level: int,
            sample_id: str
    ):
        # Spatial coordinates
        self.x = x
        self.y = y
        self.z = z

        # Unique identifier for the cell
        self.id = cell_id

        # Parcellation information
        self.parcellation_index = parcellation_index
        self.parcellation_level = parcellation_level

        # ID of the sample this cell belongs to
        self.sample_id = sample_id
        self.feature:np.ndarray = None

        # The cosine similarity between this cell and the query cell.
        self.similarity: float = 0.0


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

    def construct_adata(self, adata_list: List[AnnData]):
        cell_ids = set([c.id for c in self.cells])

        filtered_adata_list = [
            adata[adata.obs_names.isin(cell_ids)].copy()
            for adata in adata_list
        ]
        filtered_adata_list = [adata for adata in filtered_adata_list if adata.n_obs > 0]

        if len(filtered_adata_list) > 0:
            self.adata = ad.concat(filtered_adata_list, join="outer", merge="same")
        else:
            self.adata = None
            return

        coords_df = pd.DataFrame(
            {
                "x": [c.x for c in self.cells],
                "y": [c.y for c in self.cells],
            },
            index=[c.id for c in self.cells],
        )
        coords_df = coords_df.reindex(self.adata.obs_names)
        self.adata.obsm["X_spatial"] = coords_df[["x", "y"]].to_numpy(dtype=float)


class Database:
    """
    Represents a database containing multiple samples.
    """

    def __init__(self):
        self.cells: List[Cell] = []
        self.samples: Dict[str, Sample] = {}
        self.merged_cell_metadata: pd.DataFrame = pd.DataFrame()
        self.parcellation_tree = {}

    def parse_parcellation_structure(self, json_path: str) -> Dict[int, Dict[str, Union[str, int]]]:
        """
        Parse parcellation structure and return a dictionary:

        key: structure id
        value: (parent_id, name, st_level)
        """

        with open(json_path, "r") as f:
            data = json.load(f)

        result = {}

        def traverse(node: dict):
            """
            Recursively traverse the tree and extract required fields.
            """
            structure_id = node["id"]
            parent_id = node["parent_structure_id"]
            name = node["name"]
            st_level = node["st_level"]

            result[structure_id] = {
                'parent_id': parent_id,
                'name': name,
                'st_level': st_level,
            }

            # Recursively process children
            for child in node.get("children", []):
                traverse(child)

        # Root is inside data["msg"]
        for root_node in data["msg"]:
            traverse(root_node)

        return result

    def get_cell_expression(self, adata_list: List[AnnData], cell_id: str) -> Optional[np.ndarray]:
        for adata in adata_list:
            if cell_id in adata.obs.index:
                return adata[cell_id, :].X.squeeze()
        return None

    def get_cell_embedding(self, adata_list: List[AnnData], cell_id: str, embedding_key: str) -> Optional[np.ndarray]:
        for adata in adata_list:
            if embedding_key in adata.obsm.keys() and cell_id in adata.obs.index:
                return adata[cell_id, :].obsm[embedding_key].squeeze()
        return None

    """
    Loads and construct the database.
    
    `feature_name`: 'gene_expression' or the name of embeddings in adata.obsm.
    """

    def construct(self, adata_path: List[str], cell_metadata_path: List[str],
                  ccf_coordinates_path: List[str], feature_name: str, parcellation_path: Optional[str]):
        # step 1: load data from source
        logger.info('Loading adata...')
        cell_metadata_list, ccf_coordinates_list, adata_list = [], [], []
        for path in adata_path:
            adata_list.append(sc.read(path))

        logger.info('Loading cell metadata...')
        for path in cell_metadata_path:
            cell_metadata_list.append(pd.read_csv(path))
        for path in ccf_coordinates_path:
            ccf_coordinates_list.append(pd.read_csv(path))
        cell_metadata = pd.concat(cell_metadata_list, axis=0, ignore_index=True)
        ccf_coordinates = pd.concat(ccf_coordinates_list, axis=0, ignore_index=True)
        cell_metadata.drop_duplicates(subset=["cell_label"], inplace=True)
        ccf_coordinates.drop_duplicates(subset=["cell_label"], inplace=True)
        self.merged_cell_metadata = pd.merge(cell_metadata, ccf_coordinates, on="cell_label", suffixes=("", "_ccf"))
        logger.info('Constructing parcellation tree...')
        self.parcellation_tree = self.parse_parcellation_structure(parcellation_path)

        # step 2: construct cells and samples
        logger.info('Constructing database for niche query...')
        for row in self.merged_cell_metadata.itertuples(index=False):
            parcellation_info = self.parcellation_tree.get(row.parcellation_index, {})
            cell_id = row.cell_label
            sample_id = row.brain_section_label
            cell = Cell(x=row.x_ccf, y=row.y_ccf, z=row.z_ccf, cell_id=cell_id,
                        parcellation_index=row.parcellation_index,
                        parcellation_level=parcellation_info.get('st_level', -1), sample_id=sample_id)
            if feature_name == 'gene_expression':
                cell.feature = self.get_cell_expression(adata_list, cell_id)
            else:
                cell.feature = self.get_cell_embedding(adata_list, cell_id, feature_name)
            self.cells.append(cell)
            if sample_id in self.samples.keys():
                self.samples[sample_id].cells.append(cell)
            else:
                sample = Sample(sample_id=sample_id)
                sample.cells.append(cell)
                self.samples[sample_id] = sample
        logger.info(f'Processed {len(self.cells)} cells and {len(self.samples.keys())} samples.')

        logger.info('Constructing adata for each sample...')
        for sample in self.samples.values():
            sample.construct_adata(adata_list)

    def get_sample(self, sample_id: str) -> Optional[Sample]:
        return self.samples.get(sample_id, None)
