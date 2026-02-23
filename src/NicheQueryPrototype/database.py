import json
from typing import Dict, Tuple
from typing import List, Union, Optional

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData


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
        self.feature = None

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


class Database:
    """
    Represents a database containing multiple samples.
    """

    def __init__(self):
        self.cells: List[Cell] = []
        self.samples: Dict[str, Sample] = {}
        self.adata_list: List[AnnData] = []
        self.merged_cell_metadata: pd.DataFrame = pd.DataFrame()
        self.parcellation_tree = {}

    def parse_parcellation_structure(self, json_path: str) -> Dict[int, Tuple[int, str, int]]:
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

    def get_cell_expression(self, cell_id: str) -> Optional[np.ndarray]:
        for adata in self.adata_list:
            if cell_id in adata.obs.index:
                return adata[cell_id, :].X
        return None

    def get_cell_embedding(self, cell_id: str, embedding_key: str) -> Optional[np.ndarray]:
        for adata in self.adata_list:
            if embedding_key in adata.obsm.keys() and cell_id in adata.obs.index:
                return adata[cell_id, :].obsm[embedding_key]
        return None

    """
    Loads and construct the database.
    
    `feature_name`: 'gene_expression' or the name of embeddings in adata.obsm.
    """

    def construct(self, adata_path: Union[str, List[str]], cell_metadata_path: Union[str, List[str]],
                  ccf_coordinates_path: Union[str, List[str]], feature_name: str, parcellation_path: Optional[str]):
        # step 1: load data from source
        cell_metadata_list, ccf_coordinates_list = [], []
        for path in adata_path:
            self.adata_list.append(sc.read(path))
        for path in cell_metadata_path:
            cell_metadata_list.append(pd.read_csv(path))
        for path in ccf_coordinates_path:
            ccf_coordinates_list.append(pd.read_csv(path))
        cell_metadata = pd.concat(cell_metadata_list)
        ccf_coordinates = pd.concat(ccf_coordinates_list)
        cell_metadata.drop_duplicates(subset=["cell_label"], inplace=True)
        ccf_coordinates.drop_duplicates(subset=["cell_label"], inplace=True)
        self.merged_cell_metadata = pd.merge(cell_metadata, ccf_coordinates, on="cell_label")
        self.parcellation_tree = self.parse_parcellation_structure(parcellation_path)

        # step 2: construct cells and samples
        for row in self.merged_cell_metadata.itertuples(index=False):
            parcellation_level = self.parcellation_tree.get(row['parcellation_index'], -1)
            cell_id = row['cell_label']
            sample_id = row['brain_section_label']
            cell = Cell(x=row['x'], y=row['y'], z=row['z'], cell_id=cell_id,
                        parcellation_index=row['parcellation_index'],
                        parcellation_level=parcellation_level, sample_id=sample_id)
            if feature_name == 'gene_expression':
                cell.feature = self.get_cell_expression(cell_id)
            else:
                cell.feature = self.get_cell_embedding(cell_id, feature_name)
            self.cells.append(cell)
            sample = self.samples.get(sample_id, Sample(sample_id=sample_id))
            sample.cells.append(cell)

    def get_sample(self, sample_id: str) -> Optional[Sample]:
        return self.samples.get(sample_id, None)
