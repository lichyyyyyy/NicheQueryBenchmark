
import logging
from typing import List, Optional

import numpy as np
import torch
from torch_geometric.nn import knn_graph
from torch_geometric.utils import k_hop_subgraph

from src.NicheQueryPrototype.database import Cell, Database
import scanpy as sc

logger = logging.getLogger(__name__)


class Niche:
    """
    Represents a niche within a sample.
    """

    def __init__(
            self,
    ):
        # [Required] List of cells that belong to this niche
        self.cells = []

        # [Required] ID of the sample this niche belongs to
        self.sample_id = ''

        # Neighborhood size parameter
        self.k = 0

        # [Required] Parcellation index for this niche
        self.parcellation_index = 0

        # Maximum number of cells allowed in the niche
        self.cell_limit = 0

        # The center cell of the niche (optional)
        self.center_cell = None
        self.feature = None

    def compute_niche_feature(self):
        assert len(self.cells) > 0, 'Empty cells in the niche'
        cell_features = np.array([c.feature.squeeze() for c in self.cells])
        self.feature = cell_features.mean(axis=0)

    """
    Construct a niche by given cells and sample.
    """

    def construct(self, cells: List[Cell], sample_id: str):
        self.cells = cells
        self.sample_id = sample_id

    """
    Construct a niche by parcellation index.
    """

    def construct_by_parcellation(self, db: Database, center_cell: Optional[Cell],
                                  sample_id: Optional[str], parcellation_index: Optional[int]):
        assert center_cell is not None or (
                sample_id is not None and parcellation_index is not None), 'Center cell or parcellation_index should be not null'
        if center_cell is not None:
            self.center_cell = center_cell
            self.sample_id = center_cell.sample_id
            self.parcellation_index = center_cell.parcellation_index
            self.cells.append(center_cell)
        elif sample_id is not None and parcellation_index is not None:
            self.sample_id = sample_id
            self.parcellation_index = parcellation_index
        sample = db.get_sample(self.sample_id)
        assert sample is None, f"Sample {self.sample_id} not found."

        for cell in sample.cells:
            if cell.parcellation_index == self.parcellation_index:
                self.cells.append(cell)
        self.compute_niche_feature()
        logger.info(f'Constructed a niche with {len(self.cells)} cells on {self.sample_id} slice.')

    """
    Construct a niche by a center cell and k-hop. Can limited to parcellations.
    """

    def construct_by_k_hop(self, db: Database, center_cell_id: Optional[str], sample_id: Optional[str], k: int,
                           cell_limit: Optional[int] = None,
                           parcellation_index: Optional[int] = None):
        assert (center_cell_id is not None and db.get_cell(center_cell_id) is not None) or (
                sample_id is not None and parcellation_index is not None), 'Center cell or parcellation_index should be not null'
        if center_cell_id is not None and db.get_cell(center_cell_id) is not None:
            self.center_cell = db.get_cell(center_cell_id)
            self.cells.append(self.center_cell)
            self.sample_id = self.center_cell.sample_id
            self.parcellation_index = self.center_cell.parcellation_index
        elif sample_id is not None and parcellation_index is not None:
            self.sample_id = sample_id
            self.parcellation_index = parcellation_index

        self.k = k
        self.parcellation_index = parcellation_index
        self.cell_limit = cell_limit
        sample = db.get_sample(self.sample_id)
        assert sample is not None, f"Sample {self.sample_id} not found."

        center_cell_idx = -1
        coordinates_list = []
        for idx, c in enumerate(sample.cells):
            if self.center_cell is None:
                if c.parcellation_index == self.parcellation_index:
                    center_cell_idx = idx
                    self.center_cell = c
            elif c.id == self.center_cell.id:
                center_cell_idx = idx
            coordinates_list.append([c.x, c.y])
        assert center_cell_idx != -1, f"Center cell {self.center_cell.id} not found."

        logger.info('Constructing k-hop graph..')
        coordinates = torch.tensor(coordinates_list, dtype=torch.float)
        edge_index = knn_graph(coordinates, k=k, loop=False)
        subset, _, _, _ = k_hop_subgraph(
            node_idx=center_cell_idx,
            num_hops=k,
            edge_index=edge_index,
            relabel_nodes=False
        )
        if parcellation_index is not None:
            neighbour_cells = [sample.cells[i] for i in subset.cpu().tolist() ]
        else:
            neighbour_cells = [sample.cells[i] for i in subset.cpu().tolist() if sample.cells[i].parcellation_index == self.parcellation_index]

        if cell_limit is not None and len(neighbour_cells) >= cell_limit:
            self.cells.extend(neighbour_cells[:(cell_limit-1)])
        else:
            self.cells.extend(neighbour_cells)

        logger.info('Computing niche feature..')
        self.compute_niche_feature()
        logger.info(f'Finished computing niche feature with {len(self.cells)} cells on {self.sample_id} slice.')

    def visualize(self, db: Database):
        sample = db.get_sample(self.sample_id)
        sample.adata.obs["niche_to_query"] = sample.adata.obs_names.isin([c.id for c in self.cells])
        sc.pl.spatial(
            sample.adata,
            color="niche_to_query",
            palette=["lightgrey", "red"],  # False, True
            spot_size=5,title=f"Niche to Query ({sample.id})")
        sample.adata.obs["target_parcellation"] = sample.adata.obs_names.isin([c.id for c in sample.cells if c.parcellation_index == self.parcellation_index])
        sc.pl.spatial(
            sample.adata,
            color="target_parcellation",
            palette=["lightgrey", "red"],  # False, True
            spot_size=5,title=f"Target Parcellation Section ({sample.id})")
