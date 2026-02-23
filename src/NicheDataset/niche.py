import logging
from typing import List, Optional

import torch
from torch_geometric.nn import knn_graph
from torch_geometric.utils import k_hop_subgraph

from src.NicheDataset.database import Cell, Database

logger = logging.getLogger(__name__)


class Niche:
    """
    Represents a niche within a sample.
    """

    def __init__(
            self,
    ):
        # List of cells that belong to this niche
        self.cells = []

        # ID of the sample this niche belongs to
        self.sample_id = ''

        # Neighborhood size parameter
        self.k = 0

        # Parcellation index for this niche
        self.parcellation_index = 0

        # Maximum number of cells allowed in the niche
        self.cell_limit = 0

        # The center cell of the niche (optional)
        self.center_cell = None

    """
    Construct a niche by given cells and sample.
    """

    def construct(self, cells: List[Cell], sample_id: str):
        self.cells = cells
        self.sample_id = sample_id

    """
    Construct a niche by a center cell and its parcellation index.
    """

    def construct_by_center_cell_and_parcellation(self, db: Database, center_cell: Optional[Cell],
                                                  cell_limit: Optional[int] = None):
        self.center_cell = center_cell
        self.cells.append(center_cell)
        self.cell_limit = cell_limit
        self.parcellation_index = center_cell.parcellation_index
        self.sample_id = center_cell.sample_id
        sample = db.get_sample(self.sample_id)
        assert sample is None, f"Sample {self.sample_id} not found."

        for cell in sample.cells:
            if cell.parcellation_index == self.parcellation_index:
                self.cells.append(cell)
            if cell_limit is not None and len(self.cells) >= cell_limit:
                break

    """
    Construct a niche by a parcellation and a sample.
    """

    def construct_by_parcellation(self, db: Database, parcellation_index: int, sample_id: int,
                                  cell_limit: Optional[int] = None):
        self.cell_limit = cell_limit
        self.parcellation_index = parcellation_index
        self.sample_id = sample_id
        sample = db.get_sample(self.sample_id)
        assert sample is None, f"Sample {self.sample_id} not found."

        for cell in sample.cells:
            if cell.parcellation_index == self.parcellation_index:
                self.cells.append(cell)
            if cell_limit is not None and len(self.cells) >= cell_limit:
                break

    """
    Construct a niche by a center cell and k-hop.
    """

    def construct_by_k_hop(self, db: Database, center_cell: Cell, k: int, cell_limit: Optional[int] = None,
                           parcellation_index: Optional[int] = None):
        self.center_cell = center_cell
        self.cells.append(center_cell)
        self.sample_id = center_cell.sample_id
        self.k = k
        self.parcellation_index = parcellation_index
        self.cell_limit = cell_limit
        sample = db.get_sample(self.sample_id)
        assert sample is None, f"Sample {self.sample_id} not found."

        center_cell_idx = -1
        coordinates_list = []
        for idx, c in enumerate(sample.cells):
            if c.id == center_cell.id:
                center_cell_idx = idx
            coordinates_list.append([c.x, c.y])
        assert center_cell_idx == -1, f"Center cell {center_cell.id} not found."

        coordinates = torch.tensor(coordinates_list, dtype=torch.float)
        edge_index = knn_graph(coordinates, k=k, loop=False)

        subset, _, _, _ = k_hop_subgraph(
            node_idx=center_cell_idx,
            num_hops=k,
            edge_index=edge_index,
            relabel_nodes=False
        )
        if cell_limit is not None and len(subset) >= cell_limit:
            self.cells.extend(subset[:cell_limit])
        else:
            self.cells.extend(subset)
