import logging
import os
from collections import Counter
from typing import List, Optional

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
import torch
from torch_geometric.nn import knn_graph
from torch_geometric.utils import k_hop_subgraph

from src.NicheQueryPrototype.database import Cell, Database
import scanpy as sc

logger = logging.getLogger(__name__)


def _parcellation_display_label(c: Cell) -> str:
    pinfo = c.parcellation_info or {}
    name = (pinfo.get("name") or "").strip()
    if name:
        return f"{c.parcellation_index}: {name}"
    return str(c.parcellation_index)


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
        self.sample_id = ""

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
        assert len(self.cells) > 0, "Empty cells in the niche"
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

    def construct_by_parcellation(
        self,
        db: Database,
        center_cell: Optional[Cell],
        sample_id: Optional[str],
        parcellation_index: Optional[int],
    ):
        assert center_cell is not None or (
            sample_id is not None and parcellation_index is not None
        ), "Center cell or parcellation_index should be not null"
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
        logger.info(
            f"Constructed a niche with {len(self.cells)} cells on {self.sample_id} slice."
        )

    """
    Construct a niche by a center cell and k-hop. Can limited to parcellations.
    """

    def construct_by_k_hop(
        self,
        db: Database,
        center_cell_id: Optional[str],
        sample_id: Optional[str],
        k: int,
        cell_limit: Optional[int] = None,
        parcellation_index: Optional[List[int]] = None,
        niche_cells_export_path: Optional[str] = None,
    ):
        if isinstance(parcellation_index, int):
            parcellation_index = [parcellation_index]
        allowed_parcellations = (
            set(parcellation_index) if parcellation_index is not None else None
        )
        assert (
            center_cell_id is not None and db.get_cell(center_cell_id) is not None
        ) or (
            sample_id is not None and parcellation_index is not None
        ), "Center cell or parcellation_index should be not null"
        if center_cell_id is not None and db.get_cell(center_cell_id) is not None:
            self.center_cell = db.get_cell(center_cell_id)
            self.sample_id = self.center_cell.sample_id
            self.parcellation_index = self.center_cell.parcellation_index
        elif sample_id is not None and parcellation_index is not None:
            self.sample_id = sample_id
            # Keep a representative value for display/logging when constructed from a list.
            self.parcellation_index = (
                parcellation_index[0] if len(parcellation_index) > 0 else -1
            )

        self.k = k
        self.parcellation_index = parcellation_index
        self.cell_limit = cell_limit
        sample = db.get_sample(self.sample_id)
        assert sample is not None, f"Sample {self.sample_id} not found."

        center_cell_idx = -1
        coordinates_list = []
        for idx, c in enumerate(sample.cells):
            if self.center_cell is None:
                if (
                    allowed_parcellations is not None
                    and c.parcellation_index in allowed_parcellations
                ):
                    center_cell_idx = idx
                    self.center_cell = c
            elif c.id == self.center_cell.id:
                center_cell_idx = idx
            coordinates_list.append([c.x, c.y])
        assert center_cell_idx != -1, "Center cell not found."

        logger.info("Constructing k-hop graph..")
        coordinates = torch.tensor(coordinates_list, dtype=torch.float)
        edge_index = knn_graph(coordinates, k=k, loop=False)
        subset, _, _, _ = k_hop_subgraph(
            node_idx=center_cell_idx,
            num_hops=k,
            edge_index=edge_index,
            relabel_nodes=False,
        )
        if allowed_parcellations is None:
            neighbour_cells = [sample.cells[i] for i in subset.cpu().tolist()]
        else:
            neighbour_cells = [
                sample.cells[i]
                for i in subset.cpu().tolist()
                if sample.cells[i].parcellation_index in allowed_parcellations
            ]

        if cell_limit is not None and len(neighbour_cells) >= cell_limit:
            self.cells.extend(neighbour_cells[:cell_limit])
        else:
            self.cells.extend(neighbour_cells)

        logger.info("Computing niche feature..")
        self.compute_niche_feature()
        logger.info(
            f"Finished computing niche feature with {len(self.cells)} cells on {self.sample_id} slice."
        )
        if niche_cells_export_path:
            lines = [
                "cell_id\tx\ty\tz\tparcellation_index\tparcellation_name",
            ]
            for c in self.cells:
                pinfo = c.parcellation_info or {}
                pname = pinfo.get("name")
                if pname is None:
                    pname = ""
                lines.append(
                    f"{c.id}\t{c.x}\t{c.y}\t{c.z}\t{c.parcellation_index}\t{pname}"
                )
            text = "\n".join(lines) + "\n"
            print(text, end="")
            out_abs = os.path.abspath(niche_cells_export_path)
            parent = os.path.dirname(out_abs)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(niche_cells_export_path, "w", encoding="utf-8") as f:
                f.write(text)
            logger.info(
                "Wrote niche cells (n=%d) to %r.",
                len(self.cells),
                niche_cells_export_path,
            )

    def visualize(self, db: Database, spot_size: float = 0.02):
        n_cells = len(self.cells)
        logger.info(f"Niche contains {n_cells} cell(s) (sample_id={self.sample_id!r})")
        sample = db.get_sample(self.sample_id)

        cnt = Counter(c.parcellation_index for c in self.cells)
        print(f"Niche cells by parcellation (n_total={n_cells}):")
        for pidx in sorted(cnt):
            name = ""
            for c in self.cells:
                if c.parcellation_index == pidx:
                    name = (c.parcellation_info or {}).get("name") or ""
                    break
            print(
                f"  parcellation_index={pidx}  parcellation_name={name!r}  n_cells={cnt[pidx]}"
            )

        index_to_label: dict[int, str] = {}
        for c in sorted(self.cells, key=lambda x: x.parcellation_index):
            if c.parcellation_index not in index_to_label:
                index_to_label[c.parcellation_index] = _parcellation_display_label(c)

        id_to_cell = {str(c.id): c for c in self.cells}
        labels: List[str] = []
        for oid in sample.adata.obs_names:
            c = id_to_cell.get(str(oid))
            if c is None:
                labels.append("—")
            else:
                labels.append(index_to_label[c.parcellation_index])

        categories = ["—"] + [index_to_label[i] for i in sorted(index_to_label)]
        sample.adata.obs["niche_parcellation"] = pd.Categorical(
            labels, categories=categories
        )

        try:
            tab20 = matplotlib.colormaps["tab20"]
        except AttributeError:
            from matplotlib import cm

            tab20 = cm.get_cmap("tab20")
        n_parcel = len(index_to_label)
        niche_colors = [to_hex(tab20((i % 20) / 19.0)) for i in range(n_parcel)]
        palette = ["#d9d9d9"] + niche_colors

        sc.pl.spatial(
            sample.adata,
            color="niche_parcellation",
            palette=palette,
            spot_size=spot_size,
            title=f"Niche by parcellation ({sample.id})",
        )
