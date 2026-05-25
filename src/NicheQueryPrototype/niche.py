import logging
import os
from collections import Counter
from typing import List, Optional

import numpy as np
import pandas as pd
from matplotlib.colors import hsv_to_rgb, to_hex
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


def _niche_highlight_parcellation_set(niche: "Niche") -> set[int]:
    """
    Parcellation indices to color in ``visualize``; all other spots use gray.
    ``niche.parcellation_index`` may be int (e.g. center cell) or list (k-hop allowlist).
    """
    p = niche.parcellation_index
    if isinstance(p, (list, tuple, set)):
        s = {int(x) for x in p}
        if s:
            return s
    elif isinstance(p, int):
        return {p}
    return {c.parcellation_index for c in niche.cells}


def _distinct_category_hexes(n: int) -> List[str]:
    """
    Strongly separated colors for categorical parcellation plots (not including gray).
    Uses golden-ratio hue steps plus staggered saturation/value so nearby legend
    entries do not look like tab20 repeats.
    """
    if n <= 0:
        return []
    phi = (np.sqrt(5.0) - 1.0) / 2.0
    out: List[str] = []
    for i in range(n):
        h = float((0.11 + i * phi) % 1.0)
        s = 0.68 + 0.30 * ((i * 3) % 2)
        v = 0.72 + 0.24 * ((i * 5) % 2)
        rgb = np.asarray(hsv_to_rgb([h, s, v]), dtype=float)
        out.append(to_hex(np.clip(rgb, 0.0, 1.0)))
    return out


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
        center_cell: Optional[Cell] = None,
        sample_id: Optional[str] = None,
        parcellation_index: Optional[int] = None,
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
        center_cell_id: Optional[str] = None,
        sample_id: Optional[str] = None,
        k: int = 5,
        cell_limit: Optional[int] = None,
        parcellation_index: Optional[List[int]] = None,
        niche_cells_export_path: Optional[str] = None,
        niche_name: Optional[str] = None,
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
        if niche_name is not None and str(niche_name).strip() != "":
            key = str(niche_name).strip()
            adata = sample.adata
            if adata is None:
                logger.warning(
                    "niche_name=%r set but sample %r has no adata; skip writing obsm.",
                    key,
                    self.sample_id,
                )
            else:
                niche_ids = {str(c.id) for c in self.cells}
                n_obs = adata.n_obs
                flags = np.fromiter(
                    (1.0 if str(oid) in niche_ids else 0.0 for oid in adata.obs_names),
                    dtype=np.float64,
                    count=n_obs,
                )
                adata.obsm[key] = flags.reshape(n_obs, 1)
                logger.info(
                    "Wrote niche membership to adata.obsm[%r] (n_in_niche=%d, n_obs=%d).",
                    key,
                    int(flags.sum()),
                    n_obs,
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

        highlight_pidx = _niche_highlight_parcellation_set(self)

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

        highlight_sorted = sorted(highlight_pidx)
        pidx_to_label: dict[int, str] = {}
        for p in highlight_sorted:
            rep = next((c for c in self.cells if c.parcellation_index == p), None)
            if rep is None:
                rep = next((c for c in sample.cells if c.parcellation_index == p), None)
            pidx_to_label[p] = (
                _parcellation_display_label(rep) if rep is not None else str(p)
            )

        id_to_cell = {str(c.id): c for c in self.cells}
        labels: List[str] = []
        for oid in sample.adata.obs_names:
            c = id_to_cell.get(str(oid))
            if c is None or c.parcellation_index not in highlight_pidx:
                labels.append("—")
            else:
                labels.append(pidx_to_label[c.parcellation_index])

        categories = ["—"] + [pidx_to_label[p] for p in highlight_sorted]
        sample.adata.obs["niche_parcellation"] = pd.Categorical(
            labels, categories=categories
        )

        n_parcel = len(highlight_sorted)
        niche_colors = _distinct_category_hexes(n_parcel)
        palette = ["#d9d9d9"] + niche_colors

        sc.pl.spatial(
            sample.adata,
            color="niche_parcellation",
            palette=palette,
            spot_size=spot_size,
            title=f"Niche by parcellation ({sample.id})",
        )

        # Full sample: only spots whose parcellation_index is in niche.parcellation_index; else gray.
        id_to_full_cell = {str(c.id): c for c in sample.cells}
        sample_pidx_labels: List[str] = []
        for oid in sample.adata.obs_names:
            cf = id_to_full_cell.get(str(oid))
            if cf is None or cf.parcellation_index not in highlight_pidx:
                sample_pidx_labels.append("—")
            else:
                sample_pidx_labels.append(pidx_to_label[cf.parcellation_index])
        categories_all = ["—"] + [pidx_to_label[p] for p in highlight_sorted]
        sample.adata.obs["sample_parcellation_index"] = pd.Categorical(
            sample_pidx_labels, categories=categories_all
        )
        palette_all = ["#d9d9d9"] + niche_colors
        sc.pl.spatial(
            sample.adata,
            color="sample_parcellation_index",
            palette=palette_all,
            spot_size=spot_size,
            title=(f"Highlighted parcellation(s) on full sample ({sample.id})"),
        )
