from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy.optimize import linprog

if TYPE_CHECKING:
    from .database import Cell, Sample


class RmIdeal:
    """
    Compute RM-Ideal scores between a fixed query niche and every candidate
    center cell on a sample slice.

    Assumptions
    -----------
    Each cell is a ``Cell`` object from ``database.py`` with attributes
    ``x``, ``y``, and ``parcellation_index``.

    Notes
    -----
    - RM-Ideal uses only spatial coordinates + categorical labels.
    - Gene expression is intentionally not used.
    - Query niche is given directly as a list of cells.
    - Candidate niche for each slice cell is defined as its K-hop neighborhood
      on the slice graph.
    """

    def __init__(
        self,
        wl_iters: int = 3,
        graph_mode: str = "delaunay",
        knn_k: int = 6,
        radius: Optional[float] = None,
        post_transform: str = "linear",
        temperature: float = 0.1,
    ) -> None:
        """
        Parameters
        ----------
        wl_iters
            Number of WL iterations. Also used as K-hop radius for candidate niches.
        graph_mode
            One of {"delaunay", "knn", "radius"}.
        knn_k
            Number of neighbors if graph_mode == "knn".
        radius
            Radius threshold if graph_mode == "radius".
        post_transform
            Score post-processing mode. One of {"linear", "sigmoid", "rank"}.
        temperature
            Temperature for sigmoid post-processing (smaller -> stronger contrast).
        """
        self.wl_iters = wl_iters
        self.graph_mode = graph_mode
        self.knn_k = knn_k
        self.radius = radius
        self.post_transform = post_transform
        self.temperature = temperature

    # =========================
    # Public API
    # =========================
    def score_slice(
        self,
        query_niche_cells: List["Cell"],
        samples: "Sample",
    ) -> np.ndarray:
        """
        Compute RM-Ideal score for every cell in ``samples.cells``.

        Parameters
        ----------
        query_niche_cells
            List of cells that already form the query niche.
        samples
            Target sample slice.

        Returns
        -------
        scores : np.ndarray of shape (len(samples.cells),)
            scores[i] = RM-Ideal score for cell i as candidate niche center.
        """
        if not hasattr(samples, "cells"):
            raise ValueError("samples must be a Sample object with .cells")
        slice_cells = samples.cells

        self._validate_cells(query_niche_cells, name="query_niche_cells")
        self._validate_cells(slice_cells, name="samples.cells")

        # Shared categorical label vocabulary across query + slice
        all_labels = [self._cell_parcellation(c) for c in query_niche_cells] + [
            self._cell_parcellation(c) for c in slice_cells
        ]
        label_to_int, _ = self._encode_labels(all_labels)

        # Build query niche features
        query_coords = self._extract_coords(query_niche_cells)
        query_labels_int = self._extract_encoded_labels(query_niche_cells, label_to_int)
        query_adj = self._build_graph(query_coords, n_points=len(query_niche_cells))
        query_nodes = np.arange(len(query_niche_cells), dtype=int)
        query_feats = self._wl_node_features(
            labels_int=query_labels_int,
            adj=query_adj,
            nodes=query_nodes,
            wl_iters=self.wl_iters,
        )

        # Build slice graph
        slice_coords = self._extract_coords(slice_cells)
        slice_labels_int = self._extract_encoded_labels(slice_cells, label_to_int)
        slice_adj = self._build_graph(slice_coords, n_points=len(slice_cells))

        # Score every slice cell as center
        scores = np.empty(len(slice_cells), dtype=float)
        for center in range(len(slice_cells)):
            cand_nodes = self._extract_k_hop_nodes(
                adj=slice_adj,
                center=center,
                k=self.wl_iters,
            )
            cand_feats = self._wl_node_features(
                labels_int=slice_labels_int,
                adj=slice_adj,
                nodes=cand_nodes,
                wl_iters=self.wl_iters,
            )
            scores[center] = self._rm_score_from_features(query_feats, cand_feats)

        return self._post_transform_scores(scores)

    def best_match(
        self,
        query_niche_cells: List["Cell"],
        samples: "Sample",
    ) -> Tuple[int, float]:
        """
        Return the best candidate center and its RM-Ideal score.
        """
        scores = self.score_slice(query_niche_cells, samples)
        best_idx = int(np.argmax(scores))
        return best_idx, float(scores[best_idx])

    # =========================
    # Internal helpers
    # =========================
    @staticmethod
    def _validate_cells(cells: List["Cell"], name: str) -> None:
        if len(cells) == 0:
            raise ValueError(f"{name} is empty")
        for i, cell in enumerate(cells):
            try:
                _ = RmIdeal._cell_xy(cell)
                _ = RmIdeal._cell_parcellation(cell)
            except (AttributeError, TypeError, ValueError) as e:
                raise ValueError(f"{name}[{i}] is invalid cell: {e}") from e

    @staticmethod
    def _extract_coords(cells: List["Cell"]) -> np.ndarray:
        return np.array([RmIdeal._cell_xy(c) for c in cells], dtype=float)

    @staticmethod
    def _encode_labels(labels: List[Any]) -> Tuple[Dict[Any, int], np.ndarray]:
        unique = list(dict.fromkeys(labels))
        mapping = {lab: i for i, lab in enumerate(unique)}
        encoded = np.array([mapping[x] for x in labels], dtype=int)
        return mapping, encoded

    @staticmethod
    def _extract_encoded_labels(
        cells: List["Cell"],
        label_to_int: Dict[Any, int],
    ) -> np.ndarray:
        return np.array(
            [label_to_int[RmIdeal._cell_parcellation(c)] for c in cells], dtype=int
        )

    @staticmethod
    def _cell_xy(cell: "Cell") -> Tuple[float, float]:
        if not hasattr(cell, "x") or not hasattr(cell, "y"):
            raise AttributeError("Cell must have attributes 'x' and 'y'")
        return float(cell.x), float(cell.y)

    @staticmethod
    def _cell_parcellation(cell: "Cell") -> Any:
        if hasattr(cell, "parcellation_index"):
            return cell.parcellation_index
        raise AttributeError("Cell must have attribute 'parcellation_index'")

    def _build_graph(self, coords: np.ndarray, n_points: int) -> List[List[int]]:
        if self.graph_mode == "delaunay":
            if n_points < 3:
                return self._build_knn_adj(
                    coords,
                    k=min(2, max(1, n_points - 1)),
                )
            return self._build_delaunay_adj(coords)

        if self.graph_mode == "knn":
            return self._build_knn_adj(
                coords,
                k=min(self.knn_k, max(1, n_points - 1)),
            )

        if self.graph_mode == "radius":
            if self.radius is None:
                raise ValueError("radius must be provided when graph_mode='radius'")
            return self._build_radius_adj(coords, self.radius)

        raise ValueError("graph_mode must be one of {'delaunay', 'knn', 'radius'}")

    @staticmethod
    def _build_delaunay_adj(coords: np.ndarray) -> List[List[int]]:
        tri = Delaunay(coords)
        edges = set()

        for simplex in tri.simplices:
            for i in range(len(simplex)):
                for j in range(i + 1, len(simplex)):
                    a, b = int(simplex[i]), int(simplex[j])
                    edges.add(tuple(sorted((a, b))))

        n = coords.shape[0]
        adj = [[] for _ in range(n)]
        for i, j in edges:
            if i != j:
                adj[i].append(j)
                adj[j].append(i)

        return [sorted(set(nei)) for nei in adj]

    @staticmethod
    def _build_knn_adj(coords: np.ndarray, k: int) -> List[List[int]]:
        tree = cKDTree(coords)
        _, idx = tree.query(coords, k=k + 1)

        n = coords.shape[0]
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in np.atleast_1d(idx[i])[1:]:
                j = int(j)
                if i != j:
                    adj[i].append(j)
                    adj[j].append(i)

        return [sorted(set(nei)) for nei in adj]

    @staticmethod
    def _build_radius_adj(coords: np.ndarray, radius: float) -> List[List[int]]:
        tree = cKDTree(coords)
        pairs = tree.query_pairs(radius)

        n = coords.shape[0]
        adj = [[] for _ in range(n)]
        for i, j in pairs:
            if i != j:
                adj[i].append(j)
                adj[j].append(i)

        return [sorted(set(nei)) for nei in adj]

    @staticmethod
    def _extract_k_hop_nodes(adj: List[List[int]], center: int, k: int) -> np.ndarray:
        visited = {center}
        frontier = {center}

        for _ in range(k):
            nxt = set()
            for u in frontier:
                nxt.update(adj[u])
            nxt -= visited
            visited.update(nxt)
            frontier = nxt
            if not frontier:
                break

        return np.array(sorted(visited), dtype=int)

    @staticmethod
    def _induced_subgraph_adj(
        adj: List[List[int]], nodes: np.ndarray
    ) -> List[List[int]]:
        node_set = set(nodes.tolist())
        g2l = {g: i for i, g in enumerate(nodes.tolist())}

        sub_adj = [[] for _ in range(len(nodes))]
        for g in nodes:
            i = g2l[g]
            for ngh in adj[g]:
                if ngh in node_set:
                    sub_adj[i].append(g2l[ngh])

        return [sorted(set(nei)) for nei in sub_adj]

    def _wl_node_features(
        self,
        labels_int: np.ndarray,
        adj: List[List[int]],
        nodes: np.ndarray,
        wl_iters: int,
    ) -> np.ndarray:
        """
        Compute node feature vectors:
            f(v) = [h0(v), h1(v), ..., hK(v)]
        """
        sub_adj = self._induced_subgraph_adj(adj, nodes)

        h_prev = labels_int[nodes].astype(int)
        all_h = [h_prev.copy()]

        for _ in range(wl_iters):
            hash_table: Dict[Tuple[int, ...], int] = {}
            h_next = np.empty_like(h_prev)

            for v in range(len(nodes)):
                neigh_labels = [int(h_prev[u]) for u in sub_adj[v]]
                key = tuple(sorted([int(h_prev[v])] + neigh_labels))
                if key not in hash_table:
                    hash_table[key] = len(hash_table)
                h_next[v] = hash_table[key]

            h_prev = h_next
            all_h.append(h_prev.copy())

        return np.stack(all_h, axis=1)

    @staticmethod
    def _hamming_cost_matrix(feats_a: np.ndarray, feats_b: np.ndarray) -> np.ndarray:
        neq = feats_a[:, None, :] != feats_b[None, :, :]
        return neq.mean(axis=2).astype(float)

    @staticmethod
    def _emd_uniform_exact(cost: np.ndarray) -> float:
        """
        Exact Wasserstein / EMD between two uniform empirical distributions.
        """
        na, nb = cost.shape
        a = np.full(na, 1.0 / na, dtype=float)
        b = np.full(nb, 1.0 / nb, dtype=float)

        c = cost.ravel()
        A_eq = []
        b_eq = []

        # Row constraints
        for i in range(na):
            row = np.zeros(na * nb, dtype=float)
            row[i * nb : (i + 1) * nb] = 1.0
            A_eq.append(row)
            b_eq.append(a[i])

        # Column constraints
        for j in range(nb):
            col = np.zeros(na * nb, dtype=float)
            col[j::nb] = 1.0
            A_eq.append(col)
            b_eq.append(b[j])

        A_eq = np.vstack(A_eq)
        b_eq = np.array(b_eq, dtype=float)
        bounds = [(0.0, None)] * (na * nb)

        res = linprog(
            c=c,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method="highs",
        )
        if not res.success:
            raise RuntimeError(f"Optimal transport failed: {res.message}")

        return float(res.fun)

    def _rm_score_from_features(
        self, query_feats: np.ndarray, cand_feats: np.ndarray
    ) -> float:
        cost = self._hamming_cost_matrix(query_feats, cand_feats)
        w = self._emd_uniform_exact(cost)
        return float(np.clip(1.0 - w, 0.0, 1.0))

    def _post_transform_scores(self, raw_scores: np.ndarray) -> np.ndarray:
        # Base linear normalization to [0, 1].
        s_min = float(np.min(raw_scores))
        s_max = float(np.max(raw_scores))
        if np.isclose(s_max, s_min):
            linear = np.zeros_like(raw_scores, dtype=float)
        else:
            linear = (raw_scores - s_min) / (s_max - s_min)

        if self.post_transform == "linear":
            return linear
        if self.post_transform == "sigmoid":
            if self.temperature <= 0:
                raise ValueError(
                    "temperature must be > 0 when post_transform='sigmoid'"
                )
            z = (linear - 0.5) / float(self.temperature)
            return 1.0 / (1.0 + np.exp(-z))
        if self.post_transform == "rank":
            if linear.size <= 1:
                return np.zeros_like(linear, dtype=float)
            order = np.argsort(linear, kind="mergesort")
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(order.size, dtype=float)
            return ranks / float(order.size - 1)
        logger.info(
            f"No post_transform or unknown post_transform: {self.post_transform}. Returning raw scores."
        )
        return raw_scores
