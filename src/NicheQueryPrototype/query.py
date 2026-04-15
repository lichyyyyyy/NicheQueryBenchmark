from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    classification_report,
    normalized_mutual_info_score,
    precision_recall_curve,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import torch
from torch_cluster import knn_graph

from src.NicheQueryPrototype.database import (
    Database,
    Sample,
    cell_matches_parcellation_or_ancestor,
)
import scanpy as sc
from src.NicheQueryPrototype.niche import Niche, logger


def _append_result_txt(path: Optional[str], text: str) -> None:
    """Append ``text`` to ``path``; create parent directories if needed."""
    if path is None:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def _prob_vector_from_counts(counts: np.ndarray) -> np.ndarray:
    s = float(counts.sum())
    if s <= 0:
        p = np.ones_like(counts, dtype=np.float64) / len(counts)
        return p
    p = counts.astype(np.float64) / s
    return p / (p.sum() + 1e-12)


def _niche_composition_matrix(
    samples: List[Sample], spatial_k: int, n_bins: int
) -> np.ndarray:
    """
    For each cell, P_i is the normalized parcellation histogram over spatial kNN (Euclidean on x,y).
    """
    rows: List[np.ndarray] = []
    for s in samples:
        n = len(s.cells)
        coords = np.asarray([[c.x, c.y] for c in s.cells], dtype=np.float64)
        k_use = min(max(1, spatial_k), n)
        nbrs = NearestNeighbors(n_neighbors=k_use, metric="euclidean").fit(coords)
        _, ind = nbrs.kneighbors(coords)
        par = np.array([c.parcellation_index for c in s.cells], dtype=int)
        for neigh_idx in ind:
            bc = np.bincount(par[neigh_idx], minlength=n_bins)
            rows.append(_prob_vector_from_counts(bc))
    return np.stack(rows, axis=0)


def _average_ncjs_score(P: np.ndarray, emb: np.ndarray, k_embedding: int) -> float:
    """
    Average NCJS (QueST): for each cell, mean over k_embedding NN in embedding space of
    (1 - JS(P_i, P_j)), using Jensen--Shannon distance from scipy (sqrt of JS divergence).
    """
    n = P.shape[0]
    if n < 2 or k_embedding < 1:
        return float("nan")
    k_use = min(k_embedding, n - 1)
    nn = NearestNeighbors(n_neighbors=k_use + 1, metric="euclidean").fit(emb)
    _, ind = nn.kneighbors(emb)
    vals: List[float] = []
    for i in range(n):
        for j in ind[i, 1 : k_use + 1]:
            d = float(jensenshannon(P[i], P[j]))
            vals.append(1.0 - d**2)
    return float(np.mean(vals)) if vals else float("nan")


def _knn_batch_mixing(
    X: np.ndarray, batch_labels: np.ndarray, n_neighbors: int
) -> float:
    """Mean fraction of embedding kNN cells that belong to a different batch (higher = more mixed)."""
    n = X.shape[0]
    k_use = min(max(1, n_neighbors), max(1, n - 1))
    nn = NearestNeighbors(n_neighbors=k_use + 1, metric="euclidean").fit(X)
    _, ind = nn.kneighbors(X)
    fracs: List[float] = []
    for i in range(n):
        neigh = ind[i, 1 : k_use + 1]
        fracs.append(float(np.mean(batch_labels[neigh] != batch_labels[i])))
    return float(np.mean(fracs)) if fracs else float("nan")


def _avg_bio_batch(
    X: np.ndarray,
    bio_labels: np.ndarray,
    batch_labels: np.ndarray,
    batch_k_neighbors: int,
) -> Tuple[float, float]:
    """
    AvgBIO = mean(ARI, NMI, ASW_bio). AvgBATCH = mean of (1 - ASW_batch) / 2 and kNN
    batch-mixing (QueST / scIB-style batch summary; needs >=2 batches).
    """
    uniq_bio = np.unique(bio_labels)
    uniq_batch = np.unique(batch_labels)
    if len(uniq_bio) < 2:
        return float("nan"), float("nan")

    n_ct = len(uniq_bio)
    try:
        asw_bio = silhouette_score(X, bio_labels, metric="euclidean")
        n_clusters = min(n_ct, len(X))
        if n_clusters >= len(X):
            n_clusters = max(1, len(X) - 1)
        n_init = 10 if len(X) > 500 else "auto"
        km = KMeans(
            n_clusters=n_clusters,
            random_state=42,
            n_init=n_init,
        )
        pred = km.fit_predict(X)
        ari = adjusted_rand_score(bio_labels, pred)
        nmi = normalized_mutual_info_score(bio_labels, pred)
        avg_bio = float(np.mean([ari, nmi, asw_bio]))
    except ValueError:
        return float("nan"), float("nan")

    if len(uniq_batch) < 2:
        return avg_bio, float("nan")

    try:
        asw_batch = silhouette_score(X, batch_labels, metric="euclidean")
        batch_mix = float((1.0 - asw_batch) / 2.0)
        knn_mix = _knn_batch_mixing(X, batch_labels, batch_k_neighbors)
        avg_batch = float(np.mean([batch_mix, knn_mix]))
    except ValueError:
        return avg_bio, float("nan")
    return avg_bio, avg_batch


class NicheQuery:
    def __init__(self, db: Database, niche: Niche, k: int):
        assert (
            0 < k <= 10
        ), "Invalid subgraph parameter, should be between 1 and 10 (included)"
        self.db = db
        self.niche = niche
        self.k = k

    @staticmethod
    def get_parcellation_mask(sample: Sample, parcellation_index: int) -> np.ndarray:
        """
        1 if the cell lies in the target structure or under it in the CCF tree:
        ``parcellation_index`` equals the cell's index, or appears in
        ``cell.parcellation_info['parent_ids']`` (ancestor path, includes self when present).
        """
        return np.array(
            [
                1 if cell_matches_parcellation_or_ancestor(c, parcellation_index) else 0
                for c in sample.cells
            ],
            dtype=int,
        )

    """
    Generate niche embedding and parcellation mask for all samples.
    """

    def generate_niche_features_and_parcellation_mask(
        self, samples: List[Sample], overwrite=False
    ):
        def _row_normalized_adjacency(edge_index: torch.Tensor, num_nodes: int, device):
            """
            edge_index: [2, E], where edge_index[0] = source(row), edge_index[1] = target(col)
            Returns sparse COO tensor A_norm of shape [N, N] with row-normalized weights.
            """
            row = edge_index[0]
            col = edge_index[1]
            val = torch.ones(row.numel(), device=device, dtype=torch.float32)

            # Row degree: deg[i] = number of outgoing edges from i
            deg = torch.bincount(row, minlength=num_nodes).to(torch.float32)
            deg_inv = 1.0 / torch.clamp(deg, min=1.0)

            # Row-normalize: each edge from row i gets weight 1/deg[i]
            val = val * deg_inv[row]

            A_norm = torch.sparse_coo_tensor(
                indices=torch.stack([row, col], dim=0),
                values=val,
                size=(num_nodes, num_nodes),
                device=device,
            ).coalesce()

            return A_norm

        for sample in samples:
            logger.info(
                f"[START] start computing niche features for sample {sample.id}"
            )
            if (
                not overwrite
                and sample.niche_features is not None
                and sample.target_parcellation_mask is not None
            ):
                logger.info(f"[SKIP] finish sample: {sample.id}")
                continue

            # (1) features -> torch tensor (N, F)
            X_np = np.asarray(
                [np.asarray(c.feature).squeeze() for c in sample.cells],
                dtype=np.float32,
            )
            if X_np.ndim != 2:
                X_np = X_np.reshape(X_np.shape[0], -1)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            X = torch.from_numpy(X_np).to(device=device, dtype=torch.float32)

            # (2) parcellation mask
            sample.target_parcellation_mask = self.get_parcellation_mask(
                sample, self.niche.parcellation_index
            )

            # (3) KNN graph in feature space
            edge_index = knn_graph(X, k=self.k, loop=True)  # shape [2, E]

            # (4) A_norm @ X  (row-normalized adjacency)
            N = X.size(0)
            A_norm = _row_normalized_adjacency(edge_index, num_nodes=N, device=device)
            niche_X = torch.sparse.mm(A_norm, X)  # (N, F)
            sample.niche_features = niche_X.detach().cpu().numpy()

            logger.info(f"[DONE] finish sample: {sample.id}")

    """
    Perform niche query task on a sample and return cosine similarity between query niche and each niches.
    """

    def niche_query_within_a_sample(self, sample: Sample) -> np.ndarray:
        cos_sim = cosine_similarity(
            self.niche.feature.reshape(1, -1), sample.niche_features
        ).squeeze()
        for i, cell in enumerate(sample.cells):
            cell.similarity = cos_sim[i]
        return cos_sim

    """
    Get the best threshold and compute F1 scores for niche matching.
    """

    def get_best_threshold(self, samples: List[Sample]) -> Tuple[float, float]:
        cos_sim_list = []
        parcellation_mask_list = []
        for sample in samples:
            cos_sim_list.append(self.niche_query_within_a_sample(sample))
            parcellation_mask_list.append(sample.target_parcellation_mask)

        cos_sim = np.concatenate(cos_sim_list, axis=0)
        parcellation_mask = np.concatenate(parcellation_mask_list, axis=0).astype(int)

        precision, recall, thresholds = precision_recall_curve(
            y_true=parcellation_mask, y_score=cos_sim, pos_label=1
        )
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        best_idx = np.argmax(f1)
        best_threshold = thresholds[best_idx]
        best_f1 = f1[best_idx]
        logger.info(f"Best threshold found on {len(samples)} samples: {best_threshold}")
        logger.info(f"Maximized F1 score: {best_f1}")
        return best_threshold, best_f1

    """
    Perform niche query task and return samples with pred niches.
    """

    def niche_query_with_classification_report(
        self,
        best_threshold: float,
        samples: List[Sample],
        *,
        result_txt_path: Optional[str] = None,
    ) -> List[Sample]:
        cos_sim_list = []
        parcellation_mask_list = []
        for sample in samples:
            cos_sim_list.append(self.niche_query_within_a_sample(sample))
            if sample.target_parcellation_mask is not None:
                parcellation_mask_list.append(sample.target_parcellation_mask)

        pred = np.concatenate(cos_sim_list, axis=0) >= best_threshold
        parcellation_mask = np.concatenate(parcellation_mask_list, axis=0).astype(int)
        cr = classification_report(
            y_true=parcellation_mask,
            y_pred=pred,
            target_names=["Others", "Target Parcellation"],
        )
        logger.info(f"Classification report with {len(samples)} samples:")
        logger.info(f"\n{cr}\n")
        _append_result_txt(
            result_txt_path,
            f"Classification report with {len(samples)} samples:\n\n{cr}\n",
        )

        return [
            samples[i]
            for i, cos_sim in enumerate(cos_sim_list)
            if cos_sim.any() > best_threshold
        ]

    def niche_query_with_benchmark_metrics_report(
        self,
        best_threshold: float,
        samples: List[Sample],
        *,
        # NCJS: spatial k on (x,y) to build per-cell parcellation histogram P_i (see _niche_composition_matrix).
        ncjs_k_spatial: int = 15,
        # NCJS: number of nearest neighbors in niche embedding space when comparing P_i vs P_j via Jensen–Shannon.
        ncjs_k_embedding: int = 10,
        # AvgBATCH: k for embedding kNN batch-mixing fraction (paired with (1−ASW_batch)/2 in _avg_bio_batch).
        batch_k_neighbors: int = 15,
        result_txt_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Like ``niche_query_with_classification_report``, but log and return retrieval /
        embedding benchmark metrics: AUPRC, PCC (Pearson r between cosine score and
        target mask), average NCJS (QueST-style composition JS on spatial niches vs
        embedding kNN), AvgBIO (mean of ARI, NMI, ASW w.r.t. parcellation), and
        AvgBATCH (mean of ``(1 - ASW_batch)/2`` and kNN batch-mixing; needs >=2 batches).

        Requires ``generate_niche_features_and_parcellation_mask`` to have been run
        so each sample has ``niche_features`` and ``target_parcellation_mask``.
        """
        cos_sim_list: List[np.ndarray] = []
        parcellation_mask_list: List[np.ndarray] = []
        for sample in samples:
            cos_sim_list.append(self.niche_query_within_a_sample(sample))
            if sample.target_parcellation_mask is not None:
                parcellation_mask_list.append(sample.target_parcellation_mask)

        y_score = np.concatenate(cos_sim_list, axis=0)
        y_true = np.concatenate(parcellation_mask_list, axis=0).astype(int)

        if len(np.unique(y_true)) >= 2:
            auprc = float(average_precision_score(y_true, y_score))
            pcc = float(np.corrcoef(y_true.astype(np.float64), y_score)[0, 1])
        else:
            auprc = float("nan")
            pcc = float("nan")

        for s in samples:
            if s.niche_features is None:
                raise ValueError(
                    f"Sample {s.id} has no niche_features; run "
                    "generate_niche_features_and_parcellation_mask first."
                )

        emb = np.vstack([s.niche_features for s in samples])
        max_pi = max(c.parcellation_index for s in samples for c in s.cells)
        n_bins = int(max_pi) + 1
        P = _niche_composition_matrix(samples, ncjs_k_spatial, n_bins)
        avg_ncjs = _average_ncjs_score(P, emb, ncjs_k_embedding)

        bio_labels = np.concatenate(
            [[c.parcellation_index for c in s.cells] for s in samples]
        )
        batch_labels = np.concatenate([[s.id] * len(s.cells) for s in samples])
        batch_enc = LabelEncoder().fit_transform(batch_labels)
        avg_bio, avg_batch = _avg_bio_batch(
            emb, bio_labels, batch_enc, batch_k_neighbors
        )

        metrics: Dict[str, float] = {
            "auprc": auprc,
            "pcc": pcc,
            "avg_ncjs": avg_ncjs,
            "avg_bio": avg_bio,
            "avg_batch": avg_batch,
        }
        bench_msg = (
            f"Benchmark metrics ({len(samples)} samples, threshold={best_threshold}): "
            f"AUPRC={auprc:.4f}, PCC={pcc:.4f}, avg_NCJS={avg_ncjs:.4f}, "
            f"AvgBIO={avg_bio:.4f}, AvgBATCH={avg_batch:.4f}"
        )
        logger.info(bench_msg)
        _append_result_txt(result_txt_path, bench_msg + "\n")
        return metrics

    """
    Perform niche query task, find best threshold, classification report, and benchmark metrics.
    step 1: compute the best threshold with 30% samples with the parcellation section
    step 2: use the best threshold to run niche query (classification + quantitative metrics)
    step 3: compute metrics on held-out test samples (and non-target samples)
    """

    def niche_query_quantitive_metrics(
        self,
        search_samples: List[Sample],
        *,
        # NCJS: spatial k on (x,y) for parcellation histogram P_i (passed to benchmark report).
        ncjs_k_spatial: int = 15,
        # NCJS: embedding-space kNN count for JS(P_i,P_j) (passed to benchmark report).
        ncjs_k_embedding: int = 10,
        # AvgBATCH: k for kNN batch-mixing (passed to benchmark report).
        batch_k_neighbors: int = 15,
        result_txt_path: Optional[str] = None,
    ) -> Dict[str, float]:
        self.generate_niche_features_and_parcellation_mask(search_samples)
        target_samples_ = [
            s for s in search_samples if s.target_parcellation_mask.sum() > 0
        ]
        non_target_samples = [
            s for s in search_samples if s.target_parcellation_mask.sum() == 0
        ]
        val_samples, test_samples = train_test_split(
            target_samples_, test_size=0.7, random_state=42, shuffle=True
        )
        best_threshold, _ = self.get_best_threshold(val_samples)

        all_test_samples = list(test_samples) + list(non_target_samples)
        np.random.shuffle(all_test_samples)
        self.niche_query_with_classification_report(
            best_threshold,
            all_test_samples,
            result_txt_path=result_txt_path,
        )
        return self.niche_query_with_benchmark_metrics_report(
            best_threshold,
            all_test_samples,
            ncjs_k_spatial=ncjs_k_spatial,
            ncjs_k_embedding=ncjs_k_embedding,
            batch_k_neighbors=batch_k_neighbors,
            result_txt_path=result_txt_path,
        )

    """
    Perform a niche query task and visualize results using cosine similarity on given samples.
    """

    def niche_query_visualization(
        self,
        search_samples: List[Sample],
        show_target_niches: bool = False,
        spot_size: float = 0.02,
    ) -> None:
        self.generate_niche_features_and_parcellation_mask(search_samples)
        for sample in search_samples:
            self.niche_query_within_a_sample(sample)
            if "niche_query_result" not in sample.adata.obs:
                sample.adata.obs["niche_query_result"] = np.nan
            target_cells = []
            for cell in sample.cells:
                sample.adata.obs.loc[cell.id, "niche_query_result"] = float(
                    cell.similarity
                )
                if self.db.cell_matches_parcellation_or_ancestor(
                    cell, self.niche.parcellation_index
                ):
                    target_cells.append(cell)
            sample.adata.obs["target_niches"] = sample.adata.obs_names.isin(
                [c.id for c in target_cells]
            )

        fig, axes = plt.subplots(ncols=6, nrows=2, figsize=(16, 6))
        axes = axes.flatten()
        for i, sample in enumerate(search_samples):
            sc.pl.spatial(
                sample.adata,
                color="niche_query_result",
                ax=axes[i],
                show=False,
                spot_size=spot_size,
                title=f"Niche query\n{sample.id}",
            )

        if show_target_niches:
            fig, axes = plt.subplots(ncols=6, nrows=2, figsize=(16, 6))
            axes = axes.flatten()
            for i, sample in enumerate(search_samples):
                sc.pl.spatial(
                    sample.adata,
                    color="target_niches",
                    ax=axes[i],
                    show=False,
                    spot_size=spot_size,
                    title=f"Target niches\n{sample.id}",
                )
            fig.suptitle("Target niches", fontsize=18)
            plt.tight_layout()
