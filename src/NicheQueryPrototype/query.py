from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr
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
from src.NicheQueryPrototype.rm_ideal import RmIdeal


def _spatial_viz_figure(
    n_panels: int,
    *,
    max_cols: int = 6,
    panel_width_in: float = 2.75,
    panel_height_in: float = 2.75,
) -> Tuple[Any, np.ndarray]:
    """
    One matplotlib figure with a flat ``axes`` array for ``n_panels`` spatial plots.
    Unused axes are turned off. Figure size grows with ``nrows`` / ``ncols`` so the
    overall canvas stays roughly rectangular per panel (not a fixed 6×2 grid).
    """
    n_panels = max(1, int(n_panels))
    ncols = min(max_cols, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(panel_width_in * ncols, panel_height_in * nrows),
        squeeze=False,
    )
    axes_flat = np.asarray(axes).ravel()
    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].set_axis_off()
    return fig, axes_flat


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
    def get_parcellation_mask(
        sample: Sample, parcellation_index: Union[int, List[int]]
    ) -> np.ndarray:
        """
        1 if the cell lies in the target structure or under it in the CCF tree:
        ``parcellation_index`` (single int or list of ints) equals the cell's index, or appears in
        ``cell.parcellation_info['parent_ids']`` (ancestor path, includes self when present).
        """
        if isinstance(parcellation_index, int):
            targets = [parcellation_index]
        else:
            targets = list(parcellation_index)
        if len(targets) == 0:
            raise ValueError("parcellation_index cannot be an empty list")

        return np.array(
            [
                (
                    1
                    if any(
                        cell_matches_parcellation_or_ancestor(c, pidx)
                        for pidx in targets
                    )
                    else 0
                )
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

        niche_dim = int(np.asarray(self.niche.feature).reshape(-1).shape[0])

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
            if X_np.shape[1] != niche_dim:
                raise ValueError(
                    f"Sample {sample.id!r}: cell feature width {X_np.shape[1]} does not "
                    f"match query niche feature width {niche_dim}."
                )

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

    def compute_rm_ideal_score(
        self,
        samples: List[Sample],
        rm_ideal_output_key: Optional[str] = "rm_ideal_score",
        rm_ideal_post_transform: str = "linear",
        rm_ideal_temperature: float = 0.1,
        overwrite: bool = False,
    ) -> None:
        """
        Compute RM-Ideal score for each input sample.

        By default, scores are saved to ``sample.rm_ideal_score``.
        If ``rm_ideal_output_key`` is provided, scores are also written to
        ``sample.adata.obs[rm_ideal_output_key]``.
        If ``overwrite`` is False, existing ``sample.rm_ideal_score`` is reused
        when present and shape-aligned.
        """
        if len(self.niche.cells) == 0:
            raise ValueError(
                "self.niche.cells is empty; cannot compute RM-Ideal score."
            )

        rm_ideal = RmIdeal(
            wl_iters=self.k,
            post_transform=rm_ideal_post_transform,
            temperature=rm_ideal_temperature,
        )
        query_niche_cells = self.niche.cells
        n_samples = len(samples)
        t0_all = time.perf_counter()
        logger.info(
            "[START] RM-Ideal scoring for %d sample(s); wl_iters=%d; output_key=%r; transform=%s; temperature=%.3f; overwrite=%s",
            n_samples,
            self.k,
            rm_ideal_output_key,
            rm_ideal_post_transform,
            rm_ideal_temperature,
            overwrite,
        )

        for i, sample in enumerate(samples, start=1):
            t0_sample = time.perf_counter()
            logger.info(
                "[PROGRESS] RM-Ideal sample %d/%d: %s (n_cells=%d)",
                i,
                n_samples,
                sample.id,
                len(sample.cells),
            )
            used_cached = False
            if not overwrite and sample.rm_ideal_score is not None:
                cached_scores = np.asarray(sample.rm_ideal_score, dtype=float).reshape(
                    -1
                )
                if len(cached_scores) == len(sample.cells):
                    scores = cached_scores
                    used_cached = True
                else:
                    logger.warning(
                        "Sample %r has cached rm_ideal_score with mismatched length (%d vs %d); recomputing.",
                        sample.id,
                        len(cached_scores),
                        len(sample.cells),
                    )
                    scores = rm_ideal.score_slice(
                        query_niche_cells=query_niche_cells,
                        samples=sample,
                    )
            else:
                scores = rm_ideal.score_slice(
                    query_niche_cells=query_niche_cells,
                    samples=sample,
                )

            if len(scores) != len(sample.cells):
                raise ValueError(
                    f"Sample {sample.id!r}: RM-Ideal scores length ({len(scores)}) "
                    f"does not match number of cells ({len(sample.cells)})."
                )

            if overwrite or sample.rm_ideal_score is None or not used_cached:
                sample.rm_ideal_score = scores

            if rm_ideal_output_key is not None:
                if sample.adata is None:
                    raise ValueError(
                        f"Sample {sample.id!r} has no AnnData; run sample.construct_adata(...) first."
                    )
                sample.adata.obs[rm_ideal_output_key] = np.nan
                for cell_idx, cell in enumerate(sample.cells):
                    sample.adata.obs.loc[cell.id, rm_ideal_output_key] = float(
                        scores[cell_idx]
                    )
            elapsed_sample = time.perf_counter() - t0_sample
            logger.info(
                "[DONE] RM-Ideal sample %d/%d: %s finished in %.2fs (%s)",
                i,
                n_samples,
                sample.id,
                elapsed_sample,
                "cached" if used_cached else "computed",
            )
        elapsed_all = time.perf_counter() - t0_all
        logger.info(
            "[DONE] RM-Ideal scoring finished: %d sample(s) in %.2fs",
            n_samples,
            elapsed_all,
        )

    """
    Perform niche query task on a sample and return cosine similarity between query niche and each niches.
    """

    def niche_query_within_a_sample(self, sample: Sample) -> np.ndarray:
        q = int(np.asarray(self.niche.feature).reshape(-1).shape[0])
        sf = sample.niche_features.shape[1]
        if q != sf:
            raise ValueError(
                f"Query niche feature width ({q}) != sample niche_features width ({sf}) "
                f"for sample {sample.id!r}. Rebuild the database with aligned features, "
                f"or run generate_niche_features_and_parcellation_mask with overwrite=True."
            )
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
        samples: List[Sample],
        *,
        # NCJS: spatial k on (x,y) to build per-cell parcellation histogram P_i (see _niche_composition_matrix).
        ncjs_k_spatial: int = 15,
        # NCJS: number of nearest neighbors in niche embedding space when comparing P_i vs P_j via Jensen–Shannon.
        ncjs_k_embedding: int = 10,
        # AvgBATCH: k for embedding kNN batch-mixing fraction (paired with (1−ASW_batch)/2 in _avg_bio_batch).
        batch_k_neighbors: int = 15,
        result_txt_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Like ``niche_query_with_classification_report``, but log and return retrieval /
        embedding benchmark metrics: AUPRC, PCC (Pearson r between cosine score and
        y_true label), average NCJS (QueST-style composition JS on spatial niches vs
        embedding kNN), AvgBIO (mean of ARI, NMI, ASW w.r.t. parcellation), and
        AvgBATCH (mean of ``(1 - ASW_batch)/2`` and kNN batch-mixing; needs >=2 batches).

        Returns a dict with ``total`` (pooled over all cells / samples) and
        ``per_sample`` (same keys per sample; batch metrics are only defined on the
        pooled embedding, so per-sample ``avg_batch`` is typically NaN). Per-sample
        NCJS uses the same global ``n_bins`` as the pooled composition matrix.

        Requires ``generate_niche_features_and_parcellation_mask`` to have been run
        so each sample has ``niche_features``. For each sample, ``y_true`` uses
        ``sample.rm_ideal_score`` when available; otherwise falls back to
        ``sample.target_parcellation_mask``.
        """
        score_list: List[np.ndarray] = []
        y_true_list: List[np.ndarray] = []
        for sample in samples:
            # Niche-query score logic remains unchanged (cosine similarity).
            score_list.append(self.niche_query_within_a_sample(sample))

            if sample.rm_ideal_score is not None:
                yt = np.asarray(sample.rm_ideal_score, dtype=float).reshape(-1)
            elif sample.target_parcellation_mask is not None:
                yt = np.asarray(sample.target_parcellation_mask, dtype=float).reshape(
                    -1
                )
            else:
                raise ValueError(
                    f"Sample {sample.id!r} has neither rm_ideal_score nor "
                    "target_parcellation_mask; cannot build y_true."
                )
            if len(yt) != len(sample.cells):
                raise ValueError(
                    f"Sample {sample.id!r}: y_true length ({len(yt)}) "
                    f"does not match number of cells ({len(sample.cells)})."
                )
            y_true_list.append(yt)

        y_score = np.concatenate(score_list, axis=0)
        y_true = np.concatenate(y_true_list, axis=0)

        uniq_y = np.unique(y_true)
        is_binary = len(uniq_y) <= 2 and np.all(np.isin(uniq_y, [0.0, 1.0]))
        if is_binary and len(uniq_y) >= 2:
            auprc = float(average_precision_score(y_true.astype(int), y_score))
            if (
                y_true.size >= 2
                and np.std(y_true.astype(np.float64)) > 0.0
                and np.std(y_score.astype(np.float64)) > 0.0
            ):
                pcc = float(np.corrcoef(y_true.astype(np.float64), y_score)[0, 1])
            else:
                pcc = float("nan")
            spearman = float("nan")
            mae = float("nan")
            rmse = float("nan")
        else:
            auprc = float("nan")
            pcc = float("nan")
            if (
                y_true.size >= 2
                and np.std(y_true.astype(np.float64)) > 0.0
                and np.std(y_score.astype(np.float64)) > 0.0
            ):
                spearman = float(spearmanr(y_true, y_score).statistic)
            else:
                spearman = float("nan")
            mae = float(np.mean(np.abs(y_true.astype(np.float64) - y_score)))
            rmse = float(
                np.sqrt(np.mean(np.square(y_true.astype(np.float64) - y_score)))
            )

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

        total: Dict[str, float] = {
            "auprc": auprc,
            "pcc": pcc,
            "spearman": spearman,
            "mae": mae,
            "rmse": rmse,
            "avg_ncjs": avg_ncjs,
            "avg_bio": avg_bio,
            "avg_batch": avg_batch,
        }

        per_sample: List[Dict[str, Any]] = []
        for i, s in enumerate(samples):
            y_score_s = score_list[i]
            yt = y_true_list[i]
            uniq_yt = np.unique(yt)
            is_binary_s = len(uniq_yt) <= 2 and np.all(np.isin(uniq_yt, [0.0, 1.0]))
            if is_binary_s and len(uniq_yt) >= 2:
                auprc_s = float(average_precision_score(yt.astype(int), y_score_s))
                if (
                    yt.size >= 2
                    and np.std(yt.astype(np.float64)) > 0.0
                    and np.std(y_score_s.astype(np.float64)) > 0.0
                ):
                    pcc_s = float(np.corrcoef(yt.astype(np.float64), y_score_s)[0, 1])
                else:
                    pcc_s = float("nan")
                spearman_s = float("nan")
                mae_s = float("nan")
                rmse_s = float("nan")
            else:
                auprc_s = float("nan")
                pcc_s = float("nan")
                if (
                    yt.size >= 2
                    and np.std(yt.astype(np.float64)) > 0.0
                    and np.std(y_score_s.astype(np.float64)) > 0.0
                ):
                    spearman_s = float(spearmanr(yt, y_score_s).statistic)
                else:
                    spearman_s = float("nan")
                mae_s = float(np.mean(np.abs(yt.astype(np.float64) - y_score_s)))
                rmse_s = float(
                    np.sqrt(np.mean(np.square(yt.astype(np.float64) - y_score_s)))
                )

            emb_s = s.niche_features
            assert emb_s is not None
            P_s = _niche_composition_matrix([s], ncjs_k_spatial, n_bins)
            avg_ncjs_s = _average_ncjs_score(P_s, emb_s, ncjs_k_embedding)

            bio_s = np.array([c.parcellation_index for c in s.cells], dtype=int)
            batch_s = np.zeros(len(s.cells), dtype=int)
            avg_bio_s, avg_batch_s = _avg_bio_batch(
                emb_s, bio_s, batch_s, batch_k_neighbors
            )

            per_sample.append(
                {
                    "sample_id": s.id,
                    "auprc": auprc_s,
                    "pcc": pcc_s,
                    "spearman": spearman_s,
                    "mae": mae_s,
                    "rmse": rmse_s,
                    "avg_ncjs": avg_ncjs_s,
                    "avg_bio": avg_bio_s,
                    "avg_batch": avg_batch_s,
                }
            )

        out: Dict[str, Any] = {"total": total, "per_sample": per_sample}

        bench_msg = (
            f"Benchmark metrics ({len(samples)} samples) [total]: "
            f"AUPRC={auprc:.4f}, PCC={pcc:.4f}, Spearman={spearman:.4f}, "
            f"MAE={mae:.4f}, RMSE={rmse:.4f}, avg_NCJS={avg_ncjs:.4f}, "
            f"AvgBIO={avg_bio:.4f}, AvgBATCH={avg_batch:.4f}"
        )
        logger.info(bench_msg)
        _append_result_txt(result_txt_path, bench_msg + "\n")
        for row in per_sample:
            line = (
                f"  [{row['sample_id']}] AUPRC={row['auprc']:.4f}, PCC={row['pcc']:.4f}, "
                f"Spearman={row['spearman']:.4f}, MAE={row['mae']:.4f}, RMSE={row['rmse']:.4f}, "
                f"avg_NCJS={row['avg_ncjs']:.4f}, AvgBIO={row['avg_bio']:.4f}, "
                f"AvgBATCH={row['avg_batch']:.4f}"
            )
            logger.info(line)
            _append_result_txt(result_txt_path, line + "\n")
        return out

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
    ) -> List[Dict[str, Any]]:
        self.generate_niche_features_and_parcellation_mask(search_samples)
        # target_samples_ = [
        #     s for s in search_samples if s.target_parcellation_mask.sum() > 0
        # ]
        # non_target_samples = [
        #     s for s in search_samples if s.target_parcellation_mask.sum() == 0
        # ]
        # val_samples, test_samples = train_test_split(
        #     target_samples_, test_size=0.7, random_state=42, shuffle=True
        # )
        # best_threshold, _ = self.get_best_threshold(val_samples)

        # all_test_samples = list(test_samples) + list(non_target_samples)
        # np.random.shuffle(all_test_samples)
        # self.niche_query_with_classification_report(
        #     best_threshold,
        #     all_test_samples,
        #     result_txt_path=result_txt_path,
        # )
        out = self.niche_query_with_benchmark_metrics_report(
            search_samples,
            ncjs_k_spatial=ncjs_k_spatial,
            ncjs_k_embedding=ncjs_k_embedding,
            batch_k_neighbors=batch_k_neighbors,
            result_txt_path=result_txt_path,
        )
        return list(out["per_sample"])

    """
    Perform a niche query task and visualize results.
    """

    def niche_query_visualization(
        self,
        search_samples: List[Sample],
        show_target_niches: bool = False,
        spot_size: float = 0.02,
        *,
        viz_max_cols: int = 6,
        viz_panel_size_in: Tuple[float, float] = (2.75, 2.75),
    ) -> None:
        self.generate_niche_features_and_parcellation_mask(search_samples)

        for sample in search_samples:
            self.niche_query_within_a_sample(sample)
            if "niche_query_result" not in sample.adata.obs:
                sample.adata.obs["niche_query_result"] = np.nan
            for cell in sample.cells:
                sample.adata.obs.loc[cell.id, "niche_query_result"] = float(
                    cell.similarity
                )

            # Keep niche_query_result unchanged; only switch target_niches source when requested.
            if show_target_niches and sample.rm_ideal_score is not None:
                rm_score = np.asarray(sample.rm_ideal_score, dtype=float).reshape(-1)
                if len(rm_score) != len(sample.cells):
                    raise ValueError(
                        f"Sample {sample.id!r}: rm_ideal_score length ({len(rm_score)}) "
                        f"does not match number of cells ({len(sample.cells)})."
                    )
                sample.adata.obs["target_niches"] = np.nan
                for i, cell in enumerate(sample.cells):
                    sample.adata.obs.loc[cell.id, "target_niches"] = float(rm_score[i])
                continue

            target_mask = self.get_parcellation_mask(
                sample, self.niche.parcellation_index
            )
            target_cells = [
                cell for idx, cell in enumerate(sample.cells) if target_mask[idx] == 1
            ]
            sample.adata.obs["target_niches"] = sample.adata.obs_names.isin(
                [c.id for c in target_cells]
            )

        n = len(search_samples)
        pw, ph = viz_panel_size_in
        fig, axes = _spatial_viz_figure(
            n,
            max_cols=viz_max_cols,
            panel_width_in=pw,
            panel_height_in=ph,
        )
        for i, sample in enumerate(search_samples):
            sc.pl.spatial(
                sample.adata,
                color="niche_query_result",
                ax=axes[i],
                show=False,
                spot_size=spot_size,
                title=f"Niche query\n{sample.id}",
            )
        fig.suptitle("Niche query", fontsize=18)
        plt.tight_layout()

        if show_target_niches:
            fig, axes = _spatial_viz_figure(
                n,
                max_cols=viz_max_cols,
                panel_width_in=pw,
                panel_height_in=ph,
            )
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
