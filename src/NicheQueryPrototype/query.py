from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr
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


def _as_1d_float64(a: np.ndarray) -> np.ndarray:
    """1-D float64 vector for metrics / correlation (avoids (N,1) vs (N,) corrcoef issues)."""
    return np.asarray(a, dtype=np.float64).reshape(-1)


def _pearson_r(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r between two 1-D score vectors; ``nan`` if undefined."""
    a = _as_1d_float64(x)
    b = _as_1d_float64(y)
    if a.size != b.size or a.size < 2:
        return float("nan")
    if np.std(a) <= 0.0 or np.std(b) <= 0.0:
        return float("nan")
    r, _ = pearsonr(a, b)
    return float(r)


def _k_for_top_fraction(n: int, frac: float) -> int:
    """Number of top-ranked cells for ``frac`` of corpus size (at least 1, at most ``n``)."""
    if n <= 0:
        return 0
    return max(1, min(n, int(np.ceil(float(frac) * n))))


def _graded_relevance_minmax(rel: np.ndarray) -> np.ndarray:
    """Map relevance to [0, 1] for stable ``2**rel - 1`` gains; constant vector → all zeros."""
    r = _as_1d_float64(rel)
    lo, hi = float(np.min(r)), float(np.max(r))
    span = hi - lo
    if span < 1e-12:
        return np.zeros_like(r)
    return (r - lo) / span


def _dcg_from_gains_at_prefix(gains_desc: np.ndarray) -> float:
    """
    DCG over the first ``len(gains_desc)`` positions (rank 1 = index 0):
    sum_i (2^{g_i} - 1) / log2(i + 2).
    """
    if gains_desc.size == 0:
        return 0.0
    i = np.arange(gains_desc.size, dtype=np.float64)
    return float(np.sum((np.power(2.0, gains_desc) - 1.0) / np.log2(i + 2.0)))


def _ndcg_at_fractions(
    pred_scores: np.ndarray, true_relevance: np.ndarray
) -> Dict[str, float]:
    """
    NDCG@K for K = ceil(1%·n), ceil(5%·n), ceil(10%·n) cells (niche_query ranking vs RM-Ideal).

    DCG@K uses cells ranked by ``pred_scores`` descending; IDCG@K uses ideal order by
    graded relevance from ``true_relevance`` (min–max normalized to [0, 1]).
    """
    pred = _as_1d_float64(pred_scores)
    rel = _as_1d_float64(true_relevance)
    n = pred.size
    out: Dict[str, float] = {}
    if n == 0:
        for key in ("ndcg_at_1pct", "ndcg_at_5pct", "ndcg_at_10pct"):
            out[key] = float("nan")
        return out

    rel_n = _graded_relevance_minmax(rel)
    ideal_desc = np.sort(rel_n)[::-1]
    order = np.argsort(-pred, kind="mergesort")
    rel_by_pred = rel_n[order]

    for frac, key in (
        (0.01, "ndcg_at_1pct"),
        (0.05, "ndcg_at_5pct"),
        (0.10, "ndcg_at_10pct"),
    ):
        k = _k_for_top_fraction(n, frac)
        ideal_gains = ideal_desc[:k]
        idcg = _dcg_from_gains_at_prefix(ideal_gains)
        dcg = _dcg_from_gains_at_prefix(rel_by_pred[:k])
        if idcg <= 1e-12:
            out[key] = 1.0 if dcg <= 1e-12 else float("nan")
        else:
            out[key] = float(dcg / idcg)
    return out


def _rm_top_pct_fraction(top_pct: float) -> float:
    """
    Map ``top_pct`` to a fraction in (0, 1].

    - ``>= 1`` → percent (e.g. ``1`` → 1%, ``5`` → 5%, ``100`` → 100%).
    - ``< 1`` → already a fraction (e.g. ``0.01`` → 1%).
    """
    if not np.isfinite(top_pct) or top_pct <= 0:
        raise ValueError(f"top_pct must be a positive finite float, got {top_pct!r}")
    if top_pct >= 1.0:
        return float(top_pct) / 100.0
    return float(top_pct)


def _top_fraction_mask_by_score(scores: np.ndarray, top_pct: float) -> np.ndarray:
    """True for cells whose score is in the top ``top_pct`` fraction (higher is better)."""
    s = _as_1d_float64(scores)
    n = s.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    frac = _rm_top_pct_fraction(top_pct)
    k = _k_for_top_fraction(n, frac)
    order = np.argsort(-s, kind="mergesort")
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask


def _top_k_indices_by_score(scores: np.ndarray, top_pct: float) -> np.ndarray:
    """Cell indices of the top ``top_pct`` fraction by score (higher is better)."""
    s = _as_1d_float64(scores)
    n = s.size
    if n == 0:
        return np.zeros(0, dtype=int)
    frac = _rm_top_pct_fraction(top_pct)
    k = _k_for_top_fraction(n, frac)
    order = np.argsort(-s, kind="mergesort")
    return order[:k].astype(int)


def _overlap_at_fractions(
    pred_scores: np.ndarray, proxy_scores: np.ndarray
) -> Dict[str, float]:
    """
    Overlap@Q for Q = top 1%, 5%, 10% of cells (same K rule as NDCG).

    Overlap@Q = |TopK(niche_query) ∩ TopK(rm_ideal)| / K.
    """
    pred = _as_1d_float64(pred_scores)
    proxy = _as_1d_float64(proxy_scores)
    if pred.size != proxy.size:
        raise ValueError(
            f"pred_scores and proxy_scores must have the same length, "
            f"got {pred.size} vs {proxy.size}"
        )
    n = pred.size
    out: Dict[str, float] = {}
    if n == 0:
        for key in ("overlap_at_1pct", "overlap_at_5pct", "overlap_at_10pct"):
            out[key] = float("nan")
        return out

    for frac, key in (
        (0.01, "overlap_at_1pct"),
        (0.05, "overlap_at_5pct"),
        (0.10, "overlap_at_10pct"),
    ):
        top_pct = frac * 100.0
        a_k = _top_k_indices_by_score(pred, top_pct)
        b_k = _top_k_indices_by_score(proxy, top_pct)
        k = int(a_k.size)
        if k == 0:
            out[key] = float("nan")
        else:
            inter = len(set(a_k.tolist()) & set(b_k.tolist()))
            out[key] = float(inter) / float(k)
    return out


def _normalize_rm_target_top_pct_list(
    v: Optional[Union[float, int, List[float]]],
) -> List[float]:
    """Single float or list → deduplicated finite top-% values (e.g. ``[1, 5, 10]``)."""
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        raw = list(v)
    else:
        raw = [v]
    out: List[float] = []
    seen: set[float] = set()
    for x in raw:
        xf = float(x)
        if not np.isfinite(xf):
            continue
        key = round(xf, 12)
        if key not in seen:
            seen.add(key)
            out.append(xf)
    return out


def _target_niches_obs_key_for_pct(pct: float) -> str:
    """``adata.obs`` column for top-``pct``% RM-Ideal target highlight."""
    label = f"{pct:g}".replace(".", "p")
    return f"target_niches_top_{label}pct"


def _set_target_niches_top_pct_scores(
    sample: Sample, rm_score: np.ndarray, pct: float, obs_key: str
) -> None:
    """
    Top ``pct``% cells by RM-Ideal get clipped scores in [0, 1]; others NaN (plotted gray).
    """
    rm = _as_1d_float64(rm_score)
    rm_vis = np.clip(rm, 0.0, 1.0)
    top_mask = _top_fraction_mask_by_score(rm, pct)
    scores_out = np.full(rm.size, np.nan, dtype=np.float64)
    scores_out[top_mask] = rm_vis[top_mask]
    sample.adata.obs[obs_key] = (
        pd.Series(scores_out, index=[c.id for c in sample.cells])
        .reindex(sample.adata.obs_names)
        .to_numpy(dtype=np.float64)
    )


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


def _spatial_viz_figure_grid(
    nrows: int,
    ncols: int,
    *,
    panel_width_in: float = 2.75,
    panel_height_in: float = 2.75,
) -> Tuple[Any, np.ndarray]:
    """
    One figure with a 2D ``axes`` array (``nrows`` × ``ncols``) for spatial subplots.
    """
    nrows = max(1, int(nrows))
    ncols = max(1, int(ncols))
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(panel_width_in * ncols, panel_height_in * nrows),
        squeeze=False,
    )
    return fig, np.asarray(axes)


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

        Parameters
        ----------
        rm_ideal_post_transform
            Score post-processing passed to :class:`RmIdeal`. One of:

            - ``"linear"`` (default): min-max normalize raw scores to [0, 1].
            - ``"sigmoid"``: apply a sigmoid on the linear-normalized scores;
              use ``rm_ideal_temperature`` (must be > 0; smaller values yield
              stronger contrast).
            - ``"rank"``: replace scores with rank percentiles in [0, 1].

            Unknown values fall back to raw scores (see ``RmIdeal``).
        rm_ideal_temperature
            Temperature for sigmoid post-processing; ignored for other modes.
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
        )
        cos_sim = _as_1d_float64(np.asarray(cos_sim).squeeze())
        for i, cell in enumerate(sample.cells):
            cell.similarity = float(cos_sim[i])
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

        **AUPRC** is only defined when ``y_true`` is strict binary ``{0, 1}`` with both
        classes present (retrieval / classification style). Continuous
        ``rm_ideal_score`` yields ``auprc=nan`` by design; use
        ``target_parcellation_mask`` as ``y_true``, or binarize RM-Ideal, if you need
        a finite AUPRC.

        **NDCG@K** (``ndcg_at_1pct``, ``ndcg_at_5pct``, ``ndcg_at_10pct``): for each
        sample with ``rm_ideal_score``, ranks cells by niche-query cosine score and
        uses RM-Ideal as graded relevance (min–max normalized). ``K`` is
        ``max(1, ceil(frac * n_cells))`` for ``frac`` in ``{0.01, 0.05, 0.10}``.
        Pooled values concatenate all such samples. If ``rm_ideal_score`` is missing,
        per-sample NDCG is ``nan``.

        **Overlap@Q** (``overlap_at_1pct``, ``overlap_at_5pct``, ``overlap_at_10pct``):
        fraction of the model's top-K cells (by niche-query score) that also lie in the
        proxy top-K by ``rm_ideal_score``; ``K = max(1, ceil(Q·n_cells))`` for
        ``Q`` in ``{1%, 5%, 10%}``. Ranking-based, not a fixed score threshold.
        ``nan`` when ``rm_ideal_score`` is missing.

        **Pearson** (``pearson``): Pearson r between niche-query cosine similarity and
        ``rm_ideal_score`` (pooled and per sample). ``pcc`` uses the same ``y_true`` as
        above (RM-Ideal when present, else parcellation mask).
        """
        score_list: List[np.ndarray] = []
        y_true_list: List[np.ndarray] = []
        rm_ideal_for_ndcg_list: List[Optional[np.ndarray]] = []
        for sample in samples:
            # Niche-query score logic remains unchanged (cosine similarity).
            score_list.append(self.niche_query_within_a_sample(sample))

            if sample.rm_ideal_score is not None:
                yt = _as_1d_float64(np.asarray(sample.rm_ideal_score, dtype=float))
                rm_ideal_for_ndcg_list.append(yt)
            elif sample.target_parcellation_mask is not None:
                yt = _as_1d_float64(
                    np.asarray(sample.target_parcellation_mask, dtype=float)
                )
                rm_ideal_for_ndcg_list.append(None)
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

        y_score = _as_1d_float64(np.concatenate(score_list, axis=0))
        y_true = _as_1d_float64(np.concatenate(y_true_list, axis=0))

        uniq_y = np.unique(y_true)
        is_binary = len(uniq_y) <= 2 and np.all(np.isin(uniq_y, [0.0, 1.0]))
        if is_binary and len(uniq_y) >= 2:
            auprc = float(average_precision_score(y_true.astype(int), y_score))
            pcc = _pearson_r(y_true, y_score)
            spearman = float("nan")
            mae = float("nan")
            rmse = float("nan")
        else:
            auprc = float("nan")
            logger.info(
                "AUPRC is nan: y_true is not strict binary {{0, 1}} with both classes "
                "present (n_unique=%d). AUPRC needs binary labels, e.g. "
                "target_parcellation_mask; continuous rm_ideal_score will not produce AUPRC.",
                len(uniq_y),
            )
            pcc = _pearson_r(y_true, y_score)
            if y_true.size >= 2 and np.std(y_true) > 0.0 and np.std(y_score) > 0.0:
                spearman = float(spearmanr(y_true, y_score).statistic)
            else:
                spearman = float("nan")
            mae = float(np.mean(np.abs(y_true - y_score)))
            rmse = float(np.sqrt(np.mean(np.square(y_true - y_score))))

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

        pred_ndcg_parts: List[np.ndarray] = []
        rel_ndcg_parts: List[np.ndarray] = []
        for i, s in enumerate(samples):
            rv = rm_ideal_for_ndcg_list[i]
            if rv is not None:
                pred_ndcg_parts.append(score_list[i])
                rel_ndcg_parts.append(rv)
        if pred_ndcg_parts:
            ndcg_pooled = _ndcg_at_fractions(
                _as_1d_float64(np.concatenate(pred_ndcg_parts, axis=0)),
                _as_1d_float64(np.concatenate(rel_ndcg_parts, axis=0)),
            )
        else:
            ndcg_pooled = {
                "ndcg_at_1pct": float("nan"),
                "ndcg_at_5pct": float("nan"),
                "ndcg_at_10pct": float("nan"),
            }

        if pred_ndcg_parts:
            overlap_pooled = _overlap_at_fractions(
                _as_1d_float64(np.concatenate(pred_ndcg_parts, axis=0)),
                _as_1d_float64(np.concatenate(rel_ndcg_parts, axis=0)),
            )
        else:
            overlap_pooled = {
                "overlap_at_1pct": float("nan"),
                "overlap_at_5pct": float("nan"),
                "overlap_at_10pct": float("nan"),
            }

        if pred_ndcg_parts:
            pearson_pooled = _pearson_r(
                _as_1d_float64(np.concatenate(pred_ndcg_parts, axis=0)),
                _as_1d_float64(np.concatenate(rel_ndcg_parts, axis=0)),
            )
        else:
            pearson_pooled = float("nan")

        total: Dict[str, float] = {
            "auprc": auprc,
            "pcc": pcc,
            "pearson": pearson_pooled,
            "spearman": spearman,
            "mae": mae,
            "rmse": rmse,
            "avg_ncjs": avg_ncjs,
            "avg_bio": avg_bio,
            "avg_batch": avg_batch,
            "ndcg_at_1pct": ndcg_pooled["ndcg_at_1pct"],
            "ndcg_at_5pct": ndcg_pooled["ndcg_at_5pct"],
            "ndcg_at_10pct": ndcg_pooled["ndcg_at_10pct"],
            "overlap_at_1pct": overlap_pooled["overlap_at_1pct"],
            "overlap_at_5pct": overlap_pooled["overlap_at_5pct"],
            "overlap_at_10pct": overlap_pooled["overlap_at_10pct"],
        }

        per_sample: List[Dict[str, Any]] = []
        for i, s in enumerate(samples):
            y_score_s = _as_1d_float64(score_list[i])
            yt = _as_1d_float64(y_true_list[i])
            uniq_yt = np.unique(yt)
            is_binary_s = len(uniq_yt) <= 2 and np.all(np.isin(uniq_yt, [0.0, 1.0]))
            if is_binary_s and len(uniq_yt) >= 2:
                auprc_s = float(average_precision_score(yt.astype(int), y_score_s))
                pcc_s = _pearson_r(yt, y_score_s)
                spearman_s = float("nan")
                mae_s = float("nan")
                rmse_s = float("nan")
            else:
                auprc_s = float("nan")
                pcc_s = _pearson_r(yt, y_score_s)
                if yt.size >= 2 and np.std(yt) > 0.0 and np.std(y_score_s) > 0.0:
                    spearman_s = float(spearmanr(yt, y_score_s).statistic)
                else:
                    spearman_s = float("nan")
                mae_s = float(np.mean(np.abs(yt - y_score_s)))
                rmse_s = float(np.sqrt(np.mean(np.square(yt - y_score_s))))

            emb_s = s.niche_features
            assert emb_s is not None
            P_s = _niche_composition_matrix([s], ncjs_k_spatial, n_bins)
            avg_ncjs_s = _average_ncjs_score(P_s, emb_s, ncjs_k_embedding)

            bio_s = np.array([c.parcellation_index for c in s.cells], dtype=int)
            batch_s = np.zeros(len(s.cells), dtype=int)
            avg_bio_s, avg_batch_s = _avg_bio_batch(
                emb_s, bio_s, batch_s, batch_k_neighbors
            )

            rmv = rm_ideal_for_ndcg_list[i]
            if rmv is not None:
                ndcg_s = _ndcg_at_fractions(y_score_s, rmv)
                overlap_s = _overlap_at_fractions(y_score_s, rmv)
                pearson_s = _pearson_r(y_score_s, rmv)
            else:
                ndcg_s = {
                    "ndcg_at_1pct": float("nan"),
                    "ndcg_at_5pct": float("nan"),
                    "ndcg_at_10pct": float("nan"),
                }
                overlap_s = {
                    "overlap_at_1pct": float("nan"),
                    "overlap_at_5pct": float("nan"),
                    "overlap_at_10pct": float("nan"),
                }
                pearson_s = float("nan")

            per_sample.append(
                {
                    "sample_id": s.id,
                    "auprc": auprc_s,
                    "pcc": pcc_s,
                    "pearson": pearson_s,
                    "spearman": spearman_s,
                    "mae": mae_s,
                    "rmse": rmse_s,
                    "avg_ncjs": avg_ncjs_s,
                    "avg_bio": avg_bio_s,
                    "avg_batch": avg_batch_s,
                    "ndcg_at_1pct": ndcg_s["ndcg_at_1pct"],
                    "ndcg_at_5pct": ndcg_s["ndcg_at_5pct"],
                    "ndcg_at_10pct": ndcg_s["ndcg_at_10pct"],
                    "overlap_at_1pct": overlap_s["overlap_at_1pct"],
                    "overlap_at_5pct": overlap_s["overlap_at_5pct"],
                    "overlap_at_10pct": overlap_s["overlap_at_10pct"],
                }
            )

        out: Dict[str, Any] = {"total": total, "per_sample": per_sample}

        bench_msg = (
            f"Benchmark metrics ({len(samples)} samples) [total]: "
            f"AUPRC={auprc:.4f}, PCC={pcc:.4f}, Pearson={pearson_pooled:.4f}, "
            f"Spearman={spearman:.4f}, MAE={mae:.4f}, RMSE={rmse:.4f}, "
            f"avg_NCJS={avg_ncjs:.4f}, AvgBIO={avg_bio:.4f}, AvgBATCH={avg_batch:.4f}"
        )
        pearson_msg = (
            f"Pearson r (niche_query vs rm_ideal, pooled): {pearson_pooled:.4f} "
            f"(nan if no rm_ideal_score)"
        )
        ndcg_msg = (
            f"NDCG@K (niche_query vs rm_ideal, pooled): "
            f"@1%={ndcg_pooled['ndcg_at_1pct']:.4f}, "
            f"@5%={ndcg_pooled['ndcg_at_5pct']:.4f}, "
            f"@10%={ndcg_pooled['ndcg_at_10pct']:.4f} "
            f"(K=ceil(frac·n_cells) per slice; nan if no rm_ideal_score)"
        )
        overlap_msg = (
            f"Overlap@Q (proxy top-K, pooled): "
            f"@1%={overlap_pooled['overlap_at_1pct']:.4f}, "
            f"@5%={overlap_pooled['overlap_at_5pct']:.4f}, "
            f"@10%={overlap_pooled['overlap_at_10pct']:.4f} "
            f"(K=ceil(Q·n_cells); |TopK(niche_query)∩TopK(rm_ideal)|/K; "
            f"nan if no rm_ideal_score)"
        )
        logger.info(bench_msg)
        logger.info(pearson_msg)
        logger.info(ndcg_msg)
        logger.info(overlap_msg)
        _append_result_txt(result_txt_path, bench_msg + "\n")
        _append_result_txt(result_txt_path, pearson_msg + "\n")
        _append_result_txt(result_txt_path, ndcg_msg + "\n")
        _append_result_txt(result_txt_path, overlap_msg + "\n")
        for row in per_sample:
            line = (
                f"  [{row['sample_id']}] AUPRC={row['auprc']:.4f}, PCC={row['pcc']:.4f}, "
                f"Pearson={row['pearson']:.4f}, Spearman={row['spearman']:.4f}, "
                f"MAE={row['mae']:.4f}, RMSE={row['rmse']:.4f}, "
                f"avg_NCJS={row['avg_ncjs']:.4f}, AvgBIO={row['avg_bio']:.4f}, "
                f"AvgBATCH={row['avg_batch']:.4f}, "
                f"NDCG@1%={row['ndcg_at_1pct']:.4f}, NDCG@5%={row['ndcg_at_5pct']:.4f}, "
                f"NDCG@10%={row['ndcg_at_10pct']:.4f}, "
                f"Overlap@1%={row['overlap_at_1pct']:.4f}, "
                f"Overlap@5%={row['overlap_at_5pct']:.4f}, "
                f"Overlap@10%={row['overlap_at_10pct']:.4f}"
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
        """
        Run ``generate_niche_features_and_parcellation_mask`` then benchmark metrics.

        Includes **Pearson** r (niche-query cosine vs ``rm_ideal_score``), **NDCG@K**,
        and **Overlap@Q** at top 1%, 5%, and 10% of cells (by count); see
        :meth:`niche_query_with_benchmark_metrics_report`. Per-sample dicts include
        ``pearson``, ``pcc``, ``spearman``, and related benchmark keys.
        """
        self.generate_niche_features_and_parcellation_mask(search_samples)
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
        rm_target_top_pct: Optional[Union[float, List[float]]] = None,
        *,
        viz_max_cols: int = 6,
        viz_panel_size_in: Tuple[float, float] = (2.75, 2.75),
        target_viz_panel_size_in: Tuple[float, float] = (4.5, 4.5),
        target_spot_size: Optional[float] = None,
        target_rm_cmap: str = "viridis",
    ) -> None:
        """
        Visualize niche-query cosine scores per sample; optionally plot target niches.

        If ``show_target_niches`` and ``rm_target_top_pct`` is set (e.g. ``[1, 5, 10]``),
        one combined figure is drawn: each row is a top-% threshold, each column is a
        sample. Cells outside the top K% are gray; top-K cells are colored by RM-Ideal
        score in [0, 1]. Requires ``compute_rm_ideal_score``. A single float is treated
        as a one-element list. Without ``rm_target_top_pct``, RM-Ideal targets use
        continuous scores; otherwise parcellation mask defines targets.

        ``target_viz_panel_size_in`` and ``target_spot_size`` control the larger target-
        niche grid (defaults are bigger than the niche-query panels).
        """
        top_pct_list = (
            _normalize_rm_target_top_pct_list(rm_target_top_pct)
            if show_target_niches
            else []
        )
        use_rm_top_pct_list = bool(top_pct_list)

        self.generate_niche_features_and_parcellation_mask(search_samples)

        for sample in search_samples:
            self.niche_query_within_a_sample(sample)
            if "niche_query_result" not in sample.adata.obs:
                sample.adata.obs["niche_query_result"] = np.nan
            for cell in sample.cells:
                sample.adata.obs.loc[cell.id, "niche_query_result"] = float(
                    cell.similarity
                )

            if not show_target_niches:
                continue

            if sample.rm_ideal_score is not None:
                rm_score = _as_1d_float64(
                    np.asarray(sample.rm_ideal_score, dtype=float)
                )
                if len(rm_score) != len(sample.cells):
                    raise ValueError(
                        f"Sample {sample.id!r}: rm_ideal_score length ({len(rm_score)}) "
                        f"does not match number of cells ({len(sample.cells)})."
                    )
                if use_rm_top_pct_list:
                    for pct in top_pct_list:
                        _set_target_niches_top_pct_scores(
                            sample,
                            rm_score,
                            pct,
                            _target_niches_obs_key_for_pct(pct),
                        )
                    sample.adata.uns["target_niches_top_pct_list"] = list(top_pct_list)
                else:
                    sample.adata.obs["target_niches"] = np.nan
                    for i, cell in enumerate(sample.cells):
                        sample.adata.obs.loc[cell.id, "target_niches"] = float(
                            rm_score[i]
                        )
                    sample.adata.uns.pop("target_niches_top_pct_list", None)
                continue

            if use_rm_top_pct_list:
                logger.warning(
                    "Sample %r: rm_target_top_pct=%s ignored (no rm_ideal_score); "
                    "using parcellation mask for target_niches.",
                    sample.id,
                    rm_target_top_pct,
                )
            sample.adata.uns.pop("target_niches_top_pct_list", None)
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
            target_palette = ["#d9d9d9", "#d62728"]
            tpw, tph = target_viz_panel_size_in
            t_spot = (
                float(target_spot_size)
                if target_spot_size is not None
                else max(spot_size, 0.04)
            )
            if use_rm_top_pct_list:
                n_pct = len(top_pct_list)
                fig, axes_grid = _spatial_viz_figure_grid(
                    n_pct,
                    n,
                    panel_width_in=tpw,
                    panel_height_in=tph,
                )
                for row, pct in enumerate(top_pct_list):
                    obs_key = _target_niches_obs_key_for_pct(pct)
                    for col, sample in enumerate(search_samples):
                        ax = axes_grid[row, col]
                        if obs_key not in sample.adata.obs.columns:
                            if sample.rm_ideal_score is None:
                                sample.adata.obs[obs_key] = np.full(
                                    sample.adata.n_obs, np.nan, dtype=np.float64
                                )
                            else:
                                _set_target_niches_top_pct_scores(
                                    sample,
                                    _as_1d_float64(
                                        np.asarray(sample.rm_ideal_score, dtype=float)
                                    ),
                                    pct,
                                    obs_key,
                                )
                        sc.pl.spatial(
                            sample.adata,
                            color=obs_key,
                            ax=ax,
                            show=False,
                            spot_size=t_spot,
                            vmin=0.0,
                            vmax=1.0,
                            cmap=target_rm_cmap,
                            na_color="#d9d9d9",
                            colorbar_loc="right" if col == n - 1 else None,
                            title=f"Top {pct:g}% RM-Ideal\n{sample.id}",
                        )
                pct_labels = ", ".join(f"{p:g}%" for p in top_pct_list)
                fig.suptitle(
                    f"Target niches by RM-Ideal top-% ({pct_labels}; color in [0, 1])",
                    fontsize=18,
                )
                fig.subplots_adjust(top=0.93, hspace=0.28, wspace=0.12)
                plt.tight_layout()
            else:
                fig, axes = _spatial_viz_figure(
                    n,
                    max_cols=viz_max_cols,
                    panel_width_in=pw,
                    panel_height_in=ph,
                )
                for i, sample in enumerate(search_samples):
                    tn = sample.adata.obs["target_niches"]
                    plot_kw: Dict[str, Any] = dict(
                        color="target_niches",
                        ax=axes[i],
                        show=False,
                        spot_size=spot_size,
                        title=f"Target niches\n{sample.id}",
                    )
                    if isinstance(tn.dtype, pd.CategoricalDtype):
                        plot_kw["palette"] = target_palette
                    sc.pl.spatial(sample.adata, **plot_kw)
                fig.suptitle("Target niches", fontsize=18)
                plt.tight_layout()
