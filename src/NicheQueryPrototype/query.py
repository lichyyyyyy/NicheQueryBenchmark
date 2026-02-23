from typing import List, Tuple

import numpy as np
from sklearn.metrics import precision_recall_curve, classification_report
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize

from src.NicheQueryPrototype.database import Database, Sample
from src.NicheQueryPrototype.niche import Niche, logger


class NicheQuery:
    def __init__(self, db: Database, niche: Niche, k: int):
        assert 0 < k <= 10, "Invalid subgraph parameter, should be between 1 and 10 (included)"
        self.db = db
        self.niche = niche
        self.k = k
        self.generate_niche_features_and_parcellation_mask()

    def get_parcellation_mask(self, sample: Sample) -> np.ndarray:
        parcellation_indices = np.array([c.parcellation_index for c in sample.cells])
        return (parcellation_indices == self.niche.parcellation_index).astype(int)

    """
    Generate niche embedding and parcellation mask for all samples.
    """

    def generate_niche_features_and_parcellation_mask(self):
        for sample in self.db.samples:
            logger.info(f"[START] start computing niche features for sample {sample.id}")
            sample_features = np.array([c.feature for c in sample.cells])
            sample.target_parcellation_mask = self.get_parcellation_mask(sample)
            niche_connectivity = kneighbors_graph(
                sample_features,
                n_neighbors=self.k,
                mode="connectivity",
                metric="euclidean",
                include_self=True,
            )
            niche_connectivity_norm = normalize(niche_connectivity, norm='l1', axis=1)
            sample.niche_features = np.asarray(niche_connectivity_norm @ sample_features)

            logger.info(f"[DONE] finish sample: {sample.id}")

    """
    Perform niche query task on a sample and return cosine similarity between query niche and each niches.
    """

    def niche_query_within_a_sample(self, sample: Sample) -> np.ndarray:
        cos_sim = cosine_similarity(self.niche.feature.reshape(1, -1), sample.niche_features)
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
            cos_sim_list.append(self.niche_query(sample))
            parcellation_mask_list.append(sample.target_parcellation_mask)

        cos_sim = np.vstack(cos_sim_list)
        parcellation_mask = np.vstack(parcellation_mask_list)

        precision, recall, thresholds = precision_recall_curve(y_true=parcellation_mask, y_score=cos_sim,
                                                               pos_label=1)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        best_idx = np.argmax(f1)
        best_threshold = thresholds[best_idx]
        best_f1 = f1[best_idx]
        logger.info(f"best threshold: {best_threshold}")
        logger.info(f"f1 score: {best_f1}")
        return best_threshold, best_f1

    """
    Perform niche query task and return samples with pred niches.
    """

    def niche_query(self, best_threshold: float, samples) -> List[Sample]:
        cos_sim_list = []
        parcellation_mask_list = []
        for sample in samples:
            cos_sim_list.append(self.niche_query(sample))
            if sample.target_parcellation_mask is not None:
                parcellation_mask_list.append(sample.target_parcellation_mask)

        cos_sim = np.vstack(cos_sim_list) > best_threshold
        if len(parcellation_mask_list) > 0:
            parcellation_mask = np.vstack(parcellation_mask_list)
            logger.info(f"Classification report with {len(samples)} samples:")
            logger.info(classification_report(y_true=parcellation_mask, y_pred=cos_sim))

        pred_pos_sample_idx = np.where(cos_sim.sum(axis=1) > 1)[0]
        return [samples[i] for i in pred_pos_sample_idx]

    """
    Perform niche query task, find best threshold and generate classification report.
    step 1: compute the best threshold with 30% samples with the parcellation section
    step 2: use the best threshold to run niche query
    step 3: compute f1 score using all samples
    """

    def niche_query_verification(self):
        target_samples = [s for s in self.db.samples if s.target_parcellation_mask.sum() > 0]
        non_target_samples = [s for s in self.db.samples if s.target_parcellation_mask.sum() == 0]
        val_samples, test_samples = train_test_split(
            target_samples,
            test_size=0.7,
            random_state=42,
            shuffle=True
        )
        best_threshold, _ = self.get_best_threshold(val_samples)

        all_test_samples = np.vstack([test_samples, non_target_samples])
        np.random.shuffle(all_test_samples)
        self.niche_query(best_threshold, all_test_samples)
