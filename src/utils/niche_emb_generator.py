import logging

import numpy as np
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize

"""
NicheEmbGenerator: a class to generate niche embeddings for each node as the center node. Assume each cell has its own embedding.
"""


class NicheEmbGenerator:
    def __init__(self, logger=logging.getLogger(__name__)):
        self.logger = logger
        self.logger.info("initializing QueST object")

    # generate niche embedding for given masks only
    def generate_niche_embedding(self, mask, embedding_key, adata):
        assert embedding_key in adata.obsm.keys(), (
            "Invalid embedding key: embedding not exists in adata"
        )
        return adata.obsm[embedding_key][mask].mean(axis=0)

    # generate niche embedding for all nodes
    def generate_niche_embeddings(
        self, niche_embedding_key, embedding_key, coordinates_key, k, adata_list
    ):
        assert 0 < k <= 10, (
            "Invalid subgraph parameter, should be between 1 and 10 (included)"
        )
        for adata in adata_list:
            assert coordinates_key in adata.obsm.keys(), (
                "Invalid spatial coordinates key"
            )
            assert embedding_key in adata.obsm.keys(), (
                "Invalid embedding key: embedding not exists in adata"
            )
        for adata in adata_list:
            print(f"[START] start processing adata: {adata.uns['library_id']}")
            niche_connectivity = kneighbors_graph(
                adata.obsm[coordinates_key],
                n_neighbors=k,
                mode="connectivity",
                metric="euclidean",
                include_self=True,
            )
            niche_connectivity_norm = normalize(niche_connectivity, norm="l1", axis=1)
            adata.obsm[niche_embedding_key] = np.asarray(
                niche_connectivity_norm @ adata.obsm[embedding_key]
            )
            print(f"[DONE] finish adata: {adata.uns['library_id']}")
