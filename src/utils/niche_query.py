import logging

from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

"""
NicheQuery: a class to perform Niche queries. Assuming embeddings are generated for each node.
"""


class NicheQuery:
    def __init__(self, logger=logging.getLogger(__name__)):
        self.logger = logger
        self.logger.info("initializing QueST object")

    def query(self, query_adata_list, query_niche_embedding, query_result_key, niche_embedding_key):
        for adata in query_adata_list:
            assert niche_embedding_key in adata.obsm.keys(), f'{niche_embedding_key} not in adata'
        adata_list = []
        for adata in query_adata_list:
            cos_sim = cosine_similarity(query_niche_embedding.reshape(1,-1), adata.obsm[niche_embedding_key])
            adata.obs[query_result_key] = cos_sim.flatten()
            adata_list.append(adata)
        return adata_list
