import logging

from src.NicheQueryPrototype.database import Database
from src.NicheQueryPrototype.niche import Niche
from src.NicheQueryPrototype.query import NicheQuery

logging.basicConfig(level=logging.INFO)

data_folder = '../notebook/ccf/data/zhuang-abca-1/'
db = Database()
db.construct(adata_path=[data_folder + 'Zhuang-ABCA-1-raw.h5ad'],
             cell_metadata_path=[data_folder + 'cell_metadata.csv'],
             ccf_coordinates_path=[data_folder + 'ccf_coordinates.csv'],
             feature_name='gene_expression',
             parcellation_path='../notebook/ccf/data/parcellation/structure.json')

niche_to_query = Niche()
niche_to_query.construct_by_k_hop(db, center_cell=None, sample_id='Zhuang-ABCA-1.086', k=10, cell_limit=100,
                                  parcellation_index=987)

niche_query = NicheQuery(db=db, niche=niche_to_query, k=3)
target_samples = ['Zhuang-ABCA-1.086', 'Zhuang-ABCA-1.104', 'Zhuang-ABCA-1.102', 'Zhuang-ABCA-1.087',
                  'Zhuang-ABCA-1.090', 'Zhuang-ABCA-1.093', 'Zhuang-ABCA-1.084', 'Zhuang-ABCA-1.088',
                  'Zhuang-ABCA-1.096']
query_samples = [s for s in db.samples.values() if s.id in target_samples]
niche_query.niche_query_visualization(query_samples)
