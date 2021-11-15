import pandas as pd
import json
import gzip
import os

path_parent = os.path.dirname(os.getcwd())
data_path = os.path.join(path_parent, 'src/data')
mapping_corpus_path_2 = os.path.join(data_path, 'product/lspc2020_to_tablecorpus/Cleaned')
notebook_path = os.path.join(path_parent, 'notebooks')
product_path = os.path.join(data_path, 'product')

clothes_cluster_df = pd.read_json(os.path.join(mapping_corpus_path_2, 'joined_clothes_v2.json'), compression='gzip', orient='records', lines=True)

electronics_final_entities_df = pd.read_csv(os.path.join(notebook_path, 'electronics10.csv'), index_col=None)
electronics_final_entities_list = electronics_final_entities_df['cluster_id']

clothes_final_entities_df = pd.read_csv(os.path.join(notebook_path, 'clothes8.csv'), index_col=None)
clothes_final_entities_list = clothes_final_entities_df['cluster_id']

electronics_clusters_all_15_df = cluster_df[cluster_df['cluster_id'].isin(electronics_final_entities_list)]
clothes_clusters_all_10_df = clothes_cluster_df[clothes_cluster_df['cluster_id'].isin(clothes_final_entities_list)]

electronics_clusters_all_15_df.to_csv(os.path.join(mapping_corpus_path_2, 'electronics_clusters_all_15_tables.csv'), columns=None)
clothes_clusters_all_10_df.to_csv(os.path.join(mapping_corpus_path_2, 'clothes_clusters_all_8_tables_v2.csv'), columns=None)

test = 1