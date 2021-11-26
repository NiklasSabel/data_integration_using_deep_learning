
import os
import pandas as pd

pd.set_option('display.max_colwidth', None)
data_path = '../src/data'
mapping_corpus_path = data_path + r'/product/lspc2020_to_tablecorpus'
mapping_corpus_path_2 = data_path + r'/product/lspc2020_to_tablecorpus/Cleaned'
table_corpus_path = data_path + r'/product/product_top100/cleaned'
table_corpus_path_with_id = data_path + r'/product/product_top100/cleaned/with_id'
table_corpus_path2 = data_path + r'/product/product_minimum3/cleaned/with_id'
zip_files_mapping = [file for file in os.listdir(mapping_corpus_path_2) if file.endswith('.json.gz')]
zip_files_tables = [file for file in os.listdir(table_corpus_path) if file.endswith('.json.gz')]



df_joined_clothes = pd.read_json(os.path.join(mapping_corpus_path_2, 'joined_random.json'), compression='gzip', orient='records', lines=True)
df_grouped_clothes = df_joined_clothes.groupby('cluster_id').count()
# only look at clusters that have at least one brand associated
df_set_clothes = df_grouped_clothes[df_grouped_clothes['brand_y']>0].reset_index()[['cluster_id','table_id']].rename(columns={'table_id':'Amount'})
# We discard all clusters with less than 2 entries, cause we cannot match anything there, so 1,6 million clusters remain
df_set_clothes[df_set_clothes['Amount']>7].to_csv(os.path.join(mapping_corpus_path_2, 'Random_cluster_8_tables.csv'))

print('1')

df_joined_clothes = pd.read_json(os.path.join(mapping_corpus_path_2, 'joined_bikes.json'), compression='gzip', orient='records', lines=True)
df_grouped_clothes = df_joined_clothes.groupby('cluster_id').count()
# only look at clusters that have at least one brand associated
df_set_clothes = df_grouped_clothes[df_grouped_clothes['brand_y']>0].reset_index()[['cluster_id','table_id']].rename(columns={'table_id':'Amount'})
# We discard all clusters with less than 2 entries, cause we cannot match anything there, so 1,6 million clusters remain
df_set_clothes[df_set_clothes['Amount']>7].to_csv(os.path.join(mapping_corpus_path_2, 'Bikes_cluster_8_tables.csv'))

