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




df_joined_electronics = pd.read_json(os.path.join(mapping_corpus_path_2, 'joined_electronics_v3.json'), compression='gzip', orient='records', lines=True)
#df_joined_clothes = pd.read_json(os.path.join(mapping_corpus_path_2, 'joined_clothes_v3.json'), compression='gzip', orient='records', lines=True)
df_joined_bikes = pd.read_json(os.path.join(mapping_corpus_path_2, 'joined_bikes.json'), compression='gzip', orient='records', lines=True)
df_joined_cars = pd.read_json(os.path.join(mapping_corpus_path_2, 'joined_cars.json'), compression='gzip', orient='records', lines=True)
df_joined_drugstore = pd.read_json(os.path.join(mapping_corpus_path_2, 'joined_drugstore.json'), compression='gzip', orient='records', lines=True)
df_joined_technology = pd.read_json(os.path.join(mapping_corpus_path_2, 'joined_technology.json'), compression='gzip', orient='records', lines=True)
df_joined_tools = pd.read_json(os.path.join(mapping_corpus_path_2, 'joined_tools.json'), compression='gzip', orient='records', lines=True)

df_grouped_electronics = df_joined_electronics.groupby('cluster_id').count()
#df_grouped_clothes = df_joined_clothes.groupby('cluster_id').count()
df_grouped_bikes = df_joined_bikes.groupby('cluster_id').count()
df_grouped_cars = df_joined_cars.groupby('cluster_id').count()
df_grouped_drugstore = df_joined_drugstore.groupby('cluster_id').count()
df_grouped_technology = df_joined_technology.groupby('cluster_id').count()
df_grouped_tools = df_joined_tools.groupby('cluster_id').count()

# only look at clusters that have at least one brand associated
df_set_electronics = df_grouped_electronics[df_grouped_electronics['brand_y']>0].reset_index()[['cluster_id','table_id']].rename(columns={'table_id':'Amount'})
#df_set_clothes = df_grouped_clothes[df_grouped_clothes['brand_y']>0].reset_index()[['cluster_id','table_id']].rename(columns={'table_id':'Amount'})
df_set_bikes = df_grouped_bikes[df_grouped_bikes['brand_y']>0].reset_index()[['cluster_id','table_id']].rename(columns={'table_id':'Amount'})
df_set_cars = df_grouped_cars[df_grouped_cars['brand_y']>0].reset_index()[['cluster_id','table_id']].rename(columns={'table_id':'Amount'})
df_set_drugstore = df_grouped_drugstore[df_grouped_drugstore['brand_y']>0].reset_index()[['cluster_id','table_id']].rename(columns={'table_id':'Amount'})
df_set_techonlogy = df_grouped_technology[df_grouped_technology['brand_y']>0].reset_index()[['cluster_id','table_id']].rename(columns={'table_id':'Amount'})
df_set_tools = df_grouped_tools[df_grouped_tools['brand_y']>0].reset_index()[['cluster_id','table_id']].rename(columns={'table_id':'Amount'})

# We discard all clusters with less than 2 entries, cause we cannot match anything there, so 1,6 million clusters remain
df_set_electronics[df_set_electronics['Amount']>8].to_csv(os.path.join(mapping_corpus_path_2, 'Electronics_cluster_8_tables.csv'))
#df_set_clothes[df_set_clothes['Amount']>8].to_csv(os.path.join(mapping_corpus_path_2, 'Clothes_cluster_8_tables.csv'))
df_set_bikes[df_set_bikes['Amount']>8].to_csv(os.path.join(mapping_corpus_path_2, 'Bikes_cluster_8_tables.csv'))
df_set_cars[df_set_cars['Amount']>8].to_csv(os.path.join(mapping_corpus_path_2, 'Cars_cluster_8_tables.csv'))
df_set_drugstore[df_set_drugstore['Amount']>8].to_csv(os.path.join(mapping_corpus_path_2, 'Drugstore_cluster_8_tables.csv'))
df_set_techonlogy[df_set_techonlogy['Amount']>8].to_csv(os.path.join(mapping_corpus_path_2, 'Technology_cluster_8_tables.csv'))
df_set_tools[df_set_tools['Amount']>8].to_csv(os.path.join(mapping_corpus_path_2, 'Tools_cluster_8_tables.csv'))