import os
import pandas as pd
import plotly.express as px
import progressbar
import json
import numpy as np
data_path = '../src/data'
mapping_corpus_path = data_path + r'/product/lspc2020_to_tablecorpus'
mapping_corpus_path_2 = data_path + r'/product/lspc2020_to_tablecorpus/Cleaned'
table_corpus_path = data_path + r'/product/product_top100/cleaned'
table_corpus_path_with_id = data_path + r'/product/product_top100/cleaned/with_id'
table_corpus_path2 = data_path + r'/product/product_minimum3/cleaned/with_id'
mapping_corpus_path_all = data_path + r'/product/lspcV2020'
zip_files_mapping = [file for file in os.listdir(mapping_corpus_path_2) if file.endswith('.json.gz')]
zip_files_tables = [file for file in os.listdir(table_corpus_path) if file.endswith('.json.gz')]
df_large = pd.read_json(os.path.join(mapping_corpus_path_2, 'df_large_matched.json'), compression='gzip', orient='records', lines=True)
random_path = '../src/data/product/product_random/random_dict.json'
mapping_corpus_path_2 = '../src/data/product/lspc2020_to_tablecorpus/Cleaned'
with open(random_path) as f:
    random_data=json.load(f)

#clean the dictionaries by getting rid of the first key
random_dictionary={}
for value in random_data.values():
    random_dictionary.update(value)

df_random=pd.DataFrame.from_dict(random_dictionary, orient='index')
df_random_filtered=df_random.applymap(lambda x: [0,0] if x is None else x)
#clean up the tables
#split up tuples in in each column for each brand into two different columns table_id and row_id and concatente these rows
df_random_cleaned=pd.DataFrame(columns=['table_id', 'row_id'])
count = 0
with progressbar.ProgressBar(max_value=len(df_random_filtered.columns)) as bar:
    for i in range(len(df_random_filtered.columns)):
        df_random_cleaned = df_random_cleaned.append(pd.DataFrame(df_random_filtered[i].tolist(),columns=['table_id', 'row_id'], index=df_random_filtered.index))
        count += 1
        bar.update(count)
df_random_cleaned=df_random_cleaned.reset_index().rename(columns={'index':"brand"})
df_joined_random = df_large.merge(df_random_cleaned, left_on=['table_id','row_id'], right_on = ['table_id','row_id'], how='left')
df_joined_random.to_json(mapping_corpus_path_2 + '/joined_random.json', compression='gzip', orient='records', lines=True)
