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
clothes_path = '../src/data/product/product_clothes_v3/clothes_dict.json'
mapping_corpus_path_2 = '../src/data/product/lspc2020_to_tablecorpus/Cleaned'
with open(clothes_path) as f:
    clothes_data=json.load(f)

#clean the dictionaries by getting rid of the first key
cleaned_dictionary_clothes={}
for value in clothes_data.values():
    cleaned_dictionary_clothes.update(value)

df_clothes=pd.DataFrame.from_dict(cleaned_dictionary_clothes, orient='index')
df_clothes_filtered=df_clothes.applymap(lambda x: [0,0] if x is None else x)
#clean up the tables
#split up tuples in in each column for each brand into two different columns table_id and row_id and concatente these rows
df_clothes_cleaned=pd.DataFrame(columns=['table_id', 'row_id'])
count = 0
with progressbar.ProgressBar(max_value=len(df_clothes_filtered.columns)) as bar:
    for i in range(len(df_clothes_filtered.columns)):
        df_clothes_cleaned = df_clothes_cleaned.append(pd.DataFrame(df_clothes_filtered[i].tolist(),columns=['table_id', 'row_id'], index=df_clothes_filtered.index))
        count += 1
        bar.update(count)
df_clothes_cleaned=df_clothes_cleaned.reset_index().rename(columns={'index':"brand"})
df_joined_clothes = df_large.merge(df_clothes_cleaned, left_on=['table_id','row_id'], right_on = ['table_id','row_id'], how='left')
df_joined_clothes.to_json(mapping_corpus_path_2 + '/joined_clothes_v3.json', compression='gzip', orient='records', lines=True)
