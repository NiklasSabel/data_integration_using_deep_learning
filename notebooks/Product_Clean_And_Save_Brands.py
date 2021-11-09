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

zip_files_mapping = [file for file in os.listdir(mapping_corpus_path_2) if file.endswith('.json.gz')]
zip_files_tables = [file for file in os.listdir(table_corpus_path) if file.endswith('.json.gz')]

# get dictionaries
electronics_path = '../src/data/product/product_electronics/electronics_dict.json'
clothes_path = '../src/data/product/product_clothes/clothes_dict.json'
mapping_corpus_path_2 = '../src/data/product/lspc2020_to_tablecorpus/Cleaned'

with open(electronics_path) as f:
    electronics_data=json.load(f)

with open(clothes_path) as f:
    clothes_data=json.load(f)

#clean the dictionaries by getting rid of the first key
cleaned_dictionary_electronics={}
for value in electronics_data.values():
    cleaned_dictionary_electronics.update(value)

#clean the dictionaries by getting rid of the first key
cleaned_dictionary_clothes={}
for value in clothes_data.values():
    cleaned_dictionary_clothes.update(value)

#put the dictionaries into dataframes
df_electronics=pd.DataFrame.from_dict(cleaned_dictionary_electronics, orient='index')
df_clothes=pd.DataFrame.from_dict(cleaned_dictionary_clothes, orient='index')

#fill up missing values in both product category data frames to be able to split the tuples up
df_electronics_filtered=df_electronics.applymap(lambda x: [0,0] if x is None else x)
df_clothes_filtered=df_clothes.applymap(lambda x: [0,0] if x is None else x)

# clean up the tables
#split up tuples in in each column for each brand into two different columns table_id and row_id and concatente these rows
df_electronics_cleaned=pd.DataFrame(columns=['table_id', 'row_id'])
count = 0
with progressbar.ProgressBar(max_value=len(df_electronics_filtered.columns)) as bar:
    for i in range(len(df_electronics_filtered.columns)):
        df_electronics_cleaned = df_electronics_cleaned.append(pd.DataFrame(df_electronics_filtered[i].tolist(),columns=['table_id', 'row_id'], index=df_electronics_filtered.index))
        count += 1
        bar.update(count)

#clean up the tables
#split up tuples in in each column for each brand into two different columns table_id and row_id and concatente these rows
df_clothes_cleaned=pd.DataFrame(columns=['table_id', 'row_id'])
count = 0
with progressbar.ProgressBar(max_value=len(df_clothes_filtered.columns)) as bar:
    for i in range(len(df_clothes_filtered.columns)):
        df_clothes_cleaned = df_clothes_cleaned.append(pd.DataFrame(df_clothes_filtered[i].tolist(),columns=['table_id', 'row_id'], index=df_clothes_filtered.index))
        count += 1
        bar.update(count)

#rename the columns to be able to join them into the cluster_id table
df_electronics_cleaned=df_electronics_cleaned.reset_index().rename(columns={'index':"brand"})
df_electronics_cleaned

#rename the columns to be able to join them into the cluster_id table
df_clothes_cleaned=df_clothes_cleaned.reset_index().rename(columns={'index':"brand"})
df_clothes_cleaned


df_electronics_cleaned.to_json(mapping_corpus_path_2 + '/cleaned_electronics_all_brands', compression='gzip', orient='records', lines=True)

df_clothes_cleaned.to_json(mapping_corpus_path_2 + '/cleaned_clothes_all_brands', compression='gzip', orient='records', lines=True)


