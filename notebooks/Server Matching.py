import os
import pandas as pd
import plotly.express as px
import progressbar

data_path = '../src/data'
mapping_corpus_path = data_path + r'/product/lspc2020_to_tablecorpus'
mapping_corpus_path_2 = data_path + r'/product/lspc2020_to_tablecorpus/Cleaned'
table_corpus_path = data_path + r'/product/product_top100/cleaned'
table_corpus_path_with_id = data_path + r'/product/product_top100/cleaned/with_id'
table_corpus_path2 = data_path + r'/product/product_minimum3/cleaned/with_id'

zip_files_mapping = [file for file in os.listdir(mapping_corpus_path_2) if file.endswith('.json.gz')]
zip_files_tables = [file for file in os.listdir(table_corpus_path) if file.endswith('.json.gz')]

count = 0
count_1=0
with progressbar.ProgressBar(max_value=len(zip_files_tables)) as bar:
    for zip_file in zip_files_tables:
        print('/{}'.format(zip_file))
        df = pd.read_json(table_corpus_path + '/{}'.format(zip_file), compression='gzip', lines=True)
        df['cluster_id']=0
        for zip_file_map in zip_files_mapping:
            count_1=count_1+1
            print(count_1)
            df_map = pd.read_json(mapping_corpus_path_2 + '/{}'.format(zip_file_map), compression='gzip', lines=True)
            for i in range(len(df_map)):
                if df_map['table_id'][i]=='{}'.format(zip_file):
                  index_map=df_map['row_id'][i]
                  if df.index[df['row_id'] == index_map].size != 0:
                    index_table=df.index[df['row_id'] == index_map][0]
                    if df['cluster_id'][index_table]==0:
                        df['cluster_id'][index_table]=df_map['cluster_id'][i]
                    else:
                        print('double product value in table')
        df.to_json(table_corpus_path_with_id + '/{}'.format(zip_file), compression='gzip', orient='records', lines=True)
        df
        count += 1
        bar.update(count)

data_path = '../src/data'
mapping_corpus_path = data_path + r'/product/lspc2020_to_tablecorpus'
mapping_corpus_path_2 = data_path + r'/product/lspc2020_to_tablecorpus/Cleaned'
table_corpus_path = data_path + r'/product/product_top100/cleaned'
table_corpus_path_with_id = data_path + r'/product/product_top100/cleaned/with_id'
table_corpus_path2_with_id= data_path + r'/product/product_minimum3/cleaned/with_id'
table_corpus_path2= data_path + r'/product/product_minimum3/cleaned/'

zip_files_mapping = [file for file in os.listdir(mapping_corpus_path_2) if file.endswith('.json.gz')]
zip_files_tables = [file for file in os.listdir(table_corpus_path2) if file.endswith('.json.gz')]


count = 0
count_1=0
with progressbar.ProgressBar(max_value=len(zip_files_tables)) as bar:
    for zip_file in zip_files_tables:
        print('/{}'.format(zip_file))
        df = pd.read_json(table_corpus_path2 + '/{}'.format(zip_file), compression='gzip', lines=True)
        df['cluster_id']=0
        for zip_file_map in zip_files_mapping:
            count_1=count_1+1
            print(count_1)
            df_map = pd.read_json(mapping_corpus_path_2 + '/{}'.format(zip_file_map), compression='gzip', lines=True)
            for i in range(len(df_map)):
                if df_map['table_id'][i]=='{}'.format(zip_file):
                  index_map=df_map['row_id'][i]
                  if df.index[df['row_id'] == index_map].size != 0:
                    index_table=df.index[df['row_id'] == index_map][0]
                    if df['cluster_id'][index_table]==0:
                        df['cluster_id'][index_table]=df_map['cluster_id'][i]
                    else:
                        print('double product value in table')
        df.to_json(table_corpus_path2_with_id + '/{}'.format(zip_file), compression='gzip', orient='records', lines=True)
        df
        count += 1
        bar.update(count)