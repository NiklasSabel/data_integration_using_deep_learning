import pandas as pd
import os
import progressbar
import json
import gzip
import shutil

path_parent = os.path.dirname(os.getcwd())
data_path = os.path.join(path_parent, '../../../src/data')

def add_cluster_id_column():
    print('add cluster id column')
    count = 0
    with progressbar.ProgressBar(max_value=len(zip_files_tables)) as bar:
        for file in zip_files_tables:
            df = pd.read_json(os.path.join(table_corpus_path,'{}'.format(file)), compression='gzip', lines=True)
            df['cluster_id'] = None
            if df.shape[0] > 0:
                df.to_json(os.path.join(table_with_id_path, '{}').format(file), compression='gzip',
                           orient='records', lines=True)
            count += 1
            bar.update(count)

    print('adding cluster id column done')


def match_cluster_ids_to_tables():

    print('match cluster ids')

    try:
        rd = open(os.path.join(table_with_id_path, 'files_done.txt'), 'r')
        files_done = rd.readlines()
        rd.close()
    except:
        files_done = []

    count_files = 0
    for map_file in zip_files_mapping:
        if '{}\n'.format(map_file) not in files_done:
            df_map = pd.read_json(os.path.join(mapping_corpus_path, '{}'.format(map_file)), compression='gzip', lines=True)

            print(map_file)
            count = 0
            with progressbar.ProgressBar(max_value=df_map.shape[0]) as bar:
                for i in range(df_map.shape[0]):
                    table_id = df_map['table_id'][i]
                    if table_id in zip_files_with_id:
                        try:
                            df = pd.read_json(os.path.join(table_with_id_path, '{}'.format(table_id)), compression='gzip', lines=True)
                            row_id = df_map['row_id'][i]
                            cluster_id = df_map['cluster_id'][i]
                            df.loc[df['row_id']==row_id, 'cluster_id'] = int(cluster_id)
                            df.to_json(os.path.join(table_with_id_path, '{}').format(table_id), compression='gzip',
                                       orient='records', lines=True)
                        except:
                            print(table_id)
                    count += 1
                    bar.update(count)

            # update files_done status
            files_done.append('{}\n'.format(map_file))
            wr = open(os.path.join(table_with_id_path, 'files_done.txt'), 'w')
            for element in files_done:
                wr.writelines([element])
            wr.close()

            count_files += 1
            print('{} out of {} files done'.format(count_files, len(zip_files_mapping)))

    print('matching cluster ids done')


# run functions
entities = ['product_top100', 'product_minimum3']
#entities = ['product_top100']

for entity in entities:
    if entity == 'product_top100':
        mapping_corpus_path = os.path.join(data_path, 'product/lspc2020_to_tablecorpus/Cleaned')
        table_corpus_path = os.path.join(data_path, 'product/product_top100/cleaned')
        table_with_id_path = os.path.join(table_corpus_path, 'with_id')

        zip_files_mapping = os.listdir(mapping_corpus_path)
        zip_files_tables = [file for file in os.listdir(table_corpus_path) if file.endswith('.json.gz')]
        zip_files_with_id = [file for file in os.listdir(table_with_id_path) if file.endswith('.json.gz')]
    elif entity == 'product_minimum3':
        mapping_corpus_path = os.path.join(data_path, 'product/lspc2020_to_tablecorpus/Cleaned')
        table_corpus_path = os.path.join(data_path, 'product/product_minimum3/cleaned')
        table_with_id_path = os.path.join(table_corpus_path, 'with_id')

        zip_files_mapping = os.listdir(mapping_corpus_path)
        zip_files_tables = [file for file in os.listdir(table_corpus_path) if file.endswith('.json.gz')]
        zip_files_with_id = [file for file in os.listdir(table_with_id_path) if file.endswith('.json.gz')]

    print('running {} path'.format(entity))

    add_cluster_id_column()
    match_cluster_ids_to_tables()