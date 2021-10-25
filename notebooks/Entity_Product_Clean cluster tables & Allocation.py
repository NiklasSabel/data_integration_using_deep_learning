import pandas as pd
import os
import progressbar
import json
import gzip
import shutil

path_parent = os.path.dirname(os.getcwd())

cleaned_top100_path = os.path.join(path_parent, 'src/data/product/product_top100/cleaned')
cleaned_min3_path = os.path.join(path_parent, 'src/data/product/product_minimum3/cleaned')

cluster_path = os.path.join(path_parent, 'src/data/product/lspc2020_to_tablecorpus/Cleaned')

def clean_clusters():
    """
    iterate through all cluster_files;
    clean them by using only valid top100 and min3 files after language detection;
    count how much tables include a certain product
    :return:
    """

    # list all valid files after language detection
    data_files = [file for file in os.listdir(cleaned_min3_path) if file.endswith('.json.gz')]
    data_files += [file for file in os.listdir(cleaned_top100_path) if file.endswith('.json.gz')]

    cluster_files = [file for file in os.listdir(cluster_path) if file.endswith('.json.gz')]

    # generate dictionaries with different information to track product allocation
    allocation_with_table_ids_total_dict = {}
    allocation_with_table_ids_set_dict = {}
    allocation_amount_only_total_dict = {}
    allocation_amount_only_set_dict = {}

    unique_cluster_ids = []

    count_files = 0
    for cluster_file in cluster_files:
        print(cluster_file)
        df = pd.read_json(os.path.join(cluster_path, '{}'.format(cluster_file)), compression='gzip', lines=True)

        # design new dataframe with valid tables only
        df_cleaned = df[df['table_id'].isin(data_files)]
        df_cleaned = df_cleaned.reset_index()
        df_cleaned = df_cleaned.drop('index', axis=1)

        # generate a unique list of cluster IDs
        cluster_ids = df_cleaned['cluster_id'].tolist()
        if unique_cluster_ids == []:
            new_cluster_ids = list(set(cluster_ids))
        else:
            new_cluster_ids = list(set(cluster_ids) - set(unique_cluster_ids))
        unique_cluster_ids += new_cluster_ids
        unique_cluster_ids = list(set(unique_cluster_ids))

        # add dictionary keys
        new_cluster_ids_tables_dict = {key: [] for key in new_cluster_ids}
        new_cluster_ids_amount_dict = {key: 0 for key in new_cluster_ids}

        allocation_with_table_ids_total_dict.update(new_cluster_ids_tables_dict)
        allocation_with_table_ids_set_dict.update(new_cluster_ids_tables_dict)
        allocation_amount_only_total_dict.update(new_cluster_ids_amount_dict)
        allocation_amount_only_set_dict.update(new_cluster_ids_amount_dict)

        count = 0
        with progressbar.ProgressBar(max_value=df_cleaned.shape[0]) as bar:
            for i in range(df_cleaned.shape[0]):  # iterate over rows
                cluster_id = df_cleaned['cluster_id'][i]
                table_id = df_cleaned['table_id'][i]

                allocation_with_table_ids_total_dict[cluster_id].append(table_id) # write every table_id inside
                allocation_amount_only_total_dict[cluster_id] += 1 # increment for every table_id
                allocation_with_table_ids_set_dict[cluster_id] = list(set(allocation_with_table_ids_total_dict[cluster_id])) # write only unique table_ids inside
                allocation_amount_only_set_dict[cluster_id] = len(allocation_with_table_ids_set_dict[cluster_id]) # increment only for unique table_ids

                count += 1
                bar.update(count)

        count_files += 1

        print('{} out of {} cluster files done'.format(count_files, len(cluster_files)))

        # write to gzip compressed json file
        df_cleaned.to_json(os.path.join(cluster_path, '{}'.format(cluster_file)), compression='gzip', orient='records', lines=True)

    # save dictionaries with allocation of products
    with open(os.path.join(cluster_path, 'allocation_with_table_ids_total_dict.json'), 'w', encoding='utf-8') as f:
        json.dump(allocation_with_table_ids_total_dict, f)

    with open(os.path.join(cluster_path, 'allocation_with_table_ids_set_dict.json'), 'w', encoding='utf-8') as f:
        json.dump(allocation_with_table_ids_set_dict, f)

    with open(os.path.join(cluster_path, 'allocation_amount_only_total_dict.json'), 'w', encoding='utf-8') as f:
        json.dump(allocation_amount_only_total_dict, f)

    with open(os.path.join(cluster_path, 'allocation_amount_only_set_dict.json'), 'w', encoding='utf-8') as f:
        json.dump(allocation_amount_only_set_dict, f)

# run functions
clean_clusters()