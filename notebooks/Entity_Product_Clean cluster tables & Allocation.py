import pandas as pd
import os
import progressbar
import json
import gzip
import shutil
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import json
import re
import logging
import threading
import time

path_parent = os.path.dirname(os.getcwd())
product_path = os.path.join(path_parent, 'src/data/product')

cleaned_top100_path = os.path.join(product_path, 'product_top100/cleaned')
cleaned_min3_path = os.path.join(product_path, 'product_minimum3/cleaned')

cluster_path = os.path.join(product_path, 'lspc2020_to_tablecorpus/Cleaned')

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


def get_keywords():
    """
    finds all important brands for clothes and electronics
    :return: dictionary {'clothes' : [clothes_brand1, clothes_brand2, ...],
                        'electronics' : [electronics_brand1, electronics_brand2, ...]}
    """
    print('get keywords')
    # search for clothes brands top100
    clothes_html = urlopen('https://fashionunited.com/i/most-valuable-fashion-brands/')
    clothes_bsObj = BeautifulSoup(clothes_html.read(), 'lxml')
    clothes_table = clothes_bsObj.find('table')
    clothes_lines = clothes_table.find('tbody').find_all('tr')

    clothes_list = []
    for clothes_line in clothes_lines:
        clothes_brand = clothes_line.get_text().split('\n')[2].lower()
        clothes_list.append(clothes_brand)

    # search for top electronic brands
    req = Request('https://companiesmarketcap.com/electronics/largest-electronic-manufacturing-by-market-cap/',
        headers={'User-Agent': 'Mozilla/5.0'})
    electronics_html = urlopen(req)
    electronics_bsObj = BeautifulSoup(electronics_html.read(), 'lxml')
    electronics_lines = electronics_bsObj.find_all('tr')

    electronics_list = []
    for electronics_line in electronics_lines:
        electronics_brand_info = electronics_line.find('a')
        if electronics_brand_info != None:
            electronics_brand = electronics_brand_info.find('div').get_text().split('\r')[0].lower()
            electronics_list.append(electronics_brand)

    # second page
    electronics_list2 = ['intel', 'taiwan semiconductor manufacturing', 'samsung electronics', 'hon hai precision industry',
                         'hitachi', 'sony', 'panasonic', 'lg electronics', 'pegatron', 'mitsubishi electric', 'midea group',
                         'honeywell international', 'apple', 'dell technologies', 'hp', 'lenovo', 'quanta computer', 'canon',
                         'compal eLectronics', 'hewlett packard enterprise']

    brands_dict = {'clothes': clothes_list, 'electronics1':electronics_list, 'electronics2':electronics_list2,
                   'electronics_total':list(set(electronics_list + electronics_list2))}

    with open(os.path.join(product_path, 'brands_dict.json'), 'w', encoding='utf-8') as f:
        json.dump(brands_dict, f)

    print('getting keywords done')

    return brands_dict

def keyword_search(data_path):
    """
    product selection for phase 1b;
    selects only "electronic products" for structured data and "clothes" for unstructured data
    :return: two dictionaries for electronics, clothes each containing table and row ids
    """

    print('run keyword search')

    with open(os.path.join(product_path, 'brands_dict.json'), 'r', encoding='utf-8') as f:
        brands_dict = json.load(f)

    data_files = [file for file in os.listdir(data_path) if file.endswith('.json.gz')]

    # for testing
    #brands_dict['clothes'].append('nejron')  ##
    #brands_dict['electronics_total'].append('arip santoso')  ##

    entity = data_path.split('product_')[1]
    print(entity)
    # check whether dictionaries already exist
    if os.path.isfile(os.path.join(product_path,'product_clothes', 'clothes_dict.json')):
        with open(os.path.join(product_path,'product_clothes', 'clothes_dict.json'), 'r', encoding='utf-8') as f:
            clothes_dict = json.load(f)
    else:
        clothes_dict = {'top100/cleaned':{key: [] for key in brands_dict['clothes']},
                        'minimum3/cleaned':{key: [] for key in brands_dict['clothes']}}

    if os.path.isfile(os.path.join(product_path,'product_electronics', 'electronics_dict.json')):
        with open(os.path.join(product_path,'product_electronics', 'electronics_dict.json'), 'r', encoding='utf-8') as f:
            electronics_dict = json.load(f)
    else:
        electronics_dict = {'top100/cleaned':{key: [] for key in brands_dict['electronics_total']},
                            'minimum3/cleaned':{key: [] for key in brands_dict['electronics_total']}}

    count_files = 0
    for data_file in data_files:
        print(data_file)
        df = pd.read_json(os.path.join(data_path, '{}'.format(data_file)), compression='gzip', lines=True)

        clothes_row_ids = []
        electronics_row_ids = []

        # iterrate over rows and look for keywords
        count = 0
        with progressbar.ProgressBar(max_value=df.shape[0]) as bar:
            if 'brand' in df.columns: # check whether column 'brand' exists
                for i in range(df.shape[0]):  # iterate over rows
                    #if i < 1000: # only for testing
                    row_id = int(df['row_id'][i])
                    cell = df['brand'][i]
                    if cell != None:
                        cell = str(cell).lower()
                        if cell in brands_dict['clothes']:
                            clothes_dict[entity][cell].append((data_file, row_id))
                            clothes_row_ids.append(row_id)
                        elif cell in brands_dict['electronics_total']:
                            electronics_dict[entity][cell].append((data_file, row_id))
                            electronics_row_ids.append(row_id)
                    count += 1
                    bar.update(count)
            else: # if column 'brand' does not exist check for whole row in concatenated column
                df['concat'] = ''
                df['brand'] = ''
                for j in range(df.shape[1]):  # iterate over columns
                    df['concat'] = df['concat'] + df.iloc[:, j].astype('str')

                # iterrate over rows
                count = 0
                with progressbar.ProgressBar(max_value=df.shape[0]) as bar:
                    for i in range(df.shape[0]):
                        #if i < 1000: # for testing
                        row_id = int(df['row_id'][i])
                        cell = df['concat'][i]
                        if cell != None:
                            cell = str(cell).lower()
                            for brand in brands_dict['clothes']:
                                if ' {} '.format(brand) in cell:
                                    clothes_dict[entity][brand].append((data_file, row_id))
                                    clothes_row_ids.append(row_id)
                                    df['brand'] = brand
                                    break
                            for brand in brands_dict['electronics_total']:
                                if ' {} '.format(brand) in cell:
                                    electronics_dict[entity][brand].append((data_file, row_id))
                                    electronics_row_ids.append(row_id)
                                    df['brand'] = brand
                                    break
                        count += 1
                        bar.update(count)

                # drop concatenated row again
                df = df.drop('concat', axis=1)

        # save dictionaries with selected data
        with open(os.path.join(product_path,'product_clothes', 'clothes_dict.json'), 'w', encoding='utf-8') as f:
            json.dump(clothes_dict, f)

        with open(os.path.join(product_path,'product_electronics', 'electronics_dict.json'), 'w', encoding='utf-8') as f:
            json.dump(electronics_dict, f)

        # write selected data into seperate folders
        clothes_df = df[df['row_id'].isin(clothes_row_ids)]
        electronics_df = df[df['row_id'].isin(electronics_row_ids)]

        if clothes_df.shape[0] > 0:
            clothes_df.to_json(os.path.join(product_path, 'product_clothes' ,data_file), compression='gzip', orient='records',
                               lines=True)

        if electronics_df.shape[0] > 0:
            electronics_df.to_json(os.path.join(product_path, 'product_electronics' ,data_file), compression='gzip', orient='records',
                               lines=True)

        count_files += 1
        print('{} out of {} files done'.format(count_files, len(data_files)))


def thread_function(name):
    logging.info("Thread %s: starting", name)
    time.sleep(2)
    logging.info("Thread %s: finishing", name)

if __name__ == "__main__":
    format = "%(asctime)s: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO,
                        datefmt="%H:%M:%S")

    logging.info("Main    : before creating thread")
    x = threading.Thread(target=thread_function, args=(1,))
    logging.info("Main    : before running thread")
    x.start()
    logging.info("Main    : wait for the thread to finish")
    # x.join()
    logging.info("Main    : all done")

    # run functions
    #clean_clusters()
    get_keywords()
    keyword_search(cleaned_top100_path)
    keyword_search(cleaned_min3_path)