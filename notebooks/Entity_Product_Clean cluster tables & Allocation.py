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
import requests
import multiprocessing
import time

import nltk
from nltk.corpus import stopwords
import string
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

def thread_function(name):
    logging.info("Thread %s: starting", name)
    time.sleep(2)
    logging.info("Thread %s: finishing", name)

"""
session = None

def set_global_session():
    global session
    if not session:
        session = requests.Session()

def download_site(url):
    with session.get(url) as response:
        name = multiprocessing.current_process().name
        print(f"{name}:Read {len(response.content)} from {url}")

def download_all_sites(sites):
    with multiprocessing.Pool(initializer=set_global_session) as pool:
        pool.map(download_site, sites)
"""


path_parent = os.path.dirname(os.getcwd())
product_path = os.path.join(path_parent, 'src/data/product')

cleaned_top100_path = os.path.join(product_path, 'product_top100/cleaned')
cleaned_min3_path = os.path.join(product_path, 'product_minimum3/cleaned')

cluster_path = os.path.join(product_path, 'lspc2020_to_tablecorpus/Cleaned')
notebook_path = os.path.join(path_parent,'notebooks')

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

    # only top 10
    clothes_top10 = []

    brands_dict = {'clothes': clothes_list, 'electronics1':electronics_list, 'electronics2':electronics_list2,
                   'electronics_total':list(set(electronics_list + electronics_list2))}

    with open(os.path.join(product_path, 'brands_dict.json'), 'w', encoding='utf-8') as f:
        json.dump(brands_dict, f)

    print('getting keywords done')

    return brands_dict

def get_bike_keywords():
    print('get keywords')
    with open(os.path.join(product_path, 'brands_dict.json'), 'r', encoding='utf-8') as f:
        brands_dict = json.load(f)

    # search for clothes brands top100
    bikes_html = urlopen('https://bikesreviewed.com/brands/')
    bikes_bsObj = BeautifulSoup(bikes_html.read(), 'lxml')
    bikes_lines = bikes_bsObj.find_all('h3')
    bikes_list = []
    for bikes_line in bikes_lines:
        if len(bikes_line.get_text().split('. ')) > 1:
            bikes_brand = bikes_line.get_text().split('. ')[1].lower()
        else:
            bikes_brand = bikes_line.get_text().lower()
        bikes_list.append(bikes_brand)

    bikes2_html = urlopen('https://www.globalbrandsmagazine.com/top-bicycle-brands-in-the-world-2020/')
    bikes2_bsObj = BeautifulSoup(bikes2_html.read(), 'lxml')
    bikes2_lines = bikes2_bsObj.find_all('h3')
    for bikes2_line in bikes2_lines:
        bikes2_brand = bikes2_line.find('a').get_text().lower()
        if bikes2_brand != 'lifestyle':
            bikes_list.append(bikes2_brand)

    bikes_list = list(set(bikes_list))
    brands_dict['bikes'] = bikes_list

    with open(os.path.join(product_path, 'brands_dict.json'), 'w', encoding='utf-8') as f:
        json.dump(brands_dict, f)


def clean_keywords():

    print('clean keywords')
    with open(os.path.join(product_path, 'brands_dict.json'), 'r', encoding='utf-8') as f:
        brands_dict = json.load(f)

    brands_dict['clothes_cleaned'] = ['prada','calvin klein','louis vuitton','under armour','the north face',
                                      'tommy hilfiger','dolce & gabbana','adidas','puma','oakley','dior','chanel','gap',
                                      'gucci','michael kors','patagonia','moncler','armani','burberry','nike']
    brands_dict['electronics_cleaned'] = ['lenovo','canon','hitachi','resonant','sony','nvidia','nintendo','apple',
                                          'samsung','yaskawa','asus','dell','hp','amd','nikon','xiaomi','cisco',
                                          'panasonic','intel','flex']

    with open(os.path.join(product_path, 'brands_dict.json'), 'w', encoding='utf-8') as f:
        json.dump(brands_dict, f)


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
    #brands_dict['clothes_cleaned'].append('nejron')  ##
    #brands_dict['electronics_cleaned'].append('arip santoso')  ##

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

    if os.path.isfile(os.path.join(product_path,'product_bikes', 'bikes_dict.json')):
        with open(os.path.join(product_path,'product_bikes', 'bikes_dict.json'), 'r', encoding='utf-8') as f:
            bikes_dict = json.load(f)
    else:
        bikes_dict = {'top100/cleaned':{key: [] for key in brands_dict['bikes']},
                      'minimum3/cleaned':{key: [] for key in brands_dict['bikes']}}

    count = 0
    with progressbar.ProgressBar(max_value=len(data_files)) as bar:
        for data_file in data_files:
            #if data_file == 'Product_3dcartstores.com_September2020.json.gz': ## for testing
            df = pd.read_json(os.path.join(data_path, '{}'.format(data_file)), compression='gzip', lines=True)

            clothes_row_ids = []
            electronics_row_ids = []
            bikes_row_ids = []

            # iterrate over rows and look for keywords
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
                        elif cell in brands_dict['bikes']:
                            bikes_dict[entity][cell].append((data_file, row_id))
                            bikes_row_ids.append(row_id)
            elif 'name' in df.columns: # if column 'brand' does not exist check for first word in name column
                df['brand'] = ''
                # iterrate over rows
                for i in range(df.shape[0]):
                    row_id = int(df['row_id'][i])
                    if df['name'][i] != None:
                        name_split_list = str(df['name'][i]).split(' ')

                        # check for first word in name column
                        cell = str(name_split_list[0]).lower()
                        if cell in brands_dict['electronics_total']:
                            electronics_dict[entity][cell].append((data_file, row_id))
                            electronics_row_ids.append(row_id)
                            df.at[i,'brand'] = cell
                        elif cell in brands_dict['clothes']:
                            clothes_dict[entity][cell].append((data_file, row_id))
                            clothes_row_ids.append(row_id)
                            df.at[i,'brand'] = cell
                        elif cell in brands_dict['bikes']:
                            bikes_dict[entity][cell].append((data_file, row_id))
                            bikes_row_ids.append(row_id)
                            df.at[i,'brand'] = cell
                        elif len(name_split_list)>1:
                            # check for two words (since ngrams brands)
                            cell = cell + ' ' + str(name_split_list[1]).lower()
                            if cell in brands_dict['electronics_total']:
                                electronics_dict[entity][cell].append((data_file, row_id))
                                electronics_row_ids.append(row_id)
                                df.at[i, 'brand'] = cell
                            elif cell in brands_dict['clothes']:
                                clothes_dict[entity][cell].append((data_file, row_id))
                                clothes_row_ids.append(row_id)
                                df.at[i,'brand'] = cell
                            elif cell in brands_dict['bikes']:
                                bikes_dict[entity][cell].append((data_file, row_id))
                                bikes_row_ids.append(row_id)
                                df.at[i, 'brand'] = cell
                            elif len(name_split_list)>2:
                                # check for three words (since ngrams brands)
                                cell = cell + ' ' + str(name_split_list[2]).lower()
                                if cell in brands_dict['electronics_total']:
                                    electronics_dict[entity][cell].append((data_file, row_id))
                                    electronics_row_ids.append(row_id)
                                    df.at[i, 'brand'] = cell
                                elif cell in brands_dict['clothes']:
                                    clothes_dict[entity][cell].append((data_file, row_id))
                                    clothes_row_ids.append(row_id)
                                    df.at[i,'brand'] = cell
                                elif cell in brands_dict['bikes']:
                                    bikes_dict[entity][cell].append((data_file, row_id))
                                    bikes_row_ids.append(row_id)
                                    df.at[i, 'brand'] = cell

                count += 1
                bar.update(count)

                # write selected data into seperate folders
                clothes_df = df[df['row_id'].isin(clothes_row_ids)]
                electronics_df = df[df['row_id'].isin(electronics_row_ids)]
                bikes_df = df[df['row_id'].isin(bikes_row_ids)]

                if clothes_df.shape[0] > 0:
                    clothes_df.to_json(os.path.join(product_path, 'product_clothes', data_file), compression='gzip',
                                       orient='records',
                                       lines=True)

                if electronics_df.shape[0] > 0:
                    electronics_df.to_json(os.path.join(product_path, 'product_electronics', data_file),
                                           compression='gzip', orient='records',
                                           lines=True)

                if bikes_df.shape[0] > 0:
                    bikes_df.to_json(os.path.join(product_path, 'product_bikes', data_file),
                                           compression='gzip', orient='records',
                                           lines=True)

                ## nur alle paar tausend saven
                # save dictionaries with selected data
                if count % 1000 == 0:
                    with open(os.path.join(product_path,'product_clothes', 'clothes_dict.json'), 'w', encoding='utf-8') as f:
                        json.dump(clothes_dict, f)

                    with open(os.path.join(product_path,'product_electronics', 'electronics_dict.json'), 'w', encoding='utf-8') as f:
                        json.dump(electronics_dict, f)

                    with open(os.path.join(product_path,'product_bikes', 'bikes_dict.json'), 'w', encoding='utf-8') as f:
                        json.dump(bikes_dict, f)

    # save at the end of running
    with open(os.path.join(product_path, 'product_clothes', 'clothes_dict.json'), 'w', encoding='utf-8') as f:
        json.dump(clothes_dict, f)

    with open(os.path.join(product_path, 'product_electronics', 'electronics_dict.json'), 'w', encoding='utf-8') as f:
        json.dump(electronics_dict, f)

    with open(os.path.join(product_path, 'product_bikes', 'bikes_dict.json'), 'w', encoding='utf-8') as f:
        json.dump(bikes_dict, f)

def remove_stopwords(token_vector, stopwords_list):
    return token_vector.apply(lambda token_list: [word for word in token_list if word not in stopwords_list])

def remove_punctuation(token_vector):
    return token_vector.apply(lambda token_list: [word for word in token_list if word not in string.punctuation])

def jaccard_similarity_score(original, translation):
    intersect = set(original).intersection(set(translation))
    union = set(original).union(set(translation))
    try:
        return len(intersect) / len(union)
    except ZeroDivisionError:
        return 0

def post_cleaning():
    """
    Measures the similarity within a cluster_id of our final electronics and clothes entities and removes ...??
    Post-processing.
    :return:
    """
    # read final dataframes with all cluster_ids left for electronics and clothes
    electronics_clusters_all_15_df = pd.read_csv(os.path.join(cluster_path, 'electronics_clusters_all_10_tables.csv'), index_col=None)
    clothes_clusters_all_10_df = pd.read_csv(os.path.join(cluster_path, 'clothes_clusters_all_8_tables_v2.csv'), index_col=None)

    # generate lists for final cluster_ids for electronics and clothes
    electronics_final_entities_df = pd.read_csv(os.path.join(notebook_path, 'electronics10.csv'),index_col=None)
    electronics_final_entities_list = electronics_final_entities_df['cluster_id']

    clothes_final_entities_df = pd.read_csv(os.path.join(notebook_path, 'clothes8.csv'),index_col=None)
    clothes_final_entities_list = clothes_final_entities_df['cluster_id']

    # generate lists for valid electronics and clothes brands
    with open(os.path.join(product_path, 'brands_dict.json'), 'r', encoding='utf-8') as f:
        brands_dict = json.load(f)

    electronics_valid_brands = brands_dict['electronics_total']
    clothes_valid_brands = brands_dict['clothes']

    # lowercase name column for similarity measure
    electronics_clusters_all_15_df['name'] = electronics_clusters_all_15_df['name'].apply(lambda row: str(row).lower())
    clothes_clusters_all_10_df['name'] = clothes_clusters_all_10_df['name'].apply(lambda row: str(row).lower())

    # use tokenizer for name column to get tokens for training the model, remove stopwords and punctuation
    electronics_clusters_all_15_df['tokens'] = electronics_clusters_all_15_df['name'].apply(lambda row: word_tokenize(row))
    electronics_clusters_all_15_df['tokens'] = remove_stopwords(electronics_clusters_all_15_df['tokens'], stopwords.words())
    electronics_clusters_all_15_df['tokens'] = remove_punctuation(electronics_clusters_all_15_df['tokens'])

    clothes_clusters_all_10_df['tokens'] = clothes_clusters_all_10_df['name'].apply(lambda row: word_tokenize(row))
    clothes_clusters_all_10_df['tokens'] = remove_stopwords(clothes_clusters_all_10_df['tokens'],stopwords.words())
    clothes_clusters_all_10_df['tokens'] = remove_punctuation(clothes_clusters_all_10_df['tokens'])

    # get tagged words
    tagged_data_electronics = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(electronics_clusters_all_15_df['tokens'])]
    tagged_data_clothes = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(clothes_clusters_all_10_df['tokens'])]


    """
    # build model and vocabulary for electronics (do same for clothes later)
    model_electronics = Doc2Vec(vector_size=50, min_count=5, epochs=25, dm=0)
    model_electronics.build_vocab(tagged_data_electronics)
    # Train model
    model_electronics.train(tagged_data_electronics, total_examples=model_electronics.corpus_count, epochs=25)

    # compare for all cluster_ids the similarity between the entries within a cluster_id
    ## for electronics:
    valid_indices_all = []
    print('measure similarity for electronics')
    count = 0
    with progressbar.ProgressBar(max_value=len(electronics_final_entities_list)) as bar:
        for cluster_id in electronics_final_entities_list:
            electronics_single_cluster_id_df = electronics_clusters_all_15_df[electronics_clusters_all_15_df['cluster_id']==cluster_id]

            # measure similarity with Doc2Vec
            valid_brands = list(filter(lambda brand: brand in electronics_valid_brands,
                                       electronics_single_cluster_id_df['brand'].apply(lambda element: str(element).lower())))
            if len(valid_brands) > 0:
                most_common_brand = max(valid_brands, key=valid_brands.count)
                index_most_common = electronics_single_cluster_id_df[electronics_single_cluster_id_df['brand'].apply(
                    lambda element: str(element).lower()) == most_common_brand].index[0] # use this as baseline for similarity comparisons within a certain cluster

                # calculate similarity and filter for the ones which are in the current cluster
                similar_doc = model_electronics.docvecs.most_similar(f'{index_most_common}', topn=electronics_clusters_all_15_df.shape[0])
                similar_doc_cluster = [tup for tup in similar_doc if int(tup[0]) in list(electronics_single_cluster_id_df.index)] # similarities as tuples with index and similarity measure compared to baseline product
                similar_doc_cluster_df = pd.DataFrame(list(similar_doc_cluster), columns=['index','doc2vec'])
                similar_doc_cluster_df['index'] = [int(i) for i in similar_doc_cluster_df['index']] # change indices to numbers

                # measure similarity with Jaccard
                jaccard_score = electronics_single_cluster_id_df['name'].apply(lambda row: jaccard_similarity_score(
                    row,electronics_single_cluster_id_df['name'].loc[int(index_most_common)]))
                jaccard_score = jaccard_score.drop(int(index_most_common)).sort_values(ascending=False)
                jaccard_score_df = pd.DataFrame({'index':jaccard_score.index, 'jaccard':jaccard_score.values})

                # merge both similarity measures to one dataframe
                similarity_df = pd.merge(similar_doc_cluster_df, jaccard_score_df, left_on='index', right_on='index', how='left')

                # select valid cluster_ids by setting thresholds for doc2vec and jaccard similarities
                valid_cluster_id_df = similarity_df[(similarity_df['doc2vec']>0.97) | (similarity_df['jaccard']>0.5)]
                valid_cluster_id_indices = valid_cluster_id_df['index'].to_list() # list of valid indices within a cluster_id

                # creat new dataframe within cluster_id with selected indices
                #electronics_single_cluster_id_df_new = electronics_single_cluster_id_df[
                 #   electronics_single_cluster_id_df.index.isin(valid_cluster_id_indices)]

                valid_indices_all += valid_cluster_id_indices

            count += 1
            bar.update(count)

    electronics_clusters_all_15_df_new = electronics_clusters_all_15_df[
        electronics_clusters_all_15_df.index.isin(valid_indices_all)]

    electronics_clusters_all_15_df_new.to_csv(os.path.join(cluster_path,
                                                           'electronics_clusters_all_10_tables_post_processed.csv'),
                                              columns=None)

    ### Auch gleiche table_ids rauswerfen!!!!
    """

    ## for clothes:
    # build model and vocabulary for clothes (do same for clothes later)
    model_clothes = Doc2Vec(vector_size=50, min_count=5, epochs=25, dm=0)
    model_clothes.build_vocab(tagged_data_clothes)
    # Train model
    model_clothes.train(tagged_data_clothes, total_examples=model_clothes.corpus_count, epochs=25)

    valid_indices_all = []
    print('measure similarity for clothes')
    count = 0
    with progressbar.ProgressBar(max_value=len(clothes_final_entities_list)) as bar:
        for cluster_id in clothes_final_entities_list:
            clothes_single_cluster_id_df = clothes_clusters_all_10_df[
                clothes_clusters_all_10_df['cluster_id'] == cluster_id]

            # measure similarity with Doc2Vec
            valid_brands = list(filter(lambda brand: brand in clothes_valid_brands,
                                       clothes_single_cluster_id_df['brand_y'].apply(
                                           lambda element: str(element).lower())))
            if len(valid_brands) > 0:
                most_common_brand = max(valid_brands, key=valid_brands.count)
                index_most_common = clothes_single_cluster_id_df[clothes_single_cluster_id_df['brand_y'].apply(
                    lambda element: str(element).lower()) == most_common_brand].index[
                    0]  # use this as baseline for similarity comparisons within a certain cluster

                # calculate similarity and filter for the ones which are in the current cluster
                similar_doc = model_clothes.docvecs.most_similar(f'{index_most_common}',
                                                                     topn=clothes_clusters_all_10_df.shape[0])
                similar_doc_cluster = [tup for tup in similar_doc if int(tup[0]) in list(
                    clothes_single_cluster_id_df.index)]  # similarities as tuples with index and similarity measure compared to baseline product
                similar_doc_cluster_df = pd.DataFrame(list(similar_doc_cluster), columns=['index', 'doc2vec'])
                similar_doc_cluster_df['index'] = [int(i) for i in
                                                   similar_doc_cluster_df['index']]  # change indices to numbers

                # measure similarity with Jaccard
                jaccard_score = clothes_single_cluster_id_df['name'].apply(lambda row: jaccard_similarity_score(
                    row, clothes_single_cluster_id_df['name'].loc[int(index_most_common)]))
                jaccard_score = jaccard_score.drop(int(index_most_common)).sort_values(ascending=False)
                jaccard_score_df = pd.DataFrame({'index': jaccard_score.index, 'jaccard': jaccard_score.values})

                # merge both similarity measures to one dataframe
                similarity_df = pd.merge(similar_doc_cluster_df, jaccard_score_df, left_on='index', right_on='index',
                                         how='left')

                # select valid cluster_ids by setting thresholds for doc2vec and jaccard similarities
                valid_cluster_id_df = similarity_df[(similarity_df['doc2vec'] > 0.97) | (similarity_df['jaccard'] > 0.6)]
                valid_cluster_id_indices = valid_cluster_id_df[
                    'index'].to_list()  # list of valid indices within a cluster_id

                # creat new dataframe within cluster_id with selected indices
                # clothes_single_cluster_id_df_new = clothes_single_cluster_id_df[
                #   clothes_single_cluster_id_df.index.isin(valid_cluster_id_indices)]

                valid_indices_all += valid_cluster_id_indices

            count += 1
            bar.update(count)

    clothes_clusters_all_10_df_new = clothes_clusters_all_10_df[
        clothes_clusters_all_10_df.index.isin(valid_indices_all)]

    clothes_clusters_all_10_df_new.to_csv(os.path.join(cluster_path,
                                                       'clothes_clusters_all_8_tables_post_processed.csv'),
                                          columns=None)


if __name__ == "__main__":

    # for multithreading
    os.environ['NUMEXPR_MAX_THREADS'] = '24'
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


    """
    # for multiprocessing
    sites = [
                "https://www.jython.org",
                "http://olympus.realpython.org/dice",
            ] * 80
    start_time = time.time()
    download_all_sites(sites)
    duration = time.time() - start_time
    print(f"Downloaded {len(sites)} in {duration} seconds")
    """

    # run functions
    #clean_clusters()
    #get_keywords() ##
    #clean_keywords()
    keyword_search(cleaned_top100_path)
    keyword_search(cleaned_min3_path)
    #post_cleaning()
    #get_bike_keywords()

    test = 2