import os
import pandas as pd
import plotly.express as px
import progressbar
import json
import numpy as np
import nltk
from nltk.corpus import stopwords
import string
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
pd.set_option('display.max_colwidth', None)

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

zip_files_mapping = [file for file in os.listdir(mapping_corpus_path_2) if file.endswith('.json.gz')]
zip_files_tables = [file for file in os.listdir(table_corpus_path) if file.endswith('.json.gz')]

df_joined_clothes = pd.read_csv(os.path.join(mapping_corpus_path_2, 'Clothes_clusters_all_8_tables_post_processed_lower_threshold.csv'))
df_joined_clothes['category']='clothes'
df_joined_electronics = pd.read_csv(os.path.join(mapping_corpus_path_2, 'Electronics_clusters_all_8_tables_post_processed_lower_threshold.csv'))
df_joined_electronics['category']='electronics'
df_joined_bikes = pd.read_csv(os.path.join(mapping_corpus_path_2, 'Bikes_clusters_all_8_tables_post_processed_lower_threshold.csv'))
df_joined_bikes['category']='bikes'
df_joined_cars = pd.read_csv(os.path.join(mapping_corpus_path_2, 'Cars_clusters_all_8_tables_post_processed_lower_threshold.csv'))
df_joined_cars['category']='cars'
df_joined_technology = pd.read_csv(os.path.join(mapping_corpus_path_2, 'Technology_clusters_all_8_tables_post_processed_lower_threshold.csv'))
df_joined_technology['category']='technology'
df_joined_drugstore = pd.read_csv(os.path.join(mapping_corpus_path_2, 'Drugstore_clusters_all_8_tables_post_processed_lower_threshold.csv'))
df_joined_drugstore['category']='drugstore'
df_joined_tools = pd.read_csv(os.path.join(mapping_corpus_path_2, 'Tools_clusters_all_8_tables_post_processed_lower_threshold.csv'))
df_joined_tools['category']='tools'
df_joined_random = pd.read_csv(os.path.join(mapping_corpus_path_2, 'Random_clusters_all_8_tables_post_processed_lower_threshold.csv'))
df_joined_random['category']='random'
frames = [df_joined_electronics, df_joined_clothes,df_joined_bikes,df_joined_cars,df_joined_technology,df_joined_drugstore,df_joined_tools,df_joined_random]
df_concat = pd.concat(frames).drop(columns = ['Unnamed: 0','Unnamed: 0.1','Valid'])
df_concat

df_grouped= df_concat.groupby('cluster_id').count()
# only look at clusters that have at least one brand associated
df_set = df_grouped.reset_index()[['cluster_id','table_id']].rename(columns={'table_id':'Amount'})
df_set=df_set[df_set['Amount']>7]
df_10=df_set[df_set['Amount']>15]
#clean product column and lowercase
df_concat=df_concat.dropna(subset = ['name'])
df_concat['name'] = df_concat['name'].apply(lambda row: row.lower())
df_concat
#get only cluster ids with at least one brand electronics
df_compare = df_concat[df_concat['cluster_id'].isin(df_set['cluster_id'].tolist())]
#merge with set to get amount of tables per cluster in overview
df_compare = df_compare.merge(df_set, left_on=['cluster_id'], right_on = ['cluster_id'], how='left')
#use tokenizer for product names to get tokes for training the model
df_compare['product_tokes'] = df_compare['name'].apply(lambda row: word_tokenize(row))
df_compare['product_tokes'] = remove_stopwords(df_compare['product_tokes'],stopwords.words())
df_compare['product_tokes'] = remove_punctuation (df_compare['product_tokes'])
#get tagged words
tagged_data = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(df_compare['product_tokes'])]
# build model and vocabulary
model = Doc2Vec(vector_size=50, min_count = 5, epochs = 25, dm = 0)
model.build_vocab(tagged_data)
# Train model
model.train(tagged_data, total_examples=model.corpus_count, epochs=25)
top_clusters_list = df_10['cluster_id'].tolist()
index_top_clusters_list=[]
for id in top_clusters_list:
    index_top_clusters_list.append(df_compare[df_compare['cluster_id']==id].index[0])
count = 0
clusters_search=[]
with progressbar.ProgressBar(max_value=len(index_top_clusters_list)) as bar:
    for i in index_top_clusters_list:
        similar_doc = model.docvecs.most_similar(f'{i}', topn = 20)
        clusters_search.append(int(i))
        for index, similarity in similar_doc:
            if df_compare.iloc[int(index)]['Amount']>7:
                if df_compare.iloc[int(index)]['category']==df_compare.iloc[int(i)]['category']:
                    clusters_search.append(int(index))
        jaccard_score = df_compare['product_tokes'].apply(lambda row: jaccard_similarity_score(row,df_compare.iloc[int(i)]['product_tokes']) )
        indizes=sorted(range(len(jaccard_score)), key=lambda i: jaccard_score[i])[-20:]
        for index in indizes:
            if df_compare.iloc[int(index)]['Amount']>7:
                if df_compare.iloc[int(index)]['category']==df_compare.iloc[int(i)]['category']:
                    clusters_search.append(int(index))
        count=count+1
        bar.update(count)
df_final = df_compare.iloc[clusters_search]
df_final.drop_duplicates('cluster_id', keep='first').to_excel("Final_lower threshold.xlsx")

df_joined_clothes = pd.read_csv(os.path.join(mapping_corpus_path_2, 'Clothes_clusters_all_8_tables_post_processed.csv'))
df_joined_clothes['category']='clothes'
df_joined_electronics = pd.read_csv(os.path.join(mapping_corpus_path_2, 'Electronics_clusters_all_8_tables_post_processed.csv'))
df_joined_electronics['category']='electronics'
df_joined_bikes = pd.read_csv(os.path.join(mapping_corpus_path_2, 'Bikes_clusters_all_8_tables_post_processed.csv'))
df_joined_bikes['category']='bikes'
df_joined_cars = pd.read_csv(os.path.join(mapping_corpus_path_2, 'Cars_clusters_all_8_tables_post_processed.csv'))
df_joined_cars['category']='cars'
df_joined_technology = pd.read_csv(os.path.join(mapping_corpus_path_2, 'Technology_clusters_all_8_tables_post_processed.csv'))
df_joined_technology['category']='technology'
df_joined_drugstore = pd.read_csv(os.path.join(mapping_corpus_path_2, 'Drugstore_clusters_all_8_tables_post_processed.csv'))
df_joined_drugstore['category']='drugstore'
df_joined_tools = pd.read_csv(os.path.join(mapping_corpus_path_2, 'Tools_clusters_all_8_tables_post_processed.csv'))
df_joined_tools['category']='tools'
df_joined_random = pd.read_csv(os.path.join(mapping_corpus_path_2, 'Random_clusters_all_8_tables_post_processed.csv'))
df_joined_random['category']='random'
frames = [df_joined_electronics, df_joined_clothes,df_joined_bikes,df_joined_cars,df_joined_technology,df_joined_drugstore,df_joined_tools,df_joined_random]
df_concat = pd.concat(frames).drop(columns = ['Unnamed: 0','Unnamed: 0.1','Valid'])
df_concat
df_grouped= df_concat.groupby('cluster_id').count()
# only look at clusters that have at least one brand associated
df_set = df_grouped.reset_index()[['cluster_id','table_id']].rename(columns={'table_id':'Amount'})
df_set=df_set[df_set['Amount']>7]
df_10=df_set[df_set['Amount']>15]
#clean product column and lowercase
df_concat=df_concat.dropna(subset = ['name'])
df_concat['name'] = df_concat['name'].apply(lambda row: row.lower())
df_concat
#get only cluster ids with at least one brand electronics
df_compare = df_concat[df_concat['cluster_id'].isin(df_set['cluster_id'].tolist())]
#merge with set to get amount of tables per cluster in overview
df_compare = df_compare.merge(df_set, left_on=['cluster_id'], right_on = ['cluster_id'], how='left')
#use tokenizer for product names to get tokes for training the model
df_compare['product_tokes'] = df_compare['name'].apply(lambda row: word_tokenize(row))
df_compare['product_tokes'] = remove_stopwords(df_compare['product_tokes'],stopwords.words())
df_compare['product_tokes'] = remove_punctuation (df_compare['product_tokes'])
#get tagged words
tagged_data = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(df_compare['product_tokes'])]
# build model and vocabulary
model = Doc2Vec(vector_size=50, min_count = 5, epochs = 25, dm = 0)
model.build_vocab(tagged_data)
# Train model
model.train(tagged_data, total_examples=model.corpus_count, epochs=25)
top_clusters_list = df_10['cluster_id'].tolist()
index_top_clusters_list=[]
for id in top_clusters_list:
    index_top_clusters_list.append(df_compare[df_compare['cluster_id']==id].index[0])

# get most similar products for each of the base clusters and save them if they have more than 5 tables
count = 0
clusters_search=[]
with progressbar.ProgressBar(max_value=len(index_top_clusters_list)) as bar:
    for i in index_top_clusters_list:
        similar_doc = model.docvecs.most_similar(f'{i}', topn = 20)
        clusters_search.append(int(i))
        for index, similarity in similar_doc:
            if df_compare.iloc[int(index)]['Amount']>7:
                if df_compare.iloc[int(index)]['category']==df_compare.iloc[int(i)]['category']:
                        clusters_search.append(int(index))
        jaccard_score = df_compare['product_tokes'].apply(lambda row: jaccard_similarity_score(row,df_compare.iloc[int(i)]['product_tokes']) )
        indizes=sorted(range(len(jaccard_score)), key=lambda i: jaccard_score[i])[-20:]
        for index in indizes:
             if df_compare.iloc[int(index)]['Amount']>7:
                    if df_compare.iloc[int(index)]['category']==df_compare.iloc[int(i)]['category']:
                        clusters_search.append(int(index))
        count=count+1
        bar.update(count)
df_final = df_compare.iloc[clusters_search]
df_final.drop_duplicates('cluster_id', keep='first').to_excel("Final_v3.xlsx")