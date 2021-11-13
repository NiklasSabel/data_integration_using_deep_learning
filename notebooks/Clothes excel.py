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

def remove_punctuation(token_vector):
    return token_vector.apply(lambda token_list: [word for word in token_list if word not in string.punctuation])


def remove_stopwords(token_vector, stopwords_list):
    return token_vector.apply(lambda token_list: [word for word in token_list if word not in stopwords_list])


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

zip_files_mapping = [file for file in os.listdir(mapping_corpus_path_2) if file.endswith('.json.gz')]
zip_files_tables = [file for file in os.listdir(table_corpus_path) if file.endswith('.json.gz')]

df_joined_clothes = pd.read_json(os.path.join(mapping_corpus_path_2, 'joined_clothes_v2'), compression='gzip', orient='records', lines=True)

df_grouped_clothes = df_joined_clothes.groupby('cluster_id').count()

# only look at clusters that have at least one brand associated
df_set_clothes = df_grouped_clothes[df_grouped_clothes['brand_y']>0].reset_index()[['cluster_id','table_id']].rename(columns={'table_id':'Amount'})

# We discard all clusters with less than 2 entries, cause we cannot match anything there, so 1,6 million clusters remain
df_set_clothes=df_set_clothes[df_set_clothes['Amount']>1]
df_10_clothes=df_set_clothes[df_set_clothes['Amount']>10]

df_joined_clothes=df_joined_clothes.dropna(subset = ['name'])
#clean product column and lowercase
df_joined_clothes['name'] = df_joined_clothes['name'].apply(lambda row: row.lower())
df_joined_clothes
#get only cluster ids with at least one brand electronics
df_compare_clothes = df_joined_clothes[df_joined_clothes['cluster_id'].isin(df_set_clothes['cluster_id'].tolist())]
#merge with set to get amount of tables per cluster in overview
df_compare_clothes = df_compare_clothes.merge(df_set_clothes, left_on=['cluster_id'], right_on = ['cluster_id'], how='left')

#use tokenizer for product names to get tokes for training the model
df_compare_clothes['product_tokes'] = df_compare_clothes['name'].apply(lambda row: word_tokenize(row))
df_compare_clothes['product_tokes'] = remove_stopwords(df_compare_clothes['product_tokes'],stopwords.words())
df_compare_clothes['product_tokes'] = remove_punctuation (df_compare_clothes['product_tokes'])
#get tagged words
tagged_data = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(df_compare_clothes['product_tokes'])]
# build model and vocabulary
model = Doc2Vec(vector_size=50, min_count = 5, epochs = 25, dm = 0)
model.build_vocab(tagged_data)
# Train model
model.train(tagged_data, total_examples=model.corpus_count, epochs=25)

#get cluster ids and with that indices of top products to use model
top_clusters_list = df_10_clothes['cluster_id'].tolist()
index_top_clusters_list=[]
for id in top_clusters_list:
    index_top_clusters_list.append(df_compare_clothes[df_compare_clothes['cluster_id']==id].index[0])

# get most similar products for each of the base clusters and save them if they have more than 5 tables
clothes_clusters_search=[]
count = 0
with progressbar.ProgressBar(max_value=len(index_top_clusters_list)) as bar:
    for i in index_top_clusters_list:
        similar_doc = model.docvecs.most_similar(f'{i}', topn = 20)
        clothes_clusters_search.append(int(i))
        for index, similarity in similar_doc:
            if df_compare_clothes.iloc[int(index)]['Amount']>12:
                clothes_clusters_search.append(int(index))
        jaccard_score = df_compare_clothes['product_tokes'].apply(lambda row: jaccard_similarity_score(row,df_compare_clothes.iloc[int(i)]['product_tokes']) )
        indizes=sorted(range(len(jaccard_score)), key=lambda i: jaccard_score[i])[-20:]
        for index in indizes:
            if df_compare_clothes.iloc[int(index)]['Amount']>12:
                clothes_clusters_search.append(int(index))
        count += 1
        bar.update(count)
df_clothes_final = df_compare_clothes.iloc[clothes_clusters_search]


df_clothes_final.drop_duplicates('cluster_id', keep='first').to_excel("Final_Clothes_v3.xlsx")


df_joined_electronics = pd.read_json(os.path.join(mapping_corpus_path_2, 'joined_electronics_v2'), compression='gzip', orient='records', lines=True)

df_grouped_electronics = df_joined_electronics.groupby('cluster_id').count()
# only look at clusters that have at least one brand associated
df_set_electronics = df_grouped_electronics[df_grouped_electronics['brand_y']>0].reset_index()[['cluster_id','table_id']].rename(columns={'table_id':'Amount'})
# We discard all clusters with less than 2 entries, cause we cannot match anything there, so 1,6 million clusters remain
df_set_electronics=df_set_electronics[df_set_electronics['Amount']>1]
df_15_electronics=df_set_electronics[df_set_electronics['Amount']>15]

#clean product column and lowercase
df_joined_electronics=df_joined_electronics.dropna(subset = ['name'])
df_joined_electronics['name'] = df_joined_electronics['name'].apply(lambda row: row.lower())
df_joined_electronics
#get only cluster ids with at least one brand electronics
df_compare_electronics = df_joined_electronics[df_joined_electronics['cluster_id'].isin(df_set_electronics['cluster_id'].tolist())]
#merge with set to get amount of tables per cluster in overview
df_compare_electronics = df_compare_electronics.merge(df_set_electronics, left_on=['cluster_id'], right_on = ['cluster_id'], how='left')

#use tokenizer for product names to get tokes for training the model
df_compare_electronics['product_tokes'] = df_compare_electronics['name'].apply(lambda row: word_tokenize(row))
df_compare_electronics['product_tokes'] = remove_stopwords(df_compare_electronics['product_tokes'],stopwords.words())
df_compare_electronics['product_tokes'] = remove_punctuation (df_compare_electronics['product_tokes'])
#get tagged words
tagged_data = [TaggedDocument(words=_d, tags=[str(i)]) for i, _d in enumerate(df_compare_electronics['product_tokes'])]
# build model and vocabulary
model = Doc2Vec(vector_size=50, min_count = 5, epochs = 25, dm = 0)
model.build_vocab(tagged_data)
# Train model
model.train(tagged_data, total_examples=model.corpus_count, epochs=25)

#get cluster ids for basline products and with that indices of top products to use model
#1524820,47566,6076,14418,28307,33570,39040,51314,99153,215254,685416, 984421 , 1808651,2887810,34506065,47841827,620473,56116,94055, 150211,182246, 516888, 562955
top_clusters_list = df_15_electronics['cluster_id'].tolist()
index_top_clusters_list=[]
for id in top_clusters_list:
    index_top_clusters_list.append(df_compare_electronics[df_compare_electronics['cluster_id']==id].index[0])


# get most similar products for each of the base clusters and save them if they have more than 5 tables
electronics_clusters_search=[]
count = 0
with progressbar.ProgressBar(max_value=len(index_top_clusters_list)) as bar:
    for i in index_top_clusters_list:
        similar_doc = model.docvecs.most_similar(f'{i}', topn = 20)
        electronics_clusters_search.append(int(i))
        for index, similarity in similar_doc:
            if df_compare_electronics.iloc[int(index)]['Amount']>12:
                electronics_clusters_search.append(int(index))
        jaccard_score = df_compare_electronics['product_tokes'].apply(lambda row: jaccard_similarity_score(row,df_compare_electronics.iloc[int(i)]['product_tokes']) )
        indizes=sorted(range(len(jaccard_score)), key=lambda i: jaccard_score[i])[-20:]
        for index in indizes:
            if df_compare_electronics.iloc[int(index)]['Amount']>12:
                electronics_clusters_search.append(int(index))
        count += 1
        bar.update(count)
df_electroncis_final = df_compare_electronics.iloc[electronics_clusters_search]

df_electroncis_final.drop_duplicates('cluster_id', keep='first').to_excel("Final_Electronics_v3.xlsx")