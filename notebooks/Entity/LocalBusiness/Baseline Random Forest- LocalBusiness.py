#!/usr/bin/env python
# coding: utf-8

# # This notebook will give a first baseline estimation for the matching of entities via a random forest algorithm as multi-class classification

# In[1]:


import os
import pandas as pd
import gzip
import json
import numpy as np
import nltk
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# In[2]:


trainPath = r'../../../src/data/LocalBusiness/Splitting_12.20/Train_Test/train tables' + '/'
testPath = r'../../../src/data/LocalBusiness/Splitting_12.20/Train_Test/test tables' + '/'
trainTables = os.listdir(trainPath)
testTables = os.listdir(testPath)
LBData = []

for table in trainTables:
    if table != '.ipynb_checkpoints':
        with gzip.open(trainPath + table, 'r') as dataFile:
            for line in dataFile:
                lineData = json.loads(line.decode('utf-8'))
                lineData['origin'] = table
                LBData.append(lineData)
trainData = pd.DataFrame(LBData)

LBData = []
for table in testTables:
    if table != '.ipynb_checkpoints':
        with gzip.open(testPath + table, 'r') as dataFile:
            for line in dataFile:
                lineData = json.loads(line.decode('utf-8'))
                lineData['origin'] = table
                LBData.append(lineData)
testData = pd.DataFrame(LBData)   


# In[3]:


columns = ['name', 'description']
trainData['concat'] = trainData[columns].astype(str).agg(' '.join, axis=1)
testData['concat'] = testData[columns].astype(str).agg(' '.join, axis=1)
trainData = trainData[['concat', 'cluster_id', 'origin']]
testData = testData[['concat', 'cluster_id', 'origin']]
trainData = trainData.loc[trainData['cluster_id'] > -1]
testData = testData.loc[testData['cluster_id'] > -1]


# In[4]:


frames = [trainData, testData]
allData = pd.concat(frames)
allData['cluster_id_mapped'] = allData.groupby('cluster_id').ngroup()

trainData = allData.loc[allData['origin'].isin(trainData['origin'])]
testData = allData.loc[~allData['origin'].isin(trainData['origin'])]


# In[5]:


def remove_stopwords(token_vector, stopwords_list):
    return token_vector.apply(lambda token_list: [word for word in token_list if word not in stopwords_list])


# In[6]:


def remove_punctuation(token_vector):
    return token_vector.apply(lambda token_list: [word for word in token_list if word not in string.punctuation])


# In[7]:


#clusters = data.groupby(['telephoneNorm']).size().reset_index(name='counts').sort_values('counts', ascending=False)
#clusters = clusters.loc[clusters['counts'] > 1]
#clusteredData = data[data['telephoneNorm'].isin(clusters['telephoneNorm'])]
#clusteredData['ClusterID'] = clusteredData.groupby('telephoneNorm').ngroup()
#columns = ['name', 'addressregion', 'streetaddress', 'addresslocality', 'addresscountry', 'longitude', 'latitude']
#clusteredData['concat'] = clusteredData[columns].astype(str).agg(' '.join, axis=1)


# In[7]:


allData = allData.rename(columns={'origin': 'originalSource'})


# In[78]:


clusteredData = allData
clusteredData


# ### Combine tf-idf and tf vector based features

# In[79]:


#clean concated description column to use tf-idf 
clusteredData['concat'] = clusteredData['concat'].apply(lambda row: row.lower())
clusteredData['tokens'] = clusteredData['concat'].apply(lambda row: word_tokenize(row))
clusteredData['tokens'] = remove_stopwords(clusteredData['tokens'],stopwords.words())
clusteredData['tokens'] = remove_punctuation (clusteredData['tokens'])
clusteredData.drop(columns=['concat'],inplace=True)
clusteredData = clusteredData[['tokens','cluster_id_mapped', 'originalSource']]


# In[80]:


clusteredData


# In[81]:


#define vectorizer to match preprocessed tokes for term frequency
def dummy(doc):
    return doc

vectorizer  = CountVectorizer(
        tokenizer=dummy,
        preprocessor=dummy,
        max_features=5000)  
tf_value = vectorizer.fit_transform(clusteredData['tokens'])


# In[82]:


#define vectorizer to match preprocessed tokes
def dummy_fun(doc):
    return doc

tfidf = TfidfVectorizer(
    analyzer='word',
    tokenizer=dummy_fun,
    preprocessor=dummy_fun,
    token_pattern=None,
    max_features=12500)  
tfidf_value = tfidf.fit_transform(clusteredData['tokens'])


# In[83]:


df_tf = pd.DataFrame(tf_value.toarray(), columns=vectorizer.get_feature_names())
df_tfidf = pd.DataFrame(tfidf_value.toarray(), columns=tfidf.get_feature_names())
df_prepared = pd.concat([clusteredData.reset_index(), df_tfidf], axis=1)


# In[84]:


y = df_prepared[['cluster_id_mapped', 'originalSource']]
df_prepared.drop(columns=['tokens','cluster_id_mapped'], inplace=True)


# In[85]:


y


# In[86]:


y_train = y.loc[y['originalSource'].isin(trainData['origin'])]
y_test = y.loc[~y['originalSource'].isin(trainData['origin'])]
y_train = y_train['cluster_id_mapped']
y_test = y_test['cluster_id_mapped']

x_train = df_prepared.loc[df_prepared['originalSource'].isin(trainData['origin'])]
x_test = df_prepared.loc[~df_prepared['originalSource'].isin(trainData['origin'])]
x_train= x_train.drop(columns=['index', 'originalSource'])
x_test= x_test.drop(columns=['index', 'originalSource'])


# In[87]:


y_train


# In[88]:
print('Now the model:')

# Baseline random forest
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
prediction = rf.predict(x_test) 
f1_mic = f1_score(y_test,prediction,average='micro')
f1_mac = f1_score(y_test,prediction,average='macro')
accuracy = accuracy_score(y_test,prediction) 
precision = precision_score(y_test,prediction,average='micro') 
recall = recall_score(y_test,prediction,average='micro') 
precision_mac = precision_score(y_test,prediction,average='macro') 
recall_mac = recall_score(y_test,prediction,average='macro') 
print("The F1-Score micro on test set: {:.4f}".format(f1_mic))
print("The F1-Score macro on test set: {:.4f}".format(f1_mac))
print("The Precision on test set: {:.4f}".format(precision))
print("The Recall on test set: {:.4f}".format(recall))
print("The Precision macro on test set: {:.4f}".format(precision_mac))
print("The Recall macro on test set: {:.4f}".format(recall_mac))
print("The Accuracy-Score on test set: {:.4f}".format(accuracy))


# In[ ]:




