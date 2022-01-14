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

noisy_clusters = [6426, 6050, 3978, 6430, 5022, 6051, 6428, 4164, 6424, 4217, 6425, 6421, 6429, 320, 6427, 4846, 5615, 2232, 5311, 6432, 4771, 1742, 5104, 3, 7, 8 ]
noisy_clusters.extend([35, 44, 42, 52, 63, 78, 66, 67, 102, 108, 120,129, 133, 143, 153, 154, 162, 169, 174, 180, 200, 202, 203, 206, 227, 228, 229, 231, 242, 244, 255, 263, 272, 278, 280, 324, 336, 352, 353, 372, 375, 408, 413, 444, 451, 453, 454, 458, 470, 476, 488, 489, 493, 496, 499, 504 ])
noisy_clusters.extend([26, 320, 436, 437])
noisy_clusters.extend([80, 95])
noisy_clusters.extend([6432,6430, 6429, 6428, 6427, 6426, 6425, 6424, 6423, 6422, 6421, 6397, 6396, 6393, 6381, 6365, 6360, 6351, 6097, 6094, 6092, 6086, 6078, 6049, 6048, 6004, 5998, 5909, 5822, 5616, 5615, 5571, 5564, 5563, 5562, 5508, 5410, 5311, 5289, 5279, 5278, 5269, 5418, 5417, 5104, 5022, 4925, 4846, 4843, 4790, 4789, 4779, 4774, 4771, 4710, 4234, 4230, 4217, 4198, 4164, 4048, 4033])
noisy_clusters.extend([514, 524, 526, 530, 532, 537, 547, 552, 557, 559, 583, 597, 601, 627, 628, 630, 632, 653, 664, 685, 690, 695, 707, 711, 743, 745, 751, 758, 775, 800, 802, 807, 808, 812, 866, 868, 872, 878, 881, 888, 911, 924, 931, 933, 935, 968, 998, 1002, 1006, 1028, 1066, 1071, 1075, 1082, 1094, 1097, 1099, 1109, 1128, 1133, 1142, 1143, 1144, 1159, 1160, 1174, 1181, 1184, 1189, 1193, 1195, 1207, 1211, 1214, 1218, 1221, 1244, 1245, 1259, 1261, 1266, 1274, 1276, 1277, 1286,1290, 1291, 1292, 1311, 1312, 1313, 1321, 1330, 1333, 1334, 1337, 1342, 1362, 1371, 1385, 1388, 1391, 1392, 1399, 1407, 1416, 1424, 1437, 1438, 1441, 1460, 1463, 1469, 1471, 1490, 1495, 1498, 1500, 1501, 1511, 1515, 1520, 1525, 1529, 1538,  1544, 1550, 1557, 1560, 1562, 1568, 1572, 1580, 1585, 1605, 1607, 1617, 1618, 1622, 1634, 1636, 1709, 1785, 1789, 1803, 1807, 1816, 1827, 1834, 1840, 1846, 1849, 1855, 1856, 1862, 1864, 1865, 1867, 1886, 1889, 1922, 1926, 1933, 1936, 1951, 1958, 1959, 1960, 1968, 1972, 1985, 1986, 1991, 1994, 1996, 2003, 2004, 769, 773, 823, 835, 839, 950, 658, 897, 1735, 1736, 1742, 1766, 1783, 774])
noisy_clusters.extend([2009, 2016, 2068, 2070, 2071, 2121, 2123, 2132, 2144, 2148, 2162, 2174, 2188, 2193, 2199, 2202, 2232, 2233, 2247, 2254, 2255, 2256, 2257, 2269, 2274, 2277, 2287, 2289, 2293, 2309,2324, 2332, 2354, 2403, 2416, 2417, 2431, 2434, 2444, 2446, 2485, 2489, 2500, 2506, 2599, 2600, 2605, 2611, 2616, 2618, 2637, 2654, 2669, 2678, 2679, 2681, 2687, 2709, 2710, 2713, 2717, 2729, 2737, 2740, 2751, 2754, 2786, 2787, 2789, 2791, 2802, 2807, 2808, 2809, 2821, 2826, 2846, 2875, 2876, 2878, 2880, 2887, 2920, 2922, 2925, 2930, 2965, 2973, 2976, 2980, 2986,  3004])
noisy_clusters.extend([3979, 3972, 3970, 3967, 3965, 3952, 3951, 3938, 3924, 3843, 3795, 3787, 3708, 3698, 3640, 3606, 3580, 3549, 3548, 3533, 3528, 3517, 3516, 3515, 3514, 3511, 3498, 3495, 3493, 3491, 3490, 3488, 3487, 3482, 3481, 3477, 3476, 3472, 3462, 3459, 3458, 3445, 3440, 3429, 3419, 3413, 3391, 3374, 3361, 3352, 3349, 3348, 3347, 3332, 3317, 3314, 3311, 3308, 3306, 3304, 3301, 3298, 3289, 3288, 3281, 3280, 3274, 3272, 3259, 3230, 3221, 3215, 3175, 3163, 3139, 3133, 3110, 3100, 3091, 3081, 3080, 3061, 3059, 3058, 3056, 3053, 3051, 3047, 3042, 3041, 3035, 3034, 3032, 3031, 3029, 3022, 3004])

allData = allData.loc[~allData['cluster_id'].isin(noisy_clusters)]
allData['cluster_id_mapped'] = allData.groupby('cluster_id').ngroup()

trainData = allData.loc[allData['origin'].isin(trainData['origin'])]
testData = allData.loc[~allData['origin'].isin(trainData['origin'])]

trainData.to_csv(r'../../../src/data/LocalBusiness/Splitting_12.20/Train_Test/train.csv')
testData.to_csv(r'../../../src/data/LocalBusiness/Splitting_12.20/Train_Test/test.csv')

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




