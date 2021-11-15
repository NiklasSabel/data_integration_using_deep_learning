#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!pip install rank-bm25
#from rank_bm25 import BM25Okapi
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
import os
from os import listdir
from os.path import isfile, join
import re
import numpy as np
from math import floor, ceil

import json
import gzip

from os import walk
from scipy.spatial import KDTree

# !pip install geopy
# !pip install phonenumbers
# !pip install pycountry

import geopy.distance
import phonenumbers
import pycountry


# In[3]:


pd.options.display.max_columns = 100


# In[4]:


path = r"../src/data/LocalBusiness"
file_path_min3 = path + r"/LocalBusiness_minimum3/geo_preprocessed"
file_path_top100 = path + r"/LocalBusiness_top100/geo_preprocessed"

files_min3 = os.listdir(file_path_min3)
files_top100 = os.listdir(file_path_top100)


# In[5]:


print(len(files_min3))
print(len(files_top100))


# In[6]:


LB_min3 = []

for lb in files_min3:
    with gzip.open(file_path_min3 + '/' + lb, 'r') as dataFile:
        for line in dataFile:
            lineData = json.loads(line.decode('utf-8'))
            LB_min3.append(lineData)
df_min3 = pd.DataFrame(LB_min3)


# In[7]:


len(df_min3)


# In[8]:


LB_top100 = []

for lb in files_top100:
    with gzip.open(file_path_top100 + '/' + lb, 'r') as dataFile:
        for line in dataFile:
            lineData = json.loads(line.decode('utf-8'))
            LB_top100.append(lineData)
df_top100 = pd.DataFrame(LB_top100)


# In[9]:


len(df_top100)


# ## Concatenate Dataframes

# In[10]:


df_all = pd.concat([df_min3, df_top100], axis = 0, ignore_index = True)


# In[11]:


len(df_all)


# In[12]:


df_all.tail()


# ## Keep where dataframe has non-zero telephone numbers AND non-zero country codes

# In[15]:


df_clean = df_all[df_all["addresscountry"].notna()]
df_clean = df_clean[df_clean["telephone"].notna()]
len(df_clean)


# ## Format longitudes AND latitudes

# In[17]:


lon = "longitude"
lat = "latitude"


# In[18]:


# Remove entries that are not numbers or cannot be convertred to one number (list etc.)
longitudes = df_clean[lon].to_numpy()
latitudes = df_clean[lat].to_numpy()
deleteList = []
i = 0

for value in longitudes:
    if ((isinstance(value, str) == False) & (isinstance(value, float) == False)):
        deleteList.append(i)
    i = i + 1

i = 0
for value in latitudes:
    if ((isinstance(value, str) == False) & (isinstance(value, float) == False)):
        deleteList.append(i)
    i = i + 1    

df_clean.drop(df_clean.index[deleteList], axis = 0, inplace = True)        


# ### Format longitude and latitude

# In[19]:


longArray = df_clean[lon].to_numpy().astype(str)
longArray = np.char.replace(longArray, ',', '.')
longArray = np.char.replace(longArray, '--', '-')
df_clean[lon] = longArray
df_clean[lon] = pd.to_numeric(df_clean[lon], errors='coerce')

latArray = df_clean[lat].to_numpy().astype(str)
latArray = np.char.replace(latArray, ',', '.')
latArray = np.char.replace(latArray, '--', '-')
df_clean[lat] = latArray
df_clean[lat] = pd.to_numeric(df_clean[lat], errors='coerce')


# In[20]:


# Remove the entries that were set to NaN because of other errors
df_clean = df_clean[df_clean["longitude"].notna()]
df_clean = df_clean[df_clean["latitude"].notna()]
# Make sure to only include valid longitudes and latitudes
df_clean = df_clean.loc[(df_clean[lat] >= -90) & (df_clean[lat] <= 90)]
df_clean = df_clean.loc[(df_clean[lon] >= -180) & (df_clean[lon] <= 80)]


# In[21]:


len(df_clean)


# ## Second preprocessing step
# 
# ### Remove non-digits from telephone numbers

# In[22]:


def remove_non_digits(string):
    cleaned = re.sub('[^0-9]','', string)
    return cleaned


# In[23]:


df_clean['telephone_'] = df_clean['telephone'].astype('str').apply(remove_non_digits)


# ### Extract country codes to ISO-2 format using ``pycountry``

# In[24]:


countries = {}
for country in pycountry.countries:
    countries[country.name] = country.alpha_2

countries


# In[36]:


# fuction to modify the country dictionary in uppercase
def modify_dic(d):
    for key in list(d.keys()):
        new_key = key.upper()
        d[new_key] = d[key]
        print(new_key)
    return d


# In[37]:


countries_upper = modify_dic(countries)


# In[39]:


countries_upper


# In[40]:


#uppercase the df_column 
df_clean["addresscountry"] = df_clean["addresscountry"].str.upper()


# In[41]:


# Replace known countrires with ISO-2 format country code
for key, value in countries_upper.items():
    df_clean["addresscountry"] = df_clean["addresscountry"].str.replace(key, value)


# In[42]:


df_clean["addresscountry"].unique()


# ## Manually normalize countries which do not exist in country package

# In[45]:


df_clean["addresscountry"].value_counts().head(30)


# In[44]:


country_dictionary = {
                      "UNITED STATES": "US",
                      "USA":"US",
                      "UNITED KINGDOM": "GB",
                      "UK": "GB",
                      "CANADA": "CA",
                      "AUSTRALIA": "AU",
                      "UNITED ARAB EMIRATES":"AE",
                      "INDIA" : "IN",
                      "NEW ZEALAND": "NZ",
                      "SVERIGE" : "SE",
                      "DEUTSCHLAND": "DE",
                      "RUSSIA": "RU",
                      "ITALIA": "IT",
                      "IRAN": "IR",
                      ", IN" : "IN",
                      "ENGLAND": "GB"
                    }
for key, value in country_dictionary.items():
    df_clean["addresscountry"] = df_clean["addresscountry"].str.replace(key, value)


# ## In this manual step we save about 43.000 extra datapoints

# ## Remove non-covered countries
# 
# ### There are still some uncovered cases left which have to be removed

# In[47]:


df_clean.reset_index(inplace=True)


# In[48]:


liste = []
for i, row in enumerate(df_clean["addresscountry"]):
    if len(row) > 2:
        liste.append(i)
        
df_clean = df_clean.drop(liste)        


# In[49]:


df_clean["addresscountry"].unique()


# ## Drop empty phonenumbers and too lenghty phone numbers

# In[50]:


df_clean = df_clean[df_clean["telephone_"] != "" ]


# In[51]:


liste = []
df_clean.reset_index(inplace=True)
for row_index in df_clean.index:
    if len(df_clean.iloc[row_index]["telephone_"])>18:
        liste.append(row_index)


# In[52]:


df_clean.drop(labels = liste, inplace = True)


# In[53]:


df_clean = df_clean.drop(columns = ["level_0","index"])
df_clean.tail()


# In[54]:


len(df_clean)


# ## Define normalizer for telephone package phonenumbers

# In[55]:


def normalizer(entity):
    number = entity["telephone_"]
    address_country = entity["addresscountry"]
    phone_number = phonenumbers.parse(number, address_country)
    return phone_number


# ## Finally normalizing phone numbers in E.164 format
# 
# ### Ignore those which can not be identified and replace as ``unknown``

# In[56]:


df_clean.reset_index(inplace=True)
phone_objects =[]
#index = []
for row_index in df_clean.index:
    try:
        phone_object =  normalizer(df_clean.iloc[row_index])
        #index.append(row_index)
        phone_objects.append(phone_object)
    except:
        phone_objects.append("unknown")
    
    


# In[57]:


len(phone_objects)


# In[58]:


df_clean["phone_object"] = pd.Series(phone_objects)


# In[59]:


df_clean = df_clean.drop(columns = "index")
df_clean.head()


# In[60]:


unknown_rows = df_clean[df_clean["phone_object"] == "unknown"].index


# In[61]:


df_clean = df_clean.drop(unknown_rows)


# In[62]:


len(df_clean)


# ## Check whether phonenumbers are valid

# In[63]:


df_valid_numbers = df_clean[df_clean["phone_object"].apply(phonenumbers.is_valid_number)]


# In[64]:


len(df_valid_numbers)


# ## Next step: Format every telephone number into unique E.164 format

# In[37]:


#phonenumbers.format_number(df_valid_numbers["phone_object"][0], phonenumbers.PhoneNumberFormat.E164)


# In[65]:


df_valid_numbers["E.164 format"] = df_valid_numbers["phone_object"].apply(lambda objects: phonenumbers.format_number(objects, phonenumbers.PhoneNumberFormat.E164))


# In[66]:


len(df_valid_numbers)


# In[67]:


df_valid_numbers.head()


# ## After formatting phone numbers into unified format we can group by phone numbers to identify clusters

# In[79]:


df_valid_numbers["E.164 format"].value_counts().sort_values().tail(100)


# ## As one can see from the geo locations this is a *successful* match!

# In[80]:


pd.set_option('display.max_columns', 500)
df_valid_numbers[df_valid_numbers["E.164 format"] == "+442084681087"][:5]


# ## Note we also have many non-matches which is why we need geo-locations

# ## Adding the matching telephone numbers in a new column

# In[81]:


def createKDTree(tupleArray):
    tree = KDTree(tupleArray)
    return tree

# Return all values that are in a specific proximity
def queryTree(tree, point, r = 0):    
    point = [float(i) for i in point]
    idx = tree.query_ball_point(point, r)
    return idx

df_valid_numbers['telephoneNorm'] = df_valid_numbers['E.164 format'].str.replace('+','').astype(np.int64)
df_valid_numbers.reset_index(drop=True, inplace=True)
df_valid_numbers['indexValue'] = df_valid_numbers.index
telephoneArray = df_valid_numbers['telephoneNorm'].to_numpy().astype('int64')
fillArray = np.full(len(telephoneArray), 1)
tupleArray = np.array((telephoneArray, fillArray)).T.astype('int64')


# create new column with all matching points
tree = createKDTree(tupleArray)
idx = queryTree(tree, tupleArray[0])

# Search for the closest neighbour in all of the points
df_valid_numbers['MatchingNumbers'] = df_valid_numbers.apply(lambda row: queryTree(tree,[row['telephoneNorm'], 1]), axis=1) 


# In[82]:


# filter out the values which only have one value
data = df_valid_numbers[df_valid_numbers['MatchingNumbers'].apply(lambda x: len(x) > 1)]


# In[83]:


len(data)


# In[84]:


data.head()


# ## Additional Filtering by Geo Location

# In[85]:


def calcDifference(pointOne, pointTwo):
    return geopy.distance.great_circle(pointOne, pointTwo).km

def calcDifferenceFromRow (row):
    tmp = data
    indexValue = row['indexValue']
    indexPosition = (row[lat], row[lon])
    diffList = []
    for value in row['MatchingGeoPoints']:
        if not value in tmp.index:
            continue
        currRow = data.loc[data['indexValue'] == value]
        currIndex = currRow['indexValue'].values[0]
        if currIndex == indexValue:
            diffList.append(-1)
        else:            
            currPosition = (currRow[lat].values[0], currRow[lon].values[0])
            diffList.append(calcDifference(indexPosition, currPosition))
    return diffList


# In[86]:


def createKDTree(tupleArray):
    tree = KDTree(tupleArray)
    return tree

# Return all values that are in a specific proximity
def queryTree(tree, point, radius = 0.001):    
    point = [float(i) for i in point]
    idx = tree.query_ball_point(point, r=radius)
    return idx
    #idx = tree.query(point, k=neighbours)
    #return idx[1]

# convert to tuples and from string to float
lonArr = data[lon].to_numpy()
latArr = data[lat].to_numpy()
tupleArray = np.array((lonArr, latArr)).T.astype('float32')

data.reset_index(drop=True, inplace=True)
data['indexValue'] = data.index

# create new column with all matching points
tree = createKDTree(tupleArray)
idx = queryTree(tree, tupleArray[0])

# Search for the closest neighbour in all of the points
data['MatchingGeoPoints'] = data.apply(lambda row: queryTree(tree,[row[lon], row[lat]]), axis=1) 

# Keep those that have one or more matches withing the radius
data = data[data['MatchingGeoPoints'].apply(lambda x: len(x) > 1)]


# In[87]:


# Calculate the difference in km between those
data['Difference'] = data.apply(lambda row: calcDifferenceFromRow(row), axis=1) 


# In[88]:


data.head(5)


# In[89]:


len(data)


# In[92]:


data.loc[data['indexValue'] == 4]


# In[91]:


data.loc[data['indexValue'] == 6632]


# In[90]:


pwd


# In[90]:


data.to_json("test_file", compression='gzip', orient='records', lines=True)

