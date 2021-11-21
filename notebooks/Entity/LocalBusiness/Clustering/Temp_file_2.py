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

path = r"../../../../src/data"
lb_path_min3 = path + r"/LocalBusiness/LocalBusiness_minimum3/geo_preprocessed"
lb_path_top100 = path + r"/LocalBusiness/LocalBusiness_top100/geo_preprocessed"
lb_path_rest = path + r"/LocalBusiness/LocalBusiness_rest/geo_preprocessed"
rest_path_min3 = path + r"/Restaurant/Restaurant_minimum3/geo_preprocessed"
rest_path_top100 = path + r"/Restaurant/Restaurant_top100/geo_preprocessed"
rest_path_rest = path + r"/Restaurant/Restaurant_rest/geo_preprocessed"
hotel_path_min3 = path + r"/Hotel/Hotel_minimum3/geo_preprocessed"
hotel_path_top100 = path + r"/Hotel/Hotel_top100/geo_preprocessed"
hotel_path_rest = path + r"/Hotel/Hotel_rest/geo_preprocessed"


file_path_list = [lb_path_min3, lb_path_top100, rest_path_min3, rest_path_top100, hotel_path_min3, hotel_path_top100, lb_path_rest, rest_path_rest, hotel_path_rest ]

def create_df(file_path):
    files = os.listdir(file_path)
    df_as_list = []
    for lb in files:
        with gzip.open(file_path + '/' + lb, 'r') as dataFile:
            for line in dataFile:
                lineData = json.loads(line.decode('utf-8'))
                lineData["origin"] = lb
                df_as_list.append(lineData)
    df = pd.DataFrame(df_as_list)
    return df
print("creating df")
df_list = []
for file_path in file_path_list:
    df = create_df(file_path)
    df_list.append(df)

df_all = pd.concat(df_list, axis = 0, ignore_index = True)


print(df_all['origin'].nunique())

df_clean_phone = df_all[df_all["addresscountry"].notna()]
df_clean_phone = df_clean_phone[df_clean_phone["telephone"].notna()]
#len(df_clean_phone)
df_clean_phone["origin"].nunique()

def remove_non_digits(string):
    cleaned = re.sub('[^0-9]','', string)
    return cleaned

df_clean = df_clean_phone[["row_id", "address", "page_url", "origin", "telephone" , "addresscountry", "longitude", "latitude"]]
df_clean['telephone_'] = df_clean['telephone'].astype('str').apply(remove_non_digits)

countries = {}
for country in pycountry.countries:
    countries[country.name] = country.alpha_2

countries

# fuction to modify the country dictionary in uppercase
def modify_dic(d):
    for key in list(d.keys()):
        new_key = key.upper()
        d[new_key] = d[key]
    return d

countries_upper = modify_dic(countries)

#uppercase the df_column
df_clean["addresscountry"] = df_clean["addresscountry"].str.upper()

# Replace known countries with ISO-2 format country code
for key, value in countries_upper.items():
    df_clean["addresscountry"] = df_clean["addresscountry"].str.replace(key, value)

country_dictionary = {
                      "UNITED STATES": "US",
                      "USA":"US",
                      "UNITED KINGDOM": "GB",
                      "UK": "GB",
                      "CANADA": "CA",
                      "AUSTRALIA": "AU",
                      "UNITED ARAB EMIRATES":"AE",
                        "UAE": "AE",
                      "INDIA" : "IN",
                      "NEW ZEALAND": "NZ",
                      "SVERIGE" : "SE",
                      "DEUTSCHLAND": "DE",
                        "DEU": "DE",
                      "RUSSIA": "RU",
                      "ITALIA": "IT",
                      "IRAN": "IR",
                      ", IN" : "IN",
                      "ENGLAND": "GB",
                        "FRA": "FR"
                    }
for key, value in country_dictionary.items():
    df_clean["addresscountry"] = df_clean["addresscountry"].str.replace(key, value)

df_clean.reset_index(inplace=True)

print("normalized country codes")
# liste = []
# for i, row in enumerate(df_clean["addresscountry"]):
#     if len(row) > 2:
#         liste.append(i)
#
# df_clean = df_clean.drop(liste)

df_clean = df_clean[df_clean["telephone_"] != ""]
liste = []
df_clean.reset_index(inplace=True)
for row_index in df_clean.index:
    if len(df_clean.iloc[row_index]["telephone_"]) > 18:
        liste.append(row_index)

df_clean.drop(labels=liste, inplace=True)
df_clean = df_clean.drop(columns=["level_0", "index"])
df_clean.tail()


def normalizer(entity):
    number = entity["telephone_"]
    address_country = entity["addresscountry"]
    phone_number = phonenumbers.parse(number, address_country)
    return phone_number

print("phone_number normalizer")
df_clean.reset_index(inplace=True)
phone_objects = []
# index = []
for row_index in df_clean.index:
    try:
        phone_object = normalizer(df_clean.iloc[row_index])
        # index.append(row_index)
        phone_objects.append(phone_object)
    except:
        phone_objects.append("unknown")

df_clean["phone_object"] = pd.Series(phone_objects)
df_clean = df_clean.drop(columns = "index")
df_clean.head()
unknown_rows = df_clean[df_clean["phone_object"] == "unknown"].index
df_clean = df_clean.drop(unknown_rows)
print("normalizer succesful")
df_valid_numbers = df_clean[df_clean["phone_object"].apply(phonenumbers.is_valid_number)]
df_valid_numbers["E.164 format"] = df_valid_numbers["phone_object"].apply(lambda objects: phonenumbers.format_number(objects, phonenumbers.PhoneNumberFormat.E164))
df_valid_numbers["E.164 format"].value_counts().sort_values().tail(20)
df_valid_numbers['telephoneNorm'] = df_valid_numbers['E.164 format'].str.replace('+', '').astype(np.int64)

print(df_valid_numbers["origin"].nunique())

df_valid_numbers.to_json("Concatenated_MatchingFile_with_rest", compression="gzip", orient='records', lines=True)
print("success")