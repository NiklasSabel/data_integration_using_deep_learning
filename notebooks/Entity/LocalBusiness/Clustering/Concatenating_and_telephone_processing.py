#!pip install rank-bm25
# from rank_bm25 import BM25Okapi
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
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
lb_path_min3 = path + r"/LocalBusiness/LocalBusiness_minimum3/geo_preprocessed_v3"
lb_path_top100 = path + r"/LocalBusiness/LocalBusiness_top100/geo_preprocessed_v3"
rest_path_min3 = path + r"/Restaurant/Restaurant_minimum3/geo_preprocessed_v3"
rest_path_top100 = path + r"/Restaurant/Restaurant_top100/geo_preprocessed_v3"
hotel_path_min3 = path + r"/Hotel/Hotel_minimum3/geo_preprocessed_v3"
hotel_path_top100 = path + r"/Hotel/Hotel_top100/geo_preprocessed_v3"

file_path_list = [lb_path_min3, lb_path_top100, rest_path_min3, rest_path_top100, hotel_path_min3, hotel_path_top100]

for path in file_path_list:
    current = len(os.listdir(path))
    print(current)


file_path_list = [lb_path_min3, lb_path_top100, rest_path_min3, rest_path_top100, hotel_path_min3, hotel_path_top100]


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

df_all = pd.concat(df_list, axis=0, ignore_index=True)

print("concatenation done")
print("Number of unique tables:", df_all['origin'].nunique())

# First Subset is to get only entries where a telephone nuumber is available
df_clean_phone = df_all[df_all["telephone"].notna()]
# Check number of tables left
print("Number of unique tables with telephone_number:", df_clean_phone["origin"].nunique())

df_clean_phone = df_clean_phone[
    ["row_id", "origin", "name", "address", "page_url", "telephone", "addressregion", "streetaddress",
     "addresslocality", "addresscountry", "longitude", "latitude"]]


# First round of parsing with raw data
# try parsing only from telephone_numbers

df_clean_phone_parse = df_clean_phone.copy()

def normalizer_2(entity):
    number = entity["telephone"]
    phone_number = phonenumbers.parse(number)
    return phone_number

df_clean_phone_parse.reset_index(inplace=True)
phone_objects = []
for row_index in df_clean_phone_parse.index:
    try:
        phone_object = normalizer_2(df_clean_phone_parse.iloc[row_index])
        phone_objects.append(phone_object)
    except:
        phone_objects.append("unknown")


print("First Parsing done")

# append to df_clean
df_clean_phone_parse["phone_object"] = pd.Series(phone_objects)

# save unkown rows where parser failed
unknown_rows = df_clean_phone_parse[df_clean_phone_parse["phone_object"] == "unknown"]
df_clean_phone_parse = df_clean_phone_parse.drop(unknown_rows.index)

# use phone package to isolate valid phone numbers
df_clean_phone_parse = df_clean_phone_parse[df_clean_phone_parse["phone_object"].apply(phonenumbers.is_valid_number)]
df_clean_phone_parse["E.164 format"] = df_clean_phone_parse["phone_object"].apply(
    lambda objects: phonenumbers.format_number(objects, phonenumbers.PhoneNumberFormat.E164))

# Have a look at parsed numbers
df_clean_phone_parse['telephoneNorm'] = df_clean_phone_parse['E.164 format'].str.replace('+', '').astype(np.int64)
df_clean_phone_parse["E.164 format"].value_counts().sort_values().tail(20)

# Now derive all rows where parsing failed
succesful_rows = list(df_clean_phone_parse["index"])
df_residual = df_clean_phone.drop(succesful_rows)

print("Length of residual is:" , len(df_residual))
print("Length of parsed is:" , len(df_clean_phone_parse))
print("Lenght original is:", len(df_clean_phone))

# Define function for cleaning telephone numbers
def remove_non_digits(string):
    cleaned = re.sub('[^0-9]', '', string)
    return cleaned
df_residual['telephone_'] = df_residual['telephone'].astype('str').apply(remove_non_digits)

# Preprocessing for 2nd parsing
# now get the rows where we can use additional coutry code for parsing
#  
df_clean = df_residual[df_residual["addresscountry"].notna()]


# Normalize country codes for telephone package
countries = {}
for country in pycountry.countries:
    countries[country.name] = country.alpha_2


# fuction to modify the country dictionary in uppercase
def modify_dic(d):
    for key in list(d.keys()):
        new_key = key.upper()
        d[new_key] = d[key]
    return d


countries_upper = modify_dic(countries)


# uppercase the df_column
df_clean["addresscountry"] = df_clean["addresscountry"].str.upper()

# Replace known countries with ISO-2 format country code
for key, value in countries_upper.items():
    df_clean["addresscountry"] = df_clean["addresscountry"].str.replace(key, value)

# Manual country code replacement
country_dictionary = {
    "UNITED STATES": "US",
    "USA": "US",
    "UNITED KINGDOM": "GB",
    "UK": "GB",
    "CANADA": "CA",
    "AUSTRALIA": "AU",
    "UNITED ARAB EMIRATES": "AE",
    "UAE": "AE",
    "INDIA": "IN",
    "NEW ZEALAND": "NZ",
    "SVERIGE": "SE",
    "DEUTSCHLAND": "DE",
    "DEU": "DE",
    "RUSSIA": "RU",
    "ITALIA": "IT",
    "IRAN": "IR",
    ", IN": "IN",
    "ENGLAND": "GB",
    "FRA": "FR"
}
for key, value in country_dictionary.items():
    df_clean["addresscountry"] = df_clean["addresscountry"].str.replace(key, value)


print("phone_country normalizer")
def normalizer(entity):
    number = entity["telephone_"]
    address_country = entity["addresscountry"]
    phone_number = phonenumbers.parse(number, address_country)
    return phone_number


df_clean.reset_index(inplace=True)
phone_objects = []
for row_index in df_clean.index:
    try:
        phone_object = normalizer(df_clean.iloc[row_index])
        phone_objects.append(phone_object)
    except:
        phone_objects.append("unknown")


print("normalizer successful")

# append to df_clean
df_clean["phone_object"] = pd.Series(phone_objects)

# save unkown rows where parser failed
unknown_rows = df_clean[df_clean["phone_object"] == "unknown"]
df_clean = df_clean.drop(unknown_rows.index)

# use phone package to isolate valid phone numbers
df_clean = df_clean[df_clean["phone_object"].apply(phonenumbers.is_valid_number)]
df_clean["E.164 format"] = df_clean["phone_object"].apply(
    lambda objects: phonenumbers.format_number(objects, phonenumbers.PhoneNumberFormat.E164))

# Have a look at parsed numbers
df_clean['telephoneNorm'] = df_clean['E.164 format'].str.replace('+', '').astype(np.int64)
df_clean["E.164 format"].value_counts().sort_values().tail(20)


# Now derive all rows where parsing failed
succesful_rows_2 = list(df_clean["index"])
df_residual_final = df_residual.drop(succesful_rows_2)

print("Creating Concatenated Dataframes")

### Concatenate dataframes
# First, the subsets which could be parsed with telephone package

df_parsed = pd.concat([df_clean_phone_parse, df_clean], axis=0)

# append a new column for later grouping

df_residual_final["telephoneNorm"] = df_residual_final["telephone_"].astype(np.int64)


print("TESTING")
# TESTING

first = len(df_residual_final)
second = len(df_clean_phone_parse)
third = len(df_clean)

total = first + second + third
print(total)
print(df_clean_phone)

### Final Concatenation

df_total = pd.concat([df_parsed, df_residual_final], axis=0)
df_total["telephoneNorm"].value_counts().sort_values().tail(20)

### Some cleaning

df_total_clean = df_total[df_total["telephoneNorm"] != ""]
df_total_clean = df_total_clean[df_total_clean["telephoneNorm"] != "0"]
df_total_clean = df_total_clean[df_total_clean["telephoneNorm"] != "00"]


print(df_total_clean["origin"].nunique())

df_total_clean.to_json("19-12-21_MatchingFile", compression="gzip", orient='records', lines=True)
print("success")
