import os

import pandas as pd

cwd = os.getcwd()

def open_local_business_file(file_name):
    data_directory = r'..\src\data'
    path = data_directory + r'\LocalBusiness\LocalBusiness_statistics\table_statistics'
    file_path = path + r'\{}'.format(file_name)
    file = pd.read_csv(file_path)
    return file
# for the sake of variable length, LB equals Local Business
LB_statistics = open_local_business_file('LocalBusiness_September2020_statistics.csv')
LB_top100 = open_local_business_file('LocalBusiness_September2020_statistics_top100.csv')
LB_min3 = open_local_business_file('LocalBusiness_September2020_statistics_minimum3.csv')
LB_rest = open_local_business_file('LocalBusiness_September2020_statistics_rest.csv')

# Useful tables can be found in the LB_top100 as we have enough entries for matching
# the other 2 dataframes LB_min3 and LB_rest are probably less useful for matching purposes


print('break')