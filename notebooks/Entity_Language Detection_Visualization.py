import pandas as pd
import os
import progressbar
import json
import matplotlib.pyplot as plt
import dataframe_image as dfi

path_parent = os.path.dirname(os.getcwd())
data_path = os.path.join(path_parent, 'src/data')
visualization_path = os.path.join(path_parent, 'src/visualization')

def write_dicts(entity):
    """
    writes dictionaries counting for each file the number of rows before and after language detection
    :param entity: entity currently running on
    :return:
    """

    start_data_dict = {}
    end_data_dict = {}

    count = 0
    with progressbar.ProgressBar(max_value=len(cleaned_files)) as bar:
        for cleaned_file in cleaned_files:
            df = pd.read_json(os.path.join(cleaned_data_path, '{}'.format(cleaned_file)), compression='gzip', lines=True)
            end_data_dict[cleaned_file] = df.shape[0]
            count += 1
            bar.update(count)

    count2 = 0
    with progressbar.ProgressBar(max_value=len(files)) as bar:
        for file in files:
            df = pd.read_json(os.path.join(data_path, '{}'.format(file)), compression='gzip', lines=True)
            start_data_dict[file] = df.shape[0]
            if file not in cleaned_files:
                end_data_dict[file] = 0
            count2+=1
            bar.update(count2)

    with open(os.path.join(visualization_path, '{}_start_data_visualization.json'.format(entity)), 'w') as json_file:
        json.dump(start_data_dict, json_file)

    with open(os.path.join(visualization_path, '{}_end_data_visualization.json'.format(entity)), 'w') as json_file:
        json.dump(end_data_dict, json_file)


def visualize(entity):
    """
    visualize the data next to each other
    :return:
    """
    with open(os.path.join(visualization_path, '{}_start_data_visualization.json'.format(entity)), 'r') as json_file:
        start_data_dict = json.load(json_file)

    with open(os.path.join(visualization_path, '{}_end_data_visualization.json'.format(entity)), 'r') as json_file:
        end_data_dict = json.load(json_file)

    start_df = pd.DataFrame(start_data_dict.items(), columns=['File Name', 'Before Cleaning'])
    end_df = pd.DataFrame(end_data_dict.items(), columns=['File Name', 'After Cleaning'])

    result = pd.merge(start_df, end_df, on='File Name')
    result['ratio'] = result['After Cleaning'] / result['Before Cleaning']

    # look at files which have a very low ratio
    much_cleaned_df = result[(result['ratio'] <= 0.2) & (result['ratio']>0)]

    for file in much_cleaned_df['File Name']:
        old_df = pd.read_json(os.path.join(data_path, '{}'.format(file)), compression='gzip', lines=True)
        new_df = pd.read_json(os.path.join(cleaned_data_path, '{}'.format(file)), compression='gzip', lines=True)


    result.plot(kind='bar', x='File Name', y=['Before Cleaning', 'After Cleaning'])

    #plt.xticks(rotation='vertical', fontsize='xx-small')
    plt.xticks([])
    plt.legend(loc='upper left', fontsize='small')
    plt.xlabel('File')
    plt.ylabel('# of rows')
    plt.title('{} Tables All'.format(entity))
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(visualization_path, '{}_Comparison_Cleaning_All.jpg'.format(entity)))


    result2 = result[result['After Cleaning']!=0]

    result2.plot(kind='bar', x='File Name', y=['Before Cleaning', 'After Cleaning'])
    #plt.xticks(rotation='vertical', fontsize='xx-small')
    plt.xticks([])
    plt.legend(loc='upper left', fontsize='small')
    plt.xlabel('File')
    plt.ylabel('# of rows')
    plt.title('{} Tables All Cleaned'.format(entity))
    plt.tight_layout()
    #plt.show()
    plt.savefig(os.path.join(visualization_path, '{}_Comparison_Cleaning_Only Cleaned.jpg'.format(entity)))


def get_total_numbers():
    #with open(os.path.join(visualization_path, 'product_start_data_visualization.json'), 'r') as json_file:
     #  start_data_dict = json.load(json_file)

    with open(os.path.join(visualization_path, 'LocalBusiness_start_data_visualization.json'), 'r') as json_file:
        start_data_dict = json.load(json_file)

    #with open(os.path.join(visualization_path, 'product_end_data_visualization.json'), 'r') as json_file:
     #  end_data_dict = json.load(json_file)

    with open(os.path.join(visualization_path, 'LocalBusiness_end_data_visualization.json'), 'r') as json_file:
        end_data_dict = json.load(json_file)

    start_df = pd.DataFrame(start_data_dict.items(), columns=['File Name', 'Before Cleaning'])
    end_df = pd.DataFrame(end_data_dict.items(), columns=['File Name', 'After Cleaning'])

    result = pd.merge(start_df, end_df, on='File Name')

    total_before = result['Before Cleaning'].sum()
    total_after = result['After Cleaning'].sum()

    max_val = result['After Cleaning'].max()
    file_max_val = result[result['After Cleaning']==max_val]

    test2 = 2

#get_total_numbers()

def plot_data_snippet():
    df = pd.read_json(os.path.join(cleaned_data_path, 'Product_myshopify.com_September2020.json.gz'), compression='gzip', lines=True)
    df_small = df.head(5)

    dfi.export(df_small, os.path.join(visualization_path, 'Product_myshopify.com_September2020_snippet.jpg'))
    test1 =2

#plot_data_snippet()

#run functions
#entities = ['product_top100', 'product_minimum3', 'LocalBusiness_top100', 'LocalBusiness_minimum3', 'LocalBusiness_rest',
 #           'Restaurant_top100', 'Restaurant_minimum3', 'Restaurant_rest', 'Hotel_top100', 'Hotel_minimum3', 'Hotel_rest']

entities = ['product_top100', 'LocalBusiness_top100']

for entity in entities:
    if entity == 'product_top100':
        data_path = os.path.join(path_parent, 'src/data/product/product_top100')
    elif entity == 'product_minimum3':
        data_path = os.path.join(path_parent, 'src/data/product/product_minimum3')
    elif entity == 'LocalBusiness_top100':
        data_path = os.path.join(path_parent, 'src/data/LocalBusiness/LocalBusiness_top100')
    elif entity == 'LocalBusiness_minimum3':
        data_path = os.path.join(path_parent, 'src/data/LocalBusiness/LocalBusiness_minimum3')
    elif entity == 'LocalBusiness_rest':
        data_path = os.path.join(path_parent, 'src/data/LocalBusiness/LocalBusiness_rest')
    elif entity == 'Restaurant_top100':
        data_path = os.path.join(path_parent, 'src/data/Restaurant/Restaurant_top100')
    elif entity == 'Restaurant_minimum3':
        data_path = os.path.join(path_parent, 'src/data/Restaurant/Restaurant_minimum3')
    elif entity == 'Restaurant_rest':
        data_path = os.path.join(path_parent, 'src/data/Restaurant/Restaurant_rest')
    elif entity == 'Hotel_top100':
        data_path = os.path.join(path_parent, 'src/data/Hotel/Hotel_top100')
    elif entity == 'Hotel_minimum3':
        data_path = os.path.join(path_parent, 'src/data/Hotel/Hotel_minimum3')
    elif entity == 'Hotel_rest':
        data_path = os.path.join(path_parent, 'src/data/Hotel/Hotel_rest')

    cleaned_data_path = os.path.join(data_path, 'cleaned')

    files = [file for file in os.listdir(data_path) if file.endswith('.json.gz')]
    cleaned_files = [file for file in os.listdir(cleaned_data_path) if file.endswith('.json.gz')]

    write_dicts(entity)
    visualize(entity)

    test = 1