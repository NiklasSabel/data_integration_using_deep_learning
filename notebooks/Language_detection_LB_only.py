import pandas as pd
import os
import re
import fasttext
import progressbar
import json
import gzip
import shutil
fasttext.FastText.eprint = lambda x: None # avoid Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.


path_parent = os.path.dirname(os.getcwd())


def remove_irrelevant_tlds():
    """
    moves all files with valid tlds to a new path called "cleaned"
    :return:
    """
    files = [file for file in os.listdir(data_path) if file.endswith('.json.gz')]

    valid_tld = ['.com', '.net', '.org', '.uk', '.gov', '.edu', '.us', '.biz', '.au']
    valid_files = []
    file_valid = 'false'

    print('run tlds cleaning')

    count = 0
    with progressbar.ProgressBar(max_value=len(files)) as bar:
        for file in files:
            file_valid = 'false'
            for tld in valid_tld:
                if tld in file:
                    file_valid = 'true'
            if file_valid == 'true':
                valid_files.append(file)
                # copy only files with valid tlds to cleaned path
                shutil.copy(os.path.join(data_path, '{}'.format(file)), cleaned_data_path, follow_symlinks=True)

            count += 1
            bar.update(count)

    print('tlds cleaning done')


def remove_with_fasttext():
    """
    reads all files from cleaned data path and removes non-english products from the data tables
    :return:
    """
    pretrained_fasttext_path = os.path.join(path_parent, 'src/models/lid.176.bin')
    model = fasttext.load_model(pretrained_fasttext_path)

    # list all top 100 product files
    files = [file for file in os.listdir(cleaned_data_path) if file.endswith('.json.gz')]

    removed_rows_dict = {}

    count_files = 0

    for file in files:
        print(file)
        df = pd.read_json(os.path.join(cleaned_data_path, '{}'.format(file)), compression='gzip', lines=True)
        # if df.shape[0] > 20: # for top100 & min3
        # if df.shape[0] > 0: # for rest only
        if df.shape[0] > 10: # LB only
            df['concat'] = ''

            for j in range(df.shape[1]):  # iterate over columns
                df['concat'] = df['concat'] + df.iloc[:,j].astype('str')

            # iterrate over rows and save row_ids of english products
            english_products = []
            non_english_products = []
            count = 0
            with progressbar.ProgressBar(max_value=df.shape[0]) as bar:
                for i in range(df.shape[0]):  # iterate over rows
                    row_id = int(df['row_id'][i])
                    cell = df['concat'][i]
                    cell_clean = re.sub('[^A-Za-z_\s]', ' ', cell)
                    cell_pred = model.predict([cell_clean])[0]
                    if cell_pred == [['__label__en']]:
                        english_products.append(row_id)
                    else:
                        non_english_products.append(row_id)
                    count += 1
                    bar.update(count)

            # drop concatendated row again
            df = df.drop('concat', axis=1)

            # write removed row_ids to dictionary
            removed_rows_dict[file] = non_english_products

            # design new dataframe with english products only
            df_cleaned = df[df['row_id'].isin(english_products)]

            # write to gzip compressed json file
            # if df_cleaned.shape[0] > 20: # for top100 & min3
            # if df_cleaned.shape[0] > 0: # for rest only
            if df_cleaned.shape[0] > 10:  # for LB only
                df_cleaned.to_json(os.path.join(cleaned_data_path, '{}'.format(file)), compression='gzip', orient='records', lines=True)
            else:
                os.remove(os.path.join(cleaned_data_path, '{}'.format(file)))


        else:
            # if df does not contain more than 20 entries delete it from cleaned file path
            os.remove(os.path.join(cleaned_data_path, '{}'.format(file)))

        count_files += 1
        print('{} out of {} files done'.format(count_files, len(files)))

    # save dictionary with removed row_ids
    with open(os.path.join(cleaned_data_path, 'removed_rows_dict.json'), 'w', encoding='utf-8') as f:
        json.dump(removed_rows_dict, f)

    print('removed_rows_dict saved')


# run functions
#entities = ['product_top100', 'product_minimum3', 'LocalBusiness_top100', 'LocalBusiness_minimum3', 'LocalBusiness_rest',
 #           'Restaurant_top100', 'Restaurant_minimum3', 'Restaurant_rest', 'Hotel_top100', 'Hotel_minimum3', 'Hotel_rest']
# entities = ['LocalBusiness_rest', 'Restaurant_rest', 'Hotel_rest']

entities = ['LocalBusiness_top100', 'LocalBusiness_minimum3','Restaurant_top100', 'Restaurant_minimum3', 'Hotel_top100', 'Hotel_minimum3']

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

    cleaned_data_path = os.path.join(data_path, 'cleaned_new_threshold')

    print('running {} path'.format(entity))

    remove_irrelevant_tlds()
    remove_with_fasttext()