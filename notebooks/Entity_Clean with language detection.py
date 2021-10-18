import pandas as pd
import os
import fasttext
import progressbar
import json
import gzip
import shutil
fasttext.FastText.eprint = lambda x: None # avoid Warning : `load_model` does not return WordVectorModel or SupervisedModel any more, but a `FastText` object which is very similar.


path_parent = os.path.dirname(os.getcwd())
data_path = os.path.join(path_parent, 'src\data')


def remove_irrelevant_tlds():
    """
    moves all files with valid tlds to a new path called "cleaned"
    :return:
    """
    files = [file for file in os.listdir(top_100_path) if file.endswith('.json.gz')]

    valid_tld = ['.com', '.net', '.org', '.uk']
    valid_files = []
    file_valid = 'false'

    for file in files:
        file_valid = 'false'
        for tld in valid_tld:
            if tld in file:
                file_valid = 'true'
        if file_valid == 'true':
            valid_files.append(file)
            # copy only files with valid tlds to cleaned path
            shutil.copy(os.path.join(top_100_path, '{}'.format(file)), cleaned_top_100_path, follow_symlinks=True)


def remove_with_fasttext():
    """
    reads all files from cleaned data path and removes non-english products from the data tables
    :return:
    """
    pretrained_fasttext_path = os.path.join(path_parent, 'src\models\lid.176.bin')
    model = fasttext.load_model(pretrained_fasttext_path)

    # list all top 100 product files
    files = [file for file in os.listdir(cleaned_top_100_path) if file.endswith('.json.gz')]

    count_files = 0

    for file in files:
        print(file)
        df = pd.read_json(os.path.join(cleaned_top_100_path, '{}'.format(file)), compression='gzip', lines=True)
        if df.shape[0] > 20:
            df['concat'] = ''

            for j in range(df.shape[1]):  # iterate over columns
                df['concat'] = df['concat'] + df.iloc[:,j].astype('str')

            #iterrate over rows and save row_ids of english products
            english_products = []
            count = 0
            with progressbar.ProgressBar(max_value=df.shape[0]) as bar:
                for i in range(df.shape[0]):  # iterate over rows
                    row_id = df['row_id'][i]
                    cell = df['concat'][i]
                    cell_pred = model.predict([cell])[0]
                    if cell_pred == [['__label__en']]:
                        english_products.append(row_id)
                    count += 1
                    bar.update(count)

            # drop concatendated row again
            df = df.drop('concat', axis=1)

            # design new dataframe with english products only
            df_cleaned = df[df['row_id'].isin(english_products)]


            # write to gzip compressed json file
            if df_cleaned.shape[0] > 20:
                df_cleaned.to_json(os.path.join(cleaned_top_100_path, '{}'.format(file)), compression='gzip', orient='records', lines=True)
            else:
                os.remove(os.path.join(cleaned_top_100_path, '{}'.format(file)))


        else:
            # if df does not contain more than 20 entries delete it from cleaned file path
            os.remove(os.path.join(cleaned_top_100_path, '{}'.format(file)))

        count_files += 1
        print('{} out of {} files done'.format(count_files, len(files)))


# run functions
entities = ['product_top100', 'product_min3', 'localbusiness_top100', 'localbusiness_min3']

for entity in entities:
    if entity == 'product_top100':
        top_100_path = os.path.join(data_path, 'product\product_top100')
        cleaned_top_100_path = os.path.join(top_100_path, 'cleaned')
    elif entity == 'product_min3':
        top_100_path = os.path.join(data_path, 'product\product_minimum3')
        cleaned_top_100_path = os.path.join(top_100_path, 'cleaned')
    elif entity == 'localbusiness_top100':
        top_100_path = os.path.join(data_path, 'LocalBusiness\LocalBusiness_top100')
        cleaned_top_100_path = os.path.join(top_100_path, 'cleaned')
    elif entity == 'localbusiness_min3':
        top_100_path = os.path.join(data_path, 'LocalBusiness\LocalBusiness_minimum3')
        cleaned_top_100_path = os.path.join(top_100_path, 'cleaned')

    print('running {} path'.format(entity))