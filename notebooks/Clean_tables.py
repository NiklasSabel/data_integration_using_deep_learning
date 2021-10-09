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
table_corpus_path = os.path.join(data_path, 'product\lspc2020_to_tablecorpus')
top_100_path = os.path.join(data_path, 'product\product_top100')
cleaned_top_100_path = os.path.join(top_100_path, 'cleaned')

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

    for file in files:
        print(file)
        df = pd.read_json(os.path.join(cleaned_top_100_path, '{}'.format(file)), compression='gzip', lines=True)

        #iterrate over rows and save row_ids of english products
        english_products = []
        count = 0
        with progressbar.ProgressBar(max_value=df.shape[0]) as bar:
            for i in range(df.shape[0]):  # iterate over rows
                row_id = df['row_id'][i]
                for j in range(df.shape[1]): # iterate over columns
                    if df.columns[j].lower() == 'brand': # exclude brand column
                        cell_pred = 'None'
                    else:
                        cell = df.iat[i,j]
                        if type(cell) == str:
                            cell_pred = model.predict([cell])[0]
                        elif type(cell) == list:
                            if type(cell[0]) == str:
                                cell_pred = model.predict(cell)[0]
                            else:
                                cell_pred = 'None'
                        else:
                            cell_pred = 'None'
                    if cell_pred == 'None' or cell_pred == [['__label__en']]:
                        english = 'True'
                    else:
                        english = 'False'
                        break
                if english == 'True':
                    english_products.append(row_id)
                count += 1
                bar.update(count)

        # design new dataframe with english products only
        df_cleaned = df[df['row_id'].isin(english_products)]

        # write to gzip compressed json file
        df_cleaned.to_json(os.path.join(cleaned_top_100_path, '{}'.format(file)), compression='gzip', orient='records', lines=True)

# run functions
remove_irrelevant_tlds()
remove_with_fasttext()