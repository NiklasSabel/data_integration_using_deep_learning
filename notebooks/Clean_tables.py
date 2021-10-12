import pandas as pd
import os
import fasttext
import progressbar

fasttext.FastText.eprint = lambda x: None

path = os.getcwd()
data_path = os.path.join(os.getcwd(), 'src\data')
table_corpus_path = os.path.join(data_path, 'product\lspc2020_to_tablecorpus')
top_100_path = os.path.join(data_path, 'product\product_top100')


def remove_irrelevant_tlds():
    all_files = os.listdir(top_100_path)

    valid_tld = ['.com', '.net', '.org', '.uk']
    valid_files = []
    file_valid = 'false'

    for file in all_files:
        file_valid = 'false'
        for tld in valid_tld:
            if tld in file:
                file_valid = 'true'
        if file_valid == 'true':
            valid_files.append(file)

    print(len(valid_files))


pretrained_fasttext_path = os.path.join(os.getcwd(), 'src\models\lid.176.bin')
model = fasttext.load_model(pretrained_fasttext_path)

df = pd.read_json(top_100_path + r'\Product_alibaba.com_September2020.json.gz', compression='gzip', lines=True)

# iterate over rows
english_products = []
count = 0
with progressbar.ProgressBar(max_value=df.shape[0]) as bar:
    for i in df.index:
        row_id = df['row_id'][i]
        name = df['name'][i]
        description = df['description'][i]
        if name is None or model.predict([name])[0] == [['__label__en']]:
            if description is None or model.predict([description])[0] == [['__label__en']]:
                english_products.append(row_id)
        count += 1
        bar.update(count)

test = 2
