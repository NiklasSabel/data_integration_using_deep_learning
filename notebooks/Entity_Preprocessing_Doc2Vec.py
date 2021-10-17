import pandas as pd
def open_product_zip_file(file_name):
    file = pd.read_json(file_name, compression='gzip', lines=True)
    return file



def files_to_df_extended(files):
    df_list = []
    for file in files:
        df = open_product_zip_file(file)
        if 'name' in df.columns:
            df_reduced = pd.DataFrame({'name': df['name']})
            df_reduced['origin'] = file
            df_list.append(df_reduced)

    df_merged = pd.concat(df_list)
    df_merged = df_merged.reset_index()
    return df_merged


def delete_empty_rows(df):
    rows = []
    for i, row in enumerate(df['name']):
        if type(row) != str:
            rows.append(i)
    df_clean = df.drop(index=rows)
    df_clean = df_clean.reset_index(drop=True)
    return df_clean



def remove_punctuations(df, keep_numbers='yes'): # keep numbers as default
    if keep_numbers=='yes':
        df["new_column"] = df['name'].str.replace('[^\w\s]', ' ')
    else:
        df["new_column"] = df['name'].str.replace('[^A-Za-z_\s]', ' ')
    return df

def extract_most_similar(model, index, df):
    similar_doc = model.docvecs.most_similar(str(index), topn = 10)
    index_list = []
    for index, similarity in similar_doc:
        index = int(index)
        index_list.append(index)
    return df.iloc[index_list]



