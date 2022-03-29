import json
import os
import pandas as pd
pd.set_option('display.max_colwidth', 1000)



# y_pred
fp = ".\out_model\pred_test.jsonl"   # output of Tabbie  
with open(fp, 'r') as json_file:
    json_list = list(json_file)

# initiate a dataframe
df_pred = pd.DataFrame(columns=['fname','col_id','cluster_id'])

for j in json_list:
    result = json.loads(j)
    fname = result['table_id']
    col_ids = result['col_labels']
    
    for i in col_ids:
        df_pred = df_pred.append({'fname':fname, 'col_id': i, 'cluster_id':result['header'][i]}, ignore_index=True)
df_pred.sort_values(['fname', 'col_id'], ascending=[True, True], inplace=True)

selection_dict = {}
keys = df_pred['fname'].unique()
for key in keys:
    value = df_pred[df_pred['fname']==key]['col_id'].tolist()
    selection_dict[key] = value




    
# y_true
parent_path = os.path.dirname(os.getcwd()) 
test_table_path = parent_path + "/src/data/LocalBusiness/Splitting_TabbieData/Train_Test/test tables"   #TO CHANGE
test_label_path = parent_path + "/src/data/LocalBusiness/Splitting_TabbieData/Train_Test"         #TO CHANGE

# initiate a dataframe
df_true = pd.DataFrame(columns=['fname','col_id','cluster_id'])

test_labels = pd.read_csv(os.path.join(test_label_path, 'test_label.csv'))
if 'Unnamed: 0' in test_labels.columns:
    test_labels.drop(columns=['Unnamed: 0'], inplace= True)

tables = test_labels['fname'].unique()
for t in tables:
    tmp = pd.read_csv(os.path.join(test_table_path, t))
    fname = os.path.splitext(t)[0]
    for i in range(len(tmp.columns)):
        df_true = df_true.append({'fname':fname, 'col_id': i, 'cluster_id': tmp.columns[i]}, ignore_index=True)
                
# filter for including only predicted instances
df_true2= pd.DataFrame(columns=['fname','col_id','cluster_id'])

for key in selection_dict.keys():
    value = selection_dict[key]
    tmp = df_true[df_true['fname']== key]
    tmp2 = tmp[tmp['col_id'].isin(value)] 
    df_true2 = pd.concat([df_true2, tmp2], ignore_index=True) 
df_true2.sort_values(['fname', 'col_id'], ascending=[True, True], inplace=True)





# Evaluation
from sklearn.metrics import accuracy_score, f1_score
y_true = df_true2['cluster_id'].astype(str)
y_pred = df_pred['cluster_id'].astype(str)

print("accuracy: ", accuracy_score(y_true, y_pred))
print("micro f1: ", f1_score(y_true, y_pred, average='micro'))
print("macro f1: ", f1_score(y_true, y_pred, average='macro'))
print("weighted f1: ", f1_score(y_true, y_pred, average='weighted'))