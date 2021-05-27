import pandas as pd
import os
import re

labels = ['is_hired_1mo', 'is_unemployed', 'lost_job_1mo', 'job_search', 'job_offer']
method='uncertainty'
print(method)
method_dict={'our_method': 'convbert-', 'adaptive': 'adaptive', 'uncertainty': 'uncertainty'}

folder_path = '/home/manuto/Documents/world_bank/bert_twitter_labor/twitter-labor-data/data/active_learning/evaluation_inference/US'

folder_list = os.listdir(folder_path)
folder_list = [folder_name for folder_name in folder_list if method_dict[method] in folder_name]

results_dict = dict()
for folder_name in folder_list:
    data_path = os.path.join(folder_path, folder_name)
    # if 'extra_is_hired_1mo.csv' in os.listdir(data_path):
    iter_nb = int(re.findall('iter_(\d)', folder_name)[0])
    print(iter_nb)
    results_dict[iter_nb] = dict()
    for label in labels:
        df = pd.read_csv(os.path.join(data_path, f'{label}.csv'))[['rank', 'class']]
        #df = df.loc[df['rank'] < 21][['rank', 'class']]
        # extra_df = pd.read_csv(os.path.join(data_path, f'extra_{label}.csv'))
        # df = pd.concat([df, extra_df])
        df = df.loc[df['rank'] < 10001].reset_index(drop=True)
        if not df['class'].isnull().values.any():
            precision = df.loc[df['class'] == 1].shape[0]/df.shape[0]
            results_dict[iter_nb][label] = precision
        else:
            print(f'Missing labels for class {label}')

print(results_dict)