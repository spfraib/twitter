#!/usr/bin/env python
# coding: utf-8
# run using: sbatch --array=0-9 7.9-get-predictions-from-BERT.sh

print('started 10.2-BERT-deploying-1M-randomONLY_1pct.py')

import sys
import os

# column = sys.argv[1]
# column = 'is_unemployed'


####################################################################################################################################
# loading the model
####################################################################################################################################


import time

start_time = time.time()
from transformers import BertTokenizer
from pathlib import Path
import torch

from box import Box
import pandas as pd
import collections

from tqdm import tqdm, trange
# import sys
import random
import numpy as np
# import apex
from sklearn.model_selection import train_test_split

import datetime

import sys
import pickle
import os

import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LogisticRegression
from ast import literal_eval
from sklearn.metrics import classification_report
import gensim.downloader as api
from sklearn.preprocessing import scale


# sys.path.append('../')
sys.path.append('../8-training_binary/simple_transformers/')

from simpletransformers.classification import ClassificationModel


# from fast_bert.modeling import BertForMultiLabelSequenceClassification
# from fast_bert.data_cls import BertDataBunch, InputExample, InputFeatures, MultiLabelTextProcessor, \
#     convert_examples_to_features
# from fast_bert.learner_cls import BertLearner
# # from fast_bert.metrics import accuracy_multilabel, accuracy_thresh, fbeta, roc_auc, accuracy
# from fast_bert.metrics import *
import matplotlib.pyplot as plt

# torch.cuda.empty_cache()

pd.set_option('display.max_colwidth', -1)
run_start_time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

root_path='/scratch/da2734/twitter/jobs/running_on_200Msamples/'




def get_env_var(varname, default):
    if os.environ.get(varname) != None:
        var = int(os.environ.get(varname))
        print(varname, ':', var)
    else:
        var = default
        print(varname, ':', var, '(Default)')
    return var


# Choose Number of Nodes To Distribute Credentials: e.g. jobarray=0-4, cpu_per_task=20, credentials = 90 (<100)
SLURM_JOB_ID = get_env_var('SLURM_JOB_ID', 0)
SLURM_ARRAY_TASK_ID = get_env_var('SLURM_ARRAY_TASK_ID', 0)
SLURM_ARRAY_TASK_COUNT = get_env_var('SLURM_ARRAY_TASK_COUNT', 1)

# SLURM_JOB_ID = 123123123
# SLURM_ARRAY_TASK_ID = 10
# SLURM_ARRAY_TASK_COUNT = 500


print('SLURM_JOB_ID', SLURM_JOB_ID)
print('SLURM_ARRAY_TASK_ID', SLURM_ARRAY_TASK_ID)
print('SLURM_ARRAY_TASK_COUNT', SLURM_ARRAY_TASK_COUNT)


# ####################################################################################################################################
# # loading data
# ####################################################################################################################################

import time
import pyarrow.parquet as pq
from glob import glob
import os
import numpy as np

path_to_data='/scratch/spf248/twitter/data/classification/US/'


print('Load Random Tweets:')
# random contains 7.3G of data!!
start_time = time.time()

paths_to_random=list(np.array_split(
                        # glob(os.path.join(path_to_data,'random','*.parquet')),
                        glob(os.path.join(path_to_data,'random_10perct_sample','*.parquet')),
#                         glob(os.path.join(path_to_data,'random_1perct_sample','*.parquet')),
                        SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID])
print('#files:', len(paths_to_random))

tweets_random=pd.DataFrame()
for file in paths_to_random:
    print(file)
    tweets_random=pd.concat([tweets_random,pd.read_parquet(file)[['tweet_id','text']]])
    print(tweets_random.shape)

    
#     break
# tweets_random = tweets_random[:100]



print('time taken to load random sample:', str(time.time() - start_time), 'seconds')
print(tweets_random.shape)


print('dropping duplicates:')
# random contains 7.3G of data!!
start_time = time.time()
tweets_random = tweets_random.drop_duplicates('text')
print('time taken to load random sample:', str(time.time() - start_time), 'seconds')
print(tweets_random.shape)


for column in ["is_unemployed", "lost_job_1mo", "job_search", "is_hired_1mo", "job_offer"]:

    print('\n\n!!!!!', column)
#     print(x)

    start = time.time()
#     learner = create_model(column, best_epochs[column])
    model = ClassificationModel('bert', 
                            '/scratch/da2734/twitter/jobs/training_binary/simple_transformers_manu_bertbase/{}/'.format(column), 
                        args={'evaluate_during_training': True, 
                              'evaluate_during_training_verbose': True, 
                              'num_train_epochs': 20})
    print('load model:', str(time.time() - start_time), 'seconds')

#     print('Predictions of Filtered Tweets:')
#     start_time = time.time()
#     predictions_filtered = learner.predict_batch(tweets_filtered['text'].values.tolist())
#     print('time taken:', str(time.time() - start_time), 'seconds')
#     print('per tweet:', (time.time() - start_time)/tweets_filtered.shape[0], 'seconds')

#     # In[ ]:

#     data_path = "/scratch/da2734/twitter/data/may20_9Klabels/data_binary_pos_neg_balanced/"
#     print("************ {} ************".format(column))

#     train_file_name = "train_{}.csv".format(column)
#     val_file_name = "val_{}.csv".format(column)
#     #download data
#     df_train = pd.read_csv(os.path.join(data_path, train_file_name))
# #     print(df_train.head())
#     df_val = pd.read_csv(os.path.join(data_path, val_file_name))
#     #create embeddings
#     train_vecs_glove_mean = scale(np.concatenate([get_w2v_general(z, 200, glove_twitter,'mean') for z in df_train["text"]]))
#     validation_vecs_glove_mean = scale(np.concatenate([get_w2v_general(z, 200, glove_twitter,'mean') for z in df_val["text"]]))
#     #train
#     clf = LogisticRegression(max_iter=1000)
#     clf.fit(train_vecs_glove_mean,df_train["class"])
#     #evaluate
#     df_val["class_predict"] = clf.predict(validation_vecs_glove_mean)
#     TP, FP, TN, FN = perf_measure(df_val["class"], df_val["class_predict"])
#     print("Precision: ", TP/(TP+FP))
#     print("Recall: ", TP/(TP+FN))





    print('Predictions of Random Tweets:')
    start_time = time.time()
    #     predictions_random = learner.predict_batch(tweets_random['text'].values.tolist())
#     predictions_random = clf.predict_proba(random_data_vecs_glove_mean)
    predictions, predictions_random = model.predict(tweets_random['text'].values.tolist())
#     print(type(predictions_random))
    # print(predictions_random)

    print('time taken:', str(time.time() - start_time), 'seconds')
    print('per tweet:', (time.time() - start_time)/tweets_random.shape[0], 'seconds')

    # In[ ]:


    #     print('Save Predictions of Filtered Tweets:')
    #     start_time = time.time()



    #     df_filtered = predictions_filtered.set_index(tweets_filtered.tweet_id).rename(columns={
    #             '0':'pos_model',
    #             '1':'neg_model',
    #     })

    if not os.path.exists(os.path.join(root_path,'pred_output_10pct_sample_BERT', column)):
        print('>>>> directory doesnt exists, creating it')
        os.makedirs(os.path.join(root_path,'pred_output_10pct_sample_BERT', column))

    #     # if not os.path.exists(os.path.join(root_path,'pred_output_10pct_sample', column)):
    #     #     os.makedirs(os.path.join(root_path,'pred_output_10pct_sample', column))

    #     # if not os.path.exists(os.path.join(root_path,'pred_output_full', column)):
    #     #     os.makedirs(os.path.join(root_path,'pred_output_full', column))

    #     df_filtered.to_csv(
    #             # os.path.join(root_path,'pred_output', column, 'filtered'+'-'+str(SLURM_JOB_ID)+'-'+str(SLURM_ARRAY_TASK_ID)+'.csv')
    #             os.path.join(root_path,'pred_output_1pct_sample', column, 'filtered'+'-'+str(SLURM_JOB_ID)+'-'+str(SLURM_ARRAY_TASK_ID)+'.csv')
    #         )

    #     print(os.path.join(root_path,'pred_output_1pct_sample', column, 'filtered'+'-'+str(SLURM_JOB_ID)+'-'+str(SLURM_ARRAY_TASK_ID)+'.csv'), 'saved')

    #     print('time taken:', str(time.time() - start_time), 'seconds')


    print('Save Predictions of Random Tweets:')
    start_time = time.time()
    # predictions_random_df = pd.DataFrame(data=predictions_random, columns = ['neg', 'pos'])
    predictions_random_df = pd.DataFrame(data=predictions_random, columns = ['first', 'second'])
    df_random = predictions_random_df.set_index(tweets_random.tweet_id)
    # df_random = predictions_random.set_index(tweets_random.tweet_id).rename(columns={
    #         '0':'pos_model',
    #         '1':'neg_model',
    # })

    # if not os.path.exists(os.path.join(root_path,'pred_output_10pct_sample', column)):
    #     os.makedirs(os.path.join(root_path,'pred_output_10pct_sample', column))

    df_random.to_csv(
        # os.path.join(root_path,'pred_output', column, 'random'+'-'+str(SLURM_JOB_ID)+'-'+str(SLURM_ARRAY_TASK_ID)+'.csv')
        os.path.join(root_path,'pred_output_10pct_sample_BERT', column, 'random'+'-'+str(SLURM_JOB_ID)+'-'+str(SLURM_ARRAY_TASK_ID)+'.csv')
        )

#     print(os.path.join(root_path,'pred_output_1pct_sample_BERT', column, 'random'+'-'+str(SLURM_JOB_ID)+'-'+str(SLURM_ARRAY_TASK_ID)+'.csv'), 'saved')

    print('time taken:', str(time.time() - start_time), 'seconds')


#     break

    