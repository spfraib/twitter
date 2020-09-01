#!/usr/bin/env python
# coding: utf-8
# run using: sbatch --array=0-9 7.9-get-predictions-from-BERT.sh

print('started 10.2-GLOVE-deploying-10M-filteredONLY_1pct.py')

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

import matplotlib.pyplot as plt
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
# torch.cuda.empty_cache()

pd.set_option('display.max_colwidth', -1)
run_start_time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
root_path= '/archive/jobs/running_on_200Msamples/'




def get_env_var(varname, default):
    if os.environ.get(varname) != None:
        var = int(os.environ.get(varname))
        print(varname, ':', var)
    else:
        var = default
        print(varname, ':', var, '(Default)')
    return var


# Choose Number of Nodes To Distribute Credentials: e.g. jobarray=0-4, cpu_per_task=20, credentials = 90 (<100)
#SLURM_JOB_ID = get_env_var('SLURM_JOB_ID', 0)
#SLURM_ARRAY_TASK_ID = get_env_var('SLURM_ARRAY_TASK_ID', 0)
#SLURM_ARRAY_TASK_COUNT = get_env_var('SLURM_ARRAY_TASK_COUNT', 1)

# SLURM_JOB_ID = 123123123
# SLURM_ARRAY_TASK_ID = 10
# SLURM_ARRAY_TASK_COUNT = 500


#print('SLURM_JOB_ID', SLURM_JOB_ID)
#print('SLURM_ARRAY_TASK_ID', SLURM_ARRAY_TASK_ID)
#print('SLURM_ARRAY_TASK_COUNT', SLURM_ARRAY_TASK_COUNT)


# ####################################################################################################################################
# # loading data
# ####################################################################################################################################

import time
import pyarrow.parquet as pq
from glob import glob
import os
import numpy as np

path_to_data='/home/manuto/Documents/world_bank/bert_twitter_labor/data/glove_prediction_data'


# print('Load Filtered Tweets:')
# # filtered contains 8G of data!!
# start_time = time.time()

# paths_to_filtered=list(np.array_split(
#                         # glob(os.path.join(path_to_data,'filtered','*.parquet')),
#                         # glob(os.path.join(path_to_data,'filtered_10perct_sample','*.parquet')),
#                         glob(os.path.join(path_to_data,'filtered_1perct_sample','*.parquet')),
#                         SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID]
#                        )
# print('#files:', len(paths_to_filtered))

# tweets_filtered=pd.DataFrame()
# for file in paths_to_filtered:
#     print(file)
#     tweets_filtered=pd.concat([tweets_filtered,pd.read_parquet(file)[['tweet_id','text']]])
#     print(tweets_filtered.shape)

#     # break

# # tweets_filtered = tweets_filtered[:100]

# print('time taken to load keyword filtered sample:', str(time.time() - start_time), 'seconds')
# print(tweets_filtered.shape)


print('Load filtered Tweets:')
# filtered contains 7.3G of data!!
start_time = time.time()

paths_to_filtered=glob(os.path.join(path_to_data,'filtered_10perct_sample_2','*.parquet'))
print('#files:', len(paths_to_filtered))

tweets_filtered=pd.DataFrame()
for file in paths_to_filtered:
    print(file)
    tweets_filtered=pd.concat([tweets_filtered,pd.read_parquet(file)[['tweet_id','text']]])
    print(tweets_filtered.shape)

#     break

# tweets_filtered = tweets_filtered[:100]

print('time taken to load filtered sample:', str(time.time() - start_time), 'seconds')
print(tweets_filtered.shape)


def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

def get_w2v_general(tweet, size, vectors, aggregation='mean'):
    vec = np.zeros(size).reshape((1, size))
    count = 0.
    for word in tweet.split():
        try:
            vec += vectors[word.lower()].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if aggregation == 'mean':
        if count != 0:
            vec /= count
        return vec
    elif aggregation == 'sum':
        return vec

#text preprocessing pipeline
print("Setting up text preprocessing pipeline")
text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
               'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated",
              'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens

    # corpus from which the word statistics are going to be used
    # for word segmentation
    segmenter="twitter",

    # corpus from which the word statistics are going to be used
    # for spell correction
    corrector="twitter",

    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=False,  # spell correction for elongated words

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,

    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons]
)

def ekphrasis_preprocessing(tweet):
    return " ".join(text_processor.pre_process_doc(tweet))
tqdm.pandas()
print("Starting preprocessing filtered tweets")
tweets_filtered['ekphrasis_text'] = tweets_filtered['text'].progress_apply(ekphrasis_preprocessing)
print('time taken:', str(time.time() - start_time), 'seconds')


import os
#print('GENSIM_DATA_DIR', os.environ['GENSIM_DATA_DIR'] )
    
start_time = time.time()
print('loading glove')
glove_twitter = api.load("glove-twitter-200")
print('time taken:', str(time.time() - start_time), 'seconds')    

start_time = time.time()
print('calculating embeddings')    
filtered_data_vecs_glove_mean = scale(np.concatenate([get_w2v_general(z, 200, glove_twitter,'mean') for z in tqdm(tweets_filtered["text"])]))
print('time taken:', str(time.time() - start_time), 'seconds')
print('per tweet:', (time.time() - start_time)/tweets_filtered.shape[0], 'seconds')


for column in ["is_unemployed", "lost_job_1mo", "job_search", "is_hired_1mo", "job_offer"]:

    print('\n\n!!!!!', column)

#     start = time.time()
#     learner = create_model(column, best_epochs[column])
#     print('load model:', str(time.time() - start_time), 'seconds')

#     print('Predictions of Filtered Tweets:')
#     start_time = time.time()
#     predictions_filtered = learner.predict_batch(tweets_filtered['text'].values.tolist())
#     print('time taken:', str(time.time() - start_time), 'seconds')
#     print('per tweet:', (time.time() - start_time)/tweets_filtered.shape[0], 'seconds')

#     # In[ ]:

    data_path = "/home/manuto/Documents/world_bank/bert_twitter_labor/code/twitter/data/may20_9Klabels/data_binary_pos_neg_balanced/preprocessed_glove/"
    print("************ {} ************".format(column))

    train_file_name = "train_{}.csv".format(column)
    val_file_name = "val_{}.csv".format(column)
    #download data
    df_train = pd.read_csv(os.path.join(data_path, train_file_name))
#     print(df_train.head())
    df_val = pd.read_csv(os.path.join(data_path, val_file_name))
    #preprocess text
    df_train['ekphrasis_text'] = df_train['text'].apply(ekphrasis_preprocessing)
    df_val['ekphrasis_text'] = df_val['text'].apply(ekphrasis_preprocessing)
    #create embeddings
    train_vecs_glove_mean = scale(np.concatenate([get_w2v_general(z, 200, glove_twitter,'mean') for z in df_train["ekphrasis_text"]]))
    validation_vecs_glove_mean = scale(np.concatenate([get_w2v_general(z, 200, glove_twitter,'mean') for z in df_val["ekphrasis_text"]]))
    #train
    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_vecs_glove_mean,df_train["class"])
    #evaluate
    df_val["class_predict"] = clf.predict(validation_vecs_glove_mean)
    TP, FP, TN, FN = perf_measure(df_val["class"], df_val["class_predict"])
    print("Precision: ", TP/(TP+FP))
    print("Recall: ", TP/(TP+FN))





    print('Predictions of filtered Tweets:')
    start_time = time.time()
    #     predictions_filtered = learner.predict_batch(tweets_filtered['text'].values.tolist())
    predictions_filtered = clf.predict_proba(filtered_data_vecs_glove_mean)

#     print(type(predictions_filtered))
    # print(predictions_filtered)

    print('time taken:', str(time.time() - start_time), 'seconds')
    print('per tweet:', (time.time() - start_time)/tweets_filtered.shape[0], 'seconds')

    # In[ ]:


    #     print('Save Predictions of Filtered Tweets:')
    #     start_time = time.time()



    #     df_filtered = predictions_filtered.set_index(tweets_filtered.tweet_id).rename(columns={
    #             '0':'pos_model',
    #             '1':'neg_model',
    #     })

    # if not os.path.exists(os.path.join(root_path,'pred_output_1pct_sample_GLOVE', column)):
    if not os.path.exists(os.path.join(root_path,'pred_output_10pct_sample_GLOVE', column)):
        print('>>>> directory doesnt exists, creating it')
        # os.makedirs(os.path.join(root_path,'pred_output_1pct_sample_GLOVE', column))
        os.makedirs(os.path.join(root_path,'pred_output_10pct_sample_GLOVE', column))

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






    print('Save Predictions of filtered Tweets:')
    start_time = time.time()
    # predictions_filtered_df = pd.DataFrame(data=predictions_filtered, columns = ['neg', 'pos'])
    predictions_filtered_df = pd.DataFrame(data=predictions_filtered, columns = ['glove_neg_model', 'glove_pos_model'])
    df_filtered = predictions_filtered_df.set_index(tweets_filtered.tweet_id)
    # df_filtered = predictions_filtered.set_index(tweets_filtered.tweet_id).rename(columns={
    #         '0':'pos_model',
    #         '1':'neg_model',
    # })

    # if not os.path.exists(os.path.join(root_path,'pred_output_10pct_sample', column)):
    #     os.makedirs(os.path.join(root_path,'pred_output_10pct_sample', column))

    df_filtered.to_csv(
        # os.path.join(root_path,'pred_output', column, 'filtered'+'-'+str(SLURM_JOB_ID)+'-'+str(SLURM_ARRAY_TASK_ID)+'.csv')
        os.path.join(root_path,'pred_output_10pct_sample_GLOVE', column, 'filtered'+ '-' + 'manu_2' +'.csv')
        # os.path.join(root_path,'pred_output_1pct_sample_GLOVE', column, 'filtered'+'-'+str(SLURM_JOB_ID)+'-'+str(SLURM_ARRAY_TASK_ID)+'.csv')
        )


    print('time taken:', str(time.time() - start_time), 'seconds')


#     break

    
