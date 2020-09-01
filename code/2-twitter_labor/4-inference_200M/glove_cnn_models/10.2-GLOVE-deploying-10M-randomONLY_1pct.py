#!/usr/bin/env python
# coding: utf-8
# run using: sbatch --array=0-9 7.9-get-predictions-from-BERT.sh

print('started 10.2-GLOVE-deploying-1M-randomONLY_1pct.py')

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

# from fast_bert.modeling import BertForMultiLabelSequenceClassification
# from fast_bert.data_cls import BertDataBunch, InputExample, InputFeatures, MultiLabelTextProcessor, \
#     convert_examples_to_features
# from fast_bert.learner_cls import BertLearner
# # from fast_bert.metrics import accuracy_multilabel, accuracy_thresh, fbeta, roc_auc, accuracy
# from fast_bert.metrics import *
import matplotlib.pyplot as plt
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
# torch.cuda.empty_cache()

pd.set_option('display.max_colwidth', -1)
run_start_time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

root_path= '/archive/jobs/running_on_200Msamples/'




# def create_model(columnm, epoch):
#     if not os.path.exists('/scratch/da2734/twitter/jobs/running_on_200Msamples/logs/log_binary_pos_neg_{}/'.format(column)):
#         os.makedirs('/scratch/da2734/twitter/jobs/running_on_200Msamples/logs/log_binary_pos_neg_{}/'.format(column))

#     LOG_PATH = Path('/scratch/da2734/twitter/jobs/running_on_200Msamples/logs/log_binary_pos_neg_{}/'.format(column))
#     print('LOG_PATH', LOG_PATH)
#     DATA_PATH = Path('/scratch/da2734/twitter/data/may20_9Klabels/data_binary_pos_neg_balanced/')
#     LABEL_PATH = Path('/scratch/da2734/twitter/data/may20_9Klabels/data_binary_pos_neg_balanced/')
#     OUTPUT_PATH = Path('/scratch/da2734/twitter/jobs/training_binary/models_may20_9Klabels/output_{}'.format(column))
#     FINETUNED_PATH = None

#     args = Box({
#         "run_text": "100Msamples",
#         "train_size": -1,
#         "val_size": -1,
#         "log_path": LOG_PATH,
#         "full_data_dir": DATA_PATH,
#         "data_dir": DATA_PATH,
#         "task_name": "labor_market_classification",
#         "no_cuda": False,
#         #     "bert_model": BERT_PRETRAINED_PATH,
#         "output_dir": OUTPUT_PATH,
#         "max_seq_length": 512,
#         "do_train": True,
#         "do_eval": True,
#         "do_lower_case": True,
#         "train_batch_size": 8,
#         "eval_batch_size": 16,
#         "learning_rate": 5e-5,
#         "num_train_epochs": 100,
#         "warmup_proportion": 0.0,
#         "no_cuda": False,
#         "local_rank": -1,
#         "seed": 42,
#         "gradient_accumulation_steps": 1,
#         "optimize_on_cpu": False,
#         "fp16": False,
#         "fp16_opt_level": "O1",
#         "weight_decay": 0.0,
#         "adam_epsilon": 1e-8,
#         "max_grad_norm": 1.0,
#         "max_steps": -1,
#         "warmup_steps": 500,
#         "logging_steps": 50,
#         "eval_all_checkpoints": True,
#         "overwrite_output_dir": True,
#         "overwrite_cache": True,
#         "seed": 42,
#         "loss_scale": 128,
#         "task_name": 'intent',
#         "model_name": 'bert-base-uncased',
#         "model_type": 'bert'
#     })

#     import logging

#     logfile = str(LOG_PATH / 'log-{}-{}.txt'.format(run_start_time, args["run_text"]))

#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#         datefmt='%m/%d/%Y %H:%M:%S',
#         handlers=[
#             logging.FileHandler(logfile),
#             logging.StreamHandler(sys.stdout)
#         ])

#     logger = logging.getLogger()

#     logger.info(args)

#     device = torch.device('cuda')
#     if torch.cuda.device_count() > 1:
#         args.multi_gpu = True
#     else:
#         args.multi_gpu = False

#     label_cols = ['class']

#     databunch = BertDataBunch(
#         args['data_dir'],
#         LABEL_PATH,
#         args.model_name,
#         train_file='train_{}.csv'.format(column),
#         val_file='val_{}.csv'.format(column),
#         label_file='label_{}.csv'.format(column),
#         # test_data='test.csv',
#         text_col="text",  # this is the name of the column in the train file that containts the tweet text
#         label_col=label_cols,
#         batch_size_per_gpu=args['train_batch_size'],
#         max_seq_length=args['max_seq_length'],
#         multi_gpu=args.multi_gpu,
#         multi_label=False,
#         model_type=args.model_type)

#     num_labels = len(databunch.labels)
#     print('num_labels', num_labels)

#     print('time taken to load all this stuff:', str(time.time() - start_time), 'seconds')

#     # metrics defined: https://github.com/kaushaltrivedi/fast-bert/blob/d89e2aa01d948d6d3cdea7ad106bf5792fea7dfa/fast_bert/metrics.py
#     metrics = []
#     # metrics.append({'name': 'accuracy_thresh', 'function': accuracy_thresh})
#     # metrics.append({'name': 'roc_auc', 'function': roc_auc})
#     # metrics.append({'name': 'fbeta', 'function': fbeta})
#     metrics.append({'name': 'accuracy', 'function': accuracy})
#     metrics.append({'name': 'roc_auc_save_to_plot_binary', 'function': roc_auc_save_to_plot_binary})
#     # metrics.append({'name': 'accuracy_multilabel', 'function': accuracy_multilabel})

#     learner = BertLearner.from_pretrained_model(
#         databunch,
#         pretrained_path='/scratch/da2734/twitter/jobs/training_binary/models_may20_9Klabels/output_{}/model_out_{}/'.format(column, epoch),
#         metrics=metrics,
#         device=device,
#         logger=logger,
#         output_dir=args.output_dir,
#         finetuned_wgts_path=FINETUNED_PATH,
#         warmup_steps=args.warmup_steps,
#         multi_gpu=args.multi_gpu,
#         is_fp16=args.fp16,
#         multi_label=False,
#         logging_steps=0)

#     return learner


# best_epochs = {
#     'is_hired_1mo':8,
#     'lost_job_1mo':5,
#     'job_offer':4,
#     'is_unemployed':3,
#     'job_search':6
# }

# best_epochs = {
#     'is_hired_1mo':6,
#     'lost_job_1mo':4,
#     'job_offer':3,
#     'is_unemployed':3,
#     'job_search':4
# }


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


print('Load Random Tweets:')
# random contains 7.3G of data!!
start_time = time.time()

paths_to_random=glob(os.path.join(path_to_data,'random_10perct_sample_2','*.parquet'))
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
            vec += vectors[word].reshape((1, size))
            count += 1.
        except KeyError:
            continue
    if aggregation == 'mean':
        if count != 0:
            vec /= count
        return vec
    elif aggregation == 'sum':
        return vec

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
print("Starting preprocessing random tweets")
tweets_random['ekphrasis_text'] = tweets_random['text'].progress_apply(ekphrasis_preprocessing)
print('time taken:', str(time.time() - start_time), 'seconds')

import os
#print('GENSIM_DATA_DIR', os.environ['GENSIM_DATA_DIR'] )
    
start_time = time.time()
print('loading glove')
glove_twitter = api.load("glove-twitter-200")
print('time taken:', str(time.time() - start_time), 'seconds')    

start_time = time.time()
print('calculating embeddings')    
random_data_vecs_glove_mean = scale(np.concatenate([get_w2v_general(z, 200, glove_twitter,'mean') for z in tweets_random["ekphrasis_text"]]))
print('time taken:', str(time.time() - start_time), 'seconds')
print('per tweet:', (time.time() - start_time)/tweets_random.shape[0], 'seconds')


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





    print('Predictions of Random Tweets:')
    start_time = time.time()
    #     predictions_random = learner.predict_batch(tweets_random['text'].values.tolist())
    predictions_random = clf.predict_proba(random_data_vecs_glove_mean)

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






    print('Save Predictions of Random Tweets:')
    start_time = time.time()
    # predictions_random_df = pd.DataFrame(data=predictions_random, columns = ['neg', 'pos'])
    predictions_random_df = pd.DataFrame(data=predictions_random, columns = ['glove_neg_model', 'glove_pos_model'])
    df_random = predictions_random_df.set_index(tweets_random.tweet_id)
    # df_random = predictions_random.set_index(tweets_random.tweet_id).rename(columns={
    #         '0':'pos_model',
    #         '1':'neg_model',
    # })

    # if not os.path.exists(os.path.join(root_path,'pred_output_10pct_sample', column)):
    #     os.makedirs(os.path.join(root_path,'pred_output_10pct_sample', column))

    df_random.to_csv(
        # os.path.join(root_path,'pred_output', column, 'random'+'-'+str(SLURM_JOB_ID)+'-'+str(SLURM_ARRAY_TASK_ID)+'.csv')
        os.path.join(root_path,'pred_output_10pct_sample_GLOVE', column, 'random'+'_' + 'manu_2' +'.csv')
        # os.path.join(root_path,'pred_output_1pct_sample_GLOVE', column, 'random'+'-'+str(SLURM_JOB_ID)+'-'+str(SLURM_ARRAY_TASK_ID)+'.csv')
        )


    print('time taken:', str(time.time() - start_time), 'seconds')


#     break

    