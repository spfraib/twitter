#!/usr/bin/env python
# coding: utf-8

# In[ ]:

#!/usr/bin/env python
# coding: utf-8
# run using: sbatch --array=0-9 7.9-get-predictions-from-BERT.sh

import sys

column = sys.argv[1]


##################################################################################################################################
# loading the model
##################################################################################################################################


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

sys.path.append('../')

from fast_bert.modeling import BertForMultiLabelSequenceClassification
from fast_bert.data_cls import BertDataBunch, InputExample, InputFeatures, MultiLabelTextProcessor, \
    convert_examples_to_features
from fast_bert.learner_cls import BertLearner
# from fast_bert.metrics import accuracy_multilabel, accuracy_thresh, fbeta, roc_auc, accuracy
from fast_bert.metrics import *
import matplotlib.pyplot as plt

torch.cuda.empty_cache()

pd.set_option('display.max_colwidth', -1)
run_start_time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

root_path='/scratch/da2734/twitter/running_on_200Msamples/pred_output/'


def create_model(columnm, epoch):
#     if not os.path.exists('/scratch/da2734/twitter/running_on_200Msamples/log_running_on_samples_{}/'.format(column)):
#         os.makedirs('/scratch/da2734/twitter/running_on_200Msamples/log_running_on_samples_{}/'.format(column))

#     if not os.path.exists('/scratch/da2734/twitter/running_on_200Msamples/output_binary_{}'.format(column)):
#         os.makedirs('/scratch/da2734/twitter/running_on_200Msamples/output_binary_{}'.format(column))

    LOG_PATH = Path('/scratch/da2734/twitter/running_on_200Msamples/log_running_on_samples/'.format(column))
    DATA_PATH = Path('/scratch/da2734/twitter/mturk_mar6/data_binary_class_balanced_UNDERsampled/')
    LABEL_PATH = Path('/scratch/da2734/twitter/mturk_mar6/data_binary_class_balanced_UNDERsampled/')
    OUTPUT_PATH = Path(
        '/scratch/da2734/twitter/running_on_200Msamples/running_on_samples_output_binary_pos_neg_balanced_{}'.format(
            column))
    FINETUNED_PATH = None

    args = Box({
        "run_text": "labor mturk ar 6 binary",
        "train_size": -1,
        "val_size": -1,
        "log_path": LOG_PATH,
        "full_data_dir": DATA_PATH,
        "data_dir": DATA_PATH,
        "task_name": "labor_market_classification",
        "no_cuda": False,
        #     "bert_model": BERT_PRETRAINED_PATH,
        "output_dir": OUTPUT_PATH,
        "max_seq_length": 512,
        "do_train": True,
        "do_eval": True,
        "do_lower_case": True,
        "train_batch_size": 8,
        "eval_batch_size": 16,
        "learning_rate": 5e-5,
        "num_train_epochs": 100,
        "warmup_proportion": 0.0,
        "no_cuda": False,
        "local_rank": -1,
        "seed": 42,
        "gradient_accumulation_steps": 1,
        "optimize_on_cpu": False,
        "fp16": False,
        "fp16_opt_level": "O1",
        "weight_decay": 0.0,
        "adam_epsilon": 1e-8,
        "max_grad_norm": 1.0,
        "max_steps": -1,
        "warmup_steps": 500,
        "logging_steps": 50,
        "eval_all_checkpoints": True,
        "overwrite_output_dir": True,
        "overwrite_cache": True,
        "seed": 42,
        "loss_scale": 128,
        "task_name": 'intent',
        "model_name": 'bert-base-uncased',
        "model_type": 'bert'
    })

    import logging

    logfile = str(LOG_PATH / 'log-{}-{}.txt'.format(run_start_time, args["run_text"]))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler(sys.stdout)
        ])

    logger = logging.getLogger()

    logger.info(args)

    device = torch.device('cuda')
    if torch.cuda.device_count() > 1:
        args.multi_gpu = True
    else:
        args.multi_gpu = False

    label_cols = ['class']

    databunch = BertDataBunch(
        args['data_dir'],
        LABEL_PATH,
        args.model_name,
        train_file='train_{}.csv'.format(column),
        val_file='val_{}.csv'.format(column),
        label_file='label_{}.csv'.format(column),
        # test_data='test.csv',
        text_col="text",  # this is the name of the column in the train file that containts the tweet text
        label_col=label_cols,
        batch_size_per_gpu=args['train_batch_size'],
        max_seq_length=args['max_seq_length'],
        multi_gpu=args.multi_gpu,
        multi_label=False,
        model_type=args.model_type)

    num_labels = len(databunch.labels)
    print('num_labels', num_labels)

    print('time taken to load all this stuff:', str(time.time() - start_time), 'seconds')

    # metrics defined: https://github.com/kaushaltrivedi/fast-bert/blob/d89e2aa01d948d6d3cdea7ad106bf5792fea7dfa/fast_bert/metrics.py
    metrics = []
    # metrics.append({'name': 'accuracy_thresh', 'function': accuracy_thresh})
    # metrics.append({'name': 'roc_auc', 'function': roc_auc})
    # metrics.append({'name': 'fbeta', 'function': fbeta})
    metrics.append({'name': 'accuracy', 'function': accuracy})
#     metrics.append({'name': 'roc_auc_save_to_plot_binary', 'function': roc_auc_save_to_plot_binary})
    # metrics.append({'name': 'accuracy_multilabel', 'function': accuracy_multilabel})

    learner = BertLearner.from_pretrained_model(
        databunch,
        pretrained_path='../mturk_mar6/output_binary_pos_neg_balanced_{}/model_out_{}/'.format(column, epoch),
        metrics=metrics,
        device=device,
        logger=logger,
        output_dir=args.output_dir,
        finetuned_wgts_path=FINETUNED_PATH,
        warmup_steps=args.warmup_steps,
        multi_gpu=args.multi_gpu,
        is_fp16=args.fp16,
        multi_label=False,
        logging_steps=0)

    return learner


best_epochs = {
    'is_hired_1mo': 10,
    'lost_job_1mo': 9,
    'job_offer': 5,
    'is_unemployed': 6,
    'job_search': 8
}

start = time.time()
learner = create_model(column, best_epochs[column])
print('load model:', str(time.time() - start_time), 'seconds')



import time
import pyarrow.parquet as pq
from glob import glob
import os
import numpy as np


# In[ ]:


def get_env_var(varname,default):
    
    if os.environ.get(varname) != None:
        var = int(os.environ.get(varname))
        print(varname,':', var)
    else:
        var = default
        print(varname,':', var,'(Default)')
    return var

# Choose Number of Nodes To Distribute Credentials: e.g. jobarray=0-4, cpu_per_task=20, credentials = 90 (<100)
SLURM_JOB_ID            = get_env_var('SLURM_JOB_ID',0)
SLURM_ARRAY_TASK_ID     = get_env_var('SLURM_ARRAY_TASK_ID',0)
SLURM_ARRAY_TASK_COUNT  = get_env_var('SLURM_ARRAY_TASK_COUNT',1)


path_to_data='/scratch/spf248/twitter/data/classification/US/'





print('Load Filtered Tweets:')
# filtered contains 8G of data!!
start_time = time.time()

paths_to_filtered=list(np.array_split(
glob(os.path.join(path_to_data,'filtered','*.parquet')),SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID])
print('#files:', len(paths_to_filtered))

# temp = pd.read_parquet('/scratch/spf248/twitter/data/classification/US/random/part-02175-1c1e6466-49fa-411b-beb0-276d14cdffab-c000.snappy.parquet')
# print('size of temp file', temp.shape)

print('number of splits', len(np.array_split(
                        glob(os.path.join(path_to_data,'filtered','*.parquet')),
                        SLURM_ARRAY_TASK_COUNT)))

print('number of files IN split', len(np.array_split(
                        glob(os.path.join(path_to_data,'filtered','*.parquet')),
                        SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID]))


tweets_filtered=pd.DataFrame()
print('huh')
for file in paths_to_filtered:
    print(file)
    print('size of tweets_filtered', tweets_filtered.shape)
    current_file = pd.read_parquet(file)
    print('size of current file', current_file.shape)
    tweets_filtered=pd.concat([tweets_filtered,current_file])
    break

print('time taken to load keyword filtered sample:', str(time.time() - start_time), 'seconds')
print(tweets_filtered.shape)


print('Predictions of Filtered Tweets:')
start_time = time.time()
predictions_filtered = learner.predict_batch(tweets_filtered['text'].values.tolist())
print('time taken:', str(time.time() - start_time), 'seconds')

print('Save Predictions of Filtered Tweets:')
start_time = time.time()

predictions_filtered.set_index(tweets_filtered.tweet_id).rename(columns={
        '0':'pos_model',
        '1':'neg_model',
})

df_filtered.to_csv(
os.path.join(root_path,'pred','filtered'+'-'+str(SLURM_JOB_ID)+'-'+str(SLURM_ARRAY_TASK_ID)+'.csv'))

print('time taken:', str(time.time() - start_time), 'seconds')

del paths_to_filtered
del predictions_filtered
del df_filtered
del tweets_filtered







# print('Load Random Tweets:')
# # random contains 7.3G of data!!
# start_time = time.time()

# paths_to_random=list(np.array_split(
# glob(os.path.join(path_to_data,'random','*.parquet')),SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID])
# print('#files:', len(paths_to_random))

# tweets_random=pd.DataFrame()
# for file in paths_to_random:
#     print(file)
#     tweets_random=pd.concat([tweets_random,pd.read_parquet(file)[['tweet_id','text']]])
#     break

# print('time taken to load random sample:', str(time.time() - start_time), 'seconds')
# print(tweets_random.shape)



# print('Predictions of Random Tweets:')
# start_time = time.time()
# predictions_random = learner.predict_batch(tweets_random['text'].values.tolist())
# print('time taken:', str(time.time() - start_time), 'seconds')



# print('Save Predictions of Random Tweets:')
# start_time = time.time()

# predictions_random.set_index(tweets_random.tweet_id).rename(columns={
#         '0':'pos_model',
#         '1':'neg_model',
# })

# df_random.to_csv(
# os.path.join(root_path,'pred','random'+'-'+str(SLURM_JOB_ID)+'-'+str(SLURM_ARRAY_TASK_ID)+'.csv'))

# print('time taken:', str(time.time() - start_time), 'seconds')

