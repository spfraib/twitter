#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#gets all this setup
import time
start_time = time.time()
from transformers import BertTokenizer
from pathlib import Path
import torch

from box import Box
import pandas as pd
import collections
import os
from tqdm import tqdm, trange
import sys
import random
import numpy as np
# import apex
from sklearn.model_selection import train_test_split

import datetime

from fast_bert.modeling import BertForMultiLabelSequenceClassification
from fast_bert.data_cls import BertDataBunch, InputExample, InputFeatures, MultiLabelTextProcessor, convert_examples_to_features
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import accuracy_multilabel, accuracy_thresh, fbeta, roc_auc, accuracy
from fast_bert.metrics import *

torch.cuda.empty_cache()

pd.set_option('display.max_colwidth', -1)
run_start_time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

# LOG_PATH=Path('/scratch/da2734/twitter/sana/log/')
# DATA_PATH=Path('/scratch/da2734/twitter/sana/data')
# LABEL_PATH=Path('/scratch/da2734/twitter/sana/data/')
# OUTPUT_PATH=Path('/scratch/da2734/twitter/sana/output/')
root_path='/scratch/spf248/twitter/data/classification/US/BERT/twitter_sam/mturk_mar6/'
LOG_PATH=Path(root_path+'log/')
DATA_PATH=Path(root_path+'data')
LABEL_PATH=Path(root_path+'data/')
OUTPUT_PATH=Path(root_path+'output_100')
FINETUNED_PATH = None

args = Box({
    "run_text": "multilabel toxic comments with freezable layers",
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
    "eval_batch_size": 200,
    "learning_rate": 5e-5,
    "num_train_epochs": 6,
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
    "overwrite_cache": False,
    "seed": 42,
    "loss_scale": 128,
    "task_name": 'intent',
    "model_name": 'bert-base-uncased',
    "model_type": 'bert'
})

import logging

logfile = str(LOG_PATH/'log-{}-{}.txt'.format(run_start_time, args["run_text"]))

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

label_cols = ["job_loss","is_unemployed","job_search","is_hired","job_offer"]

# databunch defined here https://github.com/kaushaltrivedi/fast-bert/blob/master/fast_bert/data_cls.py
databunch = BertDataBunch(
                        args['data_dir'], 
                        LABEL_PATH, 
                        args.model_name, 
                        train_file='train.csv', 
                        val_file='val.csv',
                        # test_data='test.csv',
                        text_col="text", #this is the name of the column in the train file that containts the tweet text
                        label_col=label_cols,
                        batch_size_per_gpu=args['train_batch_size'], 
                        max_seq_length=args['max_seq_length'], 
                        multi_gpu=args.multi_gpu, 
                        multi_label=True, 
                        model_type=args.model_type)

num_labels = len(databunch.labels)
print('num_labels', num_labels)

print('__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)
print('__CUDA VERSION')
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
print('__Devices')
print('Active CUDA Device: GPU', torch.cuda.current_device())

print ('Available devices ', torch.cuda.device_count())
# print ('Current cuda device ', torch.cuda.current_device)

# metrics defined: https://github.com/kaushaltrivedi/fast-bert/blob/d89e2aa01d948d6d3cdea7ad106bf5792fea7dfa/fast_bert/metrics.py
metrics = []
metrics.append({'name': 'accuracy_thresh', 'function': accuracy_thresh})
metrics.append({'name': 'roc_auc', 'function': roc_auc})
# metrics.append({'name': 'roc_auc_save_to_plot', 'function': roc_auc_save_to_plot})
metrics.append({'name': 'fbeta', 'function': fbeta})
metrics.append({'name': 'accuracy', 'function': accuracy})
metrics.append({'name': 'accuracy_multilabel', 'function': accuracy_multilabel})


learner = BertLearner.from_pretrained_model(
                                            databunch, 
                                            pretrained_path='/scratch/da2734/twitter/mturk_mar6/output_100/model_out/', 
                                            metrics=metrics, 
                                            device=device, 
                                            logger=logger, 
                                            output_dir=args.output_dir, 
                                            finetuned_wgts_path=FINETUNED_PATH, 
                                            warmup_steps=args.warmup_steps,
                                            multi_gpu=args.multi_gpu, 
                                            is_fp16=args.fp16, 
                                            multi_label=True, 
                                            logging_steps=0)

print('time taken to load all this stuff:', str(time.time() - start_time), 'seconds')


# In[ ]:


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


# In[ ]:


path_to_data='/scratch/spf248/twitter/data/classification/US/'


# In[ ]:


print('Load Filtered Tweets:')
# filtered contains 8G of data!!
start_time = time.time()

paths_to_filtered=list(np.array_split(
glob(os.path.join(path_to_data,'filtered','*.parquet')),SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID])
print('#files:', len(paths_to_filtered))

tweets_filtered=pd.DataFrame()
for file in paths_to_filtered:
    tweets_filtered=pd.concat([tweets_filtered,pd.read_parquet(file)[['tweet_id','text']]])

print('time taken to load keyword filtered sample:', str(time.time() - start_time), 'seconds')
print(tweets_filtered.shape)


# In[ ]:


print('Load Random Tweets:')
# random contains 7.3G of data!!
start_time = time.time()

paths_to_random=list(np.array_split(
glob(os.path.join(path_to_data,'random','*.parquet')),SLURM_ARRAY_TASK_COUNT)[SLURM_ARRAY_TASK_ID])
print('#files:', len(paths_to_random))

tweets_random=pd.DataFrame()
for file in paths_to_random:
    tweets_random=pd.concat([tweets_random,pd.read_parquet(file)[['tweet_id','text']]])

print('time taken to load random sample:', str(time.time() - start_time), 'seconds')
print(tweets_random.shape)


# In[ ]:


print('Predictions of Filtered Tweets:')
start_time = time.time()
predictions_filtered = learner.predict_batch(tweets_filtered['text'].values.tolist())
print('time taken:', str(time.time() - start_time), 'seconds')


# In[ ]:


print('Predictions of Random Tweets:')
start_time = time.time()
predictions_random = learner.predict_batch(tweets_random['text'].values.tolist())
print('time taken:', str(time.time() - start_time), 'seconds')


# In[ ]:


print('Save Predictions of Filtered Tweets:')
start_time = time.time()

df_filtered = pd.DataFrame(
[dict(prediction) for prediction in predictions_filtered],
index=tweets_filtered.tweet_id).rename(columns={
'is_unemployed':'unemployed',
'job_search':'search',
'is_hired_1mo':'hired',
'lost_job_1mo':'loss',
'job_offer"':'offer',
})

df_filtered.to_csv(
os.path.join(root_path,'pred','filtered'+'-'+str(SLURM_JOB_ID)+'-'+str(SLURM_ARRAY_TASK_ID)+'.csv'))

print('time taken:', str(time.time() - start_time), 'seconds')


# In[ ]:


print('Save Predictions of Random Tweets:')
start_time = time.time()

df_random = pd.DataFrame(
[dict(prediction) for prediction in predictions_random],
index=tweets_random.tweet_id).rename(columns={
'is_unemployed':'unemployed',
'job_search':'search',
'is_hired_1mo':'hired',
'lost_job_1mo':'loss',
'job_offer"':'offer',
})

df_random.to_csv(
os.path.join(root_path,'pred','random'+'-'+str(SLURM_JOB_ID)+'-'+str(SLURM_ARRAY_TASK_ID)+'.csv'))

print('time taken:', str(time.time() - start_time), 'seconds')

