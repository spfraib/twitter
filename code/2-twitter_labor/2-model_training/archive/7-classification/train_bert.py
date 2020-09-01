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

import sys
sys.path.append('../')

from fast_bert.modeling import BertForMultiLabelSequenceClassification
from fast_bert.data_cls import BertDataBunch, InputExample, InputFeatures, \
MultiLabelTextProcessor, convert_examples_to_features
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import *

torch.cuda.empty_cache()

pd.set_option('display.max_colwidth', -1)
run_start_time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

LOG_PATH=Path('/scratch/da2734/twitter/mturk_mar6/log/')
DATA_PATH=Path('/scratch/da2734/twitter/mturk_mar6/data')
LABEL_PATH=Path('/scratch/da2734/twitter/mturk_mar6/data/')
OUTPUT_PATH=Path('/scratch/da2734/twitter/mturk_mar6/output_every_epoch/')

# LOG_PATH=Path('../mturk_mar6/log/')
# DATA_PATH=Path('../mturk_mar6/data')
# LABEL_PATH=Path('../mturk_mar6/data/')
# OUTPUT_PATH=Path('../mturk_mar6/output_every_epoch/')

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
    "eval_batch_size": 16,
    "learning_rate": 5e-5,
    "num_train_epochs": 100,
    "warmup_proportion": 0.0,
#     "no_cuda": False,
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

# label_cols = ["job_loss","is_unemployed","job_search","is_hired","job_offer"]
label_cols = ["is_unemployed", "lost_job_1mo", "job_search","is_hired_1mo","job_offer"]

databunch = BertDataBunch(
                        args['data_dir'],
                        LABEL_PATH,
                        args.model_name,
                        train_file='train.csv',
                        val_file='val.csv',
                        # test_data='test.csv',
                        #name of the column in the train file that containts the tweet text
                        text_col="text",
                        label_col=label_cols,
                        batch_size_per_gpu=args['train_batch_size'],
                        max_seq_length=args['max_seq_length'],
                        multi_gpu=args.multi_gpu,
                        multi_label=True,
                        model_type=args.model_type)

num_labels = len(databunch.labels)
print('num_labels', num_labels)


print('time taken to load all this stuff:', str(time.time() - start_time), 'seconds')

# metrics defined: https://github.com/kaushaltrivedi/fast-bert/blob/d89e2aa01d948d6d3cdea7ad106bf5792fea7dfa/fast_bert/metrics.py
metrics = []
metrics.append({'name': 'accuracy_thresh', 'function': accuracy_thresh})
metrics.append({'name': 'roc_auc', 'function': roc_auc})
metrics.append({'name': 'fbeta', 'function': fbeta})
metrics.append({'name': 'accuracy', 'function': accuracy})
metrics.append({'name': 'accuracy_multilabel', 'function': accuracy_multilabel})

learner = BertLearner.from_pretrained_model(
                                            databunch,
                                            pretrained_path=args.model_name,
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

learner.fit(args.num_train_epochs, args.learning_rate, validate=True) #this trains the model