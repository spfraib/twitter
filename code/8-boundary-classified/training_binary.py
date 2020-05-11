# gets all this setup
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
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
from fast_bert.modeling import BertForMultiLabelSequenceClassification
from fast_bert.data_cls import BertDataBunch, InputExample, InputFeatures, MultiLabelTextProcessor, \
    convert_examples_to_features
from fast_bert.learner_cls import BertLearner
from fast_bert.metrics import *


def get_args_from_command_line():
    """Parse the command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", type=str, help="dhaval or manu")
    parser.add_argument("--model_name", type=str, help="The name of the BERT model in the HuggingFace repo")
    parser.add_argument("--num_train_epochs", type=int, help="Number of epochs")
    parser.add_argument("--input_data_folder", type=str)
    parser.add_argument("--results_folder", type=str)
    parser.add_argument("--training_description", type=str)
    parser.add_argument("--label", type=str)
    args = parser.parse_args()
    return args

pre_args = get_args_from_command_line()


print(pre_args.label, 'creating model and loading..')

torch.cuda.empty_cache()

pd.set_option('display.max_colwidth', -1)
run_start_time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

if pre_args.user == "dhaval":
    if not os.path.exists('/scratch/da2734/twitter/jobs/training_binary/logs/log_binary_pos_neg_{}/'.format(pre_args.label)):
        os.makedirs('/scratch/da2734/twitter/jobs/training_binary/logs/log_binary_pos_neg_{}/'.format(pre_args.label))

    if not os.path.exists('/scratch/da2734/twitter/jobs/training_binary/models/models_may5_7Klabels_removed_allzeros/output_{}'.format(pre_args.label)):
        os.makedirs(      '/scratch/da2734/twitter/jobs/training_binary/models/models_may5_7Klabels_removed_allzeros/output_{}'.format(pre_args.label))

    LOG_PATH = Path('/scratch/da2734/twitter/jobs/training_binary/logs/log_binary_pos_neg_{}/'.format(column))
    print('LOG_PATH', LOG_PATH)
    DATA_PATH = Path('/scratch/da2734/twitter/data/may5_7Klabels/data_binary_pos_neg_balanced_removed_allzeros/')
    LABEL_PATH = Path('/scratch/da2734/twitter/data/may5_7Klabels/data_binary_pos_neg_balanced_removed_allzeros/')
    OUTPUT_PATH = Path('/scratch/da2734/twitter/jobs/training_binary/models/models_may5_7Klabels_removed_allzeros/output_{}'.format(column))
    FINETUNED_PATH = None

elif pre_args.user == "manu":
    if not os.path.exists(os.path.join(pre_args.results_folder,'logs_{}/'.format(pre_args.label))):
        os.makedirs(os.path.join(pre_args.results_folder,'log_{}/'.format(pre_args.label)))
    if not os.path.exists(os.path.join(pre_args.results_folder,'output_{}'.format(pre_args.label))):
        os.makedirs(os.path.join(pre_args.results_folder,'output_{}'.format(pre_args.label)))

    LOG_PATH = Path(os.path.join(pre_args.results_folder,'log_{}/'.format(pre_args.label)))
    print('LOG_PATH', LOG_PATH)
    DATA_PATH = Path(pre_args.input_data_folder)
    LABEL_PATH = Path(pre_args.input_data_folder)
    OUTPUT_PATH = Path(os.path.join(pre_args.results_folder,'output_{}'.format(pre_args.label)))
    FINETUNED_PATH = None

args = Box({
    "run_text": pre_args.training_description,
    "train_size": -1,
    "val_size": -1,
    "log_path": LOG_PATH,
    "full_data_dir": DATA_PATH,
    "data_dir": DATA_PATH,
    "task_name": "labor_market_classification",
    #     "bert_model": BERT_PRETRAINED_PATH,
    "output_dir": OUTPUT_PATH,
    "max_seq_length": 512,
    "do_train": True,
    "do_eval": True,
    "do_lower_case": True,
    "train_batch_size": 8,
    "eval_batch_size": 16,
    "learning_rate": 5e-5,
    "num_train_epochs": pre_args.num_train_epochs,
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
    "model_name": pre_args.model_name,
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

label_cols = ['class'] #this is the name of the column in the train and val csv files where the labels are

databunch = BertDataBunch(
    args['data_dir'],
    LABEL_PATH,
    args.model_name,
    train_file='train_{}.csv'.format(pre_args.label),
    val_file='val_{}.csv'.format(pre_args.label),
    label_file='label_{}.csv'.format(pre_args.label),
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
# metrics.append({'name': 'accuracy_multilabel', 'function': accuracy_multilabel})

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
    multi_label=False,
    logging_steps=0)

learner.fit(args.num_train_epochs, args.learning_rate, validate=True)  # this trains the model


#     break