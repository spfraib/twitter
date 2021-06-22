import os
import torch
import onnxruntime as ort
import pandas as pd
import numpy as np
import os
import time
import torch.nn.functional as F
import onnx
import getpass
from transformers import AutoTokenizer
import time
import pyarrow.parquet as pq
from glob import glob
import os
import numpy as np
import argparse
import logging
import socket
import multiprocessing
from functools import reduce

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
print('libs loaded')

parser = argparse.ArgumentParser()

parser.add_argument("--input_path", type=str, help="path to input data")
parser.add_argument("--output_path", type=str, help="path where inference csv is saved")
parser.add_argument("--country_code", type=str, help="path where inference csv is saved")
parser.add_argument("--iteration_number", type=int)
parser.add_argument("--method", type=int)
parser.add_argument("--debug_mode", type=bool, help="fast debug mode", default=True)
parser.add_argument("--drop_duplicates", type=bool, help="drop duplicated tweets from parquet files", default=False)
parser.add_argument("--resume", type=int, help="resuming a run, 0 or 1")


args = parser.parse_args()

print(args)
DEBUG_MODE = args.debug_mode

global_start = time.time()

# ####################################################################################################################################
# # loading data
# ####################################################################################################################################

path_to_data = args.input_path

print('Load random Tweets:')

start_time = time.time()

final_output_path = args.output_path #e.g. /scratch/spf248/twitter/data/user_timeline/bert_inferrred/MX

if not os.path.exists(os.path.join(final_output_path)):
    print('>>>> directory doesnt exists, creating it')
    os.makedirs(os.path.join(final_output_path))

input_files_list = glob(os.path.join(path_to_data, '*.parquet'))

"""
creating a list of unique file ids assuming this file name structure:
/scratch/spf248/twitter/data/user_timeline/user_timeline_text_preprocessed/part-00000-52fdb0a4-e509-49fe-9f3a-d809594bba7d-c000.snappy.parquet
in this case:
unique_intput_file_id_list will contain 00000-52fdb0a4-e509-49fe-9f3a-d809594bba7d
filename_prefix is /scratch/spf248/twitter/data/user_timeline/user_timeline_text_preprocessed/part-
filename_suffix is -c000.snappy.parquet
"""
unique_intput_file_id_list = [filename.split('part-')[1].split('-c000')[0]
                              for filename in input_files_list]
filename_prefix = input_files_list[0].split('part-')[0]
filename_suffix = input_files_list[0].split('part-')[1].split('-c000')[1]

already_processed_output_files = glob(os.path.join(final_output_path, '*.parquet'))
unique_already_processed_file_id_list = [filename.split('part-')[1].split('-c000')[0]
                              for filename in already_processed_output_files]

if args.resume == 1:
    unique_ids_remaining = list(set(unique_intput_file_id_list) - set(unique_already_processed_file_id_list))
    files_remaining = [filename_prefix+'part-'+filename+'-c000'+filename_suffix for filename in unique_ids_remaining]
    print(files_remaining[:3])
    print(len(files_remaining), len(unique_intput_file_id_list), len(unique_already_processed_file_id_list))
else:
    files_remaining = input_files_list

print('resume:', args.resume, '\n',
      'files already run', len(unique_already_processed_file_id_list), '\n',
      'original number of files to run', len(unique_intput_file_id_list), '\n',
      'files to run after resume:', len(files_remaining)
      )


